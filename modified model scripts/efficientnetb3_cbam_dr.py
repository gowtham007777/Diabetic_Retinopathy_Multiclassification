
"""
efficientnetb3_cbam_dr.py

Production-oriented 5-class diabetic-retinopathy classification using:

    EfficientNet-B3 (timm)
        + CBAM
        + 5-class classification head
        + domain-specific DR checkpoint initialization
        + staged fine-tuning
        + class-weighted cross-entropy
        + label smoothing
        + Grad-CAM on EfficientNet-B3 final conv_head

Dataset:
    data/organized/pooled_aptos_ddr/
        train/0_No_DR
        train/1_Mild
        train/2_Moderate
        train/3_Severe
        train/4_Proliferative_DR
        val/...
        test/...

Model input:
    RGB -> 300x300 -> ImageNet normalization

IMPORTANT:
    The domain-specific checkpoint is loaded into a reconstructed
    EfficientNet-B3 + original DR-head model first. Its EfficientNet-B3
    backbone is then transferred into the new EfficientNet-B3 + CBAM model.

    The old checkpoint attention block is not copied into CBAM because CBAM
    is a different attention mechanism. The original DR severity classifier
    Linear layers ARE semantically transferred where the tensor shapes match:
        severity_classifier.0 -> classifier.2
        severity_classifier.3 -> classifier.6
    The newly inserted BatchNorm1d and all CBAM parameters remain freshly
    initialized.

No scikit-learn dependency is used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# ============================================================
# PATHS / CONSTANTS
# ============================================================

PROJECT_ROOT = Path(r"A:\DR_classification")

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "organized"
    / "pooled_aptos_ddr"
)

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "weights"
    / "best_model_v2.pth"
)

RUN_ROOT = PROJECT_ROOT / "runs_EfficientNetB3_CBAM"
OUTPUT_ROOT = RUN_ROOT / "outputs"
GRADCAM_ROOT = OUTPUT_ROOT / "gradcam"

CLASS_DIRS = [
    "0_No_DR",
    "1_Mild",
    "2_Moderate",
    "3_Severe",
    "4_Proliferative_DR",
]

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

NUM_CLASSES = 5

IMAGE_SIZE = 300
EFFICIENTNET_FEATURES = 1536

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CBAM_REDUCTION = 8
CBAM_KERNEL_SIZE = 7

HEAD_HIDDEN = 768
HEAD_DROPOUT = 0.45

LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# A modest stochastic-depth setting. It does not change tensor shapes.
DROP_PATH_RATE = 0.10

DEFAULT_BATCH_SIZE = 4
DEFAULT_WORKERS = 2
DEFAULT_ACCUMULATION = 4

DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 6

# Requested staged protocol.
STAGE1_LR_HEAD = 3e-4

STAGE2_LR_HEAD = 1e-4
STAGE2_LR_BACKBONE = 1e-5

STAGE3_LR_HEAD = 3e-5
STAGE3_LR_BACKBONE = 5e-6

STAGE1_DEFAULT_EPOCHS = 5
STAGE2_DEFAULT_EPOCHS = 8

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# REPRODUCIBILITY / GENERAL HELPERS
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    GRADCAM_ROOT.mkdir(parents=True, exist_ok=True)


def infer_eye(filename: str) -> str:
    """
    Best-effort eye inference from filename.
    Returns N/A when the filename does not contain a recognizable token.
    """
    stem = Path(filename).stem.lower()

    if re.search(r"(?:^|[_\-.])(right|od)(?:$|[_\-.])", stem):
        return "Right"

    if re.search(r"(?:^|[_\-.])(left|os)(?:$|[_\-.])", stem):
        return "Left"

    return "N/A"


def is_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def format_pct(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{100.0 * value:.2f}%"


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def choose_device() -> torch.device:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n[Device]")
    print(f"  Device: {device}")

    if device.type == "cuda":
        print(
            f"  GPU   : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"  CUDA  : "
            f"{torch.version.cuda}"
        )

    return device


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_manifest(
    dataset_root: Path,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for split in ("train", "val", "test"):
        split_root = dataset_root / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Missing split directory:\n{split_root}"
            )

        for class_idx, class_dir in enumerate(
            CLASS_DIRS
        ):
            class_root = split_root / class_dir

            if not class_root.exists():
                raise FileNotFoundError(
                    f"Missing class directory:\n{class_root}"
                )

            for path in sorted(
                class_root.rglob("*")
            ):
                if not is_image(path):
                    continue

                rows.append(
                    {
                        "split": split,
                        "class_idx": class_idx,
                        "class_name": CLASS_NAMES[
                            class_idx
                        ],
                        "path": str(path),
                        "image_id": path.stem,
                        "eye": infer_eye(
                            path.stem
                        ),
                    }
                )

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError(
            f"No images were found under:\n{dataset_root}"
        )

    return manifest


def print_split_distribution(
    manifest: pd.DataFrame,
    title: str,
) -> None:
    table = (
        manifest.groupby(
            ["split", "class_name"]
        )
        .size()
        .unstack(fill_value=0)
    )

    # Force class-column ordering where available.
    ordered_columns = [
        name
        for name in CLASS_NAMES
        if name in table.columns
    ]

    table = table[
        ordered_columns
    ]

    print(f"\n[{title}]")
    print(table.to_string())


# ============================================================
# TRANSFORMS
# ============================================================

def build_train_transform(
    use_random_resized_crop: bool = False,
) -> transforms.Compose:
    ops: List = []

    if use_random_resized_crop:
        ops.append(
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.88, 1.0),
                ratio=(0.95, 1.05),
                interpolation=(
                    transforms.InterpolationMode.BILINEAR
                ),
            )
        )
    else:
        ops.append(
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=(
                    transforms.InterpolationMode.BILINEAR
                ),
            )
        )

    ops.extend(
        [
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                degrees=12,
                interpolation=(
                    transforms.InterpolationMode.BILINEAR
                ),
                fill=0,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.93, 1.07),
                interpolation=(
                    transforms.InterpolationMode.BILINEAR
                ),
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.06,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return transforms.Compose(ops)


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=(
                    transforms.InterpolationMode.BILINEAR
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


class DRDataset(Dataset):
    """
    Custom directory-backed dataset.

    Returns:
        image_tensor
        integer_label
        path
        image_id
        eye
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        transform: transforms.Compose,
    ):
        self.df = (
            manifest[
                manifest["split"] == split
            ]
            .reset_index(drop=True)
            .copy()
        )

        if len(self.df) == 0:
            raise RuntimeError(
                f"No samples found for split: {split}"
            )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        try:
            image = Image.open(
                row["path"]
            ).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read image:\n"
                f"{row['path']}\n"
                f"Reason: {exc}"
            ) from exc

        image_tensor = self.transform(
            image
        )

        return (
            image_tensor,
            int(row["class_idx"]),
            str(row["path"]),
            str(row["image_id"]),
            str(row["eye"]),
        )


def dr_collate(batch):
    """
    Explicitly collate tensors and metadata separately.
    """
    images = torch.stack(
        [item[0] for item in batch],
        dim=0,
    )

    labels = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long,
    )

    paths = [
        str(item[2])
        for item in batch
    ]

    image_ids = [
        str(item[3])
        for item in batch
    ]

    eyes = [
        str(item[4])
        for item in batch
    ]

    return (
        images,
        labels,
        paths,
        image_ids,
        eyes,
    )


# ============================================================
# MODEL (EFFICIENTNET-B3 + CBAM + HEAD)
# ============================================================

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Channel attention:
        GAP + GMP -> shared MLP -> sigmoid

    Spatial attention:
        channel-wise avg/max -> 7x7 conv -> sigmoid
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        kernel_size: int = 7,
    ):
        super().__init__()

        hidden = max(
            1,
            channels // reduction,
        )

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.max_pool = (
            nn.AdaptiveMaxPool2d(1)
        )

        # Shared MLP for GAP and GMP.
        self.mlp = nn.Sequential(
            nn.Linear(
                channels,
                hidden,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden,
                channels,
                bias=False,
            ),
        )

        padding = kernel_size // 2

        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        b, c, _, _ = x.shape

        avg_out = (
            self.avg_pool(x)
            .view(b, c)
        )

        max_out = (
            self.max_pool(x)
            .view(b, c)
        )

        channel_logits = (
            self.mlp(avg_out)
            +
            self.mlp(max_out)
        )

        channel_gate = torch.sigmoid(
            channel_logits
        ).view(
            b,
            c,
            1,
            1,
        )

        x = x * channel_gate

        avg_pool = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_pool = torch.amax(
            x,
            dim=1,
            keepdim=True,
        )

        spatial_input = torch.cat(
            [avg_pool, max_pool],
            dim=1,
        )

        spatial_gate = torch.sigmoid(
            self.spatial_conv(
                spatial_input
            )
        )

        return x * spatial_gate


class OriginalDRCheckpointModel(nn.Module):
    """
    Reconstruction of the original checkpoint pathway used to validate
    and load best_model_v2.pth before transferring its EfficientNet backbone.

    The original checkpoint is represented as:
        EfficientNet-B3
        -> channel attention
        -> BatchNorm feature normalization
        -> dropout
        -> severity classifier

    Auxiliary heads are included for checkpoint compatibility but are not
    used in the new CBAM model.
    """

    def __init__(
        self,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        import timm

        try:
            self.backbone = timm.create_model(
                "efficientnet_b3",
                pretrained=False,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
        except TypeError:
            self.backbone = timm.create_model(
                "efficientnet_b3",
                pretrained=False,
                num_classes=0,
            )

        if int(self.backbone.num_features) != (
            EFFICIENTNET_FEATURES
        ):
            raise RuntimeError(
                "Unexpected EfficientNet-B3 feature dimension: "
                f"{self.backbone.num_features}"
            )

        attention_hidden = (
            EFFICIENTNET_FEATURES
            // 8
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                EFFICIENTNET_FEATURES,
                attention_hidden,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                attention_hidden,
                EFFICIENTNET_FEATURES,
            ),
            nn.Sigmoid(),
        )

        self.feature_norm = nn.BatchNorm1d(
            EFFICIENTNET_FEATURES
        )

        self.dropout = nn.Dropout(
            0.40
        )

        self.severity_classifier = nn.Sequential(
            nn.Linear(
                EFFICIENTNET_FEATURES,
                768,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(
                0.20
            ),
            nn.Linear(
                768,
                NUM_CLASSES,
            ),
        )

        # Retained only to make checkpoint loading as compatible as possible.
        self.lesion_detector = nn.Sequential(
            nn.Linear(
                EFFICIENTNET_FEATURES,
                EFFICIENTNET_FEATURES // 4,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(
                EFFICIENTNET_FEATURES // 4,
                5,
            ),
        )

        self.region_predictor = nn.Sequential(
            nn.Linear(
                EFFICIENTNET_FEATURES,
                EFFICIENTNET_FEATURES // 4,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(
                EFFICIENTNET_FEATURES // 4,
                5,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = (
            self.backbone.forward_features(x)
        )

        pooled = F.adaptive_avg_pool2d(
            features,
            1,
        ).flatten(1)

        weights = self.attention(
            features
        )

        weighted = pooled * weights

        normalized = self.feature_norm(
            weighted
        )

        normalized = self.dropout(
            normalized
        )

        return self.severity_classifier(
            normalized
        )


class CBAMDRModel(nn.Module):
    """
    EfficientNet-B3 + CBAM + 5-class head.

    EfficientNet-B3 output:
        [B, 1536, H, W]

    CBAM:
        [B, 1536, H, W]

    Head:
        GAP -> 1536
        -> Linear(1536, 768)
        -> BatchNorm1d
        -> SiLU
        -> Dropout(0.45)
        -> Linear(768, 5)
    """

    def __init__(
        self,
        drop_path_rate: float = DROP_PATH_RATE,
    ):
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required.\n"
                "Install it with:\n"
                "python -m pip install timm"
            ) from exc

        try:
            self.backbone = timm.create_model(
                "efficientnet_b3",
                pretrained=False,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
        except TypeError:
            self.backbone = timm.create_model(
                "efficientnet_b3",
                pretrained=False,
                num_classes=0,
            )

        if int(self.backbone.num_features) != (
            EFFICIENTNET_FEATURES
        ):
            raise RuntimeError(
                "EfficientNet-B3 returned unexpected "
                f"feature dimension: "
                f"{self.backbone.num_features}"
            )

        self.cbam = CBAM(
            channels=EFFICIENTNET_FEATURES,
            reduction=CBAM_REDUCTION,
            kernel_size=CBAM_KERNEL_SIZE,
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                EFFICIENTNET_FEATURES,
                HEAD_HIDDEN,
            ),
            nn.BatchNorm1d(
                HEAD_HIDDEN
            ),
            nn.SiLU(inplace=True),
            nn.Dropout(
                HEAD_DROPOUT
            ),
            nn.Linear(
                HEAD_HIDDEN,
                NUM_CLASSES,
            ),
        )

        self.reset_new_parameters()

    def reset_new_parameters(self) -> None:
        """
        Explicitly initialize newly introduced CBAM/head parameters.
        """
        for module in (
            self.cbam,
            self.classifier,
        ):
            for child in module.modules():
                if isinstance(
                    child,
                    nn.Conv2d,
                ):
                    nn.init.kaiming_normal_(
                        child.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )

                    if child.bias is not None:
                        nn.init.zeros_(
                            child.bias
                        )

                elif isinstance(
                    child,
                    nn.Linear,
                ):
                    nn.init.kaiming_normal_(
                        child.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )

                    if child.bias is not None:
                        nn.init.zeros_(
                            child.bias
                        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = (
            self.backbone.forward_features(x)
        )

        attended = self.cbam(
            features
        )

        return self.classifier(
            attended
        )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def extract_state_dict(
    checkpoint,
) -> Dict[str, torch.Tensor]:
    """
    Accept common checkpoint wrappers:
        state_dict
        model_state_dict
        model
        raw state_dict
    """
    state = checkpoint

    if isinstance(checkpoint, dict):
        for candidate in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            if (
                candidate in checkpoint
                and isinstance(
                    checkpoint[candidate],
                    dict,
                )
            ):
                state = checkpoint[candidate]
                break

    if not isinstance(state, dict):
        raise RuntimeError(
            "Checkpoint does not contain a usable state_dict."
        )

    cleaned = {}

    for key, value in state.items():
        if not isinstance(
            value,
            torch.Tensor,
        ):
            continue

        k = str(key)

        for prefix in (
            "module.",
        ):
            if k.startswith(prefix):
                k = k[len(prefix):]

        cleaned[k] = value

    if not cleaned:
        raise RuntimeError(
            "No tensor parameters were found in checkpoint."
        )

    return cleaned


def load_checkpoint_into_original_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[
    OriginalDRCheckpointModel,
    Dict[str, torch.Tensor],
    Dict,
]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}\n\n"
            "Expected local file:\n"
            r"A:\DR_classification\weights\best_model_v2.pth"
        )

    print(
        "\n[Checkpoint] Loading domain-specific DR checkpoint"
    )
    print(
        f"  Path: {checkpoint_path}"
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch releases.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint could not be loaded.\n"
            f"Path: {checkpoint_path}\n"
            f"Reason: {exc}"
        ) from exc

    raw_state = extract_state_dict(
        checkpoint
    )

    source_model = OriginalDRCheckpointModel(
        drop_path_rate=0.0
    ).to(device)

    source_state = (
        source_model.state_dict()
    )

    compatible = {}
    mismatches = []

    for key, value in raw_state.items():
        if key not in source_state:
            continue

        if tuple(value.shape) != tuple(
            source_state[key].shape
        ):
            mismatches.append(
                (
                    key,
                    tuple(value.shape),
                    tuple(
                        source_state[key].shape
                    ),
                )
            )
            continue

        compatible[key] = value

    missing, unexpected = (
        source_model.load_state_dict(
            compatible,
            strict=False,
        )
    )

    backbone_loaded = sum(
        key.startswith("backbone.")
        for key in compatible
    )

    backbone_total = sum(
        key.startswith("backbone.")
        for key in source_state
    )

    report = {
        "raw_tensors": len(raw_state),
        "compatible_tensors": len(compatible),
        "backbone_loaded": backbone_loaded,
        "backbone_total": backbone_total,
        "backbone_load_ratio": (
            backbone_loaded /
            max(1, backbone_total)
        ),
        "shape_mismatch_count": len(
            mismatches
        ),
        "missing_count": len(missing),
        "unexpected_count": len(
            unexpected
        ),
    }

    print(
        f"  Raw tensors            : "
        f"{len(raw_state)}"
    )
    print(
        f"  Compatible tensors     : "
        f"{len(compatible)}"
    )
    print(
        f"  Backbone tensors loaded: "
        f"{backbone_loaded}/{backbone_total}"
    )
    print(
        f"  Backbone load ratio    : "
        f"{100.0 * report['backbone_load_ratio']:.2f}%"
    )
    print(
        f"  Shape mismatches       : "
        f"{len(mismatches)}"
    )

    if mismatches:
        print(
            "  First shape mismatches:"
        )

        for key, source_shape, target_shape in (
            mismatches[:10]
        ):
            print(
                f"    - {key}: "
                f"checkpoint={source_shape}, "
                f"target={target_shape}"
            )

    if report[
        "backbone_load_ratio"
    ] < 0.90:
        raise RuntimeError(
            "Less than 90% of EfficientNet-B3 backbone "
            "tensors loaded from the domain-specific checkpoint. "
            "Refusing to silently fall back to a different "
            "initialization."
        )

    return (
        source_model,
        raw_state,
        report,
    )


def transfer_checkpoint_weights(
    source_model: OriginalDRCheckpointModel,
    target_model: CBAMDRModel,
    raw_state: Dict[str, torch.Tensor],
) -> Dict:
    """
    Transfer the domain-specific DR checkpoint into the new CBAM model.

    Transfer policy:
        1) Copy every compatible EfficientNet-B3 backbone tensor.
        2) Semantically map the original DR severity classifier Linear layers
           into the requested new classifier when tensor shapes match exactly.
        3) Do NOT copy the original attention block into CBAM: it is a
           different architecture/operation.
        4) Leave CBAM and the inserted BatchNorm1d freshly initialized.
        5) Perform a conservative exact-name/shape pass for any remaining
           compatible tensors without overwriting explicit semantic mappings.
    """
    target_state = target_model.state_dict()
    source_full = source_model.state_dict()

    copied_target_keys = set()
    copied_source_keys = set()
    semantic_mappings = []
    mismatches = []
    ignored_keys = []

    # --------------------------------------------------------
    # 1) EfficientNet-B3 backbone: source_model is authoritative
    # because it has already validated the checkpoint.
    # --------------------------------------------------------
    for key, value in source_full.items():
        if not key.startswith("backbone."):
            continue

        if key not in target_state:
            ignored_keys.append(key)
            continue

        if tuple(value.shape) != tuple(target_state[key].shape):
            mismatches.append(
                (
                    key,
                    tuple(value.shape),
                    tuple(target_state[key].shape),
                )
            )
            continue

        target_state[key] = value.detach().clone()
        copied_target_keys.add(key)
        copied_source_keys.add(key)

    # --------------------------------------------------------
    # 2) Semantic transfer of original DR severity head.
    # Original checkpoint:
    #   severity_classifier.0 : 1536 -> 768
    #   severity_classifier.3 : 768  -> 5
    # New CBAM classifier:
    #   classifier.2 : 1536 -> 768
    #   classifier.6 : 768  -> 5
    # --------------------------------------------------------
    semantic_map = {
        "severity_classifier.0.weight": "classifier.2.weight",
        "severity_classifier.0.bias": "classifier.2.bias",
        "severity_classifier.3.weight": "classifier.6.weight",
        "severity_classifier.3.bias": "classifier.6.bias",
    }

    for source_key, target_key in semantic_map.items():
        source_value = source_full.get(source_key)

        if source_value is None:
            ignored_keys.append(source_key)
            continue

        if target_key not in target_state:
            ignored_keys.append(source_key)
            continue

        if tuple(source_value.shape) != tuple(target_state[target_key].shape):
            mismatches.append(
                (
                    f"{source_key} -> {target_key}",
                    tuple(source_value.shape),
                    tuple(target_state[target_key].shape),
                )
            )
            continue

        target_state[target_key] = source_value.detach().clone()
        copied_target_keys.add(target_key)
        copied_source_keys.add(source_key)
        semantic_mappings.append(
            {
                "source": source_key,
                "target": target_key,
                "shape": list(source_value.shape),
            }
        )

    # --------------------------------------------------------
    # 3) Conservative exact key + shape match pass.
    # Explicit semantic mappings above take precedence.
    # --------------------------------------------------------
    for key, value in raw_state.items():
        if key in copied_source_keys:
            continue
        if key not in target_state:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            continue
        if key in copied_target_keys:
            continue

        target_state[key] = value.detach().clone()
        copied_target_keys.add(key)
        copied_source_keys.add(key)

    target_model.load_state_dict(target_state, strict=True)

    backbone_loaded = sum(
        key.startswith("backbone.") for key in copied_target_keys
    )

    new_keys = [
        key
        for key in target_state
        if key.startswith("cbam.") or key.startswith("classifier.")
    ]

    new_random_keys = [
        key for key in new_keys if key not in copied_target_keys
    ]

    # This counts checkpoint tensors not transferred into the new model,
    # including the old attention tensors and auxiliary heads.
    nonmatching_source = [
        key for key in raw_state
        if key not in copied_source_keys
    ]

    report = {
        "copied_total_target_tensors": len(copied_target_keys),
        "copied_backbone": int(backbone_loaded),
        "backbone_expected": 572,
        "semantic_head_tensors": len(semantic_mappings),
        "semantic_mappings": semantic_mappings,
        "new_cbam_head_tensors": len(new_keys),
        "new_random_cbam_head_tensors": len(new_random_keys),
        "nonmatching_source_tensors": len(nonmatching_source),
        "nonmatching_source_keys": nonmatching_source,
        "shape_mismatch_count": len(mismatches),
        "shape_mismatches": mismatches,
    }

    print("\n[Checkpoint -> CBAM model]")
    print(f"  Backbone tensors transferred : {backbone_loaded}/572")
    print(
        "  Original severity-head tensors transferred : "
        f"{len(semantic_mappings)}/4"
    )
    for mapping in semantic_mappings:
        print(
            f"    {mapping['source']} -> {mapping['target']} "
            f"shape={tuple(mapping['shape'])}"
        )
    print(f"  New CBAM/head state tensors : {len(new_keys)}")
    print(f"  Still-new CBAM/head tensors: {len(new_random_keys)}")
    print(f"  Source tensors not transferred: {len(nonmatching_source)}")
    print(f"  Shape mismatches             : {len(mismatches)}")

    if backbone_loaded < 572:
        raise RuntimeError(
            "The domain-specific EfficientNet-B3 backbone transfer is incomplete. "
            f"Loaded {backbone_loaded}/572 tensors."
        )

    if len(semantic_mappings) != 4:
        raise RuntimeError(
            "The compatible original DR severity-classifier transfer is incomplete. "
            f"Loaded {len(semantic_mappings)}/4 tensors."
        )

    return report


# ============================================================
# LOSS / METRICS
# ============================================================

def compute_class_weights(
    labels: Sequence[int],
) -> torch.Tensor:
    labels_np = np.asarray(
        labels,
        dtype=np.int64,
    )

    counts = np.bincount(
        labels_np,
        minlength=NUM_CLASSES,
    ).astype(np.float64)

    total = counts.sum()

    weights = np.sqrt(
        total /
        (
            NUM_CLASSES *
            np.maximum(
                counts,
                1.0,
            )
        )
    )

    weights /= weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def per_sample_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample weighted CE. Keeping reduction='none' lets us aggregate
    training/validation loss globally over all samples.
    """
    return F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
        reduction="none",
    )


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    cm = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    for true, pred in zip(
        y_true.astype(int),
        y_pred.astype(int),
    ):
        if (
            0 <= true < NUM_CLASSES
            and
            0 <= pred < NUM_CLASSES
        ):
            cm[
                true,
                pred
            ] += 1

    return cm


def precision_recall_f1(
    cm: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tp = np.diag(cm).astype(float)

    fp = (
        cm.sum(axis=0)
        - tp
    ).astype(float)

    fn = (
        cm.sum(axis=1)
        - tp
    ).astype(float)

    precision = (
        tp /
        np.maximum(
            1.0,
            tp + fp,
        )
    )

    recall = (
        tp /
        np.maximum(
            1.0,
            tp + fn,
        )
    )

    f1 = (
        2.0 *
        precision *
        recall /
        np.maximum(
            1e-12,
            precision + recall,
        )
    )

    return (
        precision,
        recall,
        f1,
    )


def calculate_specificity(
    cm: np.ndarray,
) -> np.ndarray:
    result = []

    total = float(
        cm.sum()
    )

    for c in range(NUM_CLASSES):
        tp = float(
            cm[c, c]
        )

        fn = float(
            cm[c, :].sum()
            - tp
        )

        fp = float(
            cm[:, c].sum()
            - tp
        )

        tn = (
            total
            - tp
            - fn
            - fp
        )

        result.append(
            tn /
            max(
                1.0,
                tn + fp,
            )
        )

    return np.asarray(
        result,
        dtype=float,
    )


def calculate_qwk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Quadratic Weighted Kappa implemented directly in NumPy.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    cm = make_confusion_matrix(
        y_true,
        y_pred,
    ).astype(float)

    actual_hist = np.bincount(
        y_true,
        minlength=NUM_CLASSES,
    ).astype(float)

    pred_hist = np.bincount(
        y_pred,
        minlength=NUM_CLASSES,
    ).astype(float)

    expected = np.outer(
        actual_hist,
        pred_hist,
    ) / max(
        1.0,
        len(y_true),
    )

    denom = float(
        (NUM_CLASSES - 1) ** 2
    )

    weights = np.zeros_like(
        cm
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            weights[i, j] = (
                (i - j) ** 2
                / denom
            )

    observed = np.sum(
        weights * cm
    )

    expected_disagreement = np.sum(
        weights * expected
    )

    if expected_disagreement <= 1e-12:
        return (
            1.0
            if np.array_equal(
                y_true,
                y_pred,
            )
            else 0.0
        )

    return float(
        1.0 -
        observed /
        expected_disagreement
    )


def calculate_binary_auc(
    binary_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    AUC via rank statistic; handles ties by average rank.
    """
    binary_true = (
        binary_true
        .astype(np.int64)
    )

    n_pos = int(
        (binary_true == 1).sum()
    )

    n_neg = int(
        (binary_true == 0).sum()
    )

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    sorted_scores = scores[
        order
    ]

    ranks = np.empty(
        len(scores),
        dtype=float,
    )

    start = 0

    while start < len(
        sorted_scores
    ):
        end = start + 1

        while (
            end < len(sorted_scores)
            and
            sorted_scores[end]
            ==
            sorted_scores[start]
        ):
            end += 1

        # 1-based average rank.
        average_rank = (
            (start + 1)
            +
            end
        ) / 2.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    positive_rank_sum = (
        ranks[binary_true == 1].sum()
    )

    return float(
        (
            positive_rank_sum
            -
            n_pos *
            (n_pos + 1)
            / 2.0
        )
        /
        (
            n_pos *
            n_neg
        )
    )


def calculate_multiclass_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[
    float,
    np.ndarray,
]:
    aucs = []

    for c in range(NUM_CLASSES):
        binary = (
            y_true == c
        ).astype(np.int64)

        aucs.append(
            calculate_binary_auc(
                binary,
                probabilities[:, c],
            )
        )

    aucs_np = np.asarray(
        aucs,
        dtype=float,
    )

    if np.all(
        np.isnan(aucs_np)
    ):
        macro = float("nan")
    else:
        macro = float(
            np.nanmean(
                aucs_np
            )
        )

    return (
        macro,
        aucs_np,
    )


def build_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
) -> Dict:
    cm = make_confusion_matrix(
        y_true,
        y_pred,
    )

    precision, recall, f1 = (
        precision_recall_f1(cm)
    )

    specificity = (
        calculate_specificity(cm)
    )

    auc_macro, auc_per_class = (
        calculate_multiclass_auc(
            y_true,
            probabilities,
        )
    )

    support = (
        cm.sum(axis=1)
    )

    accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )

    macro_f1 = float(
        np.mean(f1)
    )

    weighted_f1 = float(
        np.average(
            f1,
            weights=np.maximum(
                1,
                support,
            ),
        )
    )

    return {
        "loss": float(loss),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": calculate_qwk(
            y_true,
            y_pred,
        ),
        "macro_auc": auc_macro,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "specificity": specificity.tolist(),
        "f1": f1.tolist(),
        "auc_per_class": auc_per_class.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def freeze_backbone(
    model: CBAMDRModel,
) -> None:
    for parameter in (
        model.backbone.parameters()
    ):
        parameter.requires_grad = False


def unfreeze_last_two_blocks(
    model: CBAMDRModel,
) -> None:
    freeze_backbone(
        model
    )

    if hasattr(
        model.backbone,
        "blocks",
    ):
        blocks = list(
            model.backbone.blocks
        )

        for block in blocks[-2:]:
            for parameter in (
                block.parameters()
            ):
                parameter.requires_grad = True

    for module_name in (
        "conv_head",
        "bn2",
    ):
        module = getattr(
            model.backbone,
            module_name,
            None,
        )

        if module is not None:
            for parameter in (
                module.parameters()
            ):
                parameter.requires_grad = True


def unfreeze_entire_backbone(
    model: CBAMDRModel,
) -> None:
    for parameter in (
        model.backbone.parameters()
    ):
        parameter.requires_grad = True


def enable_cbam_head(
    model: CBAMDRModel,
) -> None:
    for parameter in (
        model.cbam.parameters()
    ):
        parameter.requires_grad = True

    for parameter in (
        model.classifier.parameters()
    ):
        parameter.requires_grad = True


def freeze_backbone_bn_running_stats(
    model: CBAMDRModel,
) -> None:
    """
    Keep pretrained EfficientNet BatchNorm running statistics frozen.

    This does not prevent gradients through BN affine weights/biases when
    those parameters are trainable. It only prevents running mean/variance
    drift caused by small medical-imaging batches.
    """
    for module in (
        model.backbone.modules()
    ):
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            module.eval()


def set_stage(
    model: CBAMDRModel,
    stage: int,
) -> None:
    if stage == 1:
        freeze_backbone(
            model
        )

    elif stage == 2:
        unfreeze_last_two_blocks(
            model
        )

    elif stage == 3:
        unfreeze_entire_backbone(
            model
        )

    else:
        raise ValueError(
            f"Unsupported stage: {stage}"
        )

    enable_cbam_head(
        model
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"[Fine-tuning] Stage {stage}: "
        f"trainable {trainable:,} / {total:,} "
        f"({100.0 * trainable / max(1,total):.2f}%)"
    )


def make_optimizer(
    model: CBAMDRModel,
    stage: int,
) -> torch.optim.Optimizer:
    cbam_head_parameters = (
        list(model.cbam.parameters())
        +
        list(model.classifier.parameters())
    )

    param_groups = [
        {
            "params": cbam_head_parameters,
            "lr": (
                STAGE1_LR_HEAD
                if stage == 1
                else
                STAGE2_LR_HEAD
                if stage == 2
                else
                STAGE3_LR_HEAD
            ),
            "group_name": "cbam_head",
        }
    ]

    if stage == 2:
        backbone_parameters = [
            p
            for p in model.backbone.parameters()
            if p.requires_grad
        ]

        param_groups.append(
            {
                "params": backbone_parameters,
                "lr": STAGE2_LR_BACKBONE,
                "group_name": "backbone",
            }
        )

    elif stage == 3:
        backbone_parameters = [
            p
            for p in model.backbone.parameters()
            if p.requires_grad
        ]

        param_groups.append(
            {
                "params": backbone_parameters,
                "lr": STAGE3_LR_BACKBONE,
                "group_name": "backbone",
            }
        )

    return torch.optim.AdamW(
        param_groups,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )


def train_one_epoch(
    model: CBAMDRModel,
    loader,
    optimizer,
    scaler,
    class_weights,
    device,
    accumulation_steps: int,
    epoch_label: str,
) -> Dict:
    model.train()

    # Re-freeze pretrained backbone BN running statistics.
    freeze_backbone_bn_running_stats(
        model
    )

    total_loss_sum = 0.0
    total_samples = 0
    total_correct = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    use_amp = (
        device.type == "cuda"
    )

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=epoch_label,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_idx, batch in progress:
        images, labels = (
            batch[0],
            batch[1],
        )

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp,
        ):
            logits = model(
                images
            )

            sample_losses = (
                per_sample_ce_loss(
                    logits,
                    labels,
                    class_weights,
                )
            )

            loss = sample_losses.mean()

            backward_loss = (
                loss /
                accumulation_steps
            )

        scaler.scale(
            backward_loss
        ).backward()

        should_step = (
            (
                batch_idx + 1
            )
            %
            accumulation_steps
            == 0
            or
            batch_idx == len(loader) - 1
        )

        if should_step:
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        batch_size = (
            labels.size(0)
        )

        total_loss_sum += float(
            sample_losses.sum().item()
        )

        total_samples += batch_size

        total_correct += int(
            (
                logits.argmax(
                    dim=1
                )
                ==
                labels
            )
            .sum()
            .item()
        )

        current_acc = (
            100.0 * total_correct
            / max(1, total_samples)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_acc:.1f}%",
            refresh=False,
        )

    return {
        "loss": (
            total_loss_sum /
            max(
                1,
                total_samples,
            )
        ),
        "accuracy": (
            total_correct /
            max(
                1,
                total_samples,
            )
        ),
    }


@torch.no_grad()
def evaluate(
    model: CBAMDRModel,
    loader,
    class_weights,
    device,
    desc: str = "Validation",
) -> Dict:
    model.eval()

    total_loss_sum = 0.0
    total_samples = 0

    y_true_list = []
    y_pred_list = []
    probability_list = []

    paths = []
    image_ids = []
    eyes = []

    progress = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for batch in progress:
        images, labels = (
            batch[0],
            batch[1],
        )

        batch_paths = batch[2]
        batch_image_ids = batch[3]
        batch_eyes = batch[4]

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        )

        sample_losses = (
            per_sample_ce_loss(
                logits,
                labels,
                class_weights,
            )
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = probabilities.argmax(
            dim=1
        )

        total_loss_sum += float(
            sample_losses.sum().item()
        )

        total_samples += int(
            labels.size(0)
        )

        y_true_list.append(
            labels.cpu().numpy()
        )

        y_pred_list.append(
            predictions.cpu().numpy()
        )

        probability_list.append(
            probabilities.cpu().numpy()
        )

        paths.extend(
            batch_paths
        )

        image_ids.extend(
            batch_image_ids
        )

        eyes.extend(
            batch_eyes
        )

    y_true = np.concatenate(
        y_true_list,
        axis=0,
    )

    y_pred = np.concatenate(
        y_pred_list,
        axis=0,
    )

    probabilities = np.concatenate(
        probability_list,
        axis=0,
    )

    metrics = build_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        loss=(
            total_loss_sum /
            max(
                1,
                total_samples,
            )
        ),
    )

    metrics[
        "predictions_df"
    ] = pd.DataFrame(
        {
            "path": paths,
            "image_id": image_ids,
            "eye": eyes,
            "true_class": y_true,
            "true_label": [
                CLASS_NAMES[i]
                for i in y_true
            ],
            "pred_class": y_pred,
            "pred_label": [
                CLASS_NAMES[i]
                for i in y_pred
            ],
            "confidence": probabilities.max(
                axis=1
            ),
        }
    )

    for class_idx in range(
        NUM_CLASSES
    ):
        metrics[
            f"prob_{class_idx}"
        ] = probabilities[
            :,
            class_idx,
        ]

        metrics[
            "predictions_df"
        ][
            f"prob_{class_idx}"
        ] = probabilities[
            :,
            class_idx,
        ]

    # Numerical diagnostics useful for validating loss stability.
    logits_probability = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    true_probs = np.asarray(
        [
            logits_probability[
                i,
                y_true[i],
            ]
            for i in range(
                len(y_true)
            )
        ],
        dtype=float,
    )

    # Reconstruct max absolute logit diagnostic from probabilities is not exact,
    # so evaluation also stores the probability-based confidence diagnostics.
    metrics[
        "min_true_probability"
    ] = float(
        true_probs.min()
    )

    metrics[
        "mean_true_probability"
    ] = float(
        true_probs.mean()
    )

    return metrics


def print_epoch_metrics(
    epoch: int,
    epochs: int,
    stage: int,
    lr_string: str,
    elapsed: float,
    train_metrics: Dict,
    val_metrics: Dict,
    model: CBAMDRModel,
) -> None:
    print(
        "\n"
        f"Epoch {epoch:03d}/{epochs:03d} | "
        f"Stage {stage} | "
        f"{lr_string} | "
        f"Time {elapsed:.1f}s"
    )

    print(
        f"  Train: "
        f"loss={train_metrics['loss']:.4f} "
        f"acc={format_pct(train_metrics['accuracy'])}"
    )

    print(
        f"  Val  : "
        f"loss={val_metrics['loss']:.4f} "
        f"acc={format_pct(val_metrics['accuracy'])} "
        f"macroF1={format_pct(val_metrics['macro_f1'])} "
        f"weightedF1={format_pct(val_metrics['weighted_f1'])} "
        f"QWK={val_metrics['qwk']:.4f} "
        f"AUC={val_metrics['macro_auc']:.4f}"
    )

    msag_unused = (
        "CBAM has no scalar residual coefficient"
    )

    print(
        f"  {msag_unused}"
    )

    print(
        f"  Val confidence diagnostics: "
        f"min P(true)={val_metrics['min_true_probability']:.6f}, "
        f"mean P(true)={val_metrics['mean_true_probability']:.4f}"
    )


def print_full_metrics(
    title: str,
    metrics: Dict,
) -> None:
    print(
        "\n"
        + "=" * 96
    )

    print(
        title
    )

    print(
        "=" * 96
    )

    print(
        f"Loss         : "
        f"{metrics['loss']:.6f}"
    )

    print(
        f"Accuracy     : "
        f"{format_pct(metrics['accuracy'])}"
    )

    print(
        f"Macro F1     : "
        f"{format_pct(metrics['macro_f1'])}"
    )

    print(
        f"Weighted F1  : "
        f"{format_pct(metrics['weighted_f1'])}"
    )

    print(
        f"QWK          : "
        f"{metrics['qwk']:.4f}"
    )

    print(
        f"Macro ROC-AUC: "
        f"{metrics['macro_auc']:.4f}"
    )

    print(
        "\n"
        f"{'Class':22s}"
        f"{'Prec':>9s}"
        f"{'Recall':>9s}"
        f"{'Spec':>9s}"
        f"{'F1':>9s}"
        f"{'AUC':>9s}"
        f"{'N':>8s}"
    )

    print(
        "-" * 76
    )

    for class_idx in range(
        NUM_CLASSES
    ):
        print(
            f"{CLASS_NAMES[class_idx]:22s}"
            f"{metrics['precision'][class_idx]:9.3f}"
            f"{metrics['recall'][class_idx]:9.3f}"
            f"{metrics['specificity'][class_idx]:9.3f}"
            f"{metrics['f1'][class_idx]:9.3f}"
            f"{metrics['auc_per_class'][class_idx]:9.3f}"
            f"{metrics['support'][class_idx]:8d}"
        )

    print(
        "\nConfusion matrix:"
    )

    print(
        np.asarray(
            metrics[
                "confusion_matrix"
            ]
        )
    )


# ============================================================
# PLOTS / OUTPUTS
# ============================================================

def save_class_metrics(
    metrics: Dict,
    output_path: Path,
) -> None:
    rows = []

    for c in range(
        NUM_CLASSES
    ):
        rows.append(
            {
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "precision": metrics[
                    "precision"
                ][c],
                "recall": metrics[
                    "recall"
                ][c],
                "specificity": metrics[
                    "specificity"
                ][c],
                "f1": metrics[
                    "f1"
                ][c],
                "auc_ovr": metrics[
                    "auc_per_class"
                ][c],
                "support": metrics[
                    "support"
                ][c],
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        output_path,
        index=False,
    )


def save_confusion_matrix_plot(
    cm: np.ndarray,
    output_path: Path,
    normalize: bool,
) -> None:
    matrix = cm.astype(
        float
    )

    if normalize:
        matrix = (
            matrix /
            np.maximum(
                matrix.sum(
                    axis=1,
                    keepdims=True,
                ),
                1.0,
            )
        )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        matrix
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set_xticks(
        np.arange(NUM_CLASSES)
    )

    ax.set_yticks(
        np.arange(NUM_CLASSES)
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=35,
        ha="right",
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_title(
        (
            "Normalized Confusion Matrix"
            if normalize
            else
            "Confusion Matrix"
        )
    )

    for i in range(
        NUM_CLASSES
    ):
        for j in range(
            NUM_CLASSES
        ):
            display_value = (
                f"{matrix[i,j]:.2f}"
                if normalize
                else
                f"{int(matrix[i,j])}"
            )

            ax.text(
                j,
                i,
                display_value,
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_history_plots(
    history: pd.DataFrame,
) -> None:
    loss_path = (
        OUTPUT_ROOT /
        "training_curves.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
    )

    ax.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "Training / Validation Loss"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        loss_path,
        dpi=180,
    )

    plt.close(fig)

    performance_path = (
        OUTPUT_ROOT /
        "training_curves_performance.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        history["epoch"],
        history["train_acc"],
        label="Train Accuracy",
    )

    ax.plot(
        history["epoch"],
        history["val_acc"],
        label="Validation Accuracy",
    )

    ax.plot(
        history["epoch"],
        history["val_qwk"],
        label="Validation QWK",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Training / Validation Performance"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        performance_path,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """
    Grad-CAM based on the same hook/gradient-weighting concept used in the
    earlier EfficientNet-B3 + MSAG implementation.

    Target:
        EfficientNet-B3 final convolutional feature layer: conv_head

    Formula:
        alpha_k = mean spatial gradient
        CAM = ReLU(sum_k(alpha_k * activation_k))
        min-max normalize
    """

    def __init__(
        self,
        model: CBAMDRModel,
        target_layer: nn.Module,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[
            torch.Tensor
        ] = None

        self.gradients: Optional[
            torch.Tensor
        ] = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):
        self.gradients = grad_output[0]

    def __call__(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[
        np.ndarray,
        int,
        np.ndarray,
    ]:
        self.activations = None
        self.gradients = None

        # Needed when the target backbone is frozen.
        x = (
            image_tensor.detach()
            .clone()
            .requires_grad_(True)
        )

        self.model.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():
            logits = self.model(
                x
            )

            if class_idx is None:
                class_idx = int(
                    logits.argmax(
                        dim=1
                    ).item()
                )

            score = logits[
                0,
                class_idx,
            ]

            score.backward()

        if (
            self.activations is None
            or
            self.gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM hooks did not capture "
                "activations/gradients."
            )

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights *
            activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(
            cam
        )

        cam = cam.squeeze(
            1
        )

        cam_min = cam.amin(
            dim=(1, 2),
            keepdim=True,
        )

        cam_max = cam.amax(
            dim=(1, 2),
            keepdim=True,
        )

        cam = (
            cam - cam_min
        ) / (
            cam_max
            -
            cam_min
            +
            1e-7
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()[0]
        )

        return (
            cam.detach()
            .cpu()
            .numpy()[0],
            int(class_idx),
            probabilities,
        )

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_gradcam(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    h, w = rgb.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_u8 = np.clip(
        heatmap * 255.0,
        0,
        255,
    ).astype(np.uint8)

    color = cv2.applyColorMap(
        heatmap_u8,
        cv2.COLORMAP_JET,
    )

    color = cv2.cvtColor(
        color,
        cv2.COLOR_BGR2RGB,
    )

    return cv2.addWeighted(
        rgb,
        1.0 - alpha,
        color,
        alpha,
        0,
    )


def save_gradcam_panel(
    model: CBAMDRModel,
    row: pd.Series,
    device: torch.device,
    output_path: Path,
) -> Dict:
    model.eval()

    image_path = Path(
        row["path"]
    )

    bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    original_rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    model_transform = build_eval_transform()

    tensor = (
        model_transform(
            Image.fromarray(
                original_rgb
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    # Same Grad-CAM target used in the refined EfficientNet-B3 workflow:
    # the final EfficientNet convolutional feature layer.
    cam_engine = GradCAM(
        model,
        model.backbone.conv_head,
    )

    try:
        heatmap, pred_class, probabilities = (
            cam_engine(
                tensor,
                class_idx=None,
            )
        )
    finally:
        cam_engine.close()

    true_class = int(
        row["true_class"]
    )

    confidence = float(
        probabilities[
            pred_class
        ]
    )

    correct = (
        pred_class == true_class
    )

    input_rgb = cv2.resize(
        original_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    cam_overlay = overlay_gradcam(
        input_rgb,
        heatmap,
        alpha=0.45,
    )

    fig = plt.figure(
        figsize=(16, 9)
    )

    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.0],
        hspace=0.30,
        wspace=0.22,
    )

    ax_original = fig.add_subplot(
        grid[0, 0]
    )

    ax_input = fig.add_subplot(
        grid[0, 1]
    )

    ax_cam = fig.add_subplot(
        grid[0, 2]
    )

    ax_probs = fig.add_subplot(
        grid[0, 3]
    )

    ax_explanation = fig.add_subplot(
        grid[1, :3]
    )

    ax_summary = fig.add_subplot(
        grid[1, 3]
    )

    ax_original.imshow(
        original_rgb
    )

    ax_original.set_title(
        "Original Fundus Input",
        fontweight="bold",
    )

    ax_original.axis(
        "off"
    )

    ax_input.imshow(
        input_rgb
    )

    ax_input.set_title(
        "Model Input\nRGB → 300×300",
        fontweight="bold",
    )

    ax_input.axis(
        "off"
    )

    ax_cam.imshow(
        cam_overlay
    )

    ax_cam.set_title(
        "Grad-CAM\n"
        f"Predicted: {CLASS_NAMES[pred_class]}\n"
        f"Confidence: {100.0 * confidence:.1f}%",
        fontweight="bold",
    )

    ax_cam.axis(
        "off"
    )

    bars = ax_probs.barh(
        np.arange(NUM_CLASSES),
        probabilities * 100.0,
    )

    ax_probs.set_yticks(
        np.arange(NUM_CLASSES)
    )

    ax_probs.set_yticklabels(
        CLASS_NAMES
    )

    ax_probs.invert_yaxis()

    ax_probs.set_xlim(
        0,
        100,
    )

    ax_probs.set_xlabel(
        "Probability (%)"
    )

    ax_probs.set_title(
        "Prediction Confidence",
        fontweight="bold",
    )

    for bar, probability in zip(
        bars,
        probabilities,
    ):
        ax_probs.text(
            min(
                96.0,
                bar.get_width() + 1.0,
            ),
            bar.get_y()
            +
            bar.get_height() / 2,
            f"{100.0 * probability:.1f}%",
            va="center",
            fontsize=9,
        )

    # Dedicated explanation region.
    ax_explanation.axis(
        "off"
    )

    explanation_text = (
        "Grad-CAM interpretation\n\n"
        "Target layer:\n"
        "EfficientNet-B3 conv_head\n\n"
        "The highlighted regions show spatial areas "
        "that contributed most strongly to the selected "
        "predicted class score.\n\n"
        "This visualization is explanatory evidence, "
        "not a clinical diagnosis."
    )

    ax_explanation.text(
        0.02,
        0.95,
        explanation_text,
        va="top",
        ha="left",
        fontsize=13,
    )

    ax_summary.axis(
        "off"
    )

    status = (
        "CORRECT"
        if correct
        else
        "INCORRECT"
    )

    summary = (
        f"Image ID: {row['image_id']}\n"
        f"Eye: {row['eye']}\n\n"
        f"True Grade:\n"
        f"{true_class} ({CLASS_NAMES[true_class]})\n\n"
        f"Prediction:\n"
        f"{pred_class} ({CLASS_NAMES[pred_class]})\n\n"
        f"Confidence:\n"
        f"{100.0 * confidence:.1f}%\n\n"
        f"Status:\n"
        f"{status}"
    )

    ax_summary.text(
        0.02,
        0.95,
        summary,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(
        (
            f"Image ID: {row['image_id']} | "
            f"Eye: {row['eye']} | "
            f"True Grade: {true_class} "
            f"({CLASS_NAMES[true_class]}) | "
            f"Prediction: {pred_class} "
            f"({CLASS_NAMES[pred_class]}) | "
            f"Confidence: {100.0 * confidence:.1f}% | "
            f"Status: {status}"
        ),
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "path": str(image_path),
        "image_id": str(
            row["image_id"]
        ),
        "eye": str(
            row["eye"]
        ),
        "true_class": true_class,
        "true_label": CLASS_NAMES[
            true_class
        ],
        "pred_class": pred_class,
        "pred_label": CLASS_NAMES[
            pred_class
        ],
        "confidence": confidence,
        "status": status,
        "probabilities": probabilities.tolist(),
    }


def select_gradcam_rows(
    predictions_df: pd.DataFrame,
    number_of_samples: int,
) -> List[pd.Series]:
    selected: List[pd.Series] = []
    used = set()

    # Prefer one example per true class.
    for class_idx in range(
        NUM_CLASSES
    ):
        candidates = predictions_df[
            predictions_df["true_class"]
            ==
            class_idx
        ]

        if len(candidates):
            row = candidates.iloc[0]
            selected.append(row)
            used.add(
                str(row["path"])
            )

    # Fill remaining slots.
    for _, row in predictions_df.iterrows():
        if len(selected) >= number_of_samples:
            break

        path = str(
            row["path"]
        )

        if path in used:
            continue

        selected.append(
            row
        )

        used.add(path)

    return selected[
        :number_of_samples
    ]


# ============================================================
# CHECKPOINT / TRAINING UTILITIES
# ============================================================

def build_lr_string(
    optimizer: torch.optim.Optimizer,
) -> str:
    parts = []

    for group in optimizer.param_groups:
        name = group.get(
            "group_name",
            "group",
        )

        parts.append(
            f"{name}={group['lr']:.2e}"
        )

    return "LR " + ", ".join(parts)


def get_stage_for_epoch(
    epoch: int,
    stage1_epochs: int,
    stage2_epochs: int,
    total_epochs: int,
) -> int:
    if epoch <= stage1_epochs:
        return 1

    if epoch <= (
        stage1_epochs +
        stage2_epochs
    ):
        return 2

    if epoch <= total_epochs:
        return 3

    return 3


def save_best_checkpoint(
    path: Path,
    epoch: int,
    model: CBAMDRModel,
    optimizer: torch.optim.Optimizer,
    best_val_qwk: float,
    preprocessing_info: Dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_qwk": best_val_qwk,
            "class_names": CLASS_NAMES,
            "preprocessing": preprocessing_info,
        },
        path,
    )


def balanced_smoke_subset(
    dataset: DRDataset,
    per_class: int = 4,
) -> Subset:
    indices = []

    for class_idx in range(
        NUM_CLASSES
    ):
        class_indices = dataset.df.index[
            dataset.df["class_idx"].astype(int)
            ==
            class_idx
        ].tolist()

        if len(class_indices) < per_class:
            raise RuntimeError(
                f"Not enough samples for class "
                f"{class_idx} in smoke test."
            )

        indices.extend(
            class_indices[:per_class]
        )

    return Subset(
        dataset,
        indices,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "EfficientNet-B3 + CBAM diabetic retinopathy "
            "5-class classification."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(
            DEFAULT_DATASET_ROOT
        ),
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(
            DEFAULT_CHECKPOINT_PATH
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=DEFAULT_ACCUMULATION,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=STAGE1_DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=STAGE2_DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--random-resized-crop",
        action="store_true",
        help=(
            "Enable optional RandomResizedCrop "
            "instead of deterministic Resize."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run 1 epoch with a small subset. "
            "Validation/test smoke subsets contain 4 images per class."
        ),
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.workers < 0:
        raise ValueError(
            "--workers must be >= 0"
        )

    if args.accumulation_steps < 1:
        raise ValueError(
            "--accumulation-steps must be >= 1"
        )

    if args.gradcam_samples < 0:
        raise ValueError(
            "--gradcam-samples must be >= 0"
        )

    if args.stage1_epochs < 0:
        raise ValueError(
            "--stage1-epochs must be >= 0"
        )

    if args.stage2_epochs < 0:
        raise ValueError(
            "--stage2-epochs must be >= 0"
        )

    if (
        args.stage1_epochs
        +
        args.stage2_epochs
        >
        args.epochs
        and
        not args.smoke_test
    ):
        raise ValueError(
            "stage1-epochs + stage2-epochs cannot exceed total epochs."
        )

    seed_everything(
        args.seed
    )

    ensure_output_dirs()

    dataset_root = Path(
        args.dataset_root
    )

    checkpoint_path = Path(
        args.checkpoint_path
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EFFICIENTNET-B3 + CBAM — "
        "DIABETIC RETINOPATHY 5-CLASS CLASSIFICATION"
    )

    print(
        "=" * 100
    )

    print(
        f"Project root       : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Dataset root       : "
        f"{dataset_root}"
    )

    print(
        f"Checkpoint         : "
        f"{checkpoint_path}"
    )

    print(
        f"Model input        : "
        f"RGB → {IMAGE_SIZE}×{IMAGE_SIZE}"
    )

    print(
        f"Normalization      : "
        f"ImageNet mean/std"
    )

    print(
        f"CBAM               : "
        f"ON (reduction={CBAM_REDUCTION}, "
        f"kernel={CBAM_KERNEL_SIZE})"
    )

    print(
        f"Head               : "
        f"1536 → {HEAD_HIDDEN} → 5, "
        f"BN + SiLU + Dropout({HEAD_DROPOUT})"
    )

    print(
        f"Label smoothing    : "
        f"{LABEL_SMOOTHING}"
    )

    print(
        f"Weight decay       : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Gradient clip      : "
        f"{GRAD_CLIP_NORM}"
    )

    print(
        f"Effective batch    : "
        f"{args.batch_size * args.accumulation_steps}"
    )

    print(
        "Re-split           : OFF"
    )

    print(
        "Oversampling       : OFF"
    )

    print(
        "Grad-CAM target    : "
        "EfficientNet-B3 conv_head"
    )

    if args.random_resized_crop:
        print(
            "RRC augmentation   : ON"
        )
    else:
        print(
            "RRC augmentation   : OFF"
        )

    if args.smoke_test:
        print(
            "Smoke-test         : ON"
        )

    print(
        "=" * 100
    )

    device = choose_device()

    # --------------------------------------------------------
    # DATASET DISCOVERY
    # --------------------------------------------------------
    manifest = discover_manifest(
        dataset_root
    )

    print(
        f"\n[Dataset] Total images discovered: "
        f"{len(manifest)}"
    )

    print_split_distribution(
        manifest,
        "Dataset distribution"
    )

    train_dataset = DRDataset(
        manifest,
        split="train",
        transform=build_train_transform(
            args.random_resized_crop
        ),
    )

    val_dataset = DRDataset(
        manifest,
        split="val",
        transform=build_eval_transform(),
    )

    test_dataset = DRDataset(
        manifest,
        split="test",
        transform=build_eval_transform(),
    )

    train_labels = (
        train_dataset.df[
            "class_idx"
        ]
        .astype(int)
        .to_numpy()
    )

    class_weights_cpu = (
        compute_class_weights(
            train_labels
        )
    )

    class_weights = (
        class_weights_cpu.to(
            device
        )
    )

    print(
        "\n[Class weights]"
    )

    for c, weight in enumerate(
        class_weights_cpu.numpy()
    ):
        print(
            f"  {c} "
            f"{CLASS_NAMES[c]:20s}: "
            f"{weight:.6f}"
        )

    loader_common = {
        "num_workers": args.workers,
        "pin_memory": (
            device.type == "cuda"
        ),
        "collate_fn": dr_collate,
    }

    if args.workers > 0:
        loader_common[
            "persistent_workers"
        ] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_common,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    # Smoke loaders use balanced subsets to avoid class-order artifacts.
    # Training is also limited to a few batches for a genuinely fast smoke test.
    if args.smoke_test:
        class LimitedLoader:
            def __init__(self, loader, max_batches: int):
                self.loader = loader
                self.max_batches = max_batches

            def __iter__(self):
                for batch_index, batch in enumerate(self.loader):
                    if batch_index >= self.max_batches:
                        break
                    yield batch

            def __len__(self):
                return min(len(self.loader), self.max_batches)

        smoke_train_loader = LimitedLoader(
            train_loader,
            max_batches=8,
        )

        smoke_val_dataset = (
            balanced_smoke_subset(
                val_dataset,
                per_class=4,
            )
        )

        smoke_test_dataset = (
            balanced_smoke_subset(
                test_dataset,
                per_class=4,
            )
        )

        smoke_common = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": args.workers,
            "pin_memory": (
                device.type == "cuda"
            ),
            "collate_fn": dr_collate,
        }

        if args.workers > 0:
            smoke_common[
                "persistent_workers"
            ] = True

        smoke_val_loader = DataLoader(
            smoke_val_dataset,
            **smoke_common,
        )

        smoke_test_loader = DataLoader(
            smoke_test_dataset,
            **smoke_common,
        )

        print(
            "\n[Smoke evaluation]"
        )

        print(
            "  Training  : 8 batches only"
        )

        print(
            "  Validation: "
            "4 samples/class = 20 images"
        )

        print(
            "  Test      : "
            "4 samples/class = 20 images"
        )

        train_loader_for_run = (
            smoke_train_loader
        )

        val_loader_for_run = (
            smoke_val_loader
        )

        test_loader_for_run = (
            smoke_test_loader
        )

        effective_epochs = 1

    else:
        train_loader_for_run = (
            train_loader
        )

        val_loader_for_run = (
            val_loader
        )

        test_loader_for_run = (
            test_loader
        )

        effective_epochs = args.epochs

    # --------------------------------------------------------
    # MODEL / CHECKPOINT LOADING
    # --------------------------------------------------------
    try:
        (
            source_model,
            raw_checkpoint_state,
            checkpoint_load_report,
        ) = load_checkpoint_into_original_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        model = CBAMDRModel(
            drop_path_rate=DROP_PATH_RATE
        ).to(device)

        transfer_report = (
            transfer_checkpoint_weights(
                source_model=source_model,
                target_model=model,
                raw_state=raw_checkpoint_state,
            )
        )

    except Exception as exc:
        print(
            "\n[ERROR] Model/checkpoint initialization failed."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "\nExpected checkpoint:"
        )

        print(
            checkpoint_path
        )

        return 1

    # Free original checkpoint model after transfer.
    del source_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # SAVE MODEL CONFIG
    # --------------------------------------------------------
    preprocessing_info = {
        "model_input_size": (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "train_augmentation": {
            "horizontal_flip": 0.5,
            "rotation_degrees": 12,
            "translation": (
                0.04,
                0.04,
            ),
            "scale": (
                0.93,
                1.07,
            ),
            "brightness": 0.10,
            "contrast": 0.10,
            "saturation": 0.06,
            "hue": 0.02,
            "random_resized_crop": (
                bool(
                    args.random_resized_crop
                )
            ),
        },
        "validation_test": (
            "Resize 300x300 + ImageNet normalization"
        ),
    }

    model_info = {
        "architecture": (
            "EfficientNet-B3 + CBAM"
        ),
        "feature_dim": (
            EFFICIENTNET_FEATURES
        ),
        "cbam": {
            "reduction": CBAM_REDUCTION,
            "kernel_size": CBAM_KERNEL_SIZE,
            "channel_attention": (
                "GAP + GMP + shared MLP"
            ),
            "spatial_attention": (
                "channel avg/max + 7x7 conv"
            ),
        },
        "classifier": (
            "AdaptiveAvgPool2d -> Flatten -> "
            "Linear(1536,768) -> BatchNorm1d -> "
            "SiLU -> Dropout(0.45) -> Linear(768,5)"
        ),
        "drop_path_rate": DROP_PATH_RATE,
        "total_parameters": sum(
            p.numel()
            for p in model.parameters()
        ),
        "trainable_parameters_initial": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
        "checkpoint_loading": checkpoint_load_report,
        "checkpoint_transfer": transfer_report,
        "preprocessing": preprocessing_info,
    }

    (
        OUTPUT_ROOT /
        "model_summary.json"
    ).write_text(
        json.dumps(
            model_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    best_val_qwk = -float(
        "inf"
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    no_improvement_epochs = 0

    current_stage = None

    optimizer = None
    scheduler = None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    best_checkpoint_path = (
        OUTPUT_ROOT /
        "best_model_qwk.pth"
    )

    history_rows = []

    if args.smoke_test:
        stage1_epochs = 1
        stage2_epochs = 0
    else:
        stage1_epochs = args.stage1_epochs
        stage2_epochs = args.stage2_epochs

    for epoch in range(
        1,
        effective_epochs + 1,
    ):
        stage = get_stage_for_epoch(
            epoch=epoch,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            total_epochs=effective_epochs,
        )

        if stage != current_stage:
            current_stage = stage

            print(
                "\n"
                + "-" * 100
            )

            print(
                f"STARTING FINE-TUNING STAGE "
                f"{stage} AT EPOCH {epoch}"
            )

            print(
                "-" * 100
            )

            set_stage(
                model,
                stage,
            )

            optimizer = make_optimizer(
                model,
                stage,
            )

            # Requested scheduler: validation loss, mode=min.
            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.3,
                    patience=2,
                    min_lr=1e-7,
                )
            )

        assert optimizer is not None
        assert scheduler is not None

        start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader_for_run,
            optimizer=optimizer,
            scaler=scaler,
            class_weights=class_weights,
            device=device,
            accumulation_steps=(
                1
                if args.smoke_test
                else args.accumulation_steps
            ),
            epoch_label=(
                f"Epoch {epoch:03d}/{effective_epochs:03d} "
                f"Stage {stage} - Train"
            ),
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader_for_run,
            class_weights=class_weights,
            device=device,
            desc=(
                f"Epoch {epoch:03d}/{effective_epochs:03d} "
                f"Stage {stage} - Val"
            ),
        )

        # Scheduler reacts to global validation loss.
        scheduler.step(
            val_metrics["loss"]
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        lr_string = build_lr_string(
            optimizer
        )

        print_epoch_metrics(
            epoch=epoch,
            epochs=effective_epochs,
            stage=stage,
            lr_string=lr_string,
            elapsed=elapsed,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            model=model,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "stage": stage,
                "train_loss": train_metrics[
                    "loss"
                ],
                "train_acc": train_metrics[
                    "accuracy"
                ],
                "val_loss": val_metrics[
                    "loss"
                ],
                "val_acc": val_metrics[
                    "accuracy"
                ],
                "val_macro_f1": val_metrics[
                    "macro_f1"
                ],
                "val_weighted_f1": val_metrics[
                    "weighted_f1"
                ],
                "val_qwk": val_metrics[
                    "qwk"
                ],
                "val_auc": val_metrics[
                    "macro_auc"
                ],
                "val_min_true_probability": val_metrics[
                    "min_true_probability"
                ],
                "val_mean_true_probability": val_metrics[
                    "mean_true_probability"
                ],
                "elapsed_seconds": elapsed,
                "lr_cbam_head": (
                    optimizer.param_groups[0]["lr"]
                ),
                "lr_backbone": (
                    optimizer.param_groups[1]["lr"]
                    if len(
                        optimizer.param_groups
                    ) > 1
                    else 0.0
                ),
            }
        )

        # Best by validation QWK; loss breaks exact ties.
        improved = (
            val_metrics["qwk"]
            >
            best_val_qwk + 1e-6
        )

        if (
            abs(
                val_metrics["qwk"]
                -
                best_val_qwk
            )
            <= 1e-6
            and
            val_metrics["loss"]
            <
            best_val_loss
        ):
            improved = True

        if improved:
            best_val_qwk = (
                val_metrics["qwk"]
            )

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch

            no_improvement_epochs = 0

            save_best_checkpoint(
                path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_qwk=best_val_qwk,
                preprocessing_info=preprocessing_info,
            )

            print(
                f"  ✓ NEW BEST CHECKPOINT: "
                f"val QWK={best_val_qwk:.4f}"
            )

        else:
            no_improvement_epochs += 1

            print(
                f"  No QWK improvement: "
                f"{no_improvement_epochs}/"
                f"{args.patience}"
            )

            if (
                not args.smoke_test
                and
                no_improvement_epochs
                >=
                args.patience
            ):
                print(
                    "\nEarly stopping triggered."
                )

                break

    history_df = pd.DataFrame(
        history_rows
    )

    history_df.to_csv(
        OUTPUT_ROOT /
        "training_history.csv",
        index=False,
    )

    if len(history_df):
        save_history_plots(
            history_df
        )

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------
    if not best_checkpoint_path.exists():
        raise RuntimeError(
            "No best checkpoint was saved."
        )

    try:
        best_checkpoint = (
            torch.load(
                best_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        )
    except TypeError:
        best_checkpoint = (
            torch.load(
                best_checkpoint_path,
                map_location=device,
            )
        )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    print(
        "\n[Best model loaded]"
    )

    print(
        f"  Epoch: {best_checkpoint.get('epoch', best_epoch)}"
    )

    print(
        f"  Validation QWK: "
        f"{best_checkpoint.get('best_val_qwk', best_val_qwk):.4f}"
    )

    # --------------------------------------------------------
    # FINAL VALIDATION / TEST
    # --------------------------------------------------------
    if args.smoke_test:
        print(
            "\n[Smoke-test note] "
            "The metrics below use only the balanced smoke subsets "
            "and one epoch of training. They are pipeline diagnostics, "
            "not final model performance."
        )

    final_val = evaluate(
        model=model,
        loader=val_loader_for_run,
        class_weights=class_weights,
        device=device,
        desc="Final Validation",
    )

    final_test = evaluate(
        model=model,
        loader=test_loader_for_run,
        class_weights=class_weights,
        device=device,
        desc="Final Test",
    )

    print_full_metrics(
        (
            "SMOKE VALIDATION RESULTS"
            if args.smoke_test
            else
            "FINAL VALIDATION RESULTS"
        ),
        final_val,
    )

    print_full_metrics(
        (
            "SMOKE TEST RESULTS"
            if args.smoke_test
            else
            "FINAL TEST RESULTS"
        ),
        final_test,
    )

    # --------------------------------------------------------
    # TEST OUTPUTS
    # --------------------------------------------------------
    final_test[
        "predictions_df"
    ].to_csv(
        OUTPUT_ROOT /
        "test_predictions.csv",
        index=False,
    )

    save_class_metrics(
        final_test,
        OUTPUT_ROOT /
        "test_per_class_metrics.csv",
    )

    cm = np.asarray(
        final_test[
            "confusion_matrix"
        ],
        dtype=np.int64,
    )

    save_confusion_matrix_plot(
        cm=cm,
        output_path=(
            OUTPUT_ROOT /
            "test_confusion_matrix.png"
        ),
        normalize=False,
    )

    save_confusion_matrix_plot(
        cm=cm,
        output_path=(
            OUTPUT_ROOT /
            "test_confusion_matrix_normalized.png"
        ),
        normalize=True,
    )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------
    gradcam_reports = []

    if args.gradcam_samples > 0:
        selected_rows = select_gradcam_rows(
            final_test[
                "predictions_df"
            ],
            args.gradcam_samples,
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            "GRAD-CAM EXPLAINABILITY"
        )

        print(
            "=" * 100
        )

        for idx, row in enumerate(
            selected_rows,
            start=1,
        ):
            output_path = (
                GRADCAM_ROOT /
                (
                    f"gradcam_{idx:03d}_"
                    f"{row['image_id']}.png"
                )
            )

            try:
                report = save_gradcam_panel(
                    model=model,
                    row=row,
                    device=device,
                    output_path=output_path,
                )

                gradcam_reports.append(
                    report
                )

                print(
                    f"  ✓ {output_path}"
                )

            except Exception as exc:
                print(
                    f"  ✗ Grad-CAM failed for "
                    f"{row['path']}: {exc}"
                )

        pd.DataFrame(
            gradcam_reports
        ).to_csv(
            OUTPUT_ROOT /
            "gradcam_results.csv",
            index=False,
        )

    # --------------------------------------------------------
    # FINAL METRICS JSON
    # --------------------------------------------------------
    def strip_runtime_objects(
        metrics: Dict,
    ) -> Dict:
        result = {}

        for key, value in metrics.items():
            if (
                key in {
                    "predictions_df",
                    "y_true",
                    "y_pred",
                    "probabilities",
                }
                or str(key).startswith("prob_")
            ):
                continue

            result[key] = value

        return result

    def json_safe(value):
        """Recursively convert NumPy/PyTorch objects to JSON-safe values."""
        if isinstance(value, dict):
            return {
                str(k): json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return [json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        return value

    final_json = {
        "best_epoch": int(
            best_epoch
        ),
        "best_val_qwk": float(
            best_val_qwk
        ),
        "dataset": {
            "root": str(
                dataset_root
            ),
            "train": len(
                train_dataset
            ),
            "val": len(
                val_dataset
            ),
            "test": len(
                test_dataset
            ),
        },
        "model": model_info,
        "final_validation": (
            strip_runtime_objects(
                final_val
            )
        ),
        "final_test": (
            strip_runtime_objects(
                final_test
            )
        ),
        "gradcam_reports_generated": len(
            gradcam_reports
        ),
        "configuration": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "workers": args.workers,
            "seed": args.seed,
            "patience": args.patience,
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "stage3_epochs": max(
                0,
                args.epochs
                -
                args.stage1_epochs
                -
                args.stage2_epochs,
            ),
            "smoke_test": bool(
                args.smoke_test
            ),
        },
    }

    (
        OUTPUT_ROOT /
        "final_metrics.json"
    ).write_text(
        json.dumps(
            json_safe(final_json),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RUN COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Best checkpoint : "
        f"{best_checkpoint_path}"
    )

    print(
        f"History         : "
        f"{OUTPUT_ROOT / 'training_history.csv'}"
    )

    print(
        f"Test predictions: "
        f"{OUTPUT_ROOT / 'test_predictions.csv'}"
    )

    print(
        f"Class metrics   : "
        f"{OUTPUT_ROOT / 'test_per_class_metrics.csv'}"
    )

    print(
        f"Final metrics   : "
        f"{OUTPUT_ROOT / 'final_metrics.json'}"
    )

    print(
        f"Grad-CAM folder : "
        f"{GRADCAM_ROOT}"
    )

    print(
        "=" * 100
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        print(
            "\n[STOPPED] Training interrupted by user."
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            "\n[FATAL ERROR]"
        )
        print(
            str(exc)
        )
        raise
