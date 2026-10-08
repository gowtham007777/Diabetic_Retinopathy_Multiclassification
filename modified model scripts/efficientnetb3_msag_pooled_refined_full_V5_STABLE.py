
"""
EfficientNet-B3 + MSAG + Anti-Overfitting
Pooled APTOS + DDR
Full training / evaluation / Grad-CAM + vascular explainability

IMPORTANT DESIGN
----------------
MODEL PIPELINE
    Raw RGB
      -> resize to 512x512
      -> ImageNet normalization
      -> EfficientNet-B3 initialized from the requested DR checkpoint
      -> MSAG
      -> residual fusion (learnable, initialized to preserve checkpoint behavior)
      -> ORIGINAL checkpoint attention
      -> ORIGINAL checkpoint feature normalization
      -> ORIGINAL checkpoint severity classifier
      -> 5-class DR prediction

EXPLAINABILITY PIPELINE
    Raw RGB
      -> circular retinal crop
      -> black/background removal
      -> Graham normalization
      -> vessel map
      -> vessel density
      -> train-derived QC threshold
      -> Grad-CAM + vascular report

The explainability preprocessing is deliberately NOT fed to the classifier.
This avoids changing the input distribution expected by the selected pretrained
checkpoint.

The Grad-CAM implementation follows the uploaded explainability.py concept:
forward activation hook, full backward gradient hook, spatially averaged
gradients, weighted activation sum, ReLU, and min-max normalization.

Checkpoint architecture is reproduced from the published model structure:
timm EfficientNet-B3 backbone + channel attention + BatchNorm feature norm +
dropout + severity classifier (plus the original auxiliary heads for checkpoint
compatibility). The auxiliary heads are retained for exact checkpoint loading
but are not used by the DR severity loss.

Numerical-stability safeguards:
- pretrained BatchNorm running statistics are frozen
- discriminative low learning rates for pretrained weights
- bounded MSAG residual strength (±0.25)
- validation loss is globally averaged per sample over the full split
- validation logit/probability diagnostics are recorded

No scikit-learn dependency is required.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
DATASET_ROOT = PROJECT_ROOT / "data" / "organized" / "pooled_aptos_ddr"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "best_model_v2.pth"

RUN_ROOT = PROJECT_ROOT / "runs_EfficientNetB3_MSAG_Refined_Stable"
OUTPUT_ROOT = RUN_ROOT / "outputs"
GRADCAM_ROOT = OUTPUT_ROOT / "gradcam"
EXPLAIN_CACHE_ROOT = RUN_ROOT / "explainability_cache"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

CLASS_DIRS = [
    "0_No_DR",
    "1_Mild",
    "2_Moderate",
    "3_Severe",
    "4_Proliferative_DR",
]

NUM_CLASSES = 5
INPUT_SIZE = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Graham illumination correction.
GRAHAM_SIGMA_X = 30.0

# Vessel visualization / QC.
VESSEL_CLAHE_CLIP = 2.0
VESSEL_CLAHE_GRID = (8, 8)
VESSEL_KERNEL_SMALL = 9
VESSEL_KERNEL_LARGE = 15
QC_PERCENTILE = 5.0

# Anti-overfitting.
DROP_PATH_RATE = 0.10
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.10
GRAD_CLIP = 1.0

# Original checkpoint head dimensions for EfficientNet-B3 (1536 features).
ATTENTION_REDUCTION = 8
SEVERITY_HIDDEN = 1536 // 2
DROPOUT_ORIGINAL = 0.40
DROPOUT_SEVERITY = 0.20

# Fine-tuning stages.
STAGE1_LR = 1e-4
STAGE2_LR = 2e-5
STAGE3_LR = 5e-6
MSAG_STAGE1_LR = 3e-4
MSAG_STAGE2_LR = 2e-4
MSAG_STAGE3_LR = 5e-5

DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 6

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def sha1_short(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def infer_eye(filename: str) -> str:
    s = filename.lower()
    if re.search(r"(?:^|[_\-.])right(?:$|[_\-.])", s):
        return "Right"
    if re.search(r"(?:^|[_\-.])left(?:$|[_\-.])", s):
        return "Left"
    if re.search(r"(?:^|[_\-.])od(?:$|[_\-.])", s):
        return "Right"
    if re.search(r"(?:^|[_\-.])os(?:$|[_\-.])", s):
        return "Left"
    return "N/A"


def device_info() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[Device]")
    print(f"  Device: {device}")

    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA  : {torch.version.cuda}")

    return device


# ============================================================
# POOLED DATASET DISCOVERY
# ============================================================

def collect_dataset_records(root: Path) -> pd.DataFrame:
    rows = []

    for split in ("train", "val", "test"):
        split_root = root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split folder: {split_root}")

        for class_idx, class_dir in enumerate(CLASS_DIRS):
            class_root = split_root / class_dir
            if not class_root.exists():
                raise FileNotFoundError(f"Missing class folder: {class_root}")

            for path in sorted(class_root.rglob("*")):
                if not is_image(path):
                    continue

                rows.append(
                    {
                        "split": split,
                        "class_idx": class_idx,
                        "class_name": CLASS_NAMES[class_idx],
                        "path": str(path),
                        "image_id": path.stem,
                        "eye": infer_eye(path.stem),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root}")

    return df


# ============================================================
# MODEL INPUT PREPROCESSING
# ============================================================

def build_train_transform() -> transforms.Compose:
    # Applied to raw RGB images only.
    # These augmentations are deliberately moderate and retinal-preserving.
    return transforms.Compose(
        [
            transforms.Resize(
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                12,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.93, 1.07),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.06,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    # Matches the selected checkpoint's documented input convention:
    # resize + ImageNet normalization on the RGB fundus image.
    return transforms.Compose(
        [
            transforms.Resize(
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PooledFundusDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str,
        transform: transforms.Compose,
    ) -> None:
        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"No images in split '{split}'")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image = Image.open(row["path"]).convert("RGB")
        x = self.transform(image)

        return (
            x,
            int(row["class_idx"]),
            str(row["path"]),
            str(row["image_id"]),
            str(row["eye"]),
        )


def pooled_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor(
        [int(item[1]) for item in batch],
        dtype=torch.long,
    )

    paths = [str(item[2]) for item in batch]
    image_ids = [str(item[3]) for item in batch]
    eyes = [str(item[4]) for item in batch]

    return images, labels, paths, image_ids, eyes


# ============================================================
# EXPLAINABILITY PREPROCESSING
# ============================================================

def detect_retinal_circle(
    rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Detect the main circular fundus region and remove black/background border.
    Returns crop, circular mask and local (cx, cy, radius).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3, got {rgb.shape}")

    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    threshold = max(7, int(np.percentile(gray, 8)))
    binary = (gray > threshold).astype(np.uint8) * 255

    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        morph_kernel,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        morph_kernel,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cx = w // 2
    cy = h // 2
    radius = min(h, w) // 2 - 2

    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area > 0.12 * h * w:
            (fcx, fcy), fr = cv2.minEnclosingCircle(contour)
            candidate_radius = fr * 0.98

            if candidate_radius >= 0.28 * min(h, w):
                cx = int(round(fcx))
                cy = int(round(fcy))
                radius = int(round(candidate_radius))

    radius = max(
        2,
        min(
            radius,
            cx,
            cy,
            w - 1 - cx,
            h - 1 - cy,
        ),
    )

    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius + 1)
    y2 = min(h, cy + radius + 1)

    crop = rgb[y1:y2, x1:x2].copy()

    local_cx = cx - x1
    local_cy = cy - y1
    local_r = min(
        local_cx,
        local_cy,
        crop.shape[1] - 1 - local_cx,
        crop.shape[0] - 1 - local_cy,
    )

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(
        mask,
        (local_cx, local_cy),
        max(1, int(local_r * 0.99)),
        255,
        -1,
    )

    crop[mask == 0] = 0

    return crop, mask, (local_cx, local_cy, local_r)


def graham_normalize(
    rgb: np.ndarray,
    mask: np.ndarray,
    sigma_x: float = GRAHAM_SIGMA_X,
) -> np.ndarray:
    """
    Ben Graham style local illumination normalization:
        4*image - 4*GaussianBlur(image) + 128
    """
    blurred = cv2.GaussianBlur(
        rgb,
        ksize=(0, 0),
        sigmaX=sigma_x,
    )

    normalized = cv2.addWeighted(
        rgb,
        4.0,
        blurred,
        -4.0,
        128.0,
    )

    normalized = np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)

    normalized[mask == 0] = 0
    return normalized


def resize_rgb_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    out_rgb = cv2.resize(
        rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    out_mask = cv2.resize(
        mask,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    out_mask = (
        (out_mask > 127).astype(np.uint8) * 255
    )

    out_rgb[out_mask == 0] = 0

    return out_rgb, out_mask


def generate_vessel_map(
    processed_rgb: np.ndarray,
    retinal_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Deterministic vessel-density estimator for QC and visualization.

    It uses the green channel, local CLAHE, multi-scale black-hat filtering,
    Otsu thresholding within the retinal field, and light morphology.

    This output is NOT supplied to EfficientNet-B3.
    """
    green = processed_rgb[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=VESSEL_CLAHE_CLIP,
        tileGridSize=VESSEL_CLAHE_GRID,
    )
    enhanced = clahe.apply(green)

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (VESSEL_KERNEL_SMALL, VESSEL_KERNEL_SMALL),
    )
    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (VESSEL_KERNEL_LARGE, VESSEL_KERNEL_LARGE),
    )

    blackhat_small = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        kernel_small,
    )
    blackhat_large = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        kernel_large,
    )

    response = np.maximum(
        blackhat_small,
        blackhat_large,
    )

    retinal_pixels = response[retinal_mask > 0]

    if retinal_pixels.size == 0 or np.max(retinal_pixels) <= 0:
        return np.zeros_like(response, dtype=np.uint8), 0.0

    threshold, _ = cv2.threshold(
        retinal_pixels.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    vessel_map = (
        (response >= threshold).astype(np.uint8) * 255
    )
    vessel_map[retinal_mask == 0] = 0

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    vessel_map = cv2.morphologyEx(
        vessel_map,
        cv2.MORPH_OPEN,
        small_kernel,
    )
    vessel_map = cv2.morphologyEx(
        vessel_map,
        cv2.MORPH_CLOSE,
        small_kernel,
    )
    vessel_map[retinal_mask == 0] = 0

    vessel_pixels = np.count_nonzero(vessel_map)
    density = (
        100.0 * vessel_pixels / max(1, np.count_nonzero(retinal_mask))
    )

    return vessel_map, float(density)


def build_explainability_data(
    image_path: Path,
) -> Dict[str, np.ndarray | float]:
    bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    circular, circle_mask, _ = detect_retinal_circle(rgb)
    graham = graham_normalize(
        circular,
        circle_mask,
    )
    processed, processed_mask = resize_rgb_mask(
        graham,
        circle_mask,
    )
    vessel_map, vessel_density = generate_vessel_map(
        processed,
        processed_mask,
    )

    # Also produce the model-compatible raw RGB 512x512 image for display.
    model_input_rgb = cv2.resize(
        rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return {
        "model_input_rgb": model_input_rgb,
        "circular_rgb": circular,
        "graham_rgb": processed,
        "retinal_mask": processed_mask,
        "vessel_map": vessel_map,
        "vessel_density": vessel_density,
    }


# ============================================================
# EXACT CHECKPOINT-COMPATIBLE MODEL
# ============================================================

class CheckpointAttention(nn.Module):
    """
    Reproduces the checkpoint model card's channel attention module.
    """
    def __init__(self, feature_dim: int):
        super().__init__()

        hidden = feature_dim // ATTENTION_REDUCTION

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, pooled_4d: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_4d)


class MultiScaleSpatialAttentionGate(nn.Module):
    """
    MSAG:
      channel average/max pooling
      -> parallel 3x3/5x5/7x7 spatial branches
      -> BN
      -> 1x1 fusion
      -> sigmoid spatial gate

    Its output is the attention-refined feature map.
    """
    def __init__(
        self,
        kernel_sizes: Sequence[int] = (3, 5, 7),
    ):
        super().__init__()

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    2,
                    1,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        self.bn = nn.BatchNorm2d(len(kernel_sizes))
        self.fuse = nn.Conv2d(
            len(kernel_sizes),
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )
        max_map = torch.amax(
            x,
            dim=1,
            keepdim=True,
        )

        pooled = torch.cat(
            [avg_map, max_map],
            dim=1,
        )

        branch_outputs = [
            branch(pooled)
            for branch in self.branches
        ]

        multi_scale = torch.cat(
            branch_outputs,
            dim=1,
        )

        multi_scale = self.bn(multi_scale)

        gate = torch.sigmoid(
            self.fuse(multi_scale)
        )

        return x * gate


class RefinedDRModel(nn.Module):
    """
    The complete modified model.

    Existing checkpoint path:
        EfficientNet-B3
        -> original attention
        -> feature norm
        -> original severity classifier

    Modification:
        EfficientNet feature map
        -> MSAG
        -> learnable residual fusion
        -> original checkpoint attention
        -> original checkpoint feature norm
        -> original checkpoint severity classifier
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
                "timm is required. Install with:\n"
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

        self.feature_dim = int(self.backbone.num_features)

        if self.feature_dim != 1536:
            raise RuntimeError(
                f"Expected EfficientNet-B3 feature dimension 1536, "
                f"got {self.feature_dim}."
            )

        # New module.
        self.msag = MultiScaleSpatialAttentionGate()

        # Starts at exact identity:
        # fused = x + 0 * (MSAG(x) - x)
        # This preserves the pretrained feature representation initially.
        # Bounded residual strength. The actual alpha is limited to ±0.25
        # so MSAG cannot abruptly replace the pretrained representation.
        self.msag_alpha_raw = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        # ORIGINAL CHECKPOINT COMPONENTS.
        hidden_att = self.feature_dim // ATTENTION_REDUCTION

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                self.feature_dim,
                hidden_att,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_att,
                self.feature_dim,
            ),
            nn.Sigmoid(),
        )

        self.feature_norm = nn.BatchNorm1d(
            self.feature_dim
        )

        self.dropout = nn.Dropout(
            DROPOUT_ORIGINAL
        )

        self.severity_classifier = nn.Sequential(
            nn.Linear(
                self.feature_dim,
                self.feature_dim // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(
                DROPOUT_SEVERITY
            ),
            nn.Linear(
                self.feature_dim // 2,
                NUM_CLASSES,
            ),
        )

        # Retained for exact checkpoint compatibility.
        self.lesion_detector = nn.Sequential(
            nn.Linear(
                self.feature_dim,
                self.feature_dim // 4,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(
                self.feature_dim // 4,
                5,
            ),
        )

        self.region_predictor = nn.Sequential(
            nn.Linear(
                self.feature_dim,
                self.feature_dim // 4,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(
                self.feature_dim // 4,
                5,
            ),
        )

        # Used by the loss/reporting path only.
        self.last_feature_map: Optional[torch.Tensor] = None

    @property
    def msag_alpha(self) -> torch.Tensor:
        """Bounded residual strength in [-0.25, 0.25]."""
        return 0.25 * torch.tanh(self.msag_alpha_raw)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        features = self.backbone.forward_features(x)

        # Retain the pre-MSAG feature map for optional diagnostics.
        self.last_feature_map = features

        msag_features = self.msag(features)

        # Identity-preserving residual fusion with bounded strength.
        msag_alpha = 0.25 * torch.tanh(
            self.msag_alpha_raw
        )
        fused = features + msag_alpha * (
            msag_features - features
        )

        pooled = F.adaptive_avg_pool2d(
            fused,
            1,
        ).flatten(1)

        pooled_4d = pooled.unsqueeze(-1).unsqueeze(-1)

        attention_weights = self.attention(
            pooled_4d
        )

        attended = (
            pooled * attention_weights
        )

        normalized = self.feature_norm(
            attended
        )

        normalized = self.dropout(
            normalized
        )

        severity_logits = self.severity_classifier(
            normalized
        )

        if return_features:
            lesion_logits = self.lesion_detector(
                normalized
            )
            region_logits = self.region_predictor(
                normalized
            )

            return {
                "severity": severity_logits,
                "lesions": lesion_logits,
                "regions": region_logits,
                "features": normalized,
                "feature_map": fused,
            }

        return severity_logits


def normalize_checkpoint_state_dict(
    checkpoint,
) -> Dict[str, torch.Tensor]:
    """
    Locate the state dict and remove common wrappers.
    """
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
        )
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise RuntimeError(
            "Checkpoint does not contain a recognizable state_dict."
        )

    normalized = {}

    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue

        k = str(key)

        if k.startswith("module."):
            k = k[len("module."):]

        normalized[k] = value

    return normalized


def load_checkpoint(
    model: RefinedDRModel,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n\n"
            "Download:\n"
            "https://huggingface.co/dheeren-tejani/"
            "DiabeticRetinpathyClassifier/blob/main/best_model_v2.pth"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = normalize_checkpoint_state_dict(
        checkpoint
    )

    model_state = model.state_dict()

    compatible = {}
    mismatches = []
    unexpected = []

    for key, value in state.items():
        if key not in model_state:
            # Ignore only the new MSAG parameters.
            if key.startswith("msag.") or key.startswith("msag_alpha"):
                continue
            unexpected.append(key)
            continue

        if tuple(value.shape) != tuple(
            model_state[key].shape
        ):
            mismatches.append(
                {
                    "key": key,
                    "checkpoint": tuple(value.shape),
                    "target": tuple(model_state[key].shape),
                }
            )
            continue

        compatible[key] = value

    # Load the original checkpoint tensors into all matching components.
    missing, unexpected_after = model.load_state_dict(
        compatible,
        strict=False,
    )

    loaded_backbone = sum(
        k.startswith("backbone.")
        for k in compatible
    )

    loaded_original_attention = sum(
        k.startswith("attention.")
        for k in compatible
    )

    loaded_feature_norm = sum(
        k.startswith("feature_norm.")
        for k in compatible
    )

    loaded_severity = sum(
        k.startswith("severity_classifier.")
        for k in compatible
    )

    print("\n[Checkpoint]")
    print(f"  Path: {checkpoint_path}")
    print(f"  Raw tensors in checkpoint: {len(state)}")
    print(f"  Compatible tensors loaded: {len(compatible)}")
    print(f"  EfficientNet-B3 tensors     : {loaded_backbone}")
    print(f"  Original attention tensors  : {loaded_original_attention}")
    print(f"  Original feature norm       : {loaded_feature_norm}")
    print(f"  Original severity tensors   : {loaded_severity}")
    print(f"  Shape mismatches             : {len(mismatches)}")
    print(f"  Unexpected original keys     : {len(unexpected)}")

    new_keys = [
        k for k in model_state
        if k.startswith("msag.") or k.startswith("msag_alpha")
    ]
    print(f"  New MSAG tensors             : {len(new_keys)}")

    if mismatches:
        print("\n  First shape mismatches:")
        for item in mismatches[:10]:
            print(
                f"    - {item['key']}: "
                f"checkpoint={item['checkpoint']} "
                f"target={item['target']}"
            )
        raise RuntimeError(
            "Checkpoint tensor shape mismatch detected. "
            "Do not start full training until this is resolved."
        )

    # The backbone + original DR head are expected to be initialized.
    target_original = sum(
        k.startswith("backbone.")
        or k.startswith("attention.")
        or k.startswith("feature_norm.")
        or k.startswith("severity_classifier.")
        for k in model_state
    )

    loaded_original = sum(
        k.startswith("backbone.")
        or k.startswith("attention.")
        or k.startswith("feature_norm.")
        or k.startswith("severity_classifier.")
        for k in compatible
    )

    ratio = loaded_original / max(1, target_original)

    if ratio < 0.90:
        raise RuntimeError(
            f"Only {loaded_original}/{target_original} "
            f"({ratio*100:.1f}%) of the original checkpoint pathway "
            "loaded. Refusing to train."
        )

    return {
        "raw_tensors": len(state),
        "compatible_tensors": len(compatible),
        "loaded_backbone": loaded_backbone,
        "loaded_attention": loaded_original_attention,
        "loaded_feature_norm": loaded_feature_norm,
        "loaded_severity": loaded_severity,
        "shape_mismatches": mismatches,
        "unexpected_original_keys": unexpected,
        "load_ratio_original_pathway": ratio,
    }


# ============================================================
# BATCHNORM STABILITY
# ============================================================

def freeze_pretrained_bn_running_stats(model: RefinedDRModel) -> None:
    """
    Freeze running mean/variance of the pretrained EfficientNet BN layers
    and the original checkpoint feature_norm layer. Their affine weights and
    biases may still receive gradients when enabled by set_stage().

    This is intentionally different from `requires_grad=False`: running
    statistics are buffers and otherwise continue changing in model.train().
    The newly-created MSAG BN remains trainable.
    """
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )

    for module in model.backbone.modules():
        if isinstance(module, bn_types):
            module.eval()

    # Original checkpoint feature normalization: preserve its running stats.
    model.feature_norm.eval()


def set_stage(model: RefinedDRModel, stage: int) -> None:
    # Freeze everything first.
    for p in model.backbone.parameters():
        p.requires_grad = False

    # New MSAG + bounded residual strength always train.
    for p in model.msag.parameters():
        p.requires_grad = True

    # Train the LEAF parameter; `model.msag_alpha` is a tanh-derived tensor property.
    model.msag_alpha_raw.requires_grad = True

    # Original checkpoint DR head always train.
    for module in (
        model.attention,
        model.feature_norm,
        model.dropout,
        model.severity_classifier,
    ):
        for p in module.parameters():
            p.requires_grad = True

    if stage >= 2:
        # Last two MBConv stages plus final convolution / BN.
        if hasattr(model.backbone, "blocks"):
            for block in list(model.backbone.blocks)[-2:]:
                for p in block.parameters():
                    p.requires_grad = True

        for name in ("conv_head", "bn2"):
            module = getattr(
                model.backbone,
                name,
                None,
            )
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

    if stage >= 3:
        for p in model.backbone.parameters():
            p.requires_grad = True

    total = sum(
        p.numel()
        for p in model.parameters()
    )
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"[Fine-tuning] Stage {stage}: "
        f"trainable {trainable:,} / {total:,} "
        f"({100*trainable/max(1,total):.2f}%)"
    )


def optimizer_for_stage(
    model: RefinedDRModel,
    stage: int,
) -> torch.optim.Optimizer:
    """
    AdamW with discriminative learning rates.

    The pretrained DR pathway is already trained, so it receives a much
    smaller learning rate than the newly added MSAG module.
    """
    groups = []

    def add_group(params, lr):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({
                "params": params,
                "lr": lr,
            })

    msag_params = list(model.msag.parameters())
    msag_params.append(model.msag_alpha_raw)

    original_head_modules = [
        model.attention,
        model.feature_norm,
        model.dropout,
        model.severity_classifier,
    ]
    original_head_params = []
    for module in original_head_modules:
        original_head_params.extend(list(module.parameters()))

    backbone_params = []
    if stage >= 2:
        if hasattr(model.backbone, "blocks"):
            for block in list(model.backbone.blocks)[-2:]:
                backbone_params.extend(list(block.parameters()))
        for name in ("conv_head", "bn2"):
            module = getattr(model.backbone, name, None)
            if module is not None:
                backbone_params.extend(list(module.parameters()))

    if stage >= 3:
        backbone_params = list(model.backbone.parameters())

    if stage == 1:
        add_group(msag_params, MSAG_STAGE1_LR)
        add_group(original_head_params, STAGE1_LR)
    elif stage == 2:
        add_group(msag_params, MSAG_STAGE2_LR)
        add_group(original_head_params, STAGE1_LR)
        add_group(backbone_params, STAGE2_LR)
    else:
        add_group(msag_params, MSAG_STAGE3_LR)
        add_group(original_head_params, STAGE2_LR)
        add_group(backbone_params, STAGE3_LR)

    return torch.optim.AdamW(
        groups,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )


# ============================================================
# LOSS / METRICS
# ============================================================

def sqrt_inverse_class_weights(
    labels: Sequence[int],
) -> torch.Tensor:
    labels = np.asarray(
        labels,
        dtype=int,
    )

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES,
    ).astype(np.float64)

    total = float(counts.sum())

    weights = np.sqrt(
        total /
        np.maximum(
            1.0,
            NUM_CLASSES * counts,
        )
    )

    weights /= np.mean(weights)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def weighted_label_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Stable PyTorch CE with optional per-sample reduction.

    `reduction="none"` is used for validation so the final epoch loss is
    computed as one global sample-weighted mean across the entire split,
    rather than averaging already-averaged batch losses.
    """
    return F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
        reduction=reduction,
    )


def confusion_matrix(
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
        if 0 <= true < NUM_CLASSES and 0 <= pred < NUM_CLASSES:
            cm[true, pred] += 1

    return cm


def class_prf(
    cm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    precision = tp / np.maximum(
        1.0,
        tp + fp,
    )
    recall = tp / np.maximum(
        1.0,
        tp + fn,
    )

    f1 = (
        2 * precision * recall /
        np.maximum(
            1e-12,
            precision + recall,
        )
    )

    return precision, recall, f1


def qwk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    cm = confusion_matrix(
        y_true,
        y_pred,
    ).astype(float)

    actual = np.bincount(
        y_true.astype(int),
        minlength=NUM_CLASSES,
    ).astype(float)

    predicted = np.bincount(
        y_pred.astype(int),
        minlength=NUM_CLASSES,
    ).astype(float)

    expected = np.outer(
        actual,
        predicted,
    ) / max(
        1.0,
        len(y_true),
    )

    weights = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=float,
    )

    denom = float(
        (NUM_CLASSES - 1) ** 2
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            weights[i, j] = (
                (i - j) ** 2 / denom
            )

    observed_disagreement = np.sum(
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
        observed_disagreement /
        expected_disagreement
    )


def binary_auc(
    binary_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    positive = scores[
        binary_true == 1
    ]
    negative = scores[
        binary_true == 0
    ]

    if len(positive) == 0 or len(negative) == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]

    ranks = np.zeros(
        len(scores),
        dtype=float,
    )

    start = 0
    while start < len(scores):
        end = start + 1

        while (
            end < len(scores)
            and sorted_scores[end] ==
            sorted_scores[start]
        ):
            end += 1

        avg_rank = (
            (start + 1) + end
        ) / 2.0

        ranks[
            order[start:end]
        ] = avg_rank

        start = end

    positive_rank_sum = ranks[
        binary_true == 1
    ].sum()

    n_pos = len(positive)
    n_neg = len(negative)

    return float(
        (
            positive_rank_sum -
            n_pos * (n_pos + 1) / 2
        ) /
        (n_pos * n_neg)
    )


def multiclass_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, np.ndarray]:
    scores = []

    for class_idx in range(NUM_CLASSES):
        binary = (
            y_true == class_idx
        ).astype(int)

        scores.append(
            binary_auc(
                binary,
                probabilities[:, class_idx],
            )
        )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if np.all(np.isnan(scores)):
        macro = float("nan")
    else:
        macro = float(
            np.nanmean(scores)
        )

    return macro, scores


def build_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> Dict:
    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    precision, recall, f1 = class_prf(cm)

    support = cm.sum(axis=1)

    specificity = []

    for c in range(NUM_CLASSES):
        tp = float(cm[c, c])
        fn = float(
            cm[c, :].sum() - cm[c, c]
        )
        fp = float(
            cm[:, c].sum() - cm[c, c]
        )
        tn = float(
            cm.sum() -
            tp -
            fn -
            fp
        )

        specificity.append(
            tn /
            max(
                1.0,
                tn + fp,
            )
        )

    auc_macro, auc_per_class = (
        multiclass_auc(
            y_true,
            probabilities,
        )
    )

    accuracy = float(
        np.mean(y_true == y_pred)
    )

    macro_f1 = float(
        np.mean(f1)
    )

    weighted_f1 = float(
        np.average(
            f1,
            weights=support,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk(
            y_true,
            y_pred,
        ),
        "macro_auc": auc_macro,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "specificity": specificity,
        "f1": f1.tolist(),
        "auc_per_class": auc_per_class.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def train_one_epoch(
    model: RefinedDRModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    class_weights: torch.Tensor,
    device: torch.device,
    accumulation_steps: int = 1,
    epoch_label: str = "",
) -> Tuple[float, float, Dict[str, float]]:
    model.train()

    # Critical: prevent pretrained BN running statistics from drifting.
    freeze_pretrained_bn_running_stats(model)

    running_loss_sum = 0.0
    running_n = 0
    correct = 0
    max_abs_logit = 0.0

    optimizer.zero_grad(set_to_none=True)

    use_amp = device.type == "cuda"
    steps_since_update = 0

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=epoch_label or "Training",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_index, batch in progress:
        x, y = batch[0], batch[1]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            per_sample_loss = weighted_label_smoothed_ce(
                logits,
                y,
                class_weights,
                reduction="none",
            )
            loss = per_sample_loss.mean()
            loss_for_backward = loss / accumulation_steps

        scaler.scale(loss_for_backward).backward()
        steps_since_update += 1

        if (
            steps_since_update >= accumulation_steps
            or batch_index == len(loader) - 1
        ):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps_since_update = 0

        bs = y.size(0)
        running_loss_sum += float(per_sample_loss.detach().sum().item())
        running_n += bs
        correct += int(
            (logits.argmax(dim=1) == y).sum().item()
        )
        max_abs_logit = max(
            max_abs_logit,
            float(torch.abs(logits.detach()).max().item()),
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / max(1, running_n):.1f}%",
            maxlogit=f"{max_abs_logit:.1f}",
            refresh=False,
        )

    return (
        running_loss_sum / max(1, running_n),
        correct / max(1, running_n),
        {
            "max_abs_logit": max_abs_logit,
        },
    )


@torch.no_grad()
def evaluate(
    model: RefinedDRModel,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> Dict:
    model.eval()

    running_loss_sum = 0.0
    running_n = 0

    true_list = []
    pred_list = []
    prob_list = []
    path_list = []
    id_list = []
    eye_list = []

    max_abs_logit = 0.0
    min_true_probability = 1.0
    true_probability_sum = 0.0
    true_probability_n = 0

    progress = tqdm(
        loader,
        total=len(loader),
        desc="Validation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for batch in progress:
        x, y, paths, image_ids, eyes = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        per_sample_loss = weighted_label_smoothed_ce(
            logits,
            y,
            class_weights,
            reduction="none",
        )

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        bs = y.size(0)
        running_loss_sum += float(per_sample_loss.sum().item())
        running_n += bs

        true_probability = probabilities.gather(
            1,
            y.unsqueeze(1),
        ).squeeze(1)

        min_true_probability = min(
            min_true_probability,
            float(true_probability.min().item()),
        )
        true_probability_sum += float(
            true_probability.sum().item()
        )
        true_probability_n += bs

        max_abs_logit = max(
            max_abs_logit,
            float(torch.abs(logits).max().item()),
        )

        true_list.append(y.cpu().numpy())
        pred_list.append(predictions.cpu().numpy())
        prob_list.append(probabilities.cpu().numpy())
        path_list.extend(paths)
        id_list.extend(image_ids)
        eye_list.extend(eyes)

        progress.set_postfix(
            loss=f"{running_loss_sum / max(1, running_n):.4f}",
            maxlogit=f"{max_abs_logit:.1f}",
            refresh=False,
        )

    y_true = np.concatenate(true_list)
    y_pred = np.concatenate(pred_list)
    probabilities = np.concatenate(prob_list)

    metrics = build_metrics(
        y_true,
        y_pred,
        probabilities,
    )

    metrics["loss"] = running_loss_sum / max(1, running_n)
    metrics["max_abs_logit"] = max_abs_logit
    metrics["min_true_probability"] = min_true_probability
    metrics["mean_true_probability"] = (
        true_probability_sum / max(1, true_probability_n)
    )

    prediction_df = pd.DataFrame(
        {
            "path": path_list,
            "image_id": id_list,
            "eye": eye_list,
            "true_class": y_true,
            "true_label": [CLASS_NAMES[i] for i in y_true],
            "pred_class": y_pred,
            "pred_label": [CLASS_NAMES[i] for i in y_pred],
            "confidence": probabilities.max(axis=1),
        }
    )

    for class_idx in range(NUM_CLASSES):
        prediction_df[f"prob_{class_idx}"] = probabilities[:, class_idx]

    metrics["predictions_df"] = prediction_df
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["probabilities"] = probabilities

    return metrics


def print_metrics(
    title: str,
    metrics: Dict,
) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)

    print(
        f"Loss         : {metrics.get('loss', float('nan')):.4f}"
    )
    print(
        f"Accuracy     : {metrics['accuracy']*100:.2f}%"
    )
    print(
        f"Macro F1     : {metrics['macro_f1']*100:.2f}%"
    )
    print(
        f"Weighted F1  : {metrics['weighted_f1']*100:.2f}%"
    )
    print(
        f"QWK          : {metrics['qwk']:.4f}"
    )
    print(
        f"Macro ROC-AUC: {metrics['macro_auc']:.4f}"
    )

    print(
        "\n"
        f"{'Class':22s}"
        f"{'Prec':>8s}"
        f"{'Recall':>8s}"
        f"{'Spec':>8s}"
        f"{'F1':>8s}"
        f"{'AUC':>8s}"
        f"{'N':>7s}"
    )

    for c in range(NUM_CLASSES):
        print(
            f"{CLASS_NAMES[c]:22s}"
            f"{metrics['precision'][c]:8.3f}"
            f"{metrics['recall'][c]:8.3f}"
            f"{metrics['specificity'][c]:8.3f}"
            f"{metrics['f1'][c]:8.3f}"
            f"{metrics['auc_per_class'][c]:8.3f}"
            f"{metrics['support'][c]:7d}"
        )

    print("\nConfusion matrix:")
    print(
        np.asarray(
            metrics["confusion_matrix"]
        )
    )


# ============================================================
# GRAD-CAM — UPLOADED SCRIPT CONCEPT
# ============================================================

class GradCAM:
    """
    Adapted from the uploaded explainability.py concept.

    The target layer is a 4D feature-producing module. For EfficientNet-B3
    we use the final convolutional `conv_head`, matching the uploaded
    explainability.py architecture-specific concept.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        self.model = model
        self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output,
    ):
        self.gradients = grad_output[0]

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, torch.Tensor]:

        # Important for Stage-1 when the backbone is frozen:
        # the input still participates in a differentiable graph so the
        # target convolutional activations can receive gradients.
        x = (
            x.detach()
            .clone()
            .requires_grad_(True)
        )

        self.model.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():
            output = self.model(x)

        if class_idx is None:
            class_idx = int(
                torch.argmax(
                    output,
                    dim=1,
                ).item()
            )

        score = output[
            0,
            class_idx,
        ]

        score.backward()

        if (
            self.gradients is None
            or self.activations is None
        ):
            raise RuntimeError(
                "Grad-CAM hooks did not capture "
                "activations/gradients."
            )

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True,
        )

        cam = torch.sum(
            weights * activations,
            dim=1,
            keepdim=True,
        )

        cam = torch.relu(
            cam
        )

        cam = cam - torch.min(
            cam
        )

        cam = cam / (
            torch.max(cam) +
            1e-7
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        return (
            cam.detach()
            .cpu()
            .numpy()[0, 0],
            class_idx,
            probabilities.detach(),
        )

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# VISUALIZATION
# ============================================================

def overlay_heatmap(
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
        heatmap * 255,
        0,
        255,
    ).astype(np.uint8)

    color_bgr = cv2.applyColorMap(
        heatmap_u8,
        cv2.COLORMAP_JET,
    )

    color_rgb = cv2.cvtColor(
        color_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return cv2.addWeighted(
        rgb,
        1.0 - alpha,
        color_rgb,
        alpha,
        0,
    )


def save_confusion_matrix_plot(
    cm: np.ndarray,
    path: Path,
    normalize: bool = False,
) -> None:
    values = cm.astype(float)

    if normalize:
        values = (
            values /
            np.maximum(
                1.0,
                values.sum(
                    axis=1,
                    keepdims=True,
                ),
            )
        )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        values
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )
    ax.set_yticks(
        range(NUM_CLASSES)
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
        "Normalized Confusion Matrix"
        if normalize
        else "Confusion Matrix"
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text = (
                f"{values[i, j]:.2f}"
                if normalize
                else f"{int(values[i, j])}"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_class_metrics_csv(
    metrics: Dict,
    path: Path,
) -> None:
    rows = []

    for c in range(NUM_CLASSES):
        rows.append(
            {
                "class_id": c,
                "class": CLASS_NAMES[c],
                "precision": metrics["precision"][c],
                "recall_sensitivity": metrics["recall"][c],
                "specificity": metrics["specificity"][c],
                "f1": metrics["f1"][c],
                "auc_ovr": metrics["auc_per_class"][c],
                "support": metrics["support"][c],
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        path,
        index=False,
    )


def save_history_plots(
    history: pd.DataFrame,
    path: Path,
) -> None:
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
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=170,
    )
    plt.close(fig)

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
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path.with_name(
            path.stem +
            "_performance.png"
        ),
        dpi=170,
    )
    plt.close(fig)


def make_gradcam_report(
    model: RefinedDRModel,
    row: pd.Series,
    device: torch.device,
    qc_threshold: float,
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
            f"Could not read {image_path}"
        )

    original_rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    explainability = build_explainability_data(
        image_path
    )

    model_input_rgb = explainability[
        "model_input_rgb"
    ]

    graham_rgb = explainability[
        "graham_rgb"
    ]

    vessel_map = explainability[
        "vessel_map"
    ]

    vessel_density = float(
        explainability[
            "vessel_density"
        ]
    )

    model_tensor = (
        build_eval_transform()(
            Image.fromarray(
                original_rgb
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    # Target the final EfficientNet-B3 convolutional feature layer.
    # This follows the uploaded explainability.py concept for EfficientNet,
    # which selects the final convolutional feature representation for Grad-CAM.
    # MSAG remains active in the forward path and therefore still influences
    # the gradients that reach this target layer.
    target_layer = model.backbone.conv_head

    cam = GradCAM(
        model,
        target_layer,
    )

    try:
        heatmap, pred_cls, probability_tensor = (
            cam(
                model_tensor,
                class_idx=None,
            )
        )
    finally:
        cam.close()

    probabilities = (
        probability_tensor[0]
        .detach()
        .cpu()
        .numpy()
    )

    confidence = float(
        probabilities[pred_cls]
    )

    true_cls = int(
        row["true_class"]
    )

    correct = (
        pred_cls == true_cls
    )

    status = (
        "CORRECT"
        if correct
        else "INCORRECT"
    )

    qc_pass = (
        vessel_density >=
        qc_threshold
    )

    cam_overlay = overlay_heatmap(
        model_input_rgb,
        heatmap,
        alpha=0.45,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig = plt.figure(
        figsize=(20, 11)
    )

    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 0.95],
        hspace=0.30,
        wspace=0.18,
    )

    ax1 = fig.add_subplot(
        gs[0, 0]
    )
    ax2 = fig.add_subplot(
        gs[0, 1]
    )
    ax3 = fig.add_subplot(
        gs[0, 2]
    )
    ax4 = fig.add_subplot(
        gs[1, 0:2]
    )
    ax5 = fig.add_subplot(
        gs[1, 2:4]
    )
    ax6 = fig.add_subplot(
        gs[0, 3]
    )

    # 1. Original.
    ax1.imshow(
        original_rgb
    )
    ax1.set_title(
        "Original Fundus Input",
        fontsize=13,
        fontweight="bold",
    )
    ax1.axis("off")

    # 2. MODEL PIPELINE input.
    ax2.imshow(
        model_input_rgb
    )
    ax2.set_title(
        "Model Input\nRaw RGB → 512×512",
        fontsize=12,
        fontweight="bold",
    )
    ax2.axis("off")

    # 3. Explainability preprocessing.
    ax3.imshow(
        graham_rgb
    )
    ax3.set_title(
        "Explainability Preprocessing\n"
        "Circular Crop + Graham Normalization",
        fontsize=11,
        fontweight="bold",
    )
    ax3.axis("off")

    # 4. Grad-CAM.
    ax4.imshow(
        cam_overlay
    )
    ax4.set_title(
        f"Grad-CAM\n"
        f"Predicted Class: {CLASS_NAMES[pred_cls]}\n"
        f"Confidence: {confidence*100:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax4.axis("off")

    # 5. Confidence distribution.
    bars = ax5.barh(
        np.arange(NUM_CLASSES),
        probabilities * 100.0,
    )

    ax5.set_yticks(
        np.arange(NUM_CLASSES)
    )
    ax5.set_yticklabels(
        CLASS_NAMES
    )
    ax5.invert_yaxis()
    ax5.set_xlim(
        0,
        100,
    )

    ax5.set_xlabel(
        "Probability (%)"
    )
    ax5.set_title(
        "Prediction Confidence by Class",
        fontsize=13,
        fontweight="bold",
    )

    for bar, prob in zip(
        bars,
        probabilities,
    ):
        ax5.text(
            min(
                96,
                bar.get_width() + 1.0,
            ),
            bar.get_y() +
            bar.get_height() / 2,
            f"{prob*100:.1f}%",
            va="center",
            fontsize=9,
        )

    # 6. Vessel analysis.
    ax6.imshow(
        vessel_map,
        cmap="gray",
    )

    ax6.set_title(
        f"Vascular Analysis\n"
        f"Vessel Density: {vessel_density:.2f}%\n"
        f"QC: {'PASS' if qc_pass else 'FAIL'}",
        fontsize=12,
        fontweight="bold",
    )
    ax6.axis("off")

    image_id = str(
        row["image_id"]
    )
    eye = str(
        row["eye"]
    )

    title = (
        f"Image ID: {image_id}   |   "
        f"Eye: {eye}   |   "
        f"True Grade: {true_cls} "
        f"({CLASS_NAMES[true_cls]})   |   "
        f"Prediction: {pred_cls} "
        f"({CLASS_NAMES[pred_cls]})   |   "
        f"Confidence: {confidence*100:.1f}%   |   "
        f"Status: {status}   |   "
        f"Vessel Density: {vessel_density:.2f}%   |   "
        f"QC: {'PASS' if qc_pass else 'FAIL'}"
    )

    fig.suptitle(
        title,
        fontsize=12,
        fontweight="bold",
        y=0.988,
    )

    # Bottom text summary.
    probability_text = "\n".join(
        f"{CLASS_NAMES[i]:18s} "
        f"{probabilities[i]*100:5.1f}%"
        for i in range(NUM_CLASSES)
    )

    fig.text(
        0.70,
        0.02,
        probability_text,
        fontsize=8,
        family="monospace",
        va="bottom",
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "path": str(image_path),
        "image_id": image_id,
        "pred_class": pred_cls,
        "pred_label": CLASS_NAMES[pred_cls],
        "true_class": true_cls,
        "true_label": CLASS_NAMES[true_cls],
        "confidence": confidence,
        "status": status,
        "vessel_density": vessel_density,
        "qc_pass": qc_pass,
        "probabilities": probabilities.tolist(),
    }


# ============================================================
# SMOKE TEST HELPERS
# ============================================================

def balanced_subset(
    dataset: PooledFundusDataset,
    per_class: int,
) -> Subset:
    indices = []

    for class_idx in range(NUM_CLASSES):
        class_indices = dataset.df.index[
            dataset.df["class_idx"].astype(int) ==
            class_idx
        ].tolist()

        if len(class_indices) < per_class:
            raise RuntimeError(
                f"Class {class_idx} has only "
                f"{len(class_indices)} samples."
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EfficientNet-B3 + MSAG + anti-overfitting "
            "on pooled APTOS+DDR."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DATASET_ROOT),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
        help=(
            "Gradient accumulation. Effective batch = "
            "batch-size × accumulation-steps."
        ),
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
        default=3,
    )

    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 epoch with only a few training batches.",
    )

    args = parser.parse_args()

    if args.accumulation_steps < 1:
        raise ValueError(
            "--accumulation-steps must be >= 1"
        )

    seed_everything(
        args.seed
    )

    dataset_root = Path(
        args.dataset_root
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    safe_mkdir(
        OUTPUT_ROOT
    )
    safe_mkdir(
        GRADCAM_ROOT
    )
    safe_mkdir(
        EXPLAIN_CACHE_ROOT
    )

    print("\n" + "=" * 100)
    print(
        "EFFICIENTNET-B3 + MSAG + ANTI-OVERFITTING — "
        "REFINED POOLED APTOS + DDR"
    )
    print("=" * 100)
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Dataset root      : {dataset_root}")
    print(f"Checkpoint        : {checkpoint_path}")
    print(f"Model input       : Raw RGB → {INPUT_SIZE}×{INPUT_SIZE}")
    print("Split             : Existing train / val / test")
    print("Re-split          : OFF")
    print("Train oversample  : OFF")
    print("Model QC filtering: OFF")
    print(
        "Explainability QC : Train-derived vessel-response-density threshold"
    )
    print("MSAG              : ON")
    print("MSAG fusion       : Identity-preserving residual")
    print(
        f"MSAG alpha init   : 0.0000"
    )
    print(
        f"DropPath          : {DROP_PATH_RATE:.2f}"
    )
    print(
        f"Stable BN stats   : PRETRAINED BN RUNNING STATS FROZEN"
    )
    print(
        f"AdamW             : weight_decay={WEIGHT_DECAY:.1e}"
    )
    print(
        f"Label smoothing   : {LABEL_SMOOTHING:.2f}"
    )
    print(
        f"Gradient clip     : {GRAD_CLIP:.2f}"
    )
    print(
        f"Effective batch   : "
        f"{args.batch_size * args.accumulation_steps}"
    )
    print("Early stopping    : Validation QWK")
    print("LR scheduler      : ReduceLROnPlateau on global validation loss")
    print(
        "Grad-CAM target   : EfficientNet-B3 final conv_head"
    )

    if args.smoke_test:
        print(
            "Smoke evaluation  : Balanced 4 samples/class"
        )

    print("=" * 100)

    device = device_info()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    manifest = collect_dataset_records(
        dataset_root
    )

    print("\n[Dataset — pooled split]")
    print(
        manifest.groupby(
            ["split", "class_name"]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .to_string()
    )

    train_df = manifest[
        manifest["split"] == "train"
    ].copy()

    class_weights = sqrt_inverse_class_weights(
        train_df["class_idx"].astype(int).to_numpy()
    ).to(device)

    print("\n[Loss class weights]")
    for c, weight in enumerate(
        class_weights.cpu().numpy()
    ):
        print(
            f"  {c} {CLASS_NAMES[c]:20s}: {weight:.4f}"
        )

    train_dataset = PooledFundusDataset(
        manifest,
        "train",
        build_train_transform(),
    )

    val_dataset = PooledFundusDataset(
        manifest,
        "val",
        build_eval_transform(),
    )

    test_dataset = PooledFundusDataset(
        manifest,
        "test",
        build_eval_transform(),
    )

    print("\n[Final model dataset sizes]")
    print(
        f"  Train: {len(train_dataset)}"
    )
    print(
        f"  Val  : {len(val_dataset)}"
    )
    print(
        f"  Test : {len(test_dataset)}"
    )

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": pooled_collate,
    }

    if args.workers > 0:
        loader_kwargs[
            "persistent_workers"
        ] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # --------------------------------------------------------
    # Model + checkpoint
    # --------------------------------------------------------
    model = RefinedDRModel(
        drop_path_rate=DROP_PATH_RATE,
    ).to(device)

    checkpoint_info = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model_summary = {
        "feature_dim": model.feature_dim,
        "total_parameters": sum(
            p.numel()
            for p in model.parameters()
        ),
        "checkpoint_load": checkpoint_info,
        "model_input": {
            "size": INPUT_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "msag": {
            "enabled": True,
            "kernels": [3, 5, 7],
            "residual_identity_initialization": True,
            "alpha_initial": 0.0,
            "alpha_max_abs": 0.25,
        },
        "anti_overfitting": {
            "drop_path_rate": DROP_PATH_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "gradient_clip": GRAD_CLIP,
            "staged_finetuning": True,
        },
    }

    (OUTPUT_ROOT / "model_summary.json").write_text(
        json.dumps(
            model_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Smoke-test loaders
    # --------------------------------------------------------
    if args.smoke_test:
        class LimitedLoader:
            def __init__(
                self,
                loader,
                max_batches,
            ):
                self.loader = loader
                self.max_batches = max_batches

            def __iter__(self):
                for i, batch in enumerate(
                    self.loader
                ):
                    if i >= self.max_batches:
                        break
                    yield batch

            def __len__(self):
                return min(
                    len(self.loader),
                    self.max_batches,
                )

        train_eval_loader = LimitedLoader(
            train_loader,
            4,
        )

        val_smoke_dataset = balanced_subset(
            val_dataset,
            4,
        )

        test_smoke_dataset = balanced_subset(
            test_dataset,
            4,
        )

        smoke_loader_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": args.workers,
            "pin_memory": device.type == "cuda",
            "collate_fn": pooled_collate,
        }

        if args.workers > 0:
            smoke_loader_kwargs[
                "persistent_workers"
            ] = True

        val_eval_loader = DataLoader(
            val_smoke_dataset,
            **smoke_loader_kwargs,
        )

        test_eval_loader = DataLoader(
            test_smoke_dataset,
            **smoke_loader_kwargs,
        )

        print(
            "\n[Smoke test evaluation] "
            "4 samples/class × 5 classes = "
            "20 val + 20 test samples"
        )

        effective_epochs = 1
    else:
        train_eval_loader = train_loader
        val_eval_loader = val_loader
        test_eval_loader = test_loader
        effective_epochs = args.epochs

    # --------------------------------------------------------
    # Training setup
    # --------------------------------------------------------
    set_stage(
        model,
        1,
    )

    optimizer = optimizer_for_stage(
        model,
        stage=1,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
    )
    print("[Scheduler] ReduceLROnPlateau mode=min (validation loss)")

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_qwk = -float("inf")
    best_epoch = 0
    bad_epochs = 0

    best_model_path = (
        OUTPUT_ROOT /
        "best_model_qwk.pth"
    )

    history_rows = []

    current_stage = 1

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------
    for epoch in range(
        1,
        effective_epochs + 1,
    ):
        if args.smoke_test:
            desired_stage = 1
        elif epoch <= args.stage1_epochs:
            desired_stage = 1
        elif epoch <= (
            args.stage1_epochs +
            args.stage2_epochs
        ):
            desired_stage = 2
        else:
            desired_stage = 3

        if desired_stage != current_stage:
            if desired_stage == 1:
                lr = STAGE1_LR
            elif desired_stage == 2:
                lr = STAGE2_LR
            else:
                lr = STAGE3_LR

            print(
                "\n" + "-" * 100
            )
            print(
                f"STARTING FINE-TUNING STAGE "
                f"{desired_stage} AT EPOCH {epoch}"
            )
            print(
                "-" * 100
            )

            set_stage(
                model,
                desired_stage,
            )

            optimizer = optimizer_for_stage(
                model,
                stage=desired_stage,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.3,
                patience=2,
                min_lr=1e-7,
            )

            current_stage = desired_stage

        start_time = time.time()

        train_loss, train_acc, train_diag = train_one_epoch(
            model=model,
            loader=train_eval_loader,
            optimizer=optimizer,
            scaler=scaler,
            class_weights=class_weights,
            device=device,
            accumulation_steps=args.accumulation_steps,
            epoch_label=(
                f"Epoch {epoch:03d}/{effective_epochs:03d} "
                f"Stage {current_stage} - Train"
            ),
        )

        val_metrics = evaluate(
            model=model,
            loader=val_eval_loader,
            class_weights=class_weights,
            device=device,
        )

        scheduler.step(
            val_metrics["loss"]
        )

        elapsed = (
            time.time() - start_time
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history_rows.append(
            {
                "epoch": epoch,
                "stage": current_stage,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_qwk": val_metrics["qwk"],
                "val_auc": val_metrics["macro_auc"],
                "epoch_seconds": elapsed,
                "msag_alpha": float(
                    model.msag_alpha.detach().cpu().item()
                ),
                "train_max_abs_logit": train_diag["max_abs_logit"],
                "val_max_abs_logit": val_metrics["max_abs_logit"],
                "val_min_true_probability": val_metrics["min_true_probability"],
                "val_mean_true_probability": val_metrics["mean_true_probability"],
            }
        )

        print(
            f"\nEpoch {epoch:03d}/{effective_epochs:03d} | "
            f"Stage {current_stage} | "
            f"LR {current_lr:.2e} | "
            f"Time {elapsed:.1f}s"
        )

        print(
            f"  Train: loss={train_loss:.4f} "
            f"acc={train_acc*100:.2f}%"
        )

        print(
            f"  Val  : loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']*100:.2f}% "
            f"macroF1={val_metrics['macro_f1']*100:.2f}% "
            f"QWK={val_metrics['qwk']:.4f} "
            f"AUC={val_metrics['macro_auc']:.4f}"
        )

        print(
            f"  Diagnostics: train max|logit|={train_diag['max_abs_logit']:.2f}, "
            f"val max|logit|={val_metrics['max_abs_logit']:.2f}, "
            f"val min P(true)={val_metrics['min_true_probability']:.3e}, "
            f"val mean P(true)={val_metrics['mean_true_probability']:.4f}"
        )

        print(
            f"  MSAG residual alpha: "
            f"{model.msag_alpha.detach().cpu().item():.6f} "
            f"(bounded ±0.25)"
        )
        print(
            f"  Logit diagnostics: "
            f"train max|logit|={train_diag['max_abs_logit']:.2f}, "
            f"val max|logit|={val_metrics['max_abs_logit']:.2f}, "
            f"val min P(true)={val_metrics['min_true_probability']:.3e}"
        )

        # In a smoke test, save a diagnostic checkpoint only.
        if args.smoke_test:
            smoke_path = (
                OUTPUT_ROOT /
                "smoke_test_model.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "checkpoint_source": str(
                        checkpoint_path
                    ),
                    "preprocessing": {
                        "model": (
                            "raw RGB -> 512 -> "
                            "ImageNet normalization"
                        ),
                        "explainability": (
                            "circular crop -> "
                            "Graham normalization -> "
                            "vessel response map"
                        ),
                    },
                },
                smoke_path,
            )

            best_model_path = smoke_path
            best_epoch = epoch
            best_qwk = val_metrics["qwk"]

            print(
                "  ✓ Smoke-test checkpoint saved."
            )

        else:
            improved = (
                val_metrics["qwk"] >
                best_qwk + 1e-5
                or (
                    abs(
                        val_metrics["qwk"] -
                        best_qwk
                    ) <= 1e-5
                    and val_metrics["loss"] <
                    float(
                        history_rows[
                            best_epoch - 1
                        ]["val_loss"]
                    )
                    if best_epoch > 0
                    else False
                )
            )

            if improved or best_epoch == 0:
                best_qwk = val_metrics["qwk"]
                best_epoch = epoch
                bad_epochs = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_qwk": best_qwk,
                        "class_names": CLASS_NAMES,
                        "source_checkpoint": str(
                            checkpoint_path
                        ),
                        "model_summary": model_summary,
                    },
                    best_model_path,
                )

                print(
                    f"  ✓ NEW BEST CHECKPOINT: "
                    f"val QWK={best_qwk:.4f}"
                )

            else:
                bad_epochs += 1

                print(
                    f"  No improvement: "
                    f"{bad_epochs}/{args.patience}"
                )

                if bad_epochs >= args.patience:
                    print(
                        "\nEarly stopping triggered."
                    )
                    break

    history = pd.DataFrame(
        history_rows
    )

    history.to_csv(
        OUTPUT_ROOT /
        "training_history.csv",
        index=False,
    )

    if not history.empty:
        save_history_plots(
            history,
            OUTPUT_ROOT /
            "training_curves.png",
        )

    # --------------------------------------------------------
    # Restore best checkpoint
    # --------------------------------------------------------
    if best_model_path.exists():
        best = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            best["model_state_dict"]
        )

        if not args.smoke_test:
            best_epoch = int(
                best.get(
                    "epoch",
                    best_epoch,
                )
            )

            best_qwk = float(
                best.get(
                    "best_val_qwk",
                    best_qwk,
                )
            )

    # --------------------------------------------------------
    # Full validation / test
    # --------------------------------------------------------
    if args.smoke_test:
        print(
            "\n[Smoke test note] "
            "The complete validation/test split metrics below are "
            "diagnostic only because training used one epoch and only "
            "four training batches."
        )

        # For smoke mode use the complete val/test datasets as a pipeline
        # verification of the saved model, not the balanced tiny subsets.
        diagnostic_val_loader = val_loader
        diagnostic_test_loader = test_loader
    else:
        diagnostic_val_loader = val_loader
        diagnostic_test_loader = test_loader

    final_val = evaluate(
        model,
        diagnostic_val_loader,
        class_weights,
        device,
    )

    final_test = evaluate(
        model,
        diagnostic_test_loader,
        class_weights,
        device,
    )

    print_metrics(
        (
            "SMOKE VALIDATION RESULTS"
            if args.smoke_test
            else "FINAL VALIDATION RESULTS"
        ),
        final_val,
    )

    print_metrics(
        (
            "SMOKE TEST RESULTS"
            if args.smoke_test
            else "FINAL TEST RESULTS"
        ),
        final_test,
    )

    final_test[
        "predictions_df"
    ].to_csv(
        OUTPUT_ROOT /
        "test_predictions.csv",
        index=False,
    )

    save_class_metrics_csv(
        final_test,
        OUTPUT_ROOT /
        "test_per_class_metrics.csv",
    )

    cm = np.asarray(
        final_test["confusion_matrix"]
    )

    save_confusion_matrix_plot(
        cm,
        OUTPUT_ROOT /
        "test_confusion_matrix.png",
        normalize=False,
    )

    save_confusion_matrix_plot(
        cm,
        OUTPUT_ROOT /
        "test_confusion_matrix_normalized.png",
        normalize=True,
    )

    # --------------------------------------------------------
    # Train-derived explainability QC threshold
    # --------------------------------------------------------
    print(
        "\n[Explainability] Calculating vessel-response-density QC threshold "
        "from TRAIN split only."
    )

    train_density_values = []

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="Vessel-response QC calibration",
        unit="img",
    ):
        try:
            data = build_explainability_data(
                Path(row["path"])
            )

            train_density_values.append(
                float(
                    data["vessel_density"]
                )
            )
        except Exception as exc:
            print(
                f"\n  Warning: vessel QC failed for "
                f"{row['path']}: {exc}"
            )

    if train_density_values:
        qc_threshold = float(
            np.percentile(
                np.asarray(
                    train_density_values,
                    dtype=float,
                ),
                QC_PERCENTILE,
            )
        )
    else:
        qc_threshold = 0.0

    print(
        f"[Explainability] "
        f"Train-derived {QC_PERCENTILE:.1f}th percentile "
        f"vessel-response-density threshold: "
        f"{qc_threshold:.4f}%"
    )

    (OUTPUT_ROOT / "explainability_qc.json").write_text(
        json.dumps(
            {
                "threshold_percent": qc_threshold,
                "percentile": QC_PERCENTILE,
                "train_images_used": len(
                    train_density_values
                ),
                "important": (
                    "QC is used for explainability/reporting only. "
                    "It does not remove images from model training, "
                    "validation, or test."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Grad-CAM reports
    # --------------------------------------------------------
    print(
        "\n" + "=" * 100
    )
    print(
        "GRAD-CAM + VASCULAR EXPLAINABILITY"
    )
    print(
        "=" * 100
    )

    prediction_df = final_test[
        "predictions_df"
    ].copy()

    selected_rows = []

    # First: one sample for each true class.
    for class_idx in range(NUM_CLASSES):
        candidates = prediction_df[
            prediction_df["true_class"] ==
            class_idx
        ]

        if len(candidates) > 0:
            selected_rows.append(
                candidates.iloc[0]
            )

    # Then fill up to requested count.
    used = {
        str(row["path"])
        for row in selected_rows
    }

    for _, row in prediction_df.iterrows():
        if len(selected_rows) >= args.gradcam_samples:
            break

        if str(row["path"]) in used:
            continue

        selected_rows.append(
            row
        )
        used.add(
            str(row["path"])
        )

    gradcam_results = []

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):
        out_path = (
            GRADCAM_ROOT /
            f"gradcam_{index:03d}_"
            f"{row['image_id']}.png"
        )

        try:
            report = make_gradcam_report(
                model=model,
                row=row,
                device=device,
                qc_threshold=qc_threshold,
                output_path=out_path,
            )

            gradcam_results.append(
                report
            )

            print(
                f"  ✓ {out_path}"
            )

        except Exception as exc:
            print(
                f"  ✗ Grad-CAM failed for "
                f"{row['path']}: {exc}"
            )

    pd.DataFrame(
        gradcam_results
    ).to_csv(
        OUTPUT_ROOT /
        "gradcam_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    final_summary = {
        "best_epoch": best_epoch,
        "best_val_qwk": best_qwk,
        "dataset": {
            "root": str(dataset_root),
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "model": model_summary,
        "final_validation": {
            key: value
            for key, value in final_val.items()
            if key not in {
                "predictions_df",
                "y_true",
                "y_pred",
                "probabilities",
            }
        },
        "final_test": {
            key: value
            for key, value in final_test.items()
            if key not in {
                "predictions_df",
                "y_true",
                "y_pred",
                "probabilities",
            }
        },
        "explainability_qc_threshold_percent": (
            qc_threshold
        ),
        "gradcam_reports_generated": len(
            gradcam_results
        ),
        "numerical_stability": {
            "val_loss_global_sample_mean": True,
            "pretrained_bn_running_stats_frozen": True,
            "msag_alpha_bound": 0.25,
            "last_val_max_abs_logit": (
                float(final_val.get("max_abs_logit", float("nan")))
                if isinstance(final_val, dict) else float("nan")
            ),
            "last_val_min_true_probability": (
                float(final_val.get("min_true_probability", float("nan")))
                if isinstance(final_val, dict) else float("nan")
            ),
        },
    }

    (OUTPUT_ROOT / "final_metrics.json").write_text(
        json.dumps(
            final_summary,
            indent=2,
            default=lambda x: float(x),
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 100
    )
    print(
        "RUN COMPLETE"
    )
    print(
        "=" * 100
    )
    print(
        f"Best checkpoint : {best_model_path}"
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
        f"Final metrics   : "
        f"{OUTPUT_ROOT / 'final_metrics.json'}"
    )
    print(
        f"Grad-CAM folder : {GRADCAM_ROOT}"
    )
    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
