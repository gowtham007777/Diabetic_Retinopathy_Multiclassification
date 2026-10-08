"""
train_densenet121_paper.py

PyTorch implementation of the STANDARD DenseNet121 baseline described in:
Al-Dolat et al. (2026), "Enhancing fundus image analysis for diabetic
retinopathy using CheXNet with CBAM and Grad-CAM visualization."

Important reproducibility note:
The paper identifies DenseNet121 as a transfer-learning baseline and states
the common input size, batch size, epochs, rescaling, augmentation, and
balanced-data split. It does NOT disclose the exact DenseNet121 checkpoint,
the numerical dropout rate, or a complete optimizer/LR specification for
the baseline. Therefore this script is "paper-faithful where specified",
with clearly labeled implementation defaults for the undisclosed items.

Dataset behavior:
- Exactly ONE dataset is trained per run: aptos OR ddr.
- The selected directory is:
    data\\organized\\aptos
  or
    data\\organized\\ddr
- The script scans the whole selected dataset (existing train/val/test
  folders) and creates a fresh stratified 80/10/10 split.
- Training protocol used here:
    original data -> 80/10/10 split -> oversample TRAIN only
- This avoids putting duplicated samples into validation/test.

Model:
- DenseNet121
- ImageNet-pretrained weights by torchvision, because the paper does not
  disclose the exact standard-DenseNet checkpoint.
- Final classifier: Dropout -> Linear(1024, 5)
- Dropout default is 0.5 as an explicit implementation choice; the paper
  only says dropout was introduced for baseline stability and gives no rate.
- No CheXNet checkpoint.
- No CBAM.
- No DropBlock.
- No Transformer.

Preprocessing:
- Resize: 224 x 224
- Rescale: /255
- Standardize with dataset RGB mean/std, calculated from the original
  selected dataset before oversampling.
- Train augmentation:
    rotation up to 40 degrees
    width shift up to 0.2
    height shift up to 0.2
    shear_range 0.2 radians (matching Keras-style convention)
    zoom_range 0.2
    horizontal flip
    nearest-neighbor-like border replication

The paper discusses brightness/contrast mathematically but its stated
implementation list does not provide numeric brightness/contrast ranges,
so no unreported photometric jitter is silently added.

Class weighting:
The paper gives w_c = N / (C * n_c). Because the paper balances before
splitting, the resulting training split is class balanced and this formula
produces weights of approximately 1.0 for all classes. This script follows
that literal order.

Metrics:
- accuracy
- weighted precision / recall / F1
- macro F1
- confusion matrix
- 95% accuracy CI using the standard binomial standard-error approximation

No scikit-learn is used.

Run from:
    (dr311) A:\\DR_classification>

APTOS:
    python train_densenet121_paper.py --dataset aptos --epochs 50 --batch-size 16 --workers 2

DDR:
    python train_densenet121_paper.py --dataset ddr --epochs 50 --batch-size 16 --workers 2

Smoke test:
    python train_densenet121_paper.py --dataset aptos --epochs 1 --batch-size 16 --workers 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tqdm import tqdm

import matplotlib.pyplot as plt


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 5

CLASS_NAMES = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative_DR",
]

SUPPORTED_DATASETS = [
    "aptos",
    "ddr",
]

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 2

# Explicit defaults where the paper does not publish exact values.
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_DROPOUT = 0.5

OUTPUT_ROOT_NAME = "runs_densenet121"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-faithful DenseNet121 DR baseline."
    )

    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="aptos",
        help="Train one dataset only: aptos or ddr."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epochs. Paper: 50."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size. Paper: 16."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="DataLoader workers."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate. Not specified in paper; default=1e-4."
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay. Not specified in paper; default=1e-5."
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Dropout probability. Paper does not report a value; default=0.5."
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use random DenseNet121 instead of ImageNet-pretrained weights."
    )

    parser.add_argument(
        "--no-oversampling",
        action="store_true",
        help="Disable paper-style balancing."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom output directory."
    )

    return parser.parse_args()


# ============================================================
# PATH HELPERS
# ============================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_output_root(
    project_root: Path,
    dataset_name: str,
    custom_output: Optional[str]
) -> Path:

    if custom_output:
        return Path(custom_output).resolve()

    return (
        project_root
        / OUTPUT_ROOT_NAME
        / dataset_name
    )


# ============================================================
# CLASS FOLDER PARSING
# ============================================================

def parse_class_folder(
    folder_name: str
) -> Optional[int]:

    name = folder_name.strip()

    if len(name) < 2:
        return None

    if (
        name[0].isdigit()
        and name[1] in {"_", "-", " "}
    ):
        value = int(name[0])

        if 0 <= value < NUM_CLASSES:
            return value

    return None


# ============================================================
# DATASET MANIFEST
# ============================================================

def scan_dataset(
    dataset_name: str,
    organized_root: Path
) -> pd.DataFrame:

    dataset_root = (
        organized_root
        / dataset_name
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{dataset_root}"
        )

    files = sorted(
        [
            p
            for p in dataset_root.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not files:
        raise RuntimeError(
            f"No supported image files found under:\n{dataset_root}"
        )

    rows = []
    seen = set()

    print(
        f"\nScanning {dataset_name.upper()} images..."
    )

    for image_path in tqdm(
        files,
        desc=dataset_name.upper()
    ):

        class_id = None
        current = image_path.parent

        while current != dataset_root:
            parsed = parse_class_folder(
                current.name
            )

            if parsed is not None:
                class_id = parsed
                break

            current = current.parent

        if class_id is None:
            continue

        relative_path = image_path.relative_to(
            dataset_root
        )

        sample_id = (
            f"{dataset_name}:"
            f"{relative_path.as_posix()}"
        )

        if sample_id in seen:
            raise RuntimeError(
                f"Duplicate sample ID: {sample_id}"
            )

        seen.add(sample_id)

        source_split = (
            relative_path.parts[0]
            if len(relative_path.parts) >= 2
            else "unknown"
        )

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "id_code": image_path.stem,
                "filename": image_path.name,
                "image_path": str(image_path.resolve()),
                "relative_path": relative_path.as_posix(),
                "source_split": source_split,
                "diagnosis": class_id,
                "class_name": CLASS_NAMES[class_id],
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No class-labeled images found under:\n{dataset_root}"
        )

    return df.reset_index(drop=True)


# ============================================================
# IMAGE VALIDATION
# ============================================================

def verify_images(
    df: pd.DataFrame
) -> None:

    print("\n")
    print("=" * 80)
    print("VERIFYING ORIGINAL IMAGES")
    print("=" * 80)

    failures = []

    for image_path in tqdm(
        df["image_path"],
        desc="Validation"
    ):
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            failures.append(
                (image_path, repr(exc))
            )

    if failures:
        preview = "\n".join(
            f"{p} -> {err}"
            for p, err in failures[:25]
        )

        raise RuntimeError(
            "Image verification failed:\n"
            f"{preview}\n"
            f"Total failures: {len(failures)}"
        )

    print(
        f"Verified {len(df)} images successfully."
    )


# ============================================================
# TRAIN-ONLY OVERSAMPLING
# ============================================================

def oversample_to_majority(
    df: pd.DataFrame,
    seed: int = SEED
) -> pd.DataFrame:

    counts = (
        df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    target = int(
        counts.max()
    )

    print("\n")
    print("=" * 80)
    print("RANDOM OVERSAMPLING BEFORE 80/10/10 SPLIT")
    print("=" * 80)

    print(
        f"Target count per class: {target}"
    )

    parts = []

    for class_id in range(
        NUM_CLASSES
    ):

        class_df = df[
            df["diagnosis"] == class_id
        ].copy()

        count = len(class_df)

        if count == 0:
            raise RuntimeError(
                f"Class {class_id} ({CLASS_NAMES[class_id]}) is empty."
            )

        if count < target:
            sampled = class_df.sample(
                n=target,
                replace=True,
                random_state=seed + class_id
            )
        else:
            sampled = class_df.copy()

        parts.append(
            sampled
        )

        print(
            f"{class_id} "
            f"{CLASS_NAMES[class_id]:18s}: "
            f"{count:6d} -> {len(sampled):6d}"
        )

    balanced_df = pd.concat(
        parts,
        ignore_index=True
    )

    balanced_df = balanced_df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    return balanced_df


# ============================================================
# STRATIFIED 80/10/10 SPLIT
# ============================================================

def stratified_split_balanced(
    balanced_df: pd.DataFrame,
    seed: int = SEED
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:

    train_parts = []
    val_parts = []
    test_parts = []

    for class_id in range(
        NUM_CLASSES
    ):

        class_df = balanced_df[
            balanced_df["diagnosis"] == class_id
        ].copy()

        class_df = class_df.sample(
            frac=1.0,
            random_state=seed + class_id
        ).reset_index(drop=True)

        n = len(class_df)

        n_train = int(
            round(
                0.80 * n
            )
        )

        n_val = int(
            round(
                0.10 * n
            )
        )

        n_test = (
            n
            - n_train
            - n_val
        )

        if (
            n_train <= 0
            or n_val <= 0
            or n_test <= 0
        ):
            raise RuntimeError(
                f"Unable to split class {class_id} "
                f"with {n} samples."
            )

        train_parts.append(
            class_df.iloc[
                :n_train
            ]
        )

        val_parts.append(
            class_df.iloc[
                n_train:
                n_train + n_val
            ]
        )

        test_parts.append(
            class_df.iloc[
                n_train + n_val:
            ]
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    val_df = pd.concat(
        val_parts,
        ignore_index=True
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True
    )

    train_df = train_df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1.0,
        random_state=seed + 1
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1.0,
        random_state=seed + 2
    ).reset_index(drop=True)

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# DATASET CLASS
# ============================================================

class DRDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None
    ):
        self.df = dataframe.reset_index(
            drop=True
        )
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self,
        index: int
    ):

        row = self.df.iloc[
            index
        ]

        image = Image.open(
            row["image_path"]
        ).convert(
            "RGB"
        )

        if self.transform:
            image = self.transform(
                image
            )

        label = int(
            row["diagnosis"]
        )

        return (
            image,
            label
        )


# ============================================================
# [0,1] IMAGE TENSOR
# ============================================================

class ToFloatTensor:

    def __call__(
        self,
        image: Image.Image
    ) -> torch.Tensor:

        array = np.asarray(
            image,
            dtype=np.float32
        )

        array /= 255.0

        return torch.from_numpy(
            array
        ).permute(
            2,
            0,
            1
        )


# ============================================================
# PAPER GEOMETRIC AUGMENTATION
# ============================================================

class PaperAffine:

    """
    Keras ImageDataGenerator-style geometry.

    rotation_range = 40
    width_shift_range = 0.2
    height_shift_range = 0.2
    shear_range = 0.2 radians
    zoom_range = 0.2
    fill_mode = nearest
    """

    def __init__(
        self,
        rotation_range: float = 40.0,
        width_shift_range: float = 0.2,
        height_shift_range: float = 0.2,
        shear_range_radians: float = 0.2,
        zoom_range: float = 0.2,
    ):

        self.rotation_range = rotation_range

        self.width_shift_range = (
            width_shift_range
        )

        self.height_shift_range = (
            height_shift_range
        )

        self.shear_range_degrees = (
            math.degrees(
                shear_range_radians
            )
        )

        self.zoom_range = zoom_range

    @staticmethod
    def edge_pad(
        image: Image.Image,
        pad: int
    ) -> Image.Image:

        array = np.asarray(
            image
        )

        array = np.pad(
            array,
            (
                (pad, pad),
                (pad, pad),
                (0, 0),
            ),
            mode="edge"
        )

        return Image.fromarray(
            array
        )

    def __call__(
        self,
        image: Image.Image
    ) -> Image.Image:

        width, height = image.size

        angle = random.uniform(
            -self.rotation_range,
            self.rotation_range
        )

        dx = random.uniform(
            -self.width_shift_range * width,
            self.width_shift_range * width
        )

        dy = random.uniform(
            -self.height_shift_range * height,
            self.height_shift_range * height
        )

        shear_x = random.uniform(
            -self.shear_range_degrees,
            self.shear_range_degrees
        )

        shear_y = random.uniform(
            -self.shear_range_degrees,
            self.shear_range_degrees
        )

        scale = random.uniform(
            1.0 - self.zoom_range,
            1.0 + self.zoom_range
        )

        # Large edge padding approximates nearest-neighbor fill.
        pad = int(
            math.ceil(
                1.5
                * max(width, height)
            )
        )

        padded = self.edge_pad(
            image,
            pad
        )

        transformed = TF.affine(
            padded,
            angle=angle,
            translate=[
                int(round(dx)),
                int(round(dy)),
            ],
            scale=scale,
            shear=[
                shear_x,
                shear_y,
            ],
            interpolation=(
                T.InterpolationMode.BILINEAR
            ),
            fill=0,
        )

        return transformed.crop(
            (
                pad,
                pad,
                pad + width,
                pad + height,
            )
        )


# ============================================================
# NORMALIZATION STATISTICS
# ============================================================

def calculate_mean_std(
    df: pd.DataFrame
) -> Tuple[
    List[float],
    List[float]
]:

    """
    Calculate mean/std from the original selected dataset before
    oversampling.

    Paper only gives zero-mean/unit-variance standardization and does
    not disclose numerical RGB statistics.
    """

    dataset = DRDataset(
        df,
        transform=T.Compose([
            T.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),
            ToFloatTensor(),
        ])
    )

    channel_sum = torch.zeros(
        3,
        dtype=torch.float64
    )

    channel_squared_sum = torch.zeros(
        3,
        dtype=torch.float64
    )

    total_pixels = 0

    print("\n")
    print("=" * 80)
    print("CALCULATING DATASET MEAN / STD")
    print("=" * 80)

    with torch.inference_mode():

        for image, _ in tqdm(
            dataset,
            desc="Mean/std"
        ):

            image = image.to(
                torch.float64
            )

            pixels = image.reshape(
                3,
                -1
            )

            channel_sum += pixels.sum(
                dim=1
            )

            channel_squared_sum += (
                pixels ** 2
            ).sum(
                dim=1
            )

            total_pixels += (
                pixels.shape[1]
            )

    mean = (
        channel_sum
        / total_pixels
    )

    variance = (
        channel_squared_sum
        / total_pixels
        - mean ** 2
    )

    variance = torch.clamp(
        variance,
        min=0.0
    )

    std = torch.sqrt(
        variance
    )

    mean_list = [
        float(v)
        for v in mean.tolist()
    ]

    std_list = [
        float(v)
        for v in std.tolist()
    ]

    print(
        "Mean:",
        mean_list
    )

    print(
        "Std :",
        std_list
    )

    return (
        mean_list,
        std_list
    )


# ============================================================
# TRANSFORMS
# ============================================================

def build_transforms(
    mean: List[float],
    std: List[float]
):

    train_transform = T.Compose([

        T.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        PaperAffine(
            rotation_range=40.0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range_radians=0.2,
            zoom_range=0.2,
        ),

        T.RandomHorizontalFlip(
            p=0.5
        ),

        ToFloatTensor(),

        T.Normalize(
            mean=mean,
            std=std
        ),
    ])

    eval_transform = T.Compose([

        T.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        ToFloatTensor(),

        T.Normalize(
            mean=mean,
            std=std
        ),
    ])

    return (
        train_transform,
        eval_transform
    )


# ============================================================
# MODEL
# ============================================================

def build_model(
    use_pretrained: bool,
    dropout: float
) -> nn.Module:

    if use_pretrained:

        print(
            "\nLoading ImageNet-pretrained DenseNet121..."
        )

        weights = (
            torchvision.models
            .DenseNet121_Weights
            .DEFAULT
        )

        model = (
            torchvision.models
            .densenet121(
                weights=weights
            )
        )

    else:

        print(
            "\nBuilding DenseNet121 "
            "without pretrained weights..."
        )

        model = (
            torchvision.models
            .densenet121(
                weights=None
            )
        )

    in_features = (
        model.classifier.in_features
    )

    model.classifier = nn.Sequential(
        nn.Dropout(
            p=dropout
        ),
        nn.Linear(
            in_features,
            NUM_CLASSES
        ),
    )

    print(
        f"Classifier: Dropout({dropout}) -> "
        f"Linear({in_features}, {NUM_CLASSES})"
    )

    return model


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_paper_class_weights(
    train_df: pd.DataFrame
) -> torch.Tensor:

    """
    Literal paper formula:
        w_c = N / (C * n_c)

    This is computed on the final training dataframe. When train-only
    oversampling is enabled, the training classes are balanced and the
    resulting weights are approximately 1.0.
    """

    counts = (
        train_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    values = np.array(
        [
            counts.get(
                class_id,
                0
            )
            for class_id in range(
                NUM_CLASSES
            )
        ],
        dtype=np.float64
    )

    if np.any(values <= 0):
        raise RuntimeError(
            "Training split does not contain all classes."
        )

    N = values.sum()

    C = NUM_CLASSES

    weights = (
        N
        / (
            C
            * values
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> Tuple[
    float,
    float
]:

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Train",
        leave=False
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(
            0
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += int(
            (
                predictions
                == labels
            ).sum().item()
        )

        total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        running_loss / total,
        correct / total
    )


# ============================================================
# METRICS — NO SCIKIT-LEARN
# ============================================================

def confusion_matrix_numpy(
    labels: List[int],
    predictions: List[int]
) -> np.ndarray:

    cm = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    for true_label, pred_label in zip(
        labels,
        predictions
    ):

        cm[
            int(true_label),
            int(pred_label)
        ] += 1

    return cm


def safe_divide(
    numerator: float,
    denominator: float
) -> float:

    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def calculate_metrics(
    labels: List[int],
    predictions: List[int]
) -> Dict:

    cm = confusion_matrix_numpy(
        labels,
        predictions
    )

    total = int(
        cm.sum()
    )

    accuracy = safe_divide(
        np.trace(cm),
        total
    )

    per_class = []

    supports = cm.sum(
        axis=1
    )

    for class_id in range(
        NUM_CLASSES
    ):

        tp = int(
            cm[
                class_id,
                class_id
            ]
        )

        fp = int(
            cm[
                :,
                class_id
            ].sum()
            - tp
        )

        fn = int(
            cm[
                class_id,
                :
            ].sum()
            - tp
        )

        support = int(
            supports[
                class_id
            ]
        )

        precision = safe_divide(
            tp,
            tp + fp
        )

        recall = safe_divide(
            tp,
            tp + fn
        )

        f1 = safe_divide(
            2.0
            * precision
            * recall,
            precision
            + recall
        )

        per_class.append(
            {
                "class_id":
                    class_id,
                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],
                "precision":
                    precision,
                "recall":
                    recall,
                "f1":
                    f1,
                "support":
                    support,
            }
        )

    weighted_precision = safe_divide(
        sum(
            x["precision"]
            * x["support"]
            for x in per_class
        ),
        total
    )

    weighted_recall = safe_divide(
        sum(
            x["recall"]
            * x["support"]
            for x in per_class
        ),
        total
    )

    weighted_f1 = safe_divide(
        sum(
            x["f1"]
            * x["support"]
            for x in per_class
        ),
        total
    )

    macro_precision = float(
        np.mean(
            [
                x["precision"]
                for x in per_class
            ]
        )
    )

    macro_recall = float(
        np.mean(
            [
                x["recall"]
                for x in per_class
            ]
        )
    )

    macro_f1 = float(
        np.mean(
            [
                x["f1"]
                for x in per_class
            ]
        )
    )

    # Paper reports standard-error-based 95% accuracy intervals.
    if total > 0:

        standard_error = math.sqrt(
            max(
                accuracy
                * (
                    1.0
                    - accuracy
                ),
                0.0
            )
            / total
        )

        ci_low = max(
            0.0,
            accuracy
            - 1.96
            * standard_error
        )

        ci_high = min(
            1.0,
            accuracy
            + 1.96
            * standard_error
        )

    else:

        ci_low = 0.0
        ci_high = 0.0

    return {
        "accuracy":
            float(accuracy),
        "accuracy_ci_95":
            [
                float(ci_low),
                float(ci_high)
            ],
        "weighted_precision":
            weighted_precision,
        "weighted_recall":
            weighted_recall,
        "weighted_f1":
            weighted_f1,
        "macro_precision":
            macro_precision,
        "macro_recall":
            macro_recall,
        "macro_f1":
            macro_f1,
        "per_class":
            per_class,
        "confusion_matrix":
            cm,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Dict:

    model.eval()

    running_loss = 0.0
    total = 0

    labels = []
    predictions = []

    with torch.inference_mode():

        for images, batch_labels in tqdm(
            loader,
            desc="Eval",
            leave=False
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            batch_labels = batch_labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                batch_labels
            )

            batch_size = batch_labels.size(
                0
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            total += batch_size

            batch_predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            labels.extend(
                batch_labels.cpu().tolist()
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

    metrics = calculate_metrics(
        labels,
        predictions
    )

    metrics["loss"] = (
        running_loss
        / total
    )

    metrics["labels"] = labels
    metrics["predictions"] = predictions

    return metrics


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    mean: List[float],
    std: List[float],
    config: Dict
) -> None:

    torch.save(
        {
            "epoch":
                epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "metrics":
                {
                    "accuracy":
                        metrics[
                            "accuracy"
                        ],
                    "accuracy_ci_95":
                        metrics[
                            "accuracy_ci_95"
                        ],
                    "weighted_precision":
                        metrics[
                            "weighted_precision"
                        ],
                    "weighted_recall":
                        metrics[
                            "weighted_recall"
                        ],
                    "weighted_f1":
                        metrics[
                            "weighted_f1"
                        ],
                    "macro_f1":
                        metrics[
                            "macro_f1"
                        ],
                },
            "mean":
                mean,
            "std":
                std,
            "class_names":
                CLASS_NAMES,
            "image_size":
                IMAGE_SIZE,
            "config":
                config,
        },
        path
    )


# ============================================================
# PLOTS
# ============================================================

def save_curves(
    history: Dict,
    result_root: Path
) -> None:

    epochs = np.arange(
        1,
        len(
            history[
                "train_loss"
            ]
        )
        + 1
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        epochs,
        history[
            "train_accuracy"
        ],
        label="Train Accuracy"
    )

    ax.plot(
        epochs,
        history[
            "val_accuracy"
        ],
        label="Validation Accuracy"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "DenseNet121 Accuracy"
    )

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        result_root
        / "accuracy_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        epochs,
        history[
            "train_loss"
        ],
        label="Train Loss"
    )

    ax.plot(
        epochs,
        history[
            "val_loss"
        ],
        label="Validation Loss"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "DenseNet121 Loss"
    )

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        result_root
        / "loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_confusion_matrix_plot(
    cm: np.ndarray,
    result_root: Path
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    fig.colorbar(
        image,
        ax=ax
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
        ha="right"
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        "DenseNet121 Confusion Matrix"
    )

    threshold = (
        cm.max()
        / 2.0
        if cm.size
        and cm.max() > 0
        else 0.0
    )

    for row in range(
        NUM_CLASSES
    ):

        for col in range(
            NUM_CLASSES
        ):

            ax.text(
                col,
                row,
                str(
                    cm[
                        row,
                        col
                    ]
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[
                        row,
                        col
                    ] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    fig.savefig(
        result_root
        / "test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    np.savetxt(
        result_root
        / "test_confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=","
    )


# ============================================================
# REPORT
# ============================================================

def save_test_report(
    metrics: Dict,
    result_root: Path
) -> None:

    ci = metrics[
        "accuracy_ci_95"
    ]

    with (
        result_root
        / "test_report.txt"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DENSENET121 DIABETIC RETINOPATHY TEST REPORT\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Loss            : "
            f"{metrics['loss']:.6f}\n"
        )

        f.write(
            f"Accuracy        : "
            f"{metrics['accuracy']:.6f}\n"
        )

        f.write(
            f"Accuracy 95% CI : "
            f"({ci[0]:.6f}, {ci[1]:.6f})\n"
        )

        f.write(
            f"Weighted Prec.  : "
            f"{metrics['weighted_precision']:.6f}\n"
        )

        f.write(
            f"Weighted Recall : "
            f"{metrics['weighted_recall']:.6f}\n"
        )

        f.write(
            f"Weighted F1     : "
            f"{metrics['weighted_f1']:.6f}\n"
        )

        f.write(
            f"Macro Precision : "
            f"{metrics['macro_precision']:.6f}\n"
        )

        f.write(
            f"Macro Recall    : "
            f"{metrics['macro_recall']:.6f}\n"
        )

        f.write(
            f"Macro F1        : "
            f"{metrics['macro_f1']:.6f}\n\n"
        )

        f.write(
            "PER-CLASS METRICS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for item in (
            metrics[
                "per_class"
            ]
        ):

            f.write(
                f"{item['class_name']:20s}"
                f"Precision={item['precision']:.6f}  "
                f"Recall={item['recall']:.6f}  "
                f"F1={item['f1']:.6f}  "
                f"Support={item['support']}\n"
            )

        f.write(
            "\nCONFUSION MATRIX\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            np.array2string(
                metrics[
                    "confusion_matrix"
                ]
            )
        )


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    split_root: Path,
    full_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:

    split_root.mkdir(
        parents=True,
        exist_ok=True
    )

    full_df.to_csv(
        split_root
        / "full_manifest.csv",
        index=False
    )

    balanced_df.to_csv(
        split_root
        / "balanced_train_manifest.csv",
        index=False
    )

    train_df.to_csv(
        split_root
        / "train.csv",
        index=False
    )

    val_df.to_csv(
        split_root
        / "validation.csv",
        index=False
    )

    test_df.to_csv(
        split_root
        / "test.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    seed_everything(
        SEED
    )

    args = parse_args()

    project_root = (
        get_project_root()
    )

    organized_root = (
        project_root
        / "data"
        / "organized"
    )

    dataset_root = (
        organized_root
        / args.dataset
    )

    output_root = (
        get_output_root(
            project_root,
            args.dataset,
            args.output_dir
        )
    )

    checkpoint_root = (
        output_root
        / "checkpoints"
    )

    result_root = (
        output_root
        / "results"
    )

    split_root = (
        output_root
        / "splits"
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True
    )

    result_root.mkdir(
        parents=True,
        exist_ok=True
    )

    split_root.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n")
    print("=" * 90)
    print(
        "DENSENET121 — DIABETIC RETINOPATHY BASELINE"
    )
    print("=" * 90)

    print(
        "Project root:",
        project_root
    )

    print(
        "Dataset:",
        args.dataset.upper()
    )

    print(
        "Dataset root:",
        dataset_root
    )

    print(
        "Output:",
        output_root
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "TorchVision:",
        torchvision.__version__
    )

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{dataset_root}"
        )

    if not (
        0.0
        <= args.dropout
        < 1.0
    ):
        raise ValueError(
            "Dropout must be in [0, 1)."
        )

    # --------------------------------------------------------
    # Scan original selected dataset
    # --------------------------------------------------------

    full_df = scan_dataset(
        args.dataset,
        organized_root
    )

    print_distribution(
        "ORIGINAL DATASET",
        full_df
    )

    print(
        "\nThis run uses ONLY the selected dataset."
    )

    print(
        "The existing train/val/test directory names are retained "
        "only as source metadata; a fresh stratified 80/10/10 split "
        "is generated first, then oversampling is applied to TRAIN only."
    )

    # --------------------------------------------------------
    # Verify all images
    # --------------------------------------------------------

    verify_images(
        full_df
    )

    # --------------------------------------------------------
    # 80/10/10 SPLIT FIRST
    # --------------------------------------------------------

    (
        original_train_df,
        val_df,
        test_df
    ) = stratified_split_balanced(
        full_df,
        seed=SEED
    )

    print_distribution(
        "ORIGINAL TRAIN BEFORE OVERSAMPLING",
        original_train_df
    )

    print_distribution(
        "VALIDATION (NO OVERSAMPLING)",
        val_df
    )

    print_distribution(
        "TEST (NO OVERSAMPLING)",
        test_df
    )

    # --------------------------------------------------------
    # OVERSAMPLING — TRAINING SPLIT ONLY
    # --------------------------------------------------------

    if args.no_oversampling:

        train_df = original_train_df.copy()

        print(
            "\nOversampling disabled. Training data remains imbalanced."
        )

    else:

        train_df = (
            oversample_to_majority(
                original_train_df,
                seed=SEED
            )
        )

    print_distribution(
        "TRAIN AFTER OVERSAMPLING",
        train_df
    )

    # The balanced training dataframe is kept under the existing
    # variable name so the rest of the script and saved manifests
    # remain compatible.
    balanced_df = train_df

    # --------------------------------------------------------
    # Check partition overlap. Oversampling duplicates remain TRAIN-only.
    # --------------------------------------------------------

    train_ids = set(
        train_df["sample_id"]
    )

    val_ids = set(
        val_df["sample_id"]
    )

    test_ids = set(
        test_df["sample_id"]
    )

    overlap_info = {
        "train_val":
            len(
                train_ids & val_ids
            ),
        "train_test":
            len(
                train_ids & test_ids
            ),
        "val_test":
            len(
                val_ids & test_ids
            ),
    }

    train_duplicate_rows = (
        len(train_df)
        - train_df["sample_id"].nunique()
    )

    print(
        "\nTRAIN duplicate rows created by oversampling:",
        train_duplicate_rows
    )

    if any(
        value > 0
        for value
        in overlap_info.values()
    ):

        print("\n")
        print(
            "WARNING: sample IDs overlap across partitions."
        )

        print(
            "Train/Val overlap:",
            overlap_info["train_val"]
        )

        print(
            "Train/Test overlap:",
            overlap_info["train_test"]
        )

        print(
            "Val/Test overlap:",
            overlap_info["val_test"]
        )

    # --------------------------------------------------------
    # Save split files
    # --------------------------------------------------------

    save_splits(
        split_root,
        full_df,
        balanced_df,
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Calculate mean/std from original dataset
    # --------------------------------------------------------

    mean, std = calculate_mean_std(
        full_df
    )

    with (
        result_root
        / "normalization.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "mean":
                    mean,
                "std":
                    std,
                "source":
                    "Original selected dataset",
                "image_size":
                    IMAGE_SIZE,
            },
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    (
        train_transform,
        eval_transform
    ) = build_transforms(
        mean,
        std
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = DRDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = DRDataset(
        val_df,
        transform=eval_transform
    )

    test_dataset = DRDataset(
        test_df,
        transform=eval_transform
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    loader_args = {
        "num_workers":
            args.workers,
        "pin_memory":
            torch.cuda.is_available(),
    }

    if args.workers > 0:
        loader_args[
            "persistent_workers"
        ] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_args
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_args
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_args
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BUILDING DENSENET121")
    print("=" * 80)

    initialization = (
        "ImageNet-pretrained DenseNet121"
        if not args.no_pretrained
        else "Random DenseNet121"
    )

    model = build_model(
        use_pretrained=(
            not args.no_pretrained
        ),
        dropout=args.dropout
    )

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    class_weights = (
        compute_paper_class_weights(
            train_df
        )
        .to(DEVICE)
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print("\n")
    print("=" * 80)
    print("PAPER CLASS WEIGHTS")
    print("=" * 80)

    for class_id, weight in enumerate(
        class_weights
        .detach()
        .cpu()
        .tolist()
    ):

        print(
            f"{class_id} "
            f"{CLASS_NAMES[class_id]:18s}: "
            f"{weight:.6f}"
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = {
        "model":
            "DenseNet121 baseline",
        "dataset":
            args.dataset,
        "initialization":
            initialization,
        "image_size":
            IMAGE_SIZE,
        "num_classes":
            NUM_CLASSES,
        "class_names":
            CLASS_NAMES,
        "epochs":
            args.epochs,
        "batch_size":
            args.batch_size,
        "learning_rate":
            args.lr,
        "weight_decay":
            args.weight_decay,
        "dropout":
            args.dropout,
        "oversampling":
            (
                "Random oversampling on TRAIN only after 80/10/10 split"
                if not args.no_oversampling
                else "Disabled"
            ),
        "split":
            "80/10/10 before training-only oversampling",
        "augmentation":
            {
                "rotation":
                    "up to 40 degrees",
                "width_shift":
                    0.2,
                "height_shift":
                    0.2,
                "shear_range_radians":
                    0.2,
                "zoom_range":
                    0.2,
                "horizontal_flip_probability":
                    0.5,
                "fill_mode":
                    "nearest approximation via edge replication",
            },
        "preprocessing":
            {
                "resize":
                    "224x224",
                "rescale":
                    "1/255",
                "standardization":
                    "zero mean / unit variance",
                "mean":
                    mean,
                "std":
                    std,
            },
        "sizes":
            {
                "original":
                    len(full_df),
                "balanced":
                    len(balanced_df),
                "train":
                    len(train_df),
                "validation":
                    len(val_df),
                "test":
                    len(test_df),
            },
        "split_overlap_due_to_pre_split_oversampling":
            overlap_info,
        "seed":
            SEED,
        "no_sklearn":
            True,
    }

    with (
        result_root
        / "experiment_config.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_weighted_f1": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    best_val_accuracy = -1.0
    best_epoch = -1

    best_checkpoint = (
        checkpoint_root
        / "densenet121_best.pth"
    )

    last_checkpoint = (
        checkpoint_root
        / "densenet121_last.pth"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("START TRAINING")
    print("=" * 90)

    total_start = time.time()

    for epoch in range(
        1,
        args.epochs + 1
    ):

        epoch_start = time.time()

        print(
            f"\n"
            f"{'=' * 25} "
            f"EPOCH {epoch}/{args.epochs} "
            f"{'=' * 25}"
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "val_loss"
        ].append(
            val_metrics[
                "loss"
            ]
        )

        history[
            "val_accuracy"
        ].append(
            val_metrics[
                "accuracy"
            ]
        )

        history[
            "val_weighted_f1"
        ].append(
            val_metrics[
                "weighted_f1"
            ]
        )

        history[
            "val_macro_f1"
        ].append(
            val_metrics[
                "macro_f1"
            ]
        )

        history[
            "epoch_time_sec"
        ].append(
            epoch_time
        )

        print(
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss       : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Val Accuracy   : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Weighted F1: "
            f"{val_metrics['weighted_f1']:.4f}"
        )

        print(
            f"Val Macro F1   : "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Epoch Time     : "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            epoch,
            val_metrics,
            mean,
            std,
            config
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics[
                "accuracy"
            ]
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_metrics[
                    "accuracy"
                ]
            )

            best_epoch = epoch

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                val_metrics,
                mean,
                std,
                config
            )

            print(
                "*** BEST MODEL SAVED ***"
            )

    total_time = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # Save history and plots
    # --------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        result_root
        / "training_history.csv",
        index=False
    )

    save_curves(
        history,
        result_root
    )

    # --------------------------------------------------------
    # Reload best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        best_checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion
    )

    ci = test_metrics[
        "accuracy_ci_95"
    ]

    print(
        f"\nBest Epoch         : "
        f"{best_epoch}"
    )

    print(
        f"Best Val Accuracy  : "
        f"{best_val_accuracy:.6f}"
    )

    print(
        f"Test Loss          : "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Test Accuracy      : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Accuracy 95% CI    : "
        f"({ci[0]:.6f}, {ci[1]:.6f})"
    )

    print(
        f"Weighted Precision : "
        f"{test_metrics['weighted_precision']:.6f}"
    )

    print(
        f"Weighted Recall    : "
        f"{test_metrics['weighted_recall']:.6f}"
    )

    print(
        f"Weighted F1        : "
        f"{test_metrics['weighted_f1']:.6f}"
    )

    print(
        f"Macro F1           : "
        f"{test_metrics['macro_f1']:.6f}"
    )

    print("\nPer-class metrics:")

    print(
        f"{'Class':20s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'Support':>12s}"
    )

    for item in (
        test_metrics[
            "per_class"
        ]
    ):

        print(
            f"{item['class_name']:20s}"
            f"{item['precision']:>12.4f}"
            f"{item['recall']:>12.4f}"
            f"{item['f1']:>12.4f}"
            f"{item['support']:>12d}"
        )

    # --------------------------------------------------------
    # Save final artifacts
    # --------------------------------------------------------

    save_confusion_matrix_plot(
        test_metrics[
            "confusion_matrix"
        ],
        result_root
    )

    save_test_report(
        test_metrics,
        result_root
    )

    final_summary = {
        **config,
        "best_epoch":
            best_epoch,
        "best_validation_accuracy":
            best_val_accuracy,
        "test_loss":
            test_metrics[
                "loss"
            ],
        "test_accuracy":
            test_metrics[
                "accuracy"
            ],
        "test_accuracy_95_ci":
            test_metrics[
                "accuracy_ci_95"
            ],
        "test_weighted_precision":
            test_metrics[
                "weighted_precision"
            ],
        "test_weighted_recall":
            test_metrics[
                "weighted_recall"
            ],
        "test_weighted_f1":
            test_metrics[
                "weighted_f1"
            ],
        "test_macro_f1":
            test_metrics[
                "macro_f1"
            ],
        "total_training_time_sec":
            total_time,
        "best_checkpoint":
            str(
                best_checkpoint
            ),
        "last_checkpoint":
            str(
                last_checkpoint
            ),
    }

    with (
        result_root
        / "final_summary.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("DENSENET121 TRAINING COMPLETE")
    print("=" * 90)

    print(
        "Dataset:",
        args.dataset.upper()
    )

    print(
        "Best checkpoint:",
        best_checkpoint
    )

    print(
        "Results:",
        result_root
    )

    print(
        "Split manifests:",
        split_root
    )


def print_distribution(
    title: str,
    df: pd.DataFrame
) -> None:

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        "Total:",
        len(df)
    )

    counts = (
        df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            counts.get(
                class_id,
                0
            )
        )

        print(
            f"{class_id} "
            f"{CLASS_NAMES[class_id]:18s}: "
            f"{count}"
        )


if __name__ == "__main__":
    main()
