"""
train_chexnet_directory.py
==========================

Full PyTorch CheXNet baseline for diabetic-retinopathy grading using the
directory structure supplied by the user.

PROJECT STRUCTURE EXPECTED
--------------------------

A:\\DR_classification\\
│
├── train_chexnet_directory.py
│
├── data\\
│   ├── combined\\
│   │   ├── aptos_test.csv
│   │   ├── ddr_test.csv
│   │   ├── class_balance_report.csv
│   │   ├── train_combined_balanced.csv
│   │   ├── train_combined_raw.csv
│   │   └── val_combined.csv
│   │
│   ├── metadata\\
│   │   ├── aptos_all.csv
│   │   ├── ddr_all.csv
│   │   ├── dropped_aptos.csv
│   │   ├── dropped_ddr.csv
│   │   ├── run_config.json
│   │   └── split_summary.csv
│   │
│   └── organized\\
│       ├── aptos\\
│       │   ├── train\\
│       │   │   ├── 0_No_DR\\
│       │   │   ├── 1_Mild\\
│       │   │   ├── 2_Moderate\\
│       │   │   ├── 3_Severe\\
│       │   │   └── 4_Proliferative_DR\\
│       │   └── test\\
│       │       ├── 0_No_DR\\
│       │       ├── 1_Mild\\
│       │       ├── 2_Moderate\\
│       │       ├── 3_Severe\\
│       │       └── 4_Proliferative_DR\\
│       │
│       └── ddr\\
│           ├── train\\
│           │   ├── 0_No_DR\\
│           │   ├── 1_Mild\\
│           │   ├── 2_Moderate\\
│           │   ├── 3_Severe\\
│           │   └── 4_Proliferative_DR\\
│           └── test\\
│               ├── 0_No_DR\\
│               ├── 1_Mild\\
│               ├── 2_Moderate\\
│               ├── 3_Severe\\
│               └── 4_Proliferative_DR\\


IMPORTANT DATA PROTOCOL
-----------------------

Each execution trains exactly ONE dataset. The selected dataset is
    passed with --dataset aptos or --dataset ddr.

This script deliberately does NOT use:
    data\\combined\\train_combined_balanced.csv
    data\\combined\\val_combined.csv
    data\\combined\\aptos_test.csv
    data\\combined\\ddr_test.csv

for the new split.

Instead, it reconstructs the original image manifest by scanning:

    data\\organized\\aptos
    data\\organized\\ddr

including both the existing train and test folders of the SELECTED dataset.

Then it performs:

    ORIGINAL IMAGES
          |
          v
    STRATIFIED 80/10/10 SPLIT
          |
          +------ TRAIN --------> RANDOM OVERSAMPLING ONLY
          |
          +------ VALIDATION ----> UNTOUCHED
          |
          +------ TEST ----------> UNTOUCHED

This follows the user's requested research-clean protocol and avoids
duplicated oversampled images leaking into validation/test.

The split is stratified by:
    dataset + DR class

so the APTOS/DDR composition and five DR grades are represented in all
splits.

PAPER-BASED IMAGE PROCESSING
----------------------------

The supplied paper states:
    - resize to 224 x 224
    - rescale pixel values to [0, 1]
    - standardize to zero mean and unit variance
    - rotations up to 40 degrees
    - width shift up to 0.2
    - height shift up to 0.2
    - shear up to 0.2
    - zoom up to 0.2
    - horizontal flip
    - nearest-neighbor filling
    - brightness/contrast variation conceptually

PyTorch implementation:
    - rotation: ±40 degrees
    - translation: ±20% width/height
    - shear: ±atan(0.2) ≈ ±11.31 degrees
    - zoom: scale [0.8, 1.2]
    - border replication is implemented before affine transform to
      approximate Keras fill_mode="nearest"
    - horizontal flip p=0.5
    - ColorJitter brightness=0.2, contrast=0.2 as an explicit
      implementation approximation because the paper does not provide
      numerical brightness/contrast ranges.

NORMALIZATION
-------------

The paper does not publish numerical RGB mean/std values.

This script calculates them from the ORIGINAL TRAIN SPLIT ONLY:
    - before oversampling
    - before augmentation

CLASS WEIGHTS
-------------

The paper gives:
    w_c = N / (C * n_c)

This implementation calculates class weights from the ORIGINAL training
distribution, not the oversampled dataframe.

MODEL
-----

CheXNet is represented by its DenseNet121 backbone.

If a local original CheXNet checkpoint is supplied with:
    --chexnet-checkpoint PATH

the compatible DenseNet121 feature tensors are loaded and the original
14-disease chest-X-ray classifier is discarded.

If no CheXNet checkpoint is supplied, the script uses torchvision's
ImageNet-pretrained DenseNet121 weights. This is transfer learning, but
is NOT guaranteed to be the exact CheXNet checkpoint used by the paper.

The final classifier is:
    Linear(1024, 5)

Loss:
    CrossEntropyLoss with paper-style class weights.

No CBAM, DropBlock or Transformer is included because this script is the
BASELINE CheXNet model.

NO SCIKIT-LEARN
---------------

This script intentionally has NO scikit-learn dependency because the
user's Windows environment previously blocked one of sklearn's compiled
DLL extensions through an Application Control policy.

All splitting and metrics are implemented using NumPy/PyTorch.

RUN
---

From A:\\DR_classification:

    python train_chexnet_directory.py --epochs 50 --batch-size 16 --workers 2

APTOS only:

    python train_chexnet_directory.py --dataset aptos --epochs 50 --batch-size 16 --workers 2

DDR only:

    python train_chexnet_directory.py --dataset ddr --epochs 50 --batch-size 16 --workers 2

With original CheXNet checkpoint:

    python train_chexnet_directory.py ^
        --chexnet-checkpoint checkpoints\\chexnet\\model.pth.tar ^
        --epochs 50 --batch-size 16 --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tqdm import tqdm

import matplotlib.pyplot as plt


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# GLOBAL CONFIGURATION
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

DATASET_NAMES = [
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

DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5

OUTPUT_DIR_NAME = "runs_chexnet"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = SEED) -> None:
    """Set Python/NumPy/PyTorch random seeds."""

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
        description=(
            "CheXNet/DenseNet121 DR classification using the "
            "user-provided data/organized directory structure."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="aptos",
        help=(
            "Train on ONE dataset only. "
            "Use --dataset aptos or --dataset ddr. "
            "Default: aptos"
        )
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
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--chexnet-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional path to an original CheXNet DenseNet121 "
            "checkpoint (.pth or .pth.tar)."
        ),
    )

    parser.add_argument(
        "--no-oversampling",
        action="store_true",
        help=(
            "Disable random oversampling of the training split."
        ),
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Do not use torchvision ImageNet weights when a "
            "CheXNet checkpoint is not provided."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional output directory. By default the script creates "
            "runs_chexnet under the project root."
        ),
    )

    return parser.parse_args()


# ============================================================
# PROJECT PATHS
# ============================================================

def get_project_root() -> Path:
    """
    Assumes this script is stored directly in the project root:
        A:\\DR_classification\\train_chexnet_directory.py
    """

    return Path(__file__).resolve().parent


def get_paths(
    project_root: Path
) -> Dict[str, Path]:

    data_root = project_root / "data"

    organized_root = (
        data_root / "organized"
    )

    output_root = (
        project_root / OUTPUT_DIR_NAME
    )

    return {
        "project_root": project_root,
        "data_root": data_root,
        "organized_root": organized_root,
        "output_root": output_root,
        "checkpoint_root":
            output_root / "checkpoints",
        "result_root":
            output_root / "results",
        "split_root":
            output_root / "splits",
    }


# ============================================================
# FOLDER NAME -> CLASS
# ============================================================

def class_from_folder_name(
    folder_name: str
) -> int | None:

    """
    The provided structure uses:
        0_No_DR
        1_Mild
        2_Moderate
        3_Severe
        4_Proliferative_DR

    We use the leading integer so minor naming differences such as
    '4_Proliferative_DR' / '4_Proliferate_DR' are both accepted.
    """

    name = folder_name.strip()

    if len(name) >= 2:
        first = name[0]

        if (
            first.isdigit()
            and name[1] in {"_", "-", " "}
        ):
            class_id = int(first)

            if 0 <= class_id < NUM_CLASSES:
                return class_id

    return None


# ============================================================
# DATASET SCANNER
# ============================================================

def scan_dataset_images(
    dataset_name: str,
    organized_root: Path
) -> pd.DataFrame:

    """
    Recursively scans:
        data/organized/<dataset>/

    Only files inside a directory whose name begins with a valid
    class number are accepted.

    Both existing 'train' and 'test' directories are scanned because
    the script will create a NEW 80/10/10 split from all available
    original images.
    """

    dataset_root = (
        organized_root / dataset_name
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"  {dataset_root}"
        )

    rows = []

    print(
        f"\nScanning {dataset_name.upper()} images:"
    )

    all_files = []

    for path in dataset_root.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            all_files.append(path)

    all_files.sort()

    if not all_files:
        raise RuntimeError(
            f"No supported image files found in:\n"
            f"{dataset_root}"
        )

    seen_dataset_filename = set()

    for image_path in tqdm(
        all_files,
        desc=dataset_name.upper()
    ):

        class_id = None

        # Walk upward until the dataset root.
        current = image_path.parent

        while current != dataset_root:

            class_candidate = class_from_folder_name(
                current.name
            )

            if class_candidate is not None:
                class_id = class_candidate
                break

            current = current.parent

        if class_id is None:
            continue

        relative_path = image_path.relative_to(
            dataset_root
        )

        filename_key = image_path.name.lower()

        # The same filename twice within one dataset could represent
        # duplicated source samples in different folders.
        if filename_key in seen_dataset_filename:
            raise RuntimeError(
                f"Duplicate filename detected in dataset "
                f"{dataset_name}: {image_path.name}\n"
                "Please inspect the organized directory before training."
            )

        seen_dataset_filename.add(
            filename_key
        )

        sample_id = (
            f"{dataset_name}:"
            f"{relative_path.as_posix()}"
        )

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "id_code": image_path.stem,
                "filename": image_path.name,
                "image_path": str(
                    image_path.resolve()
                ),
                "relative_path":
                    relative_path.as_posix(),
                "source_split":
                    (
                        relative_path.parts[0]
                        if len(relative_path.parts) > 1
                        else "unknown"
                    ),
                "diagnosis": class_id,
                "class_name":
                    CLASS_NAMES[class_id],
            }
        )

    if not rows:
        raise RuntimeError(
            f"No class-labeled images were found under:\n"
            f"{dataset_root}"
        )

    return pd.DataFrame(rows)


def build_full_manifest(
    selected_datasets: List[str],
    organized_root: Path
) -> pd.DataFrame:

    manifests = []

    for dataset_name in selected_datasets:

        manifest = scan_dataset_images(
            dataset_name,
            organized_root
        )

        manifests.append(manifest)

    df = pd.concat(
        manifests,
        ignore_index=True
    )

    # Ensure every sample ID is unique.
    if df["sample_id"].duplicated().any():
        duplicates = df[
            df["sample_id"].duplicated(
                keep=False
            )
        ]

        raise RuntimeError(
            "Duplicate sample IDs detected:\n"
            f"{duplicates[['sample_id', 'image_path']].to_string(index=False)}"
        )

    return df.reset_index(drop=True)


# ============================================================
# STRATIFIED 80/10/10 SPLIT WITHOUT SCIKIT-LEARN
# ============================================================

def split_one_group(
    group_df: pd.DataFrame,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Split one dataset+class group into approximately:
        80% train
        10% validation
        10% test

    The function guarantees all partitions receive at least one image
    when a group contains >= 3 samples.
    """

    group_df = group_df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    n = len(group_df)

    if n < 3:
        raise ValueError(
            f"Group has only {n} samples; cannot make "
            "an 80/10/10 split."
        )

    n_train = int(
        round(
            n * 0.80
        )
    )

    n_val = int(
        round(
            n * 0.10
        )
    )

    n_test = n - n_train - n_val

    # Guarantee non-zero val/test when possible.
    if n_val < 1:
        n_val = 1

    if n_test < 1:
        n_test = 1

    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Unable to construct training split for "
            f"group size {n}."
        )

    train_df = group_df.iloc[
        :n_train
    ].copy()

    val_df = group_df.iloc[
        n_train:n_train + n_val
    ].copy()

    test_df = group_df.iloc[
        n_train + n_val:
    ].copy()

    return (
        train_df,
        val_df,
        test_df,
    )


def stratified_dataset_split(
    df: pd.DataFrame,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Stratification key:
        dataset + diagnosis

    Each run uses one selected dataset only.
    """

    train_parts = []
    val_parts = []
    test_parts = []

    grouped = df.groupby(
        ["dataset", "diagnosis"],
        sort=True
    )

    group_number = 0

    for (
        (dataset_name, diagnosis),
        group_df
    ) in grouped:

        (
            group_train,
            group_val,
            group_test,
        ) = split_one_group(
            group_df,
            seed + group_number
        )

        train_parts.append(
            group_train
        )

        val_parts.append(
            group_val
        )

        test_parts.append(
            group_test
        )

        group_number += 1

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

    # --------------------------------------------------------
    # Explicit leakage checks
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

    if train_ids & val_ids:
        raise RuntimeError(
            "Train/validation overlap detected."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Train/test overlap detected."
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Validation/test overlap detected."
        )

    return (
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# TRAIN-ONLY OVERSAMPLING
# ============================================================

def oversample_training_data(
    train_df: pd.DataFrame,
    seed: int = SEED
) -> pd.DataFrame:

    """
    Oversampling is performed ONLY on train_df.

    Because APTOS and DDR are combined, balancing is performed
    separately inside each dataset so one dataset does not dominate
    the balanced training pool.

    For each dataset:
        every DR class is raised to that dataset's majority-class
        training count by sampling with replacement.
    """

    output_parts = []

    print("\n")
    print("=" * 80)
    print("TRAIN-ONLY RANDOM OVERSAMPLING")
    print("=" * 80)

    for dataset_name in (
        train_df["dataset"]
        .drop_duplicates()
        .tolist()
    ):

        dataset_train = train_df[
            train_df["dataset"] == dataset_name
        ].copy()

        counts = (
            dataset_train["diagnosis"]
            .value_counts()
            .sort_index()
        )

        target = int(
            counts.max()
        )

        print(
            f"\n{dataset_name.upper()} "
            f"target per class: {target}"
        )

        for class_id in range(NUM_CLASSES):

            class_df = dataset_train[
                dataset_train["diagnosis"]
                == class_id
            ].copy()

            current_count = len(
                class_df
            )

            if current_count == 0:
                raise RuntimeError(
                    f"{dataset_name} class {class_id} "
                    f"({CLASS_NAMES[class_id]}) is empty in train."
                )

            if current_count < target:

                class_df = class_df.sample(
                    n=target,
                    replace=True,
                    random_state=(
                        seed
                        + class_id
                    )
                )

            output_parts.append(
                class_df
            )

            print(
                f"  {class_id} "
                f"{CLASS_NAMES[class_id]:18s}: "
                f"{current_count:6d} -> "
                f"{len(class_df):6d}"
            )

    balanced_df = pd.concat(
        output_parts,
        ignore_index=True
    )

    balanced_df = balanced_df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    return balanced_df


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

        row = self.df.iloc[index]

        image_path = row["image_path"]

        label = int(
            row["diagnosis"]
        )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:

            raise RuntimeError(
                f"Failed to read image:\n"
                f"{image_path}"
            ) from exc

        if self.transform is not None:
            image = self.transform(
                image
            )

        return image, label


# ============================================================
# PIL -> [0,1] FLOAT TENSOR
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

        tensor = torch.from_numpy(
            array
        ).permute(
            2,
            0,
            1
        )

        return tensor


# ============================================================
# PAPER GEOMETRIC AUGMENTATION
# ============================================================

class RandomPaperAffine:

    """
    Implements the stated paper geometric augmentation in PyTorch.

    Parameters:
        rotation: ±40°
        translation: ±20% width/height
        shear: ±atan(0.2) ≈ ±11.31°
        zoom: 0.8 to 1.2

    The image is edge-padded before affine transformation. This provides
    border replication and approximates Keras fill_mode='nearest'.
    """

    def __init__(
        self,
        degrees: float = 40.0,
        translate: Tuple[float, float] = (0.2, 0.2),
        shear_degrees: float = 11.31,
        scale_range: Tuple[float, float] = (
            0.8,
            1.2,
        ),
    ):

        self.degrees = degrees
        self.translate = translate
        self.shear_degrees = shear_degrees
        self.scale_range = scale_range

    @staticmethod
    def _edge_pad(
        image: Image.Image,
        pad: int
    ) -> Image.Image:

        """
        Replicate edge pixels using NumPy.

        Unlike a constant fill, this approximates nearest-neighbor
        border filling.
        """

        array = np.asarray(
            image
        )

        padded = np.pad(
            array,
            (
                (pad, pad),
                (pad, pad),
                (0, 0),
            ),
            mode="edge"
        )

        return Image.fromarray(
            padded
        )

    def __call__(
        self,
        image: Image.Image
    ) -> Image.Image:

        width, height = image.size

        angle = random.uniform(
            -self.degrees,
            self.degrees
        )

        max_dx = (
            self.translate[0]
            * width
        )

        max_dy = (
            self.translate[1]
            * height
        )

        translate_x = random.uniform(
            -max_dx,
            max_dx
        )

        translate_y = random.uniform(
            -max_dy,
            max_dy
        )

        shear_x = random.uniform(
            -self.shear_degrees,
            self.shear_degrees
        )

        shear_y = random.uniform(
            -self.shear_degrees,
            self.shear_degrees
        )

        scale = random.uniform(
            self.scale_range[0],
            self.scale_range[1]
        )

        # A large enough edge pad means most valid output pixels never
        # require constant fill outside the padded image.
        pad = int(
            math.ceil(
                max(width, height)
                * 1.5
            )
        )

        padded = self._edge_pad(
            image,
            pad
        )

        padded_width, padded_height = (
            padded.size
        )

        transformed = TF.affine(
            padded,
            angle=angle,
            translate=[
                int(round(translate_x)),
                int(round(translate_y)),
            ],
            scale=scale,
            shear=[
                shear_x,
                shear_y,
            ],
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
            center=[
                padded_width / 2.0,
                padded_height / 2.0,
        ])

        left = pad
        top = pad
        right = pad + width
        bottom = pad + height

        transformed = transformed.crop(
            (
                left,
                top,
                right,
                bottom,
            )
        )

        return transformed


# ============================================================
# NORMALIZATION STATISTICS
# ============================================================

def calculate_mean_std(
    train_df: pd.DataFrame
) -> Tuple[List[float], List[float]]:

    """
    Calculate channel-wise mean/std from ORIGINAL train split.

    No oversampling.
    No augmentation.
    Pixels are resized to 224x224 and scaled to [0,1].
    """

    dataset = DRDataset(
        train_df,
        transform=T.Compose([
            T.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
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
    print("CALCULATING NORMALIZATION STATISTICS")
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

            channel_sum += (
                pixels.sum(
                    dim=1
                )
            )

            channel_squared_sum += (
                (pixels * pixels).sum(
                    dim=1
                )
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
        - mean * mean
    )

    variance = torch.clamp(
        variance,
        min=0.0
    )

    std = torch.sqrt(
        variance
    )

    mean_list = [
        float(x)
        for x in mean.tolist()
    ]

    std_list = [
        float(x)
        for x in std.tolist()
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
        std_list,
    )


# ============================================================
# TRANSFORMS
# ============================================================

def build_transforms(
    mean: List[float],
    std: List[float]
):

    train_transform = T.Compose([

        # Paper input resolution
        T.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        # Rotation + translation + shear + zoom
        RandomPaperAffine(
            degrees=40.0,
            translate=(
                0.2,
                0.2,
            ),
            shear_degrees=11.31,
            scale_range=(
                0.8,
                1.2,
            ),
        ),

        # Horizontal flip
        T.RandomHorizontalFlip(
            p=0.5
        ),

        # Numerical brightness/contrast range was not supplied by
        # the paper. 0.2 is an explicit implementation approximation.
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
        ),

        # [0,1]
        ToFloatTensor(),

        # zero mean / unit variance
        T.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    eval_transform = T.Compose([

        T.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        ToFloatTensor(),

        T.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    return (
        train_transform,
        eval_transform,
    )


# ============================================================
# MODEL
# ============================================================

def build_chexnet(
    use_pretrained: bool = True
) -> nn.Module:

    """
    Build DenseNet121 and replace the ImageNet classifier by a
    5-class DR classifier.

    If use_pretrained=True:
        torchvision ImageNet weights are used.

    If use_pretrained=False:
        DenseNet121 is created without downloading ImageNet weights.

    When an original CheXNet checkpoint is supplied, the model is
    deliberately created with use_pretrained=False and the compatible
    CheXNet DenseNet121 backbone tensors are loaded afterward.
    """

    if use_pretrained:

        print(
            "\nLoading torchvision DenseNet121 "
            "ImageNet-pretrained weights..."
        )

        weights = (
            torchvision.models
            .DenseNet121_Weights
            .DEFAULT
        )

        model = torchvision.models.densenet121(
            weights=weights
        )

    else:

        print(
            "\nBuilding DenseNet121 "
            "without pretrained weights..."
        )

        model = torchvision.models.densenet121(
            weights=None
        )

    in_features = (
        model.classifier.in_features
    )

    model.classifier = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    print(
        f"Final classifier: "
        f"Linear({in_features}, {NUM_CLASSES})"
    )

    return model


# ============================================================
# CHEXNET CHECKPOINT LOADER
# ============================================================

def load_chexnet_backbone(
    model: nn.Module,
    checkpoint_path: Path
) -> nn.Module:

    """
    Load compatible DenseNet121 backbone tensors from an original
    CheXNet checkpoint.

    Classifier weights are intentionally ignored because original
    CheXNet was trained for thoracic disease labels, not DR.
    """

    print("\n")
    print("=" * 80)
    print("LOADING ORIGINAL CHEXNET CHECKPOINT")
    print("=" * 80)

    print(
        "Checkpoint:",
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = (
            checkpoint["state_dict"]
        )

    else:
        state_dict = checkpoint

    if not isinstance(
        state_dict,
        dict
    ):
        raise RuntimeError(
            "CheXNet checkpoint is not a valid state_dict."
        )

    model_state = (
        model.state_dict()
    )

    compatible = {}

    for key, value in (
        state_dict.items()
    ):

        clean_key = str(key)

        if clean_key.startswith(
            "module."
        ):

            clean_key = clean_key[
                len("module.") :
            ]

        # Common original CheXNet naming
        if clean_key.startswith(
            "densenet121."
        ):

            clean_key = clean_key[
                len("densenet121.") :
            ]

        # Some checkpoints may include a model wrapper.
        if clean_key.startswith(
            "model.densenet121."
        ):

            clean_key = clean_key[
                len("model.densenet121.") :
            ]

        if (
            clean_key in model_state
            and model_state[
                clean_key
            ].shape
            == value.shape
        ):

            # Do NOT load the original classifier
            if clean_key.startswith(
                "classifier."
            ):
                continue

            compatible[
                clean_key
            ] = value

    if not compatible:
        raise RuntimeError(
            "No compatible DenseNet121 backbone tensors were found "
            f"in checkpoint: {checkpoint_path}"
        )

    model_state.update(
        compatible
    )

    model.load_state_dict(
        model_state
    )

    print(
        f"Loaded {len(compatible)} compatible "
        "DenseNet121 backbone tensors."
    )

    print(
        "Original CheXNet classifier ignored."
    )

    print(
        "New 5-class DR classifier remains trainable."
    )

    return model


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    original_train_df: pd.DataFrame
) -> torch.Tensor:

    """
    w_c = N / (C * n_c)

    Uses ORIGINAL training distribution.
    """

    counts = (
        original_train_df[
            "diagnosis"
        ]
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
            "A DR class has no training samples."
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
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc="Train",
        leave=False
    )

    for images, labels in pbar:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
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

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        running_loss / total,
        correct / total,
    )


# ============================================================
# NUMPY METRICS
# ============================================================

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


def make_confusion_matrix(
    true_labels: List[int],
    predictions: List[int]
) -> np.ndarray:

    cm = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64
    )

    for true_label, pred_label in zip(
        true_labels,
        predictions
    ):

        cm[
            int(true_label),
            int(pred_label)
        ] += 1

    return cm


def calculate_metrics(
    true_labels: List[int],
    predictions: List[int]
) -> Dict:

    cm = make_confusion_matrix(
        true_labels,
        predictions
    )

    total = int(
        cm.sum()
    )

    accuracy = safe_divide(
        np.trace(cm),
        total
    )

    class_metrics = []

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
            supports[class_id]
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

        class_metrics.append(
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
            for x in class_metrics
        ),
        total
    )

    weighted_recall = safe_divide(
        sum(
            x["recall"]
            * x["support"]
            for x in class_metrics
        ),
        total
    )

    weighted_f1 = safe_divide(
        sum(
            x["f1"]
            * x["support"]
            for x in class_metrics
        ),
        total
    )

    macro_precision = float(
        np.mean(
            [
                x["precision"]
                for x in class_metrics
            ]
        )
    )

    macro_recall = float(
        np.mean(
            [
                x["recall"]
                for x in class_metrics
            ]
        )
    )

    macro_f1 = float(
        np.mean(
            [
                x["f1"]
                for x in class_metrics
            ]
        )
    )

    return {
        "accuracy":
            accuracy,
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
            class_metrics,
        "confusion_matrix":
            cm,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict:

    model.eval()

    total_loss = 0.0
    total_examples = 0

    true_labels = []
    predictions = []

    with torch.inference_mode():

        for images, labels in tqdm(
            loader,
            desc="Eval",
            leave=False
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            batch_size = labels.size(
                0
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_examples += (
                batch_size
            )

            batch_predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            true_labels.extend(
                labels.cpu().tolist()
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

    metrics = calculate_metrics(
        true_labels,
        predictions
    )

    metrics["loss"] = (
        total_loss
        / total_examples
    )

    metrics["labels"] = (
        true_labels
    )

    metrics["predictions"] = (
        predictions
    )

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
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "metrics": {
                "accuracy":
                    metrics["accuracy"],
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
            "mean": mean,
            "std": std,
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

def save_training_curves(
    history: Dict,
    result_dir: Path
) -> None:

    epochs = np.arange(
        1,
        len(
            history[
                "train_loss"
            ]
        ) + 1
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
        "CheXNet Accuracy"
    )

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        result_dir
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
        "CheXNet Loss"
    )

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        result_dir
        / "loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_confusion_matrix(
    cm: np.ndarray,
    result_dir: Path,
    prefix: str = "test"
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
        np.arange(
            NUM_CLASSES
        )
    )

    ax.set_yticks(
        np.arange(
            NUM_CLASSES
        )
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
        "CheXNet Confusion Matrix"
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
        result_dir
        / f"{prefix}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    np.savetxt(
        result_dir
        / f"{prefix}_confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d"
    )


# ============================================================
# REPORT WRITER
# ============================================================

def save_test_report(
    metrics: Dict,
    result_dir: Path
) -> None:

    path = (
        result_dir
        / "test_report.txt"
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CHEXNET DIABETIC RETINOPATHY TEST REPORT\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Loss       : "
            f"{metrics['loss']:.6f}\n"
        )

        f.write(
            f"Accuracy   : "
            f"{metrics['accuracy']:.6f}\n"
        )

        f.write(
            f"Weighted P : "
            f"{metrics['weighted_precision']:.6f}\n"
        )

        f.write(
            f"Weighted R : "
            f"{metrics['weighted_recall']:.6f}\n"
        )

        f.write(
            f"Weighted F1: "
            f"{metrics['weighted_f1']:.6f}\n"
        )

        f.write(
            f"Macro F1   : "
            f"{metrics['macro_f1']:.6f}\n\n"
        )

        f.write(
            "PER-CLASS METRICS\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        for item in (
            metrics["per_class"]
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
            "-" * 60
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
# PRINT DISTRIBUTIONS
# ============================================================

def print_class_distribution(
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

    if "dataset" in df.columns:

        for dataset_name in (
            df["dataset"]
            .drop_duplicates()
            .tolist()
        ):

            subset = df[
                df["dataset"]
                == dataset_name
            ]

            counts = (
                subset[
                    "diagnosis"
                ]
                .value_counts()
                .sort_index()
            )

            print(
                f"\n{dataset_name.upper()}: "
                f"{len(subset)}"
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
                    f"  {class_id} "
                    f"{CLASS_NAMES[class_id]:18s}: "
                    f"{count}"
                )

    else:

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
                f"  {class_id} "
                f"{CLASS_NAMES[class_id]:18s}: "
                f"{count}"
            )


# ============================================================
# SAVE CONFIG
# ============================================================

def make_config(
    args: argparse.Namespace,
    selected_dataset: str,
    paths: Dict[str, Path],
    mean: List[float],
    std: List[float],
    train_original: pd.DataFrame,
    train_balanced: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    initialization: str
) -> Dict:

    return {
        "model":
            "CheXNet / DenseNet121 baseline",
        "initialization":
            initialization,
        "dataset":
            selected_dataset,
        "project_root":
            str(paths["project_root"]),
        "organized_root":
            str(paths["organized_root"]),
        "image_size":
            IMAGE_SIZE,
        "num_classes":
            NUM_CLASSES,
        "class_names":
            CLASS_NAMES,
        "split":
            "Stratified 80/10/10 before oversampling",
        "stratification":
            "dataset + DR class",
        "oversampling":
            (
                "Random oversampling with replacement "
                "on TRAIN ONLY, separately within each dataset"
                if not args.no_oversampling
                else "Disabled"
            ),
        "validation_policy":
            "Untouched",
        "test_policy":
            "Untouched",
        "normalization_source":
            "Original training split before oversampling",
        "normalization_mean":
            mean,
        "normalization_std":
            std,
        "rotation_degrees":
            40,
        "translation":
            [0.2, 0.2],
        "shear_approx_degrees":
            11.31,
        "zoom_scale_range":
            [0.8, 1.2],
        "horizontal_flip_probability":
            0.5,
        "brightness_jitter":
            0.2,
        "contrast_jitter":
            0.2,
        "epochs":
            args.epochs,
        "batch_size":
            args.batch_size,
        "workers":
            args.workers,
        "learning_rate":
            args.lr,
        "weight_decay":
            args.weight_decay,
        "seed":
            SEED,
        "train_original_size":
            len(train_original),
        "train_balanced_size":
            len(train_balanced),
        "validation_size":
            len(val_df),
        "test_size":
            len(test_df),
    }


# ============================================================
# DATA QUALITY CHECK
# ============================================================

def verify_images(
    df: pd.DataFrame,
    limit: int | None = None
) -> None:

    """
    Opens images once to catch corrupt/missing images before training.

    By default all manifest images are checked.
    """

    target = df

    if limit is not None:
        target = df.head(
            limit
        )

    print("\n")
    print("=" * 80)
    print("VERIFYING IMAGE FILES")
    print("=" * 80)

    failures = []

    for _, row in tqdm(
        target.iterrows(),
        total=len(target),
        desc="Image validation"
    ):

        path = row["image_path"]

        try:

            with Image.open(
                path
            ) as image:

                image.verify()

        except Exception as exc:

            failures.append(
                (
                    path,
                    str(exc)
                )
            )

    if failures:

        message = "\n".join(
            [
                f"{path} -> {error}"
                for path, error
                in failures[:20]
            ]
        )

        raise RuntimeError(
            "Image verification failed.\n"
            f"{message}\n"
            f"Total failures: {len(failures)}"
        )

    print(
        f"Verified {len(target)} images successfully."
    )


# ============================================================
# SAVE DATA SPLITS
# ============================================================

def save_dataframes(
    output_root: Path,
    full_df: pd.DataFrame,
    train_original: pd.DataFrame,
    train_balanced: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:

    split_root = (
        output_root
        / "splits"
    )

    split_root.mkdir(
        parents=True,
        exist_ok=True
    )

    full_df.to_csv(
        split_root
        / "full_manifest.csv",
        index=False
    )

    train_original.to_csv(
        split_root
        / "train_original.csv",
        index=False
    )

    train_balanced.to_csv(
        split_root
        / "train_oversampled.csv",
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

def main():

    seed_everything(
        SEED
    )

    args = parse_args()

    project_root = (
        get_project_root()
    )

    paths = get_paths(
        project_root
    )

    # Exactly ONE dataset is selected for each run.
    # Outputs are isolated so APTOS and DDR experiments never mix.
    selected_dataset = args.dataset

    if selected_dataset not in DATASET_NAMES:
        raise ValueError(
            f"Unsupported dataset: {selected_dataset}. "
            f"Choose one of: {DATASET_NAMES}"
        )

    if args.output_dir:
        output_root = Path(
            args.output_dir
        ).resolve()

    else:
        output_root = (
            paths["output_root"]
            / selected_dataset
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
        "CHEXNET — 5-CLASS DIABETIC RETINOPATHY"
    )
    print("=" * 90)

    print(
        "Project root:",
        project_root
    )

    print(
        "Organized data:",
        paths["organized_root"]
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

    print(
        "Selected dataset:",
        selected_dataset.upper()
    )

    # --------------------------------------------------------
    # Check the exact directory structure
    # --------------------------------------------------------

    expected = (
        paths["organized_root"]
        / selected_dataset
    )

    print(
        f"\n{selected_dataset.upper()} root:"
    )

    print(
        expected
    )

    if not expected.is_dir():

        raise FileNotFoundError(
            f"\nExpected directory does not exist:\n"
            f"{expected}\n\n"
            "Make sure the script is located in the project "
            "root containing the 'data' directory."
        )

    # --------------------------------------------------------
    # Scan ONLY the selected dataset
    # --------------------------------------------------------

    full_df = build_full_manifest(
        [selected_dataset],
        paths["organized_root"]
    )

    print_class_distribution(
        "FULL IMAGE MANIFEST",
        full_df
    )

    # --------------------------------------------------------
    # Optional known paper sanity values
    # --------------------------------------------------------

    print(
        "\nThe script is using image files from the organized "
        "folders rather than pre-existing combined split CSVs."
    )

    # --------------------------------------------------------
    # Verify images
    # --------------------------------------------------------

    verify_images(
        full_df
    )

    # --------------------------------------------------------
    # NEW 80/10/10 split BEFORE oversampling
    # --------------------------------------------------------

    (
        train_original_df,
        val_df,
        test_df,
    ) = stratified_dataset_split(
        full_df,
        seed=SEED
    )

    print_class_distribution(
        "ORIGINAL TRAIN — BEFORE OVERSAMPLING",
        train_original_df
    )

    print_class_distribution(
        "VALIDATION — UNTOUCHED",
        val_df
    )

    print_class_distribution(
        "TEST — UNTOUCHED",
        test_df
    )

    # --------------------------------------------------------
    # Oversample TRAIN ONLY
    # --------------------------------------------------------

    if args.no_oversampling:

        train_balanced_df = (
            train_original_df.copy()
        )

        print(
            "\nTraining oversampling: DISABLED"
        )

    else:

        train_balanced_df = (
            oversample_training_data(
                train_original_df,
                seed=SEED
            )
        )

    print_class_distribution(
        "TRAIN — AFTER TRAIN-ONLY OVERSAMPLING",
        train_balanced_df
    )

    # --------------------------------------------------------
    # Save split manifests
    # --------------------------------------------------------

    save_dataframes(
        output_root,
        full_df,
        train_original_df,
        train_balanced_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    mean, std = calculate_mean_std(
        train_original_df
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
                "mean": mean,
                "std": std,
                "image_size": IMAGE_SIZE,
                "source":
                    "Original training split before oversampling",
            },
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    (
        train_transform,
        eval_transform,
    ) = build_transforms(
        mean,
        std
    )

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = DRDataset(
        train_balanced_df,
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

    loader_kwargs = {
        "num_workers":
            args.workers,
        "pin_memory":
            torch.cuda.is_available(),
    }

    if args.workers > 0:
        loader_kwargs[
            "persistent_workers"
        ] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
    )

    print("\n")
    print("=" * 80)
    print("DATALOADER SUMMARY")
    print("=" * 80)

    print(
        "Train images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print(
        "Test images:",
        len(test_dataset)
    )

    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print(
        "Test batches:",
        len(test_loader)
    )

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # If an original CheXNet checkpoint is supplied, DO NOT load
    # torchvision ImageNet weights first. That would unnecessarily
    # download the 30.8 MB DenseNet121 ImageNet checkpoint and then
    # overwrite the backbone with CheXNet weights.
    #
    # Instead:
    #   CheXNet checkpoint supplied
    #       -> DenseNet121 weights=None
    #       -> replace classifier with 5-class head
    #       -> load compatible CheXNet backbone tensors
    #
    # No CheXNet checkpoint supplied:
    #       -> optionally use torchvision ImageNet weights
    #
    # --------------------------------------------------------

    chexnet_checkpoint_path = None

    if args.chexnet_checkpoint:

        chexnet_checkpoint_path = Path(
            args.chexnet_checkpoint
        )

        if not chexnet_checkpoint_path.is_absolute():

            chexnet_checkpoint_path = (
                project_root
                / chexnet_checkpoint_path
            )

        chexnet_checkpoint_path = (
            chexnet_checkpoint_path.resolve()
        )

        if not chexnet_checkpoint_path.is_file():

            raise FileNotFoundError(
                "CheXNet checkpoint not found:\n"
                f"{chexnet_checkpoint_path}"
            )

        print("\n")
        print("=" * 80)
        print("CHEXNET CHECKPOINT DETECTED")
        print("=" * 80)

        print(
            "Checkpoint:",
            chexnet_checkpoint_path
        )

        print(
            "Skipping torchvision ImageNet weight download."
        )

        # Build architecture WITHOUT downloading ImageNet weights.
        model = build_chexnet(
            use_pretrained=False
        )

        # Load the actual CheXNet DenseNet121 backbone.
        model = load_chexnet_backbone(
            model,
            chexnet_checkpoint_path
        )

        initialization = (
            f"Original CheXNet checkpoint: "
            f"{chexnet_checkpoint_path}"
        )

    else:

        print("\n")
        print("=" * 80)
        print("CHEXNET CHECKPOINT")
        print("=" * 80)

        print(
            "No --chexnet-checkpoint supplied."
        )

        if args.no_pretrained:

            model = build_chexnet(
                use_pretrained=False
            )

            initialization = (
                "random DenseNet121 initialization"
            )

            print(
                "Using random DenseNet121 initialization."
            )

        else:

            model = build_chexnet(
                use_pretrained=True
            )

            initialization = (
                "torchvision DenseNet121 ImageNet weights"
            )

            print(
                "Using torchvision ImageNet-pretrained "
                "DenseNet121 initialization."
            )

            print(
                "This is transfer learning, but it is "
                "not guaranteed to be the paper's exact "
                "CheXNet checkpoint."
            )

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_original_df
        )
        .to(DEVICE)
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print("\n")
    print("=" * 80)
    print("CLASS WEIGHTS")
    print("=" * 80)

    for class_id, weight in enumerate(
        class_weights.detach().cpu().tolist()
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

    print("\n")
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)

    print(
        "Epochs:",
        args.epochs
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Learning rate:",
        args.lr
    )

    print(
        "Weight decay:",
        args.weight_decay
    )

    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "Normalization mean:",
        mean
    )

    print(
        "Normalization std:",
        std
    )

    print(
        "Initialization:",
        initialization
    )

    # --------------------------------------------------------
    # Save experiment config
    # --------------------------------------------------------

    config = make_config(
        args,
        selected_dataset,
        paths,
        mean,
        std,
        train_original_df,
        train_balanced_df,
        val_df,
        test_df,
        initialization
    )

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
    # History
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_weighted_precision": [],
        "val_weighted_recall": [],
        "val_weighted_f1": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    best_val_accuracy = -1.0
    best_epoch = -1

    best_checkpoint = (
        checkpoint_root
        / "chexnet_best.pth"
    )

    last_checkpoint = (
        checkpoint_root
        / "chexnet_last.pth"
    )

    # --------------------------------------------------------
    # TRAINING
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
                optimizer,
                DEVICE
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
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
            "val_weighted_precision"
        ].append(
            val_metrics[
                "weighted_precision"
            ]
        )

        history[
            "val_weighted_recall"
        ].append(
            val_metrics[
                "weighted_recall"
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
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Acc  : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Val Acc    : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val W-F1   : "
            f"{val_metrics['weighted_f1']:.4f}"
        )

        print(
            f"Val Macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Time       : "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save latest
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
        # Save best
        # ----------------------------------------------------

        if (
            val_metrics["accuracy"]
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
                "\n*** BEST MODEL SAVED ***"
            )

            print(
                "Path:",
                best_checkpoint
            )

    total_time = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        result_root
        / "training_history.csv",
        index=False
    )

    save_training_curves(
        history,
        result_root
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("LOADING BEST CHECKPOINT")
    print("=" * 80)

    checkpoint = torch.load(
        best_checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best validation accuracy:",
        checkpoint[
            "metrics"
        ]["accuracy"]
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print(
        f"\nTest Loss       : "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Test Accuracy   : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Weighted Precision: "
        f"{test_metrics['weighted_precision']:.6f}"
    )

    print(
        f"Weighted Recall : "
        f"{test_metrics['weighted_recall']:.6f}"
    )

    print(
        f"Weighted F1     : "
        f"{test_metrics['weighted_f1']:.6f}"
    )

    print(
        f"Macro F1        : "
        f"{test_metrics['macro_f1']:.6f}"
    )

    print("\nPer-class results:")

    print(
        f"{'Class':20s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'Support':>12s}"
    )

    for item in (
        test_metrics["per_class"]
    ):

        print(
            f"{item['class_name']:20s}"
            f"{item['precision']:>12.4f}"
            f"{item['recall']:>12.4f}"
            f"{item['f1']:>12.4f}"
            f"{item['support']:>12d}"
        )

    # --------------------------------------------------------
    # Save final outputs
    # --------------------------------------------------------

    save_confusion_matrix(
        test_metrics[
            "confusion_matrix"
        ],
        result_root,
        prefix="test"
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
            str(best_checkpoint),
        "last_checkpoint":
            str(last_checkpoint),
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
    # Final message
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("TRAINING COMPLETE")
    print("=" * 90)

    print(
        "Total training time:",
        f"{total_time / 60.0:.2f} minutes"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.6f}"
    )

    print(
        "Test accuracy:",
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        best_checkpoint
    )

    print(
        "\nResults:"
    )

    print(
        result_root
    )

    print(
        "\nSplits:"
    )

    print(
        split_root
    )


if __name__ == "__main__":
    main()
