"""
efficientnetb0_paper_replication.py

Paper-grounded EfficientNet-B0 baseline replication for:
Dharrao et al. (2025), "AI-driven detection and classification of diabetic
retinopathy stages using EfficientNetB0", Discover Applied Sciences 7:1400.
DOI: 10.1007/s42452-025-07998-9

IMPORTANT REPRODUCIBILITY NOTES
--------------------------------
The paper explicitly documents:
  * EfficientNetB0 backbone
  * transfer learning / fine-tuning
  * input 224x224x3
  * green-channel emphasis/extraction
  * CLAHE/AHE contrast enhancement
  * median filtering
  * normalization to [0,1] and/or zero-mean/unit-variance
  * training augmentation: random rotation +/-15 degrees, horizontal flip,
    vertical flip, translation, brightness, contrast, gamma, elastic
    deformation, cutout
  * class-balanced sampling with alpha=0.5
  * categorical cross-entropy
  * Adam optimizer
  * Reduce-on-plateau learning-rate scheduling: factor=0.1 after 10 epochs
  * fine-tuning / end-to-end CNN training
  * Grad-CAM explainability

The paper does NOT publish enough numerical detail to reproduce every training
hyperparameter exactly. In particular it does not state a single exact value
for initial learning rate, batch size, total epochs, CLAHE parameters, median
filter size, augmentation probabilities/ranges for several transforms, or the
exact pretrained checkpoint file.

This script therefore does two things explicitly:
  1. Implements every paper-described preprocessing/augmentation technique.
  2. Uses clearly labeled implementation defaults for quantities the paper
     leaves unspecified. These defaults are saved in config.json.

USER'S DATA PROTOCOL
--------------------
The script follows the requested protocol:
  CLEANED DATASET -> STRATIFIED 80/10/10 SPLIT -> TRAIN-ONLY OVERSAMPLING

This intentionally differs from the paper's written balancing order, which
oversampled/balanced before splitting. Validation and test are NEVER
oversampled and NEVER receive augmentation.

Dataset source
--------------
The script scans the union of all images under:
  data/organized/aptos
or
  data/organized/ddr
regardless of their existing train/val/test subfolder. This allows a fresh
80/10/10 split of your already-cleaned images without using the previous split
as the new experimental split.

Model
-----
Standard torchvision EfficientNet-B0 with local official ImageNet-1K weights.
The paper describes transfer learning/fine-tuning but does not name the exact
pretrained checkpoint file. This implementation therefore uses the official
TorchVision EfficientNet-B0 IMAGENET1K_V1 checkpoint downloaded by the user
and stored locally under weights/. The script never downloads weights at runtime.

No MSAG, no CBAM, no DropBlock, no Transformer, no other enhancement module.

Run from:
  (dr311) A:\\DR_classification>

Examples:
  python efficientnetb0_paper_replication.py --dataset aptos
  python efficientnetb0_paper_replication.py --dataset ddr

Smoke test:
  python efficientnetb0_paper_replication.py --dataset aptos --epochs 1 --smoke-test

Suggested full run on RTX 3060:
  python efficientnetb0_paper_replication.py --dataset aptos --epochs 50 --batch-size 32 --workers 2
  python efficientnetb0_paper_replication.py --dataset ddr --epochs 50 --batch-size 32 --workers 2

Dependencies already expected in the user's environment:
  torch, torchvision, pandas, numpy, Pillow, tqdm, matplotlib, opencv-python

No scikit-learn is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# PAPER / PROJECT CONSTANTS
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

SUPPORTED_DATASETS = {"aptos", "ddr"}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Paper-specified where available.
ROTATION_DEG = 15.0

# -----------------------------------------------------------------
# Explicit implementation defaults for quantities not numerically
# specified by the paper. They are intentionally surfaced in config.
# -----------------------------------------------------------------
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_WORKERS = 2
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_AUG_PROB = 0.5
DEFAULT_TRANSLATION = 0.10
DEFAULT_BRIGHTNESS = 0.20
DEFAULT_CONTRAST = 0.20
DEFAULT_GAMMA_LOW = 0.80
DEFAULT_GAMMA_HIGH = 1.20
DEFAULT_ELASTIC_ALPHA = 20.0
DEFAULT_ELASTIC_SIGMA = 4.0
DEFAULT_CUTOUT_RATIO = 0.20
DEFAULT_CLAHE_CLIP = 2.0
DEFAULT_CLAHE_TILE = 8
DEFAULT_MEDIAN_KERNEL = 3
DEFAULT_PATIENCE = 10
DEFAULT_LR_FACTOR = 0.1
LOCAL_WEIGHTS_RELATIVE = Path("weights") / "efficientnet_b0_rwightman-7f5810bc.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # Reproducible over maximum throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper-grounded EfficientNetB0 DR replication without MSAG."
    )
    p.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS), required=True)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--output-dir", type=str, default=None)

    # Explicit preprocessing choices.
    p.add_argument("--clahe-clip", type=float, default=DEFAULT_CLAHE_CLIP)
    p.add_argument("--clahe-tile", type=int, default=DEFAULT_CLAHE_TILE)
    p.add_argument("--median-kernel", type=int, default=DEFAULT_MEDIAN_KERNEL)

    # Explicit augmentation defaults for unspecified numeric ranges.
    p.add_argument("--aug-prob", type=float, default=DEFAULT_AUG_PROB)
    p.add_argument("--translation", type=float, default=DEFAULT_TRANSLATION)
    p.add_argument("--brightness", type=float, default=DEFAULT_BRIGHTNESS)
    p.add_argument("--contrast", type=float, default=DEFAULT_CONTRAST)
    p.add_argument("--gamma-low", type=float, default=DEFAULT_GAMMA_LOW)
    p.add_argument("--gamma-high", type=float, default=DEFAULT_GAMMA_HIGH)
    p.add_argument("--elastic-alpha", type=float, default=DEFAULT_ELASTIC_ALPHA)
    p.add_argument("--elastic-sigma", type=float, default=DEFAULT_ELASTIC_SIGMA)
    p.add_argument("--cutout-ratio", type=float, default=DEFAULT_CUTOUT_RATIO)

    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--gradcam-samples", type=int, default=8)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


# ============================================================
# PATHS
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parent


def dataset_root(root: Path, dataset_name: str) -> Path:
    path = root / "data" / "organized" / dataset_name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Cleaned organized dataset not found:\n{path}\n"
            "Expected class folders anywhere under this directory."
        )
    return path


def output_root(root: Path, dataset_name: str, custom: Optional[str]) -> Path:
    if custom:
        return Path(custom).resolve()
    return root / "runs_efficientnetb0_paper" / dataset_name


# ============================================================
# CLASS DISCOVERY / MANIFEST
# ============================================================

def parse_class_id(folder_name: str) -> Optional[int]:
    s = folder_name.strip()
    if len(s) >= 2 and s[0].isdigit() and s[1] in {"_", "-", " "}:
        v = int(s[0])
        if 0 <= v < NUM_CLASSES:
            return v
    return None


def infer_class_id(image_path: Path, root: Path) -> Optional[int]:
    current = image_path.parent
    while current != root.parent and current != current.parent:
        value = parse_class_id(current.name)
        if value is not None:
            return value
        if current == root:
            break
        current = current.parent
    return None


def build_manifest(root: Path, dataset_name: str) -> pd.DataFrame:
    droot = dataset_root(root, dataset_name)
    files = sorted(
        p for p in droot.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No supported images found under:\n{droot}")

    rows: List[Dict[str, object]] = []
    seen_paths = set()
    skipped = 0

    print("\n" + "=" * 80)
    print(f"SCANNING CLEANED {dataset_name.upper()} DATASET")
    print("=" * 80)
    print(f"Root: {droot}")
    print(f"Image files found: {len(files)}")

    for p in tqdm(files, desc="Manifest"):
        cid = infer_class_id(p, droot)
        if cid is None:
            skipped += 1
            continue
        rp = p.relative_to(droot).as_posix()
        if rp in seen_paths:
            raise RuntimeError(f"Duplicate relative path detected: {rp}")
        seen_paths.add(rp)
        rows.append(
            {
                "sample_id": f"{dataset_name}:{rp}",
                "dataset": dataset_name,
                "image_path": str(p.resolve()),
                "relative_path": rp,
                "source_split": rp.split("/", 1)[0] if "/" in rp else "unknown",
                "filename": p.name,
                "diagnosis": cid,
                "class_name": CLASS_NAMES[cid],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No class-labeled images were found.")
    if skipped:
        print(f"Skipped unlabeled files: {skipped}")

    counts = df["diagnosis"].value_counts().sort_index()
    print("\nCleaned dataset counts:")
    for cid in range(NUM_CLASSES):
        print(f"  {cid} {CLASS_NAMES[cid]:18s}: {int(counts.get(cid, 0)):6d}")
    print(f"  TOTAL               : {len(df):6d}")

    if set(df["diagnosis"].unique()) != set(range(NUM_CLASSES)):
        raise RuntimeError("All five DR classes must be present.")

    return df.reset_index(drop=True)


# ============================================================
# IMAGE VERIFY
# ============================================================

def verify_manifest(df: pd.DataFrame) -> None:
    failures: List[Tuple[str, str]] = []
    print("\n" + "=" * 80)
    print("VERIFYING ORIGINAL IMAGE FILES")
    print("=" * 80)
    for path in tqdm(df["image_path"], desc="Verify"):
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as exc:
            failures.append((str(path), repr(exc)))
    if failures:
        preview = "\n".join(f"{p} -> {e}" for p, e in failures[:20])
        raise RuntimeError(
            f"Image verification failed: {len(failures)} files\n{preview}"
        )
    print(f"Verified: {len(df)} images")


# ============================================================
# STRATIFIED SPLIT — USER REQUESTED PROTOCOL
# ============================================================

def stratified_split(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Invalid val/test ratios.")

    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for cid in range(NUM_CLASSES):
        part = df[df["diagnosis"] == cid].copy()
        idx = np.arange(len(part))
        rng.shuffle(idx)
        part = part.iloc[idx].reset_index(drop=True)

        n = len(part)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            n_val = max(1, int(math.floor((n - 1) * val_ratio / (val_ratio + test_ratio))))
            n_test = n - n_val - 1

        test = part.iloc[:n_test]
        val = part.iloc[n_test:n_test + n_val]
        train = part.iloc[n_test + n_val:]

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise RuntimeError(f"Split failed for class {cid}")

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)

    return train_df, val_df, test_df


# ============================================================
# TRAIN-ONLY OVERSAMPLING
# ============================================================

def oversample_train_to_majority(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = train_df["diagnosis"].value_counts().sort_index()
    target = int(counts.max())
    pieces = []

    print("\n" + "=" * 80)
    print("TRAIN-ONLY RANDOM OVERSAMPLING")
    print("=" * 80)
    print("Validation/test are NOT oversampled.")
    print(f"Target count per class: {target}")

    for cid in range(NUM_CLASSES):
        part = train_df[train_df["diagnosis"] == cid].copy()
        count = len(part)
        if count == 0:
            raise RuntimeError(f"Empty training class {cid}")
        sampled = part.sample(
            n=target,
            replace=(count < target),
            random_state=seed + 100 + cid,
        ).copy()
        sampled["oversampled"] = sampled.index.duplicated(keep=False)
        sampled = sampled.reset_index(drop=True)
        pieces.append(sampled)
        print(f"  {cid} {CLASS_NAMES[cid]:18s}: {count:6d} -> {len(sampled):6d}")

    out = pd.concat(pieces, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed + 999).reset_index(drop=True)
    out["train_row_id"] = np.arange(len(out))
    return out


# ============================================================
# PAPER PREPROCESSING
# ============================================================

def preprocess_fundus(
    image_path: str,
    clahe_clip: float,
    clahe_tile: int,
    median_kernel: int,
) -> np.ndarray:
    """
    Deterministic preprocessing applied to train/val/test.

    Paper-described techniques:
      1) green-channel extraction/emphasis
      2) CLAHE
      3) median filtering
      4) resize to 224x224
      5) later [0,1] scaling + train-set standardization

    Because the paper's architecture diagram keeps a 224x224x3 input while
    emphasizing the green channel, the green image is repeated across RGB
    channels so the pretrained 3-channel EfficientNet-B0 remains compatible.
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    green = rgb[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip),
        tileGridSize=(int(clahe_tile), int(clahe_tile)),
    )
    green = clahe.apply(green)

    k = int(median_kernel)
    if k % 2 == 0:
        k += 1
    green = cv2.medianBlur(green, k)

    green = cv2.resize(
        green,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    out = np.repeat(green[:, :, None], 3, axis=2)
    return out.astype(np.uint8)


# ============================================================
# AUGMENTATION — TRAIN ONLY
# ============================================================

def warp_affine_replicate(image: np.ndarray, angle: float, tx: float, ty: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(
        image,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def random_translation_rotation(
    image: np.ndarray,
    rng: np.random.Generator,
    p: float,
    translation: float,
) -> np.ndarray:
    if rng.random() >= p:
        return image
    h, w = image.shape[:2]
    angle = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
    tx = float(rng.uniform(-translation, translation) * w)
    ty = float(rng.uniform(-translation, translation) * h)
    return warp_affine_replicate(image, angle, tx, ty)


def random_flip(image: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    if rng.random() < p:
        image = cv2.flip(image, 1)
    if rng.random() < p:
        image = cv2.flip(image, 0)
    return image


def random_brightness_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    p: float,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    if rng.random() >= p:
        return image
    x = image.astype(np.float32) / 255.0
    alpha = float(rng.uniform(1.0 - contrast, 1.0 + contrast))
    beta = float(rng.uniform(-brightness, brightness))
    x = np.clip(alpha * x + beta, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def random_gamma(
    image: np.ndarray,
    rng: np.random.Generator,
    p: float,
    gamma_low: float,
    gamma_high: float,
) -> np.ndarray:
    if rng.random() >= p:
        return image
    gamma = float(rng.uniform(gamma_low, gamma_high))
    x = image.astype(np.float32) / 255.0
    x = np.clip(np.power(x, gamma), 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def elastic_deformation(
    image: np.ndarray,
    rng: np.random.Generator,
    p: float,
    alpha: float,
    sigma: float,
) -> np.ndarray:
    if rng.random() >= p:
        return image
    h, w = image.shape[:2]
    dx = rng.normal(0, 1, (h, w)).astype(np.float32)
    dy = rng.normal(0, 1, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma, sigmaY=sigma) * float(alpha)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma, sigmaY=sigma) * float(alpha)

    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = x + dx
    map_y = y + dy
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def cutout(
    image: np.ndarray,
    rng: np.random.Generator,
    p: float,
    ratio: float,
) -> np.ndarray:
    if rng.random() >= p:
        return image
    h, w = image.shape[:2]
    ch = max(1, int(round(h * ratio)))
    cw = max(1, int(round(w * ratio)))
    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    out = image.copy()
    out[y0:y0 + ch, x0:x0 + cw] = 0
    return out


def augment_train_image(
    image: np.ndarray,
    rng: np.random.Generator,
    aug_prob: float,
    translation: float,
    brightness: float,
    contrast: float,
    gamma_low: float,
    gamma_high: float,
    elastic_alpha: float,
    elastic_sigma: float,
    cutout_ratio: float,
) -> np.ndarray:
    # Order follows the conceptual geometric -> photometric -> regularization flow.
    image = random_translation_rotation(image, rng, aug_prob, translation)
    image = random_flip(image, rng, aug_prob)
    image = elastic_deformation(image, rng, aug_prob, elastic_alpha, elastic_sigma)
    image = random_brightness_contrast(image, rng, aug_prob, brightness, contrast)
    image = random_gamma(image, rng, aug_prob, gamma_low, gamma_high)
    image = cutout(image, rng, aug_prob, cutout_ratio)
    return image


# ============================================================
# NORMALIZATION STATISTICS
# ============================================================

def compute_train_mean_std(
    train_df: pd.DataFrame,
    clahe_clip: float,
    clahe_tile: int,
    median_kernel: int,
) -> Tuple[List[float], List[float]]:
    print("\n" + "=" * 80)
    print("COMPUTING TRAIN-SET NORMALIZATION STATISTICS")
    print("=" * 80)
    print("Statistics are computed ONLY from the original training split.")

    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    count = 0

    for path in tqdm(train_df["image_path"], desc="Mean/std"):
        img = preprocess_fundus(
            str(path),
            clahe_clip,
            clahe_tile,
            median_kernel,
        ).astype(np.float64) / 255.0
        pixels = img.reshape(-1, 3)
        sum_c += pixels.sum(axis=0)
        sumsq_c += np.square(pixels).sum(axis=0)
        count += pixels.shape[0]

    mean = sum_c / count
    var = np.maximum(sumsq_c / count - np.square(mean), 1e-12)
    std = np.sqrt(var)

    mean_list = [float(x) for x in mean]
    std_list = [float(x) for x in std]
    print(f"Mean: {mean_list}")
    print(f"Std : {std_list}")
    return mean_list, std_list


# ============================================================
# DATASET
# ============================================================

class FundusDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mean: Sequence[float],
        std: Sequence[float],
        train: bool,
        seed: int,
        preprocess_cfg: Dict[str, float],
        aug_cfg: Dict[str, float],
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.train = bool(train)
        self.seed = int(seed)
        self.preprocess_cfg = preprocess_cfg
        self.aug_cfg = aug_cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = str(row["image_path"])
        image = preprocess_fundus(
            path,
            float(self.preprocess_cfg["clahe_clip"]),
            int(self.preprocess_cfg["clahe_tile"]),
            int(self.preprocess_cfg["median_kernel"]),
        )

        if self.train:
            # Deterministic per sample/epoch is impossible without tracking epoch;
            # random state is intentionally re-seeded from global worker entropy.
            rng = np.random.default_rng()
            image = augment_train_image(
                image,
                rng,
                float(self.aug_cfg["aug_prob"]),
                float(self.aug_cfg["translation"]),
                float(self.aug_cfg["brightness"]),
                float(self.aug_cfg["contrast"]),
                float(self.aug_cfg["gamma_low"]),
                float(self.aug_cfg["gamma_high"]),
                float(self.aug_cfg["elastic_alpha"]),
                float(self.aug_cfg["elastic_sigma"]),
                float(self.aug_cfg["cutout_ratio"]),
            )

        x = image.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(x)).float()
        label = int(row["diagnosis"])
        return tensor, label


# ============================================================
# MODEL
# ============================================================

def build_model(
    root: Path,
    use_pretrained: bool
) -> nn.Module:
    """Build standard EfficientNet-B0 and optionally load local ImageNet weights."""

    # Build the architecture without triggering any network download.
    model = torchvision.models.efficientnet_b0(weights=None)
    init_name = "random initialization"
    weight_path: Optional[Path] = None

    if use_pretrained:
        weight_path = root / LOCAL_WEIGHTS_RELATIVE

        if not weight_path.is_file():
            raise FileNotFoundError(
                "Local EfficientNet-B0 pretrained weights were not found.\n"
                f"Expected: {weight_path}\n\n"
                "Download the official TorchVision ImageNet-1K EfficientNet-B0 "
                "checkpoint and place it at that exact path.\n"
                "Direct URL: https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
            )

        print("\nLoading LOCAL ImageNet-1K EfficientNet-B0 weights...")
        print(f"Weights: {weight_path}")

        state_dict = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=True,
        )

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "The local EfficientNet-B0 checkpoint does not match the "
                "TorchVision EfficientNet-B0 architecture.\n"
                f"Checkpoint: {weight_path}\n"
                f"Original error: {exc}"
            ) from exc

        init_name = (
            "Local TorchVision EfficientNet-B0 ImageNet-1K "
            "IMAGENET1K_V1 weights"
        )

    # Replace the original 1000-class ImageNet classifier with the 5-class DR head.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    print("\n" + "=" * 80)
    print("MODEL")
    print("=" * 80)
    print(f"EfficientNet-B0 initialization: {init_name}")
    if weight_path is not None:
        print(f"Local weights file       : {weight_path}")
    print(f"Classifier               : Linear({in_features}, {NUM_CLASSES})")
    print("Enhancement modules      : NONE")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"Total parameters         : {total:,}")
    print(f"Trainable parameters     : {trainable:,}")

    return model


def last_conv_layer(model: nn.Module) -> nn.Module:
    # torchvision EfficientNet-B0: features ends with Conv2dNormActivation.
    # The final Conv2d in that container is appropriate for Grad-CAM.
    container = model.features[-1]
    convs = [m for m in container.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")
    return convs[-1]


# ============================================================
# METRICS — NO SCIKIT-LEARN
# ============================================================

def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> pd.DataFrame:
    rows = []
    n_classes = cm.shape[0]
    total = cm.sum()
    for c in range(n_classes):
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)
        tn = float(total - tp - fn - fp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "support": int(cm[c, :].sum()),
            }
        )
    return pd.DataFrame(rows)


def roc_auc_ovr(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> Tuple[np.ndarray, float]:
    aucs = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        pos = probs[:, c]
        target = (y_true == c).astype(np.int32)
        n_pos = int(target.sum())
        n_neg = int(len(target) - n_pos)
        if n_pos == 0 or n_neg == 0:
            aucs[c] = float("nan")
            continue
        order = np.argsort(pos, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        sorted_scores = pos[order]
        i = 0
        rank = 1.0
        while i < len(sorted_scores):
            j = i + 1
            while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            avg_rank = (rank + rank + (j - i) - 1.0) / 2.0
            ranks[order[i:j]] = avg_rank
            rank += j - i
            i = j
        sum_ranks_pos = ranks[target == 1].sum()
        aucs[c] = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    macro = float(np.nanmean(aucs))
    return aucs, macro


def pr_auc_ovr(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full(n_classes, np.nan, dtype=np.float64)
    for c in range(n_classes):
        target = (y_true == c).astype(np.int32)
        if target.sum() == 0:
            continue
        score = probs[:, c]
        order = np.argsort(-score, kind="mergesort")
        target = target[order]
        tp = np.cumsum(target)
        fp = np.cumsum(1 - target)
        recall = tp / max(1, int(target.sum()))
        precision = tp / np.maximum(tp + fp, 1)
        # Add (recall=0, precision=1) conventionally, then trapezoid.
        r = np.concatenate(([0.0], recall.astype(np.float64)))
        p = np.concatenate(([1.0], precision.astype(np.float64)))
        out[c] = float(np.trapezoid(p, r) if hasattr(np, "trapezoid") else np.trapz(p, r))
    return out


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm = confusion_matrix_manual(y_true, y_pred, n_classes).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0
    w = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    hist_true = cm.sum(axis=1)
    hist_pred = cm.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / n
    observed_disagreement = (w * cm / n).sum()
    expected_disagreement = (w * expected / n).sum()
    if expected_disagreement == 0:
        return 1.0
    return float(1.0 - observed_disagreement / expected_disagreement)


def accuracy_ci95(accuracy: float, n: int) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    se = math.sqrt(max(0.0, accuracy * (1.0 - accuracy) / n))
    return max(0.0, accuracy - 1.96 * se), min(1.0, accuracy + 1.96 * se)


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += bs

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    prob_list: List[np.ndarray] = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        total += bs
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        prob_list.append(probs.cpu().numpy())

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    probs = np.concatenate(prob_list, axis=0) if prob_list else np.zeros((0, NUM_CLASSES), dtype=np.float32)

    cm = confusion_matrix_manual(yt, yp, NUM_CLASSES)
    pc = per_class_metrics(cm)
    accuracy = float(np.mean(yt == yp)) if len(yt) else 0.0
    macro_precision = float(pc["precision"].mean())
    macro_recall = float(pc["recall"].mean())
    macro_f1 = float(pc["f1"].mean())
    weighted_f1 = float(np.average(pc["f1"], weights=pc["support"]))
    qwk = quadratic_weighted_kappa(yt, yp, NUM_CLASSES)
    roc_aucs, macro_auc = roc_auc_ovr(yt, probs, NUM_CLASSES)
    pr_aucs = pr_auc_ovr(yt, probs, NUM_CLASSES)
    ci_lo, ci_hi = accuracy_ci95(accuracy, len(yt))

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
        "macro_roc_auc": macro_auc,
        "accuracy_ci95_low": ci_lo,
        "accuracy_ci95_high": ci_hi,
        "confusion_matrix": cm,
        "per_class": pc,
        "roc_auc_per_class": roc_aucs,
        "pr_auc_per_class": pr_aucs,
        "y_true": yt,
        "y_pred": yp,
        "probabilities": probs,
    }


# ============================================================
# PLOTS / EXPORTS
# ============================================================

def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("EfficientNet-B0 Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(NUM_CLASSES)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(range(NUM_CLASSES))
    ax.set_yticklabels(range(NUM_CLASSES))
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_accuracy"], label="Train Accuracy")
    ax.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["val_qwk"], label="Validation QWK")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("QWK")
    ax.set_title("Validation QWK")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "qwk_curve.png", dpi=200)
    plt.close(fig)


def save_roc_pr_summary(
    test_result: Dict[str, object],
    out_dir: Path,
) -> None:
    roc = np.asarray(test_result["roc_auc_per_class"], dtype=np.float64)
    pr = np.asarray(test_result["pr_auc_per_class"], dtype=np.float64)
    pd.DataFrame({
        "class_id": list(range(NUM_CLASSES)),
        "class_name": CLASS_NAMES,
        "roc_auc": roc,
        "pr_auc": pr,
    }).to_csv(out_dir / "auc_by_class.csv", index=False)


def save_test_outputs(
    result: Dict[str, object],
    test_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    cm = np.asarray(result["confusion_matrix"])
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_confusion_matrix(cm, out_dir / "confusion_matrix.png")

    pc: pd.DataFrame = result["per_class"]
    pc.to_csv(out_dir / "classification_report.csv", index=False)
    save_roc_pr_summary(result, out_dir)

    y_true = np.asarray(result["y_true"])
    y_pred = np.asarray(result["y_pred"])
    probs = np.asarray(result["probabilities"])

    pred = test_df.reset_index(drop=True).copy()
    pred["true_label"] = y_true
    pred["true_class"] = [CLASS_NAMES[int(x)] for x in y_true]
    pred["pred_label"] = y_pred
    pred["pred_class"] = [CLASS_NAMES[int(x)] for x in y_pred]
    pred["correct"] = (y_true == y_pred)
    for c in range(NUM_CLASSES):
        pred[f"prob_{c}"] = probs[:, c]
    pred.to_csv(out_dir / "test_predictions.csv", index=False)

    metrics = {
        "loss": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "macro_precision": float(result["macro_precision"]),
        "macro_recall": float(result["macro_recall"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_f1": float(result["weighted_f1"]),
        "qwk": float(result["qwk"]),
        "macro_roc_auc": float(result["macro_roc_auc"]),
        "accuracy_ci95_low": float(result["accuracy_ci95_low"]),
        "accuracy_ci95_high": float(result["accuracy_ci95_high"]),
        "support": int(len(y_true)),
        "roc_auc_per_class": [float(x) for x in result["roc_auc_per_class"]],
        "pr_auc_per_class": [float(x) for x in result["pr_auc_per_class"]],
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ============================================================
# GRAD-CAM (OPTIONAL, PAPER-ALIGNED EXPLAINABILITY)
# ============================================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._forward_hook)
        self._bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        acts = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        cam = cam[0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def tensor_to_display_image(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    x = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    x = np.clip(x * std + mean, 0.0, 1.0)
    return x


def save_gradcam_samples(
    model: nn.Module,
    test_df: pd.DataFrame,
    mean: Sequence[float],
    std: Sequence[float],
    preprocess_cfg: Dict[str, float],
    out_dir: Path,
    max_samples: int,
    seed: int,
) -> None:
    if max_samples <= 0 or len(test_df) == 0:
        return

    n = min(max_samples, len(test_df))
    sample_df = test_df.sample(n=n, random_state=seed).reset_index(drop=True)
    dataset = FundusDataset(
        sample_df,
        mean,
        std,
        train=False,
        seed=seed,
        preprocess_cfg=preprocess_cfg,
        aug_cfg={
            "aug_prob": 0,
            "translation": 0,
            "brightness": 0,
            "contrast": 0,
            "gamma_low": 1,
            "gamma_high": 1,
            "elastic_alpha": 0,
            "elastic_sigma": 1,
            "cutout_ratio": 0,
        },
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    layer = last_conv_layer(model)
    cam_engine = GradCAM(model, layer)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, (x, y) in enumerate(loader):
            x = x.to(DEVICE)
            with torch.no_grad():
                logits = model(x)
                pred = int(logits.argmax(dim=1).item())

            # Need a fresh forward/backward for Grad-CAM.
            cam, cam_class = cam_engine(x, pred)
            display = tensor_to_display_image(x[0], mean, std)
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
            overlay = np.clip(0.55 * display + 0.45 * heat, 0.0, 1.0)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(display)
            axes[0].set_title(f"Original\nTrue: {int(y.item())}")
            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title(f"Grad-CAM\nPred: {cam_class}")
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{i:02d}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    finally:
        cam_engine.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    root = project_root()
    out_root = output_root(root, args.dataset, args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    split_dir = out_root / "splits"
    results_dir = out_root / "results"
    ckpt_dir = out_root / "checkpoints"
    gradcam_dir = results_dir / "gradcam"
    for d in (split_dir, results_dir, ckpt_dir, gradcam_dir):
        d.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("=" * 90)
    print("EFFICIENTNET-B0 — PAPER-GROUNDED REPLICATION")
    print("=" * 90)
    print(f"Project root : {root}")
    print(f"Dataset      : {args.dataset}")
    print(f"Device       : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print("Model        : EfficientNet-B0 ONLY")
    print("MSAG         : OFF")
    print("CBAM         : OFF")
    print("DropBlock    : OFF")
    print("Transformer  : OFF")
    print("Split        : 80/10/10 FIRST")
    print("Oversampling : TRAIN ONLY")
    print("Augmentation : TRAIN ONLY")

    if args.val_ratio != 0.10 or args.test_ratio != 0.10:
        print("WARNING: Split ratios differ from requested 80/10/10 protocol.")

    # --------------------------------------------------------
    # Manifest + split
    # --------------------------------------------------------
    full_df = build_manifest(root, args.dataset)
    verify_manifest(full_df)

    train_df, val_df, test_df = stratified_split(
        full_df,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    print("\n" + "=" * 80)
    print("STRATIFIED SPLIT RESULTS")
    print("=" * 80)
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        counts = df["diagnosis"].value_counts().sort_index()
        print(f"{name:12s}: {len(df):6d}")
        for cid in range(NUM_CLASSES):
            print(f"  {cid} {CLASS_NAMES[cid]:18s}: {int(counts.get(cid, 0)):6d}")

    train_balanced_df = oversample_train_to_majority(train_df, args.seed)

    full_df.to_csv(split_dir / "full_manifest.csv", index=False)
    train_df.to_csv(split_dir / "train_original.csv", index=False)
    val_df.to_csv(split_dir / "validation.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    train_balanced_df.to_csv(split_dir / "train_oversampled.csv", index=False)

    # --------------------------------------------------------
    # Normalization statistics from ORIGINAL train split only
    # --------------------------------------------------------
    mean, std = compute_train_mean_std(
        train_df,
        args.clahe_clip,
        args.clahe_tile,
        args.median_kernel,
    )

    with open(results_dir / "normalization.json", "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

    # --------------------------------------------------------
    # Datasets/loaders
    # --------------------------------------------------------
    preprocess_cfg = {
        "clahe_clip": args.clahe_clip,
        "clahe_tile": args.clahe_tile,
        "median_kernel": args.median_kernel,
    }
    aug_cfg = {
        "aug_prob": args.aug_prob,
        "translation": args.translation,
        "brightness": args.brightness,
        "contrast": args.contrast,
        "gamma_low": args.gamma_low,
        "gamma_high": args.gamma_high,
        "elastic_alpha": args.elastic_alpha,
        "elastic_sigma": args.elastic_sigma,
        "cutout_ratio": args.cutout_ratio,
    }

    train_set = FundusDataset(
        train_balanced_df,
        mean,
        std,
        train=True,
        seed=args.seed,
        preprocess_cfg=preprocess_cfg,
        aug_cfg=aug_cfg,
    )
    val_set = FundusDataset(
        val_df,
        mean,
        std,
        train=False,
        seed=args.seed,
        preprocess_cfg=preprocess_cfg,
        aug_cfg=aug_cfg,
    )
    test_set = FundusDataset(
        test_df,
        mean,
        std,
        train=False,
        seed=args.seed,
        preprocess_cfg=preprocess_cfg,
        aug_cfg=aug_cfg,
    )

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=(args.workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=(args.workers > 0),
    )

    # --------------------------------------------------------
    # Model / loss / optimizer / scheduler
    # --------------------------------------------------------
    model = build_model(
        root=root,
        use_pretrained=(not args.no_pretrained),
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=DEFAULT_LR_FACTOR,
        patience=DEFAULT_PATIENCE,
    )

    if args.smoke_test:
        epochs = min(args.epochs, 1)
    else:
        epochs = args.epochs

    config = {
        "paper": {
            "title": "AI-driven detection and classification of diabetic retinopathy stages using EfficientNetB0",
            "authors": ["Deepak Dharrao", "Madhuri Dharrao", "Shreyas Patil", "Sangeeth Salvin", "Prashant Ahire", "Yashwant Dongre"],
            "journal": "Discover Applied Sciences",
            "year": 2025,
            "doi": "10.1007/s42452-025-07998-9",
        },
        "experiment": {
            "dataset": args.dataset,
            "dataset_source_root": str(dataset_root(root, args.dataset)),
            "seed": args.seed,
            "device": str(DEVICE),
            "model": "EfficientNet-B0",
            "msag": False,
            "enhancement_modules": [],
            "pretrained": not args.no_pretrained,
            "pretrained_source": (
                "Local official TorchVision EfficientNet-B0 ImageNet-1K "
                "IMAGENET1K_V1 checkpoint"
                if not args.no_pretrained
                else "none"
            ),
            "pretrained_weights_path": (
                str(root / LOCAL_WEIGHTS_RELATIVE)
                if not args.no_pretrained
                else None
            ),
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "num_classes": NUM_CLASSES,
            "split": "80/10/10 before oversampling",
            "oversampling": "random oversampling with replacement in TRAIN ONLY to majority class",
            "validation_augmentation": False,
            "test_augmentation": False,
            "normalization": "pixel /255 followed by training-split global mean/std standardization",
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss (categorical cross-entropy equivalent)",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": DEFAULT_LR_FACTOR,
                "patience_epochs": DEFAULT_PATIENCE,
            },
            "epochs": epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
        },
        "paper_specified_preprocessing": {
            "green_channel": True,
            "clahe": True,
            "median_filter": True,
            "resize": "224x224",
            "normalization": "[0,1] and/or zero-mean/unit-variance described",
        },
        "implementation_defaults_not_specified_numerically_by_paper": {
            "clahe_clip_limit": args.clahe_clip,
            "clahe_tile_grid": [args.clahe_tile, args.clahe_tile],
            "median_kernel": args.median_kernel,
            "augmentation_probability": args.aug_prob,
            "translation_fraction": args.translation,
            "brightness_fraction": args.brightness,
            "contrast_fraction": args.contrast,
            "gamma_low": args.gamma_low,
            "gamma_high": args.gamma_high,
            "elastic_alpha": args.elastic_alpha,
            "elastic_sigma": args.elastic_sigma,
            "cutout_ratio": args.cutout_ratio,
            "initial_learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": epochs,
        },
        "paper_augmentation_techniques": [
            "random rotation +/-15 degrees",
            "horizontal flip",
            "vertical flip",
            "translation",
            "brightness variation",
            "contrast variation",
            "gamma variation",
            "elastic deformation",
            "cutout",
        ],
    }
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    history_rows = []

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Epochs         : {epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Initial LR     : {args.lr}")
    print(f"Optimizer      : Adam")
    print(f"LR scheduler   : ReduceLROnPlateau(factor={DEFAULT_LR_FACTOR}, patience={DEFAULT_PATIENCE})")
    print("Train aug      : ON")
    print("Val/Test aug   : OFF")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        val_result = evaluate(model, val_loader, criterion)
        scheduler.step(float(val_result["loss"]))

        lr_now = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_result["loss"]),
            "val_accuracy": float(val_result["accuracy"]),
            "val_macro_f1": float(val_result["macro_f1"]),
            "val_qwk": float(val_result["qwk"]),
            "val_macro_roc_auc": float(val_result["macro_roc_auc"]),
            "learning_rate": lr_now,
            "epoch_seconds": elapsed,
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={float(val_result['loss']):.4f} "
            f"acc={float(val_result['accuracy']):.4f} "
            f"F1={float(val_result['macro_f1']):.4f} "
            f"QWK={float(val_result['qwk']):.4f} "
            f"AUC={float(val_result['macro_roc_auc']):.4f} | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        }
        torch.save(state, ckpt_dir / "last_model.pth")

        if float(val_result["loss"]) < best_val_loss:
            best_val_loss = float(val_result["loss"])
            best_epoch = epoch
            state["best_val_loss"] = best_val_loss
            torch.save(state, ckpt_dir / "best_model.pth")
            print("  -> saved new best_model.pth")

    history = pd.DataFrame(history_rows)
    history.to_csv(results_dir / "training_history.csv", index=False)
    save_training_curves(history, results_dir)

    # --------------------------------------------------------
    # Reload best and final test evaluation
    # --------------------------------------------------------
    best_state = torch.load(ckpt_dir / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(best_state["model_state_dict"])

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION — BEST VALIDATION CHECKPOINT")
    print("=" * 80)
    test_result = evaluate(model, test_loader, criterion)

    save_test_outputs(test_result, test_df, results_dir)

    print(f"Test accuracy     : {float(test_result['accuracy']):.4f}")
    print(f"Test macro F1     : {float(test_result['macro_f1']):.4f}")
    print(f"Test weighted F1  : {float(test_result['weighted_f1']):.4f}")
    print(f"Test QWK          : {float(test_result['qwk']):.4f}")
    print(f"Test macro ROC-AUC: {float(test_result['macro_roc_auc']):.4f}")
    print(
        f"Accuracy 95% CI  : "
        f"[{float(test_result['accuracy_ci95_low']):.4f}, "
        f"{float(test_result['accuracy_ci95_high']):.4f}]"
    )

    pc: pd.DataFrame = test_result["per_class"]
    print("\nPer-class metrics:")
    for _, row in pc.iterrows():
        print(
            f"  {int(row['class_id'])} {row['class_name']:18s} "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} "
            f"Support={int(row['support'])}"
        )

    print("\nConfusion matrix:")
    print(np.asarray(test_result["confusion_matrix"]))

    # Optional Grad-CAM explainability.
    if args.gradcam_samples > 0:
        print("\nGenerating Grad-CAM examples...")
        save_gradcam_samples(
            model,
            test_df,
            mean,
            std,
            preprocess_cfg,
            gradcam_dir,
            args.gradcam_samples,
            args.seed,
        )

    summary = {
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics": {
            "accuracy": float(test_result["accuracy"]),
            "macro_f1": float(test_result["macro_f1"]),
            "weighted_f1": float(test_result["weighted_f1"]),
            "qwk": float(test_result["qwk"]),
            "macro_roc_auc": float(test_result["macro_roc_auc"]),
            "accuracy_ci95": [
                float(test_result["accuracy_ci95_low"]),
                float(test_result["accuracy_ci95_high"]),
            ],
        },
        "counts": {
            "full": int(len(full_df)),
            "train_original": int(len(train_df)),
            "train_oversampled": int(len(train_balanced_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "elapsed_minutes": (time.time() - start_time) / 60.0,
    }
    with open(results_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"Output root : {out_root}")
    print(f"Best epoch  : {best_epoch}")
    print(f"Elapsed min : {summary['elapsed_minutes']:.2f}")
    print(f"Best model  : {ckpt_dir / 'best_model.pth'}")
    print(f"Metrics     : {results_dir / 'test_metrics.json'}")
    print(f"History     : {results_dir / 'training_history.csv'}")
    print(f"Grad-CAM    : {gradcam_dir}")


if __name__ == "__main__":
    main()
