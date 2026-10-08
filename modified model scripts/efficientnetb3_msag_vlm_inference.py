"""
EfficientNet-B3 + MSAG + BiomedCLIP VLM Ensemble Inference
============================================================

Inference-only script that combines:
  1. Trained EfficientNet-B3 + MSAG model (from checkpoint)
  2. Frozen BiomedCLIP (zero-shot classification via text prompts)

The two models' softmax probabilities are combined via weighted average:
    final_prob = w_msag * msag_prob + w_vlm * vlm_prob

No additional training is required — BiomedCLIP is used frozen at test time.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ============================================================
# PATHS / CONSTANTS
# ============================================================

PROJECT_ROOT = Path(r"A:\DR_classification")
DATASET_ROOT = PROJECT_ROOT / "data" / "organized" / "pooled_aptos_ddr"

RUN_ROOT = PROJECT_ROOT / "runs_EfficientNetB3_MSAG_Refined_Stable"
MSAG_CHECKPOINT = RUN_ROOT / "outputs" / "best_model_qwk.pth"

VLM_OUTPUT_ROOT = PROJECT_ROOT / "runs_VLM_Ensemble_Inference"
OUTPUT_DIR = VLM_OUTPUT_ROOT / "outputs"

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

DR_TEXT_PROMPTS = [
    "a retinal fundus photograph showing no diabetic retinopathy",
    "a retinal fundus photograph showing mild nonproliferative diabetic retinopathy with microaneurysms",
    "a retinal fundus photograph showing moderate nonproliferative diabetic retinopathy with hemorrhages and hard exudates",
    "a retinal fundus photograph showing severe nonproliferative diabetic retinopathy with venous beading and intraretinal microvascular abnormalities",
    "a retinal fundus photograph showing proliferative diabetic retinopathy with neovascularization",
]

# Ensemble weight — MSAG model is the primary (trained specifically for DR).
# BiomedCLIP provides complementary zero-shot signal.
DEFAULT_MSAG_WEIGHT = 0.7
DEFAULT_VLM_WEIGHT = 0.3

ATTENTION_REDUCTION = 8
DROPOUT_ORIGINAL = 0.40
DROPOUT_SEVERITY = 0.20
DROP_PATH_RATE = 0.10


# ============================================================
# HELPERS
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


def device_info() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA  : {torch.version.cuda}")
    return device


# ============================================================
# DATASET
# ============================================================

def collect_dataset_records(root: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        split_root = root / split
        if not split_root.exists():
            continue
        for class_idx, class_dir in enumerate(CLASS_DIRS):
            class_root = split_root / class_dir
            if not class_root.exists():
                continue
            for path in sorted(class_root.rglob("*")):
                if not is_image(path):
                    continue
                rows.append({
                    "split": split,
                    "class_idx": class_idx,
                    "class_name": CLASS_NAMES[class_idx],
                    "path": str(path),
                    "image_id": path.stem,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root}")
    return df


class PooledFundusDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, split: str, transform) -> None:
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
        return x, int(row["class_idx"]), str(row["path"]), str(row["image_id"])


def pooled_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)
    paths = [str(item[2]) for item in batch]
    image_ids = [str(item[3]) for item in batch]
    return images, labels, paths, image_ids


# ============================================================
# MSAG MODEL (same architecture as training script)
# ============================================================

class MultiScaleSpatialAttentionGate(nn.Module):
    def __init__(self, kernel_sizes: Sequence[int] = (3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm2d(len(kernel_sizes))
        self.fuse = nn.Conv2d(len(kernel_sizes), 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_map, max_map], dim=1)
        branch_outputs = [branch(pooled) for branch in self.branches]
        multi_scale = torch.cat(branch_outputs, dim=1)
        multi_scale = self.bn(multi_scale)
        gate = torch.sigmoid(self.fuse(multi_scale))
        return x * gate


class RefinedDRModel(nn.Module):
    def __init__(self, drop_path_rate: float = DROP_PATH_RATE):
        super().__init__()
        import timm

        try:
            self.backbone = timm.create_model(
                "efficientnet_b3", pretrained=False, num_classes=0,
                drop_path_rate=drop_path_rate,
            )
        except TypeError:
            self.backbone = timm.create_model(
                "efficientnet_b3", pretrained=False, num_classes=0,
            )

        self.feature_dim = int(self.backbone.num_features)
        self.msag = MultiScaleSpatialAttentionGate()
        self.msag_alpha_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        hidden_att = self.feature_dim // ATTENTION_REDUCTION
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, hidden_att),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_att, self.feature_dim),
            nn.Sigmoid(),
        )

        self.feature_norm = nn.BatchNorm1d(self.feature_dim)
        self.dropout = nn.Dropout(DROPOUT_ORIGINAL)

        self.severity_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_SEVERITY),
            nn.Linear(self.feature_dim // 2, NUM_CLASSES),
        )

        self.lesion_detector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(self.feature_dim // 4, 5),
        )

        self.region_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(self.feature_dim // 4, 5),
        )

        self.last_feature_map: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone.forward_features(x)
        self.last_feature_map = features

        msag_features = self.msag(features)
        msag_alpha = 0.25 * torch.tanh(self.msag_alpha_raw)
        fused = features + msag_alpha * (msag_features - features)

        pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        pooled_4d = pooled.unsqueeze(-1).unsqueeze(-1)
        attention_weights = self.attention(pooled_4d)
        attended = pooled * attention_weights
        normalized = self.feature_norm(attended)
        normalized = self.dropout(normalized)
        severity_logits = self.severity_classifier(normalized)

        if return_features:
            lesion_logits = self.lesion_detector(normalized)
            region_logits = self.region_predictor(normalized)
            return {
                "severity": severity_logits,
                "lesions": lesion_logits,
                "regions": region_logits,
                "features": normalized,
                "feature_map": fused,
            }
        return severity_logits


def normalize_checkpoint_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
        )
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a recognizable state_dict.")

    normalized = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        for prefix in ("module.", "model.", "_orig_mod."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        normalized[clean_key] = value
    return normalized


def load_msag_model(checkpoint_path: Path, device: torch.device) -> RefinedDRModel:
    print(f"\n[MSAG Model] Loading checkpoint: {checkpoint_path}")
    model = RefinedDRModel()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = normalize_checkpoint_state_dict(checkpoint)

    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        print(f"  Unexpected keys (ignored): {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"  Missing keys: {len(result.missing_keys)}")
        for k in result.missing_keys[:5]:
            print(f"    - {k}")

    model.to(device)
    model.eval()
    print("  MSAG model loaded and set to eval mode.")
    return model


# ============================================================
# BIOMEDCLIP VLM
# ============================================================

def load_biomedclip(device: torch.device):
    import open_clip

    print(f"\n[BiomedCLIP] Loading model: {BIOMEDCLIP_MODEL_NAME}")
    model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_NAME)

    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print("  BiomedCLIP loaded (frozen) and set to eval mode.")
    return model, preprocess, tokenizer


def compute_vlm_text_features(
    vlm_model, tokenizer, prompts: List[str], device: torch.device
) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = vlm_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features


def vlm_zero_shot_batch(
    vlm_model, vlm_preprocess, text_features: torch.Tensor,
    image_paths: List[str], device: torch.device,
) -> torch.Tensor:
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(vlm_preprocess(img))
    image_batch = torch.stack(images, dim=0).to(device)

    with torch.no_grad():
        image_features = vlm_model.encode_image(image_batch)
        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ text_features.t()
        probs = F.softmax(logits * 100.0, dim=-1)

    return probs


# ============================================================
# METRICS (no sklearn dependency)
# ============================================================

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    n = NUM_CLASSES
    cm = compute_confusion_matrix(y_true, y_pred, n)
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    per_class = {}
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[c, :].sum()),
        }

    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    weighted_f1 = sum(
        v["f1"] * v["support"] for v in per_class.values()
    ) / max(1, sum(v["support"] for v in per_class.values()))

    # Quadratic Weighted Kappa
    total = cm.sum()
    p_observed = np.sum(np.diag(cm)) / total
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    p_expected_matrix = np.outer(row_sum, col_sum) / total
    weight_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weight_matrix[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    numerator = np.sum(weight_matrix * cm) / total
    denominator = np.sum(weight_matrix * p_expected_matrix) / total
    qwk = 1.0 - numerator / denominator if denominator > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "qwk": round(qwk, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_weight_sensitivity(
    all_true: np.ndarray, msag_probs: np.ndarray, vlm_probs: np.ndarray,
    save_path: Path,
) -> Dict:
    weights = np.arange(0.0, 1.05, 0.05)
    results = []
    for w_msag in weights:
        w_vlm = 1.0 - w_msag
        combined = w_msag * msag_probs + w_vlm * vlm_probs
        preds = np.argmax(combined, axis=1)
        acc = np.mean(preds == all_true)
        results.append({"w_msag": round(w_msag, 2), "w_vlm": round(w_vlm, 2), "accuracy": round(acc, 4)})

    df = pd.DataFrame(results)
    best = df.loc[df["accuracy"].idxmax()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["w_msag"], df["accuracy"], "b-o", markersize=4)
    ax.axvline(best["w_msag"], color="r", linestyle="--", alpha=0.7,
               label=f'Best: w_msag={best["w_msag"]:.2f}, acc={best["accuracy"]:.4f}')
    ax.set_xlabel("MSAG Weight (w_msag)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Ensemble Weight Sensitivity Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {"sweep": results, "best_w_msag": float(best["w_msag"]), "best_accuracy": float(best["accuracy"])}


SEVERITY_DESCRIPTIONS = {
    0: "no signs of diabetic retinopathy. The retina appears healthy with no visible microaneurysms, hemorrhages, or exudates.",
    1: "mild nonproliferative diabetic retinopathy (NPDR). Early signs include a few microaneurysms in the retinal vasculature.",
    2: "moderate nonproliferative diabetic retinopathy (NPDR). Findings include scattered hemorrhages, hard exudates, and cotton-wool spots.",
    3: "severe nonproliferative diabetic retinopathy (NPDR). Extensive hemorrhages, venous beading, and intraretinal microvascular abnormalities (IRMA) are observed.",
    4: "proliferative diabetic retinopathy (PDR). Neovascularization is present, indicating advanced disease with risk of vitreous hemorrhage and retinal detachment.",
}


def generate_inference_report(
    pred_class: int,
    probs: np.ndarray,
    model_name: str,
) -> str:
    pred_name = CLASS_NAMES[pred_class]
    conf = probs[pred_class] * 100.0
    description = SEVERITY_DESCRIPTIONS[pred_class]

    lines = [
        f"[{model_name} Inference Report]",
        f"This retinal fundus image shows {description}",
        f"",
        f"Predicted severity: {pred_name} ({conf:.1f}% confidence)",
        f"",
        f"Severity probability distribution:",
    ]
    for c in range(NUM_CLASSES):
        bar_len = int(probs[c] * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        marker = " <--" if c == pred_class else ""
        lines.append(f"  {CLASS_NAMES[c]:20s} {probs[c]*100:5.1f}%  [{bar}]{marker}")

    return "\n".join(lines)


def plot_qualitative(
    all_true: np.ndarray,
    all_paths: List[str],
    msag_preds: np.ndarray,
    vlm_preds: np.ndarray,
    ensemble_preds: np.ndarray,
    msag_probs: np.ndarray,
    vlm_probs: np.ndarray,
    ensemble_probs: np.ndarray,
    save_dir: Path,
    samples_per_class: int = 3,
    seed: int = 42,
) -> None:
    rng = np.random.RandomState(seed)

    selected_indices = []
    for c in range(NUM_CLASSES):
        class_indices = np.where(all_true == c)[0]
        n_pick = min(samples_per_class, len(class_indices))
        picked = rng.choice(class_indices, size=n_pick, replace=False)
        selected_indices.extend(sorted(picked))

    n_images = len(selected_indices)

    # ---- Text report file ----
    report_path = save_dir / "vlm_inference_reports.txt"
    report_lines = [
        "=" * 80,
        "  VLM ENSEMBLE INFERENCE REPORTS",
        "  EfficientNet-B3 + MSAG  |  BiomedCLIP (zero-shot)  |  Ensemble",
        "=" * 80,
        "",
    ]

    # ---- Visual figure: image + inference text for each sample ----
    fig, axes = plt.subplots(n_images, 2, figsize=(18, 4.0 * n_images),
                             gridspec_kw={"width_ratios": [1, 1.8]})
    if n_images == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "VLM Ensemble Inference — Qualitative Analysis",
        fontsize=18, fontweight="bold", y=0.998,
    )

    for row, idx in enumerate(selected_indices):
        img = Image.open(all_paths[idx]).convert("RGB")
        true_label = int(all_true[idx])
        true_name = CLASS_NAMES[true_label]
        img_name = Path(all_paths[idx]).name

        ens_pred = int(ensemble_preds[idx])
        ens_conf = ensemble_probs[idx][ens_pred] * 100.0
        ens_correct = ens_pred == true_label

        # -- Left column: image --
        ax_img = axes[row, 0]
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        border_color = "green" if ens_correct else "red"
        for spine in ax_img.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        ax_img.set_title(f"True: {true_name}", fontsize=12, fontweight="bold", pad=6)

        # -- Right column: inference text --
        ax_txt = axes[row, 1]
        ax_txt.axis("off")

        ens_description = SEVERITY_DESCRIPTIONS[ens_pred]
        msag_pred = int(msag_preds[idx])
        vlm_pred = int(vlm_preds[idx])

        text_block = (
            f"This retinal fundus image shows {ens_description}\n\n"
            f"Ensemble prediction: {CLASS_NAMES[ens_pred]} ({ens_conf:.1f}% confidence)\n\n"
            f"Severity probability distribution:\n"
        )
        for c in range(NUM_CLASSES):
            msag_p = msag_probs[idx][c] * 100
            vlm_p = vlm_probs[idx][c] * 100
            ens_p = ensemble_probs[idx][c] * 100
            marker = "  <--" if c == ens_pred else ""
            text_block += f"  {CLASS_NAMES[c]:20s}  MSAG: {msag_p:5.1f}%  VLM: {vlm_p:5.1f}%  Ensemble: {ens_p:5.1f}%{marker}\n"

        text_block += (
            f"\nIndividual model predictions:\n"
            f"  MSAG      : {CLASS_NAMES[msag_pred]} ({msag_probs[idx][msag_pred]*100:.1f}%)\n"
            f"  BiomedCLIP: {CLASS_NAMES[vlm_pred]} ({vlm_probs[idx][vlm_pred]*100:.1f}%)\n"
            f"  Ensemble  : {CLASS_NAMES[ens_pred]} ({ens_conf:.1f}%)"
        )

        ax_txt.text(
            0.02, 0.95, text_block,
            transform=ax_txt.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        # -- Report text file entry --
        report_lines.append("-" * 80)
        report_lines.append(f"Image: {img_name}")
        report_lines.append(f"Path : {all_paths[idx]}")
        report_lines.append(f"True label: {true_name} (class {true_label})")
        report_lines.append("")
        report_lines.append(generate_inference_report(ens_pred, ensemble_probs[idx], "Ensemble"))
        report_lines.append("")
        report_lines.append(generate_inference_report(msag_pred, msag_probs[idx], "MSAG"))
        report_lines.append("")
        report_lines.append(generate_inference_report(vlm_pred, vlm_probs[idx], "BiomedCLIP VLM"))
        report_lines.append("")

    report_lines.append("=" * 80)

    # Save text report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved inference reports: {report_path}")

    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = save_dir / "qualitative_analysis_vlm_ensemble.png"
    fig.savefig(str(fig_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved qualitative figure: {fig_path}")


def save_sample_outputs(
    all_true: np.ndarray,
    all_paths: List[str],
    msag_preds: np.ndarray,
    vlm_preds: np.ndarray,
    ensemble_preds: np.ndarray,
    msag_probs: np.ndarray,
    vlm_probs: np.ndarray,
    ensemble_probs: np.ndarray,
    save_dir: Path,
    n_samples: int = 10,
    seed: int = 42,
) -> None:
    sample_dir = save_dir / "sample_outputs"
    safe_mkdir(sample_dir)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(all_true), size=min(n_samples, len(all_true)), replace=False)
    indices = sorted(indices)

    for rank, idx in enumerate(indices, 1):
        img = Image.open(all_paths[idx]).convert("RGB")
        true_label = int(all_true[idx])
        true_name = CLASS_NAMES[true_label]
        img_name = Path(all_paths[idx]).stem

        ens_pred = int(ensemble_preds[idx])
        ens_conf = ensemble_probs[idx][ens_pred] * 100.0
        msag_pred = int(msag_preds[idx])
        vlm_pred = int(vlm_preds[idx])

        # Build inference text
        description = SEVERITY_DESCRIPTIONS[ens_pred]
        lines = [
            f"INFERENCE REPORT",
            f"",
            f"Image: {Path(all_paths[idx]).name}",
            f"Ground Truth: {true_name}",
            f"",
            f"This retinal fundus image shows {description}",
            f"",
            f"Predicted Severity: {CLASS_NAMES[ens_pred]}",
            f"Confidence: {ens_conf:.1f}%",
            f"",
            f"--- Severity Probability Distribution ---",
            f"",
        ]
        for c in range(NUM_CLASSES):
            m_p = msag_probs[idx][c] * 100
            v_p = vlm_probs[idx][c] * 100
            e_p = ensemble_probs[idx][c] * 100
            marker = "  <--" if c == ens_pred else ""
            lines.append(
                f"  {CLASS_NAMES[c]:20s}"
                f"  MSAG: {m_p:5.1f}%"
                f"  VLM: {v_p:5.1f}%"
                f"  Ens: {e_p:5.1f}%{marker}"
            )
        lines += [
            f"",
            f"--- Individual Model Predictions ---",
            f"",
            f"  MSAG Model  : {CLASS_NAMES[msag_pred]:20s} ({msag_probs[idx][msag_pred]*100:.1f}%)",
            f"  BiomedCLIP  : {CLASS_NAMES[vlm_pred]:20s} ({vlm_probs[idx][vlm_pred]*100:.1f}%)",
            f"  Ensemble    : {CLASS_NAMES[ens_pred]:20s} ({ens_conf:.1f}%)",
        ]
        report_text = "\n".join(lines)

        # --- Create figure: image left, report right ---
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.05)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        correct = ens_pred == true_label
        border_color = "green" if correct else "red"
        for spine in ax_img.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        status = "CORRECT" if correct else "INCORRECT"
        ax_img.set_title(f"True: {true_name}  |  {status}", fontsize=12,
                         fontweight="bold", color=border_color, pad=8)

        ax_txt = fig.add_subplot(gs[0, 1])
        ax_txt.axis("off")
        ax_txt.text(
            0.03, 0.95, report_text,
            transform=ax_txt.transAxes,
            fontsize=9.5, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
        )

        fig.suptitle(
            f"Sample {rank}/{n_samples} — VLM Ensemble Pipeline Output",
            fontsize=13, fontweight="bold",
        )

        out_path = sample_dir / f"sample_{rank:02d}_{img_name}.png"
        fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(indices)} sample outputs to: {sample_dir}")


# ============================================================
# MAIN ENSEMBLE INFERENCE
# ============================================================

def run_ensemble_inference(
    msag_weight: float = DEFAULT_MSAG_WEIGHT,
    vlm_weight: float = DEFAULT_VLM_WEIGHT,
    batch_size: int = 8,
    seed: int = 42,
) -> None:
    seed_everything(seed)
    device = device_info()
    safe_mkdir(OUTPUT_DIR)

    import gc
    for old_png in OUTPUT_DIR.glob("*.png"):
        try:
            old_png.unlink()
        except OSError:
            pass
    gc.collect()

    print("\n" + "=" * 70)
    print("  EfficientNet-B3 + MSAG + BiomedCLIP VLM Ensemble Inference")
    print("=" * 70)
    print(f"  MSAG weight : {msag_weight}")
    print(f"  VLM weight  : {vlm_weight}")
    print(f"  Batch size  : {batch_size}")

    # ----------------------------------------------------------
    # 1. Load MSAG model
    # ----------------------------------------------------------
    msag_model = load_msag_model(MSAG_CHECKPOINT, device)

    # ----------------------------------------------------------
    # 2. Load BiomedCLIP
    # ----------------------------------------------------------
    vlm_model, vlm_preprocess, vlm_tokenizer = load_biomedclip(device)

    # Pre-compute text features for DR prompts
    print("\n[VLM] Computing text features for DR prompts...")
    for i, prompt in enumerate(DR_TEXT_PROMPTS):
        print(f"  Class {i} ({CLASS_NAMES[i]}): \"{prompt}\"")
    text_features = compute_vlm_text_features(
        vlm_model, vlm_tokenizer, DR_TEXT_PROMPTS, device
    )
    print(f"  Text features shape: {text_features.shape}")

    # ----------------------------------------------------------
    # 3. Prepare test dataset
    # ----------------------------------------------------------
    print("\n[Dataset] Scanning test set...")
    df = collect_dataset_records(DATASET_ROOT)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  Test images: {len(test_df)}")
    for c in range(NUM_CLASSES):
        count = (test_df["class_idx"] == c).sum()
        print(f"    {CLASS_NAMES[c]}: {count}")

    msag_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_dataset = PooledFundusDataset(df, "test", msag_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=pooled_collate, pin_memory=True,
    )

    # ----------------------------------------------------------
    # 4. Run inference
    # ----------------------------------------------------------
    print("\n[Inference] Running ensemble inference on test set...")
    all_true = []
    all_msag_probs = []
    all_vlm_probs = []
    all_paths = []
    all_ids = []

    t0 = time.time()

    for images, labels, paths, image_ids in tqdm(test_loader, desc="Ensemble inference"):
        # --- MSAG model ---
        images_gpu = images.to(device, non_blocking=True)
        with torch.no_grad():
            msag_logits = msag_model(images_gpu)
            msag_prob = F.softmax(msag_logits, dim=-1).cpu().numpy()

        # --- BiomedCLIP VLM ---
        vlm_prob = vlm_zero_shot_batch(
            vlm_model, vlm_preprocess, text_features, paths, device
        ).cpu().numpy()

        all_true.extend(labels.numpy().tolist())
        all_msag_probs.append(msag_prob)
        all_vlm_probs.append(vlm_prob)
        all_paths.extend(paths)
        all_ids.extend(image_ids)

    elapsed = time.time() - t0
    print(f"  Inference completed in {elapsed:.1f}s")

    all_true = np.array(all_true)
    all_msag_probs = np.concatenate(all_msag_probs, axis=0)
    all_vlm_probs = np.concatenate(all_vlm_probs, axis=0)

    # ----------------------------------------------------------
    # 5. Compute ensemble predictions
    # ----------------------------------------------------------
    ensemble_probs = msag_weight * all_msag_probs + vlm_weight * all_vlm_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    msag_preds = np.argmax(all_msag_probs, axis=1)
    vlm_preds = np.argmax(all_vlm_probs, axis=1)

    # ----------------------------------------------------------
    # 6. Compute metrics for all three
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    results = {}
    for name, preds, probs in [
        ("MSAG Only", msag_preds, all_msag_probs),
        ("VLM Only (BiomedCLIP)", vlm_preds, all_vlm_probs),
        (f"Ensemble (w_msag={msag_weight}, w_vlm={vlm_weight})", ensemble_preds, ensemble_probs),
    ]:
        metrics = compute_metrics(all_true, preds, probs)
        results[name] = metrics

        print(f"\n--- {name} ---")
        print(f"  Accuracy     : {metrics['accuracy']:.4f}")
        print(f"  Macro F1     : {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1  : {metrics['weighted_f1']:.4f}")
        print(f"  QWK          : {metrics['qwk']:.4f}")
        print(f"  Per-class:")
        for cls_name, cls_metrics in metrics["per_class"].items():
            print(f"    {cls_name:20s}  P={cls_metrics['precision']:.4f}  "
                  f"R={cls_metrics['recall']:.4f}  F1={cls_metrics['f1']:.4f}  "
                  f"N={cls_metrics['support']}")

    # ----------------------------------------------------------
    # 7. Confusion matrices
    # ----------------------------------------------------------
    print("\n[Plots] Generating confusion matrices...")
    for name_key, short_name in [
        ("MSAG Only", "msag_only"),
        ("VLM Only (BiomedCLIP)", "vlm_only"),
        (f"Ensemble (w_msag={msag_weight}, w_vlm={vlm_weight})", "ensemble"),
    ]:
        cm = np.array(results[name_key]["confusion_matrix"])
        plot_confusion_matrix(
            cm, f"Confusion Matrix — {name_key}",
            OUTPUT_DIR / f"confusion_matrix_{short_name}.png",
        )

    # ----------------------------------------------------------
    # 8. Qualitative visual analysis
    # ----------------------------------------------------------
    print("\n[Qualitative] Generating visual comparison and inference reports...")
    plot_qualitative(
        all_true, all_paths,
        msag_preds, vlm_preds, ensemble_preds,
        all_msag_probs, all_vlm_probs, ensemble_probs,
        save_dir=OUTPUT_DIR,
        samples_per_class=3, seed=seed,
    )

    # ----------------------------------------------------------
    # 9. Individual sample outputs
    # ----------------------------------------------------------
    print("\n[Samples] Saving individual sample outputs...")
    save_sample_outputs(
        all_true, all_paths,
        msag_preds, vlm_preds, ensemble_preds,
        all_msag_probs, all_vlm_probs, ensemble_probs,
        save_dir=OUTPUT_DIR,
        n_samples=10, seed=seed,
    )

    # ----------------------------------------------------------
    # 10. Weight sensitivity analysis
    # ----------------------------------------------------------
    print("\n[Analysis] Weight sensitivity sweep...")
    sensitivity = plot_weight_sensitivity(
        all_true, all_msag_probs, all_vlm_probs,
        OUTPUT_DIR / "weight_sensitivity.png",
    )
    results["weight_sensitivity"] = sensitivity

    # ----------------------------------------------------------
    # 10. Save results
    # ----------------------------------------------------------
    results_path = OUTPUT_DIR / "vlm_ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Per-image predictions CSV
    records = []
    for i in range(len(all_true)):
        records.append({
            "image_id": all_ids[i],
            "path": all_paths[i],
            "true_label": int(all_true[i]),
            "true_class": CLASS_NAMES[int(all_true[i])],
            "msag_pred": int(msag_preds[i]),
            "vlm_pred": int(vlm_preds[i]),
            "ensemble_pred": int(ensemble_preds[i]),
            "msag_correct": bool(msag_preds[i] == all_true[i]),
            "vlm_correct": bool(vlm_preds[i] == all_true[i]),
            "ensemble_correct": bool(ensemble_preds[i] == all_true[i]),
            **{f"msag_prob_{c}": float(all_msag_probs[i, c]) for c in range(NUM_CLASSES)},
            **{f"vlm_prob_{c}": float(all_vlm_probs[i, c]) for c in range(NUM_CLASSES)},
            **{f"ensemble_prob_{c}": float(ensemble_probs[i, c]) for c in range(NUM_CLASSES)},
        })
    csv_path = OUTPUT_DIR / "per_image_predictions.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"  Per-image CSV: {csv_path}")

    # ----------------------------------------------------------
    # 11. Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  MSAG Only       : Acc={results['MSAG Only']['accuracy']:.4f}  "
          f"QWK={results['MSAG Only']['qwk']:.4f}")
    print(f"  VLM Only        : Acc={results['VLM Only (BiomedCLIP)']['accuracy']:.4f}  "
          f"QWK={results['VLM Only (BiomedCLIP)']['qwk']:.4f}")
    ens_key = f"Ensemble (w_msag={msag_weight}, w_vlm={vlm_weight})"
    print(f"  Ensemble        : Acc={results[ens_key]['accuracy']:.4f}  "
          f"QWK={results[ens_key]['qwk']:.4f}")
    print(f"\n  Best weight (sweep): w_msag={sensitivity['best_w_msag']:.2f}  "
          f"acc={sensitivity['best_accuracy']:.4f}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EfficientNet-B3 + MSAG + BiomedCLIP VLM Ensemble Inference"
    )
    parser.add_argument("--msag-weight", type=float, default=DEFAULT_MSAG_WEIGHT,
                        help=f"Weight for MSAG model (default: {DEFAULT_MSAG_WEIGHT})")
    parser.add_argument("--vlm-weight", type=float, default=DEFAULT_VLM_WEIGHT,
                        help=f"Weight for VLM model (default: {DEFAULT_VLM_WEIGHT})")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference (default: 8)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_ensemble_inference(
        msag_weight=args.msag_weight,
        vlm_weight=args.vlm_weight,
        batch_size=args.batch_size,
        seed=args.seed,
    )
