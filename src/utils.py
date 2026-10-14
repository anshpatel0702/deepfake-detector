"""
utils.py – Shared utilities: seed, device, constants, class-weight computation.
"""

import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# ──────────────────────────── Project paths ────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw" / "real-vs-fake"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# ──────────────────────────── Hyper-parameters ─────────────────────────
SEED: int = 42
IMG_SIZE: int = 224  # EfficientNet-B0 default input
BATCH_SIZE: int = 64
NUM_WORKERS: int = 4  # set to 0 on Windows if multiprocessing errors occur
LABEL_MAP: Dict[str, int] = {"real": 0, "fake": 1}

# Stage-1 (frozen backbone)
STAGE1_EPOCHS: int = 5
STAGE1_LR: float = 1e-3

# Stage-2 (full fine-tune)
STAGE2_EPOCHS: int = 10
STAGE2_LR: float = 1e-5

DROPOUT: float = 0.3
BACKBONE_FEATURES: int = 1280  # EfficientNet-B0 output features


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may slow things slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


def compute_class_weights(data_dir: Path) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for BCEWithLogitsLoss.

    Returns a single scalar tensor = n_negative / n_positive (pos_weight),
    where class 0 = real (negative) and class 1 = fake (positive).
    """
    real_count = len(list((data_dir / "real").glob("*")))
    fake_count = len(list((data_dir / "fake").glob("*")))

    if real_count == 0 or fake_count == 0:
        raise ValueError(f"Empty class folder in {data_dir}")

    # pos_weight > 1 up-weights the positive (fake) class
    # pos_weight < 1 down-weights it (when fake is the majority)
    pos_weight = real_count / fake_count
    print(f"[INFO] Class counts -> real: {real_count}, fake: {fake_count}")
    print(f"[INFO] pos_weight (real/fake): {pos_weight:.4f}")
    return torch.tensor([pos_weight], dtype=torch.float32)


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "frames").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "splits").mkdir(parents=True, exist_ok=True)
