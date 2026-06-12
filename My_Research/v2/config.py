"""
config.py — Centralised configuration for the v2 pipeline
============================================================
Single source of truth for paths, feature columns, device detection.
No more duplicated constants across 8 files.

KAGGLE USERS: Change DATASET_DIR below (see comment).
"""

import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
V2_ROOT = os.path.dirname(os.path.abspath(__file__))
RESEARCH_ROOT = os.path.dirname(V2_ROOT)
PROJECT_ROOT = os.path.dirname(RESEARCH_ROOT)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  KAGGLE: Replace the line below with:                                  │
# │  DATASET_DIR = "/kaggle/input/datasets/vk2245/lob-wunderfund"         │
# └─────────────────────────────────────────────────────────────────────────┘
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")

CKPT_DIR  = os.path.join(V2_ROOT, "_checkpoints")
ONNX_DIR  = os.path.join(V2_ROOT, "_onnx")
SCALER_DIR = os.path.join(V2_ROOT, "_scalers")
MLRUNS_DIR = os.path.join(V2_ROOT, "_mlruns")

for _d in [CKPT_DIR, ONNX_DIR, SCALER_DIR, MLRUNS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Feature Schema ────────────────────────────────────────────────────────────
FEAT_COLS = (
    [f"p{i}" for i in range(12)]
    + [f"v{i}" for i in range(12)]
    + [f"dp{i}" for i in range(4)]
    + [f"dv{i}" for i in range(4)]
)
TARGET_COLS = ["t0", "t1"]
N_FEATURES  = 32
N_TARGETS   = 2
SEQ_LEN     = 100

# ── Device ────────────────────────────────────────────────────────────────────
def get_device(prefer: str = "cuda") -> str:
    """Auto-detect best available device."""
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_gpu_info() -> str:
    """Return GPU name and VRAM string, or 'CPU' if no GPU."""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"{p.name}  VRAM: {p.total_memory / 1024**3:.1f}GB"
    return "CPU only"
