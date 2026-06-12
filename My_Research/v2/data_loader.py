"""
data_loader.py — Polars-based fast data pipeline (v2)
======================================================
Tech Stack upgrade: pandas → Polars lazy API
Expected gain: 3–10x faster data loading & preprocessing

Pipeline:
  pl.scan_parquet() → lazy group_by → collect → contiguous NumPy → Dataset

KAGGLE USERS: No changes needed here — paths come from config.py.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd  # Fallback for environments without Polars

from config import (
    DATASET_DIR, SCALER_DIR,
    FEAT_COLS, TARGET_COLS,
    N_FEATURES, N_TARGETS, SEQ_LEN,
)


class LOBDataset(Dataset):
    """
    High-performance LOB dataset.
    Pre-builds a flat index of valid windows for O(1) access.
    Zero-copy windowed access via numpy slicing.
    """

    def __init__(
        self,
        features: np.ndarray,    # (N, 32) float32 contiguous
        targets: np.ndarray,     # (N, 2) float32 contiguous
        seq_starts: np.ndarray,  # (num_seqs,) int64
        seq_lengths: np.ndarray, # (num_seqs,) int64
        need_pred: np.ndarray,   # (N,) bool
        mu: np.ndarray,
        sigma: np.ndarray,
        seq_len: int = SEQ_LEN,
        need_pred_only: bool = True,
    ):
        self.features = features
        self.targets = targets
        self.mu = mu
        self.sigma = sigma
        self.seq_len = seq_len

        # Build flat index: list of global_row_index for valid windows
        indices = []
        for s in range(len(seq_starts)):
            start = seq_starts[s]
            length = seq_lengths[s]
            for local_i in range(seq_len - 1, length):
                global_i = start + local_i
                if need_pred_only and not need_pred[global_i]:
                    continue
                indices.append(global_i)

        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end_row = self.indices[idx]
        start_row = end_row - self.seq_len + 1

        X = self.features[start_row : end_row + 1].copy()  # (100, 32)
        X = (X - self.mu) / self.sigma

        y = self.targets[end_row].copy()  # (2,)

        return torch.from_numpy(X), torch.from_numpy(y)


def _load_polars(path: str):
    """Load parquet with Polars lazy API — 3-10x faster than pandas."""
    # Lazy scan — only reads columns we need, zero-copy
    lf = pl.scan_parquet(path)

    # Collect only the columns we need
    df = lf.select(
        FEAT_COLS + TARGET_COLS + ["need_prediction", "seq_ix"]
    ).collect()

    features  = df.select(FEAT_COLS).to_numpy().astype(np.float32)
    targets   = df.select(TARGET_COLS).to_numpy().astype(np.float32)
    need_pred = df["need_prediction"].to_numpy().astype(np.bool_)
    seq_ids   = df["seq_ix"].to_numpy().astype(np.int64)

    np.ascontiguousarray(features)
    np.ascontiguousarray(targets)

    # Compute sequence boundaries
    changes = np.where(np.diff(seq_ids) != 0)[0] + 1
    seq_starts = np.concatenate([[0], changes])
    seq_ends = np.concatenate([changes, [len(seq_ids)]])
    seq_lengths = seq_ends - seq_starts

    return features, targets, need_pred, seq_starts, seq_lengths


def _load_pandas(path: str):
    """Fallback: Load parquet with pandas (for envs without Polars)."""
    df = pd.read_parquet(path)

    features  = df[FEAT_COLS].to_numpy(np.float32)
    targets   = df[TARGET_COLS].to_numpy(np.float32)
    need_pred = df["need_prediction"].to_numpy(np.bool_)
    seq_ids   = df["seq_ix"].to_numpy(np.int64)

    np.ascontiguousarray(features)
    np.ascontiguousarray(targets)

    changes = np.where(np.diff(seq_ids) != 0)[0] + 1
    seq_starts = np.concatenate([[0], changes])
    seq_ends = np.concatenate([changes, [len(seq_ids)]])
    seq_lengths = seq_ends - seq_starts

    return features, targets, need_pred, seq_starts, seq_lengths


def _load_and_prepare(path: str):
    """Dispatch to Polars (preferred) or pandas (fallback)."""
    if HAS_POLARS:
        return _load_polars(path)
    else:
        print("  [WARN] Polars not found, falling back to pandas (slower)", flush=True)
        return _load_pandas(path)


def compute_scaler(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std from training features."""
    mu = features.mean(axis=0).astype(np.float32)
    sigma = (features.std(axis=0) + 1e-8).astype(np.float32)
    return mu, sigma


def build_loaders(
    batch_size: int = 2048,
    device: str = "cuda",
    num_workers: int = 0,
    pin_memory: bool = True,
    need_pred_only: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """
    Build train and validation DataLoaders.

    Returns:
        train_loader, valid_loader, scaler_dict
        scaler_dict = {"mu": ndarray, "sigma": ndarray}
    """
    train_path = os.path.join(DATASET_DIR, "train.parquet")
    valid_path = os.path.join(DATASET_DIR, "valid.parquet")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data not found: {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Valid data not found: {valid_path}")

    backend = "Polars" if HAS_POLARS else "pandas"
    print(f"  [{backend}] Loading train.parquet ...", flush=True)
    train_f, train_t, train_np, train_ss, train_sl = _load_and_prepare(train_path)
    print(f"    {len(train_f):,} rows, {len(train_ss):,} sequences", flush=True)

    print(f"  [{backend}] Loading valid.parquet ...", flush=True)
    valid_f, valid_t, valid_np, valid_ss, valid_sl = _load_and_prepare(valid_path)
    print(f"    {len(valid_f):,} rows, {len(valid_ss):,} sequences", flush=True)

    # Compute / load cached scaler
    scaler_path = os.path.join(SCALER_DIR, "scaler.npz")
    if os.path.exists(scaler_path):
        sc = np.load(scaler_path)
        mu, sigma = sc["mu"], sc["sigma"]
        print(f"  Loaded cached scaler from {scaler_path}", flush=True)
    else:
        print(f"  Computing scaler ...", flush=True)
        mu, sigma = compute_scaler(train_f)
        np.savez(scaler_path, mu=mu, sigma=sigma)
        print(f"  Saved scaler to {scaler_path}", flush=True)

    # Build datasets
    print(f"  Building train dataset ...", flush=True)
    train_ds = LOBDataset(
        train_f, train_t, train_ss, train_sl, train_np, mu, sigma,
        need_pred_only=need_pred_only,
    )
    print(f"    {len(train_ds):,} windows", flush=True)

    print(f"  Building valid dataset ...", flush=True)
    valid_ds = LOBDataset(
        valid_f, valid_t, valid_ss, valid_sl, valid_np, mu, sigma,
        need_pred_only=need_pred_only,
    )
    print(f"    {len(valid_ds):,} windows", flush=True)

    # Device-specific DataLoader settings
    if device == "cpu":
        pin_memory = False
        if num_workers == 0:
            num_workers = 4

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    scaler = {"mu": mu, "sigma": sigma}
    return train_loader, valid_loader, scaler


if __name__ == "__main__":
    print("Testing data_loader (v2) ...")
    train_loader, valid_loader, scaler = build_loaders(batch_size=64, device="cpu", num_workers=0)
    x, y = next(iter(train_loader))
    print(f"  Batch shape: X={x.shape}, y={y.shape}")
    print(f"  X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Scaler mu shape: {scaler['mu'].shape}")
    print("  [OK] data_loader v2 works")
