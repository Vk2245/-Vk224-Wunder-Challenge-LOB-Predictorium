"""
evaluate.py — Streaming evaluation harness (v2)
=================================================
Same competition-matching protocol as v1, but with Polars I/O.
Simulates exact competition server:
  - Rolling deque(maxlen=100), reset on new seq_ix
  - Only scores need_prediction=1 rows
  - Reports Pearson correlation per target + overall + latency

KAGGLE USERS: No changes needed — paths come from config.py.
"""

import os
import time
import numpy as np
from collections import deque
from typing import Dict, Optional

from config import DATASET_DIR, FEAT_COLS, SEQ_LEN

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd


def streaming_evaluate(
    onnx_path: str,
    scaler_path: str,
    valid_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run streaming evaluation exactly matching competition protocol.

    Args:
        onnx_path: Path to ONNX model
        scaler_path: Path to scaler .npz (keys: 'mu', 'sigma')
        valid_path: Path to validation parquet
        verbose: Print progress

    Returns:
        Dict with corr_t0, corr_t1, overall, latency_p50, latency_p95, latency_p99
    """
    import onnxruntime as ort

    if valid_path is None:
        valid_path = os.path.join(DATASET_DIR, "valid.parquet")

    # Load scaler
    sc = np.load(scaler_path)
    mu = sc["mu"].astype(np.float32)
    sigma = sc["sigma"].astype(np.float32)
    inv_sigma = (1.0 / sigma).astype(np.float32)

    # ONNX session — single-threaded for fair benchmarking
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Load validation data (Polars preferred)
    if verbose:
        backend = "Polars" if HAS_POLARS else "pandas"
        print(f"  [{backend}] Loading {valid_path} ...", flush=True)

    if HAS_POLARS:
        df = pl.read_parquet(valid_path)
        features  = df.select(FEAT_COLS).to_numpy().astype(np.float32)
        seq_ids   = df["seq_ix"].to_numpy().astype(np.int64)
        need_pred = df["need_prediction"].to_numpy().astype(np.bool_)
        t0_true   = df["t0"].to_numpy().astype(np.float64)
        t1_true   = df["t1"].to_numpy().astype(np.float64)
        N = df.height
    else:
        df = pd.read_parquet(valid_path)
        features  = df[FEAT_COLS].to_numpy(np.float32)
        seq_ids   = df["seq_ix"].to_numpy(np.int64)
        need_pred = df["need_prediction"].to_numpy(np.bool_)
        t0_true   = df["t0"].to_numpy(np.float64)
        t1_true   = df["t1"].to_numpy(np.float64)
        N = len(df)

    F = len(FEAT_COLS)

    # Streaming simulation (Batched for speed)
    buf = deque(maxlen=SEQ_LEN)
    cur_seq = None
    x_buf = np.empty((SEQ_LEN, F), dtype=np.float32)

    preds_t0 = []
    preds_t1 = []
    true_t0 = []
    true_t1 = []
    latencies = []
    
    batch_x = []
    BATCH_SIZE = 1024

    for i in range(N):
        if verbose and i % 100000 == 0 and i > 0:
            print(f"    [Streaming] Processed {i:,}/{N:,} rows ({(i/N)*100:.1f}%) ...", flush=True)

        if seq_ids[i] != cur_seq:
            cur_seq = seq_ids[i]
            buf.clear()

        buf.append(features[i])

        if not need_pred[i]:
            continue

        if len(buf) < SEQ_LEN:
            preds_t0.append(0.0)
            preds_t1.append(0.0)
            true_t0.append(t0_true[i])
            true_t1.append(t1_true[i])
            continue

        x_buf[:len(buf)] = buf
        
        # Scale
        scaled_x = (x_buf - mu) * inv_sigma
        batch_x.append(scaled_x)
        
        true_t0.append(t0_true[i])
        true_t1.append(t1_true[i])

        if len(batch_x) >= BATCH_SIZE or i == N - 1:
            t_start = time.perf_counter()
            
            # Try batched inference first, fall back to per-sample if model
            # was exported with fixed batch=1 (onnxscript converter bug)
            batch_arr = np.stack(batch_x)
            try:
                pred_batch = sess.run([out_name], {inp_name: batch_arr})[0]
            except Exception:
                # Fallback: run one sample at a time
                preds_single = []
                for single_x in batch_x:
                    p = sess.run([out_name], {inp_name: single_x[np.newaxis]})[0]
                    preds_single.append(p[0])
                pred_batch = np.array(preds_single)
            
            t_end = time.perf_counter()
            
            # Approximate latency per row in ms
            lat_per_row = ((t_end - t_start) * 1000) / len(batch_x)
            latencies.extend([lat_per_row] * len(batch_x))
            
            preds_t0.extend(pred_batch[:, 0].tolist())
            preds_t1.extend(pred_batch[:, 1].tolist())
            
            batch_x.clear()

    # Compute Pearson correlation
    preds_t0 = np.array(preds_t0)
    preds_t1 = np.array(preds_t1)
    true_t0 = np.array(true_t0)
    true_t1 = np.array(true_t1)

    corr_t0 = float(np.corrcoef(true_t0, preds_t0)[0, 1])
    corr_t1 = float(np.corrcoef(true_t1, preds_t1)[0, 1])
    if np.isnan(corr_t0): corr_t0 = 0.0
    if np.isnan(corr_t1): corr_t1 = 0.0
    overall = (corr_t0 + corr_t1) / 2.0

    lat_arr = np.array(latencies)

    results = {
        "corr_t0": corr_t0,
        "corr_t1": corr_t1,
        "overall": overall,
        "n_predictions": len(preds_t0),
        "latency_p50_ms": float(np.percentile(lat_arr, 50)) if len(lat_arr) > 0 else 0,
        "latency_p95_ms": float(np.percentile(lat_arr, 95)) if len(lat_arr) > 0 else 0,
        "latency_p99_ms": float(np.percentile(lat_arr, 99)) if len(lat_arr) > 0 else 0,
    }

    if verbose:
        print(f"\n  STREAMING EVALUATION RESULTS")
        print(f"  {'='*50}")
        print(f"  Model:       {os.path.basename(onnx_path)}")
        print(f"  Predictions: {results['n_predictions']:,}")
        print(f"  corr_t0:     {results['corr_t0']:.4f}")
        print(f"  corr_t1:     {results['corr_t1']:.4f}")
        print(f"  OVERALL:     {results['overall']:.4f}")
        print(f"  Latency p50: {results['latency_p50_ms']:.2f} ms")
        print(f"  Latency p99: {results['latency_p99_ms']:.2f} ms")
        print(f"  {'='*50}")

    return results
