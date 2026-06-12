"""
ensemble.py — Optimal blend of novel architectures (v2)
========================================================
Loads all trained ONNX models, runs a unified streaming simulation,
and finds the optimal weighting via constrained SLSQP optimization
to maximize overall Pearson correlation.

Usage:
  python v2/ensemble.py

KAGGLE USERS: No changes needed — paths come from config.py.
"""

import os
import sys
import time
import numpy as np
from collections import deque
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_DIR, ONNX_DIR, SCALER_DIR, FEAT_COLS, SEQ_LEN

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd


def get_available_models():
    """Find all available ONNX models, preferring INT8 if available."""
    models = {}
    base_names = ["xlstm", "ttt_linear", "sparse_moe", "ms_tcn"]

    for base in base_names:
        paths = [
            f"{base}_int8.onnx",
            f"{base}.onnx",
        ]
        for p in paths:
            full_path = os.path.join(ONNX_DIR, p)
            if os.path.exists(full_path):
                models[base] = full_path
                break

    return models


def load_onnx_sessions(model_dict):
    """Load ONNX runtime sessions for each model."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sessions = {}
    for name, path in model_dict.items():
        print(f"Loading {name} from {os.path.basename(path)}...")
        try:
            sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            sessions[name] = {"sess": sess, "in": inp_name, "out": out_name}
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    return sessions


def run_ensemble_streaming(sessions, scaler_path, valid_path=None):
    """Run all models through the validation set and collect raw predictions."""
    if valid_path is None:
        valid_path = os.path.join(DATASET_DIR, "valid.parquet")

    print(f"\nRunning streaming simulation over {valid_path}...")

    sc = np.load(scaler_path)
    mu = sc["mu"].astype(np.float32)
    sigma = sc["sigma"].astype(np.float32)
    inv_sigma = (1.0 / sigma).astype(np.float32)

    # Load with Polars (preferred) or pandas
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
    model_names = list(sessions.keys())

    all_preds = {name: {"t0": [], "t1": []} for name in model_names}
    true_vals = {"t0": [], "t1": []}
    latencies = []

    buf = deque(maxlen=SEQ_LEN)
    cur_seq = None
    x_buf = np.empty((SEQ_LEN, F), dtype=np.float32)
    x_in = np.empty((1, SEQ_LEN, F), dtype=np.float32)

    for i in range(N):
        if i % 100000 == 0 and i > 0:
            print(f"  Processed {i}/{N} rows...")

        if seq_ids[i] != cur_seq:
            cur_seq = seq_ids[i]
            buf.clear()

        buf.append(features[i])

        if not need_pred[i]:
            continue

        if len(buf) < SEQ_LEN:
            for name in model_names:
                all_preds[name]["t0"].append(0.0)
                all_preds[name]["t1"].append(0.0)
            true_vals["t0"].append(t0_true[i])
            true_vals["t1"].append(t1_true[i])
            continue

        t_start = time.perf_counter()

        for j, x in enumerate(buf):
            x_buf[j] = x

        np.subtract(x_buf, mu, out=x_in[0])
        np.multiply(x_in[0], inv_sigma, out=x_in[0])

        for name, meta in sessions.items():
            pred = meta["sess"].run([meta["out"]], {meta["in"]: x_in})[0][0]
            all_preds[name]["t0"].append(float(pred[0]))
            all_preds[name]["t1"].append(float(pred[1]))

        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

        true_vals["t0"].append(t0_true[i])
        true_vals["t1"].append(t1_true[i])

    for name in model_names:
        all_preds[name]["t0"] = np.array(all_preds[name]["t0"])
        all_preds[name]["t1"] = np.array(all_preds[name]["t1"])

    true_vals["t0"] = np.array(true_vals["t0"])
    true_vals["t1"] = np.array(true_vals["t1"])

    print(f"Simulation complete. Median Latency: {np.median(latencies):.2f}ms")
    return all_preds, true_vals


def optimize_weights(all_preds, true_vals):
    """Find optimal blending weights to maximize Pearson correlation."""
    print("\nOptimizing ensemble weights...")
    model_names = list(all_preds.keys())
    n_models = len(model_names)

    if n_models == 0:
        print("No models to ensemble.")
        return

    # Individual model performance
    print("\nIndividual Model Performance:")
    for name in model_names:
        r0 = np.corrcoef(true_vals["t0"], all_preds[name]["t0"])[0, 1]
        r1 = np.corrcoef(true_vals["t1"], all_preds[name]["t1"])[0, 1]
        ov = (r0 + r1) / 2
        print(f"  {name:15s} | t0: {r0:.4f} | t1: {r1:.4f} | OVERALL: {ov:.4f}")

    if n_models == 1:
        print("Only one model available. Skipping ensemble optimization.")
        return

    # Equal weighting baseline
    eq_w = np.ones(n_models) / n_models
    eq_pred_t0 = sum(eq_w[i] * all_preds[name]["t0"] for i, name in enumerate(model_names))
    eq_pred_t1 = sum(eq_w[i] * all_preds[name]["t1"] for i, name in enumerate(model_names))
    r0 = np.corrcoef(true_vals["t0"], eq_pred_t0)[0, 1]
    r1 = np.corrcoef(true_vals["t1"], eq_pred_t1)[0, 1]
    print(f"\nEqual Weights  | t0: {r0:.4f} | t1: {r1:.4f} | OVERALL: {(r0+r1)/2:.4f}")

    # SLSQP optimization
    def objective(w):
        w = w / np.sum(w)
        pred_t0 = sum(w[i] * all_preds[name]["t0"] for i, name in enumerate(model_names))
        pred_t1 = sum(w[i] * all_preds[name]["t1"] for i, name in enumerate(model_names))
        c0 = np.corrcoef(true_vals["t0"], pred_t0)[0, 1]
        c1 = np.corrcoef(true_vals["t1"], pred_t1)[0, 1]
        if np.isnan(c0): c0 = 0
        if np.isnan(c1): c1 = 0
        return -((c0 + c1) / 2)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_models)]
    initial_w = np.ones(n_models) / n_models

    res = minimize(objective, initial_w, method='SLSQP', bounds=bounds, constraints=cons)

    opt_w = res.x / np.sum(res.x)
    opt_score = -res.fun

    print(f"\nOPTIMAL ENSEMBLE:")
    print(f"  Overall Correlation: {opt_score:.4f}")
    print(f"  Weights:")
    for i, name in enumerate(model_names):
        print(f"    {name:15s}: {opt_w[i]:.4f}")


def main():
    scaler_path = os.path.join(SCALER_DIR, "scaler.npz")
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at {scaler_path}. Train a model first.")
        return

    models_to_load = get_available_models()
    if not models_to_load:
        print("Error: No ONNX models found. Train models first with:")
        print("  python v2/train.py --model all")
        return

    print(f"Found {len(models_to_load)} models for ensemble.")

    sessions = load_onnx_sessions(models_to_load)
    if not sessions:
        print("Error: Failed to load any models.")
        return

    all_preds, true_vals = run_ensemble_streaming(sessions, scaler_path)
    optimize_weights(all_preds, true_vals)


if __name__ == "__main__":
    main()
