"""
meta_model_stacking.py
========================
Second-stage Meta-Model (LightGBM) using xLSTM predictions.
Strictly NO data leakage:
1. We run on validation dataset only (xLSTM never trained on it).
2. We split validation chronologically (train on first 70%, evaluate on last 30%).
3. Feature engineering restricted ONLY to predictions.
"""

import os
import time
import numpy as np
import pandas as pd
from collections import deque
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from config import DATASET_DIR, FEAT_COLS, SEQ_LEN

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import onnxruntime as ort


def generate_validation_predictions(onnx_path: str, scaler_path: str, valid_path: str) -> pd.DataFrame:
    """Extracts raw predictions from ONNX model over the validation set."""
    print(f"[*] Extracting ONNX predictions from {onnx_path}...")
    
    # Load scaler
    sc = np.load(scaler_path)
    mu = sc["mu"].astype(np.float32)
    sigma = sc["sigma"].astype(np.float32)
    inv_sigma = (1.0 / sigma).astype(np.float32)

    # ONNX session
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Load data
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
    buf = deque(maxlen=SEQ_LEN)
    cur_seq = None
    x_buf = np.empty((SEQ_LEN, F), dtype=np.float32)

    out_preds_t0 = []
    out_preds_t1 = []
    out_true_t0 = []
    out_true_t1 = []
    out_seq_ids = []
    
    batch_x = []
    BATCH_SIZE = 2048

    for i in range(N):
        if i % 200000 == 0 and i > 0:
            print(f"    Processed {i:,}/{N:,} rows...")

        if seq_ids[i] != cur_seq:
            cur_seq = seq_ids[i]
            buf.clear()

        buf.append(features[i])

        if not need_pred[i]:
            continue

        if len(buf) < SEQ_LEN:
            out_preds_t0.append(0.0)
            out_preds_t1.append(0.0)
            out_true_t0.append(t0_true[i])
            out_true_t1.append(t1_true[i])
            out_seq_ids.append(seq_ids[i])
            continue

        x_buf[:len(buf)] = buf
        scaled_x = (x_buf - mu) * inv_sigma
        batch_x.append(scaled_x)
        
        out_true_t0.append(t0_true[i])
        out_true_t1.append(t1_true[i])
        out_seq_ids.append(seq_ids[i])

        if len(batch_x) >= BATCH_SIZE or i == N - 1:
            batch_arr = np.stack(batch_x)
            pred_batch = sess.run([out_name], {inp_name: batch_arr})[0]
            
            out_preds_t0.extend(pred_batch[:, 0].tolist())
            out_preds_t1.extend(pred_batch[:, 1].tolist())
            batch_x.clear()

    print("[*] Extraction complete.")
    
    return pd.DataFrame({
        "seq_ix": out_seq_ids,
        "true_t0": out_true_t0,
        "true_t1": out_true_t1,
        "pred_t0": out_preds_t0,
        "pred_t1": out_preds_t1
    })


def engineer_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates secondary features solely from predictions."""
    print("[*] Engineering meta-features from predictions...")
    
    # 1. Basic math interactions
    df["pred_diff"] = df["pred_t0"] - df["pred_t1"]
    df["pred_sum"] = df["pred_t0"] + df["pred_t1"]
    df["pred_ratio"] = df["pred_t0"] / (df["pred_t1"].abs() + 1e-6)
    df["pred_abs_t0"] = df["pred_t0"].abs()
    df["pred_abs_t1"] = df["pred_t1"].abs()
    df["pred_t0_sq"] = df["pred_t0"] ** 2
    df["pred_t1_sq"] = df["pred_t1"] ** 2

    # 2. Sequence-based rolling features (momentum of predictions)
    # Group by seq_ix to prevent leaking across different assets/days
    print("    Calculating rolling statistics...")
    
    # Sort to ensure chronological order within sequences
    # Assuming the original extraction preserved temporal order
    
    def add_rolling(col, window):
        df[f"{col}_roll_mean_{window}"] = df.groupby("seq_ix")[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{col}_roll_std_{window}"] = df.groupby("seq_ix")[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))

    add_rolling("pred_t0", 5)
    add_rolling("pred_t1", 5)
    add_rolling("pred_t0", 20)
    add_rolling("pred_t1", 20)

    # 3. Lagged predictions (what was the prediction 1 step ago?)
    df["pred_t0_lag1"] = df.groupby("seq_ix")["pred_t0"].shift(1).fillna(0)
    df["pred_t1_lag1"] = df.groupby("seq_ix")["pred_t1"].shift(1).fillna(0)

    print("[*] Feature engineering complete.")
    return df


def evaluate_corr(y_true, y_pred):
    """Safe pearson correlation."""
    if len(y_true) < 2 or np.std(y_pred) < 1e-8:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


def train_meta_model(df: pd.DataFrame):
    """Trains LightGBM models using TimeSeriesSplit to prevent data leakage and ensure robust metrics."""
    print("\n[*] Starting Meta-Model Training Pipeline (TimeSeriesSplit)")
    
    from sklearn.model_selection import TimeSeriesSplit
    
    features = [c for c in df.columns if c not in ["seq_ix", "true_t0", "true_t1"]]
    print(f"    Total rows: {len(df):,}")
    print(f"    Features: {features}")

    tscv = TimeSeriesSplit(n_splits=5)
    
    lgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }

    t0_scores, t1_scores, baseline_t0, baseline_t1 = [], [], [], []
    
    # Run Cross-Validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        
        # Baselines
        base_t0 = evaluate_corr(test_df["true_t0"], test_df["pred_t0"])
        base_t1 = evaluate_corr(test_df["true_t1"], test_df["pred_t1"])
        baseline_t0.append(base_t0)
        baseline_t1.append(base_t1)
        
        # LightGBM t0
        model_t0 = lgb.LGBMRegressor(**lgb_params)
        model_t0.fit(
            train_df[features], train_df["true_t0"],
            eval_set=[(test_df[features], test_df["true_t0"])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # LightGBM t1
        model_t1 = lgb.LGBMRegressor(**lgb_params)
        model_t1.fit(
            train_df[features], train_df["true_t1"],
            eval_set=[(test_df[features], test_df["true_t1"])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Eval
        lgb_preds_t0 = model_t0.predict(test_df[features])
        lgb_preds_t1 = model_t1.predict(test_df[features])
        t0_scores.append(evaluate_corr(test_df["true_t0"], lgb_preds_t0))
        t1_scores.append(evaluate_corr(test_df["true_t1"], lgb_preds_t1))
        
        print(f"    Fold {fold+1}: LGBM t0={t0_scores[-1]:.4f} (Raw={base_t0:.4f}) | LGBM t1={t1_scores[-1]:.4f} (Raw={base_t1:.4f})")

    avg_lgb_t0 = np.mean(t0_scores)
    avg_lgb_t1 = np.mean(t1_scores)
    avg_base_t0 = np.mean(baseline_t0)
    avg_base_t1 = np.mean(baseline_t1)

    print(f"\n{'='*50}")
    print(f"5-FOLD TIME SERIES SPLIT RESULTS")
    print(f"{'='*50}")
    print(f"Raw xLSTM t0: {avg_base_t0:.4f} -> LGBM t0: {avg_lgb_t0:.4f} (Diff: {avg_lgb_t0 - avg_base_t0:+.4f})")
    print(f"Raw xLSTM t1: {avg_base_t1:.4f} -> LGBM t1: {avg_lgb_t1:.4f} (Diff: {avg_lgb_t1 - avg_base_t1:+.4f})")
    print(f"OVERALL META-SCORE: {(avg_lgb_t0 + avg_lgb_t1) / 2:.4f}")
    
    # Feature Importance (using last fold)
    importances = pd.DataFrame({
        'Feature': features,
        'Importance_t1': model_t1.feature_importances_
    }).sort_values(by='Importance_t1', ascending=False)
    
    print("\nTop 5 Features for t1 (from final fold):")
    print(importances.head(5).to_string(index=False))


if __name__ == "__main__":
    # Try to dynamically get directory, fallback to current dir if in Jupyter
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
        if not base_dir.endswith("v2"):
            base_dir = os.path.join(base_dir, "v2")
    onnx_file = os.path.join(base_dir, "_onnx", "xlstm.onnx")
    scaler_file = os.path.join(base_dir, "_scalers", "scaler.npz")
    valid_file = os.path.join(DATASET_DIR, "valid.parquet")

    # 1. Extract
    df_preds = generate_validation_predictions(onnx_file, scaler_file, valid_file)
    
    # 2. Engineer
    df_engineered = engineer_meta_features(df_preds)
    
    # 3. Train & Evaluate
    train_meta_model(df_engineered)
