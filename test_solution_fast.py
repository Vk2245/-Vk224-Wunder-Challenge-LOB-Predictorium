"""
test_solution_fast.py — FAST TEST (10% sample, ~2 minutes)
============================================================
Tests on 10% of validation data for quick feedback.

RUN: python test_solution_fast.py
"""
import sys, os, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from utils import weighted_pearson_correlation as wpc, DataPoint

FEAT_COLS = (
    [f"p{i}" for i in range(12)] + [f"v{i}" for i in range(12)] +
    [f"dp{i}" for i in range(4)]  + [f"dv{i}" for i in range(4)]
)

print("="*60)
print("FAST TEST (10% sample)")
print("="*60)

print("\nLoading valid.parquet...")
df = pd.read_parquet(os.path.join("datasets", "valid.parquet"))

# Sample 10% of sequences
sequences = df['seq_ix'].unique()
sample_seqs = np.random.choice(sequences, size=len(sequences)//10, replace=False)
df = df[df['seq_ix'].isin(sample_seqs)].reset_index(drop=True)

print(f"Sampled: {len(df):,} rows, {len(sample_seqs):,} sequences")

print("\nLoading model...")
from solution import PredictionModel
model = PredictionModel()
print("✓ Model loaded\n")

# Pre-convert to numpy arrays for speed (avoid pandas overhead)
feat_data = df[FEAT_COLS].values.astype(np.float64)
seq_ix_arr = df['seq_ix'].values
step_arr = df['step_in_seq'].values
need_pred_arr = df['need_prediction'].values
t0_arr = df['t0'].values
t1_arr = df['t1'].values

p0, p1, y0, y1 = [], [], [], []
t_start = time.time()
n_pred = 0
n_zero = 0

# Iterate over numpy arrays (much faster than pandas)
for i in range(len(df)):
    dp = DataPoint(
        int(seq_ix_arr[i]),
        int(step_arr[i]),
        bool(need_pred_arr[i]),
        feat_data[i]
    )
    result = model.predict(dp)

    if need_pred_arr[i]:
        n_pred += 1
        if result is None or np.all(result == 0):
            p0.append(0.0); p1.append(0.0); n_zero += 1
        else:
            p0.append(float(result[0])); p1.append(float(result[1]))
        y0.append(t0_arr[i]); y1.append(t1_arr[i])

el = time.time() - t_start
r0 = wpc(np.array(y0), np.array(p0))
r1 = wpc(np.array(y1), np.array(p1))
ov = (r0 + r1) / 2
spd = n_pred / el if el > 0 else 0
est_full = el * 10 / 60  # Estimate for full dataset

print()
print("=" * 60)
print("FAST TEST RESULTS (10% sample)")
print("=" * 60)
print(f"  Predictions        : {n_pred:,}")
print(f"  Zero predictions   : {n_zero:,}")
print(f"  Time               : {el:.0f}s")
print(f"  Speed              : {spd:.0f} pred/s")
print(f"  Est. full test     : ~{est_full:.0f} min")
print(f"  t0 Pearson         : {r0:.4f}")
print(f"  t1 Pearson         : {r1:.4f}")
print(f"  OVERALL            : {ov:.4f}")
print("=" * 60)

if est_full < 15:
    print(f"\n✓ FAST ENOUGH ({est_full:.0f} min < 15 min target)")
elif est_full < 40:
    print(f"\n⚠ ACCEPTABLE ({est_full:.0f} min, might work)")
else:
    print(f"\n✗ TOO SLOW ({est_full:.0f} min, will timeout)")

if ov >= 0.28:
    print(f"✓ EXCELLENT SCORE ({ov:.4f} >= 0.28)")
elif ov >= 0.24:
    print(f"✓ GOOD SCORE ({ov:.4f} >= 0.24)")
else:
    print(f"⚠ SCORE BELOW TARGET ({ov:.4f})")
