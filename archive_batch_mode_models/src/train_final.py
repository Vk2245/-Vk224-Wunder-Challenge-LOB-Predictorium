"""
train_final.py — Single file: train + blend + verify
Novel approach: OMEN-style MLP on raw LOB structure for T1
(spatial encoding of 12-level order book, per-row, no sequence needed)
Blends with existing T0 models for best overall score.

Run: python src/train_final.py
Time: ~8 min
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, joblib, gc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from features import generate_features, generate_features_t1
from utils import weighted_pearson_correlation as wpc

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
def mp(n): return os.path.join(MODELS, n)
T0s = time.time()
def log(msg): print(f"[{time.time()-T0s:5.1f}s] {msg}", flush=True)

def lob_spatial_features(raw_sub):
    """
    OMEN-inspired: encode LOB as spatial structure.
    Key insight: normalize each level relative to spread/mid,
    then add cross-level interaction features.
    Pure per-row computation. Works in streaming.
    """
    r = raw_sub.reset_index(drop=True)
    bp = r[[f"p{i}" for i in range(6)]].values.astype(np.float32)
    ap = r[[f"p{i}" for i in range(6,12)]].values.astype(np.float32)
    bv = r[[f"v{i}" for i in range(6)]].values.astype(np.float32)
    av = r[[f"v{i}" for i in range(6,12)]].values.astype(np.float32)
    dp = r[[f"dp{i}" for i in range(4)]].values.astype(np.float32)
    dv = r[[f"dv{i}" for i in range(4)]].values.astype(np.float32)

    mid    = ((bp[:,0] + ap[:,0]) / 2).astype(np.float32)
    spread = (ap[:,0] - bp[:,0] + 1e-8).astype(np.float32)
    tv     = bv.sum(1) + av.sum(1) + 1e-8

    # Spatial encoding: price levels relative to mid/spread
    bp_s = ((bp - mid[:,None]) / spread[:,None])
    ap_s = ((ap - mid[:,None]) / spread[:,None])
    bv_s = bv / tv[:,None]
    av_s = av / tv[:,None]

    # Cross-level imbalance at each depth
    imbal = (bv - av) / (bv + av + 1e-8)

    # Cumulative depth
    cum_b = np.cumsum(bv_s, axis=1)
    cum_a = np.cumsum(av_s, axis=1)
    cum_i = (cum_b - cum_a) / (cum_b + cum_a + 1e-8)

    # Price gaps
    bgap = np.diff(bp, axis=1) / spread[:,None]   # negative (bid levels fall)
    agap = np.diff(ap, axis=1) / spread[:,None]   # positive

    # Weighted mid (microprice)
    micro = (bp[:,0]*av[:,0] + ap[:,0]*bv[:,0]) / (bv[:,0]+av[:,0]+1e-8)
    micro_s = (micro - mid) / spread

    # Trade pressure
    trade_dir = np.sign(dp.mean(1) - mid)
    trade_sz  = dv.sum(1) / tv
    signed_flow = trade_dir * trade_sz
    recent_price_move = (dp[:,0] - mid) / spread

    # LOB slope (how fast depth accumulates - resilience measure)
    bid_slope = np.polyfit(np.arange(6), bv_s.T, 1)[0] if len(r) > 1 else bv_s[:,5] - bv_s[:,0]
    ask_slope = np.polyfit(np.arange(6), av_s.T, 1)[0] if len(r) > 1 else av_s[:,5] - av_s[:,0]

    X = np.column_stack([
        bp_s, ap_s, bv_s, av_s,   # 24 features: spatial LOB
        imbal,                      # 6: per-level imbalance
        cum_i,                      # 6: cumulative imbalance
        bgap, agap,                 # 10: price gaps
        micro_s,                    # 1: microprice
        bv_s.sum(1), av_s.sum(1),  # 2: total depth fractions (=1 but normalized)
        (bv_s[:,0]-av_s[:,0]),      # 1: top-of-book imbalance
        signed_flow,                # 1: signed trade flow
        recent_price_move,          # 1: recent trade vs mid
        dv[:,0]/tv, dv[:,1]/tv,    # 2: individual trade sizes
        bid_slope, ask_slope,       # 2: depth slope
    ]).astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X

log("Loading train...")
raw_tr  = pd.read_parquet("datasets/train.parquet")
mask_tr = raw_tr["need_prediction"] == True
y_t0_tr = raw_tr[mask_tr]["t0"].values.astype(np.float32)
y_t1_tr = raw_tr[mask_tr]["t1"].values.astype(np.float32)
log(f"Train size: {mask_tr.sum()}")

log("Building spatial LOB features (train)...")
X_lob_tr = lob_spatial_features(raw_tr[mask_tr])
log(f"Feature shape: {X_lob_tr.shape}")
del raw_tr; gc.collect()

log("Loading valid...")
raw_v  = pd.read_parquet("datasets/valid.parquet")
mask_v = raw_v["need_prediction"] == True
y_t0_v = raw_v[mask_v]["t0"].values.astype(np.float32)
y_t1_v = raw_v[mask_v]["t1"].values.astype(np.float32)

log("Building spatial LOB features (valid)...")
X_lob_v = lob_spatial_features(raw_v[mask_v])

# ─── Existing model predictions ────────────────────────────────
log("Getting existing model predictions...")
t0_cols  = joblib.load(mp("xgb_features.pkl"))
t0e_cols = joblib.load(mp("t0_enriched_features.pkl"))
t1_cols  = joblib.load(mp("xgb_t1_v2_features.pkl"))
df0v = generate_features(raw_v.copy())
dt1v = generate_features_t1(raw_v.copy())
X0v  = np.nan_to_num(df0v[mask_v][t0_cols].values.astype(np.float32))
X1v  = np.nan_to_num(dt1v[mask_v][t1_cols].values.astype(np.float32))
midv = df0v.loc[mask_v,"midprice"].values.astype(np.float32)
volv = df0v.loc[mask_v,"volume_total"].values.astype(np.float32)
rmv  = raw_v[mask_v].reset_index(drop=True)
dev  = df0v[mask_v].copy().reset_index(drop=True)
for i in range(12):
    dev[f"p{i}_norm"]=(rmv[f"p{i}"].values/(midv+1e-6)).astype(np.float32)
    dev[f"v{i}_norm"]=(rmv[f"v{i}"].values/(volv+1e-6)).astype(np.float32)
for i in range(4):
    dev[f"dv{i}"]=rmv[f"dv{i}"].values.astype(np.float32)
    dev[f"dp{i}_vs_mid"]=(rmv[f"dp{i}"].values-midv).astype(np.float32)
for i in range(1,6):
    dev[f"bid_gap_{i}"]=((rmv[f"p{i-1}"].values-rmv[f"p{i}"].values)/(midv+1e-6)).astype(np.float32)
    dev[f"ask_gap_{i}"]=((rmv[f"p{6+i}"].values-rmv[f"p{6+i-1}"].values)/(midv+1e-6)).astype(np.float32)
bvv=rmv[[f"v{i}" for i in range(6)]].values.astype(np.float32)
avv=rmv[[f"v{i}" for i in range(6,12)]].values.astype(np.float32)
for i in range(1,6):
    dev[f"bid_vol_ratio_{i}"]=(bvv[:,i]/(bvv[:,0]+1e-6)).astype(np.float32)
    dev[f"ask_vol_ratio_{i}"]=(avv[:,i]/(avv[:,0]+1e-6)).astype(np.float32)
dev.replace([np.inf,-np.inf],0,inplace=True); dev.fillna(0,inplace=True)
Xev = dev[t0e_cols].values.astype(np.float32)
del df0v, dt1v, dev, rmv, bvv, avv; gc.collect()

p_xgb  = joblib.load(mp("xgb_t0.pkl")).predict(X0v)
p_lgbm = joblib.load(mp("lgbm_t0.pkl")).predict(X0v)
p_stk  = joblib.load(mp("stack_t0.pkl")).predict(np.column_stack([p_xgb,p_lgbm]))
p_fin  = joblib.load(mp("final_t0.pkl")).predict(np.column_stack([p_xgb,p_lgbm,p_stk]))
p_bt0  = (p_fin*0.5+p_lgbm*0.5).astype(np.float32)
p_le   = joblib.load(mp("lgbm_t0_enrich.pkl")).predict(Xev)
p_xe   = joblib.load(mp("xgb_t0_enrich.pkl")).predict(Xev)
p_ca   = joblib.load(mp("catboost_t0.pkl")).predict(X0v)
p_t0_exist = np.clip(p_le*0.33+p_xe*0.27+p_ca*0.20+p_bt0*0.20,-6,6)
log(f"Existing T0: {wpc(y_t0_v, p_t0_exist):.4f}")

p_xv  = joblib.load(mp("xgb_t1_v2.pkl")).predict(X1v)
p_lv  = joblib.load(mp("lgbm_t1_v2.pkl")).predict(X1v)
p_rdg = joblib.load(mp("ridge_t1_v2.pkl")).predict(X1v)
p_bv2 = (p_xv*0.5+p_lv*0.5).astype(np.float32)
p_cat = joblib.load(mp("catboost_t1_v2.pkl")).predict(np.column_stack([X1v,p_xv,p_lv,p_rdg,p_bv2]))
p_t1_exist = np.clip(p_cat*0.4+p_bv2*0.6,-6,6)
log(f"Existing T1 (per-row): {wpc(y_t1_v, p_t1_exist):.4f}")

# ─── Train MLP T1 (OMEN-style spatial encoding) ────────────────
log("Scaling features...")
sc = RobustScaler()
Xtr_s = sc.fit_transform(X_lob_tr).astype(np.float32)
Xv_s  = sc.transform(X_lob_v).astype(np.float32)

log("Training OMEN-MLP for T1...")
mlp_t1 = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128, 64),
    activation="relu", solver="adam",
    alpha=0.0005, learning_rate_init=0.001,
    max_iter=100, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=8,
    random_state=42, verbose=False, batch_size=8192
)
mlp_t1.fit(Xtr_s, y_t1_tr)
p_mlp_t1 = np.clip(mlp_t1.predict(Xv_s), -6, 6)
log(f"MLP T1 alone: {wpc(y_t1_v, p_mlp_t1):.4f}")

# Also train LGBM on spatial features for T1 (usually beats MLP on tabular)
log("Training LGBM on spatial features for T1...")
lgbm_spatial_t1 = LGBMRegressor(
    n_estimators=1500, learning_rate=0.03,
    num_leaves=127, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.5,
    random_state=42, n_jobs=-1, verbose=-1
)
# Combine spatial + existing predictions as features
X_combo_tr = np.column_stack([X_lob_tr,
    joblib.load(mp("xgb_t1_v2.pkl")).predict(
        np.nan_to_num(generate_features_t1(pd.read_parquet("datasets/train.parquet")
        )[pd.read_parquet("datasets/train.parquet")["need_prediction"]==True][t1_cols].values.astype(np.float32))
    )
]).astype(np.float32)
X_combo_v  = np.column_stack([X_lob_v, p_xv]).astype(np.float32)
lgbm_spatial_t1.fit(X_combo_tr, y_t1_tr)
p_lgbm_spatial = np.clip(lgbm_spatial_t1.predict(X_combo_v), -6, 6)
log(f"LGBM spatial T1: {wpc(y_t1_v, p_lgbm_spatial):.4f}")

# Find best T1 blend
log("Finding best T1 blend...")
best_r, best_combo = 0, None
for a in np.arange(0, 1.01, 0.1):
    for b in np.arange(0, 1.01-a, 0.1):
        c = round(1.0-a-b, 2)
        if c < 0: continue
        p = np.clip(a*p_mlp_t1 + b*p_lgbm_spatial + c*p_t1_exist, -6, 6)
        r = wpc(y_t1_v, p)
        if r > best_r:
            best_r, best_combo = r, (a, b, c)
a_best, b_best, c_best = best_combo
p_t1_best = np.clip(a_best*p_mlp_t1 + b_best*p_lgbm_spatial + c_best*p_t1_exist, -6, 6)
log(f"Best T1 blend: mlp*{a_best:.1f}+lgbm_s*{b_best:.1f}+exist*{c_best:.1f} = {best_r:.4f}")

overall = (wpc(y_t0_v, p_t0_exist) + best_r) / 2

# Save everything
joblib.dump(mlp_t1,         mp("omen_mlp_t1.pkl"),    compress=("zlib",3))
joblib.dump(lgbm_spatial_t1,mp("omen_lgbm_t1.pkl"),   compress=("zlib",3))
joblib.dump(sc,             mp("omen_scaler.pkl"),     compress=("zlib",3))
log(f"Saved models. MLP: {os.path.getsize(mp('omen_mlp_t1.pkl'))/1024/1024:.1f}MB  LGBM: {os.path.getsize(mp('omen_lgbm_t1.pkl'))/1024/1024:.1f}MB")

print()
print("="*60)
print(f"  T0 (existing, per-row)  : {wpc(y_t0_v, p_t0_exist):.4f}")
print(f"  T1 MLP alone            : {wpc(y_t1_v, p_mlp_t1):.4f}")
print(f"  T1 LGBM spatial alone   : {wpc(y_t1_v, p_lgbm_spatial):.4f}")
print(f"  T1 best blend           : {best_r:.4f}")
print(f"  OVERALL ESTIMATE        : {overall:.4f}")
print(f"  T1 weights: mlp={a_best:.1f} lgbm_s={b_best:.1f} exist={c_best:.1f}")
print("="*60)
print()
print("COPY THESE WEIGHTS INTO solution.py:")
print(f"  T1: p_mlp*{a_best:.1f} + p_lgbm_s*{b_best:.1f} + p_t1_base*{c_best:.1f}")