"""
train_t0_ensemble.py — Memory-safe version
Extracts features column-by-column into pre-allocated float32 array.
No float64 intermediate allocation. No OOM errors.
"""

import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
import catboost as cb, joblib, gc, sys, os
from scipy.stats import pearsonr
sys.path.insert(0, 'src')
from features import generate_features

CACHE_Xa = "models/cache_Xa_t0.npy"
CACHE_Xo = "models/cache_Xo_t0.npy"
CACHE_y  = "models/cache_y_t0.npy"

def safe_extract(df, cols, mask):
    """Extract columns one-by-one into pre-allocated float32 — avoids OOM."""
    n = int(mask.sum())
    X = np.zeros((n, len(cols)), dtype=np.float32)
    idx = np.where(mask)[0]
    for i, col in enumerate(cols):
        X[:, i] = df[col].values[idx].astype(np.float32)
        if i % 20 == 0: print(f"  extracting col {i}/{len(cols)}...", end="\r")
    print()
    return X

if os.path.exists(CACHE_Xa):
    print("Loading cached features...")
    Xa = np.load(CACHE_Xa); Xo = np.load(CACHE_Xo); y = np.load(CACHE_y)
    all_cols = joblib.load("models/t0_enriched_features.pkl")
    old_cols = joblib.load("models/xgb_features.pkl")
    print(f"Loaded: Xa={Xa.shape} Xo={Xo.shape}")
else:
    print("Computing features...")
    df      = pd.read_parquet("datasets/train.parquet")
    df_feat = generate_features(df)

    # Add enriched features directly as float32 series
    mid     = df_feat["midprice"].values.astype(np.float32)
    vol_tot = df_feat["volume_total"].values.astype(np.float32)

    for i in range(12):
        df_feat[f"p{i}_norm"] = (df[f"p{i}"].values / (mid + 1e-6)).astype(np.float32)
        df_feat[f"v{i}_norm"] = (df[f"v{i}"].values / (vol_tot + 1e-6)).astype(np.float32)
    for i in range(4):
        df_feat[f"dv{i}"]        = df[f"dv{i}"].values.astype(np.float32)
        df_feat[f"dp{i}_vs_mid"] = (df[f"dp{i}"].values - mid).astype(np.float32)
    for i in range(1, 6):
        df_feat[f"bid_gap_{i}"] = ((df[f"p{i-1}"].values - df[f"p{i}"].values) / (mid + 1e-6)).astype(np.float32)
        df_feat[f"ask_gap_{i}"] = ((df[f"p{6+i}"].values - df[f"p{6+i-1}"].values) / (mid + 1e-6)).astype(np.float32)
    bv = df[[f"v{i}" for i in range(6)]].values.astype(np.float32)
    av = df[[f"v{i}" for i in range(6,12)]].values.astype(np.float32)
    for i in range(1, 6):
        df_feat[f"bid_vol_ratio_{i}"] = (bv[:,i] / (bv[:,0] + 1e-6)).astype(np.float32)
        df_feat[f"ask_vol_ratio_{i}"] = (av[:,i] / (av[:,0] + 1e-6)).astype(np.float32)

    df_feat.replace([np.inf,-np.inf], 0, inplace=True)
    df_feat.fillna(0, inplace=True)

    exclude  = ["seq_ix","step_in_seq","need_prediction","t0","t1"]
    all_cols = [c for c in df_feat.columns if c not in exclude]
    old_cols = joblib.load("models/xgb_features.pkl")
    pred_mask = df_feat["need_prediction"].values.astype(bool)

    print(f"Extracting {len(all_cols)} enriched features (float32, no OOM)...")
    Xa = safe_extract(df_feat, all_cols, pred_mask)
    print(f"Extracting {len(old_cols)} old features...")
    Xo = safe_extract(df_feat, old_cols, pred_mask)
    y  = df_feat["t0"].values[pred_mask].astype(np.float32)

    joblib.dump(all_cols, "models/t0_enriched_features.pkl")
    del df, df_feat, bv, av; gc.collect()

    print("Caching to disk for next run...")
    np.save(CACHE_Xa, Xa); np.save(CACHE_Xo, Xo); np.save(CACHE_y, y)
    print("Cached!")

n = len(Xa); nv = n // 10
Xa_tr,Xa_va = Xa[:n-nv],Xa[n-nv:]
Xo_tr,Xo_va = Xo[:n-nv],Xo[n-nv:]
y_tr, y_va  = y[:n-nv], y[n-nv:]
del Xa, Xo; gc.collect()
print(f"Train: {len(y_tr)}  Val: {len(y_va)}  Enriched: {Xa_tr.shape[1]}  Old: {Xo_tr.shape[1]}")

preds = {}

# ── Existing best pipeline ────────────────────────────────────────────────────
print("\nExisting best pipeline...")
xgb_t0=joblib.load("models/xgb_t0.pkl"); lgbm_t0=joblib.load("models/lgbm_t0.pkl")
stack_t0=joblib.load("models/stack_t0.pkl"); final_t0=joblib.load("models/final_t0.pkl")
p_x=xgb_t0.predict(Xo_va); p_l=lgbm_t0.predict(Xo_va)
p_s=stack_t0.predict(np.column_stack([p_x,p_l]))
p_f=final_t0.predict(np.column_stack([p_x,p_l,p_s]))
preds["existing"]=np.clip(p_f*0.5+p_l*0.5,-6,6)
r,_=pearsonr(y_va,preds["existing"]); print(f"  existing: {r:.4f}")

# ── LGBM GBDT enriched (GPU) ──────────────────────────────────────────────────
print("\nLGBM GBDT enriched (GPU)...")
m=lgb.LGBMRegressor(boosting_type="gbdt",n_estimators=3000,learning_rate=0.05,
    max_depth=7,num_leaves=127,subsample=0.8,colsample_bytree=0.6,
    min_child_samples=50,reg_alpha=0.5,reg_lambda=3.0,
    device="gpu",n_jobs=-1,random_state=42,verbose=-1)
m.fit(Xa_tr,y_tr,eval_set=[(Xa_va,y_va)],
      callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(200)])
p=m.predict(Xa_va); r,_=pearsonr(y_va,p); preds["lgbm_enrich"]=p
print(f"  lgbm_enrich: {r:.4f}"); joblib.dump(m,"models/lgbm_t0_enrich.pkl")

# ── LGBM DART (GPU) ───────────────────────────────────────────────────────────
print("\nLGBM DART (GPU)...")
m=lgb.LGBMRegressor(boosting_type="dart",n_estimators=1500,learning_rate=0.05,
    max_depth=7,num_leaves=127,subsample=0.8,colsample_bytree=0.6,
    min_child_samples=50,reg_alpha=0.5,reg_lambda=3.0,
    drop_rate=0.1,skip_drop=0.5,
    device="gpu",n_jobs=-1,random_state=42,verbose=-1)
m.fit(Xa_tr,y_tr,eval_set=[(Xa_va,y_va)],callbacks=[lgb.log_evaluation(200)])
p=m.predict(Xa_va); r,_=pearsonr(y_va,p); preds["dart"]=p
print(f"  dart: {r:.4f}"); joblib.dump(m,"models/dart_t0.pkl")

# ── LGBM GOSS (GPU) ───────────────────────────────────────────────────────────
print("\nLGBM GOSS (GPU)...")
m=lgb.LGBMRegressor(boosting_type="goss",n_estimators=3000,learning_rate=0.05,
    max_depth=7,num_leaves=127,colsample_bytree=0.6,
    min_child_samples=50,reg_alpha=0.5,reg_lambda=3.0,
    top_rate=0.2,other_rate=0.1,
    device="gpu",n_jobs=-1,random_state=42,verbose=-1)
m.fit(Xa_tr,y_tr,eval_set=[(Xa_va,y_va)],
      callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(200)])
p=m.predict(Xa_va); r,_=pearsonr(y_va,p); preds["goss"]=p
print(f"  goss: {r:.4f}"); joblib.dump(m,"models/goss_t0.pkl")

# ── CatBoost (GPU) ────────────────────────────────────────────────────────────
print("\nCatBoost (GPU)...")
m=cb.CatBoostRegressor(
    iterations=2000,learning_rate=0.05,depth=7,
    l2_leaf_reg=3.0,subsample=0.8,
    bootstrap_type="Bernoulli",
    min_data_in_leaf=50,loss_function="RMSE",
    random_seed=42,verbose=200,early_stopping_rounds=80,
    task_type="GPU",devices="0")
m.fit(Xo_tr,y_tr,eval_set=(Xo_va,y_va))
p=m.predict(Xo_va); r,_=pearsonr(y_va,p); preds["cat"]=p
print(f"  cat: {r:.4f}"); joblib.dump(m,"models/catboost_t0.pkl")

# ── XGB enriched (GPU) ────────────────────────────────────────────────────────
print("\nXGB enriched (GPU)...")
m=xgb.XGBRegressor(n_estimators=3000,learning_rate=0.05,max_depth=7,
    subsample=0.8,colsample_bytree=0.6,min_child_weight=50,
    reg_alpha=0.5,reg_lambda=3.0,
    device="cuda",random_state=42,verbosity=0,early_stopping_rounds=80)
m.fit(Xa_tr,y_tr,eval_set=[(Xa_va,y_va)],verbose=200)
p=m.predict(Xa_va); r,_=pearsonr(y_va,p); preds["xgb_enrich"]=p
print(f"  xgb_enrich: {r:.4f}"); joblib.dump(m,"models/xgb_t0_enrich.pkl")

# ── All scores ────────────────────────────────────────────────────────────────
print("\nAll scores:")
for name,p in sorted(preds.items(),key=lambda x:-pearsonr(y_va,x[1])[0]):
    r,_=pearsonr(y_va,p); print(f"  {name}: {r:.4f}")

# ── Best blend search ─────────────────────────────────────────────────────────
print("\nSearching best blend (5000 trials)...")
keys=list(preds.keys()); P=np.array([preds[k] for k in keys])
best_r,best_w=-999,None
np.random.seed(42)
for _ in range(5000):
    w=np.random.dirichlet(np.ones(len(keys)))
    r,_=pearsonr(y_va,(P*w[:,None]).sum(0))
    if r>best_r: best_r=r; best_w=w

print("\nBest blend weights:")
for k,w in sorted(zip(keys,best_w),key=lambda x:-x[1]):
    print(f"  {k}: {w:.3f}")

joblib.dump({"keys":keys,"weights":best_w},"models/t0_blend_weights.pkl")

print(f"\n{'='*50}")
print(f"Old best t0 : 0.3481")
print(f"New best t0 : {best_r:.4f}")
print(f"Delta       : {best_r-0.3481:+.4f}")
print(f"{'='*50}")