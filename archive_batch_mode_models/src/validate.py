"""
validate.py — Ground truth validation
Baseline: t0=0.3525  t1=0.1599  overall=0.2562
Updated:   + dart blend candidates for T0 confirmation
"""
import sys, os
_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)
sys.path.insert(0, _ROOT)

import pandas as pd, numpy as np, joblib, gc
from features import generate_features, generate_features_t1
from utils import weighted_pearson_correlation as wpc

print("Loading validation dataset...")
raw    = pd.read_parquet("datasets/valid.parquet")
df_orig= generate_features(raw.copy())
df_t1v = generate_features_t1(raw.copy())

t0_cols  = joblib.load("models/xgb_features.pkl")
t0e_cols = joblib.load("models/t0_enriched_features.pkl")
t1_cols  = joblib.load("models/xgb_t1_v2_features.pkl")

mask      = df_orig["need_prediction"] == True
X_orig    = df_orig[mask][t0_cols].values.astype(np.float32)
X_t1v     = df_t1v[mask][t1_cols].values.astype(np.float32)
y_true_t0 = df_orig[mask]["t0"].values.astype(np.float32)
y_true_t1 = df_orig[mask]["t1"].values.astype(np.float32)
np.nan_to_num(X_orig, copy=False); np.nan_to_num(X_t1v, copy=False)

# ── Enriched T0 features ──────────────────────────────────────
mid     = df_orig.loc[mask,"midprice"].values.astype(np.float32)
vol_tot = df_orig.loc[mask,"volume_total"].values.astype(np.float32)
raw_m   = raw[mask].reset_index(drop=True)
df_e    = df_orig[mask].copy().reset_index(drop=True)
for i in range(12):
    df_e[f"p{i}_norm"]=(raw_m[f"p{i}"].values/(mid+1e-6)).astype(np.float32)
    df_e[f"v{i}_norm"]=(raw_m[f"v{i}"].values/(vol_tot+1e-6)).astype(np.float32)
for i in range(4):
    df_e[f"dv{i}"]       =raw_m[f"dv{i}"].values.astype(np.float32)
    df_e[f"dp{i}_vs_mid"]=(raw_m[f"dp{i}"].values-mid).astype(np.float32)
for i in range(1,6):
    df_e[f"bid_gap_{i}"] =((raw_m[f"p{i-1}"].values-raw_m[f"p{i}"].values)/(mid+1e-6)).astype(np.float32)
    df_e[f"ask_gap_{i}"] =((raw_m[f"p{6+i}"].values-raw_m[f"p{6+i-1}"].values)/(mid+1e-6)).astype(np.float32)
bv=raw_m[[f"v{i}" for i in range(6)]].values.astype(np.float32)
av=raw_m[[f"v{i}" for i in range(6,12)]].values.astype(np.float32)
for i in range(1,6):
    df_e[f"bid_vol_ratio_{i}"]=(bv[:,i]/(bv[:,0]+1e-6)).astype(np.float32)
    df_e[f"ask_vol_ratio_{i}"]=(av[:,i]/(av[:,0]+1e-6)).astype(np.float32)
df_e.replace([np.inf,-np.inf],0,inplace=True); df_e.fillna(0,inplace=True)
X_enrich=df_e[t0e_cols].values.astype(np.float32)
del df_e,raw_m,bv,av; gc.collect()
print(f"Samples: {X_orig.shape[0]}")

# ── T0 predictions ────────────────────────────────────────────
print("\nT0 predictions...")
p_xgb  = joblib.load("models/xgb_t0.pkl").predict(X_orig)
p_lgbm = joblib.load("models/lgbm_t0.pkl").predict(X_orig)
p_stk  = joblib.load("models/stack_t0.pkl").predict(np.column_stack([p_xgb,p_lgbm]))
p_fin  = joblib.load("models/final_t0.pkl").predict(np.column_stack([p_xgb,p_lgbm,p_stk]))
p_bt0  = (p_fin*0.5+p_lgbm*0.5).astype(np.float32)
p_le   = joblib.load("models/lgbm_t0_enrich.pkl").predict(X_enrich)
p_go   = joblib.load("models/goss_t0.pkl").predict(X_enrich)
p_da   = joblib.load("models/dart_t0.pkl").predict(X_enrich)
p_xe   = joblib.load("models/xgb_t0_enrich.pkl").predict(X_enrich)
p_ca   = joblib.load("models/catboost_t0.pkl").predict(X_orig)

# Base blends
full_blend = np.clip(p_le*0.25+p_go*0.25+p_xe*0.20+p_ca*0.15+p_bt0*0.15,-6,6)
t0_preds = {
    "full_blend (submitted)" : full_blend,
    "dart"                   : np.clip(p_da,-6,6),
    "lgbm_enrich"            : np.clip(p_le,-6,6),
    "goss"                   : np.clip(p_go,-6,6),
    "xgb_enrich"             : np.clip(p_xe,-6,6),
    "catboost"               : np.clip(p_ca,-6,6),
    "old_blend"              : np.clip(p_bt0,-6,6),
}
# Dart blend candidates
for dw in [0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25]:
    r = 1.0-dw
    t0_preds[f"dart@{dw:.0%}"] = np.clip(
        p_le*(0.25*r)+p_go*(0.25*r)+p_xe*(0.20*r)+
        p_ca*(0.15*r)+p_bt0*(0.15*r)+p_da*dw,-6,6)
# Monte Carlo (same seed as FAST_SCORE_BOOST)
P6   = np.array([np.clip(x,-6,6) for x in [p_le,p_go,p_da,p_xe,p_ca,p_bt0]])
best_mc_r,best_mc_w=-999,None; np.random.seed(42)
for _ in range(3000):
    w=np.random.dirichlet(np.ones(6))
    r=wpc(y_true_t0,np.clip((P6*w[:,None]).sum(0),-6,6))
    if r>best_mc_r: best_mc_r,best_mc_w=r,w
t0_preds["monte_carlo"] = np.clip((P6*best_mc_w[:,None]).sum(0),-6,6)

print("  T0 scores:")
best_t0_score,best_t0_name=-999,""
for name,p in sorted(t0_preds.items(),key=lambda x:-wpc(y_true_t0,x[1])):
    s=wpc(y_true_t0,p)
    tag=" ← SUBMITTED" if name=="full_blend (submitted)" else ""
    print(f"    [{name}]: {s:.4f}{tag}")
    if s>best_t0_score: best_t0_score,best_t0_name=s,name
print(f"  >>> Best t0: [{best_t0_name}] = {best_t0_score:.4f}")

# ── T1 predictions ────────────────────────────────────────────
print("\nT1 predictions...")
p_xv  = joblib.load("models/xgb_t1_v2.pkl").predict(X_t1v)
p_lv  = joblib.load("models/lgbm_t1_v2.pkl").predict(X_t1v)
p_rdg = joblib.load("models/ridge_t1_v2.pkl").predict(X_t1v)
p_bv2 = (p_xv*0.5+p_lv*0.5).astype(np.float32)
p_cat = joblib.load("models/catboost_t1_v2.pkl").predict(
        np.column_stack([X_t1v,p_xv,p_lv,p_rdg,p_bv2]))

print("  Building seq features...")
ds=raw.copy()
bv=ds[[f"v{i}" for i in range(6)]].values.astype(np.float32)
av=ds[[f"v{i}" for i in range(6,12)]].values.astype(np.float32)
bs=bv.sum(1); as_=av.sum(1); vt=bs+as_
ds["midprice"]    =((ds["p0"]+ds["p6"])/2).astype(np.float32)
ds["vol_imbal"]   =((bs-as_)/(vt+1e-6)).astype(np.float32)
ds["microprice"]  =((ds["p0"]*as_+ds["p6"]*bs)/(vt+1e-6)).astype(np.float32)
ds["depth_imbal"] =((ds["v0"]-ds["v6"])/(ds["v0"]+ds["v6"]+1e-6)).astype(np.float32)
tp=ds[[f"dp{i}" for i in range(4)]].values.astype(np.float32)
tv=ds[[f"dv{i}" for i in range(4)]].values.astype(np.float32)
ds["signed_trade"]=(np.sign(tp.mean(1)-ds["midprice"].values)*tv.sum(1)).astype(np.float32)
ds["spread"]      =((ds["p6"]-ds["p0"])/(ds["p0"]+1e-6)).astype(np.float32)
SIGS=["vol_imbal","microprice","depth_imbal","signed_trade","spread"]
grp=ds.groupby("seq_ix"); pm=ds["need_prediction"]==True; cols={}
for s in SIGS:
    for l in range(10): cols[f"{s}_lag{l}"]=grp[s].shift(l).fillna(0).astype(np.float32)
for s in SIGS:
    cols[f"{s}_mean"]=grp[s].transform("mean").astype(np.float32)
    cols[f"{s}_std"] =grp[s].transform("std").fillna(0).astype(np.float32)
    cols[f"{s}_cum"] =grp[s].cumsum().astype(np.float32)
fn=list(cols.keys())
Xs1=np.zeros((int(pm.sum()),len(fn)),dtype=np.float32)
for i,c in enumerate(fn): Xs1[:,i]=cols[c][pm].values
Xs1=np.nan_to_num(Xs1)

p_sv1 = joblib.load("models/lgbm_t1_seq.pkl").predict(Xs1)
Xv3   = np.column_stack([Xs1,p_fin,p_bt0,p_bv2,p_xv,p_lv])
p_sv3 = joblib.load("models/lgbm_t1_seq_v3.pkl").predict(Xv3)
p_ft1 = np.clip(joblib.load("models/final_t1.pkl").predict(
        np.column_stack([p_rdg,p_cat,p_sv1,p_sv3])),-6,6)
p_sv1_blend = np.clip(p_sv1*0.6+p_bv2*0.4,-6,6)
p_best      = np.clip(p_ft1*0.25+p_sv1_blend*0.75,-6,6)
del Xs1,Xv3; gc.collect()

t1_preds={
    "best_known (submitted)" : p_best,
    "sv1_blend"              : p_sv1_blend,
    "seq_v1"                 : np.clip(p_sv1,-6,6),
    "seq_v3"                 : np.clip(p_sv3,-6,6),
    "final_t1"               : p_ft1,
    "blend_v2"               : p_bv2,
}

print("\nT1 scores:")
best_t1_score,best_t1_name=-999,""
for name,p in sorted(t1_preds.items(),key=lambda x:-wpc(y_true_t1,x[1])):
    s=wpc(y_true_t1,p)
    tag=" ← SUBMITTED" if name=="best_known (submitted)" else ""
    print(f"  [{name}]: {s:.4f}{tag}")
    if s>best_t1_score: best_t1_score,best_t1_name=s,name
print(f"\n  >>> Best t1: [{best_t1_name}] = {best_t1_score:.4f}")

# ── FINAL SUMMARY ─────────────────────────────────────────────
overall = (best_t0_score+best_t1_score)/2
print(f"\n{'='*52}")
print(f"  Submitted:    t0=0.3525  t1=0.1599  overall=0.2562")
print(f"  Current best: t0={best_t0_score:.4f}  t1={best_t1_score:.4f}  overall={overall:.4f}")
print(f"  Delta:        t0={best_t0_score-0.3525:+.4f}  t1={best_t1_score-0.1599:+.4f}  overall={overall-0.2562:+.4f}")
print(f"{'='*52}")
if overall > 0.2562+0.0005:
    print(f"  IMPROVEMENT CONFIRMED — update solution.py and SUBMIT")
elif overall > 0.2562:
    print(f"  Marginal gain — decide whether worth submitting")
else:
    print(f"  No improvement — 0.2562 remains best")
print(f"{'='*52}")