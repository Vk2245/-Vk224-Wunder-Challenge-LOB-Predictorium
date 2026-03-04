import numpy as np
import pandas as pd
import joblib
import os
import sys
import gc

BASE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE, "models")):
    MODELS = os.path.join(BASE, "models")
else:
    MODELS = os.path.join(os.path.dirname(BASE), "models")

sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))

from features import generate_features, generate_features_t1

try:
    from utils import DataPoint
except ImportError:
    class DataPoint:
        def __init__(self, seq_ix, step_in_seq, need_prediction, state):
            self.seq_ix = seq_ix
            self.step_in_seq = step_in_seq
            self.need_prediction = need_prediction
            self.state = state

def _p(n): return os.path.join(MODELS, n)

STATE_COLS = (
    [f"p{i}" for i in range(12)] +
    [f"v{i}" for i in range(12)] +
    [f"dp{i}" for i in range(4)] +
    [f"dv{i}" for i in range(4)]
)


class PredictionModel:
    def __init__(self):
        print(f"Loading models from: {MODELS}")
        self.t0_cols     = joblib.load(_p("xgb_features.pkl"))
        self.t0e_cols    = joblib.load(_p("t0_enriched_features.pkl"))
        self.xgb_t0      = joblib.load(_p("xgb_t0.pkl"))
        self.lgbm_t0     = joblib.load(_p("lgbm_t0.pkl"))
        self.stack_t0    = joblib.load(_p("stack_t0.pkl"))
        self.final_t0    = joblib.load(_p("final_t0.pkl"))
        self.lgbm_enrich = joblib.load(_p("lgbm_t0_enrich.pkl"))
        self.xgb_enrich  = joblib.load(_p("xgb_t0_enrich.pkl"))
        self.cat_t0      = joblib.load(_p("catboost_t0.pkl"))
        self.t1_cols     = joblib.load(_p("xgb_t1_v2_features.pkl"))
        self.xgb_v2      = joblib.load(_p("xgb_t1_v2.pkl"))
        self.lgbm_v2     = joblib.load(_p("lgbm_t1_v2.pkl"))
        self.rdg_v2      = joblib.load(_p("ridge_t1_v2.pkl"))
        self.cat_v2      = joblib.load(_p("catboost_t1_v2.pkl"))
        print("All models loaded.")

    def predict(self, dp):
        if not dp.need_prediction:
            return None

        # Build single-row dataframe from state
        row = {"seq_ix": dp.seq_ix, "step_in_seq": dp.step_in_seq,
               "need_prediction": True}
        for i, col in enumerate(STATE_COLS):
            row[col] = float(dp.state[i])
        df = pd.DataFrame([row])

        # T0 features
        df_orig  = generate_features(df.copy())
        X_orig   = np.nan_to_num(df_orig[self.t0_cols].values.astype(np.float32))

        mid      = df_orig["midprice"].values.astype(np.float32)
        vol_tot  = df_orig["volume_total"].values.astype(np.float32)
        df_e     = df_orig.copy().reset_index(drop=True)
        df_r     = df.reset_index(drop=True)
        for i in range(12):
            df_e[f"p{i}_norm"] = (df_r[f"p{i}"].values / (mid + 1e-6)).astype(np.float32)
            df_e[f"v{i}_norm"] = (df_r[f"v{i}"].values / (vol_tot + 1e-6)).astype(np.float32)
        for i in range(4):
            df_e[f"dv{i}"]        = df_r[f"dv{i}"].values.astype(np.float32)
            df_e[f"dp{i}_vs_mid"] = (df_r[f"dp{i}"].values - mid).astype(np.float32)
        for i in range(1, 6):
            df_e[f"bid_gap_{i}"] = ((df_r[f"p{i-1}"].values - df_r[f"p{i}"].values) / (mid + 1e-6)).astype(np.float32)
            df_e[f"ask_gap_{i}"] = ((df_r[f"p{6+i}"].values - df_r[f"p{6+i-1}"].values) / (mid + 1e-6)).astype(np.float32)
        bv = df_r[[f"v{i}" for i in range(6)]].values.astype(np.float32)
        av = df_r[[f"v{i}" for i in range(6, 12)]].values.astype(np.float32)
        for i in range(1, 6):
            df_e[f"bid_vol_ratio_{i}"] = (bv[:, i] / (bv[:, 0] + 1e-6)).astype(np.float32)
            df_e[f"ask_vol_ratio_{i}"] = (av[:, i] / (av[:, 0] + 1e-6)).astype(np.float32)
        df_e.replace([np.inf, -np.inf], 0, inplace=True)
        df_e.fillna(0, inplace=True)
        X_enrich = df_e[self.t0e_cols].values.astype(np.float32)

        # T0 prediction - no sequence features needed
        p_xgb  = self.xgb_t0.predict(X_orig)
        p_lgbm = self.lgbm_t0.predict(X_orig)
        p_stk  = self.stack_t0.predict(np.column_stack([p_xgb, p_lgbm]))
        p_fin  = self.final_t0.predict(np.column_stack([p_xgb, p_lgbm, p_stk]))
        p_bt0  = (p_fin * 0.5 + p_lgbm * 0.5).astype(np.float32)
        p_le   = self.lgbm_enrich.predict(X_enrich)
        p_xe   = self.xgb_enrich.predict(X_enrich)
        p_ca   = self.cat_t0.predict(X_orig)
        t0_out = float(np.clip(
            p_le * 0.33 + p_xe * 0.27 + p_ca * 0.20 + p_bt0 * 0.20,
            -6, 6)[0])

        # T1 features - also per-row, no sequence lag features
        df_t1  = generate_features_t1(df.copy())
        X_t1v  = np.nan_to_num(df_t1[self.t1_cols].values.astype(np.float32))
        p_xv   = self.xgb_v2.predict(X_t1v)
        p_lv   = self.lgbm_v2.predict(X_t1v)
        p_rdg  = self.rdg_v2.predict(X_t1v)
        p_bv2  = (p_xv * 0.5 + p_lv * 0.5).astype(np.float32)
        p_cat  = self.cat_v2.predict(np.column_stack([X_t1v, p_xv, p_lv, p_rdg, p_bv2]))
        p_sv1b = np.clip(p_bv2, -6, 6)
        t1_out = float(np.clip(p_cat * 0.25 + p_sv1b * 0.75, -6, 6)[0])

        return np.array([t0_out, t1_out], dtype=np.float64)