# Archive: Batch Mode Models (Tree-Based Approach)

## Why These Files Are Archived

These models and code were designed for BATCH MODE processing:
- Require access to full sequences (all 1000 rows at once)
- Use hand-engineered features with groupby().transform()
- Tree-based models (XGBoost, LightGBM, CatBoost)

They achieved 0.2562 validation score but FAILED in competition (0.006) due to:
1. Train/inference feature mismatch
2. Streaming mode incompatibility
3. Timeout issues

## What's Here

### Models (17 total, ~27 MB)
- XGBoost: xgb_t0.pkl, xgb_t1_v2.pkl, xgb_t0_enrich.pkl
- LightGBM: lgbm_t0.pkl, lgbm_t1_v2.pkl, lgbm_t1_seq.pkl, etc.
- CatBoost: catboost_t0.pkl, catboost_t1_v2.pkl
- Ridge: ridge_t1_v2.pkl
- Stacking: stack_t0.pkl, final_t0.pkl, final_t1.pkl
- MLP: omen_mlp_t1.pkl, omen_lgbm_t1.pkl

### Source Files
- features.py: Hand-engineered feature generation (128 features)
- validate.py: Offline validation (showed 0.2562)
- train_t0_ensemble.py: Training script for T0 models
- train_final.py: Final ensemble training

## When to Use These

These models are suitable for:
- Batch processing competitions (full data access)
- Offline analysis (not real-time)
- Static prediction tasks (no streaming)
- Tabular data with hand-crafted features

NOT suitable for:
- Streaming/online prediction
- Real-time inference
- Sequence modeling tasks
- Time-series with temporal dependencies

## Current Approach (Streaming Mode)

See main project folder for:
- BiGRU, TimeMixer, Mamba-2 (sequence models)
- ONNX models for fast CPU inference
- Raw features (no hand-engineering)
- Streaming-compatible solution.py

Expected score: 0.26-0.32+ (vs 0.006 with these archived models)
