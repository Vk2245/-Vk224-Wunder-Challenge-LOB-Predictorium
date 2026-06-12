# v2 — Optimised LOB Prediction Pipeline

> **Tech Stack**: Polars · torch.compile · AMP · Optuna · Selective Quant · Structured Pruning · MLflow
>
> Same 4 novel architectures (xLSTM, TTT-Linear, Sparse MoE, MS-TCN) — but **2–3x faster training** and **smarter infrastructure**.

---

## Quick Start (Local)

```bash
# 1. Install new dependencies
pip install -r v2/requirements.txt

# 2. Train a single model
python v2/train.py --model ms_tcn --device cuda

# 3. Train all 4 models
python v2/train.py --model all --device cuda

# 4. Run HPO to find best hyperparameters
python v2/hpo.py --model xlstm --n-trials 30

# 5. Optimize ensemble weights
python v2/ensemble.py
```

---

## Kaggle Execution Guide

### Cell 1: Copy Code
```python
!cp -r /kaggle/input/datasets/vk2245/my-lob-code/v2 /kaggle/working/
!ls /kaggle/working/v2/
```

### Cell 2: Update Dataset Path
```python
# Only ONE line to change — in config.py
!sed -i 's|DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")|DATASET_DIR = "/kaggle/input/datasets/vk2245/lob-wunderfund"|g' /kaggle/working/v2/config.py
```

### Cell 3: Install & Train
```bash
!pip install polars optuna mlflow onnx onnxruntime

# Train all models (fastest to slowest)
!python /kaggle/working/v2/train.py --model ms_tcn --device cuda
!python /kaggle/working/v2/train.py --model sparse_moe --device cuda
!python /kaggle/working/v2/train.py --model ttt_linear --device cuda
!python /kaggle/working/v2/train.py --model xlstm --device cuda
```

### Cell 4: Ensemble
```bash
!python /kaggle/working/v2/ensemble.py
```

### Cell 5 (Optional): HPO
```bash
# Find optimal hyperparameters (run overnight)
!python /kaggle/working/v2/hpo.py --model ms_tcn --n-trials 30 --device cuda
```

---

## What's Different from `score_maximizers/`

| Feature | `score_maximizers/` (v1) | `v2/` |
|---------|-------------------------|-------|
| Data loading | `pd.read_parquet` (pandas) | Polars lazy scan (3–10x faster) |
| Training speed | Eager PyTorch | `torch.compile` + AMP (1.5–2x) |
| Hyperparams | Hardcoded | Optuna HPO with MedianPruner |
| Quantization | Naive full INT8 | Selective (skip sensitive layers) |
| Model FLOPs | Full dense | Structured BN pruning (30–50% less) |
| Experiment tracking | `print()` | MLflow (params, metrics, artifacts) |
| Code duplication | ~400 lines per script | Shared `engine.py` (~30 lines/model) |
| Entry point | 8 separate scripts | Single `train.py --model <name>` |

---

## Architecture

```
v2/
├── config.py          # Paths, constants, device detection
├── data_loader.py     # Polars pipeline (pandas fallback)
├── engine.py          # Shared: train/eval/export/prune/MLflow
├── evaluate.py        # Streaming evaluator
├── losses.py          # Heteroscedastic NLL, Pinball, Combined
├── train.py           # CLI: python v2/train.py --model xlstm
├── hpo.py             # CLI: python v2/hpo.py --model xlstm
├── ensemble.py        # SLSQP weight optimization
├── requirements.txt   # polars, optuna, mlflow
├── README.md          # This file
└── models/
    ├── __init__.py    # MODEL_REGISTRY + get_model()
    ├── xlstm.py       # Architecture + defaults
    ├── ttt_linear.py  # Architecture + defaults
    ├── sparse_moe.py  # Architecture + defaults
    └── ms_tcn.py      # Architecture + defaults
```

---

## CLI Reference

### train.py
```
--model       xlstm|ttt_linear|sparse_moe|ms_tcn|all
--device      cuda|cpu (default: cuda)
--epochs      Override max epochs
--lr          Override learning rate
--batch       Override batch size
--hidden      Override hidden dimension
--dropout     Override dropout
--grad-accum  Gradient accumulation steps (default: 1)
--no-compile  Disable torch.compile
--prune       Apply structured pruning before ONNX export
--prune-ratio Pruning ratio (default: 0.2)
```

### hpo.py
```
--model       xlstm|ttt_linear|sparse_moe|ms_tcn|all
--device      cuda|cpu
--n-trials    Number of Optuna trials (default: 30)
--hpo-epochs  Max epochs per trial (default: 15)
--study-name  Custom study name
```
