<<<<<<< HEAD
# LOB Prediction Competition - Final Solution

## Competition Results

**Final Score: 0.2744** (46x improvement from initial 0.006)

### Submissions Summary
1. Tree models (batch approach) - 0.0064 ❌
2. Tree models (cached) - 0.0082 ❌
3. Tree models - TIMEOUT ❌
4. Tree models - TIMEOUT ❌
5. TimeMixer INT8 (single model) - **0.2685** ✅
6. BiGRU + TimeMixer (2-model ensemble) - **0.2744** 🏆 BEST
7. 2-model retry - TIMEOUT ❌

## Solution Overview

### Winning Approach: 2-Model Ensemble
- **BiGRU** (25% weight): Bidirectional GRU for temporal patterns
- **TimeMixer** (75% weight): Multiscale MLP for seasonal/trend decomposition
- **Quantization**: INT8 for 4x speedup
- **Runtime**: 68 minutes (over limit but scored)
- **Score**: 0.2744

### Backup Solution: Single Model
- **TimeMixer** INT8 only
- **Runtime**: 15 minutes (under limit)
- **Score**: 0.2685

## Project Structure

```
LOB Prediction/
├── reports/                          # All documentation
│   ├── DOUBLE_MODEL_REPORT.txt      # Complete technical report
│   ├── WHY_WE_SCORED_LOW.txt        # Failure analysis
│   └── BATCH_VS_STREAMING_DIFFERENCE.txt  # Architecture comparison
├── src/                              # Source code
│   ├── solution.py                  # Current solution
│   ├── solution_WINNING_0.2685.py   # Backup single model
│   ├── train_dual_optimized.py      # Training script
│   └── train_timemixer.py           # TimeMixer training
├── models/                           # Trained models
│   ├── bigru_int8.onnx              # BiGRU quantized (1.80 MB)
│   ├── timemixer_int8.onnx          # TimeMixer quantized (0.34 MB)
│   ├── scaler.npz                   # BiGRU normalization
│   └── scaler_tm.npz                # TimeMixer normalization
├── checkpoints/                      # Training checkpoints
│   ├── best.pt                      # BiGRU checkpoint
│   └── best_timemixer.pt            # TimeMixer checkpoint
├── datasets/                         # Competition data
│   ├── train.parquet                # Training data
│   └── valid.parquet                # Validation data
├── archive_batch_mode_models/        # Failed tree models
├── test_solution_fast.py            # Quick test (10% sample)
├── prepare_submission.py            # Package for submission
├── quantize_models.py               # INT8 quantization
├── README.md                        # Competition instructions
└── PROJECT_README.md                # This file
```

## Key Documentation

### For Presentation
1. **reports/DOUBLE_MODEL_REPORT.txt** - Complete technical report with all details
2. **reports/WHY_WE_SCORED_LOW.txt** - Why tree models failed, how we fixed it
3. **reports/BATCH_VS_STREAMING_DIFFERENCE.txt** - Batch vs streaming comparison

### Quick Reference
- **PROJECT_README.md** - This file (project overview)
- **README.md** - Original competition instructions

## Technical Stack

### Training
- **Framework**: PyTorch 2.x
- **GPU**: NVIDIA RTX 3050 (6GB VRAM)
- **Optimization**: AMP (mixed precision), AdamW optimizer
- **Time**: 26 minutes (TimeMixer), 48 minutes (BiGRU)

### Inference
- **Engine**: ONNX Runtime
- **Quantization**: INT8 (4x speedup)
- **Optimization**: Graph optimization, sequential execution
- **Target**: 1 vCPU, <60 minutes for 1.3M predictions

## Why We Succeeded

### Problem Diagnosis
Initial tree-based models (XGBoost, LightGBM, CatBoost) failed because:
1. **Train/inference mismatch**: Features computed on full sequences (1000 rows) vs rolling buffer (100 rows)
2. **Too slow**: Feature engineering took 50ms per prediction × 1.3M = 18 hours
3. **Wrong architecture**: Tree models are stateless, can't handle streaming data

### Solution
Switched to sequence models:
1. **No feature engineering**: Use raw 32 features, let model learn patterns
2. **Temporal awareness**: BiGRU and TimeMixer designed for sequential data
3. **Fast inference**: INT8 quantization, optimized ONNX runtime
4. **Training = Inference**: Same rolling window process, no mismatch

## Performance Comparison

| Metric | Tree Models | Single Model | 2-Model Ensemble |
|--------|-------------|--------------|------------------|
| Score | 0.006 | 0.2685 | **0.2744** |
| Runtime | Timeout | 15 min | 68 min |
| Models | 17 (25 MB) | 1 (0.34 MB) | 2 (2.14 MB) |
| Features | 128 (engineered) | 32 (raw) | 32 (raw) |
| Speed | 0 pred/s | 1045 pred/s | ~320 pred/s |

## How to Use

### Test Locally
```bash
# Quick test (10% sample, 2 minutes)
python test_solution_fast.py
```

### Create Submission
```bash
# Package models and solution
python prepare_submission.py

# Upload submission.zip to competition
```

### Train Models
```bash
# Train both BiGRU and TimeMixer
python src/train_dual_optimized.py

# Quantize to INT8
python quantize_models.py
```

## Key Insights

1. **Architecture matters more than complexity**: Simple TimeMixer (0.34 MB) beat 17 tree models (25 MB)
2. **Match model to problem**: Streaming requires sequence models, not tree models
3. **Speed is critical**: 0.30 score with timeout = 0.00 score
4. **Quantization is powerful**: INT8 gives 4x speedup with <1% accuracy loss
5. **Ensemble helps**: 2-model gave +2.2% boost over single model

## Lessons Learned

### What Worked ✅
- Switching to sequence models (critical decision)
- INT8 quantization for speed
- Raw features instead of hand-engineered features
- Ensemble diversity (BiGRU + TimeMixer)
- Testing on 10% sample for fast iteration

### What Didn't Work ❌
- Tree models (train/inference mismatch)
- Hand-engineered features (too slow, wrong for streaming)
- Transformer for t1 (too weak, t1=0.06)
- TCN (failed to converge)
- Training specialized models (not enough time)

## Competition Statistics

- **Total submissions**: 7
- **Successful submissions**: 2 (0.2685, 0.2744)
- **Failed submissions**: 5 (timeouts, low scores)
- **Best improvement**: 46x (0.006 → 0.2744)
- **Time spent**: 5 days (3 days on wrong approach, 2 days on correct approach)

## Future Improvements

If we had more time:
1. Train for more epochs (25 instead of 3)
2. Hyperparameter tuning (grid search)
3. Longer context window (200 steps instead of 100)
4. 3-model ensemble (add specialized t1 model)
5. Model distillation (large model → small model)

Realistic ceiling: 0.30-0.33 with current approach

---

**Competition**: LOB Prediction Challenge  
**Date**: March 2, 2026  
**Team**: VK224  
**Final Score**: 0.2744 🏆  
**Status**: SUCCESS ✅

For detailed technical information, see `reports/` folder.
=======
# -Vk224-Wunder-Challenge-LOB-Predictorium
LOB Prediction Challenge: Rank 120/4917 (Top 2.4%) | Failed with 17 tree models (0.006), pivoted to BiGRU+TimeMixer ensemble with INT8 quantization (0.2812). Complete journey from batch processing mistake to streaming success. Includes models, training scripts, and detailed analysis.
>>>>>>> d4d2b5a27aa7e099895c215bb4ec71cecb955655
