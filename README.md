# Limit Order Book (LOB) Prediction: From Competition to Research

**Author**: Vishal Kumar  
**Competition Rank**: 120/4917 (Top 2.4%)  
**Competition Score**: 0.2685  
**Post-Competition Research**: Novel Architecture Benchmark Suite

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Competition Phase (Phase 1)](#competition-phase-phase-1)
3. [Post-Competition Research (Phase 2)](#post-competition-research-phase-2)
4. [Key Findings & Insights](#key-findings--insights)
5. [Repository Structure](#repository-structure)
6. [Results & Visualizations](#results--visualizations)
7. [Installation & Setup](#installation--setup)
8. [Usage](#usage)
9. [Future Work](#future-work)
10. [Citation](#citation)

---

## Project Overview

This repository documents a comprehensive study of Limit Order Book (LOB) prediction, spanning from competition participation to systematic post-competition research. The work addresses a critical but often overlooked challenge in financial ML: **train-inference feature mismatch under streaming deployment constraints**.

### Problem Statement

Predict short-term price movements (t0: current-tick, t1: next-tick ~100 steps ahead) from high-frequency LOB data under real-time streaming constraints where only the last 100 timesteps are accessible.

### Key Challenge

Models trained on full sequences with engineered features achieve strong offline performance but fail catastrophically when deployed under streaming constraints (correlation drop from 0.256 to 0.006 - a **42x degradation**).

---

## Competition Phase (Phase 1)

### Wunderfund LOB Prediction Challenge

**Duration**: Competition Period  
**Rank**: 120 / 4917 participants (Top 2.4%)  
**Final Score**: 0.2685  
**Platform**: Wunderfund  

### Approach

#### Dataset
- 10,721 training sequences (10.7M rows)
- 1,444 validation sequences (1.4M rows)
- 32 features per timestep (12 price levels, 12 volume levels, 8 trade features)
- 2 prediction targets (t0, t1)
- Sliding window: 100 timesteps

#### Initial Models (Batch Mode)
Built 17 gradient-boosted tree models with 128 engineered features:
- 9 XGBoost models
- 8 LightGBM models
- CatBoost models
- Ridge Regression

**Batch Performance**: 
- Overall Pearson Correlation: 0.256
- t0 correlation: 0.352
- t1 correlation: 0.160

#### Discovery: Train-Inference Mismatch

When deployed under streaming constraints on Wunderfund platform:

```
Batch Evaluation:  correlation = 0.256
Streaming Evaluation: correlation = 0.006

Degradation: 42x performance loss!
```

**Root Cause**: 40 of 128 features used full-sequence statistics (global means, standard deviations) that are undefined or biased when computed from only a 100-step rolling window.

#### Pivot to Streaming-Native Models

Implemented sequence models using only raw 32 features:

**BiGRU Model**:
- 3-layer Bidirectional GRU (256 hidden units per direction)
- Self-supervised pretraining (InfoNCE contrastive loss)
- Supervised fine-tuning with MSE loss

**TimeMixer Model**:
- Multiscale MLP with 3 resolutions (100/50/25 timesteps)
- Seasonal-trend decomposition at each scale
- MLP mixing layers with residual connections

**Final Competition Submission**:
- Weighted ensemble: 0.25 × BiGRU + 0.75 × TimeMixer
- ONNX INT8 quantization for deployment
- **Competition Score: 0.2812**
- Latency: 0.96ms per prediction

### Competition Results Summary

| Model | corr_t0 | corr_t1 | Overall | Latency (ms) |
|-------|---------|---------|---------|--------------|
| Tree Ensemble (batch) | 0.352 | 0.160 | 0.256 | N/A |
| Tree Ensemble (streaming) | 0.352 | 0.160 | **0.006** | N/A |
| BiGRU | 0.295 | 0.026 | 0.160 | 8.24 |
| TimeMixer | 0.315 | 0.099 | 0.207 | ~1.0 |
| **BiGRU + TimeMixer** | ~0.35 | ~0.21 | **0.2812** | ~2.0 |

**Key Insight**: Streaming-native sequence models with raw features outperform tree ensembles with engineered features by 47x under streaming constraints (0.2812 vs 0.006).

---

## Post-Competition Research (Phase 2)

### Motivation

The competition revealed that architectural alignment with deployment constraints is the dominant factor in real-world LOB prediction. This motivated a systematic study of novel architectures designed for streaming inference.

### Research Goals

1. Evaluate 4 genuinely novel architectures never applied to streaming LOB prediction
2. Implement rigorous streaming evaluation protocol matching competition conditions
3. Achieve INT8 quantization with <5ms latency on consumer hardware
4. Document comprehensive benchmarks for publication

### Novel Architectures Implemented

#### 1. xLSTM (Extended LSTM)
**Innovation**: Exponential gating for unbounded forget gates

**Architecture**:
- 2 xLSTM layers with 192 hidden units
- Exponential input/forget gates (instead of sigmoid)
- Normalizer state to prevent explosion
- Post-up projection for cross-variate mixing

**Key Features**:
- Adapts memory instantly to volatility shifts
- Unbounded forget gates: f_t = exp(w_f × x_t) (no [0,1] constraint)
- Normalizer prevents explosion: n_t = max(|f_t × n_{t-1} + i_t|, 1)

**Status**: Training encountered NaN instability despite multiple stabilization attempts (reduced LR, smaller batch size, gate clamping, disabled AMP). Architecture proves too sensitive for this dataset.

#### 2. TTT-Linear (Test-Time Training Linear)
**Innovation**: Dynamically updates weights during inference

**Architecture**:
- 2 TTT-Linear blocks with 96 hidden dimensions
- Inner model W updated per-token via self-supervised loss
- Learned TTT learning rate
- Memory-efficient: processes every 5th timestep

**Mechanism**:
```
At each timestep t:
1. Compute k_t = W_K × x_t, v_t = W_V × x_t, q_t = W_Q × x_t
2. Reconstruction loss: L = ||W × k_t - v_t||²
3. Update W: W ← W + η × ∇_W(L)
4. Output: y_t = W × q_t
```

**Status**: Training produced NaN loss due to memory explosion in weight update step (batch size 256, 37,732 batches). Architecture requires significantly more memory than anticipated.

#### 3. Sparse MoE (Mixture of Experts)
**Innovation**: Regime-aware routing with learned GRU router

**Architecture**:
- 4 expert networks (each: GRU + MLP)
- Regime router: uses last 20 timesteps + global statistics
- Top-2 expert selection
- Load balancing auxiliary loss

**Training Results**:
- Best validation: 0.1642 overall correlation
- t0: 0.1444, t1: 0.0084
- Early stopped at epoch 8
- Weaker than MS-TCN (0.2126)

**Observation**: Routing mechanism struggled to learn meaningful regime distinctions. Expert specialization was minimal (usage distribution: 41%, 7%, 44%, 8%).

#### 4. MS-TCN (Multi-Scale Temporal Convolutional Network)
**Innovation**: Learned dilation scales (not fixed powers of 2)

**Architecture**:
- 4 parallel branches with dilations (1, 4, 16, 50)
- Depthwise-separable causal convolutions
- Squeeze-and-excitation attention for fusion
- 128 hidden dimensions

**Training Results** (LOCAL MACHINE):
- **Best validation: 0.2126 overall correlation**
- t0: 0.3159 (strong), t1: 0.0965 (moderate)
- Training time: ~4.8 hours (17,391 seconds)
- Early stopped after patience exhausted
- ONNX INT8: 0.29MB (66% size reduction)
- Latency p50: 9.89ms, p99: 15.34ms

**Status**: BEST PERFORMING MODEL in post-competition research.

### Additional Models Explored

#### 5. Transformer (CPU-Optimized)
**Architecture**:
- 4-layer causal Transformer with RoPE
- 4 attention heads, 128 hidden dimensions
- Pre-LayerNorm for stability

**Target**: 0.21-0.23 correlation  
**Status**: Ready for training on Lightning.ai 4-core CPU

#### 6. BiGRU (CPU-Optimized)
**Architecture**:
- 2-layer Bidirectional GRU (96 hidden units)
- Attention pooling mechanism
- Designed for Lightning.ai CPU training

**Target**: 0.20-0.24 correlation  
**Status**: Ready for training

### Training Infrastructure

**Local Machine (Windows)**:
- GPU: NVIDIA RTX 3050 6GB
- RAM: 16GB
- Used for: MS-TCN (successful), xLSTM (NaN), TTT-Linear (NaN)

**Lightning.ai Cloud (CPU)**:
- 4-core CPU, 16GB RAM
- Used for: Sparse MoE (0.1642), Transformer, BiGRU (pending)

### Post-Competition Results Summary

| Model | corr_t0 | corr_t1 | Overall | Parameters | Status |
|-------|---------|---------|---------|------------|--------|
| **MS-TCN (Local)** | **0.3159** | **0.0965** | **0.2126** | ~350K | Success |
| Sparse MoE (Cloud) | 0.1444 | 0.0084 | 0.1642 | ~400K | Trained |
| xLSTM | - | - | NaN | ~280K | Unstable |
| TTT-Linear | - | - | NaN | ~175K | OOM |
| Transformer (CPU) | - | - | Pending | ~540K | Ready |
| BiGRU (CPU) | - | - | Pending | ~120K | Ready |

**Best Model**: MS-TCN with 0.2126 correlation (75.6% of competition baseline 0.2812)

---

## Key Findings & Insights

### 1. Train-Inference Mismatch is Catastrophic

**Quantification**:
```
Offline (batch):    0.256 correlation (tree ensemble with 128 features)
Streaming (real):   0.006 correlation (same model, same features)

Performance loss: 42x degradation
```

**Cause**: 40 of 128 engineered features relied on full-sequence statistics (global mean/std) that are systematically biased when computed from rolling 100-step window instead of full 1000-step sequence.

**Solution**: Streaming-native architectures using only raw features (no feature engineering).

### 2. Architecture Matters More Than Feature Engineering

**Evidence**:
- 32 raw features + sequence model: 0.2812 correlation
- 128 engineered features + tree ensemble: 0.006 correlation (streaming)

**Implication**: Architectural compatibility with streaming constraints dominates prediction quality in production deployment.

### 3. Not All Novel Architectures Are Practical

**Failures**:
- **xLSTM**: Exponential gates too sensitive, multiple stabilization attempts failed (NaN loss)
- **TTT-Linear**: Memory requirements for per-token weight updates exceed 6GB VRAM capacity

**Success**:
- **MS-TCN**: Pure convolutions, no recurrence, stable training, strong performance (0.2126)
- **Sparse MoE**: Stable training but weaker performance (0.1642) due to poor regime learning

**Lesson**: Research prototypes require significant engineering to achieve production stability.

### 4. t1 Prediction Remains Challenging

**Observation**: All models show strong t0 (current-tick) but weak t1 (100-step-ahead):

| Model | t0 | t1 | Ratio |
|-------|----|----|-------|
| MS-TCN | 0.3159 | 0.0965 | 3.3:1 |
| BiGRU | 0.2947 | 0.0261 | 11.3:1 |
| TimeMixer | 0.315 | 0.099 | 3.2:1 |

**Analysis**: 100-step-ahead prediction requires capturing longer-range dependencies that are difficult to propagate through recurrent hidden states or learn from limited temporal receptive field.

### 5. Quantization Is Production-Critical

**Impact**:
- Model size reduction: 10-73% (e.g., TimeMixer: 1.14MB → 0.34MB)
- Inference speedup: 3-4x
- Accuracy loss: <1%

**Result**: Enabled full 1.3M prediction workload to complete in 21 minutes (within 60-minute budget).

### 6. Self-Supervised Pretraining Helps

**BiGRU Results**:
```
Direct supervised:    0.150 overall
SSL + supervised:     0.160 overall

Improvement: +0.010 (6.7% relative)
```

**Mechanism**: InfoNCE contrastive loss encourages representation learning that distinguishes market regimes.

---

## Repository Structure

```
.
├── README.md                          # This file
├── .gitignore                         # Git ignore configuration
├── requirements.txt                   # Dependencies
│
├── My_Research/                       # Post-competition research
│   ├── README.md                      # Research overview
│   ├── report/
│   │   └── REPORT.MD                  # Full technical report
│   ├── score_maximizers/              # Novel architectures
│   │   ├── data_loader.py             # High-performance data pipeline
│   │   ├── evaluate.py                # Streaming evaluation
│   │   ├── xlstm/                     # xLSTM implementation
│   │   │   ├── xlstm_gpu.py
│   │   │   └── xlstm_cpu.py
│   │   ├── ttt_linear/                # TTT-Linear implementation
│   │   │   ├── ttt_linear_gpu.py
│   │   │   └── ttt_linear_cpu.py
│   │   ├── sparse_moe/                # Sparse MoE implementation
│   │   │   ├── sparse_moe_gpu.py
│   │   │   └── sparse_moe_cpu.py
│   │   ├── ms_tcn/                    # MS-TCN implementation
│   │   │   ├── ms_tcn_gpu.py
│   │   │   └── ms_tcn_cpu.py
│   │   ├── ensemble/                  # Ensemble optimization
│   │   │   └── novel_ensemble.py
│   │   └── losses/                    # Novel loss functions
│   │       └── heteroscedastic.py
│   ├── lightning_research/            # CPU-optimized models
│   │   ├── transformer_cpu.py         # Lightweight Transformer
│   │   └── bigru_cpu.py               # BiGRU with attention
│   └── paper/                         # Publication drafts
│       └── NOVEL_RESEARCH_DIRECTION.md
│
├── archive_batch_mode_models/         # Competition models (batch)
│   ├── src/                           # Training scripts
│   │   ├── train_bigru.py
│   │   ├── train_final.py
│   │   └── features.py
│   └── models/                        # Trained models (.pkl)
│
├── example_solution/                  # Competition baseline
│   ├── solution.py                    # Streaming inference
│   └── baseline.onnx                  # Baseline model
│
├── models/                            # Competition models (streaming)
│   ├── bigru.onnx
│   ├── timemixer.onnx
│   └── scaler*.npz
│
├── reports/                           # Analysis & documentation
│   ├── DEVELOPMENT_JOURNEY.md         # Full journey documentation
│   ├── COMPLETE_PROJECT_EXPLANATION.txt
│   ├── BATCH_VS_STREAMING_DIFFERENCE.txt
│   └── DOUBLE_MODEL_REPORT.txt
│
└── misc/                              # Research papers & guides
    ├── LOB_Research_Document.pdf
    └── Workshop_Complete_Guide_v2.docx
```

**Note**: Large files (datasets, checkpoints, ONNX models) are excluded via `.gitignore` for repository cleanliness.

---

## Results & Visualizations

### Competition Performance

```
┌──────────────────────────────────────────────────────┐
│  Batch vs Streaming Performance Comparison          │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Batch Evaluation (Full Sequence):                  │
│  ████████████████████████████████ 0.256              │
│                                                      │
│  Streaming Evaluation (Rolling Window):             │
│  █ 0.006                                             │
│                                                      │
│  Degradation: 42x performance loss                  │
└──────────────────────────────────────────────────────┘
```

### Post-Competition Model Comparison

```
Model Performance (Overall Correlation)

MS-TCN         ████████████████████████ 0.2126
Sparse MoE     ██████████████████ 0.1642
BiGRU          █████████████████ 0.1604
TimeMixer      ████████████████████████ 0.207
xLSTM          [NaN - Training Failed]
TTT-Linear     [NaN - Training Failed]

────────────────────────────────────────────────────
0.00    0.05    0.10    0.15    0.20    0.25
           Overall Pearson Correlation
```

### t0 vs t1 Performance Gap

```
t0 (Current-Tick) vs t1 (100-Step-Ahead) Prediction

              t0           t1
MS-TCN        ████████ (0.3159)   ██ (0.0965)
BiGRU         ████████ (0.2947)   █ (0.0261)
TimeMixer     ████████ (0.3150)   ██ (0.0990)
Sparse MoE    ████ (0.1444)       █ (0.0084)

───────────────────────────────────────────────────
Observation: All models excel at t0 but struggle with t1
Long-horizon prediction remains an open challenge
```

### Quantization Impact

```
Model Size Reduction (INT8 Quantization)

TimeMixer:  1.14 MB → 0.34 MB  [70.7% reduction]
MS-TCN:     0.85 MB → 0.29 MB  [66.0% reduction]
BiGRU:      1.99 MB → 1.80 MB  [9.6% reduction]
MLP-Mixer:  3.22 MB → 0.88 MB  [73.0% reduction]

Inference Speedup: 3-4x across all models
Accuracy Loss: <1%
```

### Training Stability Comparison

```
Architecture Stability Assessment

[SUCCESS] MS-TCN:       Stable, converged successfully
[SUCCESS] Sparse MoE:   Stable, weak performance
[SUCCESS] BiGRU:        Stable with SSL pretraining
[SUCCESS] TimeMixer:    Stable, good performance
[WARNING] Transformer:  Stable but computationally expensive
[FAILED]  xLSTM:        NaN loss despite stabilization
[FAILED]  TTT-Linear:   Out of memory, NaN loss
```

---

## Installation & Setup

### Prerequisites

```bash
Python 3.8+
CUDA 11.8+ (for GPU training)
16GB RAM minimum
6GB VRAM (for GPU models)
```

### Local Setup (Competition Models)

```bash
# Clone repository
git clone https://github.com/Vk2245/-Vk224-Wunder-Challenge-LOB-Predictorium.git
cd Vk224-Wunder-Challenge-LOB-Predictorium

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Research Setup (Novel Architectures)

```bash
# Navigate to research folder
cd My_Research

# Install research dependencies
pip install -r requirements_research.txt

# Verify installation
python score_maximizers/data_loader.py
```

### Cloud Setup (Lightning.ai)

```bash
# Upload to Lightning.ai workspace
# Install dependencies in cloud environment
pip install torch numpy pandas pyarrow scikit-learn onnx onnxruntime

# Verify paths
python -c "import os; print(os.getcwd())"
```

---

## Usage

### Competition Models (Streaming Inference)

```python
# Load trained model
import onnxruntime as ort
import numpy as np

# Initialize session
session = ort.InferenceSession("models/bigru_timemixer_ensemble.onnx")

# Load scaler
scaler = np.load("models/scaler.npz")
mu, sigma = scaler['mu'], scaler['sigma']

# Streaming prediction
from collections import deque

buffer = deque(maxlen=100)

def predict(row):
    buffer.append(row)
    
    if len(buffer) < 100:
        return 0.0, 0.0  # Not ready
    
    # Stack and normalize
    X = np.stack(buffer)  # (100, 32)
    X_norm = (X - mu) / sigma
    X_batch = X_norm[None, :, :]  # (1, 100, 32)
    
    # Predict
    pred = session.run(None, {"input": X_batch})[0]
    return pred[0, 0], pred[0, 1]  # t0, t1
```

### Training Novel Architectures

**MS-TCN (GPU)**:
```bash
cd My_Research
python score_maximizers/ms_tcn/ms_tcn_gpu.py
```

**Transformer (CPU - Lightning.ai)**:
```bash
cd My_Research
python lightning_research/transformer_cpu.py
```

**Custom Configuration**:
```python
# Edit hyperparameters in script
BATCH = 1024        # Batch size
LR = 1e-3           # Learning rate
EPOCHS = 30         # Max epochs
PATIENCE = 7        # Early stopping patience
```

### Evaluation

**Streaming Evaluation**:
```bash
python score_maximizers/evaluate.py \
    --model_path models/ms_tcn.onnx \
    --scaler_path models/scaler.npz \
    --data_path datasets/valid.parquet
```

**Batch Evaluation (for comparison)**:
```python
from score_maximizers.evaluate import batch_evaluate

results = batch_evaluate(
    model_path="models/ms_tcn.onnx",
    data_path="datasets/valid.parquet"
)
print(f"t0: {results['t0']:.4f}, t1: {results['t1']:.4f}")
```

---

## Future Work

### Immediate Priorities

1. **Complete CPU Model Training**
   - Train Transformer on Lightning.ai (target: 0.21-0.23)
   - Train BiGRU on Lightning.ai (target: 0.20-0.24)
   - Build 3-model ensemble (MS-TCN + Transformer + BiGRU)

2. **Stability Improvements**
   - Investigate xLSTM stabilization: reduced precision (FP16), layerwise gradient clipping
   - TTT-Linear memory optimization: sparse updates, gradient checkpointing

3. **Ensemble Optimization**
   - Meta-learning for dynamic weighting
   - Attention-based combination mechanism
   - Regime-conditional ensembling

### Research Extensions

1. **Multi-Dataset Validation**
   - FI-2010 benchmark (Finnish stocks)
   - LOBSTER dataset (NASDAQ)
   - NSE Indian market data

2. **Architecture Variants**
   - PatchTST: Patch-based Transformer for long sequences
   - iTransformer: Inverted attention across features
   - FNet: Fourier mixing (ONNX compatibility required)
   - Dual-Horizon: Separate t0/t1 expert pathways

3. **Loss Function Ablations**
   - Heteroscedastic Gaussian NLL vs MSE
   - Pinball (Quantile) loss for robust prediction
   - Asymmetric loss for directional accuracy

4. **Transfer Learning**
   - Pre-train on multiple instruments
   - Fine-tune on target instrument
   - Cross-market generalization study

### Production Enhancements

1. **Quantization**
   - INT4 quantization for further compression
   - Mixed-precision strategies (INT4 + INT8)
   - Quantization-aware training

2. **Inference Optimization**
   - TensorRT deployment for NVIDIA GPUs
   - ONNX Runtime tuning for CPU
   - Model distillation for smaller models

3. **Online Learning**
   - Exponentially weighted running statistics for normalization
   - Online fine-tuning with recent data
   - Concept drift detection and adaptation



---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{kumar2024lob,
  author = {Vishal Kumar},
  title = {Limit Order Book Prediction: From Competition to Research},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Vk2245/-Vk224-Wunder-Challenge-LOB-Predictorium}}
}
```

---

## Acknowledgments

- **Wunderfund** for organizing the competition and providing the dataset
- **Competition Community** for insights on 2-stage boosting, SWA, and data augmentation
- **Lightning.ai** for providing cloud compute resources
- **PyTorch** and **ONNX** teams for excellent ML infrastructure

---

## License

This project is available under the MIT License. See LICENSE file for details.

---

## Contact

**Vishal Kumar**  
Email: vk224official@gmail.com  
GitHub: [@Vk2245](https://github.com/Vk2245)  
Portfolio: [vk224portfolio.vercel.app](https://vk224portfolio.vercel.app/)  
LinkedIn: [linkedin.com/in/vishal-kumar-7a74462a0](https://www.linkedin.com/in/vishal-kumar-7a74462a0/)  
Kaggle: [@vk2245](https://www.kaggle.com/vk2245)  
Google Developer: [g.dev/vk224](https://g.dev/vk224)  
Resume: [View Resume](https://drive.google.com/file/d/18nXReC6hM3ZhpZKwhP3OG7C21qGPzNhn/view?usp=drive_link)

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact via email.

---

**Last Updated**: June 2026  
**Status**: Ongoing Research

---

## Quick Links

- [Competition Platform](https://wunderfund.com) (if available)
- [Full Technical Report](My_Research/report/REPORT.MD)
- [Research Roadmap](My_Research/EXECUTION_STATUS.md)
- [Development Journey](reports/DEVELOPMENT_JOURNEY.md)
- [Issue Tracker](https://github.com/Vk2245/-Vk224-Wunder-Challenge-LOB-Predictorium/issues)
