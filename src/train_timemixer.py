"""
train_timemixer.py — TimeMixer with multiscale decomposition for LOB
=====================================================================
Based on TSMixer/TimeMixer research (Google 2023, ICLR 2024)
- Multiscale decomposition (100 -> 50 -> 25 steps)
- Seasonal/Trend separation via moving average
- Pure MLP mixing (no RNN, no attention)
- Fast training, excellent for ensemble with BiGRU

PLACE AT:  src/train_timemixer.py
RUN:       python src/train_timemixer.py
"""
import os, sys, math, random, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR    = os.path.join(ROOT, "checkpoints");  os.makedirs(CKPT_DIR, exist_ok=True)
MODEL_DIR   = os.path.join(ROOT, "models");        os.makedirs(MODEL_DIR, exist_ok=True)
ONNX_PATH   = os.path.join(MODEL_DIR, "timemixer.onnx")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_tm.npz")
BEST_CKPT   = os.path.join(CKPT_DIR,  "best_tm.pt")

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    BATCH      = 2048  # Increased from 1024 for better GPU utilization
    NUM_WORKERS = 0
    PIN_MEM    = True
    USE_AMP    = True
else:
    BATCH      = 256
    NUM_WORKERS = 0
    PIN_MEM    = False
    USE_AMP    = False

SEQ_LEN     = 100
SPATIAL_DIM = 32
HIDDEN      = 256
LR          = 1e-3
PATIENCE    = 7

t0 = time.time()
def log(s): print(f"[{time.time()-t0:7.1f}s] {s}", flush=True)

# ── TimeMixer Model ───────────────────────────────────────────────────────────
class MovingAvg(nn.Module):
    """Moving average for trend extraction"""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    
    def forward(self, x):
        # x: (B, L, C)
        # Manual padding to preserve length
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_trend = self.avg(x_padded.transpose(1, 2)).transpose(1, 2)
        # Trim to original length
        return x_trend[:, :x.size(1), :]

class SeriesDecomp(nn.Module):
    """Decompose into seasonal + trend"""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)
    
    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

class MixingBlock(nn.Module):
    """MLP mixing along time dimension"""
    def __init__(self, seq_len, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (B, L, C)
        x_norm = self.norm(x)
        x_mixed = self.mlp(x_norm.transpose(1, 2)).transpose(1, 2)
        return x + x_mixed

class TimeMixer(nn.Module):
    """
    TimeMixer: Multiscale MLP-based time series model
    - Decomposes into 3 scales: 100, 50, 25
    - Separates seasonal/trend at each scale
    - Mixes with MLP blocks
    - Fuses and predicts
    """
    def __init__(self, input_dim=32, seq_len=100, hidden=256, output_dim=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden)
        
        # Decomposition modules for each scale
        self.decomp_100 = SeriesDecomp(kernel_size=25)
        self.decomp_50  = SeriesDecomp(kernel_size=13)
        self.decomp_25  = SeriesDecomp(kernel_size=7)
        
        # Mixing blocks for each scale
        self.mix_100 = MixingBlock(100, hidden, dropout)
        self.mix_50  = MixingBlock(50, hidden, dropout)
        self.mix_25  = MixingBlock(25, hidden, dropout)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output head
        self.head = nn.Linear(hidden // 2, output_dim)
    
    def forward(self, x):
        # x: (B, 100, 32)
        B = x.size(0)
        
        # Project to hidden
        x = self.input_proj(x)  # (B, 100, hidden)
        
        # Scale 1: 100 steps
        seasonal_100, trend_100 = self.decomp_100(x)
        mixed_100 = self.mix_100(seasonal_100 + trend_100)
        feat_100 = mixed_100[:, -1, :]  # Take last step
        
        # Scale 2: 50 steps (downsample by 2)
        x_50 = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        seasonal_50, trend_50 = self.decomp_50(x_50)
        mixed_50 = self.mix_50(seasonal_50 + trend_50)
        feat_50 = mixed_50[:, -1, :]
        
        # Scale 3: 25 steps (downsample by 4)
        x_25 = F.avg_pool1d(x.transpose(1, 2), kernel_size=4, stride=4).transpose(1, 2)
        seasonal_25, trend_25 = self.decomp_25(x_25)
        mixed_25 = self.mix_25(seasonal_25 + trend_25)
        feat_25 = mixed_25[:, -1, :]
        
        # Fuse all scales
        fused = torch.cat([feat_100, feat_50, feat_25], dim=1)  # (B, hidden*3)
        fused = self.fusion(fused)  # (B, hidden//2)
        
        # Predict
        out = self.head(fused)  # (B, 2)
        return out

# ── Dataset ───────────────────────────────────────────────────────────────────
class FTDataset(Dataset):
    def __init__(self, df, mu, sig, need_pred_only=True):
        self.df = df
        self.mu = mu
        self.sig = sig
        self.need_pred_only = need_pred_only
        
        # Build index
        self.indices = []
        for seq_ix, grp in df.groupby("seq_ix"):
            steps = grp["step_in_seq"].values
            for i, step in enumerate(steps):
                if step >= 99:  # Need 100 steps
                    if need_pred_only:
                        if grp.iloc[i]["need_prediction"]:
                            self.indices.append((seq_ix, i))
                    else:
                        self.indices.append((seq_ix, i))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        seq_ix, end_idx = self.indices[idx]
        grp = self.df[self.df["seq_ix"] == seq_ix]
        
        start_idx = max(0, end_idx - 99)
        window = grp.iloc[start_idx:end_idx+1]
        
        # Get features
        feat_cols = ([f"p{i}" for i in range(12)] + [f"v{i}" for i in range(12)] +
                     [f"dp{i}" for i in range(4)] + [f"dv{i}" for i in range(4)])
        X = window[feat_cols].values.astype(np.float32)
        
        # Pad if needed
        if len(X) < 100:
            pad = np.zeros((100 - len(X), 32), dtype=np.float32)
            X = np.vstack([pad, X])
        
        # Normalize
        X = (X - self.mu) / self.sig
        
        # Get targets
        y = window.iloc[-1][["t0", "t1"]].values.astype(np.float32)
        
        return torch.from_numpy(X), torch.from_numpy(y)

# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, scaler):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        
        if USE_AMP:
            with torch.amp.autocast(device_type='cuda'):
                pred = model(x)
                loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def eval_model(model, loader):
    model.eval()
    p0, p1, y0, y1 = [], [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()
            y = y.numpy()
            
            p0.extend(pred[:, 0])
            p1.extend(pred[:, 1])
            y0.extend(y[:, 0])
            y1.extend(y[:, 1])
    
    def pearson(y, p):
        r = np.corrcoef(y, p)[0, 1]
        return 0.0 if np.isnan(r) else float(r)
    
    r0 = pearson(y0, p0)
    r1 = pearson(y1, p1)
    return r0, r1, (r0 + r1) / 2

def main():
    log(f"Device: {DEVICE}  Batch={BATCH}  Workers={NUM_WORKERS}  AMP={USE_AMP}")
    
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        log(f"GPU: {props.name}  VRAM: {vram:.1f}GB")
    
    # Load data
    log("Loading train.parquet ...")
    train_df = pd.read_parquet(os.path.join(ROOT, "datasets", "train.parquet"))
    log(f"Train: {len(train_df):,} rows  |  {train_df['seq_ix'].nunique():,} sequences")
    
    # Fit scaler
    log("Fitting scaler ...")
    feat_cols = ([f"p{i}" for i in range(12)] + [f"v{i}" for i in range(12)] +
                 [f"dp{i}" for i in range(4)] + [f"dv{i}" for i in range(4)])
    X = train_df[feat_cols].to_numpy(np.float32)
    mu = X.mean(0).astype(np.float32)
    sig = (X.std(0) + 1e-8).astype(np.float32)
    
    # Build datasets
    log("Building train dataset ...")
    train_ds = FTDataset(train_df, mu, sig, need_pred_only=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEM)
    log(f"  Train windows: {len(train_ds):,}")
    
    log("Loading valid.parquet ...")
    valid_df = pd.read_parquet(os.path.join(ROOT, "datasets", "valid.parquet"))
    log("Building valid dataset ...")
    valid_ds = FTDataset(valid_df, mu, sig, need_pred_only=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEM)
    log(f"  Valid windows: {len(valid_ds):,}")
    
    # Model
    model = TimeMixer(SPATIAL_DIM, SEQ_LEN, HIDDEN, 2, dropout=0.1).to(DEVICE)
    
    if sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log("torch.compile() enabled")
        except:
            log("torch.compile() skipped")
    else:
        log("torch.compile() skipped (Windows)")
    
    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    
    # Training
    log(f"Training 25 epochs ...")
    best_score = -999
    no_improve = 0
    
    for ep in range(1, 26):
        ep_start = time.time()
        loss = train_epoch(model, train_loader, opt, scaler)
        r0, r1, ov = eval_model(model, valid_loader)
        ep_time = time.time() - ep_start
        
        vram_str = ""
        if DEVICE == "cuda":
            vram_used = torch.cuda.max_memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_str = f"  VRAM {vram_used:.1f}/{vram_total:.1f}GB"
        
        log(f"  ep{ep:02d}/25  loss={loss:.4f}  t0={r0:.4f}  t1={r1:.4f}  ov={ov:.4f}  {int(ep_time)}s{vram_str}")
        
        if ov > best_score:
            best_score = ov
            torch.save(model.state_dict(), BEST_CKPT)
            log(f"  *** BEST={best_score:.4f} saved ***")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log(f"  Early stop ({no_improve} no-improve)")
                break
    
    log(f"Training done. Best = {best_score:.4f}")
    
    # Export ONNX
    log("Exporting ONNX ...")
    model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    
    dummy = torch.randn(1, SEQ_LEN, SPATIAL_DIM).to(DEVICE)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14
    )
    
    onnx_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
    log(f"ONNX: {onnx_mb:.2f}MB  Scaler saved.")
    
    # Save scaler
    np.savez(SCALER_PATH, mu=mu, sig=sig)
    
    # Benchmark
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    
    dummy_np = np.random.randn(1, SEQ_LEN, SPATIAL_DIM).astype(np.float32)
    t_start = time.time()
    for _ in range(1000):
        sess.run([out_name], {inp_name: dummy_np})
    t_per_call = (time.time() - t_start) / 1000 * 1000
    est_min = t_per_call * 1_400_000 / 1000 / 60
    
    log(f"ONNX: {t_per_call:.2f}ms/call  →  {est_min:.1f} min for 1.4M preds")
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best Pearson (valid) : {best_score:.4f}")
    print(f"ONNX                 : {ONNX_PATH}")
    print(f"Scaler               : {SCALER_PATH}")
    if best_score >= 0.27:
        print("GOOD — ready for ensemble")
    elif best_score >= 0.20:
        print("OK — can ensemble")
    else:
        print("LOW — check hyperparameters")
    print("=" * 60)
    print("NEXT: Create ensemble solution")

if __name__ == "__main__":
    main()
