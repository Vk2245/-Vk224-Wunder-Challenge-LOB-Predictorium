"""
train_bigru.py  —  STLOB, Windows-safe, GPU-maximized
======================================================
Fixes vs previous version:
  - Windows multiprocessing: all training inside if __name__=='__main__'
  - GradScaler deprecation: torch.amp.GradScaler('cuda', ...)
  - num_workers set to 0 on Windows (spawn overhead > benefit for small batches)
    → use prefetch thread instead via prefetch_generator
  - AMP autocast: torch.amp.autocast(device_type='cuda')
  - torch.compile wrapped in try/except (silently skips if unsupported)
  - persistent_workers only when num_workers > 0

PLACE AT:  src/train_bigru.py
RUN:       python src/train_bigru.py
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
ONNX_PATH   = os.path.join(MODEL_DIR, "bigru.onnx")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.npz")
BEST_CKPT   = os.path.join(CKPT_DIR,  "best.pt")
SSL_CKPT    = os.path.join(CKPT_DIR,  "ssl.pt")

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    BATCH      = 2048  # Increased for better GPU utilization
    # Windows: num_workers > 0 causes spawn issues unless inside __main__
    # We handle this by setting num_workers=0 (still fast with big batch+AMP)
    NUM_WORKERS = 0
    PIN_MEM    = True
    USE_AMP    = True
else:
    BATCH      = 256
    NUM_WORKERS = 0
    PIN_MEM    = False
    USE_AMP    = False

SEQ_LEN     = 100
HIDDEN      = 128
SPATIAL_DIM = 64
EPOCHS_SSL  = 2
EPOCHS_FT   = 25
LR          = 1e-3
MAX_SSL     = 200_000   # reduced to speed up SSL on laptop GPU
MAX_FT_TR   = 600_000
SEED        = 42
F_DIM       = 32

FEAT_COLS = (
    [f"p{i}"  for i in range(12)] +
    [f"v{i}"  for i in range(12)] +
    [f"dp{i}" for i in range(4)]  +
    [f"dv{i}" for i in range(4)]
)

T0s = time.time()
def log(m): print(f"[{time.time()-T0s:7.1f}s] {m}", flush=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
class SpatialEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.bid_enc   = nn.Linear(12, out_dim // 2)
        self.ask_enc   = nn.Linear(12, out_dim // 2)
        self.trade_enc = nn.Linear(8,  out_dim // 4)
        self.cross     = nn.Linear(out_dim, out_dim // 2)
        self.out       = nn.Sequential(
            nn.Linear(out_dim // 2 + out_dim // 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        hb = F.gelu(self.bid_enc(x[..., :12]))
        ha = F.gelu(self.ask_enc(x[..., 12:24]))
        ht = F.gelu(self.trade_enc(x[..., 24:32]))
        hx = F.gelu(self.cross(torch.cat([hb, ha], dim=-1)))
        return self.out(torch.cat([hx, ht], dim=-1))


class STLOB(nn.Module):
    def __init__(self, spatial_dim=64, hidden=128, dropout=0.1):
        super().__init__()
        self.hidden  = hidden
        self.spatial = SpatialEncoder(spatial_dim)
        self.gru     = nn.GRU(
            spatial_dim, hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.q    = nn.Linear(hidden, hidden)
        self.k    = nn.Linear(hidden, hidden)
        self.v    = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        h, _ = self.gru(self.spatial(x))
        hl   = h[:, -1, :]
        h1, h2 = hl[:, :self.hidden], hl[:, self.hidden:]
        gate = torch.sigmoid(
            (self.q(h1) * self.k(h2)).sum(-1, keepdim=True) / math.sqrt(self.hidden)
        )
        return self.head(torch.cat([h1, gate * self.v(h2)], dim=-1))


class Projector(nn.Module):
    def __init__(self, hidden=128, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# DATASETS  — lazy pointer-based, no giant pre-allocation
# ══════════════════════════════════════════════════════════════════════════════
class SSLDataset(Dataset):
    def __init__(self, df, mu, sig):
        self.seqs  = []
        self.index = []
        for _, g in df.groupby("seq_ix", sort=False):
            x  = (g[FEAT_COLS].to_numpy(np.float32) - mu) / sig
            si = len(self.seqs)
            self.seqs.append(x)
            for e in range(SEQ_LEN - 1, len(g)):
                self.index.append((si, e))
        if len(self.index) > MAX_SSL:
            chosen = np.random.choice(len(self.index), MAX_SSL, replace=False)
            self.index = [self.index[i] for i in chosen]
        log(f"  SSL windows: {len(self.index):,} (lazy)")

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        si, e = self.index[i]
        x  = self.seqs[si][e - SEQ_LEN + 1: e + 1].copy()
        x1 = x + (np.random.randn(*x.shape) * 0.04).astype(np.float32)
        x2 = x.copy()
        ts = np.random.randint(0, SEQ_LEN // 2)
        x2[ts: ts + np.random.randint(5, 25)] = 0.0
        return torch.from_numpy(x1), torch.from_numpy(x2)


class FTDataset(Dataset):
    def __init__(self, df, mu, sig, max_samples=None):
        self.seqs  = []
        self.index = []
        for _, g in df.groupby("seq_ix", sort=False):
            x    = (g[FEAT_COLS].to_numpy(np.float32) - mu) / sig
            y    = g[["t0", "t1"]].to_numpy(np.float32)
            need = g["need_prediction"].to_numpy(bool)
            si   = len(self.seqs)
            self.seqs.append(x)
            for e in range(SEQ_LEN - 1, len(g)):
                if need[e]:
                    self.index.append((si, e, y[e]))
        if max_samples and len(self.index) > max_samples:
            chosen = np.random.choice(len(self.index), max_samples, replace=False)
            self.index = [self.index[i] for i in chosen]
        log(f"  FT windows: {len(self.index):,} (lazy)")

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        si, e, y = self.index[i]
        x = self.seqs[si][e - SEQ_LEN + 1: e + 1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loader(ds, shuffle, batch=None):
    b = batch or BATCH
    return DataLoader(
        ds, batch_size=b, shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=False,   # must be False when num_workers=0
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fit_scaler(df):
    s = df.sample(min(1_000_000, len(df)), random_state=SEED)
    X = s[FEAT_COLS].to_numpy(np.float32)
    return X.mean(0).astype(np.float32), (X.std(0) + 1e-8).astype(np.float32)

def info_nce(z1, z2, temp=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2])
    s = torch.mm(z, z.t()) / temp
    # Use -65000 instead of -1e9 for float16 compatibility
    s.fill_diagonal_(-65000.0)
    lbl = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(s, lbl)

def pearson(y, p):
    r = np.corrcoef(y, p)[0, 1]
    return 0.0 if np.isnan(r) else float(r)

def gpu_info():
    if DEVICE == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        tot  = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"VRAM {used:.1f}/{tot:.1f}GB"
    return "CPU"

def get_state(m):
    # handles both compiled and uncompiled models
    sd = getattr(m, "_orig_mod", m).state_dict()
    return {k: v.cpu().clone() for k, v in sd.items()}

def load_state(m, sd):
    getattr(m, "_orig_mod", m).load_state_dict(sd)

def make_amp_scaler():
    if USE_AMP:
        return torch.amp.GradScaler("cuda")
    # dummy scaler that does nothing
    class NoopScaler:
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass
        def unscale_(self, opt): pass
    return NoopScaler()

def amp_context():
    if USE_AMP:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  — must be inside if __name__ == '__main__' on Windows
# ══════════════════════════════════════════════════════════════════════════════
def main():
    seed_all(SEED)
    log(f"Device: {DEVICE}  Batch={BATCH}  Workers={NUM_WORKERS}  AMP={USE_AMP}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f}GB")

    # ── STEP 1: Load & scale ──────────────────────────────────────────────────
    log("Loading train.parquet ...")
    train_df = pd.read_parquet(os.path.join(ROOT, "datasets", "train.parquet"))
    log(f"Train: {len(train_df):,} rows  |  {train_df['seq_ix'].nunique():,} sequences")
    log("Fitting scaler ...")
    mu, sig = fit_scaler(train_df)

    # ── STEP 2: SSL pretrain ──────────────────────────────────────────────────
    log("Building SSL dataset ...")
    ssl_ds = SSLDataset(train_df, mu, sig)
    ssl_dl = make_loader(ssl_ds, shuffle=True)

    model     = STLOB(SPATIAL_DIM, HIDDEN, dropout=0.1).to(DEVICE)
    projector = Projector(HIDDEN, 64).to(DEVICE)

    # Disable torch.compile on Windows (Triton not available)
    if sys.platform != "win32":
        try:
            model     = torch.compile(model,     mode="reduce-overhead")
            projector = torch.compile(projector, mode="reduce-overhead")
            log("torch.compile() enabled")
        except Exception as ex:
            log(f"torch.compile() skipped: {ex}")
    else:
        log("torch.compile() skipped (Windows)")

    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt_ssl    = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=LR, weight_decay=1e-4
    )
    amp_scaler = make_amp_scaler()

    log(f"\nSSL pretraining {EPOCHS_SSL} epochs ...")
    for ep in range(EPOCHS_SSL):
        model.train(); projector.train()
        tot = 0; n = 0; t0 = time.time()
        for x1, x2 in ssl_dl:
            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)
            with amp_context():
                s1 = model.spatial(x1); h1 = model.gru(s1)[0][:, -1, :]
                s2 = model.spatial(x2); h2 = model.gru(s2)[0][:, -1, :]
                loss = info_nce(projector(h1), projector(h2))
            opt_ssl.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(opt_ssl)
            amp_scaler.update()
            tot += loss.item(); n += 1
        log(f"  SSL ep{ep+1}/{EPOCHS_SSL}  loss={tot/n:.4f}  {time.time()-t0:.0f}s  {gpu_info()}")

    torch.save({"model": get_state(model)}, SSL_CKPT)
    log("SSL checkpoint saved.")
    del ssl_ds, ssl_dl, projector; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # ── STEP 3: Fine-tune ─────────────────────────────────────────────────────
    log("\nBuilding FT train dataset ...")
    ft_tr = FTDataset(train_df, mu, sig, max_samples=MAX_FT_TR)
    del train_df; gc.collect()

    log("Loading valid.parquet ...")
    valid_df = pd.read_parquet(os.path.join(ROOT, "datasets", "valid.parquet"))
    log("Building FT valid dataset ...")
    ft_vl = FTDataset(valid_df, mu, sig)
    del valid_df; gc.collect()

    tr_dl = make_loader(ft_tr, shuffle=True)
    vl_dl = make_loader(ft_vl, shuffle=False, batch=BATCH * 2)

    load_state(model, torch.load(SSL_CKPT, map_location=DEVICE, weights_only=False)["model"])
    log("SSL weights loaded.")

    opt_ft    = torch.optim.AdamW(model.parameters(), lr=LR * 0.5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt_ft, max_lr=LR * 0.5,
        steps_per_epoch=len(tr_dl),
        epochs=EPOCHS_FT, pct_start=0.1,
    )
    criterion  = nn.HuberLoss(delta=0.5)
    amp_scaler = make_amp_scaler()

    best_score = -999.0; best_state = None; no_improve = 0

    log(f"\nFine-tuning {EPOCHS_FT} epochs ...")
    for ep in range(EPOCHS_FT):
        model.train()
        tr_loss = 0; n = 0; t0 = time.time()
        for xb, yb in tr_dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            with amp_context():
                loss = criterion(model(xb), yb)
            opt_ft.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt_ft)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(opt_ft)
            amp_scaler.update()
            scheduler.step()
            tr_loss += loss.item(); n += 1

        model.eval()
        Ps, Ys = [], []
        with torch.no_grad():
            for xb, yb in vl_dl:
                with amp_context():
                    p = model(xb.to(DEVICE, non_blocking=True))
                Ps.append(p.float().cpu().numpy())
                Ys.append(yb.numpy())
        P = np.concatenate(Ps); Y = np.concatenate(Ys)
        r0 = pearson(Y[:, 0], P[:, 0])
        r1 = pearson(Y[:, 1], P[:, 1])
        ov = (r0 + r1) / 2
        log(f"  ep{ep+1:02d}/{EPOCHS_FT}  loss={tr_loss/n:.4f}  t0={r0:.4f}  t1={r1:.4f}  ov={ov:.4f}  {time.time()-t0:.0f}s  {gpu_info()}")

        if ov > best_score:
            best_score = ov
            best_state = get_state(model)
            torch.save({
                "model_state": best_state, "mu": mu, "sig": sig,
                "hidden": HIDDEN, "spatial_dim": SPATIAL_DIM,
                "seq_len": SEQ_LEN, "score": best_score
            }, BEST_CKPT)
            log(f"  *** BEST={best_score:.4f} saved ***")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 7:
                log("  Early stop (7 no-improve)")
                break

    log(f"\nTraining done. Best = {best_score:.4f}")

    # ── STEP 4: Export ONNX ───────────────────────────────────────────────────
    log("Exporting ONNX ...")
    export_model = STLOB(SPATIAL_DIM, HIDDEN, dropout=0.0).cpu()
    export_model.load_state_dict(best_state)
    export_model.eval()

    dummy = torch.zeros(1, SEQ_LEN, F_DIM)
    torch.onnx.export(
        export_model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["output"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    np.savez(SCALER_PATH, mu=mu, sig=sig)
    log(f"ONNX: {os.path.getsize(ONNX_PATH)/1e6:.2f}MB  Scaler saved.")

    # ── STEP 5: Speed check ───────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        sess     = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, SEQ_LEN, F_DIM), dtype=np.float32)
        for _ in range(10): sess.run(None, {"input": dummy_np})
        t0 = time.time()
        for _ in range(500): sess.run(None, {"input": dummy_np})
        ms = (time.time() - t0) / 500 * 1000
        log(f"ONNX: {ms:.2f}ms/call  →  {ms*1_400_000/1000/60:.1f} min for 1.4M preds")
    except Exception as e:
        log(f"ONNX speed check skipped: {e}")

    print("\n" + "="*60)
    print(f"  TRAINING COMPLETE")
    print(f"  Best Pearson (valid) : {best_score:.4f}")
    print(f"  ONNX                 : models/bigru.onnx")
    print(f"  Scaler               : models/scaler.npz")
    print()
    if   best_score >= 0.33: print("  EXCELLENT — exceeds target. SUBMIT.")
    elif best_score >= 0.27: print("  GOOD — run test_solution.py then submit.")
    elif best_score >= 0.20: print("  OK — below target, consider more epochs.")
    else:                     print("  LOW — check data paths and try again.")
    print("="*60)
    print("NEXT: python test_solution.py")


if __name__ == "__main__":
    main()