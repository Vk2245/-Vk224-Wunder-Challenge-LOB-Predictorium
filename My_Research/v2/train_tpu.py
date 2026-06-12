"""
train_tpu.py — TPU-optimized training harness
================================================
Designed for Kaggle TPU v3-8 (single-core mode).

Key differences from GPU engine:
  - torch_xla for device management
  - bfloat16 natively (no GradScaler needed)
  - xm.optimizer_step() for proper XLA graph materialization
  - xm.mark_step() at training boundaries
  - No torch.compile (XLA IS the compiler)
  - ParallelLoader for async host→device transfer

Usage on Kaggle:
  !python train_tpu.py --model transformer_timemixer
  !python train_tpu.py --model triple_fusion
  !python train_tpu.py --model wavenet_dense
"""

import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── XLA imports ───────────────────────────────────────────────────────────────
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from config import CKPT_DIR, ONNX_DIR, SCALER_DIR, SEQ_LEN, N_FEATURES, N_TARGETS
from data_loader import build_loaders
from models import MODEL_REGISTRY, get_model, get_export_wrapper

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(ONNX_DIR, exist_ok=True)

t0 = time.time()
def log(s): print(f"[{time.time()-t0:7.1f}s] {s}", flush=True)


# ── TPU Training Loop ─────────────────────────────────────────────────────────
def train_one_epoch_tpu(model, loader, optimizer, device, epoch_num, use_bf16=True):
    """Single epoch training on TPU with bfloat16."""
    model.train()
    total_loss = 0.0
    n = 0
    nan_count = 0
    batch_start = time.time()
    num_batches = len(loader)

    optimizer.zero_grad()

    for batch_idx, (x, y) in enumerate(loader, 1):
        x = x.to(device)
        y = y.to(device)

        if use_bf16:
            x = x.to(torch.bfloat16)

        output = model(x)
        if isinstance(output, tuple):
            pred, aux_loss = output[0], output[1]
        else:
            pred, aux_loss = output, None

        # Cast pred back to float32 for loss computation
        pred = pred.float()
        loss = F.mse_loss(pred, y)

        if aux_loss is not None:
            loss = loss + 0.01 * aux_loss.float()

        # NaN guard
        if torch.isnan(loss):
            nan_count += 1
            if nan_count <= 3:
                log(f"    WARN: NaN loss at batch {batch_idx}, skipping")
            xm.mark_step()
            continue

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # XLA-aware optimizer step
        xm.optimizer_step(optimizer)
        optimizer.zero_grad()

        batch_loss = loss.item()
        total_loss += batch_loss * x.size(0)
        n += x.size(0)

        # Progress
        if batch_idx % 200 == 0 or batch_idx == num_batches:
            elapsed = time.time() - batch_start
            bps = batch_idx / elapsed if elapsed > 0 else 0
            eta_min = int((num_batches - batch_idx) / bps / 60) if bps > 0 else 0
            print(f"    Ep{epoch_num} | {batch_idx}/{num_batches} | "
                  f"Loss: {batch_loss:.4f} | {bps:.1f} b/s | "
                  f"ETA: {eta_min}m" + (f" | NaN: {nan_count}" if nan_count else ""),
                  flush=True)

    return total_loss / n if n > 0 else float("nan")


# ── TPU Evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_tpu(model, loader, device):
    """Evaluate on TPU, gather predictions to CPU."""
    model.eval()
    p0, p1, y0, y1 = [], [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output
        pred = pred.float().cpu().numpy()
        y_np = y.cpu().numpy()
        p0.extend(pred[:, 0])
        p1.extend(pred[:, 1])
        y0.extend(y_np[:, 0])
        y1.extend(y_np[:, 1])

    r0 = float(np.corrcoef(y0, p0)[0, 1])
    r1 = float(np.corrcoef(y1, p1)[0, 1])
    r0 = 0.0 if np.isnan(r0) else r0
    r1 = 0.0 if np.isnan(r1) else r1
    return {"corr_t0": r0, "corr_t1": r1, "overall": (r0 + r1) / 2}


# ── ONNX Export (on CPU, after training) ──────────────────────────────────────
def export_onnx_from_tpu(model, model_name, do_quantize=True):
    """Move model to CPU and export ONNX."""
    model_cpu = copy.deepcopy(model).cpu().float()
    model_cpu.eval()

    onnx_path = os.path.join(ONNX_DIR, f"{model_name}.onnx")
    dummy = torch.randn(1, SEQ_LEN, N_FEATURES)

    os.environ["TORCH_ONNX_USE_NEW_EXPORTER"] = "0"
    torch.onnx.export(
        model_cpu, dummy, onnx_path,
        input_names=["features"],
        output_names=["predictions"],
        dynamic_axes={"features": {0: "batch"}, "predictions": {0: "batch"}},
        opset_version=17,
    )
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    log(f"  FP32 ONNX: {size_mb:.2f} MB")

    if do_quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            from onnxruntime.quantization import QuantizationMode
            import onnx
            onnx_model = onnx.load(onnx_path)
            sensitive = {"LayerNorm", "layer_norm", "head", "norm", "gate"}
            nodes_to_skip = [
                n.name for n in onnx_model.graph.node
                if any(s in n.name.lower() for s in sensitive)
            ]
            int8_path = os.path.join(ONNX_DIR, f"{model_name}_int8.onnx")
            quantize_dynamic(
                onnx_path, int8_path,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=nodes_to_skip[:20],
            )
            log(f"  Selective quant: skipping {len(nodes_to_skip)} sensitive nodes")
            size_int8 = os.path.getsize(int8_path) / 1024 / 1024
            log(f"  INT8 ONNX: {size_int8:.2f} MB")
            return int8_path
        except Exception as e:
            log(f"  Quantization failed: {e}, using FP32")
    return onnx_path


# ── Main Training Orchestrator ────────────────────────────────────────────────
def train_model_tpu(model_name: str, epochs: int = None, batch_size: int = None, lr: float = None):
    """Full training pipeline on TPU."""
    global t0
    t0 = time.time()

    device = xm.xla_device()
    log(f"TPU device: {device}")

    # Get model and config
    model, config = get_model(model_name)
    if epochs is not None:
        config["epochs"] = epochs
    if batch_size is not None:
        config["batch_size"] = batch_size
    if lr is not None:
        config["lr"] = lr

    actual_epochs = config.get("epochs", 6)
    actual_batch = config.get("batch_size", 2048)
    actual_lr = config.get("lr", 5e-4)
    weight_decay = config.get("weight_decay", 1e-4)

    # Build data loaders
    train_loader, valid_loader, _ = build_loaders(batch_size=actual_batch, num_workers=0)

    # Move model to TPU
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Training {model_name} | {n_params:,} params | Device=TPU | bf16=True")

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_epochs)

    # Training loop
    best_score = -999.0
    no_improve = 0
    patience = config.get("patience", 7)
    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_best.pt")

    for ep in range(1, actual_epochs + 1):
        ep_start = time.time()
        log(f"Epoch {ep}/{actual_epochs} ...")

        loss = train_one_epoch_tpu(model, train_loader, optimizer, device, ep)
        metrics = evaluate_tpu(model, valid_loader, device)

        scheduler.step()

        ep_time = int(time.time() - ep_start)
        vram = "TPU"
        log(f"  ep{ep:02d}/{actual_epochs}  loss={loss:.4f}  "
            f"t0={metrics['corr_t0']:.4f}  t1={metrics['corr_t1']:.4f}  "
            f"ov={metrics['overall']:.4f}  {ep_time}s  {vram}")

        if metrics["overall"] > best_score:
            best_score = metrics["overall"]
            # Save CPU copy of state_dict
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"  Early stopping after {patience} epochs with no improvement")
                break

    log(f"Training done. Best = {best_score:.4f}")

    # Reload best
    best_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.cpu()
    model.load_state_dict(best_state)
    model = model.to(device)

    # ONNX export (on CPU)
    log(f"Exporting ONNX → {ONNX_DIR}/{model_name}.onnx")
    do_quantize = model_name != "enc_dec"
    onnx_path = export_onnx_from_tpu(model, model_name, do_quantize=do_quantize)
    log("Streaming evaluation disabled.")

    # Final summary
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} (v2 TPU) COMPLETE")
    print(f"  Best validation:   {best_score:.4f}")
    print(f"  ONNX: {onnx_path}")
    print(f"{'='*60}")

    return {"best_score": best_score, "onnx_path": onnx_path}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    available = list(MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser(description="TPU Training Harness (v2)")
    parser.add_argument("--model", required=True, choices=available + ["all"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    if args.model == "all":
        for name in available:
            train_model_tpu(name, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    else:
        train_model_tpu(args.model, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
