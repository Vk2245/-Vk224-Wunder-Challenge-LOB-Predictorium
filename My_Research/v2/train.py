"""
train.py — Unified training entry point (v2)
==============================================
Tech Stack: All optimisations applied automatically via engine.py.

Usage:
  python v2/train.py --model xlstm             # Train xLSTM on GPU
  python v2/train.py --model ms_tcn --device cpu # Train MS-TCN on CPU
  python v2/train.py --model all                # Train all 4 sequentially
  python v2/train.py --model xlstm --no-compile # Disable torch.compile
  python v2/train.py --model xlstm --prune      # Apply structured pruning

KAGGLE USERS: No changes needed — paths come from config.py.
  Just change DATASET_DIR in config.py (see comment there).
"""

import argparse
import sys
import os

# Ensure v2/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_device, get_gpu_info
from data_loader import build_loaders
from models import get_model, get_export_wrapper, MODEL_REGISTRY
import engine


def train_single(args, model_name: str):
    """Train a single model."""
    device = get_device(args.device)

    # Get model + merged config (defaults + CLI overrides)
    overrides = {}
    if args.lr is not None:        overrides["lr"] = args.lr
    if args.epochs is not None:    overrides["epochs"] = args.epochs
    if args.batch is not None:     overrides["batch_size"] = args.batch
    if args.hidden is not None:    overrides["hidden_dim"] = args.hidden
    if args.dropout is not None:   overrides["dropout"] = args.dropout

    model, config = get_model(model_name, **overrides)

    # Build data loaders
    batch_size = config.get("batch_size", 2048)
    train_loader, valid_loader, scaler = build_loaders(
        batch_size=batch_size, device=device, num_workers=0, pin_memory=(device == "cuda"),
    )

    # Get export wrapper (for MoE)
    wrapper_cls = get_export_wrapper(model_name)

    # Use compile unless explicitly disabled or model says no
    use_compile = config.get("use_compile", True) and not args.no_compile
    use_amp = config.get("use_amp", True)

    # Run training via engine
    results = engine.run_training(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        valid_loader=valid_loader,
        scaler_dict=scaler,
        device=device,
        epochs=config.get("epochs", 30),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
        patience=config.get("patience", 7),
        use_amp=use_amp,
        use_compile=use_compile,
        grad_accum_steps=args.grad_accum,
        aux_weight=config.get("aux_weight", 0.01),
        export_wrapper_cls=wrapper_cls,
        do_prune=args.prune,
        prune_ratio=args.prune_ratio,
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="v2 Training — Optimised LOB Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v2/train.py --model xlstm
  python v2/train.py --model ms_tcn --device cpu --epochs 10
  python v2/train.py --model all --prune
  python v2/train.py --model ttt_linear --lr 5e-4 --hidden 128
        """,
    )

    parser.add_argument("--model", required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Model to train, or 'all' for all 4")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device (default: cuda)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Override hidden dimension")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--prune", action="store_true",
                        help="Apply structured pruning before ONNX export")
    parser.add_argument("--prune-ratio", type=float, default=0.2,
                        help="Pruning ratio (default: 0.2)")

    args = parser.parse_args()

    if args.model == "all":
        # Train all models sequentially
        all_results = {}
        for name in MODEL_REGISTRY:
            print(f"\n{'#'*60}")
            print(f"# Training: {name}")
            print(f"{'#'*60}\n")
            all_results[name] = train_single(args, name)

        # Summary
        print(f"\n{'='*60}")
        print("ALL MODELS COMPLETE — SUMMARY")
        print(f"{'='*60}")
        for name, res in all_results.items():
            print(f"  {name:15s} | best={res['best_score']:.4f} | "
                  f"stream={res['streaming']['overall']:.4f} | "
                  f"p50={res['streaming']['latency_p50_ms']:.2f}ms")
        print(f"{'='*60}")

    else:
        train_single(args, args.model)


if __name__ == "__main__":
    main()
