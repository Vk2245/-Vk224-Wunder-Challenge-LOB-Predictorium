"""
hpo.py — Optuna Hyperparameter Optimisation (v2)
==================================================
Tech Stack: Optuna with MedianPruner — kills bad trials early,
finds optimal configs 5–10x faster than grid search.

Usage:
  python v2/hpo.py --model xlstm --n-trials 30
  python v2/hpo.py --model ms_tcn --n-trials 50 --device cuda
  python v2/hpo.py --model all --n-trials 20

Each trial trains for a few epochs and reports the validation
correlation. Bad trials are pruned early by MedianPruner.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config import get_device, MLRUNS_DIR
from data_loader import build_loaders
from models import get_model, get_export_wrapper, MODEL_REGISTRY
import engine


# ── Search Spaces per Model ──────────────────────────────────────────────────
SEARCH_SPACES = {
    "xlstm": {
        "hidden_dim":   ("int",   [64, 256]),
        "proj_dim":     ("int",   [128, 512]),
        "n_blocks":     ("int",   [1, 3]),
        "dropout":      ("float", [0.05, 0.3]),
        "lr":           ("loguniform", [1e-5, 1e-3]),
        "batch_size":   ("categorical", [256, 512, 1024]),
        "weight_decay": ("loguniform", [1e-6, 1e-3]),
    },
    "ttt_linear": {
        "hidden_dim":   ("int",   [64, 192]),
        "n_blocks":     ("int",   [1, 3]),
        "dropout":      ("float", [0.05, 0.25]),
        "lr":           ("loguniform", [1e-4, 5e-3]),
        "batch_size":   ("categorical", [128, 256, 512]),
        "weight_decay": ("loguniform", [1e-6, 1e-3]),
    },
    "sparse_moe": {
        "hidden_dim":   ("int",   [64, 256]),
        "n_experts":    ("int",   [2, 6]),
        "lr":           ("loguniform", [1e-4, 5e-3]),
        "batch_size":   ("categorical", [512, 1024, 2048]),
        "weight_decay": ("loguniform", [1e-6, 1e-3]),
        "aux_weight":   ("loguniform", [1e-3, 0.1]),
    },
    "ms_tcn": {
        "hidden_dim":   ("int",   [64, 256]),
        "n_layers_per_branch": ("int", [1, 3]),
        "dropout":      ("float", [0.05, 0.25]),
        "lr":           ("loguniform", [1e-4, 5e-3]),
        "batch_size":   ("categorical", [512, 1024, 2048, 4096]),
        "weight_decay": ("loguniform", [1e-6, 1e-3]),
    },
}


def sample_hparams(trial, model_name: str) -> dict:
    """Sample hyperparameters from the search space."""
    space = SEARCH_SPACES[model_name]
    params = {}

    for name, (dist_type, bounds) in space.items():
        if dist_type == "int":
            params[name] = trial.suggest_int(name, bounds[0], bounds[1])
        elif dist_type == "float":
            params[name] = trial.suggest_float(name, bounds[0], bounds[1])
        elif dist_type == "loguniform":
            params[name] = trial.suggest_float(name, bounds[0], bounds[1], log=True)
        elif dist_type == "categorical":
            params[name] = trial.suggest_categorical(name, bounds)

    return params


def create_objective(model_name: str, device: str, hpo_epochs: int):
    """Create an Optuna objective function for the given model."""

    # Pre-load data once (shared across trials)
    # We'll load with max batch size and adjust per trial
    print("Pre-loading data for HPO ...", flush=True)

    def objective(trial):
        # Sample hyperparameters
        hparams = sample_hparams(trial, model_name)

        # Build model with sampled params
        model, config = get_model(model_name, **hparams)

        # Build loaders with sampled batch size
        batch_size = hparams.get("batch_size", config.get("batch_size", 1024))
        train_loader, valid_loader, scaler = build_loaders(
            batch_size=batch_size, device=device, num_workers=0,
            pin_memory=(device == "cuda"),
        )

        wrapper_cls = get_export_wrapper(model_name)

        try:
            results = engine.run_training(
                model=model,
                model_name=f"{model_name}_trial{trial.number}",
                train_loader=train_loader,
                valid_loader=valid_loader,
                scaler_dict=scaler,
                device=device,
                epochs=hpo_epochs,
                lr=hparams.get("lr", config.get("lr", 1e-3)),
                weight_decay=hparams.get("weight_decay", config.get("weight_decay", 1e-4)),
                patience=config.get("patience", 5),
                use_amp=config.get("use_amp", True),
                use_compile=False,  # Skip compile during HPO (warmup overhead)
                aux_weight=hparams.get("aux_weight", config.get("aux_weight", 0.01)),
                export_wrapper_cls=wrapper_cls,
                hpo_trial=trial,
            )
            return results["best_score"]

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}", flush=True)
            return float("-inf")

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="v2 HPO — Optuna Hyperparameter Optimisation",
        epilog="""
Examples:
  python v2/hpo.py --model xlstm --n-trials 30
  python v2/hpo.py --model ms_tcn --n-trials 50 --hpo-epochs 10
  python v2/hpo.py --model all --n-trials 20
        """,
    )

    parser.add_argument("--model", required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Model to optimise")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--hpo-epochs", type=int, default=15,
                        help="Max epochs per trial (default: 15)")
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name (default: auto)")

    args = parser.parse_args()
    device = get_device(args.device)

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    for model_name in models_to_run:
        study_name = args.study_name or f"v2_hpo_{model_name}"

        print(f"\n{'#'*60}")
        print(f"# HPO: {model_name} | {args.n_trials} trials | {args.hpo_epochs} epochs each")
        print(f"{'#'*60}\n")

        # Create study with TPE sampler + MedianPruner
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )

        objective = create_objective(model_name, device, args.hpo_epochs)

        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        # Results
        print(f"\n{'='*60}")
        print(f"HPO COMPLETE: {model_name}")
        print(f"  Best score: {study.best_value:.4f}")
        print(f"  Best params:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")
        print(f"  Trials: {len(study.trials)} total, "
              f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
