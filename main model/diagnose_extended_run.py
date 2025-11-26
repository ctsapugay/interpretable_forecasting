"""
Lightweight diagnostics for Extended Interpretable Forecasting Model runs.

Usage (from the `main model` directory):
    python diagnose_extended_run.py --run-dir ../training_outputs/20251119_124133

This will:
- Plot train/val MSE and MAE and LR over epochs from best_model.pt
- Run a single forward+backward pass to record per-parameter gradient norms
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_splitting import ETTDataSplitter, DataSplitConfig
from extended_model import ExtendedModelConfig, InterpretableForecastingModel


def plot_history(run_dir: Path) -> None:
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("history", {})
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not train_loss or not val_loss:
        print("No train/val history stored in checkpoint.")
        return

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "train_val_loss.png", dpi=150)
    plt.close()

    train_mae = history.get("train_mae")
    val_mae = history.get("val_mae")
    if train_mae and val_mae:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_mae, label="train_mae")
        plt.plot(epochs, val_mae, label="val_mae")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / "train_val_mae.png", dpi=150)
        plt.close()

    lr_hist = history.get("lr")
    if lr_hist:
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, lr_hist)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / "lr_schedule.png", dpi=150)
        plt.close()


def check_gradients(run_dir: Path) -> None:
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        print("No config stored in checkpoint; cannot rebuild model.")
        return

    config = ExtendedModelConfig(**cfg_dict)
    model = InterpretableForecastingModel(config)

    split_cfg = DataSplitConfig()
    splitter = ETTDataSplitter(
        file_path="ETT-small/ETTh1.csv",
        split_config=split_cfg,
        normalize="standard",
    )
    # Use a moderate window length
    inputs, _ = splitter.get_forecasting_data(
        split="val",
        input_length=min(config.max_len, 96),
        prediction_length=config.forecast_horizon,
        stride=1,
        as_torch=True,
    )
    if inputs.shape[0] == 0:
        print("No validation data available for gradient diagnostics.")
        return

    batch = inputs[:4]
    batch.requires_grad_(True)

    out = model(batch)
    loss = out["forecasts"].sum()
    loss.backward()

    grad_summary = {
        name: float(p.grad.norm())
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    with open(run_dir / "gradient_norms.json", "w") as f:
        json.dump(grad_summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose extended model training run")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a training output directory containing best_model.pt",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    plot_history(run_dir)
    check_gradients(run_dir)
    print(f"Diagnostics written to {run_dir}")


if __name__ == "__main__":
    main()


