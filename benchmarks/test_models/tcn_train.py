import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent 'main model' for data utilities (repo_root / "main model")
_main_model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "main model")
)
if _main_model_path not in sys.path:
    sys.path.insert(0, _main_model_path)
from data_splitting import ETTDataSplitter, DataSplitConfig

from train_utils import set_seed, evaluate_loop, train_one_epoch, plot_forecast_examples
from tcn_baseline import TCNBaseline, TCNBaselineConfig


def main():
    p = argparse.ArgumentParser()
    # Data / splits
    p.add_argument("--file-path", type=str, default="interpretable_forecasting/ETT-small/ETTh1.csv")
    p.add_argument("--normalize", type=str, default="standard", choices=["standard", "minmax", "none"])
    p.add_argument("--input-length", type=int, default=96)
    p.add_argument("--forecast-horizon", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)

    # Model
    p.add_argument("--channels", type=str, default="64,64,64")
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # Train
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # IO
    p.add_argument("--save-dir", type=str, default="runs/tcn_baseline")
    p.add_argument("--run-name", type=str, default=None)  # auto stamp if None

    args = p.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    save_root = Path(args.save_dir)
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    save_dir = save_root / run_name
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Data using the same splitter as the extended model
    split_cfg = DataSplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    splitter = ETTDataSplitter(
        file_path=args.file_path,
        split_config=split_cfg,
        normalize=args.normalize,
    )
    num_vars = len(splitter.variables)

    def _make_loader(split: str, shuffle: bool) -> DataLoader:
        x, y = splitter.get_forecasting_data(
            split=split,
            input_length=args.input_length,
            prediction_length=args.forecast_horizon,
            stride=args.stride,
            as_torch=True,
        )
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=False)

    train_dl = _make_loader("train", shuffle=True)
    val_dl = _make_loader("val", shuffle=False)
    test_dl = _make_loader("test", shuffle=False)

    norm_stats = splitter.splitter.norm_stats
    variable_names = splitter.variables

    # Model
    channels = tuple(int(x) for x in args.channels.split(",") if x.strip())
    cfg = TCNBaselineConfig(
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        num_variables=num_vars,
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )
    model = TCNBaseline(cfg).to(device)

    # Optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # Train loop
    best_val = math.inf
    best_path = save_dir / "checkpoints" / "best.pt"
    history = {"train_loss": [], "val_metrics": []}

    print(f"\n[INFO] Training TCN Baseline on {device}")
    print(
        f"   Variables (M): {num_vars} | Input length (T_in): {args.input_length} "
        f"| Horizon (H): {args.forecast_horizon}"
    )
    print(f"   Params: {TCNBaseline.count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
        history["train_loss"].append(float(train_loss))

        # Validation metrics (denormalized, overall)
        val_metrics = evaluate_loop(model, val_dl, device, norm_stats, denorm=True)
        history["val_metrics"].append(val_metrics)

        val_loss = val_metrics["mse"]
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val,
                    "config": vars(cfg),
                    "args": vars(args),
                },
                best_path,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mse={val_metrics['mse']:.6f}  "
            f"(mae={val_metrics['mae']:.6f}, rmse={val_metrics['rmse']:.6f}, mape={val_metrics['mape']:.6f}) "
            f"{'[BEST]' if improved else ''}"
        )

    # Load best & test
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\n[OK] Loaded best checkpoint from epoch {ckpt['epoch']} with val_mse={ckpt['val_loss']:.6f}")

    test_metrics = evaluate_loop(model, test_dl, device, norm_stats, denorm=True)
    print(
        f"\n[INFO] Test — mse={test_metrics['mse']:.6f}, "
        f"mae={test_metrics['mae']:.6f}, rmse={test_metrics['rmse']:.6f}, mape={test_metrics['mape']:.6f}"
    )

    # Simple forecast visualization for this baseline run
    plot_forecast_examples(
        model=model,
        data_loader=test_dl,
        device=device,
        norm_stats=norm_stats,
        save_path=save_dir / "forecast_examples.png",
        variable_names=variable_names,
    )

    # Save run summary
    with open(save_dir / "results.json", "w") as f:
        json.dump(
            {
                "best_val_mse": float(best_val),
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                "history": history,
                "config": vars(cfg),
                "args": vars(args),
            },
            f,
            indent=2,
        )

    # Extended-style artifacts
    best_epoch = ckpt["epoch"] if "ckpt" in locals() else args.epochs
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(
            {
                "test_loss": float(test_metrics["mse"]),
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                "best_epoch": int(best_epoch),
            },
            f,
            indent=2,
        )

    with open(save_dir / "config.json", "w") as f:
        json.dump(
            {
                "model_config": vars(cfg),
                "train_args": vars(args),
            },
            f,
            indent=2,
        )

    print(f"\n[OK] Done. Artifacts saved to: {save_dir}")
    print(f"   Best checkpoint: {best_path}")
    print(f"   Results: {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()






