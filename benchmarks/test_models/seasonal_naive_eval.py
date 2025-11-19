import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch

# Add parent 'main model' for data utilities
_main_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main model'))
if _main_model_path not in sys.path:
    sys.path.insert(0, _main_model_path)
from data_utils import ETTDataLoader

from train_utils import set_seed, prepare_dataloaders, evaluate_loop
from seasonal_naive_baseline import SeasonalNaive, SeasonalNaiveConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file-path", type=str, default="interpretable_forecasting/ETT-small/ETTh1.csv")
    p.add_argument("--normalize", type=str, default="standard", choices=["standard", "minmax", "none"])
    p.add_argument("--input-length", type=int, default=96)
    p.add_argument("--forecast-horizon", type=int, default=24)
    p.add_argument("--season-length", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default="runs/seasonal_naive")
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    save_root = Path(args.save_dir)
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    save_dir = save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = ETTDataLoader(file_path=args.file_path, normalize=args.normalize)
    num_vars = loader.data.shape[1]
    _, val_dl, test_dl, _ = prepare_dataloaders(
        loader=loader,
        input_len=args.input_length,
        pred_len=args.forecast_horizon,
        batch_size=256,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stride=args.stride,
    )

    cfg = SeasonalNaiveConfig(
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        num_variables=num_vars,
        season_length=args.season_length,
    )
    model = SeasonalNaive(cfg).to(device)

    print(f"\n🧪 Evaluating Seasonal Naive (S={args.season_length}) on {device}")
    val_metrics = evaluate_loop(model, val_dl, device, loader.norm_stats, denorm=True)
    test_metrics = evaluate_loop(model, test_dl, device, loader.norm_stats, denorm=True)
    print(f"Val — mse={val_metrics['mse']:.6f}, mae={val_metrics['mae']:.6f}, rmse={val_metrics['rmse']:.6f}, mape={val_metrics['mape']:.6f}")
    print(f"Test — mse={test_metrics['mse']:.6f}, mae={test_metrics['mae']:.6f}, rmse={test_metrics['rmse']:.6f}, mape={test_metrics['mape']:.6f}")

    with open(save_dir / "results.json", "w") as f:
        json.dump(
            {
                "best_val_mse": float(val_metrics["mse"]),
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                "history": {},
                "config": vars(cfg),
                "args": vars(args),
            },
            f,
            indent=2,
        )
    print(f"\n✅ Done. Results: {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()





