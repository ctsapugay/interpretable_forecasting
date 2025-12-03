import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Add parent 'main model' for data utilities (repo_root / "main model")
_main_model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "main model")
)
if _main_model_path not in sys.path:
    sys.path.insert(0, _main_model_path)

from data_splitting import ETTDataSplitter, DataSplitConfig
from train_utils import set_seed, evaluate_loop, plot_forecast_examples
from seasonal_naive_baseline import SeasonalNaive, SeasonalNaiveConfig


def main() -> None:
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

    def _make_loader(split: str, batch_size: int) -> DataLoader:
        x, y = splitter.get_forecasting_data(
            split=split,
            input_length=args.input_length,
            prediction_length=args.forecast_horizon,
            stride=args.stride,
            as_torch=True,
        )
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    val_dl = _make_loader("val", batch_size=256)
    test_dl = _make_loader("test", batch_size=256)

    norm_stats = splitter.splitter.norm_stats
    variable_names = splitter.variables

    cfg = SeasonalNaiveConfig(
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        num_variables=num_vars,
        season_length=args.season_length,
    )
    model = SeasonalNaive(cfg).to(device)

    print(f"\n[INFO] Evaluating Seasonal Naive (S={args.season_length}) on {device}")
    val_metrics = evaluate_loop(model, val_dl, device, norm_stats, denorm=True)
    test_metrics = evaluate_loop(model, test_dl, device, norm_stats, denorm=True)
    print(
        f"Val mse={val_metrics['mse']:.6f}, mae={val_metrics['mae']:.6f}, "
        f"rmse={val_metrics['rmse']:.6f}, mape={val_metrics['mape']:.6f}"
    )
    print(
        f"Test mse={test_metrics['mse']:.6f}, mae={test_metrics['mae']:.6f}, "
        f"rmse={test_metrics['rmse']:.6f}, mape={test_metrics['mape']:.6f}"
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

    # Extended-style artifacts
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(
            {
                "test_loss": float(test_metrics["mse"]),
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                "best_epoch": 0,
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

    print(f"\n[OK] Done. Results: {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()

