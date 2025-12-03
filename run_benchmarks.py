"""
Run all benchmark forecasting models (LSTM, TCN, Seasonal Naive) on ETTh1
and save their outputs under `benchmark_outputs/` for comparison with the
extended interpretable model.

Usage (from repo root):

    python run_benchmarks.py

This will create:

    benchmark_outputs/
      lstm_baseline/<timestamp>/
      tcn_baseline/<timestamp>/
      seasonal_naive/<timestamp>/

Each subdirectory contains a `results.json` with test metrics that can be
directly compared to the extended model's `test_results.json`.
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse


def run(cmd, cwd: Path) -> None:
    """Run a subprocess command and stream output."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all benchmark models on ETTh1.")
    parser.add_argument(
        "--epochs-lstm",
        type=int,
        default=30,
        help="Number of training epochs for the LSTM baseline.",
    )
    parser.add_argument(
        "--epochs-tcn",
        type=int,
        default=30,
        help="Number of training epochs for the TCN baseline.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    repo_root = Path(__file__).resolve().parent
    bench_dir = repo_root / "benchmarks" / "test_models"
    data_path = repo_root / "ETT-small" / "ETTh1.csv"
    out_root = repo_root / "benchmark_outputs"

    out_root.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"ETT file not found at {data_path}. Adjust data_path in run_benchmarks.py.")

    # Prefer the repo's virtualenv Python if available; otherwise fall back to
    # the current interpreter. This avoids `ModuleNotFoundError` for `torch`
    # when the system Python does not have the required packages.
    venv_python = None
    venv_candidates = [
        repo_root / "venv" / "Scripts" / "python.exe",  # Windows
        repo_root / "venv" / "bin" / "python",          # POSIX
    ]
    for cand in venv_candidates:
        if cand.exists():
            venv_python = cand
            break
    python_exe = str(venv_python) if venv_python is not None else sys.executable

    # 1. LSTM baseline
    lstm_out = out_root / "lstm_baseline"
    lstm_out.mkdir(parents=True, exist_ok=True)

    run(
        [
            python_exe,
            str(bench_dir / "lstm_train.py"),
            "--file-path",
            str(data_path),
            "--input-length",
            "96",
            "--forecast-horizon",
            "24",
            "--epochs",
            str(args.epochs_lstm),
            "--save-dir",
            str(lstm_out),
        ],
        cwd=repo_root,
    )

    # 2. TCN baseline
    tcn_out = out_root / "tcn_baseline"
    tcn_out.mkdir(parents=True, exist_ok=True)

    run(
        [
            python_exe,
            str(bench_dir / "tcn_train.py"),
            "--file-path",
            str(data_path),
            "--input-length",
            "96",
            "--forecast-horizon",
            "24",
            "--epochs",
            str(args.epochs_tcn),
            "--save-dir",
            str(tcn_out),
        ],
        cwd=repo_root,
    )

    # 3. Seasonal naive baseline (no training, just evaluation)
    naive_out = out_root / "seasonal_naive"
    naive_out.mkdir(parents=True, exist_ok=True)

    run(
        [
            python_exe,
            str(bench_dir / "seasonal_naive_eval.py"),
            "--file-path",
            str(data_path),
            "--input-length",
            "96",
            "--forecast-horizon",
            "24",
            "--season-length",
            "24",
            "--save-dir",
            str(naive_out),
        ],
        cwd=repo_root,
    )

    print("\n✅ All benchmark runs completed. Check `benchmark_outputs/` for results.")


if __name__ == "__main__":
    main(parse_args())


