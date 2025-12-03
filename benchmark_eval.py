"""
Run LSTM, TCN, and Seasonal Naive baselines on ETTh1 with a shared
configuration, saving outputs under `benchmark_outputs/`.

Usage:
    python benchmark_eval.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import argparse


def get_python_executable(repo_root: Path) -> str:
    """Prefer repo virtualenv Python if available."""
    candidates = [
        repo_root / "venv" / "Scripts" / "python.exe",  # Windows
        repo_root / "venv" / "bin" / "python",          # POSIX
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return sys.executable


def run(cmd, cwd: Path) -> None:
    print("\n>>> Running:", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(map(str, cmd))}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ETTh1 baselines with shared config.")
    p.add_argument("--epochs", type=int, default=50, help="Epochs for LSTM and TCN baselines.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    bench_dir = repo_root / "benchmarks" / "test_models"
    data_path = repo_root / "ETT-small" / "ETTh1.csv"
    out_root = repo_root / "benchmark_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"ETT file not found at {data_path}")

    python_exe = get_python_executable(repo_root)

    # LSTM baseline
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
            str(args.epochs),
            "--batch-size",
            "32",
            "--lr",
            "1e-3",
            "--hidden-size",
            "128",
            "--num-layers",
            "2",
            "--dropout",
            "0.1",
            "--save-dir",
            str(lstm_out),
        ],
        cwd=repo_root,
    )

    # TCN baseline
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
            str(args.epochs),
            "--batch-size",
            "32",
            "--lr",
            "1e-3",
            "--channels",
            "64,64,64",
            "--kernel-size",
            "3",
            "--dropout",
            "0.1",
            "--save-dir",
            str(tcn_out),
        ],
        cwd=repo_root,
    )

    # Seasonal naive baseline
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

    print("\nAll benchmark evaluations completed.")
    print(f"Results under: {out_root}")


if __name__ == "__main__":
    main()

