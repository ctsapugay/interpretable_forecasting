"""
Run multiple Extended Interpretable Forecasting Model configurations
to compare hyperparameters (spline settings, residual head on/off, etc.).

Usage (from repo root):

    python "main model/model_eval.py"

This will run several training jobs using `train_extended_model.py` and
store their outputs under:

    model_eval_outputs/<experiment_name>/<timestamp>/

Each subdirectory will contain the usual `config.json`, `test_results.json`,
and interpretability artifacts saved by `train_extended_model.py`.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for a single model evaluation run."""

    name: str
    epochs: int
    lr: float
    weight_decay: float
    num_control_points: int
    num_compression_queries: int
    spline_smooth_alpha: float
    use_residual_head: bool
    use_spline_head: bool = True
    scheduler: bool = True
    warmup_epochs: int = 3
    patience: int = 20
    batch_size: int = 32
    input_length: int = 96
    forecast_horizon: int = 24


def get_python_executable(repo_root: Path) -> str:
    """
    Prefer the repo's virtualenv Python if available; otherwise fall back
    to the current interpreter.
    """
    venv_candidates = [
        repo_root / "venv" / "Scripts" / "python.exe",  # Windows
        repo_root / "venv" / "bin" / "python",          # POSIX
    ]
    for cand in venv_candidates:
        if cand.exists():
            return str(cand)
    return sys.executable


def build_command(
    python_exe: str,
    repo_root: Path,
    base_output_dir: Path,
    data_path: Path,
    exp: ExperimentConfig,
) -> List[str]:
    """Construct the subprocess command for a given experiment."""
    main_script = repo_root / "main model" / "train_extended_model.py"
    exp_out_dir = base_output_dir / exp.name

    cmd: List[str] = [
        python_exe,
        str(main_script),
        "--data-path",
        str(data_path),
        "--input-length",
        str(exp.input_length),
        "--forecast-horizon",
        str(exp.forecast_horizon),
        "--batch-size",
        str(exp.batch_size),
        "--epochs",
        str(exp.epochs),
        "--lr",
        str(exp.lr),
        "--weight-decay",
        str(exp.weight_decay),
        "--num-control-points",
        str(exp.num_control_points),
        "--num-compression-queries",
        str(exp.num_compression_queries),
        "--spline-smooth-alpha",
        str(exp.spline_smooth_alpha),
        "--warmup-epochs",
        str(exp.warmup_epochs),
        "--patience",
        str(exp.patience),
        "--save-dir",
        str(exp_out_dir),
    ]

    if exp.scheduler:
        cmd.append("--scheduler")

    # Head configuration
    if not exp.use_spline_head:
        cmd.append("--no-spline-head")
    if not exp.use_residual_head:
        cmd.append("--no-residual-head")

    return cmd


def run_experiment(
    python_exe: str,
    repo_root: Path,
    base_output_dir: Path,
    data_path: Path,
    exp: ExperimentConfig,
) -> None:
    """Run a single experiment configuration."""
    cmd = build_command(
        python_exe=python_exe,
        repo_root=repo_root,
        base_output_dir=base_output_dir,
        data_path=data_path,
        exp=exp,
    )

    print("\n" + "=" * 80)
    print(f"▶ Running experiment: {exp.name}")
    print("Command:", " ".join(cmd))
    print("=" * 80 + "\n")

    result = subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        raise SystemExit(
            f"Experiment '{exp.name}' failed with exit code {result.returncode}"
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "ETT-small" / "ETTh1.csv"
    base_output_dir = repo_root / "model_eval_outputs"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"ETT file not found at {data_path}")

    python_exe = get_python_executable(repo_root)

    # Define a small suite of diverse configurations around ~50 epochs each.
    experiments: List[ExperimentConfig] = [
        # 1) Baseline: spline-only, moderate smoothness
        ExperimentConfig(
            name="spline_only_alpha0.3_cp16_q2",
            epochs=50,
            lr=5e-4,
            weight_decay=1e-5,
            num_control_points=16,
            num_compression_queries=2,
            spline_smooth_alpha=0.3,
            use_residual_head=False,
        ),
        # 2) More flexible spline: more control points, lower smoothness
        ExperimentConfig(
            name="spline_only_alpha0.1_cp24_q3",
            epochs=60,
            lr=5e-4,
            weight_decay=1e-5,
            num_control_points=24,
            num_compression_queries=3,
            spline_smooth_alpha=0.1,
            use_residual_head=False,
        ),
        # 3) Very smooth spline with fewer control points (strong bias, high smoothness)
        ExperimentConfig(
            name="spline_only_alpha0.6_cp12_q1",
            epochs=50,
            lr=5e-4,
            weight_decay=1e-5,
            num_control_points=12,
            num_compression_queries=1,
            spline_smooth_alpha=0.6,
            use_residual_head=False,
        ),
        # 4) Residual head ON with moderate smoothness
        ExperimentConfig(
            name="residual_on_alpha0.3_cp16_q2",
            epochs=50,
            lr=5e-4,
            weight_decay=1e-5,
            num_control_points=16,
            num_compression_queries=2,
            spline_smooth_alpha=0.3,
            use_residual_head=True,
        ),
        # 5) Residual head ON, more flexible spline
        ExperimentConfig(
            name="residual_on_alpha0.15_cp24_q3",
            epochs=60,
            lr=5e-4,
            weight_decay=1e-5,
            num_control_points=24,
            num_compression_queries=3,
            spline_smooth_alpha=0.15,
            use_residual_head=True,
        ),
    ]

    print(f"Found {len(experiments)} experiments to run.")
    for exp in experiments:
        run_experiment(
            python_exe=python_exe,
            repo_root=repo_root,
            base_output_dir=base_output_dir,
            data_path=data_path,
            exp=exp,
        )

    print("\nAll model evaluation experiments completed.")
    print(f"Results saved under: {base_output_dir}")


if __name__ == "__main__":
    main()

