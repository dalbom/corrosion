from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def build_checkpoint_metadata(
    *,
    config: dict[str, Any],
    architecture: str,
    metric_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_config = dict(config.get("dataset", {}))
    channels = list(dataset_config.get("channels", []))
    loss = dict(config.get("loss", {}))
    metadata = {
        "method": config["method"],
        "architecture": architecture,
        "domain": config["domain"],
        "dataset_config": dataset_config,
        "channel_list": channels,
        "conditioning_normalization_type": config.get("conditioning", {}).get("norm_type", "batchnorm"),
        "loss_weights": {
            "lambda_l1": float(loss.get("lambda_l1", 100.0)),
            "lambda_ssim": float(loss.get("lambda_ssim", 50.0)),
            "lambda_perceptual": float(loss.get("lambda_perceptual", 10.0)),
            "denoising_weight": float(loss.get("denoising_weight", 1.0)),
        },
        "seed": int(config.get("seed", config.get("training", {}).get("seed", 0))),
        "metric_summary": metric_summary or {},
        "git_commit": current_git_commit(),
    }
    return metadata


def checkpoint_payload(
    *,
    config: dict[str, Any],
    architecture: str,
    metric_summary: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    metadata = build_checkpoint_metadata(
        config=config,
        architecture=architecture,
        metric_summary=metric_summary,
    )
    return {
        "metadata": metadata,
        **state,
    }


def find_best_checkpoint(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "checkpoints" / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {path}")
    return path
