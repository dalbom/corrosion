from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sensor_image_recon.core.identity import attach_config_identity


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path
    samples_dir: Path
    inference_dir: Path
    metrics_dir: Path


def create_run_layout(config: dict, run_id: str | None = None) -> RunLayout:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(config.get("run", {}).get("root", "runs"))
    identity = config.get("config_identity") or attach_config_identity(config)["config_identity"]
    run_dir = (
        root
        / identity["domain"]
        / identity["method"]
        / identity["study"]
        / identity["sensor_set"]
        / identity["recipe"]
        / identity["seed_name"]
        / run_id
    )
    layout = RunLayout(
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        tensorboard_dir=run_dir / "tensorboard",
        samples_dir=run_dir / "samples",
        inference_dir=run_dir / "inference",
        metrics_dir=run_dir / "metrics",
    )
    for path in (
        layout.run_dir,
        layout.checkpoints_dir,
        layout.tensorboard_dir,
        layout.samples_dir,
        layout.inference_dir,
        layout.metrics_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout
