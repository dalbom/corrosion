from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn
from torch.utils.data import DataLoader

from sensor_image_recon.core.checkpoint import build_checkpoint_metadata
from sensor_image_recon.core.config import save_config
from sensor_image_recon.core.paths import RunLayout
from sensor_image_recon.data.dataset import SensorImageDataset
from sensor_image_recon.domains import get_domain_adapter


def build_dataset(config: dict[str, Any], split: str) -> SensorImageDataset:
    dataset_cfg = config["dataset"]
    key = f"{split}_csv"
    csv_path = dataset_cfg.get(key) or dataset_cfg.get("train_csv")
    adapter = get_domain_adapter(config["domain"], dataset_cfg)
    return SensorImageDataset(
        csv_path=csv_path,
        domain_adapter=adapter,
        channels=dataset_cfg["channels"],
        image_size=int(dataset_cfg.get("image_size", config.get("architecture", {}).get("image_size", 128))),
    )


def build_loader(config: dict[str, Any], dataset: SensorImageDataset, *, shuffle: bool) -> DataLoader:
    training_cfg = config.get("training", {})
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(training_cfg.get("num_workers", 4)),
        drop_last=shuffle and len(dataset) >= int(training_cfg.get("batch_size", 32)),
    )


def init_run_files(config: dict[str, Any], layout: RunLayout) -> None:
    save_config(config, layout.run_dir / "config.yaml")


def write_metadata(config: dict[str, Any], layout: RunLayout, architecture: str, metrics: dict[str, Any]) -> dict[str, Any]:
    metadata = build_checkpoint_metadata(
        config=config,
        architecture=architecture,
        metric_summary=metrics,
    )
    (layout.run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def reconstruction_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lpips_fn=None,
) -> dict[str, torch.Tensor]:
    l1 = F.l1_loss(prediction, target)
    ssim_loss = 1.0 - ssim_fn(prediction, target, data_range=2.0, size_average=True)
    if lpips_fn is None:
        perceptual = torch.zeros((), device=prediction.device)
    else:
        perceptual = lpips_fn(
            prediction.repeat(1, 3, 1, 1),
            target.repeat(1, 3, 1, 1),
        ).mean()
    return {
        "l1": l1,
        "ssim": ssim_loss,
        "perceptual": perceptual,
    }


def weighted_reconstruction_loss(config: dict[str, Any], losses: dict[str, torch.Tensor]) -> torch.Tensor:
    loss_cfg = config.get("loss", {})
    return (
        float(loss_cfg.get("lambda_l1", 100.0)) * losses["l1"]
        + float(loss_cfg.get("lambda_ssim", 50.0)) * losses["ssim"]
        + float(loss_cfg.get("lambda_perceptual", 10.0)) * losses["perceptual"]
    )


def select_score(config: dict[str, Any], metrics: dict[str, float]) -> float:
    loss_cfg = config.get("loss", {})
    lambda_l1 = float(loss_cfg.get("lambda_l1", 100.0))
    if lambda_l1 <= 0:
        return float(metrics.get("mace", float("inf")))
    score = float(metrics.get("l1", 0.0))
    score += float(loss_cfg.get("lambda_ssim", 50.0)) / lambda_l1 * (1.0 - float(metrics.get("ssim", 0.0)))
    lpips_value = metrics.get("lpips")
    if lpips_value is not None:
        score += float(loss_cfg.get("lambda_perceptual", 10.0)) / lambda_l1 * float(lpips_value)
    return score


def maybe_init_lpips(config: dict[str, Any], device: torch.device):
    if float(config.get("loss", {}).get("lambda_perceptual", 10.0)) <= 0:
        return None
    import lpips

    fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for param in fn.parameters():
        param.requires_grad = False
    return fn


def save_best_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)


def copy_checkpoint_to_best(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)
