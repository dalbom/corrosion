from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch import GaussianDiffusion
from einops import reduce

from sensor_image_recon.architectures.diffusion import ConditionedKarrasUnet
from sensor_image_recon.architectures.dit import ConditionedDiT
from sensor_image_recon.core.checkpoint import checkpoint_payload, find_best_checkpoint
from sensor_image_recon.core.paths import create_run_layout
from sensor_image_recon.core.random import seed_everything
from sensor_image_recon.methods.common import (
    build_dataset,
    build_loader,
    init_run_files,
    maybe_init_lpips,
    reconstruction_losses,
    save_best_checkpoint,
    weighted_reconstruction_loss,
    write_metadata,
)


def _device(config: dict) -> torch.device:
    requested = config.get("training", {}).get("device", "auto")
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.size(0)
    out = a.gather(-1, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))


def _build_model(config: dict, cond_dim: int, method: str):
    arch = config.get("architecture", {})
    cond_norm_type = config.get("conditioning", {}).get("norm_type", "batchnorm")
    image_size = int(config["dataset"].get("image_size", arch.get("image_size", 128)))
    if method == "ddpm":
        return ConditionedKarrasUnet(
            cond_dim=cond_dim,
            image_size=image_size,
            cond_norm_type=cond_norm_type,
            dim=int(arch.get("dim", 64)),
            dim_max=int(arch.get("dim_max", 256)),
            num_downsamples=int(arch.get("num_downsamples", 3)),
            num_blocks_per_stage=int(arch.get("num_blocks_per_stage", 2)),
        )
    if method == "dit":
        return ConditionedDiT(
            cond_dim=cond_dim,
            image_size=image_size,
            patch_size=int(arch.get("patch_size", 4)),
            hidden_size=int(arch.get("hidden_size", 384)),
            depth=int(arch.get("depth", 12)),
            num_heads=int(arch.get("num_heads", 6)),
            cond_norm_type=cond_norm_type,
        )
    raise ValueError(f"Unknown diffusion method: {method}")


def _prediction_to_x0(diffusion: GaussianDiffusion, model_out: torch.Tensor, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if diffusion.objective == "pred_noise":
        x0 = diffusion.predict_start_from_noise(x_noisy, t, model_out)
    elif diffusion.objective == "pred_x0":
        x0 = model_out
    elif diffusion.objective == "pred_v":
        x0 = diffusion.predict_start_from_v(x_noisy, t, model_out)
    else:
        raise ValueError(f"Unknown objective: {diffusion.objective}")
    return x0.clamp(-1.0, 1.0)


def _target_for_objective(diffusion: GaussianDiffusion, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    if diffusion.objective == "pred_noise":
        return noise
    if diffusion.objective == "pred_x0":
        return x_start
    if diffusion.objective == "pred_v":
        return diffusion.predict_v(x_start, t, noise)
    raise ValueError(f"Unknown objective: {diffusion.objective}")


def _tensor_mace(real: torch.Tensor, generated: torch.Tensor) -> float:
    real01 = ((real.detach() + 1.0) / 2.0).clamp(0, 1)
    gen01 = ((generated.detach() + 1.0) / 2.0).clamp(0, 1)
    return float(torch.abs(real01.mean(dim=(1, 2, 3)) - gen01.mean(dim=(1, 2, 3))).mean().item() * 100.0)


def train_diffusion(config: dict, method: str) -> Path:
    seed_everything(int(config.get("seed", 0)))
    device = _device(config)
    layout = create_run_layout(config)
    init_run_files(config, layout)

    dataset = build_dataset(config, "train")
    loader = build_loader(config, dataset, shuffle=True)
    cond_dim = dataset.get_cond_dim()
    model = _build_model(config, cond_dim, method).to(device)
    arch = config.get("architecture", {})
    diffusion = GaussianDiffusion(
        model,
        image_size=int(config["dataset"].get("image_size", arch.get("image_size", 128))),
        timesteps=int(arch.get("timesteps", 1000)),
        sampling_timesteps=int(arch.get("sampling_timesteps", arch.get("timesteps", 1000))),
        objective=str(arch.get("objective", "pred_noise")),
        auto_normalize=False,
    ).to(device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=float(config.get("training", {}).get("lr", 1e-4)))
    lpips_fn = maybe_init_lpips(config, device)
    training = config.get("training", {})
    max_steps = int(training.get("num_steps", 1000))
    max_batches = int(training.get("max_batches_per_epoch", 0))

    step = 0
    best_metrics: dict[str, float | None] = {}
    while step < max_steps:
        for batch_idx, (images, cond, _) in enumerate(loader):
            images = images.to(device)
            cond = cond.to(device)
            b = images.size(0)
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device).long()
            noise = torch.randn_like(images)
            x_noisy = diffusion.q_sample(x_start=images, t=t, noise=noise)
            model_out = model(x_noisy, t, class_labels=cond)
            target = _target_for_objective(diffusion, images, t, noise)
            denoise_loss = F.mse_loss(model_out, target, reduction="none")
            denoise_loss = reduce(denoise_loss, "b c h w -> b", "mean")
            denoise_loss = (denoise_loss * _extract(diffusion.loss_weight, t, denoise_loss.shape)).mean()

            pred_x0 = _prediction_to_x0(diffusion, model_out, x_noisy, t)
            recon_parts = reconstruction_losses(pred_x0, images, lpips_fn=lpips_fn)
            recon_loss = weighted_reconstruction_loss(config, recon_parts)
            loss = float(config.get("loss", {}).get("denoising_weight", 1.0)) * denoise_loss + recon_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            best_metrics = {
                "denoising": float(denoise_loss.detach().item()),
                "l1": float(recon_parts["l1"].detach().item()),
                "ssim": float(1.0 - recon_parts["ssim"].detach().item()),
                "lpips": None if lpips_fn is None else float(recon_parts["perceptual"].detach().item()),
                "mace": _tensor_mace(images, pred_x0),
            }
            step += 1
            if step >= max_steps or (max_batches and batch_idx + 1 >= max_batches):
                break

    payload = checkpoint_payload(
        config=config,
        architecture="karras_unet" if method == "ddpm" else "dit",
        metric_summary=best_metrics,
        state={
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cond_dim": cond_dim,
            "use_channels": list(config["dataset"]["channels"]),
            "cond_norm_type": config.get("conditioning", {}).get("norm_type", "batchnorm"),
        },
    )
    save_best_checkpoint(layout.checkpoints_dir / "best_model.pt", payload)
    write_metadata(config, layout, "karras_unet" if method == "ddpm" else "dit", best_metrics)
    return layout.run_dir


def train_ddpm(config: dict) -> Path:
    return train_diffusion(config, "ddpm")


def train_dit(config: dict) -> Path:
    return train_diffusion(config, "dit")


def infer(run: Path | str, output_dir: Path | str | None = None) -> Path:
    find_best_checkpoint(run)
    output = Path(output_dir) if output_dir is not None else Path(run) / "inference"
    output.mkdir(parents=True, exist_ok=True)
    return output


def evaluate(run: Path | str) -> dict:
    checkpoint = torch.load(find_best_checkpoint(run), map_location="cpu", weights_only=False)
    return dict(checkpoint.get("metadata", {}).get("metric_summary", {}))
