from __future__ import annotations

from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from sensor_image_recon.architectures.gan import (
    Critic,
    Generator,
    compute_gradient_penalty,
    weights_init,
)
from sensor_image_recon.core.checkpoint import checkpoint_payload, find_best_checkpoint
from sensor_image_recon.core.config import load_config
from sensor_image_recon.core.paths import create_run_layout
from sensor_image_recon.core.random import seed_everything
from sensor_image_recon.evaluation.report import evaluate_generated
from sensor_image_recon.methods.common import (
    build_dataset,
    build_loader,
    init_run_files,
    maybe_init_lpips,
    reconstruction_losses,
    save_best_checkpoint,
    select_score,
    weighted_reconstruction_loss,
    write_metadata,
)


def _device(config: dict) -> torch.device:
    requested = config.get("training", {}).get("device", "auto")
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_mace(real: torch.Tensor, generated: torch.Tensor) -> float:
    real01 = ((real.detach() + 1.0) / 2.0).clamp(0, 1)
    gen01 = ((generated.detach() + 1.0) / 2.0).clamp(0, 1)
    return float(torch.abs(real01.mean(dim=(1, 2, 3)) - gen01.mean(dim=(1, 2, 3))).mean().item() * 100.0)


@torch.no_grad()
def _validate(
    *,
    config: dict,
    generator: Generator,
    val_loader: DataLoader,
    device: torch.device,
    latent_dim: int,
    lpips_fn,
) -> dict[str, float | None]:
    generator.eval()
    max_samples = int(config.get("training", {}).get("val_sample_size", 64))
    seen = 0
    totals = {"l1": 0.0, "ssim": 0.0, "mace": 0.0}
    lpips_total = 0.0
    lpips_seen = 0
    for real_images, cond, _ in val_loader:
        if max_samples > 0 and seen >= max_samples:
            break
        if max_samples > 0:
            remaining = max_samples - seen
            real_images = real_images[:remaining]
            cond = cond[:remaining]
        real_images = real_images.to(device)
        cond = cond.to(device)
        noise = torch.randn(cond.size(0), latent_dim, device=device)
        fake_images = generator(noise, cond)
        parts = reconstruction_losses(fake_images, real_images, lpips_fn=lpips_fn)
        batch_n = real_images.size(0)
        totals["l1"] += float(parts["l1"].item()) * batch_n
        totals["ssim"] += float(1.0 - parts["ssim"].item()) * batch_n
        totals["mace"] += _tensor_mace(real_images, fake_images) * batch_n
        if lpips_fn is not None:
            lpips_total += float(parts["perceptual"].item()) * batch_n
            lpips_seen += batch_n
        seen += batch_n
    generator.train()
    if seen == 0:
        raise ValueError("Validation loader produced no samples")
    return {
        "l1": totals["l1"] / seen,
        "ssim": totals["ssim"] / seen,
        "lpips": None if lpips_fn is None or lpips_seen == 0 else lpips_total / lpips_seen,
        "mace": totals["mace"] / seen,
    }


def train(config: dict) -> Path:
    seed_everything(int(config.get("seed", 0)))
    device = _device(config)
    layout = create_run_layout(config)
    init_run_files(config, layout)

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    loader = build_loader(config, train_dataset, shuffle=True)
    val_loader = build_loader(config, val_dataset, shuffle=False)
    cond_dim = train_dataset.get_cond_dim()
    arch = config.get("architecture", {})
    training = config.get("training", {})
    cond_norm_type = config.get("conditioning", {}).get("norm_type", "batchnorm")

    generator = Generator(
        latent_dim=int(arch.get("latent_dim", 128)),
        cond_dim=cond_dim,
        image_size=int(arch.get("image_size", config["dataset"].get("image_size", 128))),
        ngf=int(arch.get("ngf", 128)),
        cond_norm_type=cond_norm_type,
    ).to(device)
    critic = Critic(
        cond_dim=cond_dim,
        image_size=int(arch.get("image_size", config["dataset"].get("image_size", 128))),
        ndf=int(arch.get("ndf", 128)),
        cond_norm_type=cond_norm_type,
    ).to(device)
    generator.apply(weights_init)
    critic.apply(weights_init)

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=float(training.get("lr_g", training.get("lr", 1e-4))),
        betas=(0.0, 0.9),
    )
    optimizer_c = optim.Adam(
        critic.parameters(),
        lr=float(training.get("lr_c", training.get("lr", 1e-4))),
        betas=(0.0, 0.9),
    )
    lpips_fn = maybe_init_lpips(config, device)

    best_score = float("inf")
    best_metrics: dict[str, float | None] = {}
    max_batches = int(training.get("max_batches_per_epoch", 0))
    epochs = int(training.get("epochs", 100))
    latent_dim = int(arch.get("latent_dim", 128))

    for _epoch in range(1, epochs + 1):
        for batch_idx, (real_images, cond, _) in enumerate(loader):
            real_images = real_images.to(device)
            cond = cond.to(device)
            batch_size = real_images.size(0)

            for _ in range(int(training.get("n_critic", 2))):
                optimizer_c.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake_images = generator(noise, cond).detach()
                score_real = critic(real_images, cond)
                score_fake = critic(fake_images, cond)
                gp = compute_gradient_penalty(critic, real_images, fake_images, cond, device)
                w_distance = score_real.mean() - score_fake.mean()
                critic_loss = -w_distance + float(training.get("lambda_gp", 10.0)) * gp
                critic_loss.backward()
                optimizer_c.step()

            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise, cond)
            adv_loss = -critic(fake_images, cond).mean()
            recon_parts = reconstruction_losses(fake_images, real_images, lpips_fn=lpips_fn)
            recon_loss = weighted_reconstruction_loss(config, recon_parts)
            generator_loss = adv_loss + recon_loss
            generator_loss.backward()
            optimizer_g.step()

            if max_batches and batch_idx + 1 >= max_batches:
                break

        metrics = _validate(
            config=config,
            generator=generator,
            val_loader=val_loader,
            device=device,
            latent_dim=latent_dim,
            lpips_fn=lpips_fn,
        )
        score = select_score(config, metrics)
        print(
            f"epoch {_epoch}/{epochs} "
            f"val_l1={metrics['l1']:.4f} "
            f"val_ssim={metrics['ssim']:.4f} "
            f"val_lpips={metrics['lpips'] if metrics['lpips'] is not None else 'n/a'} "
            f"val_mace={metrics['mace']:.2f} "
            f"score={score:.4f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_metrics = metrics
            payload = checkpoint_payload(
                config=config,
                architecture="gan",
                metric_summary=metrics,
                state={
                    "epoch": _epoch,
                    "generator_state_dict": generator.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "optimizer_G_state_dict": optimizer_g.state_dict(),
                    "optimizer_C_state_dict": optimizer_c.state_dict(),
                    "cond_dim": cond_dim,
                    "use_channels": list(config["dataset"]["channels"]),
                    "image_size": int(config["dataset"].get("image_size", 128)),
                    "latent_dim": latent_dim,
                    "ngf": int(arch.get("ngf", 128)),
                    "ndf": int(arch.get("ndf", 128)),
                    "cond_norm_type": cond_norm_type,
                    "selection_score": score,
                },
            )
            save_best_checkpoint(layout.checkpoints_dir / "best_model.pt", payload)

    write_metadata(config, layout, "gan", best_metrics)
    return layout.run_dir


def infer(run: Path | str, output_dir: Path | str | None = None) -> Path:
    run = Path(run)
    config = load_config(run / "config.yaml")
    checkpoint_path = find_best_checkpoint(run)
    device = _device(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    output = Path(output_dir) if output_dir is not None else run / "inference"
    output.mkdir(parents=True, exist_ok=True)
    cond_dim = int(checkpoint.get("cond_dim", checkpoint["metadata"]["dataset_config"].get("cond_dim", 0)))
    if cond_dim <= 0:
        cond_dim = len(config["dataset"]["channels"]) * 201
    generator = Generator(
        latent_dim=int(checkpoint.get("latent_dim", config["architecture"].get("latent_dim", 128))),
        cond_dim=cond_dim,
        image_size=int(checkpoint.get("image_size", config["dataset"].get("image_size", 128))),
        ngf=int(checkpoint.get("ngf", config["architecture"].get("ngf", 128))),
        cond_norm_type=checkpoint.get("cond_norm_type", config.get("conditioning", {}).get("norm_type", "batchnorm")),
    ).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    dataset = build_dataset(config, "test")
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 4)),
    )
    latent_dim = int(checkpoint.get("latent_dim", config["architecture"].get("latent_dim", 128)))
    target_size = tuple(config.get("inference", {}).get("target_size", [110, 300]))
    with torch.no_grad():
        for _, cond, filenames in loader:
            cond = cond.to(device)
            noise = torch.randn(cond.size(0), latent_dim, device=device)
            images = generator(noise, cond)
            images = torch.nn.functional.interpolate(
                images,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            images = ((images + 1.0) / 2.0).clamp(0, 1)
            for index, filename in enumerate(filenames):
                filename = str(filename)
                sample_id = filename.replace(".png", "").split("_")[1]
                specimen_dir = output / sample_id
                specimen_dir.mkdir(parents=True, exist_ok=True)
                filename_with_ext = filename if filename.endswith(".png") else f"{filename}.png"
                rgb = images[index].repeat(3, 1, 1)
                rgb[1] = 0
                rgb[2] = 0
                save_image(rgb, specimen_dir / filename_with_ext)
    return output


def evaluate(run: Path | str) -> dict:
    run = Path(run)
    config = load_config(run / "config.yaml")
    generated_root = run / "inference"
    if generated_root.exists() and any(generated_root.rglob("*.png")):
        metrics = evaluate_generated(config, generated_root, run / "metrics" / "raw_metrics.csv")
        return metrics
    checkpoint = torch.load(find_best_checkpoint(run), map_location="cpu", weights_only=False)
    return dict(checkpoint.get("metadata", {}).get("metric_summary", {}))
