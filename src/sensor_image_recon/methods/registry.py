from __future__ import annotations

from sensor_image_recon.methods import cgan
from sensor_image_recon.methods.base import MethodSpec
from sensor_image_recon.methods.diffusion import evaluate as diffusion_evaluate
from sensor_image_recon.methods.diffusion import infer as diffusion_infer
from sensor_image_recon.methods.diffusion import train_ddpm, train_dit


_REGISTRY = {
    "cgan": MethodSpec("cgan", cgan.train, cgan.infer, cgan.evaluate),
    "ddpm": MethodSpec("ddpm", train_ddpm, diffusion_infer, diffusion_evaluate),
    "dit": MethodSpec("dit", train_dit, diffusion_infer, diffusion_evaluate),
}


def get_method(name: str) -> MethodSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown method: {name}") from exc


def list_methods() -> list[str]:
    return sorted(_REGISTRY)
