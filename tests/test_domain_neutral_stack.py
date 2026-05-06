import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from sensor_image_recon.core.checkpoint import build_checkpoint_metadata
from sensor_image_recon.core.config import load_config
from sensor_image_recon.core.paths import create_run_layout
from sensor_image_recon.data.dataset import SensorImageDataset
from sensor_image_recon.domains.corrosion import CorrosionDomainAdapter
from sensor_image_recon.domains.corrosion.metrics import mean_absolute_corrosion_error
from sensor_image_recon.methods.registry import get_method, list_methods


def _write_corrosion_fixture(root: Path, n: int = 4) -> tuple[Path, Path]:
    dataset_root = root / "datasets"
    image_root = dataset_root / "corrosion_img"
    rows = []
    for i in range(n):
        specimen = f"{61 + i}"
        filename = f"0525_{specimen}_{30 + i}.0_augmented"
        specimen_dir = image_root / specimen
        specimen_dir.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        arr[..., 0] = 32 + i * 20
        arr[..., 1] = 7
        arr[..., 2] = 3
        Image.fromarray(arr).save(specimen_dir / f"{filename}.png")
        rows.append(
            {
                "filename": filename,
                "S11": " ".join(str(float(j + i)) for j in range(6)),
                "S21": " ".join(str(float(j + i + 10)) for j in range(6)),
                "Phase11": " ".join(str(float(j + i + 20)) for j in range(6)),
                "Phase21": " ".join(str(float(j + i + 30)) for j in range(6)),
            }
        )
    csv_path = dataset_root / "Corrosion_train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, dataset_root


def _base_config(tmp_path: Path, method: str) -> Path:
    csv_path, dataset_root = _write_corrosion_fixture(tmp_path, n=4)
    config_path = tmp_path / f"{method}.yaml"
    config_path.write_text(
        f"""
domain: corrosion
method: {method}
experiment: smoke
seed: 7
run:
  root: {tmp_path / "runs"}
dataset:
  train_csv: {csv_path}
  val_csv: {csv_path}
  test_csv: {csv_path}
  img_root: {dataset_root}
  image_size: 128
  channels: [S11, S21]
training:
  epochs: 1
  num_steps: 1
  batch_size: 2
  num_workers: 0
  max_batches_per_epoch: 1
  val_sample_size: 2
  lr: 0.0002
  n_critic: 1
loss:
  lambda_l1: 100.0
  lambda_ssim: 50.0
  lambda_perceptual: 0.0
conditioning:
  norm_type: batchnorm
architecture:
  latent_dim: 8
  ngf: 4
  ndf: 4
  image_size: 128
  timesteps: 4
  sampling_timesteps: 2
  dim: 8
  dim_max: 16
  num_downsamples: 1
  num_blocks_per_stage: 1
  patch_size: 4
  hidden_size: 16
  depth: 1
  num_heads: 2
""",
        encoding="utf-8",
    )
    return config_path


def test_corrosion_dataset_uses_adapter_condition_parser_and_red_target(tmp_path):
    csv_path, dataset_root = _write_corrosion_fixture(tmp_path, n=2)
    adapter = CorrosionDomainAdapter({"img_root": str(dataset_root)})

    dataset = SensorImageDataset(
        csv_path=csv_path,
        domain_adapter=adapter,
        channels=["S11", "S21"],
        image_size=16,
    )

    target, cond, sample_id = dataset[0]

    assert target.shape == (1, 16, 16)
    assert target.min() >= -1.0 and target.max() <= 1.0
    assert cond.shape == (12,)
    assert cond.dtype == torch.float32
    assert sample_id.startswith("0525_61_")
    assert dataset.get_cond_dim() == 12


def test_corrosion_mace_metric_is_raw_mean_difference():
    real = np.full((4, 4), 0.25, dtype=np.float32)
    generated = np.full((4, 4), 0.40, dtype=np.float32)

    assert mean_absolute_corrosion_error(real, generated) == pytest.approx(15.0)


def test_method_registry_exposes_active_methods_only():
    assert set(list_methods()) == {"cgan", "ddpm", "dit"}
    assert get_method("cgan").name == "cgan"
    with pytest.raises(KeyError):
        get_method("cvae")


def test_run_layout_and_checkpoint_metadata_include_required_fields(tmp_path):
    config = load_config(_base_config(tmp_path, "cgan"))
    layout = create_run_layout(config, run_id="20260506-test")
    metadata = build_checkpoint_metadata(
        config=config,
        architecture="gan",
        metric_summary={"mace": 5.27},
    )

    assert layout.run_dir == tmp_path / "runs" / "corrosion" / "cgan" / "smoke" / "20260506-test"
    for name in ["checkpoints", "tensorboard", "samples", "inference", "metrics"]:
        assert (layout.run_dir / name).is_dir()
    assert metadata["method"] == "cgan"
    assert metadata["architecture"] == "gan"
    assert metadata["domain"] == "corrosion"
    assert metadata["channel_list"] == ["S11", "S21"]
    assert metadata["conditioning_normalization_type"] == "batchnorm"
    assert metadata["loss_weights"]["lambda_l1"] == 100.0
    assert "git_commit" in metadata


@pytest.mark.parametrize("method", ["cgan", "ddpm", "dit"])
def test_cpu_smoke_train_writes_standard_run_metadata(tmp_path, method):
    config = load_config(_base_config(tmp_path, method))
    if method in {"ddpm", "dit"}:
        config["dataset"]["image_size"] = 16
        config["architecture"]["image_size"] = 16

    run_dir = get_method(method).train(config)

    metadata_path = run_dir / "metadata.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    assert metadata_path.exists()
    assert checkpoint_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == method
    assert metadata["domain"] == "corrosion"
    assert metadata["conditioning_normalization_type"] == "batchnorm"
