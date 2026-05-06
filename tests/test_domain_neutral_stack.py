import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from sensor_image_recon.core.catalog import generate_catalog
from sensor_image_recon.core.checkpoint import build_checkpoint_metadata
from sensor_image_recon.core.config import load_config
from sensor_image_recon.core.paths import create_run_layout
from sensor_image_recon.core.sweep import expand_sweep_config
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
study:
  name: smoke_study
variant:
  sensor_set: s11_s21
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

    assert layout.run_dir == (
        tmp_path
        / "runs"
        / "corrosion"
        / "cgan"
        / "smoke_study"
        / "s11_s21"
        / "seed_007"
        / "20260506-test"
    )
    for name in ["checkpoints", "tensorboard", "samples", "inference", "metrics"]:
        assert (layout.run_dir / name).is_dir()
    assert metadata["method"] == "cgan"
    assert metadata["architecture"] == "gan"
    assert metadata["domain"] == "corrosion"
    assert metadata["channel_list"] == ["S11", "S21"]
    assert metadata["conditioning_normalization_type"] == "batchnorm"
    assert metadata["loss_weights"]["lambda_l1"] == 100.0
    assert metadata["config_identity"]["sensor_set"] == "s11_s21"
    assert "recipe" not in metadata["config_identity"]
    assert metadata["config_identity_key"] == "corrosion/cgan/smoke_study/s11_s21/seed_007"
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


def _write_fake_catalog_run(root: Path, run_id: str, mace: float) -> Path:
    run_dir = root / "corrosion" / "cgan" / "study_a" / "s11_s21" / "seed_001" / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "inference").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "best_model.pt").write_text(run_id, encoding="utf-8")
    (run_dir / "checkpoints" / "best_mace_model.pt").write_text(run_id, encoding="utf-8")
    (run_dir / "inference" / "sample.png").write_text(run_id, encoding="utf-8")
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        """
domain: corrosion
method: cgan
study:
  name: study_a
variant:
  sensor_set: s11_s21
seed: 1
dataset:
  channels: [S11, S21]
""",
        encoding="utf-8",
    )
    metadata = {
        "domain": "corrosion",
        "method": "cgan",
        "channel_list": ["S11", "S21"],
        "config_identity": {
            "domain": "corrosion",
            "method": "cgan",
            "study": "study_a",
            "sensor_set": "s11_s21",
            "seed": 1,
            "seed_name": "seed_001",
            "channels": ["S11", "S21"],
        },
        "config_identity_key": "corrosion/cgan/study_a/s11_s21/seed_001",
        "metric_summary": {"mace": mace},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "metrics" / "summary.json").write_text(json.dumps({"mace": mace}), encoding="utf-8")
    return run_dir


def test_catalog_registers_latest_run_per_configuration(tmp_path):
    runs_root = tmp_path / "runs"
    older = _write_fake_catalog_run(runs_root, "20260506-111111", mace=6.0)
    newer = _write_fake_catalog_run(runs_root, "20260506-222222", mace=5.2)

    result = generate_catalog(runs_root)

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.selected_run == newer
    assert entry.skipped_runs == [older]
    checkpoint_link = (
        runs_root
        / "catalog"
        / "corrosion"
        / "cgan"
        / "study_a"
        / "checkpoints"
        / "s11_s21"
        / "seed_001"
        / "best_model.pt"
    )
    inference_link = (
        runs_root
        / "catalog"
        / "corrosion"
        / "cgan"
        / "study_a"
        / "inference"
        / "s11_s21"
        / "seed_001"
    )
    assert checkpoint_link.is_symlink()
    assert checkpoint_link.resolve() == newer / "checkpoints" / "best_model.pt"
    assert inference_link.is_symlink()
    registered = json.loads(
        (runs_root / "catalog" / "corrosion" / "cgan" / "study_a" / "registered_runs.json").read_text(
            encoding="utf-8"
        )
    )
    assert registered[0]["selected_run"] == str(newer)
    assert registered[0]["skipped_runs"] == [str(older)]


def test_sweep_config_expands_selected_methods_sensor_sets_and_seeds(tmp_path):
    (tmp_path / "methods").mkdir()
    (tmp_path / "methods" / "cgan.yaml").write_text("method: cgan\narchitecture:\n  latent_dim: 8\n", encoding="utf-8")
    (tmp_path / "methods" / "dit.yaml").write_text("method: dit\narchitecture:\n  depth: 1\n", encoding="utf-8")
    domain_config = tmp_path / "domain.yaml"
    domain_config.write_text(
        """
domain: corrosion
run:
  root: runs
datasets:
  default:
    train_csv: train.csv
    val_csv: val.csv
    test_csv: test.csv
    img_root: datasets
    image_size: 128
sensor_sets:
  s11: [S11]
  s11_s21: [S11, S21]
methods:
  cgan:
    config: methods/cgan.yaml
  dit:
    config: methods/dit.yaml
""",
        encoding="utf-8",
    )
    sweep_config = tmp_path / "sweep.yaml"
    sweep_config.write_text(
        f"""
domain_config: {domain_config}
study:
  name: selected_sweep
selection:
  methods: [cgan, dit]
  sensor_sets: [s11, s11_s21]
  seeds: [1, 3]
stages: [train, infer, evaluate, catalog]
""",
        encoding="utf-8",
    )

    expanded = expand_sweep_config(sweep_config)

    assert len(expanded) == 8
    identity_keys = {config["config_identity_key"] for config in expanded}
    assert "corrosion/cgan/selected_sweep/s11/seed_001" in identity_keys
    assert "corrosion/dit/selected_sweep/s11_s21/seed_003" in identity_keys
    s11_config = next(config for config in expanded if config["variant"]["sensor_set"] == "s11")
    assert s11_config["dataset"]["channels"] == ["S11"]
    assert "recipe" not in s11_config["variant"]
    assert s11_config["loss"]["lambda_l1"] == 100.0
    assert s11_config["loss"]["lambda_ssim"] == 50.0
    assert s11_config["loss"]["lambda_perceptual"] == 10.0
    assert s11_config["conditioning"]["norm_type"] == "batchnorm"
    assert s11_config["stages"] == ["train", "infer", "evaluate", "catalog"]
