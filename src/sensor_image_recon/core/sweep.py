from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sensor_image_recon.core.catalog import find_latest_run, generate_catalog
from sensor_image_recon.core.config import DEFAULTS, _deep_merge
from sensor_image_recon.core.identity import attach_config_identity


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(base: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return base / path


def load_domain_inventory(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    inventory = _load_yaml(path)
    inventory["_path"] = str(path)
    inventory["_base_dir"] = str(path.parent)
    return inventory


def _available_names(inventory: dict[str, Any], section: str) -> list[str]:
    return sorted(str(name) for name in inventory.get(section, {}).keys())


def _selected_names(selection: Any, available: list[str], label: str) -> list[str]:
    if selection in (None, "all"):
        return available
    if isinstance(selection, str):
        names = [selection]
    else:
        names = list(selection)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return [str(name) for name in names]


def _load_fragment(inventory: dict[str, Any], section: str, name: str) -> dict[str, Any]:
    entry = inventory.get(section, {}).get(name, {})
    if isinstance(entry, dict) and entry.get("config"):
        return _load_yaml(_resolve(Path(inventory["_base_dir"]), entry["config"]))
    return {}


def compose_run_config(
    inventory: dict[str, Any],
    *,
    method: str,
    sensor_set: str,
    seed: int,
    study_name: str,
    stages: list[str],
) -> dict[str, Any]:
    sensors = inventory.get("sensor_sets", {}).get(sensor_set)
    if sensors is None:
        raise ValueError(f"Unknown sensor set: {sensor_set}")

    dataset_name = str(inventory.get("dataset", "default"))
    dataset_cfg = deepcopy(inventory.get("datasets", {}).get(dataset_name, {}))
    dataset_cfg["channels"] = list(sensors)

    config = deepcopy(DEFAULTS)
    config = _deep_merge(
        config,
        {key: deepcopy(inventory[key]) for key in ("domain", "run") if key in inventory},
    )
    config = _deep_merge(config, _load_fragment(inventory, "methods", method))
    config = _deep_merge(
        config,
        {
            "domain": inventory["domain"],
            "method": method,
            "study": {"name": study_name},
            "variant": {
                "sensor_set": sensor_set,
            },
            "seed": int(seed),
            "training": {"seed": int(seed)},
            "dataset": dataset_cfg,
            "stages": list(stages),
        },
    )
    return attach_config_identity(config)


def expand_sweep(sweep_config: dict[str, Any], sweep_path: str | Path | None = None) -> list[dict[str, Any]]:
    if "domain_config" not in sweep_config:
        raise ValueError("Sweep config requires domain_config")
    base_dir = Path(sweep_path).parent if sweep_path is not None else Path.cwd()
    domain_config_path = _resolve(base_dir, sweep_config["domain_config"])
    inventory = load_domain_inventory(domain_config_path)

    selection = sweep_config.get("selection", {})
    methods = _selected_names(selection.get("methods"), _available_names(inventory, "methods"), "methods")
    sensor_sets = _selected_names(
        selection.get("sensor_sets"),
        _available_names(inventory, "sensor_sets"),
        "sensor sets",
    )
    seeds = selection.get("seeds", [1])
    if isinstance(seeds, int):
        seeds = [seeds]
    stages = list(sweep_config.get("stages", ["train", "infer", "evaluate"]))
    study_name = str(sweep_config.get("study", {}).get("name", "sweep"))

    expanded = []
    for method in methods:
        for sensor_set in sensor_sets:
            for seed in seeds:
                expanded.append(
                    compose_run_config(
                        inventory,
                        method=method,
                        sensor_set=sensor_set,
                        seed=int(seed),
                        study_name=study_name,
                        stages=stages,
                    )
                )
    return expanded


def expand_sweep_config(path: str | Path) -> list[dict[str, Any]]:
    return expand_sweep(_load_yaml(path), sweep_path=path)


def write_sweep_config(
    *,
    domain_config: str,
    study_name: str,
    methods: list[str] | str,
    sensor_sets: list[str] | str,
    seeds: list[int],
    stages: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain_config": domain_config,
        "study": {"name": study_name},
        "selection": {
            "methods": methods,
            "sensor_sets": sensor_sets,
            "seeds": seeds,
        },
        "stages": stages,
        "execution": {
            "skip_existing": True,
            "catalog_at_end": "catalog" in stages,
        },
    }
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return output_path


def run_sweep_config(path: str | Path) -> list[Path]:
    sweep_path = Path(path)
    sweep_config = _load_yaml(sweep_path)
    expanded = expand_sweep(sweep_config, sweep_path=sweep_path)
    execution = sweep_config.get("execution", {})
    skip_existing = bool(execution.get("skip_existing", True))
    run_dirs: list[Path] = []

    from sensor_image_recon.methods.registry import get_method

    for config in expanded:
        stages = list(config.get("stages", sweep_config.get("stages", ["train", "infer", "evaluate"])))
        runs_root = config.get("run", {}).get("root", "runs")
        run_dir = find_latest_run(runs_root, config["config_identity_key"]) if skip_existing else None
        if "train" in stages and run_dir is None:
            run_dir = get_method(config["method"]).train(config)
        if run_dir is None:
            raise FileNotFoundError(f"No existing run for {config['config_identity_key']}")
        run_dirs.append(Path(run_dir))
        method = get_method(config["method"])
        if "infer" in stages:
            method.infer(run_dir, None)
        if "evaluate" in stages:
            metrics = method.evaluate(run_dir)
            metrics_dir = Path(run_dir) / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / "summary.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    if "catalog" in sweep_config.get("stages", []) or execution.get("catalog_at_end"):
        runs_roots = sorted({str(config.get("run", {}).get("root", "runs")) for config in expanded})
        for runs_root in runs_roots:
            generate_catalog(runs_root)
    return run_dirs
