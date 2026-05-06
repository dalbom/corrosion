#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sensor_image_recon.core.sweep import load_domain_inventory, write_sweep_config


def _split_csv(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if value == "all":
        return "all"
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _prompt_selection(label: str, available: list[str], default: str = "all"):
    print(f"{label}:")
    for index, name in enumerate(available, start=1):
        print(f"  {index}. {name}")
    raw = input(f"Select {label.lower()} [default: {default}; use all, names, or numbers]: ").strip()
    if not raw:
        raw = default
    if raw == "all":
        return "all"
    selected = []
    for item in raw.split(","):
        item = item.strip()
        if item.isdigit():
            selected.append(available[int(item) - 1])
        else:
            selected.append(item)
    return selected


def _prompt_list(prompt: str, default: str) -> str:
    value = input(f"{prompt} [default: {default}]: ").strip()
    return value or default


def _relative_reference(output_path: Path, source_path: Path) -> str:
    try:
        return os.path.relpath(source_path, start=output_path.parent)
    except ValueError:
        return str(source_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interview the user and write a sweep config.")
    parser.add_argument("--domain-config", default="configs/corrosion/domain.yaml")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated method names or all.")
    parser.add_argument("--sensor-sets", default=None, help="Comma-separated sensor set names or all.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--stages", default=None, help="Comma-separated stages, e.g. train,infer,evaluate,catalog.")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    domain_config = Path(args.domain_config)
    inventory = load_domain_inventory(domain_config)
    methods_available = sorted(inventory.get("methods", {}).keys())
    sensors_available = sorted(inventory.get("sensor_sets", {}).keys())

    study_name = args.study_name or _prompt_list("Study name", f"{inventory['domain']}_sweep")
    methods = _split_csv(args.methods)
    if methods is None:
        methods = _prompt_selection("Methods", methods_available)
    sensor_sets = _split_csv(args.sensor_sets)
    if sensor_sets is None:
        sensor_sets = _prompt_selection("Sensor sets", sensors_available)
    seeds = _split_ints(args.seeds)
    if seeds is None:
        seeds = _split_ints(_prompt_list("Seeds", "1"))
    stages = _split_csv(args.stages)
    if stages is None:
        stages = _split_csv(_prompt_list("Stages", "train,infer,evaluate,catalog"))
    if stages == "all":
        stages = ["train", "infer", "evaluate", "catalog"]

    output = Path(args.output) if args.output else Path("configs") / "sweeps" / inventory["domain"] / f"{study_name}.yaml"
    domain_ref = _relative_reference(output, domain_config)
    path = write_sweep_config(
        domain_config=domain_ref,
        study_name=study_name,
        methods=methods,
        sensor_sets=sensor_sets,
        seeds=seeds or [1],
        stages=stages,
        output_path=output,
    )
    print(path)


if __name__ == "__main__":
    main()
