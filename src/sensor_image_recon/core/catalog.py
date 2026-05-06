from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sensor_image_recon.core.config import load_config
from sensor_image_recon.core.identity import build_config_identity, config_identity_key


@dataclass(frozen=True)
class CatalogEntry:
    identity_key: str
    identity: dict[str, Any]
    selected_run: Path
    skipped_runs: list[Path]
    metrics: dict[str, Any]
    symlinks: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class CatalogResult:
    catalog_root: Path
    entries: list[CatalogEntry]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _run_sort_key(run_dir: Path) -> tuple[float, str]:
    try:
        stamp = datetime.strptime(run_dir.name, "%Y%m%d-%H%M%S").timestamp()
    except ValueError:
        stamp = run_dir.stat().st_mtime
    return stamp, run_dir.name


def _run_record(metadata_path: Path) -> dict[str, Any] | None:
    run_dir = metadata_path.parent
    config_path = run_dir / "config.yaml"
    metadata = _read_json(metadata_path)
    if not metadata and not config_path.exists():
        return None
    config: dict[str, Any] = {}
    if config_path.exists():
        config = load_config(config_path)
    identity = metadata.get("config_identity")
    if not identity and config:
        identity = build_config_identity(config)
    if not identity:
        return None
    key = metadata.get("config_identity_key") or config_identity_key(identity)
    summary = _read_json(run_dir / "metrics" / "summary.json")
    metrics = summary or metadata.get("metric_summary", {})
    return {
        "key": key,
        "identity": identity,
        "run_dir": run_dir,
        "metrics": metrics,
    }


def discover_run_records(
    runs_root: str | Path,
    *,
    domain: str | None = None,
    method: str | None = None,
) -> list[dict[str, Any]]:
    runs_root = Path(runs_root)
    catalog_root = runs_root / "catalog"
    records = []
    for metadata_path in runs_root.rglob("metadata.json"):
        if catalog_root in metadata_path.parents:
            continue
        record = _run_record(metadata_path)
        if record is None:
            continue
        identity = record["identity"]
        if domain is not None and identity.get("domain") != domain:
            continue
        if method is not None and identity.get("method") != method:
            continue
        records.append(record)
    return records


def find_latest_run(
    runs_root: str | Path,
    identity_key: str,
) -> Path | None:
    records = [
        record
        for record in discover_run_records(runs_root)
        if record["key"] == identity_key
    ]
    if not records:
        return None
    records.sort(key=lambda record: _run_sort_key(record["run_dir"]), reverse=True)
    return records[0]["run_dir"]


def _replace_link(link: Path, target: Path) -> Path | None:
    if not target.exists():
        return None
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    relative_target = os.path.relpath(target, start=link.parent)
    link.symlink_to(relative_target, target_is_directory=target.is_dir())
    return link


def _catalog_base(catalog_root: Path, identity: dict[str, Any]) -> Path:
    return catalog_root / identity["domain"] / identity["method"] / identity["study"]


def _entry_symlinks(base: Path, identity: dict[str, Any], run_dir: Path) -> list[Path]:
    leaf = Path(identity["sensor_set"]) / identity["recipe"] / identity["seed_name"]
    links: list[Path] = []
    for checkpoint_name in ("best_model.pt", "best_mace_model.pt", "last_model.pt"):
        link = _replace_link(
            base / "checkpoints" / leaf / checkpoint_name,
            run_dir / "checkpoints" / checkpoint_name,
        )
        if link is not None:
            links.append(link)
    for folder in ("inference", "samples"):
        link = _replace_link(base / folder / leaf, run_dir / folder)
        if link is not None:
            links.append(link)
    link = _replace_link(base / "runs" / leaf, run_dir)
    if link is not None:
        links.append(link)
    return links


def _write_catalog_files(catalog_root: Path, entries: list[CatalogEntry]) -> None:
    grouped: dict[Path, list[CatalogEntry]] = {}
    for entry in entries:
        grouped.setdefault(_catalog_base(catalog_root, entry.identity), []).append(entry)

    for base, base_entries in grouped.items():
        base.mkdir(parents=True, exist_ok=True)
        serializable = [
            {
                "identity_key": entry.identity_key,
                "identity": entry.identity,
                "selected_run": str(entry.selected_run),
                "skipped_runs": [str(path) for path in entry.skipped_runs],
                "metrics": entry.metrics,
                "symlinks": [str(path) for path in entry.symlinks],
            }
            for entry in sorted(base_entries, key=lambda item: item.identity_key)
        ]
        (base / "registered_runs.json").write_text(
            json.dumps(serializable, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        fieldnames = [
            "identity_key",
            "domain",
            "method",
            "study",
            "sensor_set",
            "recipe",
            "seed",
            "selected_run",
            "mace",
            "mae",
            "mse",
            "psnr",
            "ssim",
        ]
        with (base / "leaderboard.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for entry in sorted(base_entries, key=lambda item: item.identity_key):
                row = {
                    "identity_key": entry.identity_key,
                    "selected_run": str(entry.selected_run),
                    **{
                        key: entry.identity.get(key)
                        for key in ("domain", "method", "study", "sensor_set", "recipe", "seed")
                    },
                }
                for metric in ("mace", "mae", "mse", "psnr", "ssim"):
                    row[metric] = entry.metrics.get(metric)
                writer.writerow(row)


def generate_catalog(
    runs_root: str | Path = "runs",
    *,
    domain: str | None = None,
    method: str | None = None,
    refresh: bool = True,
) -> CatalogResult:
    runs_root = Path(runs_root)
    catalog_root = runs_root / "catalog"
    if refresh and catalog_root.exists() and domain is None and method is None:
        shutil.rmtree(catalog_root)
    elif refresh and catalog_root.exists() and domain is not None and method is not None:
        shutil.rmtree(catalog_root / domain / method, ignore_errors=True)
    elif refresh and catalog_root.exists() and domain is not None:
        shutil.rmtree(catalog_root / domain, ignore_errors=True)
    elif refresh and catalog_root.exists() and method is not None:
        for domain_dir in catalog_root.iterdir():
            if domain_dir.is_dir():
                shutil.rmtree(domain_dir / method, ignore_errors=True)
    catalog_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in discover_run_records(runs_root, domain=domain, method=method):
        grouped.setdefault(record["key"], []).append(record)

    entries: list[CatalogEntry] = []
    for key, records in sorted(grouped.items()):
        records.sort(key=lambda record: _run_sort_key(record["run_dir"]), reverse=True)
        selected = records[0]
        skipped = [record["run_dir"] for record in records[1:]]
        identity = selected["identity"]
        base = _catalog_base(catalog_root, identity)
        symlinks = _entry_symlinks(base, identity, selected["run_dir"])
        entries.append(
            CatalogEntry(
                identity_key=key,
                identity=identity,
                selected_run=selected["run_dir"],
                skipped_runs=skipped,
                metrics=selected["metrics"],
                symlinks=symlinks,
            )
        )

    _write_catalog_files(catalog_root, entries)
    return CatalogResult(catalog_root=catalog_root, entries=entries)


def format_catalog_result(result: CatalogResult) -> str:
    lines = [f"Catalog root: {result.catalog_root}", "Registered catalog entries:"]
    if not result.entries:
        lines.append("  none")
        return "\n".join(lines)
    for entry in result.entries:
        lines.append(entry.identity_key)
        lines.append(f"  selected: {entry.selected_run}")
        if entry.skipped_runs:
            lines.append("  skipped older:")
            for run in entry.skipped_runs:
                lines.append(f"    {run}")
        if "mace" in entry.metrics:
            lines.append(f"  mace: {entry.metrics['mace']}")
        lines.append(f"  symlinks: {len(entry.symlinks)}")
    return "\n".join(lines)
