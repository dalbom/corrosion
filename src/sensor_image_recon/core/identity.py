from __future__ import annotations

import re
from typing import Any


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "default"


def seed_name(seed: int | str) -> str:
    return f"seed_{int(seed):03d}"


def sensor_set_name(config: dict[str, Any]) -> str:
    variant = config.get("variant", {})
    if variant.get("sensor_set"):
        return slugify(str(variant["sensor_set"]))
    channels = config.get("dataset", {}).get("channels", [])
    return slugify("_".join(str(channel) for channel in channels) or "no_sensors")


def study_name(config: dict[str, Any]) -> str:
    study = config.get("study", {})
    if study.get("name"):
        return slugify(str(study["name"]))
    return slugify(str(config.get("experiment", "default")))


def build_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", config.get("training", {}).get("seed", 0)))
    identity = {
        "domain": slugify(str(config["domain"])),
        "method": slugify(str(config["method"])),
        "study": study_name(config),
        "sensor_set": sensor_set_name(config),
        "seed": seed,
        "seed_name": seed_name(seed),
        "channels": list(config.get("dataset", {}).get("channels", [])),
    }
    dataset = config.get("dataset", {})
    for key in ("train_csv", "val_csv", "test_csv"):
        if dataset.get(key):
            identity[key] = str(dataset[key])
    return identity


def normalize_config_identity(identity: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(identity)
    normalized.pop("recipe", None)
    return normalized


def config_identity_key(identity: dict[str, Any]) -> str:
    identity = normalize_config_identity(identity)
    return "/".join(
        [
            str(identity["domain"]),
            str(identity["method"]),
            str(identity["study"]),
            str(identity["sensor_set"]),
            str(identity["seed_name"]),
        ]
    )


def attach_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    identity = build_config_identity(config)
    config["config_identity"] = identity
    config["config_identity_key"] = config_identity_key(identity)
    return config
