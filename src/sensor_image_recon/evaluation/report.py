from __future__ import annotations

from pathlib import Path

import pandas as pd

from sensor_image_recon.domains import get_domain_adapter
from sensor_image_recon.evaluation.metrics import aggregate, image_metrics, load_red01


def evaluate_generated(config: dict, generated_root: str | Path, out_csv: str | Path | None = None) -> dict:
    if config["domain"] != "corrosion":
        raise NotImplementedError("Raw-image evaluation is currently implemented for corrosion")
    dataset_cfg = config["dataset"]
    adapter = get_domain_adapter(config["domain"], dataset_cfg)
    csv_path = dataset_cfg.get("test_csv") or dataset_cfg.get("val_csv") or dataset_cfg.get("train_csv")
    df = pd.read_csv(csv_path)
    generated_root = Path(generated_root)
    rows = []
    for _, row in df.iterrows():
        real_path = adapter.target_path(row)
        filename = str(row["filename"])
        filename_with_ext = filename if filename.endswith(".png") else f"{filename}.png"
        sample_id = filename.replace(".png", "").split("_")[1]
        gen_path = generated_root / sample_id / filename_with_ext
        if not real_path.exists() or not gen_path.exists():
            continue
        metrics = image_metrics(load_red01(real_path), load_red01(gen_path))
        rows.append({"filename": filename, "sample_index": sample_id, **metrics})
    if not rows:
        raise FileNotFoundError(f"No comparable raw generated images found under {generated_root}")
    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
    return aggregate(rows)
