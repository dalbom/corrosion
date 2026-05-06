from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_image_recon.core.config import load_config
from sensor_image_recon.evaluation.report import evaluate_generated
from sensor_image_recon.methods.registry import get_method


def _cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = get_method(config["method"]).train(config)
    print(run_dir)


def _cmd_infer(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    method = metadata["method"]
    output_dir = get_method(method).infer(run_dir, args.output_dir)
    print(output_dir)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    config = load_config(run_dir / "config.yaml")
    generated_root = Path(args.generated_root) if args.generated_root else run_dir / "inference"
    out_csv = run_dir / "metrics" / "raw_metrics.csv"
    metrics = evaluate_generated(config, generated_root, out_csv)
    (run_dir / "metrics" / "summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sensor_image_recon.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train from a YAML config")
    train.add_argument("--config", required=True)
    train.set_defaults(func=_cmd_train)

    infer = sub.add_parser("infer", help="Generate raw images from a run")
    infer.add_argument("--run", required=True)
    infer.add_argument("--output-dir", default=None)
    infer.set_defaults(func=_cmd_infer)

    evaluate = sub.add_parser("evaluate", help="Evaluate raw generated images")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--generated-root", default=None)
    evaluate.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
