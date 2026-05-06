#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sensor_image_recon.core.catalog import format_catalog_result, generate_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate symlink catalog views from run directories.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_catalog(
        args.runs_root,
        domain=args.domain,
        method=args.method,
        refresh=not args.no_refresh,
    )
    print(format_catalog_result(result))


if __name__ == "__main__":
    main()
