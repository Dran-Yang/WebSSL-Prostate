from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

from prostate.utils.pretrain import load_config, run_pretrain


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web-DINO MRI pretraining entry point.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("D:\Project\Web-Dino-SSL\prostate\configs\pretrain.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        "-o",
        metavar="KEY=VALUE",
        nargs="*",
        help="Override configuration entries using dotted keys.",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Use KEY=VALUE format.")
        key, value = override.split("=", 1)
        key_parts = key.split(".")
        target = cfg
        for part in key_parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[key_parts[-1]] = yaml.safe_load(value)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config_path: Path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = load_config(config_path)

    if args.override:
        apply_overrides(cfg, args.override)

    run_pretrain(cfg)


if __name__ == "__main__":
    main()
