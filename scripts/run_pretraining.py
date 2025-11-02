from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dinov2.logging import setup_logging

from pre_ssl_prostate.engine.trainer import ProstateSSLTrainer
from pre_ssl_prostate.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Prostate SSL pretraining runner")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path("pre-ssl-prostate/configs/prostate_ssl.yaml").resolve()
        ),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional overrides in the format KEY=VALUE.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts, save_to=Path(args.config).with_suffix(".resolved.yaml"))

    setup_logging(output=cfg.train.output_dir, name="pre_ssl_prostate", level=logging.INFO)

    trainer = ProstateSSLTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
