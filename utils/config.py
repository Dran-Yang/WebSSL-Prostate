from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import OmegaConf

from dinov2.configs import dinov2_default_config
from dinov2.utils.config import apply_scaling_rules_to_cfg
import torch


def _expanduser_in_cfg(cfg):
    cfg.train.output_dir = str(Path(cfg.train.output_dir).expanduser())
    if "pretrained_weights" in cfg.student and cfg.student.pretrained_weights:
        cfg.student.pretrained_weights = str(
            Path(cfg.student.pretrained_weights).expanduser()
        )
    return cfg


def load_config(
    config_path: str,
    overrides: Sequence[str] | None = None,
    *,
    save_to: Path | None = None,
):
    """
    Load the MM-DINOv2 default SSL configuration, merge it with a custom yaml file,
    and optionally apply CLI-style overrides (e.g. [\"train.seed=42\", \"optim.epochs=100\"]).
    """

    base_cfg = OmegaConf.create(dinov2_default_config)
    custom_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, custom_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(list(overrides)))

    cfg = _expanduser_in_cfg(cfg)
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)

    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()  # type: ignore[attr-defined]
    except Exception:
        cuda_available = False

    fsdp_scaler_available = False
    try:
        from torch.distributed.fsdp.sharded_grad_scaler import (  # noqa: F401
            ShardedGradScaler as _ShardedGradScaler,
        )

        fsdp_scaler_available = True
    except Exception:
        fsdp_scaler_available = False

    if not cuda_available or not fsdp_scaler_available:
        cfg.compute_precision.grad_scaler = False

    cfg = apply_scaling_rules_to_cfg(cfg)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=cfg, f=str(save_to))

    return cfg
