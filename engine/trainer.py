from __future__ import annotations

import json
import logging
import math
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from dinov2.data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast
from dinov2.logging import MetricLogger
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.utils import CosineScheduler

try:
    # Prefer absolute import when package is installed
    from pre_ssl_prostate.data.prostate_ssl_dataset import build_prostate_ssl_dataset
except ImportError:
    # Fallback to a relative import when running from the source tree
    from ..data.prostate_ssl_dataset import build_prostate_ssl_dataset

logger = logging.getLogger("pre_ssl_prostate.trainer")


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )


def build_schedulers(cfg):
    official_epoch_length = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * official_epoch_length,
        warmup_iters=cfg.optim["warmup_epochs"] * official_epoch_length,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * official_epoch_length,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * official_epoch_length,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * official_epoch_length,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * official_epoch_length,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)
    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * official_epoch_length
    ] = 0

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


class ProstateSSLTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for SSL pretraining. Please ensure a CUDA capable "
                "device is available or adjust the configuration accordingly."
            )
        self.device = torch.device("cuda")
        self.output_dir = Path(cfg.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "training_metrics.jsonl"
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SSLMetaArch] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_schedule = None
        self.wd_schedule = None
        self.momentum_schedule = None
        self.teacher_temp_schedule = None
        self.last_layer_lr_schedule = None
        self.dataloader: Optional[DataLoader] = None
        self.mask_generator = None

    def _build_data_transform(self):
        cfg = self.cfg
        return DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            intensity_aug_name=cfg.crops.intensity_aug,
            crop_from_tumor_foreground=cfg.crops.crop_from_tumor_foreground,
            max_blur_radius=cfg.crops.max_blur_radius,
            gamma_range=cfg.crops.gamma_range,
        )

    def _build_collate_fn(self, n_tokens, inputs_dtype):
        cfg = self.cfg
        mask_generator = MaskingGenerator(
            input_size=(cfg.crops.global_crops_size // cfg.student.patch_size,) * 2,
            max_num_patches=n_tokens // 2,
        )
        self.mask_generator = mask_generator
        collate_fn = partial(
            collate_data_and_cast,
            mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
            mask_probability=cfg.ibot.mask_sample_probability,
            mask_per_channel=cfg.ibot.mask_per_channel,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            dtype=inputs_dtype,
        )
        return collate_fn

    def build_dataloader(self):
        cfg = self.cfg
        transform = self._build_data_transform()
        global_patch_tokens = cfg.crops.global_crops_size // cfg.student.patch_size
        n_tokens = int(global_patch_tokens**2)
        inputs_dtype = torch.float16

        collate_fn = self._build_collate_fn(n_tokens, inputs_dtype)
        dataset = build_prostate_ssl_dataset(
            cfg.train.dataset_path,
            transform=transform,
        )

        dataset_size = len(dataset)
        steps_per_epoch = max(1, math.ceil(dataset_size / cfg.train.batch_size_per_gpu))
        if bool(getattr(cfg.train, "auto_epoch_length", False)):
            cfg.train.OFFICIAL_EPOCH_LENGTH = steps_per_epoch
            logger.info(
                "Auto OFFICIAL_EPOCH_LENGTH set to %d steps (dataset=%d, batch=%d).",
                cfg.train.OFFICIAL_EPOCH_LENGTH,
                dataset_size,
                cfg.train.batch_size_per_gpu,
            )
        else:
            logger.info(
                "Dataset size=%d, batch=%d, steps_per_epoch=%d (manual OFFICIAL_EPOCH_LENGTH=%d).",
                dataset_size,
                cfg.train.batch_size_per_gpu,
                steps_per_epoch,
                cfg.train.OFFICIAL_EPOCH_LENGTH,
            )

        persistent_workers = bool(getattr(cfg.train, "persistent_workers", False)) and cfg.train.num_workers > 0
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )

        prefetch_factor = getattr(cfg.train, "prefetch_factor", None)
        if prefetch_factor is not None and cfg.train.num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

        loader = DataLoader(**loader_kwargs)
        self.dataloader = loader
        return loader

    def build_model(self):
        cfg = self.cfg
        model = SSLMetaArch(cfg).to(self.device)

        if cfg.optim.freeze_backbone_epochs > 0:
            for param in model.student.backbone.parameters():
                param.requires_grad = False

        self.model = model
        return model

    def _log_metrics(self, iteration: int, metrics: Dict[str, float]):
        record = {"iteration": iteration}
        record.update(metrics)
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_checkpoint(self, iteration: int):
        if self.model is None or self.optimizer is None:
            return
        state = {
            "iteration": iteration,
            "student": self.model.student.state_dict(),
            "teacher": self.model.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        ckpt_path = self.ckpt_dir / f"iter_{iteration:07d}.pth"
        torch.save(state, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)

    def train(self):
        cfg = self.cfg
        model = self.build_model()
        data_loader = self.build_dataloader()

        optimizer = build_optimizer(cfg, model.get_params_groups())
        (
            self.lr_schedule,
            self.wd_schedule,
            self.momentum_schedule,
            self.teacher_temp_schedule,
            self.last_layer_lr_schedule,
        ) = build_schedulers(cfg)

        self.optimizer = optimizer
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", ":", "{:.6f}")
        metric_logger.add_meter("wd", ":", "{:.6f}")
        metric_logger.add_meter("total_loss", ":", "{:.4f}")

        max_iter = cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH
        save_period = cfg.train.saveckp_freq * cfg.train.OFFICIAL_EPOCH_LENGTH
        log_period = getattr(cfg.logging, "log_period_iterations", 100)

        data_iterator = iter(cycle(data_loader))
        scaler = model.fp16_scaler
        freeze_backbone_iters = cfg.optim.freeze_backbone_epochs * cfg.train.OFFICIAL_EPOCH_LENGTH

        for iteration in range(max_iter):
            batch = next(data_iterator)
            lr = self.lr_schedule[iteration]
            wd = self.wd_schedule[iteration]
            momentum = self.momentum_schedule[iteration]
            teacher_temp = self.teacher_temp_schedule[iteration]
            last_layer_lr = self.last_layer_lr_schedule[iteration]

            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

            if iteration == freeze_backbone_iters:
                logger.info("Unfreezing student backbone parameters.")
                for param in model.student.backbone.parameters():
                    param.requires_grad = True

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.forward_backward(batch, teacher_temp=teacher_temp)

            if scaler is not None:
                if cfg.optim.clip_grad:
                    scaler.unscale_(optimizer)
                    for module in model.student.values():
                        torch.nn.utils.clip_grad_norm_(
                            module.parameters(), cfg.optim.clip_grad
                        )
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.optim.clip_grad:
                    for module in model.student.values():
                        torch.nn.utils.clip_grad_norm_(
                            module.parameters(), cfg.optim.clip_grad
                        )
                optimizer.step()

            model.update_teacher(momentum)

            loss_value = sum(v.item() for v in loss_dict.values())
            metric_logger.update(lr=lr, wd=wd, total_loss=loss_value)

            if (iteration + 1) % log_period == 0 or iteration == 0:
                metrics = {k: float(v) for k, v in loss_dict.items()}
                metrics.update({"lr": lr, "wd": wd, "total_loss": loss_value})
                self._log_metrics(iteration + 1, metrics)
                logger.info(
                    "Iter [%d/%d] total_loss=%.4f lr=%.6f wd=%.6f",
                    iteration + 1,
                    max_iter,
                    loss_value,
                    lr,
                    wd,
                )

            if save_period > 0 and (iteration + 1) % save_period == 0:
                self.save_checkpoint(iteration + 1)

        # final checkpoint
        self.save_checkpoint(max_iter)
        logger.info("Training complete.")
