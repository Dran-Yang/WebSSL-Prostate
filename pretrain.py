"""
Utilities for running Web-DINO style MRI pretraining.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from prostate.datasets.mri_slices import build_mri_dataset
from prostate.losses.dino_loss import DINOLoss
from prostate.model.build_webdino import build_webdino
from prostate.utils.augment import build_mri_dino_augmentor


def load_config(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data


def resolve_env_path(path_value: str) -> str:
    if not isinstance(path_value, str):
        return str(path_value)
    if path_value.startswith("${env:") and path_value.endswith("}"):
        inner = path_value[6:-1]
        if ":" in inner:
            env_name, default = inner.split(":", 1)
        else:
            env_name, default = inner, ""
        return os.path.expanduser(os.environ.get(env_name, default))
    return os.path.expanduser(path_value)


def init_distributed(cfg: Dict[str, object]) -> Tuple[int, int]:
    world_size = int(cfg.get("world_size", 1))
    if world_size <= 1:
        return 0, 1

    rank = int(os.environ.get("RANK", 0))
    dist_url = cfg.get("dist_url", "env://")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> List[float]:
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep

    schedule = []
    if warmup_iters > 0:
        for it in range(warmup_iters):
            progress = (it + 1) / max(1, warmup_iters)
            value = start_warmup_value + (base_value - start_warmup_value) * progress
            schedule.append(value)
    for it in range(total_iters - warmup_iters):
        progress = it / max(1, total_iters - warmup_iters)
        value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        schedule.append(value)
    return schedule


def momentum_schedule(
    base_m: float,
    total_iters: int,
    warmup_iters: int = 0,
) -> List[float]:
    schedule: List[float] = []
    for t in range(total_iters):
        if t < warmup_iters:
            progress = t / max(1, warmup_iters)
            schedule.append(1.0 - (1.0 - base_m) * progress)
            continue
        progress = (t - warmup_iters) / max(1, total_iters - warmup_iters)
        value = 1 - (1 - base_m) * (math.cos(math.pi * progress) + 1) / 2
        schedule.append(value)
    return schedule


def create_dataloader(
    dataset_cfg: Dict[str, object],
    batch_size: int,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = build_mri_dataset(dataset_cfg)

    dataloader_cfg = dict(dataset_cfg.get("dataloader") or {})
    shuffle = bool(dataloader_cfg.get("shuffle", True))
    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(dataloader_cfg.get("num_workers", 8)),
        pin_memory=bool(dataloader_cfg.get("pin_memory", True)),
        persistent_workers=bool(dataloader_cfg.get("persistent_workers", True)),
        prefetch_factor=int(dataloader_cfg.get("prefetch_factor", 2)),
        drop_last=bool(dataloader_cfg.get("drop_last", True)),
    )
    return loader, sampler


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    cfg: Dict[str, object],
) -> optim.Optimizer:
    optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("base_lr", 5e-4))
    weight_decay = float(cfg.get("weight_decay", 0.04))

    if optimizer_name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.95)))
        optimizer = optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == "lion":
        try:
            from lion_pytorch import Lion  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("lion-pytorch not installed") from exc
        optimizer = Lion(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer


def adjust_weight_decay(optimizer: optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["weight_decay"] = value


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    lr_schedule: List[float],
    wd_schedule: List[float],
    momentum_values: List[float],
    augmentor,
    criterion: DINOLoss,
    scaler: GradScaler,
    device: torch.device,
    precision: str,
    epoch: int,
    print_freq: int,
    clip_grad: float,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    num_steps = len(dataloader)

    start_time = time.time()
    for step, batch in enumerate(dataloader):
        global_step = epoch * num_steps + step
        x = batch["image"]

        # Multi-crop augmentation across the batch
        views: List[List[torch.Tensor]] = []
        total_views = augmentor.num_global_crops + augmentor.num_local_crops
        for _ in range(total_views):
            views.append([])

        for sample in x:
            sample_cpu = sample.detach().cpu()
            crops = augmentor(sample_cpu)
            for idx, crop in enumerate(crops):
                views[idx].append(crop)

        # Stack each view into batch tensors
        student_inputs: List[torch.Tensor] = []
        for crops in views:
            batch_tensor = torch.stack(crops, dim=0).to(device, non_blocking=True)
            student_inputs.append(batch_tensor)
        teacher_inputs = student_inputs[: augmentor.num_global_crops]

        # Adjust learning rate and weight decay
        lr = lr_schedule[global_step]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        adjust_weight_decay(optimizer, wd_schedule[global_step])

        use_amp = precision in {"amp_fp16", "amp_bfloat16"}
        autocast_dtype = torch.float16 if precision == "amp_fp16" else torch.bfloat16

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp, dtype=autocast_dtype):
            student_outputs = [model.forward_student(inp) for inp in student_inputs]
            with torch.no_grad():
                teacher_outputs = [model.forward_teacher(inp) for inp in teacher_inputs]
            loss = criterion(student_outputs, teacher_outputs, epoch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.student_backbone.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        # Update teacher
        momentum = momentum_values[global_step]
        model.update_teacher(momentum)

        running_loss += loss.item()

        if (step + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / (step + 1)
            print(
                f"Epoch[{epoch+1}] Step[{step+1}/{num_steps}] "
                f"Loss: {avg_loss:.4f} LR: {lr:.6f} Time/step: {elapsed / (step + 1):.3f}s"
            )

    avg_loss = running_loss / num_steps
    return avg_loss, lr_schedule[epoch * num_steps : (epoch + 1) * num_steps][-1]


def run_pretrain(cfg: Dict[str, object]) -> None:
    distributed_cfg = dict(cfg.get("distributed") or {})
    rank, world_size = init_distributed(distributed_cfg)

    experiment_cfg = dict(cfg.get("experiment") or {})
    output_dir = Path(experiment_cfg.get("output_dir", "outputs/pretrain"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = dict(cfg.get("dataset") or {})
    dataset_root = resolve_env_path(str(dataset_cfg.get("root")))
    dataset_cfg["root"] = dataset_root

    dataloader_cfg = dict(dataset_cfg.get("dataloader") or {})
    batch_size = int(dataloader_cfg.get("batch_size", 32))
    dataloader, sampler = create_dataloader(
        dataset_cfg,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )

    image_size = int(dataset_cfg.get("image_size", 224))
    augment_cfg = dict(cfg.get("augmentations") or {})
    augmentor = build_mri_dino_augmentor(augment_cfg, image_size=image_size)

    model_cfg = dict(cfg.get("model") or {})
    model = build_webdino(model_cfg, image_size=image_size)
    model.to(device)

    optimization_cfg = dict(cfg.get("optimization") or {})
    optimizer = build_optimizer(model.student_backbone.parameters(), optimization_cfg)

    epochs = int(optimization_cfg.get("epochs", 100))
    niter = len(dataloader)
    if niter == 0:
        raise RuntimeError("Empty dataloader. Ensure dataset contains samples.")

    lr_schedule = cosine_scheduler(
        base_value=float(optimization_cfg.get("base_lr", 5e-4)),
        final_value=float(optimization_cfg.get("final_lr", 1e-5)),
        epochs=epochs,
        niter_per_ep=niter,
        warmup_epochs=int(optimization_cfg.get("warmup_epochs", 10)),
        start_warmup_value=float(optimization_cfg.get("warmup_lr", 1e-6)),
    )
    wd_schedule = cosine_scheduler(
        base_value=float(optimization_cfg.get("weight_decay", 0.04)),
        final_value=float(optimization_cfg.get("weight_decay_end", 0.4)),
        epochs=epochs,
        niter_per_ep=niter,
        warmup_epochs=0,
        start_warmup_value=float(optimization_cfg.get("weight_decay", 0.04)),
    )

    momentum_base = float(model.momentum_schedule["base"])
    warmup_epochs = int(model.momentum_schedule.get("warmup_epochs", 0))  # type: ignore[attr-defined]
    momentum_values = momentum_schedule(
        momentum_base,
        epochs * niter,
        warmup_iters=warmup_epochs * niter,
    )

    temperature_cfg = dict(cfg.get("temperature") or {})
    criterion = DINOLoss(
        out_dim=int(model_cfg.get("out_dim", 65536)),
        warmup_teacher_temp=float(temperature_cfg.get("teacher_base", 0.04)),
        teacher_temp=float(temperature_cfg.get("teacher_final", 0.04)),
        warmup_teacher_epochs=int(temperature_cfg.get("warmup_epochs", 30)),
        total_epochs=epochs,
        student_temp=float(temperature_cfg.get("student", 0.1)),
    ).to(device)

    precision = str(experiment_cfg.get("precision", "amp_bfloat16"))
    scaler = GradScaler(enabled=precision == "amp_fp16")

    print_freq = int(cfg.get("logging", {}).get("log_every", 20))

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss, current_lr = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_values=momentum_values,
            augmentor=augmentor,
            criterion=criterion,
            scaler=scaler,
            device=device,
            precision=precision,
            epoch=epoch,
            print_freq=print_freq,
            clip_grad=float(optimization_cfg.get("clip_grad", 0.0)),
        )
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} LR: {current_lr:.6f}")
            # save_checkpoint(model, optimizer, epoch, output_dir)    #保存每个epoch的checkpoint
    # # 循环结束后只保存一次
    # if rank == 0:
    #     print("Training finished, saving final checkpoint...")
    #     save_checkpoint(model, optimizer, epoch, output_dir)

# def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, output_dir: Path) -> None:
#     checkpoint_dir = output_dir / "checkpoints"
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     state = {
#         "epoch": epoch + 1,
#         "student": model.student_backbone.state_dict(),
#         "teacher": model.teacher_backbone.state_dict(),
#         "student_head": model.student_head.state_dict(),
#         "teacher_head": model.teacher_head.state_dict(),
#         "optimizer": optimizer.state_dict(),
#     }
#     torch.save(state, checkpoint_dir / f"checkpoint_{epoch+1:03d}.pth")
