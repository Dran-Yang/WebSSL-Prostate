"""
Multi-view augmentations tailored for prostate MRI self-supervised pretraining.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import (
    gaussian_blur,
    resize,
    rotate,
)
from torchvision.transforms.functional import InterpolationMode


def _clamp_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _random_resized_crop(
    image: torch.Tensor,
    size: int,
    scale: Tuple[float, float],
    ratio: Tuple[float, float] = (0.75, 1.33),
) -> torch.Tensor:
    _, h, w = image.shape
    area = h * w

    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect = random.uniform(ratio[0], ratio[1])

        new_h = int(round(math.sqrt(target_area / aspect)))
        new_w = int(round(math.sqrt(target_area * aspect)))

        if 0 < new_h <= h and 0 < new_w <= w:
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            cropped = image[:, top : top + new_h, left : left + new_w]
            return resize(
                cropped,
                size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

    # Fallback to central crop
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    cropped = image[:, start_h : start_h + min_dim, start_w : start_w + min_dim]
    return resize(
        cropped,
        size,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def _maybe_horizontal_flip(
    image: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    if random.random() < probability:
        return torch.flip(image, dims=(2,))
    return image


def _maybe_rotate(image: torch.Tensor, max_degrees: float) -> torch.Tensor:
    if max_degrees <= 0:
        return image
    angle = random.uniform(-max_degrees, max_degrees)
    return rotate(
        image,
        angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )


def _intensity_jitter(
    image: torch.Tensor,
    contrast: float,
    brightness: float,
    gamma: float,
) -> torch.Tensor:
    out = image
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = out.mean(dim=(1, 2), keepdim=True)
        out = (out - mean) * factor + mean
    if brightness > 0:
        offset = random.uniform(-brightness, brightness)
        out = out + offset
    if gamma > 0:
        gamma_factor = 1.0 + random.uniform(-gamma, gamma)
        gamma_factor = max(0.1, gamma_factor)
        out = torch.pow(_clamp_tensor(out), gamma_factor)
    return _clamp_tensor(out)


def _maybe_blur(image: torch.Tensor, probability: float, kernel: int = 15) -> torch.Tensor:
    if random.random() < probability:
        kernel_size = kernel if kernel % 2 == 1 else kernel + 1
        sigma = random.uniform(0.1, 2.0)
        return gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return image


def _add_noise(image: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return image
    noise = torch.randn_like(image) * std
    return _clamp_tensor(image + noise)


class MultiCropAugmentor:
    """Generates multiple global and local crops for DINO-style training."""

    def __init__(
        self,
        global_transform: Callable[[torch.Tensor], torch.Tensor],
        local_transform: Callable[[torch.Tensor], torch.Tensor],
        num_global_crops: int,
        num_local_crops: int,
    ) -> None:
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

    def __call__(self, image: torch.Tensor) -> List[torch.Tensor]:
        crops: List[torch.Tensor] = []
        for _ in range(self.num_global_crops):
            crops.append(self.global_transform(image))
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))
        return crops


def build_view_transform(
    *,
    output_size: int,
    scale: Tuple[float, float],
    flip_prob: float,
    rotate_deg: float,
    blur_prob: float,
    noise_std: float,
    jitter: Dict[str, float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    size = int(output_size)
    scale = (float(scale[0]), float(scale[1]))
    flip_prob = float(flip_prob)
    rotate_deg = float(rotate_deg)
    blur_prob = float(max(0.0, min(1.0, blur_prob)))

    contrast = float(jitter.get("contrast", 0.0))
    brightness = float(jitter.get("brightness", 0.0))
    gamma = float(jitter.get("gamma", 0.0))

    def transform(image: torch.Tensor) -> torch.Tensor:
        out = image
        out = _random_resized_crop(out, size=size, scale=scale)
        out = _maybe_horizontal_flip(out, flip_prob)
        out = _maybe_rotate(out, rotate_deg)
        out = _intensity_jitter(out, contrast=contrast, brightness=brightness, gamma=gamma)
        out = _maybe_blur(out, blur_prob)
        out = _add_noise(out, std=noise_std)
        return out

    return transform


def build_mri_dino_augmentor(
    cfg: Dict[str, object],
    image_size: int,
) -> MultiCropAugmentor:
    num_global = int(cfg.get("global_crops", 2))
    num_local = int(cfg.get("local_crops", 6))
    global_scale = tuple(cfg.get("global_crop_scale", (0.35, 1.0)))
    local_scale = tuple(cfg.get("local_crop_scale", (0.10, 0.35)))
    blur_prob = float(cfg.get("gaussian_blur_prob", 0.5))
    local_blur_prob = float(cfg.get("local_blur_prob", 0.1))
    noise_std = float(cfg.get("noise_std", 0.02))
    flip_prob = float(cfg.get("random_flip_prob", 0.5))
    rotate_deg = float(cfg.get("random_rotate_deg", 0.0))
    jitter_cfg = dict(cfg.get("intensity_jitter", {}))

    local_output = max(64, image_size // 2)

    global_transform = build_view_transform(
        output_size=image_size,
        scale=(float(global_scale[0]), float(global_scale[1])),
        flip_prob=flip_prob,
        rotate_deg=rotate_deg,
        blur_prob=blur_prob,
        noise_std=noise_std,
        jitter=jitter_cfg,
    )
    local_transform = build_view_transform(
        output_size=local_output,
        scale=(float(local_scale[0]), float(local_scale[1])),
        flip_prob=flip_prob,
        rotate_deg=rotate_deg,
        blur_prob=local_blur_prob,
        noise_std=noise_std,
        jitter=jitter_cfg,
    )

    return MultiCropAugmentor(
        global_transform=global_transform,
        local_transform=local_transform,
        num_global_crops=num_global,
        num_local_crops=num_local,
    )
