"""
Dataset utilities for handling multi-modal prostate MRI slices.

The dataset expects the filesystem to be organised by patient/case folders,
with one subdirectory per requested modality. Each modality directory should
contain a single volumetric file (NIfTI ``.nii/.nii.gz``).

Example structure::

    <root>/
        case_0001/
            t2w/volume.nii.gz
            adc/volume.nii.gz
            dwi/volume.nii.gz

During indexing we inspect volumes to determine their depth (number of axial
planes). For each case we select a subset of slice indices according to the
configured policy and expose them as individual training samples while keeping
modalities aligned.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover - nibabel is optional
    nib = None


@dataclass(frozen=True)
class SliceRecord:
    """Metadata describing a single slice entry."""

    case_id: str
    slice_index: int
    modalities: Tuple[str, ...]
    paths: Tuple[Path, ...]
    depth: int


def _ensure_sequence(value: Sequence[float], length: int) -> Tuple[float, ...]:
    if len(value) != length:
        raise ValueError(f"Expected sequence of length {length}, got {len(value)}")
    return tuple(float(v) for v in value)


def _load_nifti_array(path: Path) -> np.ndarray:
    if nib is None:
        raise ImportError(
            f"nibabel is required to read NIfTI files but is not installed for {path}"
        )
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def _load_volume(path: Path) -> np.ndarray:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        return _load_nifti_array(path)
    raise ValueError(f"Unsupported volume extension for {path}")


def _peek_volume_shape(path: Path) -> Tuple[int, int, int]:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        if nib is None:
            raise ImportError(
                f"nibabel is required to inspect NIfTI shapes but is not installed for {path}"
            )
        image = nib.load(str(path))
        shape = image.shape
    else:
        raise ValueError(f"Unsupported volume extension for {path}")

    if len(shape) == 4 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 3:
        raise ValueError(f"Expected 3D volume at {path}, got shape {shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


class VolumeCache:
    """LRU cache for volumetric MRI data."""

    def __init__(self, max_items: int) -> None:
        self.max_items = max(1, int(max_items))
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]

        volume = _load_volume(path)
        self._store[key] = volume
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)
        return volume


class MRISliceDataset(Dataset):
    """Dataset returning registered multi-modal MRI slices."""

    def __init__(
        self,
        root: Path,
        modalities: Sequence[str],
        allowed_suffixes: Sequence[str],
        slice_policy: str,
        slice_stride: int,
        max_slices_per_volume: Optional[int],
        cache_size: int,
        clip_percentiles: Sequence[float],
        normalize_per_volume: bool,
        epsilon: float,
    ) -> None:
        self.root = Path(root)
        self.modalities = tuple(modalities)
        self.allowed_suffixes = tuple(s.lower() for s in allowed_suffixes)
        self.slice_policy = slice_policy
        self.slice_stride = max(1, int(slice_stride))
        self.max_slices_per_volume = (
            int(max_slices_per_volume) if max_slices_per_volume else None
        )
        self.normalize_per_volume = bool(normalize_per_volume)
        self.epsilon = float(epsilon)
        self.clip_percentiles = _ensure_sequence(clip_percentiles, 2)
        self.cache = VolumeCache(cache_size)
        self._volume_stats: Dict[str, Tuple[float, float]] = {}

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        if not self.modalities:
            raise ValueError("At least one modality is required")

        self.records: List[SliceRecord] = self._index_cases()
        if not self.records:
            raise RuntimeError(
                f"No slices found in {self.root}. "
                "Check directory structure and modality names."
            )

    def _index_cases(self) -> List[SliceRecord]:
        records: List[SliceRecord] = []
        for case_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            modal_paths: List[Path] = []
            # collect first matching volume per modality
            for modality in self.modalities:
                modality_dir = case_dir / modality
                if not modality_dir.exists():
                    modal_paths = []
                    break
                candidate = self._find_volume(modality_dir)
                if candidate is None:
                    modal_paths = []
                    break
                modal_paths.append(candidate)
            if len(modal_paths) != len(self.modalities):
                continue

            depths = [self._peek_depth(path) for path in modal_paths]
            depth = int(min(depths))
            slice_indices = self._select_slices(depth)
            for slice_idx in slice_indices:
                records.append(
                    SliceRecord(
                        case_id=case_dir.name,
                        slice_index=slice_idx,
                        modalities=self.modalities,
                        paths=tuple(modal_paths),
                        depth=depth,
                    )
                )
        return records

    def _find_volume(self, modality_dir: Path) -> Optional[Path]:
        for suffix in self.allowed_suffixes:
            candidates = sorted(
                p for p in modality_dir.glob(f"*{suffix}") if p.is_file()
            )
            if candidates:
                return candidates[0]
        return None

    def _peek_depth(self, path: Path) -> int:
        _, _, depth = _peek_volume_shape(path)
        return depth

    def _select_slices(self, depth: int) -> List[int]:
        policy = self.slice_policy.lower()
        stride = self.slice_stride

        if policy == "all":
            indices = list(range(0, depth, stride))
            if self.max_slices_per_volume:
                indices = indices[: self.max_slices_per_volume]
            return indices

        if policy == "middle":
            base = list(range(0, depth, stride))
            total = min(len(base), int(self.max_slices_per_volume) if self.max_slices_per_volume else len(base))
            if total == 0:
                return [depth // 2]
            center = depth // 2
            half = max(1, total // 2)
            start = max(0, center - half * stride)
            new_indices = list(range(start, min(depth, start + total * stride), stride))
            return new_indices[:total]

        if policy == "percentile":
            # 均匀从 [p_lo, p_hi] 区间取 N 个切片（避免极端首尾）
            p_lo, p_hi = 0.2, 0.8
            num = int(self.max_slices_per_volume or max(1, depth // stride))
            grid = np.linspace(p_lo, p_hi, num=num)
            cand = sorted(set(int(round(g * (depth - 1))) for g in grid))
            return cand

        if policy == "random":
            num = int(self.max_slices_per_volume or max(1, depth // stride))
            rng = np.random.default_rng()
            cand = sorted(set(int(rng.integers(0, depth)) for _ in range(num)))
            return cand

        if policy == "centered":
            # 以中位数为中心，取一个窗口（窗口宽约 = num * stride）
            num = int(self.max_slices_per_volume or max(1, depth // stride))
            center = depth // 2
            halfw = max(1, (num // 2) * stride)
            start = max(0, center - halfw)
            stop = min(depth - 1, center + halfw)
            base = list(range(start, stop + 1, stride))
            return base[:num]

        raise ValueError(f"Unknown slice selection policy: {self.slice_policy}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        slices = []
        for modality, path in zip(record.modalities, record.paths):
            volume = self.cache.get(path)
            if volume.ndim == 4:
                volume = volume[0]
            if volume.ndim != 3:
                raise ValueError(
                    f"Expected 3D array for {path}, got shape {volume.shape}"
                )
            stats = self._get_volume_stats(path, volume) if self.normalize_per_volume else None
            slice_data = volume[:, :, record.slice_index]
            slice_data = self._preprocess(slice_data, stats)
            slices.append(slice_data)

        stacked = np.stack(slices, axis=0)
        tensor = torch.from_numpy(stacked).float()
        return {
            "image": tensor,
            "case_id": record.case_id,
            "slice_index": record.slice_index,
        }

    def _get_volume_stats(
        self, path: Path, volume: np.ndarray
    ) -> Tuple[float, float]:
        key = str(path)
        if key not in self._volume_stats:
            self._volume_stats[key] = (
                float(np.min(volume)),
                float(np.max(volume)),
            )
        return self._volume_stats[key]

    def _preprocess(
        self,
        array: np.ndarray,
        stats: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        data = np.asarray(array, dtype=np.float32)
        if stats is not None:
            min_val, max_val = stats
            if not math.isclose(max_val, min_val):
                data = (data - min_val) / max(max_val - min_val, self.epsilon)
        low, high = np.percentile(data, self.clip_percentiles)
        if math.isclose(high, low):
            return np.zeros_like(data, dtype=np.float32)
        data = np.clip(data, low, high)
        data = (data - low) / max(high - low, self.epsilon)
        return data


def build_mri_dataset(cfg: Dict[str, object]) -> MRISliceDataset:
    """Factory helper aligning with YAML configuration."""

    root = Path(str(cfg["root"]))
    slice_cfg = dict(cfg.get("slice_selection") or {})
    cache_cfg = dict(cfg.get("cache") or {})
    preprocess_cfg = dict(cfg.get("preprocessing") or {})

    dataset = MRISliceDataset(
        root=root,
        modalities=cfg.get("modalities", []),
        allowed_suffixes=cfg.get("allowed_suffixes", [".nii", ".nii.gz"]),
        slice_policy=slice_cfg.get("policy", "middle"),
        slice_stride=slice_cfg.get("stride", 2),
        max_slices_per_volume=slice_cfg.get("max_slices_per_volume"),
        cache_size=cache_cfg.get("max_items", 16),
        clip_percentiles=preprocess_cfg.get("clip_percentiles", (0.5, 99.5)),
        normalize_per_volume=preprocess_cfg.get("normalize_per_volume", True),
        epsilon=preprocess_cfg.get("epsilon", 1e-6),
    )
    return dataset
