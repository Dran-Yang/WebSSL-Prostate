from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import unquote

import numpy as np
import torch
from monai.transforms import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRangePercentilesd
from torch.utils.data import Dataset

from dinov2.data.monai_transforms.io import LoadTumorSliced

logger = logging.getLogger(__name__)


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def from_string(cls, value: str) -> "Split":
        value_upper = value.upper()
        try:
            return cls[value_upper]
        except KeyError as exc:
            raise ValueError(f"Unsupported split '{value}'.") from exc


@dataclass
class ProstateSSLConfig:
    root: Path
    split: Split
    sequences: Sequence[str]
    segmentation_key: Optional[str]
    spatial_size: Sequence[int]
    random_axes: bool
    random_slices: bool
    min_tumor_size: int
    split_file: Optional[Path]
    file_template: str
    search_suffixes: Sequence[str]
    allow_nested_dirs: bool
    recursive_search: bool


class SubjectToSequencePaths:
    """
    Resolve sequence identifiers to actual NIfTI file paths for a given subject directory.
    """

    def __init__(
        self,
        *,
        root: Path,
        sequences: Sequence[str],
        segmentation_key: Optional[str],
        file_template: str,
        search_suffixes: Sequence[str],
        allow_nested_dirs: bool,
        recursive_search: bool,
    ) -> None:
        self.root = root
        self.sequences = list(sequences)
        self.segmentation_key = segmentation_key
        self.file_template = file_template
        self.search_suffixes = tuple(search_suffixes)
        self.allow_nested_dirs = allow_nested_dirs
        self.recursive_search = recursive_search

    def _resolve(self, subject_dir: Path, sequence: str) -> Path:
        """Return the expected file path for a sequence."""
        relative = subject_dir.relative_to(self.root)
        subject_id = subject_dir.name
        dataset_id = relative.parts[0] if len(relative.parts) > 1 else ""
        template = self.file_template.format(
            subject=subject_id, sequence=sequence, dataset=dataset_id, relative=str(relative)
        )
        candidate = subject_dir / Path(template)
        resolved = self._resolve_candidate(candidate, sequence, subject_dir)
        if resolved is None:
            raise FileNotFoundError(
                f"Unable to resolve sequence '{sequence}' for subject '{subject_dir.name}'. "
                f"Checked template '{candidate}'."
            )
        return resolved

    def _resolve_candidate(self, candidate: Path, sequence: str, subject_dir: Path) -> Optional[Path]:
        resolved = self._check_file_variants(candidate)
        if resolved is not None:
            return resolved

        if candidate.is_dir() and self.allow_nested_dirs:
            resolved = self._pick_from_directory(candidate, sequence)
            if resolved is not None:
                return resolved

        if self.allow_nested_dirs:
            resolved = self._search_within_subject(subject_dir, sequence)
            if resolved is not None:
                return resolved

        return None

    def _check_file_variants(self, base: Path) -> Optional[Path]:
        parent = base.parent
        base_name = base.name
        candidates: list[Path] = []

        def _append(candidate: Path | str) -> None:
            path = candidate if isinstance(candidate, Path) else parent / candidate
            if path not in candidates:
                candidates.append(path)

        _append(base)

        if base.is_file():
            return base

        base_root = self._strip_known_suffixes(base_name)
        suffixes = list(self.search_suffixes) + [".nii", ".nii.gz"]

        for suffix in suffixes:
            suffix_norm = suffix if suffix.startswith(".") else f".{suffix}"
            _append(base_root + suffix_norm)

        # Consider original name without any suffix (for directories mimicking file names).
        _append(base_root)

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _strip_known_suffixes(self, name: str) -> str:
        suffixes = set(self.search_suffixes)
        suffixes.update({".nii", ".nii.gz"})
        for suffix in sorted(suffixes, key=len, reverse=True):
            suffix_norm = suffix if suffix.startswith(".") else f".{suffix}"
            if name.lower().endswith(suffix_norm.lower()):
                return name[: -len(suffix_norm)]
        return name

    def _pick_from_directory(self, directory: Path, sequence: str) -> Optional[Path]:
        nii_files = sorted(directory.glob("*.nii*"))
        if not nii_files:
            return None

        sequence_lower = sequence.lower()
        matching = [fp for fp in nii_files if sequence_lower in fp.name.lower()]
        if len(matching) == 1:
            return matching[0]

        if matching:
            logger.warning(
                "Multiple files matched sequence '%s' in directory '%s'. Using '%s'.",
                sequence,
                directory,
                matching[0],
            )
            return matching[0]

        if len(nii_files) == 1:
            return nii_files[0]

        logger.warning(
            "Could not uniquely match sequence '%s' within directory '%s'. Selected '%s'.",
            sequence,
            directory,
            nii_files[0],
        )
        return nii_files[0]

    def _search_within_subject(self, subject_dir: Path, sequence: str) -> Optional[Path]:
        patterns = [f"{sequence}.nii", f"{sequence}.nii.gz", f"*{sequence}*.nii", f"*{sequence}*.nii.gz"]
        for pattern in patterns:
            if self.recursive_search:
                matches = sorted(subject_dir.glob(f"**/{pattern}"))
            else:
                matches = sorted(subject_dir.glob(pattern))
            if matches:
                if len(matches) > 1:
                    logger.warning(
                        "Sequence '%s' for subject '%s' matched multiple files. Using '%s'.",
                        sequence,
                        subject_dir.name,
                        matches[0],
                    )
                return matches[0]
        return None

    def __call__(self, subject_dir: Path) -> dict[str, Path]:
        paths = {seq: self._resolve(subject_dir, seq) for seq in self.sequences}
        if self.segmentation_key is not None:
            paths[self.segmentation_key] = self._resolve(subject_dir, self.segmentation_key)
        return paths


class ProstateSSL(Dataset):
    """
    Slice-based self-supervised dataset for multi-modal prostate MRI studies.

    The dataset expects a directory structure compatible with the BIDS-inspired layout
    used in MM-DINOv2. Each subject directory listed in the split file should contain
    the modality files expressed through the provided template (e.g.,
    ``preop/sub-{subject}_ses-preop_space-sri_{sequence}.nii.gz``).
    """

    def __init__(
        self,
        cfg: ProstateSSLConfig,
        transform=None,
        target_transform=None,
    ) -> None:
        self.cfg = cfg
        self.root = cfg.root.expanduser()
        self.sequences = list(cfg.sequences)
        self.segmentation_key = cfg.segmentation_key
        self.spatial_size = tuple(int(v) for v in cfg.spatial_size)
        self.random_axes = cfg.random_axes
        self.random_slices = cfg.random_slices
        self.min_tumor_size = int(cfg.min_tumor_size)
        self.split = cfg.split

        self.transform = transform
        self.target_transform = target_transform

        self.subjects = self._load_subjects(cfg.split_file)
        self.pipeline = self._build_pipeline()

    def _load_subjects(self, split_file: Optional[Path]) -> List[str]:
        if split_file is None:
            subject_dirs = sorted(
                p for p in (self.root / self.split.value).iterdir() if p.is_dir()
            )
            return [str(p.relative_to(self.root)) for p in subject_dirs]

        split_path = split_file.expanduser()
        if not split_path.exists():
            raise FileNotFoundError(f"Split file {split_path} not found.")
        with split_path.open("r", encoding="utf-8") as f:
            subjects = [line.strip() for line in f if line.strip()]
        return subjects

    def _build_pipeline(self):
        if self.segmentation_key is None:
            raise ValueError(
                "segmentation_key must be provided to use LoadTumorSliced. "
                "Provide a prostate or lesion segmentation mask."
            )

        mapper = SubjectToSequencePaths(
            root=self.root,
            sequences=self.sequences,
            segmentation_key=self.segmentation_key,
            file_template=self.cfg.file_template,
            search_suffixes=self.cfg.search_suffixes,
            allow_nested_dirs=self.cfg.allow_nested_dirs,
            recursive_search=self.cfg.recursive_search,
        )

        axes = [0, 1, 2] if (self.random_axes and self.split is Split.TRAIN) else [2]
        select_random_slices = self.random_slices and self.split is Split.TRAIN

        return Compose(
            [
                mapper,
                LoadTumorSliced(
                    keys=[*self.sequences, self.segmentation_key],
                    tumor_key=self.segmentation_key,
                    spatial_size=self.spatial_size,
                    axes=axes,
                    min_tumor_size=self.min_tumor_size,
                    select_random_slices=select_random_slices,
                ),
                ScaleIntensityRangePercentilesd(
                    keys=self.sequences, b_min=0.0, b_max=1.0, lower=1, upper=99, clip=True
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int):
        subject_rel = self.subjects[index]
        subject_dir = (self.root / subject_rel).expanduser()
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory {subject_dir} could not be found.")

        sample = self.pipeline(subject_dir)
        image = torch.stack([sample[seq] for seq in self.sequences], dim=0)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        target = torch.tensor(-1, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


def parse_dataset_str(dataset_str: str) -> dict:
    tokens = dataset_str.split(":")
    name = tokens[0]
    if name != "ProstateSSL":
        raise ValueError(f"Unsupported dataset '{name}'")

    params: dict[str, object] = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        value = unquote(value)
        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
        elif value.replace(".", "", 1).isdigit():
            if "." in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        elif "," in value and key in {"sequences", "search_suffixes"}:
            params[key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            params[key] = value
    return params


def build_prostate_ssl_dataset(
    dataset_str: str,
    *,
    split_override: Optional[str] = None,
    transform=None,
    target_transform=None,
) -> ProstateSSL:
    parsed = parse_dataset_str(dataset_str)

    split_name = split_override or parsed.get("split")
    if split_name is None:
        raise ValueError("split parameter is required for ProstateSSL dataset.")

    split = Split.from_string(str(split_name))
    root = Path(parsed.get("root", ".")).expanduser()

    splits_root = Path(parsed.get("splits_root", root / "splits")).expanduser()
    split_file = splits_root / f"{split.value}.txt"

    sequences = parsed.get("sequences", ["t2w"])
    if isinstance(sequences, str):
        sequences = [seq.strip() for seq in sequences.split(",") if seq.strip()]

    segmentation_key = parsed.get("segmentation_key")
    spatial_size = parsed.get("spatial_size", 112)
    if isinstance(spatial_size, (int, float)):
        spatial_size = (int(spatial_size), int(spatial_size))

    search_suffixes = parsed.get("search_suffixes", [".nii", ".nii.gz"])
    if isinstance(search_suffixes, str):
        search_suffixes = [s.strip() for s in search_suffixes.split(",") if s.strip()]
    allow_nested_dirs = bool(parsed.get("allow_nested_dirs", True))
    recursive_search = bool(parsed.get("recursive_search", True))

    cfg = ProstateSSLConfig(
        root=root,
        split=split,
        sequences=sequences,
        segmentation_key=str(segmentation_key) if segmentation_key not in (None, "none", "") else None,
        spatial_size=spatial_size,
        random_axes=bool(parsed.get("random_axes", False)),
        random_slices=bool(parsed.get("random_slices", False)),
        min_tumor_size=int(parsed.get("min_tumor_size", 250)),
        split_file=split_file,
        file_template=str(parsed.get("file_template", "{sequence}.nii.gz")),
        search_suffixes=tuple(search_suffixes),
        allow_nested_dirs=allow_nested_dirs,
        recursive_search=recursive_search,
    )

    return ProstateSSL(cfg, transform=transform, target_transform=target_transform)
