from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List all patient folders under the dataset root and write train.txt."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory that contains one sub-folder per patient (e.g. X:/AIS_processed_20251015update).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pre-ssl-prostate/splits/train.txt"),
        help="Where to write the train split file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing train.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{root}' does not exist.")

    patient_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not patient_dirs:
        raise ValueError(f"No patient directories found under '{root}'.")

    output_path = args.output
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for patient_dir in patient_dirs:
            rel = patient_dir.relative_to(root).as_posix()
            handle.write(f"{rel}\n")

    print(f"Wrote {len(patient_dirs)} patients to {output_path}")


if __name__ == "__main__":
    main()
