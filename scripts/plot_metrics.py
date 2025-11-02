from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training metrics stored in JSONL format."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("out/pre_ssl/prostate_ssl_pretrain/training_metrics.jsonl"),
        help="Path to the training_metrics.jsonl file produced during pretraining.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Metric names to plot. Defaults to all numeric columns except lr and wd.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the plot (e.g. metrics.png). If omitted, shows interactively.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=1,
        help="Size of the moving average window to smooth curves (>=1).",
    )
    parser.add_argument(
        "--include-lr",
        action="store_true",
        help="Include learning-rate and weight-decay curves on a secondary axis.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file '{path}' not found.")

    records: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to decode line: {line}") from exc
            records.append(record)

    if not records:
        raise ValueError(f"No records found in '{path}'.")
    return records


def moving_average(values: Iterable[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)

    window_values: list[float] = []
    smoothed: list[float] = []
    cum_sum = 0.0
    for idx, value in enumerate(values):
        window_values.append(value)
        cum_sum += value
        if len(window_values) > window:
            cum_sum -= window_values.pop(0)
        divisor = min(window, idx + 1)
        smoothed.append(cum_sum / divisor)
    return smoothed


def main() -> None:
    args = parse_args()

    records = load_metrics(args.metrics)
    iterations = [record["iteration"] for record in records if "iteration" in record]
    if not iterations:
        raise ValueError("No 'iteration' field found in metrics records.")

    sample_record = records[0]
    numeric_keys = [
        key
        for key, value in sample_record.items()
        if isinstance(value, (int, float)) and key != "iteration"
    ]

    lr_keys = [key for key in ("lr", "wd") if key in numeric_keys]
    default_keys = [key for key in numeric_keys if key not in lr_keys]

    selected_keys = args.columns or default_keys
    missing = [key for key in selected_keys if key not in numeric_keys]
    if missing:
        raise ValueError(f"Requested columns not found in metrics: {missing}")

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 6))

    for key in selected_keys:
        values = [record.get(key, float("nan")) for record in records]
        smoothed = moving_average(values, max(1, args.rolling_window))
        ax.plot(iterations, smoothed, label=key)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training metrics from {args.metrics.name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    if args.include_lr and lr_keys:
        ax2 = ax.twinx()
        for key in lr_keys:
            values = [record.get(key, float("nan")) for record in records]
            smoothed = moving_average(values, max(1, args.rolling_window))
            ax2.plot(iterations, smoothed, label=key, linestyle=":", linewidth=1.2)
        ax2.set_ylabel("LR / WD")
        lines_labels = [ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
        lines = sum((handles for handles, _ in lines_labels), [])
        labels = sum((labels for _, labels in lines_labels), [])
        ax2.legend(lines, labels, loc="upper right")

    fig.tight_layout()

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
