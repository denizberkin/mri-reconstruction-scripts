#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["MSE", "NMSE", "PSNR", "SSIM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot metric comparisons for acceleration factors 4 and 8."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/metrics_full.csv"),
        help="Path to metrics file (.csv or .json).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/figures_metrics"),
        help="Directory where plots are saved.",
    )
    parser.add_argument(
        "--accelerations",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Acceleration factors to compare.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["PSNR", "SSIM"],
        help="Metrics to include in plots.",
    )
    parser.add_argument(
        "--legend_fontsize",
        type=float,
        default=8,
        help="Legend font size for plots.",
    )
    return parser.parse_args()


def _coerce_row_types(row: Dict[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = dict(row)
    out["method"] = str(row["method"])
    out["acquisition"] = str(row["acquisition"])
    out["acceleration"] = int(row["acceleration"])
    out["center_fraction"] = float(row["center_fraction"])
    out["volumes"] = int(row["volumes"])

    for key in METRICS:
        out[key] = float(row[key])

    return out


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [_coerce_row_types(row) for row in reader]

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "rows" in data:
            data = data["rows"]
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of metric rows or an object with 'rows'.")
        return [_coerce_row_types(row) for row in data]

    raise ValueError("Unsupported input format. Use .csv or .json")


def unique_in_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def get_row(
    rows: List[Dict[str, object]],
    method: str,
    acceleration: int,
    acquisition: str,
) -> Dict[str, object] | None:
    for row in rows:
        if (
            row["method"] == method
            and row["acceleration"] == acceleration
            and row["acquisition"] == acquisition
        ):
            return row
    return None


def format_metric_value(metric: str, value: float) -> str:
    if np.isnan(value):
        return "nan"
    if metric in {"MSE", "NMSE"}:
        return f"{value:.3g}"
    return f"{value:.3f}"


def plot_overall(
    rows: List[Dict[str, object]],
    accels: List[int],
    metrics: List[str],
    output_dir: Path,
    legend_fontsize: float,
) -> None:
    overall_rows = [row for row in rows if row["acquisition"] == "ALL"]
    methods = unique_in_order([str(row["method"]) for row in overall_rows])

    if not methods:
        print("No acquisition='ALL' rows found; skipping overall plot.")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(methods))
    width = 0.35

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        for accel_idx, accel in enumerate(accels):
            values = []
            for method in methods:
                row = get_row(overall_rows, method, accel, "ALL")
                values.append(np.nan if row is None else float(row[metric]))

            offset = (accel_idx - (len(accels) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width=width, label=f"R={accel}")

            if metric in {"MSE", "NMSE"}:
                label_values = [format_metric_value(metric, value) for value in values]
                ax.bar_label(bars, labels=label_values, padding=2, fontsize=8)
            else:
                ax.bar_label(
                    bars,
                    labels=[format_metric_value(metric, value) for value in values],
                    fmt="%.3f",
                    padding=2,
                    fontsize=8,
                )

        ax.set_title(f"ALL: {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.grid(axis="y", alpha=0.25)
        if metric in {"MSE", "NMSE"}:
            ax.set_yscale("log")

    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="best", fontsize=legend_fontsize)
    fig.tight_layout()

    out_file = output_dir / "accel_compare_overall.png"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_by_acquisition(
    rows: List[Dict[str, object]],
    accels: List[int],
    metrics: List[str],
    output_dir: Path,
    legend_fontsize: float,
) -> None:
    acquisitions = unique_in_order([str(row["acquisition"]) for row in rows if row["acquisition"] != "ALL"])
    methods = unique_in_order([str(row["method"]) for row in rows])

    if not acquisitions:
        print("No per-acquisition rows found; skipping acquisition plot.")
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(acquisitions))
    method_markers = {"zero_filled": "o", "unet": "s"}

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        for method in methods:
            for accel in accels:
                y = []
                for acquisition in acquisitions:
                    row = get_row(rows, method, accel, acquisition)
                    y.append(np.nan if row is None else float(row[metric]))

                marker = method_markers.get(method, "o")
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linewidth=1.7,
                    label=f"{method} (R={accel})",
                )

        ax.set_title(f"Per-acquisition {metric}")
        ax.grid(alpha=0.25)
        if metric in {"MSE", "NMSE"}:
            ax.set_yscale("log")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(acquisitions)
    axes[0].legend(ncols=2, fontsize=legend_fontsize)
    axes[0].set_ylabel("Metric value")
    fig.tight_layout()

    out_file = output_dir / "accel_compare_by_acquisition.png"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_file}")


def main() -> None:
    args = parse_args()

    metrics = [m.upper() for m in args.metrics]
    invalid = [m for m in metrics if m not in METRICS]
    if invalid:
        raise ValueError(f"Unsupported metric(s): {invalid}. Allowed: {METRICS}")

    rows = load_rows(args.input)
    rows = [row for row in rows if int(row["acceleration"]) in args.accelerations]

    if not rows:
        raise ValueError("No rows matched requested accelerations.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_overall(rows, args.accelerations, metrics, args.output_dir, args.legend_fontsize)
    plot_by_acquisition(rows, args.accelerations, metrics, args.output_dir, args.legend_fontsize)


if __name__ == "__main__":
    main()
