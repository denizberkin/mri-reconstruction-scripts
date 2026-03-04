#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_named_path(items) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --method '{item}'. Expected name=path format.")
        name, path_str = item.split("=", 1)
        output[name.strip()] = Path(path_str.strip())
    return output


def normalize(img: np.ndarray) -> np.ndarray:
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    if p99 <= p1:
        return np.zeros_like(img)
    out = np.clip((img - p1) / (p99 - p1), 0.0, 1.0)
    return out


def get_slice(volume: np.ndarray, slice_idx: int) -> np.ndarray:
    if slice_idx < 0:
        idx = volume.shape[0] // 2
    else:
        idx = min(slice_idx, volume.shape[0] - 1)
    return volume[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side qualitative reconstruction figures."
    )
    parser.add_argument(
        "--target_path",
        type=Path,
        default=Path("data/multicoil_val"),
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default="reconstruction_rss",
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method recon dir in name=path form. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/figures"),
    )
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Specific .h5 filenames. Repeat or comma-separate.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=6,
        help="Used if --file not provided.",
    )
    parser.add_argument(
        "--slice_index",
        type=int,
        default=-1,
        help="Slice index; -1 uses middle slice.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def expand_file_list(items) -> List[str]:
    if not items:
        return []
    files = []
    for item in items:
        for token in item.split(","):
            token = token.strip()
            if token:
                files.append(token)
    return files


def main() -> None:
    args = parse_args()
    methods = parse_named_path(args.method)

    selected = expand_file_list(args.file)
    if not selected:
        selected = [p.name for p in sorted(args.target_path.glob("*.h5"))[: args.max_files]]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm(selected, desc="Plot comparisons"):
        target_file = args.target_path / file_name
        if not target_file.exists():
            print(f"Skip missing target: {file_name}")
            continue

        with h5py.File(target_file, "r") as hf:
            target_vol = hf[args.target_key][()]

        target_img = normalize(get_slice(target_vol, args.slice_index))
        panels = [("Target", target_img)]

        for method_name, method_dir in methods.items():
            pred_file = method_dir / file_name
            if not pred_file.exists():
                continue
            with h5py.File(pred_file, "r") as hf:
                pred_vol = hf["reconstruction"][()]
            pred_img = normalize(get_slice(pred_vol, args.slice_index))
            panels.append((method_name, pred_img))

        cols = len(panels)
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        if cols == 1:
            axes = [axes]

        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        fig.tight_layout()
        out_file = args.output_dir / f"{Path(file_name).stem}_comparison.png"
        fig.savefig(out_file, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
