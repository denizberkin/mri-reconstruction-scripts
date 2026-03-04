#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def normalize(img: np.ndarray) -> np.ndarray:
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    if p99 <= p1:
        return np.zeros_like(img)
    return np.clip((img - p1) / (p99 - p1), 0.0, 1.0)


def get_slice(volume: np.ndarray, slice_idx: int) -> np.ndarray:
    if slice_idx < 0:
        idx = volume.shape[0] // 2
    else:
        idx = min(slice_idx, volume.shape[0] - 1)
    return volume[idx]


def to_2d_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            return np.linalg.norm(arr, axis=0)
        if arr.shape[-1] <= 8 and arr.shape[0] > 8 and arr.shape[1] > 8:
            return np.linalg.norm(arr, axis=-1)
        return arr[0]

    while arr.ndim > 2:
        arr = arr[0]

    return arr


def expand_file_list(items) -> List[str]:
    if not items:
        return []
    files: List[str] = []
    for item in items:
        for token in item.split(","):
            token = token.strip()
            if token:
                files.append(token)
    return files


def center_crop(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape
    start_h = max((h - out_h) // 2, 0)
    start_w = max((w - out_w) // 2, 0)
    return img[start_h : start_h + out_h, start_w : start_w + out_w]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and save qualitative outputs for zero-filled vs U-Net."
    )
    parser.add_argument(
        "--target_path",
        type=Path,
        default=Path("data/multicoil_val"),
        help="Path to validation .h5 files with target reconstruction key.",
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default="reconstruction_rss",
        help="Target key in ground-truth files.",
    )
    parser.add_argument(
        "--zero_filled_path",
        type=Path,
        default=Path("results/zero_filled_subset"),
        help="Path to zero-filled reconstruction .h5 files.",
    )
    parser.add_argument(
        "--unet_path",
        type=Path,
        default=Path("results/unet_subset/reconstructions"),
        help="Path to U-Net reconstruction .h5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/figures_zf_unet"),
    )
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Specific .h5 filenames. Repeat or comma-separate.",
    )
    parser.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help="Optional text file with one .h5 file name per line.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=10,
        help="Used if --file and --file_list are not provided.",
    )
    parser.add_argument(
        "--slice_index",
        type=int,
        default=-1,
        help="Slice index; -1 uses middle slice.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected = expand_file_list(args.file)
    if args.file_list is not None and args.file_list.exists():
        selected.extend(
            [line.strip() for line in args.file_list.read_text().splitlines() if line.strip()]
        )

    if not selected:
        selected = [p.name for p in sorted(args.target_path.glob("*.h5"))[: args.max_files]]

    combined_dir = args.output_dir / "comparison"
    target_dir = args.output_dir / "target"
    zf_dir = args.output_dir / "zero_filled"
    unet_dir = args.output_dir / "unet"
    error_zf_dir = args.output_dir / "error_zero_filled"
    error_unet_dir = args.output_dir / "error_unet"

    for directory in [
        combined_dir,
        target_dir,
        zf_dir,
        unet_dir,
        error_zf_dir,
        error_unet_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm(selected, desc="Saving qualitative outputs"):
        target_file = args.target_path / file_name
        zf_file = args.zero_filled_path / file_name
        unet_file = args.unet_path / file_name

        if not target_file.exists() or not zf_file.exists() or not unet_file.exists():
            print(f"Skip missing file(s): {file_name}")
            continue

        with h5py.File(target_file, "r") as hf:
            target_vol = hf[args.target_key][()]
        with h5py.File(zf_file, "r") as hf:
            zf_vol = hf["reconstruction"][()]
        with h5py.File(unet_file, "r") as hf:
            unet_vol = hf["reconstruction"][()]

        target = to_2d_image(get_slice(target_vol, args.slice_index))
        zf = to_2d_image(get_slice(zf_vol, args.slice_index))
        unet = to_2d_image(get_slice(unet_vol, args.slice_index))

        target_n = normalize(target)
        zf_n = normalize(zf)
        unet_n = normalize(unet)

        common_h = min(target_n.shape[0], zf_n.shape[0], unet_n.shape[0])
        common_w = min(target_n.shape[1], zf_n.shape[1], unet_n.shape[1])

        if target_n.shape != zf_n.shape or target_n.shape != unet_n.shape:
            print(
                f"Shape mismatch for {file_name}: "
                f"target={target_n.shape}, zero_filled={zf_n.shape}, unet={unet_n.shape}. "
                f"Using center crop to ({common_h}, {common_w}) for error maps."
            )

        target_err = center_crop(target_n, common_h, common_w)
        zf_err = center_crop(zf_n, common_h, common_w)
        unet_err = center_crop(unet_n, common_h, common_w)

        err_zf = np.abs(zf_err - target_err)
        err_unet = np.abs(unet_err - target_err)

        stem = Path(file_name).stem

        plt.imsave(target_dir / f"{stem}.png", target_n, cmap="gray")
        plt.imsave(zf_dir / f"{stem}.png", zf_n, cmap="gray")
        plt.imsave(unet_dir / f"{stem}.png", unet_n, cmap="gray")
        plt.imsave(error_zf_dir / f"{stem}.png", err_zf, cmap="magma")
        plt.imsave(error_unet_dir / f"{stem}.png", err_unet, cmap="magma")

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        panels = [
            ("Target", target_n, "gray"),
            ("Zero-filled", zf_n, "gray"),
            ("U-Net", unet_n, "gray"),
            ("|Zero-filled - Target|", err_zf, "magma"),
            ("|U-Net - Target|", err_unet, "magma"),
        ]

        for ax, (title, img, cmap) in zip(axes, panels):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        fig.tight_layout()
        output_path = combined_dir / f"{stem}_zf_unet_qualitative.png"
        fig.savefig(output_path, dpi=args.dpi)
        plt.close(fig)

    print(f"Saved qualitative outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
