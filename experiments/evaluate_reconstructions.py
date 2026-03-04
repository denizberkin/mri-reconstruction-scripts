#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def load_runtime_modules() -> Dict[str, object]:
    from fastmri.data import transforms
    from fastmri.evaluate import METRIC_FUNCS, Metrics

    return {
        "transforms": transforms,
        "METRIC_FUNCS": METRIC_FUNCS,
        "Metrics": Metrics,
    }


def add_fastmri_repo_to_path(fastmri_repo: Path) -> None:
    repo = fastmri_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def parse_named_path(items) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --method '{item}'. Expected name=path format.")
        name, path_str = item.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip())
        if not name:
            raise ValueError(f"Invalid --method '{item}': empty name")
        output[name] = path
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more reconstruction folders against fastMRI targets."
    )
    parser.add_argument("--fastmri_repo", type=Path, default=Path("fastMRI"))
    parser.add_argument(
        "--target_path",
        type=Path,
        default=Path("data/multicoil_val"),
        help="Directory with target .h5 files.",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        default="multicoil",
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method recon dir in name=path form. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--acquisition",
        default="",
        help="Optional acquisition filter: AXT1, AXT2, AXFLAIR, etc.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/metrics_summary.csv"),
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/metrics_summary.json"),
    )
    return parser.parse_args()


def load_arrays(
    target_file: Path,
    pred_file: Path,
    target_key: str,
    transforms_module,
):
    with h5py.File(target_file, "r") as target_hf, h5py.File(pred_file, "r") as pred_hf:
        target = target_hf[target_key][()]
        pred = pred_hf["reconstruction"][()]

    min_slices = min(target.shape[0], pred.shape[0])
    target = target[:min_slices]
    pred = pred[:min_slices]

    target = transforms_module.center_crop(target, (target.shape[-1], target.shape[-1]))
    pred = transforms_module.center_crop(pred, (target.shape[-1], target.shape[-1]))

    return target, pred


def evaluate_method(
    target_dir: Path,
    pred_dir: Path,
    target_key: str,
    acquisition_filter: str,
    method_name: str,
    runtime_modules: Dict[str, object],
) -> Tuple[dict, int]:
    transforms_module = runtime_modules["transforms"]
    metrics_class = runtime_modules["Metrics"]
    metric_funcs = runtime_modules["METRIC_FUNCS"]

    metrics = metrics_class(metric_funcs)
    used = 0

    target_files = sorted(target_dir.glob("*.h5"))
    for target_file in tqdm(target_files, desc=f"Eval {method_name}", leave=False):
        pred_file = pred_dir / target_file.name
        if not pred_file.exists():
            continue

        with h5py.File(target_file, "r") as target_hf:
            acquisition = str(target_hf.attrs.get("acquisition", ""))
        if acquisition_filter and acquisition_filter != acquisition:
            continue

        target, pred = load_arrays(target_file, pred_file, target_key, transforms_module)
        metrics.push(target, pred)
        used += 1

    means = metrics.means() if used > 0 else {k: np.nan for k in ["MSE", "NMSE", "PSNR", "SSIM"]}
    return means, used


def main() -> None:
    args = parse_args()
    add_fastmri_repo_to_path(args.fastmri_repo)
    runtime_modules = load_runtime_modules()

    methods = parse_named_path(args.method)
    target_key = "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"

    summary = []
    print("Method,Volumes,MSE,NMSE,PSNR,SSIM")
    for method_name, pred_dir in tqdm(methods.items(), total=len(methods), desc="Methods"):
        means, used = evaluate_method(
            args.target_path,
            pred_dir,
            target_key,
            args.acquisition.strip(),
            method_name,
            runtime_modules,
        )
        row = {
            "method": method_name,
            "volumes": used,
            "MSE": float(means["MSE"]),
            "NMSE": float(means["NMSE"]),
            "PSNR": float(means["PSNR"]),
            "SSIM": float(means["SSIM"]),
        }
        summary.append(row)
        print(
            f"{method_name},{used},{row['MSE']:.6g},{row['NMSE']:.6g},{row['PSNR']:.6g},{row['SSIM']:.6g}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["method", "volumes", "MSE", "NMSE", "PSNR", "SSIM"])
        writer.writeheader()
        writer.writerows(summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
