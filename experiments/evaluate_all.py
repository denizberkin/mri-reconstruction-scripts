from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_ACQUISITIONS = ["AXT1", "AXT1POST", "AXT2", "AXFLAIR"]


def load_runtime_modules(methods: List[str]) -> Dict[str, object]:
    import fastmri
    from fastmri.data import transforms
    from fastmri.data.mri_data import et_query
    from fastmri.data.subsample import create_mask_for_mask_type
    from fastmri.evaluate import METRIC_FUNCS, Metrics

    modules: Dict[str, object] = {
        "fastmri": fastmri,
        "transforms": transforms,
        "et_query": et_query,
        "create_mask_for_mask_type": create_mask_for_mask_type,
        "METRIC_FUNCS": METRIC_FUNCS,
        "Metrics": Metrics,
    }

    if "unet" in methods:
        from fastmri.data import SliceDataset
        from fastmri.models import Unet

        modules["SliceDataset"] = SliceDataset
        modules["Unet"] = Unet

    return modules


def add_fastmri_repo_to_path(fastmri_repo: Path) -> None:
    repo = fastmri_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def is_readable_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate zero-filled and/or pretrained U-Net on full fastMRI validation data "
            "without saving reconstruction volumes to disk."
        )
    )
    parser.add_argument("--fastmri_repo", type=Path, default=Path("fastMRI"))
    parser.add_argument("--data_path", type=Path, default=Path("data/multicoil_val"))
    parser.add_argument("--challenge", choices=["singlecoil", "multicoil"], default="multicoil")
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="Method(s) to evaluate. Repeat from: zero_filled, unet. Defaults to both.",
    )
    parser.add_argument(
        "--unet_challenge",
        default="unet_brain_mc",
        choices=("unet_knee_sc", "unet_knee_mc", "unet_brain_mc"),
    )
    parser.add_argument(
        "--state_dict_file",
        type=Path,
        default=Path("brain_leaderboard_state_dict.pt"),
        help="Path to pretrained U-Net state dict.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for U-Net inference: auto, cuda, or cpu.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--mask_type",
        choices=["random", "equispaced", "equispaced_fraction", "magic", "magic_fraction"],
        default="random",
        help="Retrospective undersampling mask type.",
    )
    parser.add_argument(
        "--accelerations",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Acceleration factors to evaluate.",
    )
    parser.add_argument(
        "--center_fractions",
        type=float,
        nargs="+",
        default=[0.08, 0.04],
        help="Center fractions aligned with --accelerations.",
    )
    parser.add_argument(
        "--acquisition",
        action="append",
        default=None,
        help="Acquisition type(s) to include; repeat or comma-separate. Defaults: AXT1, AXT1POST, AXT2, AXFLAIR.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/metrics_full_nosave.csv"),
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/metrics_full_nosave.json"),
    )
    return parser.parse_args()


def resolve_methods(method_args: Optional[List[str]]) -> List[str]:
    allowed = {"zero_filled", "unet"}
    if not method_args:
        return ["zero_filled", "unet"]

    methods: List[str] = []
    for item in method_args:
        for part in item.split(","):
            m = part.strip()
            if not m:
                continue
            if m not in allowed:
                raise ValueError(f"Invalid --method '{m}'. Allowed: zero_filled, unet")
            if m not in methods:
                methods.append(m)

    if not methods:
        return ["zero_filled", "unet"]
    return methods


def resolve_acquisitions(acquisition_args: Optional[List[str]]) -> List[str]:
    if not acquisition_args:
        return list(DEFAULT_ACQUISITIONS)

    acquisitions: List[str] = []
    for item in acquisition_args:
        for part in item.split(","):
            value = part.strip()
            if value and value not in acquisitions:
                acquisitions.append(value)

    return acquisitions or list(DEFAULT_ACQUISITIONS)


def list_valid_target_files(
    data_path: Path,
    target_key: str,
    acquisitions: List[str],
) -> Tuple[List[Tuple[Path, str]], List[Path], int]:
    files = sorted(data_path.glob("*.h5"))
    good: List[Tuple[Path, str]] = []
    skipped: List[Path] = []
    skipped_by_acquisition = 0

    for path in files:
        try:
            with h5py.File(path, "r") as hf:
                if target_key not in hf:
                    skipped.append(path)
                    continue

                acquisition = str(hf.attrs.get("acquisition", ""))
                if acquisition in acquisitions:
                    good.append((path, acquisition))
                else:
                    skipped_by_acquisition += 1
        except OSError:
            skipped.append(path)

    return good, skipped, skipped_by_acquisition


def align_target_and_pred(target: np.ndarray, pred: np.ndarray, transforms_module) -> Tuple[np.ndarray, np.ndarray]:
    min_slices = min(target.shape[0], pred.shape[0])
    target = target[:min_slices]
    pred = pred[:min_slices]

    crop_hw = (target.shape[-1], target.shape[-1])
    target = transforms_module.center_crop(target, crop_hw)
    pred = transforms_module.center_crop(pred, crop_hw)

    return target, pred


def to_scalar(value) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def save_rows(rows: List[dict], output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "method",
                "acceleration",
                "center_fraction",
                "acquisition",
                "volumes",
                "MSE",
                "NMSE",
                "PSNR",
                "SSIM",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def reconstruct_zero_filled_volume(
    file_path: Path,
    challenge: str,
    fastmri_module,
    transforms_module,
    et_query_func,
    mask_func,
):
    with h5py.File(file_path, "r") as hf:
        et_root = etree.fromstring(hf["ismrmrd_header"][()])
        kspace = transforms_module.to_tensor(hf["kspace"][()])
        target = hf["reconstruction_rss" if challenge == "multicoil" else "reconstruction_esc"][()]

    seed = tuple(map(ord, file_path.name))
    masked_kspace = transforms_module.apply_mask(kspace, mask_func, seed=seed)[0]

    enc = ["encoding", "encodedSpace", "matrixSize"]
    crop_size = (
        int(et_query_func(et_root, enc + ["x"])),
        int(et_query_func(et_root, enc + ["y"])),
    )

    image = fastmri_module.ifft2c(masked_kspace)
    if image.shape[-2] < crop_size[1]:
        crop_size = (image.shape[-2], image.shape[-2])

    image = transforms_module.complex_center_crop(image, crop_size)
    image = fastmri_module.complex_abs(image)

    if challenge == "multicoil":
        image = fastmri_module.rss(image, dim=1)

    pred = image.cpu().numpy()
    return target, pred


def load_unet_model(args: argparse.Namespace, device, runtime_modules: Dict[str, object]):
    unet_class = runtime_modules["Unet"]
    model = unet_class(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    model.load_state_dict(torch.load(args.state_dict_file, map_location="cpu"))
    model = model.eval().to(device)
    return model


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    if device_arg not in {"cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    return torch.device(device_arg)


def run_unet_batch(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch

    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()
    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()

    return output.squeeze(0).numpy(), int(slice_num[0]), fname[0]


def reconstruct_unet_volume(
    file_path: Path,
    challenge: str,
    model,
    device,
    num_workers: int,
    mask_func,
    runtime_modules: Dict[str, object],
):
    transforms_module = runtime_modules["transforms"]
    slice_dataset_class = runtime_modules["SliceDataset"]

    which_challenge = "multicoil" if challenge == "multicoil" else "singlecoil"

    with tempfile.TemporaryDirectory(prefix="unet_eval_", dir=".") as tmp_dir:
        tmp_path = Path(tmp_dir)
        linked_file = tmp_path / file_path.name
        os.symlink(file_path.resolve(), linked_file)

        dataset = slice_dataset_class(
            root=tmp_path,
            transform=transforms_module.UnetDataTransform(
                which_challenge=which_challenge,
                mask_func=mask_func,
                use_seed=True,
            ),
            challenge=which_challenge,
            use_dataset_cache=False,
        )
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)

        slice_outputs: Dict[int, np.ndarray] = {}
        for batch in dataloader:
            with torch.no_grad():
                output, slice_num, _ = run_unet_batch(batch, model, device)
            slice_outputs[slice_num] = output

    pred = np.stack([slice_outputs[idx] for idx in sorted(slice_outputs)])

    with h5py.File(file_path, "r") as hf:
        target = hf["reconstruction_rss" if challenge == "multicoil" else "reconstruction_esc"][()]

    return target, pred


def evaluate_zero_filled(
    file_records: List[Tuple[Path, str]],
    challenge: str,
    acquisitions: List[str],
    acceleration: int,
    center_fraction: float,
    mask_type: str,
    runtime_modules: Dict[str, object],
):
    fastmri_module = runtime_modules["fastmri"]
    transforms_module = runtime_modules["transforms"]
    et_query_func = runtime_modules["et_query"]
    create_mask = runtime_modules["create_mask_for_mask_type"]
    metric_funcs = runtime_modules["METRIC_FUNCS"]
    metrics_class = runtime_modules["Metrics"]

    mask_func = create_mask(mask_type, [center_fraction], [acceleration])
    metrics_by_group = {acq: metrics_class(metric_funcs) for acq in acquisitions}
    metrics_by_group["ALL"] = metrics_class(metric_funcs)
    counts_by_group = {acq: 0 for acq in acquisitions}
    counts_by_group["ALL"] = 0

    for file_path, acquisition in tqdm(file_records, desc=f"Eval zero_filled R{acceleration}"):
        target, pred = reconstruct_zero_filled_volume(
            file_path,
            challenge,
            fastmri_module,
            transforms_module,
            et_query_func,
            mask_func,
        )
        target, pred = align_target_and_pred(target, pred, transforms_module)
        metrics_by_group[acquisition].push(target, pred)
        metrics_by_group["ALL"].push(target, pred)
        counts_by_group[acquisition] += 1
        counts_by_group["ALL"] += 1

    rows = []
    for acquisition in acquisitions + ["ALL"]:
        used = counts_by_group[acquisition]
        means = (
            metrics_by_group[acquisition].means()
            if used > 0
            else {k: np.nan for k in ["MSE", "NMSE", "PSNR", "SSIM"]}
        )
        rows.append(
            {
                "method": "zero_filled",
                "acquisition": acquisition,
                "acceleration": acceleration,
                "center_fraction": center_fraction,
                "volumes": used,
                "MSE": to_scalar(means["MSE"]),
                "NMSE": to_scalar(means["NMSE"]),
                "PSNR": to_scalar(means["PSNR"]),
                "SSIM": to_scalar(means["SSIM"]),
            }
        )

    return rows


def evaluate_unet(
    file_records: List[Tuple[Path, str]],
    args: argparse.Namespace,
    challenge: str,
    acquisitions: List[str],
    acceleration: int,
    center_fraction: float,
    mask_type: str,
    runtime_modules: Dict[str, object],
):
    create_mask = runtime_modules["create_mask_for_mask_type"]
    transforms_module = runtime_modules["transforms"]
    metric_funcs = runtime_modules["METRIC_FUNCS"]
    metrics_class = runtime_modules["Metrics"]

    mask_func = create_mask(mask_type, [center_fraction], [acceleration])
    device = resolve_device(args.device)
    print(f"U-Net device: {device}")
    model = load_unet_model(args, device, runtime_modules)

    metrics_by_group = {acq: metrics_class(metric_funcs) for acq in acquisitions}
    metrics_by_group["ALL"] = metrics_class(metric_funcs)
    counts_by_group = {acq: 0 for acq in acquisitions}
    counts_by_group["ALL"] = 0

    for file_path, acquisition in tqdm(file_records, desc=f"Eval unet R{acceleration}"):
        target, pred = reconstruct_unet_volume(
            file_path=file_path,
            challenge=challenge,
            model=model,
            device=device,
            num_workers=args.num_workers,
            mask_func=mask_func,
            runtime_modules=runtime_modules,
        )
        target, pred = align_target_and_pred(target, pred, transforms_module)
        metrics_by_group[acquisition].push(target, pred)
        metrics_by_group["ALL"].push(target, pred)
        counts_by_group[acquisition] += 1
        counts_by_group["ALL"] += 1

    rows = []
    for acquisition in acquisitions + ["ALL"]:
        used = counts_by_group[acquisition]
        means = (
            metrics_by_group[acquisition].means()
            if used > 0
            else {k: np.nan for k in ["MSE", "NMSE", "PSNR", "SSIM"]}
        )
        rows.append(
            {
                "method": "unet",
                "acquisition": acquisition,
                "acceleration": acceleration,
                "center_fraction": center_fraction,
                "volumes": used,
                "MSE": to_scalar(means["MSE"]),
                "NMSE": to_scalar(means["NMSE"]),
                "PSNR": to_scalar(means["PSNR"]),
                "SSIM": to_scalar(means["SSIM"]),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    add_fastmri_repo_to_path(args.fastmri_repo)

    methods = resolve_methods(args.method)
    runtime_modules = load_runtime_modules(methods)
    acquisitions = resolve_acquisitions(args.acquisition)
    if len(args.accelerations) != len(args.center_fractions):
        raise ValueError("--accelerations and --center_fractions must have the same length.")

    target_key = "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"

    file_records, skipped, skipped_by_acquisition = list_valid_target_files(
        args.data_path,
        target_key,
        acquisitions,
    )
    if not file_records:
        raise RuntimeError(f"No valid .h5 files with target key '{target_key}' in {args.data_path}")

    if skipped:
        print(f"Skipping {len(skipped)} unreadable/no-target file(s).")
    if skipped_by_acquisition:
        print(f"Skipping {skipped_by_acquisition} file(s) outside selected acquisitions: {', '.join(acquisitions)}")

    rows = []
    print("Method,Acceleration,CenterFraction,Acquisition,Volumes,MSE,NMSE,PSNR,SSIM")

    for acceleration, center_fraction in zip(args.accelerations, args.center_fractions):
        for method in methods:
            if method == "zero_filled":
                method_rows = evaluate_zero_filled(
                    file_records=file_records,
                    challenge=args.challenge,
                    acquisitions=acquisitions,
                    acceleration=acceleration,
                    center_fraction=center_fraction,
                    mask_type=args.mask_type,
                    runtime_modules=runtime_modules,
                )
            elif method == "unet":
                method_rows = evaluate_unet(
                    file_records=file_records,
                    args=args,
                    challenge=args.challenge,
                    acquisitions=acquisitions,
                    acceleration=acceleration,
                    center_fraction=center_fraction,
                    mask_type=args.mask_type,
                    runtime_modules=runtime_modules,
                )
            else:
                continue

            rows.extend(method_rows)
            for row in method_rows:
                print(
                    f"{row['method']},{row['acceleration']},{row['center_fraction']},"
                    f"{row['acquisition']},{row['volumes']},{row['MSE']:.6g},"
                    f"{row['NMSE']:.6g},{row['PSNR']:.6g},{row['SSIM']:.6g}"
                )

            save_rows(rows, args.output_csv, args.output_json)
            print(f"Checkpoint saved: {args.output_csv} / {args.output_json}")

    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
