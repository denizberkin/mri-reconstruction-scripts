#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Set

import h5py
import numpy as np
import requests
import torch
from tqdm import tqdm

UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    "unet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def load_runtime_modules() -> dict[str, object]:
    import fastmri
    import fastmri.data.transforms as T
    from fastmri.data import SliceDataset
    from fastmri.models import Unet

    return {
        "fastmri": fastmri,
        "T": T,
        "SliceDataset": SliceDataset,
        "Unet": Unet,
    }


def add_fastmri_repo_to_path(fastmri_repo: Path) -> None:
    repo = fastmri_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def download_model(url: str, fname: Path) -> None:
    response = requests.get(url, timeout=20, stream=True)
    response.raise_for_status()

    chunk_size = 1024 * 1024
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as handle:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                progress_bar.update(len(chunk))
                handle.write(chunk)
    progress_bar.close()


def parse_file_filter(files: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if not files:
        return None
    out: Set[str] = set()
    for item in files:
        for name in item.split(","):
            name = name.strip()
            if name:
                out.add(name)
    return out or None


def parse_file_list(file_list: Optional[Path]) -> Optional[Set[str]]:
    if file_list is None:
        return None

    wanted = {
        line.strip()
        for line in file_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return wanted or None


def is_readable_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def select_input_files(
    data_path: Path,
    include_files: Optional[Set[str]],
    max_files: int,
) -> tuple[list[Path], list[Path]]:
    files = sorted(data_path.glob("*.h5"))

    if include_files is not None:
        files = [path for path in files if path.name in include_files]

    if include_files is None and max_files > 0:
        files = files[:max_files]

    readable_files: list[Path] = []
    skipped_files: list[Path] = []
    for path in files:
        if is_readable_h5(path):
            readable_files.append(path)
        else:
            skipped_files.append(path)
    return readable_files, skipped_files


def stage_files(selected_files: list[Path], stage_dir: Path) -> Path:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for source in selected_files:
        destination = stage_dir / source.name
        os.symlink(source.resolve(), destination)

    return stage_dir


def run_unet_model(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch

    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()
    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()

    return output, int(slice_num[0]), fname[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained fastMRI U-Net inference with optional subset filtering."
    )
    parser.add_argument("--fastmri_repo", type=Path, default=Path("fastMRI"))
    parser.add_argument(
        "--challenge",
        default="unet_brain_mc",
        choices=("unet_knee_sc", "unet_knee_mc", "unet_brain_mc"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--state_dict_file",
        type=Path,
        default=None,
        help="Path to saved state_dict; downloads official model if omitted.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/multicoil_val"),
        help="Path to folder containing input .h5 files.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/unet_val"),
        help="Output directory; reconstructions go under output_path/reconstructions.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Specific .h5 filename(s) to include; repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help="Optional text file with one .h5 filename per line.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If > 0 and --file is not set, use first N sorted volumes.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_fastmri_repo_to_path(args.fastmri_repo)
    runtime_modules = load_runtime_modules()
    fastmri_module = runtime_modules["fastmri"]
    transforms_module = runtime_modules["T"]
    slice_dataset_class = runtime_modules["SliceDataset"]
    unet_class = runtime_modules["Unet"]

    model = unet_class(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)

    state_dict_file = args.state_dict_file
    if state_dict_file is None:
        default_fname = Path(MODEL_FNAMES[args.challenge])
        if not default_fname.exists():
            download_model(UNET_FOLDER + default_fname.name, default_fname)
        state_dict_file = default_fname

    model.load_state_dict(torch.load(state_dict_file, map_location="cpu"))
    model = model.eval().to(torch.device(args.device))

    if "_mc" in args.challenge:
        which_challenge = "multicoil"
    else:
        which_challenge = "singlecoil"

    include_files = parse_file_filter(args.file)
    if include_files is None:
        include_files = parse_file_list(args.file_list)

    selected_files, skipped_files = select_input_files(args.data_path, include_files, args.max_files)
    if not selected_files:
        raise RuntimeError("No readable .h5 files selected for U-Net inference.")

    if skipped_files:
        print(f"Warning: skipped {len(skipped_files)} unreadable/truncated file(s).")
        for skipped_path in skipped_files[:10]:
            print(f"  - {skipped_path.name}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")

    stage_root = args.output_path / "_subset_input"
    dataset_root = stage_files(selected_files, stage_root)

    dataset = slice_dataset_class(
        root=dataset_root,
        transform=transforms_module.UnetDataTransform(which_challenge=which_challenge),
        challenge=which_challenge,
        raw_sample_filter=None,
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.num_workers)

    start_time = time.perf_counter()
    outputs = defaultdict(list)

    for batch in tqdm(dataloader, desc="Running U-Net inference"):
        with torch.no_grad():
            output, slice_num, fname = run_unet_model(batch, model, torch.device(args.device))
        outputs[fname].append((slice_num, output))

    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    recon_dir = args.output_path / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    fastmri_module.save_reconstructions(outputs, recon_dir)

    elapsed = time.perf_counter() - start_time
    print(f"Saved U-Net reconstructions to: {recon_dir}")
    print(f"Processed {len(outputs)} volumes in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
