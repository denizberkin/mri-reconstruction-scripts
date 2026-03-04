#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Iterable, Optional, Set

import h5py
from tqdm import tqdm


def load_runtime_modules() -> dict[str, object]:
    import fastmri
    from fastmri.data import transforms
    from fastmri.data.mri_data import et_query

    return {
        "fastmri": fastmri,
        "transforms": transforms,
        "et_query": et_query,
    }


def add_fastmri_repo_to_path(fastmri_repo: Path) -> None:
    repo = fastmri_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


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


def save_zero_filled(
    data_dir: Path,
    out_dir: Path,
    challenge: str,
    runtime_modules: dict[str, object],
    include_files: Optional[Set[str]] = None,
    max_files: int = 0,
) -> None:
    fastmri_module = runtime_modules["fastmri"]
    transforms_module = runtime_modules["transforms"]
    et_query_func = runtime_modules["et_query"]

    reconstructions = {}

    files = sorted(data_dir.glob("*.h5"))
    if include_files is not None:
        files = [path for path in files if path.name in include_files]
    if max_files > 0:
        files = files[:max_files]

    for fname in tqdm(files, desc="Zero-filled"):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            masked_kspace = transforms_module.to_tensor(hf["kspace"][()])

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

            reconstructions[fname.name] = image

    out_dir.mkdir(parents=True, exist_ok=True)
    fastmri_module.save_reconstructions(reconstructions, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-filled baseline on a folder of fastMRI .h5 files."
    )
    parser.add_argument(
        "--fastmri_repo",
        type=Path,
        default=Path("fastMRI"),
        help="Path to fastMRI repository root.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/multicoil_val"),
        help="Path to a folder containing .h5 files.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/zero_filled_val"),
        help="Output directory for reconstructions.",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        default="multicoil",
        help="Challenge type.",
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
        help="If > 0 and --file/--file_list are not set, use first N sorted volumes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_fastmri_repo_to_path(args.fastmri_repo)
    runtime_modules = load_runtime_modules()
    include_files = parse_file_filter(args.file)
    if include_files is None:
        include_files = parse_file_list(args.file_list)

    save_zero_filled(
        args.data_path,
        args.output_path,
        args.challenge,
        runtime_modules=runtime_modules,
        include_files=include_files,
        max_files=args.max_files,
    )
    print(f"Saved zero-filled reconstructions to: {args.output_path}")


if __name__ == "__main__":
    main()
