#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm


HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def is_valid_h5(path: Path) -> bool:
    try:
        if path.stat().st_size <= 0:
            return False
        with path.open("rb") as handle:
            return handle.read(8) == HDF5_SIGNATURE
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fastMRI ESPIRiT (BART) on a selected subset of volumes."
    )
    parser.add_argument("--fastmri_repo", type=Path, default=Path("fastMRI"))
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/multicoil_val"),
        help="Directory containing source .h5 files.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/espirit_val_subset"),
        help="Directory where reconstructions and staged subset input are written.",
    )
    parser.add_argument("--challenge", choices=["singlecoil", "multicoil"], default="multicoil")
    parser.add_argument("--split", choices=["val", "test", "challenge"], default="val")
    parser.add_argument("--mask_type", choices=["random", "equispaced"], default="equispaced")
    parser.add_argument("--reg_wt", type=float, default=0.01)
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--num_procs", type=int, default=2)
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If > 0, cap selected files to first N after filters; default 0 means no cap.",
    )
    parser.add_argument(
        "--bart_toolbox_path",
        type=Path,
        default=Path(os.environ["TOOLBOX_PATH"]) if "TOOLBOX_PATH" in os.environ else None,
        help="Path to BART toolbox root (contains `bart` binary and `python/bart.py`).",
    )
    parser.add_argument(
        "--include_pattern",
        type=str,
        default="",
        help="Optional substring filter for volume names, e.g. 'AXFLAIR'.",
    )
    parser.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help="Optional text file with one .h5 filename per line.",
    )
    parser.add_argument(
        "--keep_staging",
        action="store_true",
        help="Keep staged subset directory after run (default behavior).",
    )
    return parser.parse_args()


def prepare_env(fastmri_repo: Path, bart_toolbox_path: Path | None) -> dict:
    env = os.environ.copy()

    py_paths = [str(fastmri_repo)]
    if bart_toolbox_path is not None:
        py_paths.append(str((bart_toolbox_path / "python").resolve()))

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(py_paths + ([existing] if existing else []))

    if bart_toolbox_path is not None:
        env["TOOLBOX_PATH"] = str(bart_toolbox_path.resolve())
        env["BART_TOOLBOX_PATH"] = str(bart_toolbox_path.resolve())

    return env


def preflight_bart(python_executable: str, env: dict, bart_toolbox_path: Path | None) -> None:
    if bart_toolbox_path is None:
        raise RuntimeError(
            "Missing BART toolbox path!"
        )

    bart_bin = bart_toolbox_path / "bart"
    bart_py = bart_toolbox_path / "python" / "bart.py"
    if not bart_bin.exists() or not bart_py.exists():
        raise RuntimeError(
            f"Invalid BART path: {bart_toolbox_path}\n"
            "Expected both files:\n"
            f"  - {bart_bin}\n"
            f"  - {bart_py}"
        )

    check = subprocess.run(
        [python_executable, "-c", "import bart; print('bart_ok')"],
        env=env,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "Could not import Python module `bart`.\n"
            "Make sure BART is built and paths are set correctly.\n"
            f"Command stderr:\n{check.stderr.strip()}"
        )


def choose_files(args: argparse.Namespace) -> List[Path]:
    all_files = sorted(args.data_path.glob("*.h5"))

    if args.include_pattern:
        all_files = [p for p in all_files if args.include_pattern in p.name]

    all_files = [p for p in all_files if is_valid_h5(p)]

    if args.file_list:
        wanted = {
            line.strip()
            for line in args.file_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        all_files = [p for p in all_files if p.name in wanted]

    if args.max_files > 0:
        all_files = all_files[: args.max_files]

    if not all_files:
        raise RuntimeError("No files selected for ESPIRiT run.")

    return all_files


def stage_subset(selected_files: List[Path], stage_root: Path, challenge: str, split: str) -> None:
    stage_data_dir = stage_root / f"{challenge}_{split}"
    if stage_data_dir.exists():
        shutil.rmtree(stage_data_dir)
    stage_data_dir.mkdir(parents=True, exist_ok=True)

    for src in tqdm(selected_files, desc="Staging subset"):
        dst = stage_data_dir / src.name
        os.symlink(src.resolve(), dst)


def write_selected_files_log(selected_files: List[Path], output_path: Path) -> Path:
    log_path = output_path / "selected_files.txt"
    lines = [f"{path.name}\n" for path in selected_files]
    log_path.write_text("".join(lines), encoding="utf-8")
    return log_path


def main() -> None:
    args = parse_args()

    args.data_path = args.data_path.resolve()
    args.output_path = args.output_path.resolve()
    if args.bart_toolbox_path is not None:
        args.bart_toolbox_path = args.bart_toolbox_path.resolve()

    selected_files = choose_files(args)
    args.output_path.mkdir(parents=True, exist_ok=True)
    selected_files_log = write_selected_files_log(selected_files, args.output_path)

    stage_root = args.output_path / "_subset_input"
    stage_subset(selected_files, stage_root, args.challenge, args.split)

    fastmri_repo = args.fastmri_repo.resolve()
    cs_dir = fastmri_repo / "fastmri_examples" / "cs"
    run_bart_py = cs_dir / "run_bart.py"

    env = prepare_env(fastmri_repo, args.bart_toolbox_path)
    preflight_bart(sys.executable, env, args.bart_toolbox_path)

    cmd = [
        sys.executable,
        str(run_bart_py),
        "--challenge",
        args.challenge,
        "--data_path",
        str(stage_root),
        "--output_path",
        str(args.output_path / "reconstructions"),
        "--split",
        args.split,
        "--mask_type",
        args.mask_type,
        "--reg_wt",
        str(args.reg_wt),
        "--num_iters",
        str(args.num_iters),
        "--num_procs",
        str(args.num_procs),
    ]

    print("Selected files:")
    for path in selected_files:
        print(f"  - {path.name}")
    print(f"Selected files log: {selected_files_log}")
    print(f"BART toolbox path: {args.bart_toolbox_path}")
    print("\nRunning:", " ".join(cmd))

    subprocess.run(cmd, check=True, cwd=cs_dir, env=env)

    print(f"\nESPIRiT reconstructions saved to: {args.output_path / 'reconstructions'}")
    print(f"Staged subset directory: {stage_root}")


if __name__ == "__main__":
    main()
