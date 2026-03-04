import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified random subset of fastMRI .h5 filenames."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/multicoil_val"),
        help="Directory containing .h5 files.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of filenames to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=727,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("selected_samples_stratified.txt"),
        help="Output text file (one filename per line).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random_generator = random.Random(args.seed)

    all_files = sorted(args.data_path.glob("*.h5"))
    if not all_files:
        raise RuntimeError(f"No .h5 files found in: {args.data_path}")

    grouped_files = defaultdict(list)
    for file_path in all_files:
        parts = file_path.name.split("_")
        if len(parts) < 4:
            continue
        grouped_files[parts[2]].append(file_path.name)

    if not grouped_files:
        raise RuntimeError("Could not parse acquisition groups from filenames.")

    for acquisition in grouped_files:
        random_generator.shuffle(grouped_files[acquisition])

    target = min(args.num_samples, sum(len(values) for values in grouped_files.values()))
    acquisition_types = sorted(grouped_files)
    base_per_type = target // len(acquisition_types)

    selected_files = [
        name
        for acquisition in acquisition_types
        for name in grouped_files[acquisition][:base_per_type]
    ]

    remaining_pool = [
        name
        for acquisition in acquisition_types
        for name in grouped_files[acquisition][base_per_type:]
    ]
    random_generator.shuffle(remaining_pool)
    selected_files += remaining_pool[: target - len(selected_files)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected_files) + "\n", encoding="utf-8")

    print(f"Saved {len(selected_files)} filenames to: {args.output}")
    print("Acquisition counts:", Counter(name.split("_")[2] for name in selected_files))


if __name__ == "__main__":
    main()