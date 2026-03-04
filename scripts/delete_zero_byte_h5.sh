# This script finds and deletes all zero-byte .h5 files in the specified target directory (default: data/multicoil_val). 
# Usage: ./scripts/delete_zero_byte_h5.sh [target_directory]

#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-data/multicoil_val}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Directory not found: $TARGET_DIR" >&2
    exit 1
fi

echo "Scanning for zero-byte .h5 files in: $TARGET_DIR"
count=$(find "$TARGET_DIR" -type f -name '*.h5' -size 0c | wc -l)

if [[ "$count" -eq 0 ]]; then
    echo "No zero-byte .h5 files found."
    exit 0
fi

echo "Deleting $count zero-byte .h5 file(s)..."
find "$TARGET_DIR" -type f -name '*.h5' -size 0c -print -delete
echo "Done."
