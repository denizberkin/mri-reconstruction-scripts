# unzip all .tar.xz files in a given input directory, with progress bar
# Usage: ./scripts/unzip_all.sh -i /path/to/input_directory

#!/bin/bash

INPUT_DIR="."

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) INPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Count files for the summary
files=("$INPUT_DIR"/*.tar.xz)
total_files=${#files[@]}

if [ ! -e "${files[0]}" ]; then
    echo "No .tar.xz files found."
    exit 1
fi

echo "Processing $total_files files in: $INPUT_DIR"

for file in "${files[@]}"; do
    echo "Extracting: $(basename "$file")"
    
    # pv reads the file, shows progress, and pipes it to tar
    pv "$file" | pixz -d | tar -xC "$INPUT_DIR"
done

echo "Extraction complete."