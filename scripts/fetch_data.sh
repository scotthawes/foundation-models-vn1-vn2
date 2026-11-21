#!/bin/sh
# Download and unpack the VN2 competition data.
#
# The data is free to use; the competition asks that it is cited (see README.md).
# Everything under data/ is ignored by git, so this script is how a fresh clone
# gets the files.
set -eu

URL="https://supchains.com/downloads/vn2-inventory-competition.zip"
HERE=$(cd "$(dirname "$0")/.." && pwd)
RAW="$HERE/data/raw"
ZIP="$RAW/vn2-inventory-competition.zip"

mkdir -p "$RAW"
if [ ! -f "$ZIP" ]; then
    echo "Downloading $URL"
    curl -sSL --max-time 180 -o "$ZIP" "$URL"
fi
echo "Unpacking $ZIP"
unzip -o -q "$ZIP" -d "$RAW"

echo
echo "Data is in $RAW/vn2-inventory-competition"
echo "Reproduce the official benchmark with:"
echo "  python scripts/reproduce_benchmark.py"
