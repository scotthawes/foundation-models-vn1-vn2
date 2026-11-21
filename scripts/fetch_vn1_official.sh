#!/usr/bin/env bash
# Download the official VN1 bundle: Phase 2 actuals, the winners' forecasts, the scorer.
# The data is free to use with citation; it stays in data/, which git ignores.
set -euo pipefail
cd "$(dirname "$0")/.."

url="https://supchains.com/downloads/vn1-forecasting-competition.zip"
mkdir -p data/raw/vn1-official
curl -fsSL -o data/raw/vn1-official/vn1.zip "$url"
unzip -oq data/raw/vn1-official/vn1.zip -d data/raw/vn1-official/
echo "unpacked into data/raw/vn1-official/vn1-forecasting-competition/"
