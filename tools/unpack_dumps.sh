#!/bin/bash
# Large trajectory dumps and logs are stored gzipped (*.jsonl.gz, *.log.gz).
# Run this once after cloning; the analysis scripts read the plain .jsonl files.
#   bash tools/unpack_dumps.sh          # keeps the .gz next to the unpacked file
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
find . -path ./.git -prune -o -name '*.jsonl.gz' -print | while read -r f; do
    [ -f "${f%.gz}" ] || gunzip -k "$f"
done
echo "unpacked $(find . -name '*.jsonl.gz' | wc -l) dump files"
