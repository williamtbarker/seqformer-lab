#!/usr/bin/env bash
set -euo pipefail

project_root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"

python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
python -m build

smoke_dir=$(mktemp -d "${TMPDIR:-/tmp}/seqformer-lab.XXXXXX")
trap 'rm -rf "$smoke_dir"' EXIT HUP INT TERM
seqformer-lab demo --output "$smoke_dir/demo" --epochs 1 --seed 23 >/dev/null
seqformer-lab evaluate \
  --data "$smoke_dir/demo/dataset.jsonl" \
  --checkpoint "$smoke_dir/demo" >/dev/null

printf 'All checks passed, including the deterministic CPU demo.\n'
