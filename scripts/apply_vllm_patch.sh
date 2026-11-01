#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/vllm-checkout" >&2
  exit 2
fi

target_root=$(cd "$1" && pwd)
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -f "$target_root/vllm/v1/core/block_pool.py" ]]; then
  echo "Target is not a compatible vLLM source checkout: $target_root" >&2
  exit 1
fi

cp -R "$script_root/patches/vllm/vllm/." "$target_root/vllm/"
cp -R "$script_root/patches/vllm/tests/." "$target_root/tests/"

echo "Applied the MnemoRAG vLLM overlay to $target_root"
