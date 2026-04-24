#!/usr/bin/env bash
# Apply in-tree vLLM-Ascend worker hook for PTO (see README.md in this directory).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/apply_vllm_ascend_pto_hook.py" "$@"
