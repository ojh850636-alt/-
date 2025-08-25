#!/usr/bin/env bash
set -euo pipefail

PATCH_FILE="${1:-}"
if [[ -z "$PATCH_FILE" || ! -f "$PATCH_FILE" ]]; then
  echo "Usage: $0 path/to/patch.diff" >&2
  exit 1
fi

# Try git apply first; fallback to patch
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git apply --index "$PATCH_FILE" || {
    echo "git apply failed; trying patch -p0" >&2
    patch -p0 < "$PATCH_FILE"
  }
else
  patch -p0 < "$PATCH_FILE"
fi

echo "Applied $PATCH_FILE"
