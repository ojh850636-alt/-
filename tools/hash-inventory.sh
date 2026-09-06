#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(pwd)}"
OUT_DIR="$ROOT_DIR/artifacts"
mkdir -p "$OUT_DIR"

HASH_TSV="$OUT_DIR/hash_inventory.tsv"
NAME_COLLISIONS_TSV="$OUT_DIR/name_collisions.tsv"

# Exclude common binary/build dirs
exclude_dirs=(.git node_modules dist build out .venv venv .direnv artifacts .cache target vendor)

prune_expr=()
for d in "${exclude_dirs[@]}"; do
  prune_expr+=( -name "$d" -prune -o )
done

# Exclude obvious binaries by extension
binary_exts="jpg|jpeg|png|gif|bmp|svg|ico|pdf|zip|gz|tar|tgz|7z|rar|mp3|mp4|mov|avi|ogg|wasm|class|jar|exe|dll|so|dylib|bin"

# Build file list
mapfile -t files < <(find "$ROOT_DIR" \
  \( ${prune_expr[*]} -type f -print \) 2>/dev/null | \
  grep -Ev "\\.($binary_exts)$" | \
  grep -Ev "/\\.git/|/artifacts/|/node_modules/|/dist/|/build/|/out/|/target/|/vendor/|/\\.venv/|/venv/|/\\.direnv/")

# Compute hashes
{
  printf "hash\tsize\tpath\n"
  for f in "${files[@]}"; do
    if [[ -f "$f" ]]; then
      size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f") || size=0
      hash=$(sha256sum "$f" | awk '{print $1}')
      printf "%s\t%s\t%s\n" "$hash" "$size" "${f#$ROOT_DIR/}"
    fi
  done | sort -k1,1 -k3,3
} > "$HASH_TSV"

# Name collisions: same basename different hash
{
  printf "basename\thash\tsize\tpath\n"
  tail -n +2 "$HASH_TSV" | awk -F"\t" '{print $3"\t"$1"\t"$2"\t"$3}' | \
  awk -F"\t" '{ n=split($1, a, "/"); base=a[n]; print base"\t"$2"\t"$3"\t"$4 }' | \
  sort -k1,1 -k2,2 | \
  awk -F"\t" 'NR==1{prev=$1; prevline=$0; next} {if($1==prev){print prevline; print $0} prev=$1; prevline=$0}' | \
  sort -u || true
} > "$NAME_COLLISIONS_TSV"

printf "Wrote %s and %s\n" "$HASH_TSV" "$NAME_COLLISIONS_TSV"
