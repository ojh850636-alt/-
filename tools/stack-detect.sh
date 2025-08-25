#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(pwd)}"
OUT_DIR="$ROOT_DIR/artifacts"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/stack_report.txt"

cd "$ROOT_DIR"

langs=$(find . -type f \
  -not -path "*/.git/*" -not -path "*/node_modules/*" -not -path "*/dist/*" -not -path "*/build/*" -not -path "*/artifacts/*" \
  -regex ".*\\.\(ts\|tsx\|js\|jsx\|py\|go\|rs\|java\|kt\|kts\|php\|rb\|cs\|cpp\|c\|m\|mm\|swift\)" \
  | awk -F. '{print $NF}' | sort | uniq -c | sort -nr)

pmaps=()
for f in package.json pnpm-lock.yaml yarn.lock bun.lockb requirements.txt pyproject.toml Pipfile poetry.lock \
         go.mod Cargo.toml pom.xml build.gradle build.gradle.kts composer.json Gemfile \
         mix.exs shard.yml setup.py Makefile CMakeLists.txt; do
  [[ -f "$f" ]] && pmaps+=("$f")
done

entries=$(grep -RIl --include='*.{ts,tsx,js,jsx,py,go,rs,java,kt,kts}' -e 'main\(' -e 'main\s*:' -e 'if __name__ == "__main__"' 2>/dev/null || true)

{
  echo "# Stack Report"
  echo "Date: $(date -Iseconds)"
  echo
  echo "## Languages (by extension count)"
  if [[ -n "${langs}" ]]; then
    echo "$langs"
  else
    echo "(none detected)"
  fi
  echo
  echo "## Package/Build markers"
  if [[ ${#pmaps[@]} -gt 0 ]]; then
    printf '%s\n' "${pmaps[@]}"
  else
    echo "(none detected)"
  fi
  echo
  echo "## Potential entry points"
  if [[ -n "${entries:-}" ]]; then
    printf '%s\n' ${entries}
  else
    echo "(none detected)"
  fi
} > "$REPORT"

echo "Wrote $REPORT"
