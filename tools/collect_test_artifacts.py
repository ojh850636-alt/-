"""Simple helper to gather common test artifacts into a folder for CI upload.
Usage: python tools/collect_test_artifacts.py <output_dir>
"""
import sys
from pathlib import Path
import shutil

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_artifacts")
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).parent.parent

CANDIDATES = [
    ROOT / "smoke_debug.log",
    ROOT / "smoke_test_result.json",
    ROOT / "tools" / "webhook_events.log",
]

for p in CANDIDATES:
    if p.exists():
        try:
            shutil.copy(p, OUT / p.name)
        except Exception:
            pass

print("collected")
