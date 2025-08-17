"""Local verification runner

Run this script locally (not inside the assistant) to execute pytest and the
two smoke scripts. It captures stdout/stderr and exceptions and writes a JSON
report to `tools/local_verify.out.json` so CI or a developer can inspect
results even if terminal output isn't captured.

Usage (PowerShell):
  python .\tools\local_verify.py
"""

import json
import subprocess
import importlib
import sys
from pathlib import Path
from datetime import datetime, timezone


def write_report(path: Path, data: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("failed to write report:", e)


def run_pytest(repo_root: Path):
    cmd = [sys.executable, "-m", "pytest", "-q"]
    p = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def run_smoke(module_path: str, name: str, repo_root: Path):
    # import and call run() if available; capture exceptions
    result = {"ok": False, "error": None}
    try:
        sys.path.insert(0, str(repo_root))
        mod = importlib.import_module(module_path)
        if hasattr(mod, "run"):
            mod.run()
            result["ok"] = True
        else:
            result["error"] = "no run() in module"
    except Exception as e:
        result["error"] = repr(e)
    finally:
        try:
            sys.path.remove(str(repo_root))
        except Exception:
            pass
    return result


def main():
    repo_root = Path(__file__).parents[1]
    out = Path(__file__).parent / "local_verify.out.json"
    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "pytest": None,
        "smoke_ai_quota": None,
        "run_concurrency_smoke": None,
    }

    try:
        report["pytest"] = run_pytest(repo_root)
    except Exception as e:
        report["pytest"] = {"returncode": None, "stdout": "", "stderr": repr(e)}

    try:
        report["smoke_ai_quota"] = run_smoke(
            "tools.smoke_ai_quota", "smoke_ai_quota", repo_root
        )
    except Exception as e:
        report["smoke_ai_quota"] = {"ok": False, "error": repr(e)}

    try:
        report["run_concurrency_smoke"] = run_smoke(
            "tools.run_concurrency_smoke", "run_concurrency_smoke", repo_root
        )
    except Exception as e:
        report["run_concurrency_smoke"] = {"ok": False, "error": repr(e)}

    write_report(out, report)
    print("wrote report to", out)


if __name__ == "__main__":
    main()
