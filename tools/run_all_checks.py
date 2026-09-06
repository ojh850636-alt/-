import subprocess
import sys
import os

root = os.path.dirname(os.path.dirname(__file__))
print("PROJECT ROOT:", root)


def run(cmd, check=False):
    print("\n$", cmd)
    r = subprocess.run(cmd, shell=True, cwd=root)
    if check and r.returncode != 0:
        print("COMMAND FAILED:", r.returncode)
        sys.exit(r.returncode)
    return r.returncode


# 1. import check
run("python tools/import_check_local.py", check=True)
# 2. run smoke
run("python tools/run_smoke.py", check=True)
# 3. run direct tests
run("python tools/run_tests_direct.py", check=True)
# 4. run pytest
run("python -m pytest -q", check=False)
print("\nALL CHECKS COMPLETED")
