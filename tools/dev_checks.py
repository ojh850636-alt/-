import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    return p.returncode


PY = Path(sys.executable)

print("1) pytest")
rc = run([str(PY), "-m", "pytest", "-q"])
if rc != 0:
    sys.exit(rc)

print("\n2) ruff --fix")
rc = run([str(PY), "-m", "ruff", "--fix", "."])
if rc != 0:
    sys.exit(rc)

print("\n3) docker build (skipped if docker missing)")
from shutil import which

if which("docker"):
    rc = run(["docker", "build", "-t", "lucia-app", "."])
    if rc != 0:
        sys.exit(rc)
else:
    print("docker not found; skipped")

print("\nAll checks completed")
