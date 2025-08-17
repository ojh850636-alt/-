import subprocess, sys, textwrap

script = textwrap.dedent(r"""
import sys
from pathlib import Path
proj = Path(r"c:\Users\Hi\Desktop\루시아 에이전트\2")
sys.path.insert(0, str(proj))
print('starting import')
import importlib
m = importlib.import_module('lucia_ultimate_quantum_enhanced_modular')
print('imported', m)
""")

p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
try:
    out, err = p.communicate(timeout=8)
except subprocess.TimeoutExpired:
    p.kill()
    out, err = p.communicate()
    print('timeout')
    print('stdout:', out)
    print('stderr:', err)
    sys.exit(2)

print('returncode', p.returncode)
print('stdout:', out)
print('stderr:', err)
if p.returncode != 0:
    sys.exit(1)
