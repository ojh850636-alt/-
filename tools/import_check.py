import sys
from pathlib import Path

project_root = Path(r"c:\Users\Hi\Desktop\루시아 에이전트\2")
print("project_root =", project_root)

sys.path.insert(0, str(project_root))
print("sys.path[0]=", sys.path[0])

try:
    import importlib

    importlib.import_module("lucia_ultimate_quantum_enhanced_modular")
    print("import ok")
except Exception as e:
    import traceback

    traceback.print_exc()
    print("import failed:", repr(e))
