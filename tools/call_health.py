import sys
from pathlib import Path

proj = Path(r"c:\Users\Hi\Desktop\루시아 에이전트\2")
sys.path.insert(0, str(proj))
print("sys.path[0]=", sys.path[0])
try:
    import importlib

    mod = importlib.import_module("lucia_ultimate_quantum_enhanced_modular")
    print("imported module, app=", getattr(mod, "app", None))
    try:
        res = mod.health()
        print("health res:", res)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("health call failed:", repr(e))
except Exception as e:
    import traceback

    traceback.print_exc()
    print("import failed:", repr(e))
