import importlib
import inspect
import pydantic
import sys
from pathlib import Path

out = Path("patches") / "diag_pyd.txt"
with out.open("w", encoding="utf-8") as f:
    f.write("PYDANTIC: %s\n" % getattr(pydantic, "__version__", "unknown"))
    for name in [
        "lucia_ultimate_quantum_integrated_fixed",
        "lucia_ultimate_quantum_integrated",
    ]:
        f.write("\nMODULE: %s\n" % name)
        try:
            mod = importlib.import_module(name)
            f.write("module file: %s\n" % getattr(mod, "__file__", "??"))
            try:
                src = inspect.getsource(mod)
                lines = src.splitlines()
                for i, line in enumerate(lines[:400], start=1):
                    f.write(f"{i:04d}: {line}\n")
                # search for .dict occurrences
                occurrences = [
                    (i + 1, line)
                    for i, line in enumerate(lines)
                    if ".dict(" in line or "req.dict" in line
                ]
                f.write("\nOCCURRENCES OF .dict(:\n")
                for ln, line in occurrences:
                    f.write(f"{ln}: {line}\n")
            except Exception as e:
                f.write("inspect.getsource failed: %r\n" % e)
        except Exception as e:
            f.write("import failed: %r\n" % e)
    # also list loaded modules with similar names
    f.write("\nLOADED MODULES SIMILAR:\n")
    for k in sorted(sys.modules.keys()):
        if "lucia_ultimate" in k:
            f.write(
                k
                + " -> "
                + str(getattr(sys.modules[k], "__file__", "built-in or none"))
                + "\n"
            )
print("WROTE", out)
