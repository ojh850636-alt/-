import importlib
import inspect
import pydantic

out = []
try:
    mod = importlib.import_module("lucia_ultimate_quantum_integrated_fixed")
    out.append("MODULE_FILE: " + str(getattr(mod, "__file__", "unknown")))
    try:
        src = inspect.getsource(mod).splitlines()
        out.append("---SOURCE LINES 60-90---")
        for i in range(59, 91):
            if i < len(src):
                out.append(f"{i + 1:04d}: {src[i]}")
            else:
                out.append(f"{i + 1:04d}: <no line>")
        occ = [
            (i + 1, line) for i, line in enumerate(src) if ".dict(" in line or "req.dict" in line
        ]
        out.append("---OCCURRENCES .dict ---")
        if occ:
            for ln, line in occ:
                out.append(f"{ln}: {line}")
        else:
            out.append("<none>")
    except Exception as e:
        out.append("inspect.getsource failed: " + repr(e))
except Exception as e:
    out.append("import failed: " + repr(e))
out.append("PYDANTIC_VERSION: " + getattr(pydantic, "__version__", "unknown"))
# write to file
with open("tools/detect_module.out", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("WROTE tools/detect_module.out")
