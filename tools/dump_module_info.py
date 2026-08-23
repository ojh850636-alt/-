import importlib
import inspect
import pydantic

out = []
try:
    mod = importlib.import_module("lucia_ultimate_quantum_integrated_fixed")
    out.append("MODULE_FILE: " + str(getattr(mod, "__file__", "??")))
    src = inspect.getsource(mod).splitlines()
    out.append(
        "\n".join(f"{i + 1:04d}: {line}" for i, line in enumerate(src[59:91], start=60))
    )
    occ = [(i + 1, line) for i, line in enumerate(src) if ".dict(" in line or "req.dict" in line]
    out.append("\nOCCURRENCES:")
    out.append(str(occ))
except Exception as e:
    out.append("ERR: " + repr(e))
out.append("PYDANTIC: " + getattr(pydantic, "__version__", "unknown"))
with open("tools/dump_module_info.out", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("WROTE tools/dump_module_info.out")
