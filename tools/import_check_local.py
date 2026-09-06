import importlib
import sys

modules = [
    "lucia_ultimate_quantum_integrated_fixed",
    "lucia_core.command_parser",
    "lucia_core.dispatcher",
    "lucia_core.file_ops",
    "lucia_core.state",
    "lucia_core.schemas",
    "ai_quota",
    "ai_providers",
]
ok = True
for m in modules:
    try:
        importlib.import_module(m)
        print("OK", m)
    except Exception as e:
        ok = False
        print("FAIL", m, type(e).__name__, str(e))
if not ok:
    sys.exit(2)
