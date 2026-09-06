import sys, traceback, importlib
p = r'C:\Users\Hi\Desktop\루시아 에이전트\2'
if p not in sys.path:
    sys.path.insert(0, p)
try:
    importlib.import_module('lucia_ultimate_quantum_integrated_fixed')
    print('IMPORT_OK')
except Exception:
    traceback.print_exc()
