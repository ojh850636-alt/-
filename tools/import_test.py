import importlib
import time
from pathlib import Path

outf = Path(__file__).parent / 'import_result.txt'
try:
    importlib.import_module('lucia_ultimate_quantum_enhanced_modular')
    with open(outf, 'w', encoding='utf-8') as f:
        f.write('ok')
except Exception as e:
    with open(outf, 'w', encoding='utf-8') as f:
        f.write('error: ' + str(e))
