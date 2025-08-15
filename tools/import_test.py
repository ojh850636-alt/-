import importlib
import time
from pathlib import Path

outf = Path(__file__).parent / 'import_result.txt'
import traceback
import sys

def main() -> int:
    try:
        importlib.import_module('lucia_ultimate_quantum_enhanced_modular')
        outf.write_text('ok', encoding='utf-8')
        print('ok')
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"error: {e}\n{tb}"
        outf.write_text(msg, encoding='utf-8')
        print('error:', e)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
