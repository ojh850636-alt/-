import pkgutil
import importlib
import sys
from pathlib import Path
out = Path(__file__).parent.parent / 'patches' / 'pydantic_inspect.txt'
with open(out, 'w', encoding='utf-8') as f:
    try:
        p = importlib.import_module('pydantic')
        f.write('pydantic version: ' + getattr(p, '__version__', 'unknown') + '\n')
        found = []
        for attr in dir(p):
            if 'Deprecated' in attr or 'Pydantic' in attr or 'warning' in attr.lower():
                found.append(attr)
        f.write('matches in pydantic top-level:\n')
        for a in found:
            f.write('  ' + a + '\n')
        # try to import common locations
        try:
            import pydantic.errors as pe
            f.write('\nattributes in pydantic.errors:\n')
            for attr in dir(pe):
                if 'Deprecated' in attr or 'Pydantic' in attr or 'warning' in attr.lower():
                    f.write('  ' + attr + '\n')
        except Exception as e:
            f.write('\nno pydantic.errors: ' + str(e) + '\n')
        try:
            import pydantic.warnings as pw
            f.write('\nattributes in pydantic.warnings:\n')
            for attr in dir(pw):
                if 'Deprecated' in attr or 'Pydantic' in attr or 'warning' in attr.lower():
                    f.write('  ' + attr + '\n')
        except Exception as e:
            f.write('\nno pydantic.warnings: ' + str(e) + '\n')
    except Exception as e:
        f.write('pydantic import failed: ' + str(e) + '\n')
print('WROTE', out)
