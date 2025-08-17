import sys, json, traceback
from pathlib import Path
proj = Path(r"c:\Users\Hi\Desktop\루시아 에이전트\2")
sys.path.insert(0, str(proj))
out = []
try:
    import importlib
    out.append('importlib OK')
    mod = importlib.import_module('lucia_ultimate_quantum_enhanced_modular')
    out.append('module imported: ' + repr(mod))
    for name in ('state','quantum_core','emotion','sensor','threat','evolution','crypto','tokens','comms','decision'):
        val = getattr(mod, name, None)
        out.append(f"{name}: {type(val)} repr={repr(val)[:200]}")
    try:
        res = mod.health()
        out.append('health(): ' + json.dumps(res, default=str, ensure_ascii=False))
    except Exception as e:
        out.append('health() raised:')
        out.append(traceback.format_exc())
except Exception:
    out.append('import failed:')
    out.append(traceback.format_exc())

p = Path(__file__).parent / 'health_debug_out.txt'
p.write_text('\n'.join(out), encoding='utf8')
print('wrote', p)
