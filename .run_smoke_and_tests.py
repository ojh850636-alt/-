import sys, os, traceback, subprocess, json
p = os.path.dirname(__file__)
paths = [p, os.path.join(p, '')]
sys.path[:0] = paths
out = {'py_compile': None, 'cli': None, 'pytest': None, 'errors': []}
try:
    import py_compile
    try:
        py_compile.compile(os.path.join(p, 'lucia_quantum_fusion.py'), doraise=True)
        out['py_compile'] = 'ok'
    except Exception as e:
        out['py_compile'] = 'error'
        out['errors'].append({'py_compile': str(e)})
except Exception as e:
    out['errors'].append({'py_compile_import': str(e)})
# run CLI
try:
    import importlib
    m = importlib.import_module('lucia_quantum_fusion')
    try:
        rc = m.cli(['ingest-samples'])
        out['cli'] = {'rc': rc}
    except Exception as e:
        out['cli'] = {'exception': str(e)}
        out['errors'].append({'cli': traceback.format_exc()})
except Exception as e:
    out['errors'].append({'cli_import': traceback.format_exc()})
# run pytest
try:
    pr = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=p, capture_output=True, text=True, timeout=600)
    out['pytest'] = {'returncode': pr.returncode, 'stdout': pr.stdout[-2000:], 'stderr': pr.stderr[-2000:]}
except Exception as e:
    out['errors'].append({'pytest_run': traceback.format_exc()})
# write result
with open(os.path.join(p, 'smoke_test_result.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('written smoke_test_result.json')
