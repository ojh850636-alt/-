# Small programmatic test runner to execute selected test functions without pytest CLI.
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

mod = importlib.import_module('tests.test_ai_quota_modular')

failed = []
for name in dir(mod):
    if name.startswith('test_'):
        func = getattr(mod, name)
        if callable(func):
            try:
                # provide tmp_path and monkeypatch minimal substitutes when needed
                import tempfile
                from types import SimpleNamespace
                tmp = Path(tempfile.mkdtemp())
                # create a very small monkeypatch replacement
                class MP:
                    def setenv(self, k, v):
                        import os
                        os.environ[k] = v
                mp = MP()
                # call with heuristics
                try:
                    func(tmp_path=tmp, monkeypatch=mp)
                except TypeError:
                    try:
                        func(tmp_path=tmp)
                    except TypeError:
                        func()
                print(f"PASS: {name}")
            except AssertionError as e:
                print(f"FAIL: {name} - {e}")
                failed.append((name, e))
            except Exception as e:
                print(f"ERROR: {name} - {e}")
                failed.append((name, e))

if failed:
    print(f"{len(failed)} tests failed")
    sys.exit(2)
print("All done")
