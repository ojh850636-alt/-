import importlib
import inspect
import sys

TEST_MODULES = [
    "tests.test_file_ops",
    "tests.test_ai_quota",
    "tests.test_ai_quota_modular",
    "tests.test_ai_quota_concurrency",
    "tests.test_endpoint_schemas",
]

failed = False
for m in TEST_MODULES:
    print("\n=== running", m, "===")
    try:
        mod = importlib.import_module(m)
    except Exception as e:
        print("IMPORT FAILED", m, type(e).__name__, e)
        failed = True
        continue
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            print("running", name)
            try:
                fn()
            except Exception as e:
                print("FAILED", name, type(e).__name__, e)
                failed = True

print("\nALL_OK", not failed)
if failed:
    sys.exit(2)
