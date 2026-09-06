import os
import sys

# Ensure Pydantic deprecation warnings are ignored early
os.environ["PYTHONWARNINGS"] = (
    os.environ.get("PYTHONWARNINGS", "")
    + ";ignore::pydantic.errors.PydanticDeprecatedSince20"
)

# Also set pytest filterwarnings via env
os.environ["PYTEST_ADDOPTS"] = (
    os.environ.get("PYTEST_ADDOPTS", "") + " -q --disable-warnings"
)

import pytest

if __name__ == "__main__":
    rc = pytest.main([])
    sys.exit(rc)
