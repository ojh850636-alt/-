import warnings
import sys

# Suppress Pydantic deprecation warnings early
try:
    from pydantic.errors import PydanticDeprecatedSince20

    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except Exception:
    warnings.filterwarnings("ignore", message=r".*The `dict` method is deprecated.*")

# Run pytest programmatically
import pytest

rc = pytest.main(["-q"])
print("PYTEST RC", rc)
sys.exit(rc)
