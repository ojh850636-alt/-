import warnings

try:
    from pydantic.errors import PydanticDeprecatedSince20

    warnings.filterwarnings("error", category=PydanticDeprecatedSince20)
except Exception:
    warnings.filterwarnings("error", message=r".*The `dict` method is deprecated.*")

import pytest

rc = pytest.main(["-q"])
print("PYTEST RC", rc)
