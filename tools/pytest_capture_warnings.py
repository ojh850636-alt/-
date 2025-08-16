import warnings, traceback, sys

# Treat Pydantic deprecation warnings as errors to get a traceback
try:
    from pydantic.errors import PydanticDeprecatedSince20
    warnings.filterwarnings('error', category=PydanticDeprecatedSince20)
except Exception:
    warnings.filterwarnings('error', message=r".*The `dict` method is deprecated.*")

import pytest
rc = pytest.main(['-q'])
print('PYTEST RC', rc)
sys.exit(rc)
