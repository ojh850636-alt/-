"""Robust startup shim: wrap warnings.warn to suppress the specific
Pydantic deprecation message about `.dict()` and provide a safe
BaseModel.dict -> model_dump fallback when available.
"""

import warnings
from pathlib import Path

# Wrap warnings.warn to quietly ignore the pydantic dict() deprecation.
_orig_warn = warnings.warn


def _wrapped_warn(message, category=None, *args, **kwargs):
    try:
        msg_text = str(message)
    except Exception:
        msg_text = ""
    # If the message matches the pydantic dict deprecation, ignore it.
    if "The `dict` method is deprecated" in msg_text:
        return
    # Some environments pass a warning subclass; match by name too.
    if category is not None:
        try:
            cname = getattr(category, "__name__", "")
            if "PydanticDeprecatedSince" in cname:
                return
        except Exception:
            pass
    return _orig_warn(message, category=category, *args, **kwargs)


warnings.warn = _wrapped_warn

# Also provide a safe BaseModel.dict shim when pydantic is present
try:
    from pydantic import BaseModel

    if hasattr(BaseModel, "model_dump"):

        def _base_model_safe_dict(self, *args, **kwargs):
            try:
                return self.model_dump(*args, **kwargs)
            except Exception:
                return getattr(self, "__dict__", {})

        try:
            BaseModel.dict = _base_model_safe_dict
        except Exception:
            pass
except Exception:
    pass

try:
    # Import early so this runs before most test imports
    import traceback as _tb
    from pathlib import Path

    try:
        from pydantic import BaseModel as _BaseModel

        def _debug_dict(self, *a, **k):
            st = "".join(_tb.format_stack())
            p = Path(__file__).parent / "patches" / "dict_shim.log"
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a", encoding="utf-8") as f:
                    f.write("DEBUG BaseModel.dict called from:\n")
                    f.write(st + "\n----\n")
            except Exception:
                pass
            # raise so pytest shows the stack where .dict() was invoked
            raise RuntimeError(
                "DEBUG: BaseModel.dict called; see patches/dict_shim.log for stack"
            )

        # install shim
        try:
            _BaseModel.dict = _debug_dict
        except Exception:
            pass
    except Exception:
        # pydantic not available; nothing to do
        pass
except Exception:
    pass
"""Startup shim to suppress Pydantic v2 dict() deprecation warnings and
provide a temporary BaseModel.dict shim that delegates to model_dump.

This file is intentionally small and placed at the repo root so test runs
and CI that run from the repository will import it early and avoid
polluting logs with PydanticDeprecation warnings.
"""
import warnings

try:
    from pydantic.errors import PydanticDeprecatedSince20

    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except Exception:
    warnings.filterwarnings("ignore", message=r".*The `dict` method is deprecated.*")

try:
    from pydantic import BaseModel

    # Only install the shim when model_dump exists (v2) and dict is the old API
    if hasattr(BaseModel, "model_dump"):

        def _base_model_safe_dict(self, *args, **kwargs):
            try:
                return self.model_dump(*args, **kwargs)
            except Exception:
                return getattr(self, "__dict__", {})

        BaseModel.dict = _base_model_safe_dict
except Exception:
    # best-effort shim; don't fail startup if pydantic isn't present
    pass
