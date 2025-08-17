import warnings
import os
from pathlib import Path

# Ensure deterministic test behavior for AI provider mocks
os.environ.setdefault('AI_USE_MOCK', 'true')
os.environ.setdefault('AI_DAILY_CALL_LIMIT', '1000')
os.environ.setdefault('AI_MAX_CALLS_PER_MINUTE', '100')

# Ensure downloads directory exists for file operations tests
Path('downloads').mkdir(exist_ok=True)

# Some pydantic installations don't expose the internal warning class name.
# Use a message-based filter which is portable across versions.
warnings.filterwarnings('ignore', message=r".*The `dict` method is deprecated.*")
