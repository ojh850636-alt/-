import warnings

# Some pydantic installations don't expose the internal warning class name.
# Use a message-based filter which is portable across versions.
warnings.filterwarnings('ignore', message=r".*The `dict` method is deprecated.*")
