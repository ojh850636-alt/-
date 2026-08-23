from typing import Optional
from pydantic import BaseModel


class CommandRequest(BaseModel):
    text: Optional[str] = None


class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    text: Optional[str] = None
    use_stub: Optional[bool] = False
    mock_error: Optional[bool] = False


class FileActionRequest(BaseModel):
    action: str
    filename: Optional[str] = None
