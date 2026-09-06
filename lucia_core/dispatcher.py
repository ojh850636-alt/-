from typing import Dict, Any
from . import file_ops
from .state import state


async def execute_command(command: Dict[str, Any]) -> Dict[str, Any]:
    try:
        t = command.get("type")
        if t == "file":
            action = command.get("action")
            if action == "create_python":
                return await file_ops.handle_create_python()
            if action == "create_html":
                return await file_ops.handle_create_html()
            if action == "create_text":
                return await file_ops.handle_create_text()
            if action == "list":
                return await file_ops.handle_list_files()
            if action == "download":
                return await file_ops.handle_download(command.get("filename", ""))
            return {"ok": False, "message": "unknown file action"}
        if t == "quantum":
            # lightweight simulation
            state.psi = round(max(0.1, min(0.9, state.psi + 0.01)), 3)
            return {"ok": True, "psi": state.psi}
        if t == "ai_chat":
            return {"ok": False, "message": "AI not configured in cleaned build"}
        return {"ok": False, "message": "unknown command type"}
    except Exception as e:
        return {"ok": False, "message": str(e)}
