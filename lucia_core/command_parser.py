from typing import Dict, Any


def parse_enhanced_commands(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    if any(k in t for k in ["파이썬", "python", ".py"]):
        return {"type": "file", "action": "create_python"}
    if any(k in t for k in ["html", ".html"]):
        return {"type": "file", "action": "create_html"}
    if any(k in t for k in ["텍스트", ".txt"]):
        return {"type": "file", "action": "create_text"}
    if any(k in t for k in ["목록", "리스트"]):
        return {"type": "file", "action": "list"}
    if "다운로드" in t or "download" in t:
        # try to extract filename
        parts = t.split()
        for p in parts:
            if "." in p:
                return {"type": "file", "action": "download", "filename": p}
        return {"type": "file", "action": "list"}
    if "양자" in t or "위협" in t:
        return {"type": "quantum"}
    return {"type": "ai_chat", "text": text}
