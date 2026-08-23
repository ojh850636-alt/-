Lucia cleaned server (fixed)

Files created:
- lucia_ultimate_quantum_integrated_fixed.py : cleaned, runnable FastAPI server (minimal)

How to run (PowerShell):

# Create virtual env (optional)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python lucia_ultimate_quantum_integrated_fixed.py

The API endpoints:
- POST /command {"text": "파이썬 파일 만들어줘"}
- GET /downloads/{filename}

This is a safe, minimal starting point while we iteratively restore advanced features from the original file.
