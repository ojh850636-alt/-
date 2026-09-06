# PowerShell helper to run the local app
$env:LUCIA_HOST = "127.0.0.1"
$env:LUCIA_PORT = "8000"
python -m pip install uvicorn -q
python run_uvicorn.py
