python -m uvicorn lucia_ultimate_quantum_integrated:app --host 127.0.0.1 --port 8002# Run pytest against the local running server (expects server at :8002)
python -m pytest -q tests\test_file_ops.py -k test_create_list_delete_cycle
python -m pytest -q tests\test_ai.py -k test_ai_status_and_chat_stub
python -m pytest -q tests\test_quantum.py -k test_quantum_stub
