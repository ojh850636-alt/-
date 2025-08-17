import requests
import sys

try:
    r = requests.get('http://127.0.0.1:8002/health', timeout=3)
    print('status', r.status_code)
    print(r.text)
except Exception as e:
    print('error', e)
    sys.exit(1)
