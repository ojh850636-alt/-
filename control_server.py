import os
import subprocess
import time
import sys

PORT = int(os.environ.get("SERVER_PORT", "8002"))
PY = sys.executable


def pid_on_port(port):
    try:
        out = subprocess.check_output(["netstat", "-ano"]).decode(errors="ignore")
    except Exception:
        return None
    for line in out.splitlines():
        if f":{port} " in line or f":{port}\r" in line:
            parts = line.split()
            if parts:
                pid = parts[-1]
                if pid.isdigit():
                    return int(pid)
    return None


if __name__ == "__main__":
    pid = pid_on_port(PORT)
    if pid:
        print(f"Killing PID {pid}")
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=True)
        except subprocess.CalledProcessError:
            print("Failed to kill PID", pid)
    else:
        print("No process on port", PORT)

    # Start server via start_with_env.ps1
    print("Starting server using start_with_env.ps1")
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "start_with_env.ps1",
        ]
    )

    time.sleep(1)

    try:
        import requests

        h = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=3).json()
        print("health", h)
    except Exception as e:
        print("health-failed", e)

    try:
        import requests

        a = requests.get(f"http://127.0.0.1:{PORT}/ai/status", timeout=3).json()
        print("ai-status", a)
    except Exception as e:
        print("ai-status-failed", e)

    print("Running pytest")
    subprocess.run([PY, "-m", "pytest", "-q"])
