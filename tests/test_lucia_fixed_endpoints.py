import shutil
from fastapi.testclient import TestClient

from lucia_ultimate_quantum_integrated_fixed import app, DOWNLOADS

client = TestClient(app)


def setup_module(module):
    # ensure clean downloads
    if DOWNLOADS.exists():
        shutil.rmtree(DOWNLOADS)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)


def teardown_module(module):
    # cleanup after tests
    if DOWNLOADS.exists():
        shutil.rmtree(DOWNLOADS)


def test_create_python_file_via_command():
    resp = client.post("/command", json={"text": "파이썬 파일 생성"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True
    filename = data.get("filename")
    assert filename
    filepath = DOWNLOADS / filename
    assert filepath.exists()
    content = filepath.read_text(encoding="utf-8")
    assert "Hello from Lucia" in content


def test_list_and_download():
    # ensure at least one file exists
    files = list(DOWNLOADS.iterdir())
    assert len(files) >= 1
    name = files[0].name

    # test list via command
    resp = client.post("/command", json={"text": "목록"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True
    assert isinstance(data.get("files"), list)
    assert any(f["name"] == name for f in data.get("files"))

    # test download endpoint
    resp2 = client.get(f"/downloads/{name}")
    assert resp2.status_code == 200
    # Content should be present
    assert resp2.content
