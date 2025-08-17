from fastapi.testclient import TestClient
from lucia_ultimate_quantum_integrated_fixed import app, DOWNLOADS

client = TestClient(app)


def test_command_create_and_download(tmp_path):
    # ensure downloads dir is empty for test
    if DOWNLOADS.exists():
        for p in DOWNLOADS.iterdir():
            if p.is_file():
                p.unlink()
    else:
        DOWNLOADS.mkdir(parents=True)

    # request creation of a python file
    resp = client.post("/command", json={"text": "생성 파이썬 파일"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True
    filename = data.get("filename")
    assert filename

    # file should exist
    filepath = DOWNLOADS / filename
    assert filepath.exists()

    # download via endpoint
    dl = client.get(f"/downloads/{filename}")
    assert dl.status_code == 200
    # content-disposition should include filename
    assert filename in dl.headers.get("content-disposition", "")

    # request list
    resp2 = client.post("/command", json={"text": "목록"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2.get("ok") is True
    files = data2.get("files")
    assert any(f["name"] == filename for f in files)
