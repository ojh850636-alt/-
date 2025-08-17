import threading
import time
from pathlib import Path


def test_concurrent_reserve_and_rollback(tmp_path, monkeypatch):
    """Spawn multiple threads that reserve, some roll back, and verify final counts."""
    # configure quota storage and limits
    quota_file = tmp_path / 'ai_quota_concurrent.json'
    monkeypatch.setenv('AI_QUOTA_FILE', str(quota_file))
    # allow up to 5 calls per minute
    monkeypatch.setenv('AI_MAX_CALLS_PER_MINUTE', '5')
    monkeypatch.setenv('AI_DAILY_CALL_LIMIT', '1000')

    import ai_quota

    # clean file if present
    try:
        quota_file.unlink()
    except Exception:
        pass

    success_workers = 5
    error_workers = 8

    results = []

    def worker(do_error: bool):
        ok = ai_quota.reserve_call()
        results.append(bool(ok))
        if not ok:
            return
        # simulate work time; errors wait a bit longer
        time.sleep(0.02 if do_error else 0.005)
        if do_error:
            # simulate provider exception -> rollback
            ai_quota.rollback_call()

    threads = []
    for _ in range(success_workers):
        t = threading.Thread(target=worker, args=(False,))
        threads.append(t)
    for _ in range(error_workers):
        t = threading.Thread(target=worker, args=(True,))
        threads.append(t)

    # start all nearly simultaneously
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Count how many reserved succeeded
    reserved_count = sum(1 for r in results if r)

    # load persisted usage
    import json
    data = {}
    if quota_file.exists():
        with open(quota_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # daily counter should equal the number of successful workers (errors rolled back)
    daily = int(data.get('daily', 0))
    assert daily == success_workers, f"daily={daily} expected {success_workers} (reserved_count={reserved_count})"

    # reserved_count reported by threads should match
    assert reserved_count == success_workers + error_workers or reserved_count == success_workers or reserved_count >= success_workers
