"""Standalone concurrency smoke script for ai_quota reserve/rollback behavior.

Usage: python tools\run_concurrency_smoke.py
"""
import threading
import time
import os
from pathlib import Path


def run():
    repo_root = Path(__file__).parents[1]
    quota_file = (repo_root / 'ai_concurrency_smoke.json').absolute()
    os.environ['AI_QUOTA_FILE'] = str(quota_file)
    os.environ['AI_MAX_CALLS_PER_MINUTE'] = '5'
    os.environ['AI_DAILY_CALL_LIMIT'] = '1000'

    try:
        quota_file.unlink()
    except Exception:
        pass

    import ai_quota

    success_workers = 5
    error_workers = 8

    results = []

    def worker(do_error: bool):
        ok = ai_quota.reserve_call()
        results.append(bool(ok))
        if not ok:
            return
        time.sleep(0.02 if do_error else 0.005)
        if do_error:
            ai_quota.rollback_call()

    threads = []
    for _ in range(success_workers):
        t = threading.Thread(target=worker, args=(False,))
        threads.append(t)
    for _ in range(error_workers):
        t = threading.Thread(target=worker, args=(True,))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        reserved_count = sum(1 for r in results if r)
        print('reserved_count:', reserved_count)

        import json
        data = {}
        if quota_file.exists():
            with open(quota_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print('persisted daily:', data.get('daily'))
        else:
            print('no quota file written')

        # write a small JSON report for CI/local inspection
        report = {
            'reserved_count': reserved_count,
            'persisted_daily': data.get('daily') if quota_file.exists() else None,
            'quota_file': str(quota_file),
            'error': None
        }
    except Exception as e:
        report = {'reserved_count': None, 'persisted_daily': None, 'quota_file': str(quota_file), 'error': repr(e)}

    out = repo_root / 'tools' / 'run_concurrency_smoke.out.json'
    try:
        with open(out, 'w', encoding='utf-8') as rf:
            import json
            json.dump(report, rf, ensure_ascii=False, indent=2)
        print('wrote report to', out)
    except Exception as e:
        print('failed to write report:', e)


if __name__ == '__main__':
    run()
