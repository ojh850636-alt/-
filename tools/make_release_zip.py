import os
import zipfile

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out_dir = os.path.join(root, 'patches')
os.makedirs(out_dir, exist_ok=True)
zip_path = os.path.join(out_dir, 'lucia_fix_quick_start.zip')

exclude_dirs = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'env', '.vscode', 'node_modules'}
exclude_files = {'pytest_results.xml'}

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(root):
        # compute relative path
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == '.':
            rel_dir = ''
        # skip excluded dirs
        parts = rel_dir.split(os.sep) if rel_dir else []
        if any(p in exclude_dirs for p in parts):
            continue
        for fname in filenames:
            if fname in exclude_files:
                continue
            # skip the release zip itself
            if fname == os.path.basename(zip_path):
                continue
            abspath = os.path.join(dirpath, fname)
            arcname = os.path.join(rel_dir, fname) if rel_dir else fname
            try:
                zf.write(abspath, arcname)
            except Exception:
                # skip problematic files
                continue

print('WROTE', zip_path)
