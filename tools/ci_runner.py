import subprocess
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out_dir = os.path.join(root, 'patches')
os.makedirs(out_dir, exist_ok=True)
stdout_path = os.path.join(out_dir, 'pytest_stdout.log')
stderr_path = os.path.join(out_dir, 'pytest_stderr.log')
return_code_path = os.path.join(out_dir, 'pytest_return_code.txt')

# Add a Python-level warning filter to suppress Pydantic v2 deprecation warnings
# which may be emitted during test execution and pollute CI logs.
cmd = ['python', 'tools/run_pytest_wrapper.py']
print('Running:', ' '.join(cmd))
with open(stdout_path, 'w', encoding='utf-8') as so, open(stderr_path, 'w', encoding='utf-8') as se:
    p = subprocess.Popen(cmd, cwd=root, stdout=so, stderr=se)
    rc = p.wait()
    with open(return_code_path, 'w', encoding='utf-8') as rcfile:
        rcfile.write(str(rc))
print('WROTE', stdout_path, stderr_path, return_code_path)
sys.exit(rc)
