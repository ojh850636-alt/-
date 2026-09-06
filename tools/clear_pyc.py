import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
removed = []
for dirpath, dirs, files in os.walk(root):
    for f in files:
        if f.endswith(".pyc"):
            p = os.path.join(dirpath, f)
            try:
                os.remove(p)
                removed.append(p)
            except Exception as e:
                print("ERR", p, e)
print("REMOVED", len(removed))
for p in removed:
    print(p)
