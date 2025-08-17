import sys
from pathlib import Path

outf = Path(__file__).parent / "import_result.txt"
p = Path(r"c:\Users\Hi\Desktop\루시아 에이전트\2")
sys.path.insert(0, str(p))
with open(outf, "w", encoding="utf8") as f:
    f.write("sys.path[0]=" + sys.path[0] + "\n")
    try:
        import lucia_ultimate_quantum_enhanced_modular as mod

        f.write("OK app=" + str(getattr(mod, "app", None)) + "\n")
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        f.write("ERROR\n")
        f.write(tb)
        f.write("\n" + repr(e) + "\n")
