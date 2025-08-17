from pathlib import Path

p = Path("lucia_ultimate_quantum_integrated_fixed.py")
out = Path("patches") / "lucia_fixed_numbered.txt"
text = p.read_text(encoding="utf-8")
with open(out, "w", encoding="utf-8") as f:
    for i, line in enumerate(text.splitlines(), start=1):
        f.write(f"{i:04d}: {line}\n")
print("WROTE", out)
