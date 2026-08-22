from pathlib import Path
import hashlib
src=Path('.github/r22592/c73_mlx_static_runner.py')
out=Path('.github/r22592/c73_mlx_static_runner_retry.py')
s=src.read_text()
old=""" for pat in [r'r=8',r'alpha=20',r'dropout=0',r'Layers:\\s*32',r'Training iters:\\s*600']:\n  if not re.search(pat,readme,re.I):raise RuntimeError('TRAINING_FACT_DRIFT_'+pat)\n"""
new=""" if hashlib.sha256(readme.encode()).hexdigest()!='fd848f7ca0028c8e66660233204dafc313c45b4f3f217af0b5cf23381b819dc6':raise RuntimeError('IMMUTABLE_README_HASH_DRIFT')\n if hashlib.sha256(cfg_txt.encode()).hexdigest()!='b9391a6e91e636f23e17f93b886cb69e522a0747e5889e2667bc34eefce2c5b9':raise RuntimeError('IMMUTABLE_ADAPTER_CONFIG_HASH_DRIFT')\n"""
assert s.count(old)==1, s.count(old)
s2=s.replace(old,new)
assert s2.count(old)==0
out.write_text(s2)
print({'PASS':True,'replacements':1,'scientific_logic_changes':0,'source_logic_changes':0,'patched_sha256':hashlib.sha256(s2.encode()).hexdigest()})
