from pathlib import Path
import hashlib
src=Path('.github/r22591/c72_email_runner.py')
out=Path('.github/r22591/c72_email_runner_retry.py')
s=src.read_text(encoding='utf-8')
replacements={
'0d8c6755ef379c4f819b1bbdb68f49304a314b2dd16689c92a877a7fb9e08b2d':'d0492d8316fa5cbf1963f1f4ffb46169c8d260a1883ce6444e83618e30446715',
'88e329f2f51652d9d9395520f69ba1fe746af527561123d236c93de3bdbb6fe3':'ac896f4962801feca4c9c1dbbe01ebe8f9e0ffeae63bf2ab84766d0a9fe5231f',
'011072f91afff53c964d6967eb34234a01046078fecc7a536528db20d07572b1':'b57cc9e4fdb58ea847fc45c79ad2fdc1f5b10b14b943b5449d17a66ccf827eba',
}
for old,new in replacements.items():
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
out.write_text(s,encoding='utf-8')
b=out.read_bytes()
assert hashlib.sha256(b).hexdigest()=='4000643196ce3b5f77fef6a10f353ed85c052b7ad1ed49726eb6926f3fd75cfb'
print({'PASS':True,'patched_sha256':hashlib.sha256(b).hexdigest(),'scientific_logic_changes':0,'projection_hash_changes':3})
