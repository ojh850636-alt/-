from __future__ import annotations
import hashlib,json,os,shutil,ssl,time,urllib.request,zipfile
from pathlib import Path

OWNER='ojh850636-alt';REPO='-';SOURCE_RUN=31484392389;RUNTIME_RUN=31491939184
TOKEN=os.environ['GITHUB_TOKEN'];CTX=ssl.create_default_context()
HEAD={'Authorization':f'Bearer {TOKEN}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'LUCIA-R22542-C39'}

def get_json(url):
    req=urllib.request.Request(url,headers=HEAD)
    with urllib.request.urlopen(req,context=CTX,timeout=60) as r:return json.load(r)

def artifacts(run):
    u=f'https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{run}/artifacts?per_page=100';d=get_json(u);return {x['name']:x for x in d['artifacts'] if not x.get('expired')}

def download(a,dst):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    want=(a.get('digest') or '').removeprefix('sha256:')
    for attempt in range(1,4):
        tmp=dst.with_suffix(dst.suffix+'.part');tmp.unlink(missing_ok=True);h=hashlib.sha256()
        try:
            req=urllib.request.Request(a['archive_download_url'],headers=HEAD)
            with urllib.request.urlopen(req,context=CTX,timeout=180) as r, tmp.open('wb') as f:
                while True:
                    b=r.read(8*1024*1024)
                    if not b:break
                    f.write(b);h.update(b)
            got=h.hexdigest()
            if want and got!=want:raise RuntimeError(f'digest mismatch {a["name"]} {got} != {want}')
            tmp.replace(dst);return got
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt==3:raise
            time.sleep(2**attempt)

def extract(a,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);z=out.parent/(a['name']+'.zip');got=download(a,z)
    with zipfile.ZipFile(z) as q:q.extractall(out)
    z.unlink();return got

def main():
    s=artifacts(SOURCE_RUN);r=artifacts(RUNTIME_RUN)
    expected=['r22542-c39-escrow-adapter','r22542-c39-escrow-meta']+[f'r22542-c39-base-{i:02d}' for i in range(16)]
    miss=[x for x in expected if x not in s];assert not miss,miss
    assert 'r22542-c39-runtime-wheelhouse' in r
    shutil.rmtree('recovered',ignore_errors=True);shutil.rmtree('runtime_escrow',ignore_errors=True)
    receipt={'schema':'LUCIA_AA_R22542_C39_EPHEMERAL_ESCROW_STREAM_RECEIPT_V1','source_run':SOURCE_RUN,'runtime_run':RUNTIME_RUN,'artifacts':[]}
    receipt['artifacts'].append({'name':expected[0],'archive_sha256':extract(s[expected[0]],'recovered/adapter')})
    receipt['artifacts'].append({'name':expected[1],'archive_sha256':extract(s[expected[1]],'recovered/meta')})
    for i in range(16):
        n=f'r22542-c39-base-{i:02d}';receipt['artifacts'].append({'name':n,'archive_sha256':extract(s[n],f'recovered/base_parts/{i:02d}')})
    n='r22542-c39-runtime-wheelhouse';receipt['artifacts'].append({'name':n,'archive_sha256':extract(r[n],'runtime_escrow')})
    Path('escrow_stream_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'artifact_count':len(receipt['artifacts']),'source_run':SOURCE_RUN,'runtime_run':RUNTIME_RUN},sort_keys=True))
if __name__=='__main__':main()
