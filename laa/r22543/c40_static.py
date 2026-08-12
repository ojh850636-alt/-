from __future__ import annotations
import argparse,hashlib,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
from safetensors import safe_open

ROLES=['storage','retrieval_plan','consolidation']
PROJS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

def files(root):return {r:Path(root)/r/'adapter_model.safetensors' for r in ROLES}
def pair_keys(path):
    with safe_open(str(path),framework='np') as f:ks=list(f.keys())
    pairs={}
    for k in ks:
        if '.lora_A.' in k:
            base=k.replace('.lora_A.weight','').replace('.lora_A.default.weight','')
            bk=k.replace('.lora_A.','.lora_B.')
            if bk in ks:pairs[base]=(k,bk)
    return pairs

def meta(base):
    m=re.search(r'layers\.(\d+)\.',base);layer=int(m.group(1)) if m else -1
    proj=next((p for p in PROJS if p in base),None)
    return layer,proj

def load_delta(path,ak,bk,scale=2.0):
    with safe_open(str(path),framework='np') as f:a=f.get_tensor(ak).astype(np.float32);b=f.get_tensor(bk).astype(np.float32)
    return (b@a)*scale,a,b

def cosine(x,y):
    nx=float(np.linalg.norm(x));ny=float(np.linalg.norm(y));return float(np.sum(x*y)/(nx*ny)) if nx and ny else 0.0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();fs=files(a.root)
    allpairs={r:pair_keys(p) for r,p in fs.items()};common=set.intersection(*(set(x) for x in allpairs.values()));per_role={};module_rows=[]
    for r,p in fs.items():
      stats=[];zero=nonfinite=0;ranks=[]
      for base,(ak,bk) in sorted(allpairs[r].items()):
        d,A,B=load_delta(p,ak,bk);rank=int(np.linalg.matrix_rank(A));ranks.append(rank);z=bool(np.count_nonzero(A)==0 or np.count_nonzero(B)==0);nf=bool(not np.isfinite(A).all() or not np.isfinite(B).all());zero+=z;nonfinite+=nf;layer,proj=meta(base)
        stats.append({'module':base,'layer':layer,'projection':proj,'delta_fro':float(np.linalg.norm(d)),'rank_A':rank,'zero':z,'nonfinite':nf})
      per_role[r]={'pair_count':len(stats),'zero_pairs':zero,'nonfinite_pairs':nonfinite,'min_rank_A':min(ranks),'max_rank_A':max(ranks),'module_stats':stats}
    cross=[]
    for base in sorted(common):
      ds={};norms={}
      for r in ROLES:
        ak,bk=allpairs[r][base];d,_,_=load_delta(fs[r],ak,bk);ds[r]=d;norms[r]=float(np.linalg.norm(d))
      layer,proj=meta(base);cross.append({'module':base,'layer':layer,'projection':proj,'storage_retrieval_cosine':cosine(ds['storage'],ds['retrieval_plan']),'storage_consolidation_cosine':cosine(ds['storage'],ds['consolidation']),'retrieval_consolidation_cosine':cosine(ds['retrieval_plan'],ds['consolidation']),'storage_fro':norms['storage'],'retrieval_fro':norms['retrieval_plan'],'consolidation_fro':norms['consolidation']})
    agg={}
    for proj in PROJS:
      rows=[x for x in cross if x['projection']==proj]
      if rows:
        agg[proj]={k:float(np.median([x[k] for x in rows])) for k in ('storage_retrieval_cosine','storage_consolidation_cosine','retrieval_consolidation_cosine','storage_fro','retrieval_fro','consolidation_fro')}
    out={'schema':'LUCIA_AA_R22543_C40_STATIC_TRIAD_V1','roles':ROLES,'common_pair_count':len(common),'per_role':per_role,'cross_adapter_module_stats':cross,'projection_medians':agg,'interpretation_boundary':'weight energy/cosine is descriptive only; no causal credit','raw_tensors_saved':False,'delta_matrices_saved':False}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
