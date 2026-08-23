from __future__ import annotations
import argparse, gc, hashlib, json, math, os, re, shutil, tempfile, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np

EXPECTED_ADAPTER_SHA='551975ad96235ee5a125df3bef460111d46ed51e95061fd3c37398be4cf5b53a'
EXPECTED_ADAPTER_SIZE=18485794
EXPECTED_SUITE_SHA='beea0021151e66c81163d9098fd6eb859125a8cbebdffc94082d312af649e705'
BASE_REPO='Qwen/Qwen2.5-1.5B-Instruct'
BASE_REV='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
SCALE=20.0
RANK=8
EXPECTED_MODULES=('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')
LABELS=('VALID','HALLUCINATED','UNCERTAIN')
LAYER_RE=re.compile(r'(?:^|\.)layers\.(\d+)\.(.+)\.lora_([ab])$')

def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def sha256_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()
def writej(p:Path,o): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n",encoding='utf-8')
def finite_stats(a:np.ndarray):
 x=np.asarray(a,dtype=np.float64)
 return {'shape':list(a.shape),'dtype':str(a.dtype),'count':int(a.size),'finite':bool(np.isfinite(x).all()),'zero_fraction':float(np.mean(x==0)),'mean':float(x.mean()),'std':float(x.std()),'fro_norm':float(np.linalg.norm(x)),'max_abs':float(np.max(np.abs(x)))}
def pair_key(name:str):
 m=LAYER_RE.search(name)
 if not m: return None
 return int(m.group(1)),m.group(2),m.group(3)
def spectral_pair(A,B):
 A=np.asarray(A,dtype=np.float64);B=np.asarray(B,dtype=np.float64)
 qa,ra=np.linalg.qr(A,mode='reduced');qb,rb=np.linalg.qr(B.T,mode='reduced')
 s=np.linalg.svd(ra@rb.T,compute_uv=False)
 s=np.asarray(s,dtype=np.float64)*SCALE
 s2=s*s; total=float(s2.sum()); top=float(s[0]) if len(s) else 0.0
 p=s2/total if total>0 else np.zeros_like(s2)
 ent=float(math.exp(-sum(float(q)*math.log(float(q)) for q in p if q>0))) if total>0 else 0.0
 return {'singular_values':[float(v) for v in s],'effective_rank_1e6':int(np.sum(s>top*1e-6)) if top>0 else 0,'delta_fro':float(math.sqrt(total)),'delta_spectral':top,'stable_rank':float(total/(top*top)) if top>0 else 0.0,'entropy_rank':ent,'top_component_energy_fraction':float(p[0]) if len(p) else 0.0}
def principal_similarity(X,Y):
 X=np.asarray(X,dtype=np.float64);Y=np.asarray(Y,dtype=np.float64)
 qx,_=np.linalg.qr(X,mode='reduced');qy,_=np.linalg.qr(Y,mode='reduced')
 s=np.linalg.svd(qx.T@qy,compute_uv=False)
 return float(np.mean(s*s))
def make_variant(weights,kind,seed=42,dose=1.0,ablate_layers=None,ablate_module=None,ablate_rank=None):
 rng=np.random.default_rng(seed); out={k:np.array(v,copy=True) for k,v in weights.items()}
 if kind=='random':
  for k,v in out.items():
   x=rng.standard_normal(v.shape)
   n=float(np.linalg.norm(x)); target=float(np.linalg.norm(np.asarray(weights[k],dtype=np.float64)))
   x=x*(target/n if n else 0.0);out[k]=x.astype(v.dtype)
 elif kind=='shuffle':
  parsed=[(k,pair_key(k)) for k in out]; layers=sorted({p[0] for _,p in parsed if p})
  perm=layers.copy();rng.shuffle(perm); mp=dict(zip(layers,perm))
  src=dict(weights)
  idx={(p[0],p[1],p[2]):k for k,p in parsed if p}
  for k,p in parsed:
   if p:
    sk=idx[(mp[p[0]],p[1],p[2])];out[k]=np.array(src[sk],copy=True)
 elif kind=='dose':
  for k,v in out.items():
   if pair_key(k) and pair_key(k)[2]=='b': out[k]=(np.asarray(v,dtype=np.float64)*dose).astype(v.dtype)
 elif kind=='ablate':
  for k,v in out.items():
   p=pair_key(k)
   if not p: continue
   layer,module,factor=p
   dozero=(ablate_layers is not None and layer in ablate_layers) or (ablate_module is not None and module==ablate_module)
   if dozero: out[k]=np.zeros_like(v)
   if ablate_rank is not None:
    x=np.array(out[k],copy=True)
    if factor=='a' and x.ndim==2 and ablate_rank<x.shape[1]: x[:,ablate_rank]=0
    if factor=='b' and x.ndim==2 and ablate_rank<x.shape[0]: x[ablate_rank,:]=0
    out[k]=x
 return out

def static_audit(adapter:Path, outdir:Path):
 from safetensors.numpy import load_file
 weights=load_file(str(adapter))
 tensor_stats={k:finite_stats(v) for k,v in weights.items()}
 parsed={};unexpected=[]
 for k in weights:
  p=pair_key(k)
  if not p: unexpected.append(k);continue
  layer,module,factor=p;parsed.setdefault((layer,module),{})[factor]=k
 pair_summaries=[]; failures=[]
 for (layer,module),ab in sorted(parsed.items()):
  if set(ab)!={'a','b'}: failures.append(f'MISSING_PAIR:{layer}:{module}');continue
  A=weights[ab['a']];B=weights[ab['b']]
  if A.ndim!=2 or B.ndim!=2 or A.shape[1]!=RANK or B.shape[0]!=RANK or A.shape[1]!=B.shape[0]: failures.append(f'SHAPE:{layer}:{module}:{A.shape}:{B.shape}');continue
  sp=spectral_pair(A,B);sp.update({'layer':layer,'module':module,'a_key':ab['a'],'b_key':ab['b'],'a_shape':list(A.shape),'b_shape':list(B.shape),'a_norm':float(np.linalg.norm(A)),'b_norm':float(np.linalg.norm(B))})
  if sp['effective_rank_1e6']==0: failures.append(f'ZERO_OPERATOR:{layer}:{module}')
  pair_summaries.append(sp)
 layers=sorted({x['layer'] for x in pair_summaries});modules=sorted({x['module'] for x in pair_summaries})
 if len(weights)!=196: failures.append(f'TENSOR_COUNT:{len(weights)}')
 if len(pair_summaries)!=98: failures.append(f'PAIR_COUNT:{len(pair_summaries)}')
 if len(layers)!=14: failures.append(f'LAYER_COUNT:{len(layers)}')
 if tuple(modules)!=tuple(sorted(EXPECTED_MODULES)): failures.append('MODULE_SET')
 if unexpected: failures.append(f'UNEXPECTED_KEYS:{len(unexpected)}')
 if not all(v['finite'] for v in tensor_stats.values()): failures.append('NONFINITE')
 layer_energy={str(l):float(math.sqrt(sum(x['delta_fro']**2 for x in pair_summaries if x['layer']==l))) for l in layers}
 module_energy={m:float(math.sqrt(sum(x['delta_fro']**2 for x in pair_summaries if x['module']==m))) for m in modules}
 rank_energy=[]
 for r in range(RANK):
  e=0.0
  for x in pair_summaries:
   A=np.asarray(weights[x['a_key']],dtype=np.float64);B=np.asarray(weights[x['b_key']],dtype=np.float64)
   e+=(SCALE*np.linalg.norm(A[:,r])*np.linalg.norm(B[r,:]))**2
  rank_energy.append(float(math.sqrt(e)))
 coherence={}
 for m in modules:
  xs=[x for x in pair_summaries if x['module']==m];inp=[];outp=[]
  for i in range(len(xs)):
   for j in range(i+1,len(xs)):
    inp.append(principal_similarity(weights[xs[i]['a_key']],weights[xs[j]['a_key']]))
    outp.append(principal_similarity(weights[xs[i]['b_key']].T,weights[xs[j]['b_key']].T))
  coherence[m]={'input_subspace_mean_cos2':float(np.mean(inp)) if inp else None,'output_subspace_mean_cos2':float(np.mean(outp)) if outp else None,'pair_count':len(inp)}
 learned_er=float(np.mean([x['entropy_rank'] for x in pair_summaries])); random_er=[]
 for seed in (42,224,1337,9001,17001):
  rw=make_variant(weights,'random',seed=seed); vals=[]
  for x in pair_summaries: vals.append(spectral_pair(rw[x['a_key']],rw[x['b_key']])['entropy_rank'])
  random_er.append(float(np.mean(vals)))
 audit={'schema':'R22601_C76K_STATIC_OPERATOR_ATLAS_V1','source_identity':{'sha256':sha256_file(adapter),'size':adapter.stat().st_size},'tensor_count':len(weights),'operator_pair_count':len(pair_summaries),'layers':layers,'modules':modules,'unexpected_keys':unexpected,'failures':failures,'static_viability_pass':not failures,'tensor_stats':tensor_stats,'pairs':pair_summaries,'layer_energy':layer_energy,'module_energy':module_energy,'rank_component_energy_proxy':rank_energy,'subspace_coherence':coherence,'structural_random_control':{'learned_mean_entropy_rank':learned_er,'random_seed_mean_entropy_rank':random_er,'learned_minus_random_mean':learned_er-float(np.mean(random_er))}}
 writej(outdir/'R22601_C76_K_STATIC_OPERATOR_ATLAS.json',audit)
 total=sum(v*v for v in module_energy.values()); modshare={k:(v*v/total if total else 0.0) for k,v in module_energy.items()}
 ltotal=sum(v*v for v in layer_energy.values()); lshare={k:(v*v/ltotal if ltotal else 0.0) for k,v in layer_energy.items()}
 genome={'schema':'R22601_C76K_RAWFREE_OPERATOR_GENOME_V1','claim_boundary':'Static learned-operator material only; not task capability evidence.','adapted_layers':layers,'module_families':modules,'rank':RANK,'scale':SCALE,'operator_pairs':len(pair_summaries),'module_energy_share':modshare,'layer_energy_share':lshare,'rank_component_energy_proxy_normalized':([x/sum(rank_energy) for x in rank_energy] if sum(rank_energy)>0 else [0]*RANK),'subspace_coherence':coherence,'static_viability_pass':not failures}
 writej(outdir/'R22601_C76_K_RAWFREE_OPERATOR_GENOME.json',genome)
 return weights,audit

def load_cases(path:Path):
 raw=path.read_bytes()
 if sha256_bytes(raw)!=EXPECTED_SUITE_SHA: raise RuntimeError('PRESEAL_SUITE_SHA_MISMATCH')
 cases=[json.loads(x) for x in raw.splitlines() if x.strip()]
 if len(cases)!=128: raise RuntimeError('PRESEAL_CASE_COUNT')
 splits={0:[],1:[],2:[],3:[]}
 for c in cases:
  v=int(c['case_id'].rsplit('-',1)[-1]);splits[v].append(c)
 assert all(len(v)==32 for v in splits.values())
 return splits

def prompt_for(c):
 return ('You are a citation-evidence auditor. Compare OBSERVED metadata against CANONICAL evidence. Choose exactly one class for whether the observed citation is valid, contradicted/hallucinated, or too incomplete to decide. Mapping: A = VALID; B = HALLUCINATED; C = UNCERTAIN. Return only A, B, or C.\n'+f"INPUT_TYPE={c['input_type']}\nCANONICAL={json.dumps(c['canonical'],sort_keys=True,separators=(',',':'))}\nOBSERVED={json.dumps(c['observed'],sort_keys=True,separators=(',',':'))}\nANSWER=")
def label_letter(c): return {'VALID':'A','HALLUCINATED':'B','UNCERTAIN':'C'}[c['expected']['label']]
def find_single_tokens(tokenizer):
 options=[(' A',' B',' C'),('A','B','C'),(' 1',' 2',' 3'),('1','2','3')]
 for trio in options:
  ids=[tokenizer.encode(x,add_special_tokens=False) for x in trio]
  if all(len(x)==1 for x in ids) and len({x[0] for x in ids})==3: return trio,[x[0] for x in ids]
 raise RuntimeError('NO_SINGLE_TOKEN_CLASS_CODES')
def score_cases(model,tokenizer,cases):
 import mlx.core as mx
 from mlx_lm.generate import generate_step
 symbols,ids=find_single_tokens(tokenizer); code_to_idx={'A':0,'B':1,'C':2};rows=[]
 for c in cases:
  toks=tokenizer.encode(prompt_for(c),add_special_tokens=False);gen=generate_step(mx.array(toks),model,max_tokens=1);_tok,logp=next(gen);mx.eval(logp)
  vals=[float(logp[i].item()) for i in ids];ci=code_to_idx[label_letter(c)];best_wrong=max(v for j,v in enumerate(vals) if j!=ci);pred='ABC'[int(np.argmax(vals))]
  rows.append({'case_id':c['case_id'],'correct':pred==label_letter(c),'margin':vals[ci]-best_wrong,'pred':pred,'target':label_letter(c)})
 return rows
def summary_rows(rows): return {'n':len(rows),'accuracy':float(np.mean([r['correct'] for r in rows])),'mean_margin':float(np.mean([r['margin'] for r in rows])),'median_margin':float(np.median([r['margin'] for r in rows])),'positive_margin_fraction':float(np.mean([r['margin']>0 for r in rows]))}
def paired_bootstrap_delta(a,b,seed=20260823,nboot=2000):
 da=np.array([x['margin'] for x in a]);db=np.array([x['margin'] for x in b]);d=da-db;rng=np.random.default_rng(seed);means=[]
 for _ in range(nboot): means.append(float(np.mean(d[rng.integers(0,len(d),len(d))])))
 return {'mean':float(np.mean(d)),'ci95':[float(np.quantile(means,.025)),float(np.quantile(means,.975))]}
def save_variant(path:Path,weights,config_src:Path):
 from safetensors.numpy import save_file
 path.mkdir(parents=True,exist_ok=True);shutil.copy2(config_src,path/'adapter_config.json');save_file(weights,str(path/'adapters.safetensors'))
def clear_mlx():
 try:
  import mlx.core as mx;mx.clear_cache()
 except Exception: pass
 gc.collect()
def behavior_audit(adapter:Path,config:Path,cases_path:Path,outdir:Path,work:Path):
 from mlx_lm import load
 splits=load_cases(cases_path)
 from safetensors.numpy import load_file
 src=load_file(str(adapter))
 variants={'FULL':None,'RANDOM42':make_variant(src,'random',seed=42),'RANDOM224':make_variant(src,'random',seed=224),'RANDOM1337':make_variant(src,'random',seed=1337),'SHUFFLE42':make_variant(src,'shuffle',seed=42),'DOSE025':make_variant(src,'dose',dose=.25),'DOSE050':make_variant(src,'dose',dose=.5),'DOSE150':make_variant(src,'dose',dose=1.5)}
 vdirs={}
 for name,w in variants.items():
  if w is None: vdirs[name]=adapter.parent
  else:
   d=work/'variants'/name;save_variant(d,w,config);vdirs[name]=d
 results={'schema':'R22601_C76K_SOURCEFREE_BEHAVIOR_V1','claim_boundary':'Source-free synthetic citation-metadata classification surrogate. No public Hallmark score is counted as LAA capability. Positive claim requires locked control separation.','splits':{'DISCOVERY':32,'CONFIRMATION':32,'CAUSAL':32,'STRESS':32},'conditions':{},'control_separated_gain':False,'confirmation_pass':False,'causal':{'run':False}}
 model,tok=load(BASE_REPO,revision=BASE_REV);base_disc=score_cases(model,tok,splits[0]);results['conditions']['BASE_DISCOVERY']={'summary':summary_rows(base_disc),'rows':base_disc};del model;clear_mlx()
 cond_rows={}
 for name in ('FULL','RANDOM42','RANDOM224','RANDOM1337','SHUFFLE42','DOSE025','DOSE050','DOSE150'):
  model,tok=load(BASE_REPO,revision=BASE_REV,adapter_path=str(vdirs[name]));rr=score_cases(model,tok,splits[0]);cond_rows[name]=rr;results['conditions'][name+'_DISCOVERY']={'summary':summary_rows(rr),'rows':rr};del model;clear_mlx()
 full=cond_rows['FULL'];dbase=paired_bootstrap_delta(full,base_disc);results['discovery_full_minus_base_margin']=dbase
 ctrl_deltas={k:paired_bootstrap_delta(full,cond_rows[k]) for k in ('RANDOM42','RANDOM224','RANDOM1337','SHUFFLE42')};results['discovery_full_minus_controls_margin']=ctrl_deltas
 positive=(dbase['mean']>0 and dbase['ci95'][0]>0 and all(x['mean']>0 and x['ci95'][0]>0 for x in ctrl_deltas.values()));results['control_separated_gain']=bool(positive)
 if positive:
  confirm={};model,tok=load(BASE_REPO,revision=BASE_REV);confirm['BASE']=score_cases(model,tok,splits[1]);del model;clear_mlx()
  for name in ('FULL','RANDOM42','SHUFFLE42'):
   model,tok=load(BASE_REPO,revision=BASE_REV,adapter_path=str(vdirs[name]));confirm[name]=score_cases(model,tok,splits[1]);del model;clear_mlx()
  results['confirmation']={k:summary_rows(v) for k,v in confirm.items()};cbase=paired_bootstrap_delta(confirm['FULL'],confirm['BASE']);cr=paired_bootstrap_delta(confirm['FULL'],confirm['RANDOM42']);cs=paired_bootstrap_delta(confirm['FULL'],confirm['SHUFFLE42']);results['confirmation_deltas']={'base':cbase,'random42':cr,'shuffle42':cs};cpass=all(x['mean']>0 and x['ci95'][0]>0 for x in (cbase,cr,cs));results['confirmation_pass']=bool(cpass)
  if cpass:
   results['causal']['run']=True;causal_cases=splits[2];layers=sorted({pair_key(k)[0] for k in src if pair_key(k)});bands=[layers[i:i+2] for i in range(0,len(layers),2)];model,tok=load(BASE_REPO,revision=BASE_REV,adapter_path=str(vdirs['FULL']));full_c=score_cases(model,tok,causal_cases);del model;clear_mlx();results['causal']['full']=summary_rows(full_c);ablations={};specs=[]
   for band in bands: specs.append(('LAYER_BAND_'+('_'.join(map(str,band))),dict(ablate_layers=set(band))))
   for m in EXPECTED_MODULES: specs.append(('MODULE_'+m.replace('.','_'),dict(ablate_module=m)))
   for r in range(RANK): specs.append((f'RANK_{r}',dict(ablate_rank=r)))
   for name,kw in specs:
    d=work/'variants'/'ABLATIONS'/name;save_variant(d,make_variant(src,'ablate',**kw),config);model,tok=load(BASE_REPO,revision=BASE_REV,adapter_path=str(d));rr=score_cases(model,tok,causal_cases);del model;clear_mlx();delta=paired_bootstrap_delta(full_c,rr,seed=20260823+len(ablations));ablations[name]={'summary':summary_rows(rr),'full_minus_ablation_margin':delta}
   results['causal']['ablations']=ablations;necessities=[(k,v['full_minus_ablation_margin']['mean']) for k,v in ablations.items() if v['full_minus_ablation_margin']['mean']>0 and v['full_minus_ablation_margin']['ci95'][0]>0];results['causal']['necessity_positive']=sorted(necessities,key=lambda x:x[1],reverse=True)
 writej(outdir/'R22601_C76_K_SOURCEFREE_BEHAVIOR.json',results)
 compact={'schema':'R22601_C76K_BEHAVIOR_COMPACT_V1','control_separated_gain':results['control_separated_gain'],'confirmation_pass':results['confirmation_pass'],'discovery_full_minus_base_margin':results.get('discovery_full_minus_base_margin'),'discovery_full_minus_controls_margin':results.get('discovery_full_minus_controls_margin'),'causal_run':results['causal']['run'],'claim_boundary':results['claim_boundary']}
 if results['causal']['run']: compact['necessity_positive']=results['causal'].get('necessity_positive',[])
 writej(outdir/'R22601_C76_K_BEHAVIOR_COMPACT.json',compact);return results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);ap.add_argument('--config',required=True);ap.add_argument('--cases',required=True);ap.add_argument('--out',required=True);ap.add_argument('--work',required=True);a=ap.parse_args();adapter=Path(a.adapter);config=Path(a.config);cases=Path(a.cases);out=Path(a.out);work=Path(a.work);out.mkdir(parents=True,exist_ok=True)
 identity={'expected_sha256':EXPECTED_ADAPTER_SHA,'observed_sha256':sha256_file(adapter),'expected_size':EXPECTED_ADAPTER_SIZE,'observed_size':adapter.stat().st_size};identity['pass']=identity['expected_sha256']==identity['observed_sha256'] and identity['expected_size']==identity['observed_size'];writej(out/'R22601_C76_K_SOURCE_IDENTITY.json',identity)
 if not identity['pass']: writej(out/'R22601_C76_K_RUN_SUMMARY.json',{'status':'TERMINAL_IDENTITY_MISMATCH','PASS':False});return 10
 weights,static=static_audit(adapter,out)
 if not static['static_viability_pass']: writej(out/'R22601_C76_K_RUN_SUMMARY.json',{'status':'TERMINAL_STATIC_VIABILITY_FAIL','PASS':False,'base_bytes':0});return 20
 behavior_status='NOT_RUN';behavior=None
 try:
  behavior=behavior_audit(adapter,config,cases,out,work);behavior_status='PASS_EXECUTED'
 except Exception as e:
  behavior_status='ERROR_AFTER_SOURCE_CONSUMED_NO_RETRY';writej(out/'R22601_C76_K_BEHAVIOR_ERROR.json',{'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
 summary={'schema':'R22601_C76K_RUN_SUMMARY_V1','PASS':True,'status':'STATIC_PASS_BEHAVIOR_'+behavior_status,'source_consumed':True,'adapter_bytes':adapter.stat().st_size,'base_runtime_attempted':True,'behavior_status':behavior_status,'static_operator_pairs':static['operator_pair_count'],'control_separated_gain':bool(behavior and behavior.get('control_separated_gain')),'confirmation_pass':bool(behavior and behavior.get('confirmation_pass')),'causal_run':bool(behavior and behavior.get('causal',{}).get('run'))};writej(out/'R22601_C76_K_RUN_SUMMARY.json',summary);return 0
if __name__=='__main__': raise SystemExit(main())
