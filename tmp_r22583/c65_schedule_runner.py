from __future__ import annotations
import argparse, gc, hashlib, itertools, json, math, os, random, re, shutil, struct
from pathlib import Path
from c65_schedule_generator import make_cases, metadata, feasible_start, hhmm, WORK_START, WORK_END

ADAPTER_REPO='sumitsrv/qwen3-0.6b-task-planner'
EXPECTED_BASE='unsloth/Qwen3-0.6B'
LICENSE='apache-2.0'
SEED=22583

def sha256_path(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def norm_base(x):
    if not x:return ''
    s=str(x).lower().strip().replace('-bnb-4bit','').replace('-unsloth','')
    return s

def parse_st_header(p:Path):
    with p.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8: raise SystemExit('short safetensors header')
        n=struct.unpack('<Q',raw)[0]
        if n<=2 or n>100_000_000: raise SystemExit(f'unsafe header length {n}')
        hdr=json.loads(f.read(n))
    tensors={k:v for k,v in hdr.items() if k!='__metadata__'}
    dtypes={}
    for v in tensors.values(): dtypes[v.get('dtype','?')]=dtypes.get(v.get('dtype','?'),0)+1
    return {'tensor_count':len(tensors),'dtype_counts':dtypes,'lora_key_count':sum('lora_' in k for k in tensors),'non_lora_key_count':sum('lora_' not in k for k in tensors),'header_sha256':hashlib.sha256(json.dumps(hdr,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

def discover(meta,suffix):
    xs=[x.get('rfilename','') for x in meta.get('siblings',[]) if x.get('rfilename','').endswith(suffix)]
    if len(xs)!=1: raise SystemExit(f'path discovery {suffix}: {xs}')
    return xs[0]

def verify_seal(seal_path:Path):
    seal=json.loads(seal_path.read_text()); cs=make_cases(); md=metadata(cs)
    dig=hashlib.sha256(json.dumps(md,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if seal['fingerprint']!='DAILY_SCHEDULE_CONSTRAINT_SATISFACTION_MINIMAL_DEFERRAL': raise SystemExit(21)
    if len(cs)!=seal['case_count'] or dig!=seal['cases_sha256']: raise SystemExit(22)
    return seal,cs

def capture(out:Path,escrow:Path,seal_path:Path):
    import requests
    from huggingface_hub import hf_hub_download
    out.mkdir(parents=True,exist_ok=True);escrow.mkdir(parents=True,exist_ok=True)
    seal,cases=verify_seal(seal_path)
    am=requests.get(f'https://huggingface.co/api/models/{ADAPTER_REPO}',timeout=30);am.raise_for_status();am=am.json()
    bm=requests.get(f'https://huggingface.co/api/models/{EXPECTED_BASE}',timeout=30);bm.raise_for_status();bm=bm.json()
    rev=am['sha']; base_runtime_rev=bm['sha']; card=am.get('cardData') or {}; lic=(card.get('license') or am.get('license') or '').lower(); declared=card.get('base_model')
    if isinstance(declared,list): declared=declared[0] if len(declared)==1 else declared
    if lic!=LICENSE: raise SystemExit(31)
    if norm_base(declared or EXPECTED_BASE)!=norm_base(EXPECTED_BASE): raise SystemExit(32)
    cfgpath=discover(am,'adapter_config.json'); apath=discover(am,'adapter_model.safetensors')
    dl=escrow/'_download';dl.mkdir(exist_ok=True)
    cfgsrc=Path(hf_hub_download(repo_id=ADAPTER_REPO,filename=cfgpath,revision=rev,local_dir=dl))
    cfg=json.loads(cfgsrc.read_text()); actual=cfg.get('base_model_name_or_path'); training_rev=cfg.get('revision'); mts=cfg.get('modules_to_save')
    coherent=(norm_base(actual)==norm_base(declared or EXPECTED_BASE)==norm_base(EXPECTED_BASE))
    if not coherent: raise SystemExit(32)
    if mts not in (None,[],()): raise SystemExit(34)
    asrc=Path(hf_hub_download(repo_id=ADAPTER_REPO,filename=apath,revision=rev,local_dir=dl))
    shallow=parse_st_header(asrc)
    shutil.copy2(cfgsrc,escrow/'adapter_config.json');shutil.copy2(asrc,escrow/'adapter_model.safetensors');shutil.rmtree(dl,ignore_errors=True)
    rec={'schema':'R22583_C65_CAPTURE_V1','repo':ADAPTER_REPO,'resolved_revision':rev,'license':lic,'declared_base':declared,'actual_adapter_config_base':actual,'training_base_revision':training_rev,'base_name_coherent':coherent,'training_revision_pinned':bool(training_rev),'base_runtime_repo':EXPECTED_BASE,'base_runtime_revision':base_runtime_rev,'config_path':cfgpath,'adapter_path':apath,'config_sha256':sha256_path(escrow/'adapter_config.json'),'adapter_file':{'size':(escrow/'adapter_model.safetensors').stat().st_size,'sha256':sha256_path(escrow/'adapter_model.safetensors')},'shallow_header':shallow,'modules_to_save':mts,'target_modules':sorted(cfg.get('target_modules') or []),'r':cfg.get('r'),'lora_alpha':cfg.get('lora_alpha'),'binary_ingress_count':1,'hf_adapter_redownload':0,'base_downloaded':False,'source_reported_metrics_are_evidence':False,'source_reported_performance_credit':0,'case_count':len(cases),'seal_sha256':sha256_path(seal_path),'raw_committed_to_git':False}
    (out/'C65_CAPTURE.json').write_text(json.dumps(rec,indent=2,sort_keys=True))
    print(json.dumps({k:rec[k] for k in ['resolved_revision','license','declared_base','actual_adapter_config_base','training_base_revision','base_runtime_revision','modules_to_save','target_modules','r','lora_alpha','shallow_header']},indent=2))

def static(out:Path,adapter:Path):
    import torch
    from safetensors import safe_open
    out.mkdir(parents=True,exist_ok=True); stats=[]; aux=[]; ap=adapter/'adapter_model.safetensors'
    with safe_open(str(ap),framework='pt',device='cpu') as f:
        keys=list(f.keys()); ks=set(keys)
        for k in keys:
            if '.lora_A.' in k and k.endswith('.weight'):
                bk=k.replace('.lora_A.','.lora_B.')
                if bk not in ks: continue
                A=f.get_tensor(k).float(); B=f.get_tensor(bk).float(); an=float(torch.linalg.vector_norm(A)); bn=float(torch.linalg.vector_norm(B)); prod=an*bn
                m=re.search(r'layers\.(\d+)\.',k); layer=int(m.group(1)) if m else None
                target=next((t for t in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if t in k),'other')
                stats.append({'pair_id':hashlib.sha256(k.encode()).hexdigest()[:16],'layer':layer,'target':target,'a_shape':list(A.shape),'b_shape':list(B.shape),'a_norm':an,'b_norm':bn,'norm_product':prod,'rank_a':int(torch.linalg.matrix_rank(A)),'rank_b':int(torch.linalg.matrix_rank(B)),'alive':bool(prod>0 and math.isfinite(prod))})
            elif 'lora_' not in k: aux.append({'key_hash':hashlib.sha256(k.encode()).hexdigest()[:16],'shape':list(f.get_slice(k).get_shape())})
    rec={'schema':'R22583_C65_STATIC_V1','tensor_count':len(keys),'pair_count':len(stats),'alive_pair_count':sum(x['alive'] for x in stats),'dead_pair_count':sum(not x['alive'] for x in stats),'nonfinite_count':sum(not math.isfinite(x['norm_product']) for x in stats),'target_counts':{},'ranks_a':sorted(set(x['rank_a'] for x in stats)),'ranks_b':sorted(set(x['rank_b'] for x in stats)),'auxiliary_non_lora_keys':len(aux),'pair_stats':stats}
    for x in stats: rec['target_counts'][x['target']]=rec['target_counts'].get(x['target'],0)+1
    if rec['pair_count']==0 or rec['alive_pair_count']!=rec['pair_count'] or rec['nonfinite_count']!=0 or rec['auxiliary_non_lora_keys']!=0: raise SystemExit(33)
    (out/'C65_STATIC.json').write_text(json.dumps(rec,indent=2,sort_keys=True))
    print(json.dumps({k:rec[k] for k in ['tensor_count','pair_count','alive_pair_count','dead_pair_count','nonfinite_count','target_counts','ranks_a','ranks_b','auxiliary_non_lora_keys']},indent=2))

def variants(adapter:Path,tmp:Path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    tmp.mkdir(parents=True,exist_ok=True)
    with safe_open(str(adapter/'adapter_model.safetensors'),framework='pt',device='cpu') as f: t={k:f.get_tensor(k).cpu() for k in f.keys()}
    gen=torch.Generator().manual_seed(SEED); rnd={}; sh={k:v.clone() for k,v in t.items()}; d05={k:v.clone() for k,v in t.items()}; d15={k:v.clone() for k,v in t.items()}
    for k,v in t.items():
        if 'lora_' in k:
            x=torch.randn(v.shape,generator=gen,dtype=torch.float32); n=float(torch.linalg.vector_norm(x)); target=float(torch.linalg.vector_norm(v.float())); rnd[k]=(x*(target/n if n else 0)).to(v.dtype)
            if '.lora_B.' in k: d05[k]=(v.float()*0.5).to(v.dtype); d15[k]=(v.float()*1.5).to(v.dtype)
        else: rnd[k]=v.clone()
    groups={}
    for k,v in t.items():
        m=re.search(r'layers\.(\d+)\.',k); target=next((q for q in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if q in k),'other'); ab='A' if '.lora_A.' in k else ('B' if '.lora_B.' in k else 'X')
        if m and ab!='X': groups.setdefault((target,ab,tuple(v.shape)),[]).append((int(m.group(1)),k))
    for items in groups.values():
        items=sorted(items); vals=[t[k] for _,k in items]; vals=vals[1:]+vals[:1]
        for (_,k),v in zip(items,vals): sh[k]=v.clone()
    out={}
    for name,data in [('RANDOM_RANK_MATCHED',rnd),('LAYER_SHUFFLED',sh),('DOSE_0_5',d05),('DOSE_1_5',d15)]:
        d=tmp/name;d.mkdir(parents=True,exist_ok=True);shutil.copy2(adapter/'adapter_config.json',d/'adapter_config.json');save_file(data,str(d/'adapter_model.safetensors'));out[name]=d
    return out

def ablate(adapter:Path,tmp:Path,group:str):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(str(adapter/'adapter_model.safetensors'),framework='pt',device='cpu') as f: t={k:f.get_tensor(k).cpu().clone() for k in f.keys()}
    layers=[int(m.group(1)) for k in t for m in [re.search(r'layers\.(\d+)\.',k)] if m]; mx=max(layers); one=(mx+1)/3
    for k,v in list(t.items()):
        if '.lora_B.' not in k: continue
        m=re.search(r'layers\.(\d+)\.',k); layer=int(m.group(1)) if m else -1
        hit=(group=='Q_PROJ' and 'q_proj' in k) or (group=='V_PROJ' and 'v_proj' in k) or (group=='MLP_ALL' and any(x in k for x in ['gate_proj','up_proj','down_proj'])) or (group=='EARLY' and 0<=layer<one) or (group=='MIDDLE' and one<=layer<2*one) or (group=='LATE' and layer>=2*one)
        if hit: t[k]=torch.zeros_like(v)
    d=tmp/group;d.mkdir(parents=True,exist_ok=True);shutil.copy2(adapter/'adapter_config.json',d/'adapter_config.json');save_file(t,str(d/'adapter_model.safetensors'));return d

def parse_decision(text):
    try:
        s=text.find('{'); e=text.rfind('}')
        if s<0 or e<s: return None
        d=json.loads(text[s:e+1])
        if not isinstance(d,dict): return None
        start=d.get('start'); end=d.get('end'); de=d.get('deferred_tasks')
        if not (isinstance(start,str) and isinstance(end,str) and isinstance(de,list) and all(isinstance(x,str) for x in de)): return None
        return {'start':start,'end':end,'deferred_tasks':sorted(de)}
    except Exception:return None

def _mins(s):
    try:h,m=map(int,s.split(':')); return h*60+m
    except Exception:return None

def valid_decision(dec,case):
    if dec is None:return False
    s=_mins(dec['start']);e=_mins(dec['end']); dur=case['input']['new_task']['duration_minutes']
    if s is None or e is None or s<WORK_START or e>WORK_END or e-s!=dur or s%5:return False
    ids={x['id'] for x in case['input']['existing_items'] if not x['fixed']}
    if len(set(dec['deferred_tasks']))!=len(dec['deferred_tasks']) or not set(dec['deferred_tasks']).issubset(ids):return False
    items=[]
    for x in case['input']['existing_items']:
        y=dict(x);y['start']=_mins(y['start']);y['end']=_mins(y['end']);items.append(y)
    from c65_schedule_generator import intervals_blocked
    blocked=intervals_blocked(items,dec['deferred_tasks'])
    if any(not (e<=a or s>=b) for a,b in blocked):return False
    return True

def behavior(out:Path,adapter:Path,base:Path,tmp:Path,seal_path:Path,capture_path:Path):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    seal,cases=verify_seal(seal_path); TH=seal['thresholds']; FT=seal['failure_thresholds']; out.mkdir(parents=True,exist_ok=True);cap=json.loads(capture_path.read_text())
    tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True); tok.pad_token=tok.eos_token; tok.padding_side='left'
    def chat(c):
        msgs=[{'role':'system','content':c['prompt'].split('\nUSER\n',1)[0]},{'role':'user','content':c['prompt'].split('\nUSER\n',1)[1]}]
        try:return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError:return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    def load(ad=None):
        b=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,torch_dtype=torch.float32,device_map=None);b.eval()
        if ad is None:return b
        return PeftModel.from_pretrained(b,str(ad),is_trainable=False).eval()
    def run(model,cs,batch=6):
        rows=[]
        for s in range(0,len(cs),batch):
            ch=cs[s:s+batch]; texts=[chat(c) for c in ch]; enc=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=1024)
            with torch.inference_mode(): y=model.generate(**enc,max_new_tokens=80,do_sample=False,pad_token_id=tok.eos_token_id)
            for j,c in enumerate(ch):
                comp=tok.decode(y[j,enc['input_ids'].shape[1]:],skip_special_tokens=True); dec=parse_decision(comp); valid=valid_decision(dec,c); expected={**c['expected'],'deferred_tasks':sorted(c['expected']['deferred_tasks'])}; exact=(dec==expected)
                rows.append({'case_id':c['case_id'],'pair_id':c['pair_id'],'decision':dec,'valid':valid,'exact':exact,'expected_deferral':bool(c['expected']['deferred_tasks']),'deferral_exact':bool(dec is not None and dec['deferred_tasks']==expected['deferred_tasks']),'output_hash':hashlib.sha256(comp.encode()).hexdigest()})
        return rows
    def agg(rs):
        pairs={}
        for x in rs:pairs.setdefault(x['pair_id'],[]).append(x)
        defcases=[x for x in rs if x['expected_deferral']]
        return {'n':len(rs),'exact_optimal_rate':sum(x['exact'] for x in rs)/len(rs),'valid_rate':sum(x['valid'] for x in rs)/len(rs),'pair_exact_rate':sum(len(v)==2 and all(z['exact'] for z in v) for v in pairs.values())/len(pairs),'deferral_exact_rate':sum(x['deferral_exact'] for x in defcases)/len(defcases) if defcases else 1.0,'parse_rate':sum(x['decision'] is not None for x in rs)/len(rs)}
    abi={}
    for name,ad in [('BASE',None),('FULL',adapter)]:
        m=load(ad);a=run(m,cases[:4],1);b=run(m,cases[:4],4);abi[name]={'decision_equal':[x['decision'] for x in a]==[x['decision'] for x in b],'hash_equal':[x['output_hash'] for x in a]==[x['output_hash'] for x in b]};del m;gc.collect()
    (out/'C65_RUNTIME_ABI.json').write_text(json.dumps(abi,indent=2,sort_keys=True))
    if not all(v['decision_equal'] and v['hash_equal'] for v in abi.values()): raise SystemExit(41)
    vs=variants(adapter,tmp); conds=[('BASE',None),('FULL',adapter),('RANDOM_RANK_MATCHED',vs['RANDOM_RANK_MATCHED']),('LAYER_SHUFFLED',vs['LAYER_SHUFFLED']),('DOSE_0_5',vs['DOSE_0_5']),('DOSE_1_5',vs['DOSE_1_5'])]
    summary={}; hashes=[]; semantic=[]
    for name,ad in conds:
        m=load(ad);rs=run(m,cases,6);del m;gc.collect();summary[name]=agg(rs);hashes += [{'condition':name,'case_id':x['case_id'],'output_hash':x['output_hash']} for x in rs];semantic += [{'condition':name,'case_id':x['case_id'],'decision':x['decision'],'valid':x['valid'],'exact':x['exact']} for x in rs]
    f=summary['FULL']; b=summary['BASE']; rnd=summary['RANDOM_RANK_MATCHED']; sh=summary['LAYER_SHUFFLED']
    checks={'full_exact_optimal':f['exact_optimal_rate']>=TH['full_exact_optimal_min'],'full_valid':f['valid_rate']>=TH['full_valid_min'],'full_pair_exact':f['pair_exact_rate']>=TH['full_pair_exact_min'],'beats_base':f['exact_optimal_rate']-b['exact_optimal_rate']>=TH['full_minus_base_exact_min'],'beats_random':f['exact_optimal_rate']-rnd['exact_optimal_rate']>=TH['full_minus_random_exact_min'],'beats_shuffle':f['exact_optimal_rate']-sh['exact_optimal_rate']>=TH['full_minus_shuffle_exact_min'],'deferral_exact':f['deferral_exact_rate']>=TH['full_deferral_precision_min']};positive=all(checks.values())
    fchecks={'exact_degradation':f['exact_optimal_rate']-b['exact_optimal_rate']<=FT['full_minus_base_exact_max_for_degradation'],'valid_degradation':f['valid_rate']-b['valid_rate']<=FT['full_minus_base_valid_max_for_degradation'],'deferral_degradation':f['deferral_exact_rate']-b['deferral_exact_rate']<=FT['full_minus_base_deferral_exact_max_for_degradation']};failure=any(fchecks.values())
    six_hash_identity=all(len({next(h['output_hash'] for h in hashes if h['condition']==cond and h['case_id']==c['case_id']) for cond,_ in conds})==1 for c in cases);six_decision_identity=all(len({json.dumps(next(s['decision'] for s in semantic if s['condition']==cond and s['case_id']==c['case_id']),sort_keys=True) for cond,_ in conds})==1 for c in cases)
    causal={}
    if positive:
        sub=cases[:12];fm=load(adapter);fr=run(fm,sub,6);del fm;gc.collect();full_sub=agg(fr)
        for g in seal['causal_groups']:
            ad=ablate(adapter,tmp/'ablations',g);m=load(ad);rs=run(m,sub,6);del m;gc.collect();aa=agg(rs);causal[g]={'exact_optimal_rate':aa['exact_optimal_rate'],'valid_rate':aa['valid_rate'],'drop_vs_full_exact':full_sub['exact_optimal_rate']-aa['exact_optimal_rate']}
    rec={'schema':'R22583_C65_BEHAVIOR_V1','summary':summary,'positive_gate':positive,'positive_gate_checks':checks,'failure_gate':failure,'failure_gate_checks':fchecks,'causal_run':positive,'causal':causal,'six_condition_exact_output_hash_identity':six_hash_identity,'six_condition_decision_identity':six_decision_identity,'output_hashes':hashes,'semantic_decisions':semantic,'raw_outputs_exported':False,'source_reported_metrics_are_evidence':False,'case_count':len(cases),'base_runtime':{'repo':cap['base_runtime_repo'],'revision':cap['base_runtime_revision']}}
    (out/'C65_BEHAVIOR.json').write_text(json.dumps(rec,indent=2,sort_keys=True));print(json.dumps({'summary':summary,'positive_gate':positive,'checks':checks,'failure_gate':failure,'failure_checks':fchecks,'six_hash_identity':six_hash_identity,'six_decision_identity':six_decision_identity,'causal':causal},indent=2))

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
    q=sp.add_parser('capture');q.add_argument('--out',required=True);q.add_argument('--escrow',required=True);q.add_argument('--seal',required=True)
    q=sp.add_parser('static');q.add_argument('--out',required=True);q.add_argument('--adapter',required=True)
    q=sp.add_parser('behavior');q.add_argument('--out',required=True);q.add_argument('--adapter',required=True);q.add_argument('--base',required=True);q.add_argument('--tmp',required=True);q.add_argument('--seal',required=True);q.add_argument('--capture-json',required=True)
    a=p.parse_args()
    if a.cmd=='capture':capture(Path(a.out),Path(a.escrow),Path(a.seal))
    elif a.cmd=='static':static(Path(a.out),Path(a.adapter))
    else:behavior(Path(a.out),Path(a.adapter),Path(a.base),Path(a.tmp),Path(a.seal),Path(a.capture_json))
if __name__=='__main__':main()
