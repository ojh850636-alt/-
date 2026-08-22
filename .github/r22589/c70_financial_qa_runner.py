from __future__ import annotations
import json, hashlib, os, pathlib, re, shutil, time, traceback, urllib.request, random, math, collections
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

ADAPTER_REPO='fotapol/qwen3-1.7b-financial-qa-lora'
BASE_REPO='Qwen/Qwen3-1.7B'
BASE_REV='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
QUESTION='CONTEXT_GROUNDED_FINANCIAL_ARITHMETIC_WITH_EXACT_UNIT_NORMALIZATION_AND_SUPPORT_ONLY_REASONING'
ROOT=pathlib.Path('work'); DL=ROOT/'download'; ESC=ROOT/'escrow'; OUT=ROOT/'out'
for p in (DL,ESC,OUT): p.mkdir(parents=True,exist_ok=True)
UA={'User-Agent':'LUCIA-AA-R22589-C70-financial-one-use'}
Q=Decimal('0.01')
SYSTEM='Answer the financial question using only the provided context. Return only the final answer.'

def jwrite(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def api_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read().decode())
def text_url(url,limit=4_000_000):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:
        b=r.read(limit+1)
        if len(b)>limit: raise RuntimeError('TEXT_METADATA_TOO_LARGE')
        return b.decode('utf-8','replace')
def sibling(info,name):
    for x in info.get('siblings') or []:
        if x.get('rfilename')==name:
            l=x.get('lfs') or {}
            return {'size':l.get('size') or x.get('size'),'sha256':l.get('sha256') or l.get('oid'),'blob_id':x.get('blobId')}
    return {}
def direct_download(repo,rev,name,dst):
    u=f'https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true'
    req=urllib.request.Request(u,headers=UA); h=hashlib.sha256(); n=0
    with urllib.request.urlopen(req,timeout=300) as r,dst.open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b: break
            f.write(b); h.update(b); n+=len(b)
    return {'name':name,'size':n,'sha256':h.hexdigest()}

def rnd2(x): return x.quantize(Q,rounding=ROUND_HALF_UP)
def d(n): return Decimal(n)/100
def dec_answer(c):
    k=c['kind']; a=d(c['a']); b=d(c['b']); x=d(c.get('c',0))
    if k=='pct_increase': return rnd2((b-a)/a*100),'percent'
    if k=='pct_decrease': return rnd2((a-b)/a*100),'percent'
    if k=='gross_margin': return rnd2((a-b)/a*100),'percent'
    if k=='share': return rnd2(a/b*100),'percent'
    if k=='difference': return rnd2(b-a),'million'
    if k=='average3': return rnd2((a+b+x)/3),'million'
    raise KeyError(k)
def round_frac(f):
    s=f*100; q,r=divmod(s.numerator,s.denominator)
    if r*2>=s.denominator: q+=1
    return Fraction(q,100)
def frac_answer(c):
    k=c['kind']; a=Fraction(c['a'],100); b=Fraction(c['b'],100); x=Fraction(c.get('c',0),100)
    if k=='pct_increase': y=(b-a)/a*100; u='percent'
    elif k=='pct_decrease': y=(a-b)/a*100; u='percent'
    elif k=='gross_margin': y=(a-b)/a*100; u='percent'
    elif k=='share': y=a/b*100; u='percent'
    elif k=='difference': y=b-a; u='million'
    elif k=='average3': y=(a+b+x)/3; u='million'
    else: raise KeyError(k)
    z=round_frac(y); return Decimal(z.numerator)/Decimal(z.denominator),u
def make_prompt(c):
    k=c['kind']; a=d(c['a']); b=d(c['b']); x=d(c.get('c',0)); co=c['company']; y=c['year']; f=lambda z:f'{z:.2f}'
    if k=='pct_increase': ctx=f'{co} revenue was ${f(a)} million in {y} and ${f(b)} million in {y+1}.'; q='What was the percentage increase in revenue?'
    elif k=='pct_decrease': ctx=f'{co} operating expense was ${f(a)} million in {y} and ${f(b)} million in {y+1}.'; q='What was the percentage decrease in operating expense?'
    elif k=='gross_margin': ctx=f'{co} reported revenue of ${f(a)} million and cost of goods sold of ${f(b)} million in {y}.'; q='What was the gross margin as a percentage of revenue?'
    elif k=='share': ctx=f'{co} total revenue was ${f(b)} million in {y}, of which the services segment contributed ${f(a)} million.'; q='What percentage of total revenue came from the services segment?'
    elif k=='difference': ctx=f'{co} cash balance was ${f(a)} million at the start of {y} and ${f(b)} million at year end.'; q='By how many million dollars did the cash balance change?'
    elif k=='average3': ctx=f'{co} revenue was ${f(a)} million in {y}, ${f(b)} million in {y+1}, and ${f(x)} million in {y+2}.'; q='What was the average annual revenue across the three years, in millions?'
    return f'Context:\n{ctx}\n\nQuestion:\n{q}'
def build_cases(seed=22589,n=64):
    rng=random.Random(seed); kinds=['pct_increase','pct_decrease','gross_margin','share','difference','average3']; out=[]
    for i in range(n):
        k=kinds[i%6]; a=rng.randint(250,9500); b=rng.randint(250,9500); x=rng.randint(250,9500)
        if k=='pct_increase' and b<=a: b=a+rng.randint(50,4000)
        if k=='pct_decrease' and b>=a: b=max(50,a-rng.randint(50,max(51,a//2)))
        if k=='gross_margin': a=max(a,1000); b=rng.randint(100,a-50)
        if k=='share': b=max(b,1000); a=rng.randint(50,b)
        if k=='difference' and b<a: a,b=b,a
        c={'id':f'fq{i:03d}','kind':k,'a':a,'b':b,'c':x,'company':f'Company-{chr(65+i%26)}{i}','year':2011+i%13}
        da,u=dec_answer(c); fa,u2=frac_answer(c)
        if da!=fa or u!=u2: raise RuntimeError('ORACLE_DISAGREEMENT')
        c.update({'expected':format(da,'.2f'),'unit':u,'prompt':make_prompt(c)})
        out.append(c)
    return out
def parse_answer(t):
    t=t.strip().lower().replace(',','')
    if len(t)>120: return {'ok':False,'reason':'too_long'}
    nums=re.findall(r'[-+]?\d+(?:\.\d+)?',t)
    if len(nums)!=1: return {'ok':False,'reason':'numeric_count'}
    try: v=Decimal(nums[0]).quantize(Q,rounding=ROUND_HALF_UP)
    except Exception: return {'ok':False,'reason':'parse'}
    pct=('%' in t or 'percent' in t); mil=('million' in t or re.search(r'\bmn\b',t) is not None)
    if pct and mil: return {'ok':False,'reason':'unit_conflict'}
    return {'ok':True,'value':v,'unit':'percent' if pct else ('million' if mil else 'none')}
def verify_text(t,c):
    p=parse_answer(t)
    if not p.get('ok'): return False,p.get('reason')
    ok=(p['value']==Decimal(c['expected']) and p['unit']==c['unit'])
    return ok,'PASS' if ok else 'value_or_unit'
def preseal(cases):
    blob=json.dumps(cases,sort_keys=True,separators=(',',':')).encode()
    return {'schema':'R22589_C70_FINANCIAL_QA_PRESEAL_V1','cases':len(cases),'sha256':sha_bytes(blob),'oracle_decimal_fraction_exact':all(dec_answer(c)==frac_answer(c) for c in cases),'training_rows_used':0,'model_outputs_used':0,'split':{'primary':12,'holdout':8,'ood':8,'unused_oracle_only':36}}

def main():
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(min(4,os.cpu_count() or 2)); torch.manual_seed(22589)
    start=time.time(); cases=build_cases(); ps=preseal(cases); jwrite('R22589_C70_FINANCIAL_QA_PRESEAL.json',ps)
    inf=api_json('https://huggingface.co/api/models/'+ADAPTER_REPO+'?blobs=true'); arev=inf.get('sha')
    if not re.fullmatch(r'[0-9a-f]{40}',str(arev)): raise RuntimeError('NO_IMMUTABLE_ADAPTER_SHA')
    card=inf.get('cardData') or {}; meta=sibling(inf,'adapter_model.safetensors')
    readme=text_url(f'https://huggingface.co/{ADAPTER_REPO}/raw/{arev}/README.md'); cfg_txt=text_url(f'https://huggingface.co/{ADAPTER_REPO}/raw/{arev}/adapter_config.json'); cfg=json.loads(cfg_txt)
    if card.get('license')!='apache-2.0': raise RuntimeError('LICENSE_DRIFT:'+str(card.get('license')))
    if BASE_REV not in readme or BASE_REPO not in readme: raise RuntimeError('EXACT_TRAINING_BASE_NOT_IN_IMMUTABLE_CARD')
    if cfg.get('base_model_name_or_path')!=BASE_REPO or cfg.get('peft_type')!='LORA': raise RuntimeError('BASE_OR_PEFT_DRIFT')
    if cfg.get('r')!=16 or cfg.get('lora_alpha')!=32 or cfg.get('modules_to_save') not in (None,[]): raise RuntimeError('LORA_OR_AUX_DRIFT')
    prov={'schema':'R22589_C70_ATOMIC_SOURCE_PIN_V1','question':QUESTION,'adapter_repo':ADAPTER_REPO,'adapter_revision':arev,'adapter_lfs':meta,'adapter_license':card.get('license'),'base_repo':BASE_REPO,'training_base_revision':BASE_REV,'training_base_revision_source':'immutable adapter README','adapter_config_revision':cfg.get('revision'),'r':cfg.get('r'),'lora_alpha':cfg.get('lora_alpha'),'target_modules_declared':sorted(cfg.get('target_modules') or []),'modules_to_save':cfg.get('modules_to_save'),'weight_gets_before_pin':0,'model_weight_bytes_before_pin':0,'preseal':ps,'publisher_metrics_credit':0}
    jwrite('R22589_C70_ATOMIC_SOURCE_PIN.json',prov)
    ad_stage=DL/'adapter_model.safetensors'; adrec=direct_download(ADAPTER_REPO,arev,'adapter_model.safetensors',ad_stage)
    if meta.get('size') and int(meta['size'])!=adrec['size']: raise RuntimeError('ADAPTER_SIZE_MISMATCH')
    if meta.get('sha256') and str(meta['sha256']).replace('sha256:','')!=adrec['sha256']: raise RuntimeError('ADAPTER_SHA_MISMATCH')
    ad_dir=ESC/'adapter'; ad_dir.mkdir(); ad_path=ad_dir/'adapter_model.safetensors'; os.replace(ad_stage,ad_path); (ad_dir/'adapter_config.json').write_text(cfg_txt); (ad_dir/'README.md').write_text(readme)
    A={};B={};other=[]
    with safe_open(ad_path,framework='numpy') as f:
        for n in f.keys():
            if '.lora_A.' in n: A[n.replace('.lora_A.default.weight','').replace('.lora_A.weight','')]=np.asarray(f.get_tensor(n),dtype=np.float32)
            elif '.lora_B.' in n: B[n.replace('.lora_B.default.weight','').replace('.lora_B.weight','')]=np.asarray(f.get_tensor(n),dtype=np.float32)
            else: other.append({'name':n,'shape':list(f.get_slice(n).get_shape()),'dtype':str(f.get_slice(n).get_dtype())})
    keys=sorted(set(A)|set(B)); incomplete=[k for k in keys if k not in A or k not in B]
    if incomplete: raise RuntimeError('INCOMPLETE_LORA_PAIRS')
    bymod=collections.defaultdict(lambda:{'pairs':0,'proxy':0.0,'zero_pairs':0,'rank_min':999999,'rank_max':0}); bylayer=collections.defaultdict(lambda:{'pairs':0,'proxy':0.0,'modules':set()}); rows=[]; total=0.0; zero=0; finite=True; scale=cfg['lora_alpha']/cfg['r']
    for k in keys:
        a=A[k]; b=B[k]; z=not(np.count_nonzero(a) and np.count_nonzero(b)); zero+=int(z); finite &= bool(np.isfinite(a).all() and np.isfinite(b).all()); er=min(int(np.linalg.matrix_rank(a.astype(np.float64))),int(np.linalg.matrix_rank(b.astype(np.float64)))); proxy=float((scale*np.linalg.norm(a.astype(np.float64))*np.linalg.norm(b.astype(np.float64)))**2); total+=proxy
        mod=k.split('.')[-1]; lm=re.search(r'\.layers\.(\d+)\.',k); layer=int(lm.group(1)) if lm else -1
        m=bymod[mod]; m['pairs']+=1;m['proxy']+=proxy;m['zero_pairs']+=int(z);m['rank_min']=min(m['rank_min'],er);m['rank_max']=max(m['rank_max'],er)
        l=bylayer[layer];l['pairs']+=1;l['proxy']+=proxy;l['modules'].add(mod)
        rows.append({'component_id':k,'layer':layer,'module':mod,'a_shape':list(a.shape),'b_shape':list(b.shape),'effective_rank_upper_bound':er,'zero_pair':z})
    for v in bymod.values(): v['energy_ratio']=v.pop('proxy')/total if total else 0
    for v in bylayer.values(): v['energy_ratio']=v.pop('proxy')/total if total else 0;v['modules']=sorted(v['modules'])
    static={'schema':'R22589_C70_ADAPTER_OPERATOR_ARCHAEOLOGY_V1','source':{'repo':ADAPTER_REPO,'revision':arev,'sha256':adrec['sha256'],'size':adrec['size']},'operator':{'tensor_count':len(A)+len(B)+len(other),'complete_pairs':len(keys),'alive_pairs':len(keys)-zero,'zero_pairs':zero,'all_finite':finite,'auxiliary_tensor_count':len(other),'target_modules_actual':sorted(bymod),'by_module':dict(sorted(bymod.items())),'by_layer':{str(k):v for k,v in sorted(bylayer.items())},'pair_shape_rank_inventory':rows,'energy_proxy_definition':'(alpha/r*||A||F*||B||F)^2 ranking only; NOT Delta-W norm or causal importance'},'claim_boundary':'STATIC_ONLY_UNTIL_FRESH_NUMERIC_BEHAVIOR'}
    jwrite('R22589_C70_ADAPTER_OPERATOR_ARCHAEOLOGY.json',static)
    if zero or not finite or other: raise RuntimeError('ADAPTER_OPERATOR_VIABILITY_FAIL')
    binfo=api_json(f'https://huggingface.co/api/models/{BASE_REPO}/revision/{BASE_REV}?blobs=true')
    if binfo.get('sha')!=BASE_REV: raise RuntimeError('BASE_REVISION_API_MISMATCH')
    sibs=[x.get('rfilename') for x in binfo.get('siblings') or [] if x.get('rfilename')]
    need=[x for x in sibs if re.fullmatch(r'model-\d+-of-\d+\.safetensors',x)]
    need+= [x for x in ['model.safetensors','model.safetensors.index.json','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json'] if x in sibs]
    need=list(dict.fromkeys(need))
    if not any(x.endswith('.safetensors') for x in need): raise RuntimeError('NO_BASE_SAFETENSORS')
    bdir=ESC/'base'; bdir.mkdir(); base_manifest=[]
    for name in need:
        p=DL/name; rec=direct_download(BASE_REPO,BASE_REV,name,p); os.replace(p,bdir/name); base_manifest.append(rec)
    prov['adapter_actual']=adrec;prov['base_files']=base_manifest;jwrite('R22589_C70_ATOMIC_SOURCE_PIN.json',prov)
    tok=AutoTokenizer.from_pretrained(str(bdir),local_files_only=True,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    base=AutoModelForCausalLM.from_pretrained(str(bdir),local_files_only=True,dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    primary=cases[:12]; holdout=cases[12:20]; ood=cases[20:28]
    neg=[{'id':f'neg{i}','prompt':f'Context:\nCompany N{i} reported revenue of $10.00 million in 2020.\n\nQuestion:\nWhat was its gross margin percentage? If the context is insufficient, return INSUFFICIENT.','expected':'INSUFFICIENT'} for i in range(4)]
    def chat(c):
        return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':c['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    def gen(m,c,max_new=24):
        p=chat(c); ins=tok(p,return_tensors='pt')
        with torch.inference_mode(): out=m.generate(**ins,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.eos_token_id,use_cache=True)
        return tok.decode(out[0,ins['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    def eval_cases(m,cs):
        out=[]
        for c in cs:
            t=gen(m,c); ok,why=verify_text(t,c); out.append({'id':c['id'],'kind':c['kind'],'expected_sha256':sha_bytes((c['expected']+'|'+c['unit']).encode()),'pass':ok,'reason':why,'completion_sha256':sha_bytes(t.encode()),'completion_chars':len(t)})
            del t
        return out
    def eval_neg(m):
        out=[]
        for c in neg:
            t=gen(m,c); ok=(t.strip().upper()=='INSUFFICIENT');out.append({'id':c['id'],'pass':ok,'completion_sha256':sha_bytes(t.encode()),'completion_chars':len(t)});del t
        return out
    def target(c): return c['expected']+('%' if c['unit']=='percent' else ' million')
    def nll(m,c):
        p=chat(c); a=tok(p,add_special_tokens=False)['input_ids']; b=tok(target(c),add_special_tokens=False)['input_ids']; ids=torch.tensor([a+b]); lab=torch.tensor([[-100]*len(a)+b])
        with torch.inference_mode(): return float(m(input_ids=ids,labels=lab,use_cache=False).loss.float().item())
    behavior={'schema':'R22589_C70_FRESH_NUMERIC_BEHAVIOR_V1','question':QUESTION,'preseal':ps,'publisher_metrics_credit':0,'primary':{},'controls':{},'holdout':{'status':'LOCKED_NOT_READ_UNLESS_GATE'},'ood':{'status':'LOCKED_NOT_READ_UNLESS_HOLDOUT_GATE'},'collateral_negative':{},'causal':{'status':'NOT_RUN'},'claim_boundary':{}}
    bp=eval_cases(base,primary); bn=[nll(base,c) for c in primary]; bneg=eval_neg(base)
    behavior['primary']['BASE']={'outcomes':bp,'strict_pass':sum(x['pass'] for x in bp),'target_nll':bn};behavior['collateral_negative']['BASE']={'outcomes':bneg,'pass':sum(x['pass'] for x in bneg)}
    model=PeftModel.from_pretrained(base,str(ad_dir),is_trainable=False).eval()
    original={n:p.detach().clone() for n,p in model.named_parameters() if 'lora_' in n}; named=dict(model.named_parameters())
    def restore():
        with torch.no_grad():
            for n,x in original.items(): named[n].copy_(x)
    def apply(cond):
        restore()
        with torch.no_grad():
            if cond.startswith('dose'):
                z=float(cond[4:])
                for n in original:
                    if 'lora_B' in n: named[n].mul_(z)
            elif cond.startswith('random'):
                g=torch.Generator(device='cpu');g.manual_seed(22589+int(cond[-1]))
                for n,x0 in original.items():
                    x=torch.randn(tuple(named[n].shape),generator=g,dtype=torch.float32);norm=x0.float().norm();x=x/(x.norm()+1e-12)*norm;named[n].copy_(x.to(named[n].dtype))
            elif cond=='shuffle':
                groups={}
                for n in original:
                    m=re.search(r'\.layers\.(\d+)\.',n)
                    if m: groups.setdefault(re.sub(r'\.layers\.\d+\.', '.layers.*.',n),[]).append((int(m.group(1)),n))
                for arr in groups.values():
                    arr.sort(); src=[original[n] for _,n in arr]
                    for i,(_,n) in enumerate(arr): named[n].copy_(src[(i+1)%len(src)])
    restore(); fp=eval_cases(model,primary); fn=[nll(model,c) for c in primary]; fneg=eval_neg(model)
    behavior['primary']['FULL']={'outcomes':fp,'strict_pass':sum(x['pass'] for x in fp),'target_nll':fn};behavior['collateral_negative']['FULL']={'outcomes':fneg,'pass':sum(x['pass'] for x in fneg)}
    rescue=[i for i,(b,f) in enumerate(zip(bp,fp)) if (not b['pass']) and f['pass']];harm=[i for i,(b,f) in enumerate(zip(bp,fp)) if b['pass'] and not f['pass']]; gain=len(rescue)>len(harm) and len(rescue)>0
    paired={'rescue_indices':rescue,'harm_indices':harm,'primary_gain':gain};behavior['primary']['paired']=paired
    for cond in ['random1','random2','shuffle','dose0.5','dose1.5']:
        apply(cond); rec={'target_nll':[nll(model,c) for c in primary],'generation':'NOT_RUN_NO_PRIMARY_RESCUE'}
        if rescue:
            inds=rescue[:4]; outs=eval_cases(model,[primary[i] for i in inds]);rec['generation']={'indices':inds,'outcomes':outs,'strict_pass':sum(x['pass'] for x in outs)}
        behavior['controls'][cond]=rec
    restore(); sep=False
    if rescue:
        n=min(4,len(rescue));sep=(behavior['controls']['random1']['generation']['strict_pass']<n and behavior['controls']['shuffle']['generation']['strict_pass']<n)
    paired['random_shuffle_separated']=sep
    hold_gain=False; hres=[];hharm=[]
    if gain and sep:
        with model.disable_adapter(): hb=eval_cases(model,holdout)
        restore(); hf=eval_cases(model,holdout);hres=[i for i,(b,f) in enumerate(zip(hb,hf)) if not b['pass'] and f['pass']];hharm=[i for i,(b,f) in enumerate(zip(hb,hf)) if b['pass'] and not f['pass']];hold_gain=len(hres)>len(hharm) and len(hres)>0
        behavior['holdout']={'status':'READ_AFTER_PRIMARY_CONTROL_GATE','BASE':hb,'FULL':hf,'rescue_indices':hres,'harm_indices':hharm,'same_environment_replicated_gain':hold_gain}
    if gain and sep and hold_gain:
        with model.disable_adapter(): ob=eval_cases(model,ood)
        restore(); of=eval_cases(model,ood);behavior['ood']={'status':'READ_AFTER_HOLDOUT_GATE','BASE':ob,'FULL':of,'full_pass':sum(x['pass'] for x in of),'base_pass':sum(x['pass'] for x in ob)}
        probe=primary[rescue[0]]; mods=sorted(static['operator']['target_modules_actual']); caus={'probe_id':probe['id'],'module_ablation':{},'module_sufficiency':{},'layer_quartile_ablation':{}}
        def zero_b(pred):
            restore()
            with torch.no_grad():
                for n in original:
                    if 'lora_B' in n and pred(n): named[n].zero_()
        for mod in mods:
            zero_b(lambda n,m=mod:f'.{m}.' in n); caus['module_ablation'][mod]=eval_cases(model,[probe])[0]
        for mod in mods:
            restore()
            with torch.no_grad():
                for n in original:
                    if 'lora_B' in n and f'.{mod}.' not in n: named[n].zero_()
            caus['module_sufficiency'][mod]=eval_cases(model,[probe])[0]
        for lo,hi in [(0,6),(7,13),(14,20),(21,27)]:
            zero_b(lambda n,lo=lo,hi=hi: (lambda m:bool(m) and lo<=int(m.group(1))<=hi)(re.search(r'\.layers\.(\d+)\.',n)));caus['layer_quartile_ablation'][f'{lo}_{hi}']=eval_cases(model,[probe])[0]
        restore();behavior['causal']={'status':'E3_BOUNDED_SAME_ENVIRONMENT_ONLY','results':caus,'not_e4_or_e5':True}
    behavior['claim_boundary']={'fresh_primary_gain':gain,'random_shuffle_separated':sep,'same_environment_holdout_gain':hold_gain,'external_positive_capability_e4_plus_increment':0,'e5_increment':0,'training_provenance_full_pipeline_available':False,'reason_e4e5_zero':'single physical GitHub runner and release omits normalized training examples/split manifest/full notebook/source pipeline; exact Base revision alone is not E5.'}
    jwrite('R22589_C70_FRESH_NUMERIC_BEHAVIOR.json',behavior)
    brain={'schema':'R22589_C70_RAWFREE_BRAIN_MATERIAL_V1','question':QUESTION,'observed':{'adapter_live':True,'primary_gain':gain,'control_separated':sep,'holdout_gain':hold_gain,'bounded_e3':behavior['causal']['status'].startswith('E3')},'method_laws':['EXACT_BASE_REVISION_ENABLES_BEHAVIOR_BUT_DOES_NOT_ALONE_ENABLE_E5','NUMERIC_ORACLE_CAN_AVOID_UNTRUSTED_CODE_EXECUTION','PUBLISHER_METRICS_ARE_ADMISSION_PRIOR_NOT_LAA_EVIDENCE','FIRST_CONTROL_SEPARATED_POSITIVE_MUST_STOP_BREADTH'],'promotion':'QUARANTINED_UNTIL_INDEPENDENT_E5_AND_FULL_TRAINING_PROVENANCE'}
    jwrite('R22589_C70_RAWFREE_BRAIN_MATERIAL.json',brain)
    return {'status':'PASS','adapter_revision':arev,'adapter_sha256':adrec['sha256'],'adapter_size':adrec['size'],'behavior':behavior['claim_boundary'],'elapsed_seconds':time.time()-start}

result=None;err=None
try:
    result=main()
except Exception as e:
    err={'type':type(e).__name__,'message':str(e),'traceback_tail':'\n'.join(traceback.format_exc().splitlines()[-14:])}
finally:
    for p in (DL,ESC):
        if p.exists(): shutil.rmtree(p,ignore_errors=True)
    raw=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and (p.suffix.lower() in {'.safetensors','.bin','.pt','.pth','.pkl','.pickle'} or p.name in {'tokenizer.json','vocab.json','merges.txt'}): raw.append(str(p))
    receipt={'schema':'R22589_C70_FINAL_DELETION_RECEIPT_V1','result':result,'error':err,'raw_remaining':raw,'raw_remaining_count':len(raw),'post_delete_pass':not raw,'source_consumed':(OUT/'R22589_C70_ADAPTER_OPERATOR_ARCHAEOLOGY.json').exists(),'base_behavior_executed':(OUT/'R22589_C70_FRESH_NUMERIC_BEHAVIOR.json').exists()}
    jwrite('R22589_C70_FINAL_DELETION_RECEIPT.json',receipt)
if err or not result or raw:
    print(json.dumps({'PASS':False,'error':err,'deletion':not raw},indent=2));raise SystemExit(1)
print(json.dumps({'PASS':True,'result':result,'deletion':True},indent=2))
