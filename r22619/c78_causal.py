import hashlib,json,math,os,random,re,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import c78_behavior as B
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

OUT=Path('out');OUT.mkdir(exist_ok=True)
BASE=Path('src_base');ADAPTER=Path('src_adapter')
BEHAV=Path(os.environ.get('C78_BEHAVIOR_RESULT','behavior/C78_BEHAVIOR_RESULT.json'))
beh=json.loads(BEHAV.read_text())
ids=[c['id'] for c in B.SUITE if c['split']=='CONFIRMATION'][:24]+[c['id'] for c in B.SUITE if c['split']=='OOD'][:24]
case_by={c['id']:c for c in B.SUITE}
mods=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
pre={'schema':'R22619_C78_CAUSAL_PRESEAL_V2','model_outputs_observed_at_seal':0,'behavior_gate_required':True,'causal_subset_ids':ids,'candidate':B.PRESEAL['candidate'],'candidate_revision':B.PRESEAL['candidate_revision'],'mandatory_claim_suffix':'_BASE_REVISION_HOLD','claim_ceiling':'E3_BASE_REVISION_HOLD','broad_groups':{'ATTENTION_ALL':{'layers':list(range(18)),'modules':['q_proj','k_proj','v_proj','o_proj']},'MLP_ALL':{'layers':list(range(18)),'modules':['gate_proj','up_proj','down_proj']},'EARLY_ALL':{'layers':list(range(0,6)),'modules':mods},'MIDDLE_ALL':{'layers':list(range(6,12)),'modules':mods},'LATE_ALL':{'layers':list(range(12,18)),'modules':mods}},'fine_layers':list(range(18)),'fine_modules':mods,'necessity_rule':'CI95 lower of FULL-minus-ABLATE >0','sufficiency_rule':'CI95 lower of ONLY_CANDIDATE-minus-BASE >0','survivor_rule':'necessity AND sufficiency; Holm alpha=.05 applied to paired sign-flip p-values','matched_random_rule':'site survivor must exceed all 8 same-module random-layer sufficiency controls; otherwise redundant/localization only','minimality_rule':'no minimal circuit unless site survivor has necessity+sufficiency+Holm+matched-random and LOO irredundancy','safety_red_blocks_student_promotion':True}
(OUT/'C78_CAUSAL_PRESEAL.json').write_text(json.dumps(pre,indent=2,sort_keys=True)+'\n')

if not beh.get('control_separated_positive_green',False):
    (OUT/'C78_CAUSAL_NOT_RUN.json').write_text(json.dumps({'schema':'R22619_C78_CAUSAL_NOT_RUN_V1','reason':'BEHAVIOR_CONTROL_GATE_RED','behavior_result_sha256':hashlib.sha256(BEHAV.read_bytes()).hexdigest(),'causal_interventions_run':0},indent=2,sort_keys=True)+'\n')
    print('CAUSAL NOT RUN: behavior gate RED');raise SystemExit(0)


def boot(vals,seed,n=3000):
    a=np.asarray(vals,dtype=float);rng=np.random.default_rng(seed);m=np.empty(n)
    for i in range(n):m[i]=a[rng.integers(0,len(a),len(a))].mean()
    return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def signflip_p(vals,seed,n=10000):
    a=np.asarray(vals,dtype=float);obs=float(a.mean());rng=np.random.default_rng(seed);cnt=0
    for _ in range(n):
        signs=rng.choice([-1.0,1.0],size=len(a));cnt+=abs(float((a*signs).mean()))>=abs(obs)
    return (cnt+1)/(n+1)

def holm(ps,alpha=.05):
    items=sorted(ps.items(),key=lambda kv:kv[1]);m=len(items);out={}
    running=0.0
    for i,(k,p) in enumerate(items):
        adj=min(1.0,(m-i)*p);running=max(running,adj);out[k]=running
    return out

def grp_match(v,layers,modules):return v['layer'] in layers and v['module'] in modules

def zero_group(lp,orig,layers,modules):
    B.restore(lp,orig)
    with torch.no_grad():
        for n,v in lp.items():
            if v['ab']=='B' and grp_match(v,layers,modules):v['p'].zero_()

def only_group(lp,orig,layers,modules):
    B.restore(lp,orig)
    with torch.no_grad():
        for n,v in lp.items():
            if v['ab']=='B' and not grp_match(v,layers,modules):v['p'].zero_()

def score_set(model,tok,cases):return {c['id']:B.score_one(model,tok,c) for c in cases}

def paired(a,b):return [a[i]-b[i] for i in ids]

def stat(vals,seed):return {'n':len(vals),'mean':float(np.mean(vals)),'ci95':boot(vals,seed),'p_signflip':signflip_p(vals,seed+10000)}

ah=hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest();bh=hashlib.sha256((BASE/'model.safetensors').read_bytes()).hexdigest()
assert ah=='414f0d5db8d6560a12f5841eb4a7fdc8ce558db37f4fe4e1c3c19a249223f918';assert bh=='d3ae9628a0ed2cfa21b5b8345be4e1d4126794e734f3483dcc275b4735bf5a15'
tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True,trust_remote_code=False,use_fast=True)
if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,trust_remote_code=False,device_map={'':'cpu'})
model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=False,local_files_only=True);model.eval()
lp=B.lora_params(model);assert len(lp)==252
orig=B.snapshot(lp);cases=[case_by[i] for i in ids]
B.restore(lp,orig);full=score_set(model,tok,cases)
base_scores={c['id']:B.score_one(model,tok,c,disabled=True) for c in cases}

results={'schema':'R22619_C78_CAUSAL_RESULT_V1','source':{'adapter_sha256':ah,'base_sha256':bh},'claim_suffix':'_BASE_REVISION_HOLD','broad':{},'fine_layers':{},'fine_modules':{},'fine_sites':{},'matched_random':{},'raw_activations_retained':0,'raw_logits_retained':0}
# broad
for ix,(name,g) in enumerate(pre['broad_groups'].items()):
    zero_group(lp,orig,g['layers'],g['modules']);abl=score_set(model,tok,cases)
    only_group(lp,orig,g['layers'],g['modules']);only=score_set(model,tok,cases)
    nec=paired(full,abl);suf=paired(only,base_scores)
    results['broad'][name]={'necessity':stat(nec,100+ix),'sufficiency':stat(suf,200+ix)}
B.restore(lp,orig)
# Holm broad across both required tests
pnec={k:v['necessity']['p_signflip'] for k,v in results['broad'].items()};psuf={k:v['sufficiency']['p_signflip'] for k,v in results['broad'].items()}
hnec=holm(pnec);hsuf=holm(psuf)
for k,v in results['broad'].items():
    v['necessity']['holm_p']=hnec[k];v['sufficiency']['holm_p']=hsuf[k]
    v['survivor']=v['necessity']['ci95'][0]>0 and v['sufficiency']['ci95'][0]>0 and hnec[k]<.05 and hsuf[k]<.05
broad_surv=[k for k,v in results['broad'].items() if v['survivor']]
results['broad_survivors']=broad_surv

if broad_surv:
    # layer search
    for l in range(18):
        zero_group(lp,orig,[l],mods);abl=score_set(model,tok,cases)
        only_group(lp,orig,[l],mods);only=score_set(model,tok,cases)
        results['fine_layers'][f'L{l}']={'necessity':stat(paired(full,abl),300+l),'sufficiency':stat(paired(only,base_scores),400+l)}
    # module search
    for j,m in enumerate(mods):
        zero_group(lp,orig,list(range(18)),[m]);abl=score_set(model,tok,cases)
        only_group(lp,orig,list(range(18)),[m]);only=score_set(model,tok,cases)
        results['fine_modules'][m]={'necessity':stat(paired(full,abl),500+j),'sufficiency':stat(paired(only,base_scores),600+j)}
    for bucket in ('fine_layers','fine_modules'):
        pn={k:v['necessity']['p_signflip'] for k,v in results[bucket].items()};ps={k:v['sufficiency']['p_signflip'] for k,v in results[bucket].items()};hn=holm(pn);hs=holm(ps)
        for k,v in results[bucket].items():
            v['necessity']['holm_p']=hn[k];v['sufficiency']['holm_p']=hs[k];v['survivor']=v['necessity']['ci95'][0]>0 and v['sufficiency']['ci95'][0]>0 and hn[k]<.05 and hs[k]<.05
    ls=[int(k[1:]) for k,v in results['fine_layers'].items() if v['survivor']];ms=[k for k,v in results['fine_modules'].items() if v['survivor']]
    results['layer_survivors']=ls;results['module_survivors']=ms
    # intersections only
    for l in ls:
        for j,m in enumerate(ms):
            sid=f'L{l}_{m}';zero_group(lp,orig,[l],[m]);abl=score_set(model,tok,cases);only_group(lp,orig,[l],[m]);only=score_set(model,tok,cases)
            results['fine_sites'][sid]={'layer':l,'module':m,'necessity':stat(paired(full,abl),700+l*10+j),'sufficiency':stat(paired(only,base_scores),800+l*10+j)}
    if results['fine_sites']:
        pn={k:v['necessity']['p_signflip'] for k,v in results['fine_sites'].items()};ps={k:v['sufficiency']['p_signflip'] for k,v in results['fine_sites'].items()};hn=holm(pn);hs=holm(ps)
        for k,v in results['fine_sites'].items():
            v['necessity']['holm_p']=hn[k];v['sufficiency']['holm_p']=hs[k];v['survivor']=v['necessity']['ci95'][0]>0 and v['sufficiency']['ci95'][0]>0 and hn[k]<.05 and hs[k]<.05
        ss=[k for k,v in results['fine_sites'].items() if v['survivor']];results['site_survivors']=ss
        rng=random.Random(22619)
        for sid in ss:
            v=results['fine_sites'][sid];l=v['layer'];m=v['module'];others=[x for x in range(18) if x!=l];rng.shuffle(others);ctrl=others[:8];cand=v['sufficiency']['mean'];vals=[]
            for q in ctrl:
                only_group(lp,orig,[q],[m]);sc=score_set(model,tok,cases);st=stat(paired(sc,base_scores),900+q);vals.append({'site':f'L{q}_{m}','sufficiency':st})
            results['matched_random'][sid]={'candidate_sufficiency_mean':cand,'controls':vals,'candidate_exceeds_all_controls':all(cand>x['sufficiency']['mean'] for x in vals)}
B.restore(lp,orig)
# final conservative claim
if not broad_surv:claim='E2_CONTROL_SEPARATED_BEHAVIOR_NO_CAUSAL_LOCALIZATION_BASE_REVISION_HOLD'
elif any(v.get('survivor') for v in results['fine_sites'].values()):claim='E3_SITE_LOCALIZATION_BASE_REVISION_HOLD'
elif any(v.get('survivor') for v in results['fine_layers'].values()) or any(v.get('survivor') for v in results['fine_modules'].values()):claim='E3_LAYER_OR_MODULE_LOCALIZATION_BASE_REVISION_HOLD'
else:claim='E3_BROAD_LOCALIZATION_ONLY_BASE_REVISION_HOLD'
results['max_supported_claim']=claim
results['e4_claim']=False;results['e5_claim']=False;results['algorithm_claim']=False;results['student_claim']=False;results['promotion_claim']=False
(OUT/'C78_CAUSAL_RESULT.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
print(json.dumps({'broad_survivors':results.get('broad_survivors',[]),'layer_survivors':results.get('layer_survivors',[]),'module_survivors':results.get('module_survivors',[]),'site_survivors':results.get('site_survivors',[]),'claim':claim},indent=2))
