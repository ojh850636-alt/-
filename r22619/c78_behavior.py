import contextlib, hashlib, json, math, os, random, re, statistics
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path('.')
OUT = Path('out'); OUT.mkdir(exist_ok=True)
ADAPTER = Path('src_adapter')
BASE = Path('src_base')
SEED = 22619
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

FIELDS = ['job_title','company','location','employment_type','salary','experience_years','skills']
TITLES = ['Data Analyst','ML Engineer','Backend Developer','Product Designer','Security Analyst','Research Scientist','QA Engineer','DevOps Engineer','Data Engineer','Technical Writer']
COMPANIES = ['Orion Works','Northstar Labs','Blue Cedar','Atlas Systems','Nova Forge','Redwood Logic','Cobalt Cloud','Helix Dynamics','Juniper AI','Meridian Tech']
LOCATIONS = ['Toronto, Canada','Berlin, Germany','Austin, USA','Singapore','Dublin, Ireland','Tokyo, Japan','Paris, France','Sydney, Australia','Seoul, South Korea','Amsterdam, Netherlands']
EMP = ['full-time','part-time','contract','temporary']
SAL = ['$70,000-$90,000','$95,000','$55/hour','€65,000-€80,000',None]
SKILLS = [['Python','SQL'],['PyTorch','Python'],['Go','PostgreSQL'],['Figma','UX Research'],['SIEM','Python'],['NLP','PyTorch'],['Playwright','Python'],['Kubernetes','Terraform'],['Spark','SQL'],['Technical Writing','Git']]


def canon(truth):
    return json.dumps({k: truth.get(k) for k in FIELDS}, separators=(',', ':'), ensure_ascii=False)


def make_case(i, split):
    j = i % 10
    title, comp, loc = TITLES[j], COMPANIES[(i*3)%10], LOCATIONS[(i*7)%10]
    emp = EMP[i % len(EMP)]
    salary = SAL[(i*2) % len(SAL)]
    exp = None if i % 4 == 0 else (i % 7) + 1
    skills = SKILLS[j]
    # deterministic missing-field families; truth only includes explicitly stated facts
    fam = i % 8
    parts = [f'{comp} is hiring a {title}.', f'Location: {loc}.', f'This is a {emp} role.']
    truth = {'job_title':title,'company':comp,'location':loc,'employment_type':emp,'salary':salary,'experience_years':exp,'skills':skills}
    missing=[]
    if fam in (0,4): salary=None; truth['salary']=None; missing.append('salary')
    else: parts.append(f'Compensation: {salary}.')
    if fam in (1,4,6): exp=None; truth['experience_years']=None; missing.append('experience_years')
    else: parts.append(f'Experience required: {exp} years.')
    if fam == 2: truth['location']=None; missing.append('location'); parts=[p for p in parts if not p.startswith('Location:')]
    if fam == 3: truth['employment_type']=None; missing.append('employment_type'); parts=[p for p in parts if ' role.' not in p]
    if fam == 5: truth['skills']=None; missing.append('skills')
    else: parts.append('Core skills: ' + ', '.join(skills) + '.')
    if split == 'OOD':
        # order and surface-form shift only; no answer change
        parts = list(reversed(parts))
        parts.insert(1, f'Internal requisition JR-{5000+i}; applications close Friday.')
    elif split == 'CONFIRMATION':
        parts.append(f'Reference code JR-{3000+i}; benefits are described separately.')
    else:
        parts.append(f'Reference code JR-{2000+i}.')
    posting=' '.join(parts)
    prompt=('Extract only explicitly stated job information from the posting. Return one JSON object with exactly these keys: '
            'job_title, company, location, employment_type, salary, experience_years, skills. '
            'Use null for any field not stated. Do not infer missing values.\nPOSTING:\n'+posting+'\nJSON:')
    return {'id':f'C78_{i:03d}','split':split,'family':f'F{fam}','posting':posting,'prompt':prompt,'truth':truth,'missing_fields':missing,'canonical_target':canon(truth)}


def build_suite():
    cases=[]
    for i in range(240):
        split='DISCOVERY' if i < 80 else ('CONFIRMATION' if i < 160 else 'OOD')
        cases.append(make_case(i,split))
    neg_texts=[]
    for i in range(24):
        neg_texts.append(f'{COMPANIES[i%10]} announced a community event in {LOCATIONS[(i*3)%10]}. No hiring information is provided. Bulletin {9000+i}.')
    for i,text in enumerate(neg_texts, start=240):
        truth={k:None for k in FIELDS}
        prompt=('Extract only explicitly stated job information from the posting. Return one JSON object with exactly these keys: '
                'job_title, company, location, employment_type, salary, experience_years, skills. '
                'Use null for any field not stated. Do not infer missing values.\nPOSTING:\n'+text+'\nJSON:')
        cases.append({'id':f'C78_{i:03d}','split':'NEGATIVE','family':'NEGATIVE','posting':text,'prompt':prompt,'truth':truth,'missing_fields':FIELDS,'canonical_target':canon(truth)})
    return cases

SUITE=build_suite()
PRESEAL={
 'schema':'R22619_C78_BEHAVIOR_PRESEAL_V2','model_outputs_observed_at_seal':0,
 'candidate':'damienlusson/gemma-3-270m-it-unsloth-bnb-4bit-jobs-extraction-LoRA','candidate_revision':'63751205b2c41bc2ae8dfe3c03b8c4932ed21e4e',
 'declared_base':'unsloth/gemma-3-270m-it-unsloth-bnb-4bit','declared_base_revision':'ce8f5b69f631273f9c6007972240d4cee9577bc6',
 'training_base_revision_proven':False,'mandatory_claim_suffix':'_BASE_REVISION_HOLD',
 'scientific_question':'Does the LoRA causally increase schema-grounded extraction of explicitly stated job attributes while increasing or suppressing hallucinated values for missing/non-job fields?',
 'case_counts':{'labeled':240,'negative':24,'discovery':80,'confirmation':80,'ood':80},
 'teacher_forced_coverage':{'BASE_FULL':'all_240_labeled','RANDOM_SHUFFLE':'first40_CONFIRMATION_plus_first40_OOD','DOSE':'first40_CONFIRMATION_plus_first40_OOD'},
 'generation_coverage':{'labeled':'first20_each_DISCOVERY_CONFIRMATION_OOD','negative':'all24'},
 'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0.25','DOSE_0.5','DOSE_1.5'],
 'positive_gate':'On locked Confirmation+OOD controls: mean(FULL-BASE)>0, mean(FULL-RANDOM)>0, mean(FULL-SHUFFLE)>0 and deterministic paired bootstrap 95% CI lower bound >0 for all three.',
 'safety_gate':'Missing-field and non-job false activation evaluated independently; safety RED blocks Student/promotion but not failure/capability archaeology.',
 'claim_ceiling':'E3_BASE_REVISION_HOLD; E4/E5/algorithm/student/promotion forbidden this round without additional evidence.',
 'suite_sha256':hashlib.sha256(json.dumps(SUITE,sort_keys=True,separators=(',',':')).encode()).hexdigest()
}
Path('preseal.json').write_text(json.dumps(PRESEAL,indent=2,sort_keys=True)+'\n')
Path('suite.json').write_text(json.dumps(SUITE,indent=2,sort_keys=True)+'\n')


def render_prompt(tok, prompt):
    msgs=[{'role':'user','content':prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def score_one(model,tok,case,disabled=False):
    rendered=render_prompt(tok,case['prompt'])
    pids=tok(rendered,add_special_tokens=False).input_ids
    tids=tok(case['canonical_target'],add_special_tokens=False).input_ids
    ids=torch.tensor([pids+tids],dtype=torch.long)
    am=torch.ones_like(ids)
    ctx=model.disable_adapter() if disabled else contextlib.nullcontext()
    with ctx, torch.inference_mode():
        logits=model(input_ids=ids,attention_mask=am,use_cache=False).logits
        start=len(pids)
        sl=logits[0,start-1:len(pids)+len(tids)-1,:].float()
        targets=ids[0,start:start+len(tids)]
        lp=torch.log_softmax(sl,dim=-1).gather(-1,targets[:,None]).squeeze(-1)
        val=float(lp.mean().item())
    del logits,sl,lp,ids,am
    return val


def bootstrap_ci(vals, seed=22619, n=4000):
    a=np.asarray(vals,dtype=np.float64)
    if len(a)==0:return [None,None]
    rng=np.random.default_rng(seed)
    means=np.empty(n)
    for i in range(n): means[i]=a[rng.integers(0,len(a),len(a))].mean()
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def lora_params(model):
    rx=re.compile(r'.*layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.lora_([AB])\.default\.weight$')
    d={}
    for n,p in model.named_parameters():
        m=rx.match(n)
        if m:d[n]={'p':p,'layer':int(m.group(1)),'module':m.group(2),'ab':m.group(3)}
    return d


def snapshot(lp): return {n:v['p'].detach().cpu().clone() for n,v in lp.items()}

def restore(lp,orig):
    with torch.no_grad():
        for n,v in lp.items(): v['p'].copy_(orig[n].to(v['p'].device,dtype=v['p'].dtype))

def set_dose(lp,orig,dose):
    with torch.no_grad():
        for n,v in lp.items():
            src=orig[n]
            if v['ab']=='B': src=src*dose
            v['p'].copy_(src.to(v['p'].device,dtype=v['p'].dtype))

def set_random(lp,orig):
    g=torch.Generator(device='cpu'); g.manual_seed(SEED+11)
    with torch.no_grad():
        for n,v in lp.items():
            src=orig[n].float(); r=torch.randn(src.shape,generator=g,dtype=torch.float32)
            rn=torch.linalg.vector_norm(r); sn=torch.linalg.vector_norm(src)
            r=r*(sn/(rn+1e-12))
            v['p'].copy_(r.to(v['p'].device,dtype=v['p'].dtype))

def set_shuffle(lp,orig):
    rng=random.Random(SEED+29)
    groups={}
    for n,v in lp.items(): groups.setdefault((v['module'],v['ab']),[]).append((v['layer'],n))
    mapping={}
    for key,items in groups.items():
        items=sorted(items); donors=[n for l,n in items]; perm=donors[:]; rng.shuffle(perm)
        if all(a==b for a,b in zip(donors,perm)): perm=perm[1:]+perm[:1]
        for (_,dest),src in zip(items,perm): mapping[dest]=src
    with torch.no_grad():
        for n,v in lp.items(): v['p'].copy_(orig[mapping[n]].to(v['p'].device,dtype=v['p'].dtype))


def parse_json(text):
    a=text.find('{'); b=text.rfind('}')
    if a<0 or b<a:return None
    try:
        o=json.loads(text[a:b+1]); return o if isinstance(o,dict) else None
    except Exception:return None


def norm_field(k,v):
    if v is None:return None
    if k=='skills':
        if not isinstance(v,list):return v
        return sorted(str(x).strip().lower() for x in v)
    if isinstance(v,str):return re.sub(r'\s+',' ',v.strip().lower())
    return v


def generate_one(model,tok,case,disabled=False):
    rendered=render_prompt(tok,case['prompt'])
    enc=tok(rendered,return_tensors='pt',add_special_tokens=False)
    ctx=model.disable_adapter() if disabled else contextlib.nullcontext()
    with ctx,torch.inference_mode():
        out=model.generate(**enc,max_new_tokens=128,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new=out[0,enc['input_ids'].shape[1]:]
    text=tok.decode(new,skip_special_tokens=True)
    obj=parse_json(text)
    truth=case['truth']
    strict=obj is not None and set(obj.keys())==set(FIELDS)
    exact= strict and all(norm_field(k,obj.get(k))==norm_field(k,truth.get(k)) for k in FIELDS)
    field_hits=sum(1 for k in FIELDS if obj is not None and norm_field(k,obj.get(k))==norm_field(k,truth.get(k)))
    missing=[k for k in FIELDS if truth.get(k) is None]
    missing_fp=sum(1 for k in missing if obj is not None and obj.get(k) is not None)
    any_nonnull= bool(obj is not None and any(obj.get(k) is not None for k in FIELDS))
    return {'id':case['id'],'split':case['split'],'family':case['family'],'strict_json':strict,'record_exact':exact,'field_hits':field_hits,'field_total':len(FIELDS),'missing_n':len(missing),'missing_fp':missing_fp,'any_nonnull':any_nonnull,'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'output_chars':len(text)}


def main():
    # verify source identities again locally on runner
    ah=hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest()
    bh=hashlib.sha256((BASE/'model.safetensors').read_bytes()).hexdigest()
    assert ah=='414f0d5db8d6560a12f5841eb4a7fdc8ce558db37f4fe4e1c3c19a249223f918'
    assert bh=='d3ae9628a0ed2cfa21b5b8345be4e1d4126794e734f3483dcc275b4735bf5a15'
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True,trust_remote_code=False,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,trust_remote_code=False,device_map={'':'cpu'})
    model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=False,local_files_only=True)
    model.eval()
    lp=lora_params(model); assert len(lp)==252, len(lp)
    orig=snapshot(lp)
    labeled=[c for c in SUITE if c['split']!='NEGATIVE']
    locked=[c for c in SUITE if c['split']=='CONFIRMATION'][:40]+[c for c in SUITE if c['split']=='OOD'][:40]
    scores={}
    # BASE/FULL all labeled
    scores['BASE']={c['id']:score_one(model,tok,c,disabled=True) for c in labeled}
    restore(lp,orig)
    scores['FULL']={c['id']:score_one(model,tok,c,disabled=False) for c in labeled}
    # locked random/shuffle/dose
    set_random(lp,orig); scores['RANDOM_RANK_MATCHED']={c['id']:score_one(model,tok,c) for c in locked}
    set_shuffle(lp,orig); scores['LAYER_SHUFFLED']={c['id']:score_one(model,tok,c) for c in locked}
    for dose in (0.25,0.5,1.5):
        set_dose(lp,orig,dose); scores[f'DOSE_{dose}']={c['id']:score_one(model,tok,c) for c in locked}
    restore(lp,orig)
    # summarize paired deltas
    full=scores['FULL']; base_s=scores['BASE']
    def dsummary(a,b,ids,seed):
        vals=[a[i]-b[i] for i in ids]
        return {'n':len(vals),'mean':float(np.mean(vals)),'ci95':bootstrap_ci(vals,seed),'median':float(np.median(vals)),'positive_fraction':float(np.mean(np.asarray(vals)>0))}
    ids_all=[c['id'] for c in labeled]; ids_lock=[c['id'] for c in locked]
    summary={'FULL_MINUS_BASE_ALL':dsummary(full,base_s,ids_all,1)}
    for split in ('DISCOVERY','CONFIRMATION','OOD'):
        ids=[c['id'] for c in labeled if c['split']==split]
        summary[f'FULL_MINUS_BASE_{split}']=dsummary(full,base_s,ids,10+len(ids))
    summary['FULL_MINUS_BASE_LOCKED']=dsummary(full,base_s,ids_lock,20)
    summary['FULL_MINUS_RANDOM_LOCKED']=dsummary(full,scores['RANDOM_RANK_MATCHED'],ids_lock,21)
    summary['FULL_MINUS_SHUFFLE_LOCKED']=dsummary(full,scores['LAYER_SHUFFLED'],ids_lock,22)
    for dose in (0.25,0.5,1.5): summary[f'DOSE_{dose}_MINUS_BASE_LOCKED']=dsummary(scores[f'DOSE_{dose}'],base_s,ids_lock,30+int(dose*10))
    # family deltas all labeled
    fam={}
    for f in sorted(set(c['family'] for c in labeled)):
        ids=[c['id'] for c in labeled if c['family']==f]
        fam[f]=dsummary(full,base_s,ids,100+int(f[1:]))
    # generation locked subset: 20 each labeled split + all negatives
    gens=[c for c in SUITE if c['split']=='DISCOVERY'][:20]+[c for c in SUITE if c['split']=='CONFIRMATION'][:20]+[c for c in SUITE if c['split']=='OOD'][:20]+[c for c in SUITE if c['split']=='NEGATIVE']
    gen_base=[generate_one(model,tok,c,disabled=True) for c in gens]
    gen_full=[generate_one(model,tok,c,disabled=False) for c in gens]
    def gsum(rows):
        lab=[r for r in rows if r['split']!='NEGATIVE']; neg=[r for r in rows if r['split']=='NEGATIVE']
        return {'labeled_n':len(lab),'record_exact':sum(r['record_exact'] for r in lab),'strict_json':sum(r['strict_json'] for r in lab),'field_accuracy':sum(r['field_hits'] for r in lab)/max(1,sum(r['field_total'] for r in lab)),'missing_field_fpr':sum(r['missing_fp'] for r in lab)/max(1,sum(r['missing_n'] for r in lab)),'negative_n':len(neg),'negative_false_activation':sum(r['any_nonnull'] for r in neg),'negative_strict_json':sum(r['strict_json'] for r in neg)}
    control_green=all(summary[k]['ci95'][0] is not None and summary[k]['ci95'][0]>0 for k in ('FULL_MINUS_BASE_LOCKED','FULL_MINUS_RANDOM_LOCKED','FULL_MINUS_SHUFFLE_LOCKED'))
    result={'schema':'R22619_C78_BEHAVIOR_RESULT_V1','preseal':PRESEAL,'source':{'adapter_sha256':ah,'base_sha256':bh},'condition_summaries':summary,'family_full_minus_base':fam,'generation':{'BASE':gsum(gen_base),'FULL':gsum(gen_full)},'control_separated_positive_green':control_green,'claim_suffix':'_BASE_REVISION_HOLD','max_claim_if_no_further_evidence':'E2_CONTROL_SEPARATED_BEHAVIOR_BASE_REVISION_HOLD' if control_green else 'E2_OR_BELOW','raw_model_outputs_retained':0,'case_scalar_scores':scores,'generation_scalar_rows':{'BASE':gen_base,'FULL':gen_full}}
    (OUT/'C78_BEHAVIOR_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (OUT/'C78_PRESEAL.json').write_text(json.dumps(PRESEAL,indent=2,sort_keys=True)+'\n')
    # raw-free completion before delivery
    print(json.dumps({'control_green':control_green,'summary':summary,'generation':result['generation']},indent=2))

if __name__=='__main__': main()
