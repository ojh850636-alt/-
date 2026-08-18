from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, random, re, shutil, statistics, sys, time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
sys.dont_write_bytecode=True
SEED=22573
ADAPTER_REPO='waliaMuskaan011/calendar-event-extractor-smollm'
ADAPTER_REV='08d0bd53801b5bf035b44ff4ae084c94a51126ee'
ADAPTER_SHA='31340a2b7b846836279df0966c14ccab0dcca69c25eb2a4c9927c3bf2eda20ac'
BASE_REPO='HuggingFaceTB/SmolLM-360M'
BASE_REV='59f7ef243ee09a72cbc14cb054393a3e3b771d41'
BASE_SHA='e91f05d8506ee5efbd8c0fbfc1799c49af2b2f2cce824bc2d801d5af2a716cc2'
FIELDS=['action','date','time','attendees','location','duration','recurrence','notes']
INSTR='Extract calendar fields from: "{event}".\nReturn ONLY valid JSON with keys [action,date,time,attendees,location,duration,recurrence,notes].\nUse null for unknown.'

def jd(x): return json.dumps(x,separators=(',',':'),ensure_ascii=False)
def sha_obj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def clean(s): return re.sub(r'\s+',' ',str(s or '').strip().lower())
def target(**kw): return {k:kw.get(k) for k in FIELDS}

def build_suite():
    cases=[]
    def add(part,event,tgt,ref=None,hint=None,contract='absolute'):
        prompt=INSTR.format(event=event)
        if ref is not None:
            prompt=f'Reference date: {ref.strftime("%d/%m/%Y")}\n'+prompt
        if hint is not None:
            prompt=f'Reference date: {ref.strftime("%d/%m/%Y")}\nResolved date hint: {hint}\n'+INSTR.format(event=event)
        cases.append({'id':f'C57-{len(cases):03d}','partition':part,'event':event,'prompt':prompt,'target':tgt,'contract':contract,'reference_date':ref.isoformat() if ref else None,'hint':hint})
    abs_specs=[
      ('meeting','Alice','conference room A','10 May 2026','2pm','1 hour',date(2026,5,10),'02:00 PM'),
      ('call','Bob','Zoom','2026-06-03','14:30','45 minutes',date(2026,6,3),'02:30 PM'),
      ('interview','Carol','main office','07/07/2026','9:15 AM','30 minutes',date(2026,7,7),'09:15 AM'),
      ('lunch','David','local cafe','August 21, 2026','12pm','90 minutes',date(2026,8,21),'12:00 PM'),
      ('review','Emily','meeting room B','2026-09-30','16:45','2 hours',date(2026,9,30),'04:45 PM'),
      ('workshop','Frank','auditorium','11 October 2026','10am','3 hours',date(2026,10,11),'10:00 AM'),
      ('demo','Grace','Google Meet','12/11/2026','11:30am','45 mins',date(2026,11,12),'11:30 AM'),
      ('appointment','Helen','clinic','December 5, 2026','8am','30 mins',date(2026,12,5),'08:00 AM'),
    ]
    for i in range(24):
        a,n,loc,ds,ts,dur,dv,tv=abs_specs[i%len(abs_specs)]
        note=None if i%3 else 'important'
        event=f'{a} with {n} at {loc} on {ds} at {ts} for {dur}'+(' important' if note else '')
        add('absolute',event,target(action=a,date=dv.strftime('%d/%m/%Y'),time=tv,attendees=[n],location=loc,duration=dur,recurrence=None,notes=note),contract='absolute')
    rel_specs=[('tomorrow','meeting','Alice','office','2pm','1 hour','02:00 PM'),('next Monday','call','Bob','Zoom','9am','30 minutes','09:00 AM'),('this Friday','review','Carol','boardroom','3:30pm','45 minutes','03:30 PM')]
    for i in range(12):
        phrase,a,n,loc,ts,dur,tv=rel_specs[i%3]
        event=f'{a} with {n} {phrase} at {ts} in {loc} for {dur}'
        add('relative_literal',event,target(action=a,date=phrase,time=tv,attendees=[n],location=loc,duration=dur,recurrence=None,notes=None),contract='relative_literal')
    refs=[date(2026,1,31),date(2026,2,27),date(2026,3,30),date(2026,12,31),date(2027,1,15),date(2028,2,28)]
    for i in range(12):
        ref=refs[i%len(refs)];a='meeting' if i%2==0 else 'call';n='Alice' if i%2==0 else 'Bob';loc='office' if i%2==0 else 'Zoom';ts='2pm' if i%2==0 else '9am';tv='02:00 PM' if i%2==0 else '09:00 AM';dur='1 hour' if i%2==0 else '30 minutes';resolved=ref+timedelta(days=1)
        event=f'{a} with {n} tomorrow at {ts} in {loc} for {dur}'
        add('relative_reference',event,target(action=a,date=resolved.strftime('%d/%m/%Y'),time=tv,attendees=[n],location=loc,duration=dur,recurrence=None,notes=None),ref=ref,contract='relative_reference')
    srefs=[date(2026,4,30),date(2026,6,30),date(2026,9,30),date(2026,12,30)]
    for i in range(8):
        ref=srefs[i%4];resolved=ref+timedelta(days=1);a='meeting';n='Taylor';loc='conference room A';tv='10:00 AM';dur='45 minutes';event=f'{a} with {n} tomorrow at 10am in {loc} for {dur}';hint=f'tomorrow = {resolved.strftime("%d/%m/%Y")}'
        add('scaffolded_reference',event,target(action=a,date=resolved.strftime('%d/%m/%Y'),time=tv,attendees=[n],location=loc,duration=dur,recurrence=None,notes=None),ref=ref,hint=hint,contract='scaffolded_reference')
    neg=['Order number 4500 is pending','OTP 660044 for verification','Flight 602 status is delayed','Invoice 1200 needs review','Room 305 is unavailable','Ticket 2207 is a support reference','Product model 500 is discontinued','Account reference 7001 needs update']
    for s in neg:add('hard_negative',s,target(action=None,date=None,time=None,attendees=None,location=None,duration=None,recurrence=None,notes=None),contract='hard_negative')
    assert len(cases)==64
    return cases

def parse_json_text(s):
    s=str(s or '').strip()
    for cand in re.findall(r'\{.*?\}',s,re.S):
        try:
            o=json.loads(cand)
            if isinstance(o,dict) and any(k in o for k in FIELDS):return {k:o.get(k) for k in FIELDS}
        except Exception:pass
    try:
        o=json.loads(s);return {k:o.get(k) for k in FIELDS} if isinstance(o,dict) else None
    except Exception:return None

def norm_date(v):
    if v is None:return None
    s=clean(v)
    if s in {'tomorrow','next monday','this friday','today'}:return s
    raw=str(v).strip()
    for f in ['%d/%m/%Y','%Y-%m-%d','%B %d, %Y','%d %B %Y','%m/%d/%Y','%d-%m-%Y']:
        try:return datetime.strptime(raw,f).date().isoformat()
        except Exception:pass
    return s

def norm_time(v):
    if v is None:return None
    raw=str(v).strip().upper().replace('.','')
    for f in ['%I:%M %p','%I%p','%I:%M%p','%H:%M']:
        try:return datetime.strptime(raw.replace('  ',' '),f).strftime('%H:%M')
        except Exception:pass
    return clean(v)
def norm_duration(v):
    if v is None:return None
    s=clean(v).replace('hrs','hours').replace('hr','hour').replace('mins','minutes').replace('min','minute')
    m=re.match(r'^(\d+(?:\.\d+)?)\s*(hour|hours|minute|minutes)$',s)
    if not m:return s
    x=float(m.group(1));mins=x*60 if m.group(2).startswith('hour') else x
    return str(int(mins) if mins.is_integer() else mins)+'m'
def norm_list(v):
    if v is None:return None
    if isinstance(v,list):return sorted(clean(x) for x in v)
    return [clean(v)]
def field_score(pred,gold,field):
    pv=pred.get(field) if pred else None;gv=gold.get(field)
    if field=='date':return float(norm_date(pv)==norm_date(gv))
    if field=='time':return float(norm_time(pv)==norm_time(gv))
    if field=='duration':return float(norm_duration(pv)==norm_duration(gv))
    if field=='attendees':return float(norm_list(pv)==norm_list(gv))
    return float((clean(pv) if pv is not None else None)==(clean(gv) if gv is not None else None))
def score_case(pred,gold):
    if pred is None:return {'json_valid':0,'semantic':0,'date':0,'time':0,'action':0,'attendees':0,'location':0,'duration':0,'recurrence':0,'false_event':0 if any(gold.values()) else 1}
    fs={f:field_score(pred,gold,f) for f in ['date','time','action','attendees','location','duration','recurrence']};weights={'date':.30,'time':.15,'action':.15,'attendees':.10,'location':.10,'duration':.15,'recurrence':.05};sem=sum(fs[k]*weights[k] for k in weights);false_event=int(not any(gold.values()) and any(pred.get(k) not in (None,'',[],{}) for k in FIELDS));return {'json_valid':1,'semantic':sem,'false_event':false_event,**fs}
def aggregate(rows):
    if not rows:return {}
    return {k:sum(float(r[k]) for r in rows)/len(rows) for k in rows[0]}
def bootstrap_ci(vals,B=1000,seed=SEED):
    if not vals:return [0,0]
    rng=random.Random(seed);n=len(vals);xs=[sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(B)];xs.sort();return [xs[int(.025*B)],xs[min(B-1,int(.975*B))]]
def tier_ids(cases,name):
    quotas={'LARGE':{'absolute':24,'relative_literal':12,'relative_reference':12,'scaffolded_reference':8,'hard_negative':8},'MEDIUM':{'absolute':16,'relative_literal':8,'relative_reference':8,'scaffolded_reference':8,'hard_negative':8},'SMALL':{'absolute':12,'relative_literal':6,'relative_reference':6,'scaffolded_reference':4,'hard_negative':4}}[name];out=[]
    for part,q in quotas.items():out.extend([c['id'] for c in cases if c['partition']==part][:q])
    return out
def generation_ids(cases,name):
    by={p:[c['id'] for c in cases if c['partition']==p] for p in ['absolute','relative_literal','relative_reference','scaffolded_reference','hard_negative']}
    if name=='LARGE':return by['absolute'][:4]+by['relative_literal'][:3]+[by['relative_reference'][0],by['relative_reference'][2],by['relative_reference'][4],by['relative_reference'][1]]+by['scaffolded_reference'][:3]+by['hard_negative'][:2]
    if name=='MEDIUM':return by['absolute'][:3]+by['relative_literal'][:2]+[by['relative_reference'][0],by['relative_reference'][2],by['relative_reference'][4]]+by['scaffolded_reference'][:2]+by['hard_negative'][:2]
    return by['absolute'][:2]+by['relative_literal'][:1]+[by['relative_reference'][0],by['relative_reference'][2]]+by['scaffolded_reference'][:1]+by['hard_negative'][:2]

def preseal(out):
    out.mkdir(parents=True,exist_ok=True);cases=build_suite();perfect=[score_case(c['target'],c['target']) for c in cases];null=[score_case(None,c['target']) for c in cases];date_mut=[]
    for c in cases:
        p=dict(c['target'])
        if p.get('date') and re.match(r'\d{2}/\d{2}/\d{4}',str(p['date'])):p['date']=(datetime.strptime(p['date'],'%d/%m/%Y').date()+timedelta(days=1)).strftime('%d/%m/%Y')
        elif p.get('date')=='tomorrow':p['date']='today'
        date_mut.append(score_case(p,c['target']))
    tiers={n:{'ids':tier_ids(cases,n)} for n in ['LARGE','MEDIUM','SMALL']};tiers['LARGE'].update({'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p50','DOSE_1p50'],'generation_ids':generation_ids(cases,'LARGE'),'generation_n':16,'causal_n':12});tiers['MEDIUM'].update({'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED'],'generation_ids':generation_ids(cases,'MEDIUM'),'generation_n':12,'causal_n':8});tiers['SMALL'].update({'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED'],'generation_ids':generation_ids(cases,'SMALL'),'generation_n':8,'causal_n':8})
    rec={'schema':'R22573_C57_PREOUTPUT_SEAL_V1','candidate':'C57','weights_present':0,'model_outputs_observed':0,'training_rows_used':0,'scientific_question':'Does the SmolLM calendar LoRA add control-separated calendar semantic transduction while distinguishing absolute-date normalization from relative-date literal preservation, and does it show genuine reference-date-sensitive resolution rather than training-time temporal anchoring or scaffold copying?','suite_n':len(cases),'suite_sha256':sha_obj(cases),'partitions':dict(Counter(c['partition'] for c in cases)),'tiers':tiers,'primary_behavior_gate':'paired target-NLL Full gain over Base CI low>0.01; Full beats random and shuffle by >=0.01 nats/token; generation absolute semantic gain>=0.05; hard-negative FP increase<=0.10','dynamic_temporal_gate':'on relative_reference generation: Full date accuracy gain over Base>=0.20 and Full date accuracy>=0.50; paired target-NLL gain positive; same utterance under changed reference dates must change to corresponding exact dates','scaffold_diagnostic':'scaffolded_reference date gain substantially exceeds relative_reference date gain while minimal reference resolution remains weak => scaffold-copy candidate, not dynamic temporal reasoning','failure_gate':'hard-negative FP increase>=0.20 with Full worse than Base and learned-control target-NLL separation, or control-separated semantic degradation<=-0.10','causal_if_admitted':['MINUS_ATTENTION','MINUS_MLP','MINUS_Q','MINUS_K','MINUS_V','MINUS_O','MINUS_GATE','MINUS_UP','MINUS_DOWN','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE'],'causal_E3_gate':'on two disjoint halves, same group removes >=50% of admitted Full-vs-Base target-NLL gain and absolute loss-of-gain>=0.02 nats/token','workload_selection':'binary-free exact-architecture benchmark chooses largest presealed tier whose worst-case estimate including causal is <=65% of 45-minute job budget; if none, abort before weights','verifier_smoke':{'perfect':aggregate(perfect),'null_semantic':aggregate(null).get('semantic',0),'date_mutation_semantic':aggregate(date_mut).get('semantic',0)},'source_contract_warning':'current public training-source lineage contains both literal-relative few-shot targets (date=tomorrow) and enhanced-generator paths that can pair relative text with absolute dates; exact source commit is not cryptographically bound to adapter revision.','claim_boundary':'fresh synthetic calendar assay only; no scheduling/compliance guarantee; current public source code is contextual lineage, not proof of exact training rows.'};(out/'C57_PREOUTPUT_SEAL.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2));print(json.dumps(rec,indent=2))

def runtime_smoke(out):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    torch.set_grad_enabled(False);cfg=AutoConfig.from_pretrained(BASE_REPO,revision=BASE_REV,trust_remote_code=False);model=AutoModelForCausalLM.from_config(cfg);model.eval();lc=LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM');p=get_peft_model(model,lc);p.eval();ids=['control_random','control_shuffle','dose_0p50','dose_1p50','minus_q','minus_early']
    for n in ids:p.add_adapter(n,lc);p.set_adapter(n);p.delete_adapter(n)
    x=torch.randint(3,int(cfg.vocab_size),(4,96));mask=torch.ones_like(x);_=p(input_ids=x,attention_mask=mask).logits;t0=time.perf_counter();_=p(input_ids=x,attention_mask=mask).logits;forward_s=time.perf_counter()-t0
    pad=0;a=torch.tensor([11,12,13,14,15]);b=torch.tensor([21,22,23]);batch=torch.tensor([[11,12,13,14,15],[pad,pad,21,22,23]]);am=torch.tensor([[1,1,1,1,1],[0,0,1,1,1]]);genkw=dict(max_new_tokens=4,do_sample=False,pad_token_id=pad,eos_token_id=None);t0=time.perf_counter();gb=p.generate(input_ids=batch,attention_mask=am,**genkw);gen_s=time.perf_counter()-t0;ga=p.generate(input_ids=a.unsqueeze(0),attention_mask=torch.ones(1,len(a),dtype=torch.long),**genkw);gc=p.generate(input_ids=b.unsqueeze(0),attention_mask=torch.ones(1,len(b),dtype=torch.long),**genkw);eq1=torch.equal(gb[0,-4:],ga[0,-4:]);eq2=torch.equal(gb[1,-4:],gc[0,-4:])
    tiers=json.loads((out/'C57_PREOUTPUT_SEAL.json').read_text())['tiers'];estimates={}
    for name,t in tiers.items():
        n=len(t['ids']);conds=len(t['conditions']);g=t['generation_n'];ca=t['causal_n'];estimates[name]=300+math.ceil(n/4)*conds*forward_s*1.7+math.ceil(ca/4)*12*forward_s*1.7+math.ceil(g/2)*2*gen_s*(96/4)*1.35
    budget=45*60*.65;selected=next((n for n in ['LARGE','MEDIUM','SMALL'] if estimates[n]<=budget),None);rec={'schema':'R22573_C57_RUNTIME_PREFLIGHT_V1','pass':bool(selected and eq1 and eq2),'weights_present':0,'forward_b4_l96_seconds':forward_s,'generation_b2_new4_seconds':gen_s,'left_padding_batch_single_equivalence':[bool(eq1),bool(eq2)],'tier_estimates_seconds':estimates,'budget_seconds':budget,'selected_tier':selected,'identifiers':ids};out.mkdir(exist_ok=True);(out/'C57_RUNTIME_PREFLIGHT.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2));
    if not rec['pass']:raise SystemExit('RUNTIME_PREFLIGHT_RED')

def execute(out,work):
    import torch,torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file
    torch.set_grad_enabled(False);torch.manual_seed(SEED);random.seed(SEED);pre=json.loads((out/'C57_PREOUTPUT_SEAL.json').read_text());rt=json.loads((out/'C57_RUNTIME_PREFLIGHT.json').read_text());tier=rt['selected_tier'];T=pre['tiers'][tier];allcases=build_suite();selected=[c for c in allcases if c['id'] in set(T['ids'])];raw=work/'raw';base_dir=raw/'base';ad_dir=raw/'adapter';base_dir.mkdir(parents=True);ad_dir.mkdir(parents=True)
    for fn in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
        try:src=hf_hub_download(BASE_REPO,filename=fn,revision=BASE_REV);shutil.copy2(src,base_dir/fn)
        except Exception:
            if fn in {'config.json','model.safetensors','tokenizer.json','tokenizer_config.json'}:raise
    for fn in ['adapter_config.json','adapter_model.safetensors']:
        src=hf_hub_download(ADAPTER_REPO,filename=fn,revision=ADAPTER_REV);shutil.copy2(src,ad_dir/fn)
    assert hashlib.sha256((base_dir/'model.safetensors').read_bytes()).hexdigest()==BASE_SHA;assert hashlib.sha256((ad_dir/'adapter_model.safetensors').read_bytes()).hexdigest()==ADAPTER_SHA;ac=json.loads((ad_dir/'adapter_config.json').read_text());assert ac['base_model_name_or_path']==BASE_REPO and int(ac['r'])==16
    st=load_file(str(ad_dir/'adapter_model.safetensors'),device='cpu');pairmap=defaultdict(dict);energy=defaultdict(float);zero=nonfinite=0
    for k,t in st.items():
        zero+=int(float(t.float().norm())==0);nonfinite+=int(not torch.isfinite(t).all());fam=next((x for x in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if x in k),'other');energy[fam]+=float(t.float().pow(2).sum());m=re.sub(r'\.lora_[AB]\.weight$','',k);side='A' if '.lora_A.' in k else 'B' if '.lora_B.' in k else None
        if side:pairmap[m][side]=t
    total=sum(energy.values()) or 1;static={'schema':'R22573_C57_STATIC_FORENSICS_V1','tensor_count':len(st),'complete_pairs':sum(set(v)=={'A','B'} for v in pairmap.values()),'zero_tensor_count':zero,'nonfinite_tensor_count':nonfinite,'rank_config':ac['r'],'alpha':ac['lora_alpha'],'target_modules':sorted(ac['target_modules']),'energy_proxy_by_family':{k:v/total for k,v in sorted(energy.items())},'evidence_grade':'E1_STATIC_ONLY'};(out/'C57_STATIC_FORENSICS.json').write_text(json.dumps(static,indent=2));del st
    tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token;base=AutoModelForCausalLM.from_pretrained(base_dir,local_files_only=True,dtype=torch.float32,device_map='cpu');base.eval();peft=PeftModel.from_pretrained(base,ad_dir,is_trainable=False);peft.eval();smoke_prompts=[selected[0]['prompt'],selected[1]['prompt']];enc=tok(smoke_prompts,return_tensors='pt',padding=True);bg=peft.generate(**enc,max_new_tokens=4,do_sample=False,pad_token_id=tok.pad_token_id);eq=[]
    for i,s in enumerate(smoke_prompts):
        e=tok(s,return_tensors='pt');sg=peft.generate(**e,max_new_tokens=4,do_sample=False,pad_token_id=tok.pad_token_id);eq.append(bool(torch.equal(bg[i,-4:],sg[0,-4:])))
    if not all(eq):raise RuntimeError('EXACT_TOKENIZER_LEFT_PADDING_EQUIVALENCE_RED')
    params={n:p for n,p in peft.named_parameters() if 'lora_A' in n or 'lora_B' in n};originals={n:p.detach().clone() for n,p in params.items()}
    def restore():
        with torch.no_grad():
            for n,p in params.items():p.copy_(originals[n])
    def mutate(cond):
        restore();g=torch.Generator().manual_seed(SEED+91)
        with torch.no_grad():
            if cond=='FULL':return
            if cond.startswith('DOSE_'):
                fac={'DOSE_0p50':.5,'DOSE_1p50':1.5}[cond]
                for n,p in params.items():
                    if 'lora_B' in n:p.mul_(fac)
            elif cond=='RANDOM_RANK_MATCHED':
                for n,p in params.items():
                    o=originals[n];z=torch.randn(o.shape,generator=g,dtype=o.dtype);zn=float(z.norm());on=float(o.norm());p.copy_(z*(on/zn if zn else 0))
            elif cond=='LAYER_SHUFFLED':
                groups=defaultdict(list)
                for n,o in originals.items():groups[(tuple(o.shape),'A' if 'lora_A' in n else 'B')].append(n)
                rng=random.Random(SEED+92)
                for names in groups.values():
                    src=names[:];rng.shuffle(src)
                    for dst,ss in zip(names,src):params[dst].copy_(originals[ss])
            else:raise KeyError(cond)
    def model_ctx(cond):
        if cond=='BASE':return peft.disable_adapter()
        mutate(cond);return contextlib.nullcontext()
    def nll_rows_current(cases,batch_size=4):
        rows=[]
        for bi in range(0,len(cases),batch_size):
            chunk=cases[bi:bi+batch_size];seqs=[];labs=[]
            for c in chunk:
                pids=tok(c['prompt'],add_special_tokens=True)['input_ids'];tids=tok(jd(c['target']),add_special_tokens=False)['input_ids'];seqs.append(pids+tids);labs.append([-100]*len(pids)+tids)
            mx=max(map(len,seqs));ids=[];am=[];lb=[]
            for s,l in zip(seqs,labs):padn=mx-len(s);ids.append([tok.pad_token_id]*padn+s);am.append([0]*padn+[1]*len(s));lb.append([-100]*padn+l)
            ids=torch.tensor(ids);am=torch.tensor(am);lb=torch.tensor(lb);log=peft(input_ids=ids,attention_mask=am).logits;sl=log[:,:-1,:].contiguous();yl=lb[:,1:].contiguous();loss=F.cross_entropy(sl.view(-1,sl.size(-1)),yl.view(-1),ignore_index=-100,reduction='none').view(yl.shape);mask=yl.ne(-100)
            for j,c in enumerate(chunk):rows.append({'id':c['id'],'partition':c['partition'],'nll':float(loss[j][mask[j]].mean()),'tokens':int(mask[j].sum())})
        return rows
    def nll_rows(cond,cases,batch_size=4):
        with model_ctx(cond):return nll_rows_current(cases,batch_size)
    nll={cond:nll_rows(cond,selected) for cond in T['conditions']};gen_ids=set(T['generation_ids']);gen=[c for c in allcases if c['id'] in gen_ids];assert len(gen)==int(T['generation_n'])
    def generate(cond,cases,batch_size=2):
        outs=[]
        with model_ctx(cond):
            for bi in range(0,len(cases),batch_size):
                ch=cases[bi:bi+batch_size];e=tok([c['prompt'] for c in ch],return_tensors='pt',padding=True);inp=e['input_ids'].shape[1];g=peft.generate(**e,max_new_tokens=96,do_sample=False,pad_token_id=tok.pad_token_id)
                for j,c in enumerate(ch):
                    pred=parse_json_text(tok.decode(g[j,inp:],skip_special_tokens=True));outs.append({'id':c['id'],'partition':c['partition'],**score_case(pred,c['target'])})
        return outs
    gen_base=generate('BASE',gen);gen_full=generate('FULL',gen)
    def bypart(rows):
        return {p:aggregate([r for r in rows if r['partition']==p]) for p in sorted(set(r['partition'] for r in rows))}
    nb={r['id']:r for r in nll['BASE']};nf={r['id']:r for r in nll['FULL']};gains=[nb[i]['nll']-nf[i]['nll'] for i in nb];means={c:sum(r['nll'] for r in rows)/len(rows) for c,rows in nll.items()};ci=bootstrap_ci(gains);gb=aggregate(gen_base);gf=aggregate(gen_full);pb=bypart(gen_base);pf=bypart(gen_full);random_gap=means.get('RANDOM_RANK_MATCHED',999)-means['FULL'];shuffle_gap=means.get('LAYER_SHUFFLED',999)-means['FULL'];overall_gain=gf.get('semantic',0)-gb.get('semantic',0);abs_gain=pf.get('absolute',{}).get('semantic',0)-pb.get('absolute',{}).get('semantic',0);ref_gain=pf.get('relative_reference',{}).get('date',0)-pb.get('relative_reference',{}).get('date',0);ref_acc=pf.get('relative_reference',{}).get('date',0);fp_inc=pf.get('hard_negative',{}).get('false_event',0)-pb.get('hard_negative',{}).get('false_event',0);cmap={c['id']:c for c in allcases};fmap={r['id']:r for r in gen_full};groups=defaultdict(list)
    for cid in T['generation_ids']:
        c=cmap[cid]
        if c['partition']=='relative_reference':groups[c['event']].append(cid)
    pair_total=pair_ok=0
    for ids2 in groups.values():
        for i in range(len(ids2)):
            for j in range(i+1,len(ids2)):
                if cmap[ids2[i]]['reference_date']!=cmap[ids2[j]]['reference_date']:pair_total+=1;pair_ok+=int(fmap[ids2[i]]['date']==1 and fmap[ids2[j]]['date']==1)
    pair_acc=(pair_ok/pair_total) if pair_total else 0.0;positive=bool(ci[0]>0.01 and random_gap>=0.01 and shuffle_gap>=0.01 and abs_gain>=0.05 and fp_inc<=0.10);dynamic=bool(positive and ref_gain>=0.20 and ref_acc>=0.50 and pair_acc>=0.50);failure=bool((overall_gain<=-0.10 and ci[1]<0 and random_gap>=0.01 and shuffle_gap>=0.01) or (fp_inc>=0.20 and random_gap>=0.01 and shuffle_gap>=0.01));behavior={'schema':'R22573_C57_BEHAVIOR_V1','tier':tier,'nll_means':means,'full_minus_base_nll_gain_mean':statistics.mean(gains),'paired_gain_ci95':ci,'full_vs_random_nll_gap':random_gap,'full_vs_shuffle_nll_gap':shuffle_gap,'generation_n':len(gen),'generation_base':gb,'generation_full':gf,'generation_base_by_partition':pb,'generation_full_by_partition':pf,'overall_semantic_gain':overall_gain,'absolute_semantic_gain':abs_gain,'relative_reference_date_gain':ref_gain,'relative_reference_full_date_accuracy':ref_acc,'reference_pair_resolution_accuracy':pair_acc,'reference_pair_count':pair_total,'hard_negative_fp_increase':fp_inc,'positive_behavior_gate':positive,'dynamic_temporal_gate':dynamic,'failure_gate':failure,'exact_tokenizer_left_padding_equivalence':eq,'raw_output_text_exported':False};(out/'C57_BEHAVIOR.json').write_text(json.dumps(behavior,indent=2))
    causal={'schema':'R22573_C57_CAUSAL_V1','executed':False,'results':[],'E3_localization_admitted':False,'admitted_groups':[]}
    if positive:
        cset=selected[:int(T['causal_n'])];base_rows=nll_rows('BASE',cset);full_rows=nll_rows('FULL',cset);bm={r['id']:r['nll'] for r in base_rows};fm={r['id']:r['nll'] for r in full_rows};basegain=sum(bm[i]-fm[i] for i in bm)/len(bm);layers=[]
        for n in params:
            m=re.search(r'layers\.(\d+)\.',n)
            if m:layers.append(int(m.group(1)))
        maxlayer=max(layers) if layers else 0
        def ablate(group):
            restore()
            with torch.no_grad():
                for n,p in params.items():
                    hit=False
                    if group=='ATTENTION':hit=any(x in n for x in ['q_proj','k_proj','v_proj','o_proj'])
                    elif group=='MLP':hit=any(x in n for x in ['gate_proj','up_proj','down_proj'])
                    elif group in ['Q','K','V','O','GATE','UP','DOWN']:hit=(group.lower()+'_proj') in n
                    else:
                        m=re.search(r'layers\.(\d+)\.',n);li=int(m.group(1)) if m else -1;third=max(1,math.ceil((maxlayer+1)/3));band='EARLY' if li<third else 'MIDDLE' if li<2*third else 'LATE';hit=(band==group)
                    if hit and 'lora_B' in n:p.zero_()
        groups2=['ATTENTION','MLP','Q','K','V','O','GATE','UP','DOWN','EARLY','MIDDLE','LATE'];ids3=[c['id'] for c in cset];half=max(1,len(ids3)//2);halves=[ids3[:half],ids3[half:]]
        for group in groups2:
            ablate(group);ar=nll_rows_current(cset);amap={r['id']:r['nll'] for r in ar};halfstats=[];passes=[]
            for hids in halves:
                if not hids:continue
                bg2=sum(bm[i]-fm[i] for i in hids)/len(hids);loss2=sum(amap[i]-fm[i] for i in hids)/len(hids);frac=(loss2/bg2) if bg2>1e-9 else 0.0;hp=bool(loss2>=0.02 and frac>=0.50);halfstats.append({'base_full_gain':bg2,'ablation_loss_of_gain':loss2,'fraction_removed':frac,'pass':hp});passes.append(hp)
            overall_loss=sum(amap[i]-fm[i] for i in ids3)/len(ids3);overall_frac=(overall_loss/basegain) if basegain>1e-9 else 0.0;gp=bool(len(passes)==2 and all(passes));causal['results'].append({'group':group,'base_full_gain':basegain,'loss_of_gain':overall_loss,'fraction_removed':overall_frac,'half_checks':halfstats,'E3_gate':gp});
            if gp:causal['admitted_groups'].append(group)
        causal['executed']=True;causal['E3_localization_admitted']=bool(causal['admitted_groups'])
    (out/'C57_CAUSAL.json').write_text(json.dumps(causal,indent=2));prov={'schema':'R22573_C57_PROVENANCE_V1','adapter_repo':ADAPTER_REPO,'adapter_revision':ADAPTER_REV,'adapter_sha256':ADAPTER_SHA,'adapter_license':'mit','base_repo':BASE_REPO,'base_revision':BASE_REV,'base_model_safetensors_sha256':BASE_SHA,'base_license':'apache-2.0','training_time_base_revision_proven':False,'source_repo_pickle_risk_files_excluded':True,'source_consumed_once':True,'raw_committed_to_git':False,'selected_tier':tier};(out/'C57_PROVENANCE.json').write_text(json.dumps(prov,indent=2));shutil.rmtree(raw,ignore_errors=True);cleanup={'schema':'R22573_C57_CLEANUP_V1','raw_root_deleted':not raw.exists(),'raw_weights_remaining':0,'raw_tokenizer_remaining':0,'hf_cache_external_cleanup_required_by_workflow':True};(out/'C57_CLEANUP.json').write_text(json.dumps(cleanup,indent=2));print(json.dumps({'static':static,'behavior':behavior,'causal':causal},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preseal','runtime-smoke','execute']);ap.add_argument('--out',required=True);ap.add_argument('--work',default='c57-work');a=ap.parse_args();out=Path(a.out)
    if a.mode=='preseal':preseal(out)
    elif a.mode=='runtime-smoke':runtime_smoke(out)
    else:execute(out,Path(a.work))
if __name__=='__main__':main()
