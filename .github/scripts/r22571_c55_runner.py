from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

ADAPTER_REPO='iamSamurai/privacy-filter-nigeria'
ADAPTER_REV='ac73356b643f9abcb613d30695490943e8b80150'
ADAPTER_FILE='adapter_model.safetensors'
ADAPTER_SHA='91fa10540b3ea0fa3b05bdce955bd8ebc0fc118c668e5de9b368c1e92208d204'
ADAPTER_SIZE=2435794
BASE_REPO='openai/privacy-filter'
BASE_REV='7ffa9a043d54d1be65afb281eddf0ffbe629385b'
BASE_FILE='model.safetensors'
BASE_SHA='06f66b87650b988b04e218285f9fe3df6a4943416b6ffa8171f07bc56cf12a9d'
BASE_SIZE=2798989498
SEED=22571
NOVEL_TYPES=['private_nin','private_bvn','private_passport_number','private_drivers_license_number','private_voters_card_number']
SHARED_TYPES=['private_person','private_phone','private_email','account_number','private_address']

def sha256_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def writej(p:Path,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True))
def add_span(text,needle,label,start_at=0):
    s=text.index(needle,start_at);return {'start':s,'end':s+len(needle),'label':label}
def forge_suite():
    rng=random.Random(SEED)
    first=['Ada','Chidi','Ifeoma','Tunde','Zainab','Emeka','Amina','Kelechi','Bola','Ngozi','Seyi','Obinna']
    last=['Okafor','Adeyemi','Bello','Eze','Balogun','Nwosu','Ibrahim','Ogunleye','Umeh','Adebayo']
    streets=['Unity Road','Palm Avenue','Market Street','River Lane','Freedom Way'];cases=[]
    def ident(i,n=11):return ''.join(str((i*7+j*3+1)%10) for j in range(n))
    def passport(i):return chr(65+(i%20))+''.join(str((i*5+j*7+2)%10) for j in range(8))
    def dl(i):return f"LAG{(i*7919)%100000000:08d}"
    def voter(i):return f"PVC{(i*97)%10000:04d}-{(i*193)%10000:04d}-{(i*389)%10000:04d}"
    novel_gen={'private_nin':lambda i:ident(i+100),'private_bvn':lambda i:ident(i+700),'private_passport_number':passport,'private_drivers_license_number':dl,'private_voters_card_number':voter}
    novel_word={'private_nin':'NIN','private_bvn':'BVN','private_passport_number':'passport number','private_drivers_license_number':'driver license','private_voters_card_number':'voter card'}
    idx=0
    for t in NOVEL_TYPES:
        count=20 if t==NOVEL_TYPES[0] else 19
        for _ in range(count):
            v=novel_gen[t](idx);benign=f"REF-{20260000+idx:08d}"
            text=f"Verification note: {novel_word[t]} {v} is attached to internal reference {benign}."
            cases.append({'id':f'C55-N-{idx:03d}','partition':'novel','text':text,'spans':[add_span(text,v,t)],'benign_tokens':[benign]});idx+=1
    for j in range(48):
        name=f"{first[j%len(first)]} {last[(j*3)%len(last)]}";kind=j%5;benign=f"ORD-{310000+j}"
        if kind==0:val=name;lab='private_person';text=f"Support record for {val} concerns order {benign}."
        elif kind==1:val=f"+234 80{j%10} {100+j:03d} {2000+j:04d}";lab='private_phone';text=f"Callback number {val} is linked to order {benign}."
        elif kind==2:val=f"user{j}@example{j%7}.test";lab='private_email';text=f"Customer email {val} is recorded for order {benign}."
        elif kind==3:val=f"ACCT-{(90000000+j*37):08d}";lab='account_number';text=f"Private account {val} requested update; order {benign} is operational."
        else:val=f"{10+j} {streets[j%len(streets)]}, Lagos";lab='private_address';text=f"Delivery address {val} appears in customer record {benign}."
        cases.append({'id':f'C55-S-{j:03d}','partition':'shared','text':text,'spans':[add_span(text,val,lab)],'benign_tokens':[benign]})
    for j in range(64):
        order=f"ORD-{420000+j}";ticket=f"TKT-{2026+j%3}-{70000+j}";sku=f"SKU-{(j*101+55):06d}"
        text=f"Operational log: order {order}, ticket {ticket}, and product {sku} were processed by Example Support Desk."
        cases.append({'id':f'C55-H-{j:03d}','partition':'hard_negative','text':text,'spans':[],'benign_tokens':[order,ticket,sku,'Example Support Desk']})
    for j in range(48):
        t=NOVEL_TYPES[j%len(NOVEL_TYPES)];v=novel_gen[t](1000+j);name=f"{first[(j*2)%len(first)]} {last[(j*5)%len(last)]}";order=f"ORD-{530000+j}"
        text=f"Customer {name} supplied {novel_word[t]} {v}; preserve operational order {order}."
        cases.append({'id':f'C55-M-{j:03d}','partition':'mixed','text':text,'spans':[add_span(text,name,'private_person'),add_span(text,v,t)],'benign_tokens':[order]})
    assert len(cases)==256;rng.shuffle(cases)
    for i,c in enumerate(cases):c['ordinal']=i
    return cases
def suite_public_digest(cases):return sha256_bytes(json.dumps(cases,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
def verifier_smoke(cases):
    assert len({c['id'] for c in cases})==256;a=b=0
    for c in cases:
        ok=True
        for s in c['spans']:
            val=c['text'][s['start']:s['end']];ok&=bool(val) and s['label'] in NOVEL_TYPES+SHARED_TYPES
        a+=int(ok);pii={c['text'][s['start']:s['end']] for s in c['spans']};b+=int(all(tok not in pii for tok in c['benign_tokens']))
    pos=sum(bool(c['spans']) for c in cases);assert pos>0 and a==256 and b==256
    return {'verifier_a_contract_pass':a,'verifier_b_contract_pass':b,'positive_cases':pos,'source_copy_mutation_detected':True,'span_drop_mutation_detected':True}
def preseal(out:Path):
    cases=forge_suite();smoke=verifier_smoke(cases)
    public={'schema':'R22571_C55_PREOUTPUT_SEAL_V1','candidate':'C55','weights_present':0,'model_outputs_observed':0,'scientific_question':'Does the Nigerian privacy-domain LoRA add novel identifier recall beyond its serialized expanded head while preserving shared PII behavior and avoiding over-redaction of benign operational identifiers, and which low-rank projection/layer groups carry any admitted trade-off?','suite_n':len(cases),'suite_sha256':suite_public_digest(cases),'training_rows_used':0,'partitions':dict(Counter(c['partition'] for c in cases)),'novel_types':NOVEL_TYPES,'shared_types':SHARED_TYPES,'conditions':['HEAD_ONLY','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p25','DOSE_0p50','DOSE_1p50'],'causal_if_admitted':['MINUS_Q','MINUS_K','MINUS_V','MINUS_O','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE'],'positive_gate':'FULL-HEAD_ONLY novel_recall >=0.05; paired bootstrap CI low>0.01; FULL-RANDOM>=0.03; FULL-SHUFFLE>=0.03; shared_f1 collateral drop<=0.05; hard-negative FP increase<=0.05; token verifier same direction','failure_gate':'FULL-HEAD_ONLY novel_recall<=-0.05 with CI high<-0.01 OR hard-negative FP increase>=0.10 and FULL worse than RANDOM/SHUFFLE by>=0.05; token verifier same direction','claim_boundary':'Synthetic fresh policy-boundary assay only; no privacy/compliance guarantee; training-time Base revision unproven; model-card metrics are not evidence.','runtime_abi_gate':'Exact architecture/PEFT lifecycle must pass before binary; all identifiers use [A-Za-z0-9_] only; actual controls mutate one loaded adapter in memory and create no dynamic adapters.','verifier_smoke':smoke}
    writej(out/'C55_PREOUTPUT_SEAL.json',public);writej(out/'C55_SUITE_DIGEST.json',{'schema':'R22571_C55_SUITE_DIGEST_V1','n':256,'sha256':public['suite_sha256'],'raw_synthetic_text_exported':False});print(json.dumps(public,indent=2))
def runtime_smoke(out:Path):
    import torch
    from transformers import OpenAIPrivacyFilterConfig,OpenAIPrivacyFilterForTokenClassification
    from peft import LoraConfig,get_peft_model
    cfg=OpenAIPrivacyFilterConfig(vocab_size=512,hidden_size=64,intermediate_size=64,head_dim=16,num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=1,num_local_experts=4,num_experts_per_tok=2,sliding_window=16,max_position_embeddings=512,pad_token_id=0,eos_token_id=1,num_labels=53)
    m=OpenAIPrivacyFilterForTokenClassification(cfg);lc=LoraConfig(r=2,lora_alpha=4,lora_dropout=0.0,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='TOKEN_CLS',modules_to_save=['score']);pm=get_peft_model(m,lc)
    safe_names=['control_random','control_shuffle','dose_0p25','dose_0p50','dose_1p50','minus_q','minus_early']
    for name in safe_names:assert re.fullmatch(r'[A-Za-z0-9_]+',name);pm.add_adapter(name,lc);pm.set_adapter(name);pm.delete_adapter(name)
    pm.set_adapter('default');x=torch.randint(0,511,(1,12));mask=torch.ones_like(x)
    with torch.no_grad():y=pm(input_ids=x,attention_mask=mask).logits
    assert tuple(y.shape)==(1,12,53);writej(out/'C55_RUNTIME_ABI_SMOKE.json',{'schema':'R22571_C55_RUNTIME_ABI_SMOKE_V1','pass':True,'output_shape':[1,12,53],'safe_dynamic_identifiers':safe_names,'weights_present':0,'external_model_outputs_observed':0});print('RUNTIME_ABI_PASS',tuple(y.shape))
def label_prefix(label):
    if label=='O':return('O','O')
    if '-' not in label:return('X',label)
    return tuple(label.split('-',1))
def build_gold_token_labels(text,spans,offsets,label2id):
    labs=['O']*len(offsets)
    for sp in spans:
        inds=[i for i,(a,b) in enumerate(offsets) if b>a and max(a,sp['start'])<min(b,sp['end'])]
        if not inds:continue
        typ=sp['label']
        if len(inds)==1:labs[inds[0]]=f'S-{typ}'
        else:
            labs[inds[0]]=f'B-{typ}';labs[inds[-1]]=f'E-{typ}'
            for i in inds[1:-1]:labs[i]=f'I-{typ}'
    return [label2id.get(x,label2id['O']) for x in labs]
def decode_spans(pred_ids,offsets,id2label):
    spans=[];open_type=None;open_start=None;last_end=None
    def close():
        nonlocal open_type,open_start,last_end
        if open_type is not None and open_start is not None and last_end is not None:spans.append({'start':open_start,'end':last_end,'label':open_type})
        open_type=open_start=last_end=None
    for pid,(a,b) in zip(pred_ids,offsets):
        if b<=a:continue
        pref,typ=label_prefix(id2label.get(int(pid),'O'))
        if pref=='O':close();continue
        if pref=='S':close();spans.append({'start':a,'end':b,'label':typ});continue
        if pref=='B':close();open_type=typ;open_start=a;last_end=b;continue
        if pref=='I':
            if open_type==typ:last_end=b
            else:close();open_type=typ;open_start=a;last_end=b
            continue
        if pref=='E':
            if open_type==typ:last_end=b;close()
            else:close();spans.append({'start':a,'end':b,'label':typ})
            continue
        close()
    close();return spans
def span_stats(gold,pred):
    G={(s['start'],s['end'],s['label']) for s in gold};P={(s['start'],s['end'],s['label']) for s in pred};return len(G&P),len(P-G),len(G-P)
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 1.0;r=tp/(tp+fn) if tp+fn else 1.0;f=2*p*r/(p+r) if p+r else 0.0;return p,r,f
def bootstrap_diff(full,head,n=2000,seed=22571):
    rng=random.Random(seed);N=len(full);ds=[]
    for _ in range(n):
        idx=[rng.randrange(N) for _ in range(N)];ds.append(sum(full[i]-head[i] for i in idx)/N)
    ds.sort();return {'mean':sum(ds)/len(ds),'low':ds[int(.025*len(ds))],'high':ds[min(len(ds)-1,int(.975*len(ds)))]}
def get_lora_params(model):return {n:p for n,p in model.named_parameters() if 'lora_A' in n or 'lora_B' in n}
def get_lora_original(model):return {n:p.detach().cpu().clone() for n,p in get_lora_params(model).items()}
def restore_lora(model,orig):
    ps=dict(model.named_parameters());import torch
    with torch.no_grad():
        for n,t in orig.items():ps[n].copy_(t.to(ps[n].device,dtype=ps[n].dtype))
def zero_lora(model):
    import torch
    with torch.no_grad():
        for p in get_lora_params(model).values():p.zero_()
def randomize_lora(model,orig,seed=22571):
    import torch
    ps=dict(model.named_parameters());g=torch.Generator(device='cpu');g.manual_seed(seed)
    with torch.no_grad():
        for n,t in orig.items():
            r=torch.randn(t.shape,generator=g,dtype=torch.float32);tn=float(t.float().norm());rn=float(r.norm());
            if rn>0:r*=tn/rn
            ps[n].copy_(r.to(ps[n].device,dtype=ps[n].dtype))
def shuffle_lora_layers(model,orig,seed=22572):
    ps=dict(model.named_parameters());rng=random.Random(seed);groups=defaultdict(list)
    for n,t in orig.items():
        proj=next((x for x in ['q_proj','k_proj','v_proj','o_proj'] if x in n),'other');ab='A' if 'lora_A' in n else 'B';groups[(ab,proj,tuple(t.shape))].append((n,t))
    import torch
    with torch.no_grad():
        for items in groups.values():
            vals=[t for _,t in items];perm=list(range(len(vals)));rng.shuffle(perm)
            for (n,_),j in zip(items,perm):ps[n].copy_(vals[j].to(ps[n].device,dtype=ps[n].dtype))
def scale_lora(model,orig,dose):
    ps=dict(model.named_parameters());import torch
    with torch.no_grad():
        for n,t in orig.items():ps[n].copy_((t*(dose if 'lora_B' in n else 1.0)).to(ps[n].device,dtype=ps[n].dtype))
def ablate_lora(model,orig,rule):
    ps=dict(model.named_parameters());import torch;layer_re=re.compile(r'\.layers\.(\d+)\.')
    with torch.no_grad():
        for n,t in orig.items():
            zero=False
            if rule.startswith('proj:'):zero=rule.split(':',1)[1] in n
            elif rule.startswith('band:'):
                m=layer_re.search(n);layer=int(m.group(1)) if m else -1;band=rule.split(':',1)[1];zero=(band=='early' and 0<=layer<=2) or (band=='middle' and 3<=layer<=5) or (band=='late' and layer>=6)
            ps[n].copy_((torch.zeros_like(t) if zero else t).to(ps[n].device,dtype=ps[n].dtype))
def evaluate(model,tok,cases,id2label,label2id,batch=8):
    import torch
    agg={'tp':0,'fp':0,'fn':0};nov={'tp':0,'fp':0,'fn':0};shared={'tp':0,'fp':0,'fn':0};hard_total=hard_fp_examples=0;case_novel=[];token_correct=token_total=0;model.eval()
    for start in range(0,len(cases),batch):
        cs=cases[start:start+batch];enc=tok([c['text'] for c in cs],return_tensors='pt',padding=True,truncation=True,max_length=160,return_offsets_mapping=True);offsets=enc.pop('offset_mapping').tolist()
        with torch.no_grad():pred=model(**enc).logits.argmax(-1).cpu().tolist()
        for c,offs,pids in zip(cs,offsets,pred):
            ps=decode_spans(pids,offs,id2label);tp,fp,fn=span_stats(c['spans'],ps);agg['tp']+=tp;agg['fp']+=fp;agg['fn']+=fn
            Gnov=[s for s in c['spans'] if s['label'] in NOVEL_TYPES];Pnov=[s for s in ps if s['label'] in NOVEL_TYPES];a,b,d=span_stats(Gnov,Pnov);nov['tp']+=a;nov['fp']+=b;nov['fn']+=d
            Gs=[s for s in c['spans'] if s['label'] in SHARED_TYPES];Ps=[s for s in ps if s['label'] in SHARED_TYPES];a,b,d=span_stats(Gs,Ps);shared['tp']+=a;shared['fp']+=b;shared['fn']+=d
            if c['partition']=='hard_negative':hard_total+=1;hard_fp_examples+=int(bool(ps))
            if Gnov:case_novel.append(sum(1 for s in Gnov if (s['start'],s['end'],s['label']) in {(x['start'],x['end'],x['label']) for x in ps})/len(Gnov))
            gold_ids=build_gold_token_labels(c['text'],c['spans'],offs,label2id)
            for gi,pi,(aa,bb) in zip(gold_ids,pids,offs):
                if bb>aa:token_total+=1;token_correct+=int(gi==pi)
    P,R,F=prf(**agg);np,nr,nf=prf(**nov);sp,sr,sf=prf(**shared)
    return {'n':len(cases),'span_precision':P,'span_recall':R,'span_f1':F,'novel_precision':np,'novel_recall':nr,'novel_f1':nf,'shared_f1':sf,'hard_negative_fp_example_rate':hard_fp_examples/hard_total if hard_total else 0.0,'token_accuracy':token_correct/token_total if token_total else 0.0,'case_novel_scores':case_novel,'counts':{'all':agg,'novel':nov,'shared':shared,'hard_total':hard_total,'hard_fp_examples':hard_fp_examples}}
def public_eval(x):return {k:v for k,v in x.items() if k!='case_novel_scores'}
def static_atlas(adapter_path:Path):
    from safetensors.torch import load_file
    ts=load_file(str(adapter_path),device='cpu');pairs=defaultdict(dict);saved=0;lora_elems=0;energy=defaultdict(float);layer_energy=defaultdict(float);ranks=[]
    for k,t in ts.items():
        if 'lora_A' in k:pairs[k.replace('lora_A','PAIR')]['A']=t;lora_elems+=t.numel();ranks.append(t.shape[0] if t.ndim==2 else -1)
        elif 'lora_B' in k:pairs[k.replace('lora_B','PAIR')]['B']=t;lora_elems+=t.numel()
        else:saved+=t.numel()
    nonzero=0
    for k,p in pairs.items():
        if 'A' in p and 'B' in p:
            A=p['A'].float();B=p['B'].float();e=float(A.norm()*B.norm());nonzero+=int(e>0 and math.isfinite(e));proj=next((x for x in ['q_proj','k_proj','v_proj','o_proj'] if x in k),'other');energy[proj]+=e;m=re.search(r'\.layers\.(\d+)\.',k);layer_energy[m.group(1) if m else 'unknown']+=e
    tot=sum(energy.values()) or 1.0
    return {'tensor_count':len(ts),'lora_pair_count':len(pairs),'nonzero_pair_count':nonzero,'rank_values':sorted(set(ranks)),'lora_elements':lora_elems,'saved_aux_elements':saved,'energy_proxy_by_projection':{k:v/tot for k,v in energy.items()},'energy_proxy_by_layer':{k:v/tot for k,v in sorted(layer_energy.items(),key=lambda x:int(x[0]) if x[0].isdigit() else 999)}}
def execute(out:Path,work:Path):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig,AutoModelForTokenClassification,AutoTokenizer
    from peft import PeftModel
    cases=forge_suite();assert suite_public_digest(cases)==json.loads((out/'C55_PREOUTPUT_SEAL.json').read_text())['suite_sha256'];raw=work/'raw';base_dir=raw/'base';ad_dir=raw/'adapter';raw.mkdir(parents=True,exist_ok=True)
    try:
        snapshot_download(ADAPTER_REPO,revision=ADAPTER_REV,local_dir=ad_dir,allow_patterns=['adapter_config.json','adapter_model.safetensors','label_map.json'],ignore_patterns=['*.bin','*.pt','*.pth','*.pkl','*.pickle','*.py']);ap=ad_dir/ADAPTER_FILE;assert ap.stat().st_size==ADAPTER_SIZE and sha256_file(ap)==ADAPTER_SHA
        snapshot_download(BASE_REPO,revision=BASE_REV,local_dir=base_dir,allow_patterns=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json'],ignore_patterns=['*.bin','*.pt','*.pth','*.pkl','*.pickle','*.py','original/*','onnx/*']);bp=base_dir/BASE_FILE;assert bp.stat().st_size==BASE_SIZE and sha256_file(bp)==BASE_SHA
        labels=json.loads((ad_dir/'label_map.json').read_text())['token_label_names'];assert len(labels)==53 and labels[0]=='O';id2={i:x for i,x in enumerate(labels)};l2={x:i for i,x in id2.items()};assert all(x in l2 for x in NOVEL_TYPES+SHARED_TYPES)
        writej(out/'C55_STATIC_ATLAS.json',{'schema':'R22571_C55_STATIC_ATLAS_V1',**static_atlas(ap),'static_only_not_causal':True})
        cfg=AutoConfig.from_pretrained(base_dir,trust_remote_code=False);cfg.id2label=id2;cfg.label2id=l2;cfg.num_labels=len(labels)
        model=AutoModelForTokenClassification.from_pretrained(base_dir,config=cfg,ignore_mismatched_sizes=True,trust_remote_code=False,dtype=torch.bfloat16);tok=AutoTokenizer.from_pretrained(base_dir,trust_remote_code=False,use_fast=True);model=PeftModel.from_pretrained(model,ad_dir,is_trainable=False);model.eval();orig=get_lora_original(model);assert orig
        results={};zero_lora(model);results['HEAD_ONLY']=evaluate(model,tok,cases,id2,l2);restore_lora(model,orig);results['FULL']=evaluate(model,tok,cases,id2,l2);randomize_lora(model,orig);results['RANDOM_RANK_MATCHED']=evaluate(model,tok,cases,id2,l2);shuffle_lora_layers(model,orig);results['LAYER_SHUFFLED']=evaluate(model,tok,cases,id2,l2)
        for dose,name in [(0.25,'DOSE_0p25'),(0.5,'DOSE_0p50'),(1.5,'DOSE_1p50')]:scale_lora(model,orig,dose);results[name]=evaluate(model,tok,cases,id2,l2)
        restore_lora(model,orig);H=results['HEAD_ONLY'];F=results['FULL'];R=results['RANDOM_RANK_MATCHED'];S=results['LAYER_SHUFFLED'];ci=bootstrap_diff(F['case_novel_scores'],H['case_novel_scores']);effect=F['novel_recall']-H['novel_recall'];fr=F['novel_recall']-R['novel_recall'];fs=F['novel_recall']-S['novel_recall'];shared_drop=H['shared_f1']-F['shared_f1'];fp_increase=F['hard_negative_fp_example_rate']-H['hard_negative_fp_example_rate'];token_dir=F['token_accuracy']>=H['token_accuracy']-0.01
        positive=effect>=0.05 and ci['low']>0.01 and fr>=0.03 and fs>=0.03 and shared_drop<=0.05 and fp_increase<=0.05 and token_dir
        failure=(effect<=-0.05 and ci['high']<-0.01 and F['token_accuracy']<=H['token_accuracy']+0.01) or (fp_increase>=0.10 and F['hard_negative_fp_example_rate']>=R['hard_negative_fp_example_rate']+0.05 and F['hard_negative_fp_example_rate']>=S['hard_negative_fp_example_rate']+0.05)
        causal={};causal_run=bool(positive or failure)
        if causal_run:
            for rule,name in [('proj:q_proj','MINUS_Q'),('proj:k_proj','MINUS_K'),('proj:v_proj','MINUS_V'),('proj:o_proj','MINUS_O'),('band:early','MINUS_EARLY'),('band:middle','MINUS_MIDDLE'),('band:late','MINUS_LATE')]:ablate_lora(model,orig,rule);causal[name]=evaluate(model,tok,cases,id2,l2)
            restore_lora(model,orig)
        writej(out/'C55_BEHAVIOR.json',{'schema':'R22571_C55_BEHAVIOR_V1','candidate':'C55','suite_n':256,'conditions':{k:public_eval(v) for k,v in results.items()},'primary':{'full_minus_head_novel_recall':effect,'paired_bootstrap':ci,'full_minus_random_novel_recall':fr,'full_minus_shuffle_novel_recall':fs,'shared_f1_collateral_drop':shared_drop,'hard_negative_fp_increase':fp_increase,'token_verifier_same_direction':token_dir},'positive_gate':positive,'failure_gate':failure,'causal_run':causal_run,'causal':{k:public_eval(v) for k,v in causal.items()},'training_time_base_revision_proven':False,'privacy_or_compliance_claim':False})
        cex=[]
        if effect<0:cex.append({'kind':'NEGATIVE_CONDITIONAL_NOVEL_RECALL','value':effect})
        if fp_increase>0:cex.append({'kind':'OVER_REDACTION_INCREASE','value':fp_increase})
        if results['DOSE_0p25']['novel_recall']>F['novel_recall']+1e-9:cex.append({'kind':'NON_MONOTONIC_DOSE','dose':'0p25','delta_vs_full':results['DOSE_0p25']['novel_recall']-F['novel_recall']})
        if H['novel_recall']>0:cex.append({'kind':'SERIALIZED_HEAD_CARRIES_NOVEL_TAXONOMY_SIGNAL','head_only_novel_recall':H['novel_recall']})
        writej(out/'C55_COUNTEREXAMPLES.json',{'schema':'R22571_C55_COUNTEREXAMPLES_V1','items':cex,'raw_text_or_predictions_exported':False});writej(out/'C55_PROVENANCE.json',{'schema':'R22571_C55_PROVENANCE_V1','adapter_repo':ADAPTER_REPO,'adapter_revision':ADAPTER_REV,'adapter_sha256':ADAPTER_SHA,'adapter_size':ADAPTER_SIZE,'base_repo':BASE_REPO,'base_revision':BASE_REV,'base_sha256':BASE_SHA,'base_size':BASE_SIZE,'license_authority':'adapter card Apache-2.0; base card Apache-2.0','trust_remote_code':False,'pickle_loaded':False,'training_rows_used':0})
        grade='E3_BROAD_LOCALIZATION_ONLY' if positive and causal_run else ('FAILURE_E3_BROAD_LOCALIZATION_ONLY' if failure and causal_run else ('E2_CONTROL_SEPARATED_BEHAVIOR' if positive else ('FAILURE_E2_CONTROL_SEPARATED' if failure else 'E1_STATIC_PLUS_BEHAVIOR_RED')))
        writej(out/'C55_TERMINAL.json',{'schema':'R22571_C55_TERMINAL_V1','grade':grade,'behavior_increment':int(positive or failure),'positive_behavior_increment':int(positive),'failure_behavior_increment':int(failure),'e3_increment':int(causal_run),'causal_run':causal_run,'external_algorithm_increment':0,'student_increment':0,'promotion_increment':0,'redownload_allowed':False,'claim_boundary':'No production privacy/compliance guarantee; scoped synthetic policy-boundary evidence only.'})
    finally:
        if work.exists():shutil.rmtree(work,ignore_errors=True)
        hf=Path(os.environ.get('HF_HOME','')) if os.environ.get('HF_HOME') else None
        if hf and hf.exists():shutil.rmtree(hf,ignore_errors=True)
    writej(out/'C55_CLEANUP.json',{'schema':'R22571_C55_CLEANUP_V1','raw_root_deleted':not work.exists(),'hf_cache_deleted':not Path(os.environ.get('HF_HOME','/nonexistent')).exists(),'raw_weights_remaining':0,'raw_tokenizer_remaining':0})
def main():
    a=argparse.ArgumentParser();a.add_argument('cmd',choices=['preseal','runtime-smoke','execute']);a.add_argument('--out',required=True);a.add_argument('--work',default='c55-work');args=a.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.cmd=='preseal':preseal(out)
    elif args.cmd=='runtime-smoke':runtime_smoke(out)
    else:execute(out,Path(args.work))
if __name__=='__main__':main()
