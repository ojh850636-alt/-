#!/usr/bin/env python3
import argparse, copy, hashlib, json, math, os, random, re, shutil, sys
from pathlib import Path

ADAPTER_REPO='ambrosehui/flan-t5-small-rag-hallucinations-judgment'
ADAPTER_REV='0abe2dc5cfd732b20488f712806355108b772028'
ADAPTER_SHA='421ce544da12db31f569db6f6e5ab0599a4e20eb670468a59dab403185b3e022'
BASE_REPO='google/flan-t5-small'
SEED=22569

NAMES=['Nara','Kivo','Talen','Mira','Soren','Vela','Rumi','Daro','Lina','Pavo','Eris','Juno','Kelan','Omi','Rava','Tori']
CITIES=['Luma','Kora','Venn','Sora','Dema','Novi','Pira','Rell']
COLORS=['amber','teal','violet','silver','crimson','indigo','golden','white']
JOBS=['medic','baker','pilot','teacher','carpenter','gardener','chemist','tailor']
OBJECTS=['violin','camera','lantern','bicycle','telescope','guitar','compass','kettle']


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def make_suite(n=512, seed=SEED):
    rng=random.Random(seed); cases=[]
    parts=[('discovery',128),('confirmation',128),('lexical_ood',96),('compositional_ood',96),('negative_oos',64)]
    cid=0
    for part,count in parts:
        for j in range(count):
            a,b,c=rng.sample(NAMES,3); city,city2=rng.sample(CITIES,2); color,color2=rng.sample(COLORS,2); job,job2=rng.sample(JOBS,2); obj=rng.choice(OBJECTS)
            faithful=(j%2==0)
            fam=j%8
            distract=[f'{b} lives in {city2}.',f'{c} works as a {job2}.',f'{b} is {color2}.']
            if part in ('lexical_ood','negative_oos'):
                if fam==0:
                    ctx=f'Residence record: {a} resides in {city}. '+ ' '.join(distract)
                    q=f'Where does {a} live?'; ans=f'{a} lives in {city}.' if faithful else f'{a} lives in {city2}.'
                elif fam==1:
                    ctx=f'Employment ledger states that {a} has the occupation {job}. '+ ' '.join(distract)
                    q=f'What is {a} employed as?'; ans=f'{a} is a {job}.' if faithful else f'{a} is a {job2}.'
                elif fam==2:
                    ctx=f'Appearance note: the color associated with {a} is {color}. '+ ' '.join(distract)
                    q=f'What color is associated with {a}?'; ans=f'The color is {color}.' if faithful else f'The color is {color2}.'
                elif fam==3:
                    ctx=f'Property log: {a} does not own a {obj}. '+ ' '.join(distract)
                    q=f'Does {a} own a {obj}?'; ans=f'No, {a} does not own a {obj}.' if faithful else f'Yes, {a} owns a {obj}.'
                elif fam==4:
                    ctx=f'Two verified facts: {a} is a {job}; {a} lives in {city}. '+ ' '.join(distract)
                    q=f'Summarize the verified job and city for {a}.'; ans=f'{a} is a {job} in {city}.' if faithful else f'{a} is a {job} in {city} and owns a {obj}.'
                elif fam==5:
                    ctx=f'{a} lives in {city}. {a} is {color}. '+ ' '.join(distract)
                    q=f'Is the following answer fully supported: {a} lives in {city} and is {color}?'; ans=f'{a} lives in {city} and is {color}.' if faithful else f'{a} lives in {city} and is {color2}.'
                elif fam==6:
                    ctx=f'Archive: {a} works as a {job}. No ownership information is recorded for {a}. '+ ' '.join(distract)
                    q=f'What does the archive support about {a}?'; ans=f'{a} works as a {job}.' if faithful else f'{a} works as a {job} and owns a {obj}.'
                else:
                    ctx=f'Confirmed: {a} lives in {city}. Confirmed: {a} is a {job}. '+ ' '.join(distract*2)
                    q=f'Which statement is supported about {a}?'; ans=f'{a} lives in {city} and is a {job}.' if faithful else f'{a} lives in {city2} and is a {job}.'
            else:
                if fam==0:
                    ctx=f'{a} lives in {city}. '+ ' '.join(distract); q=f'Where does {a} live?'; ans=f'{a} lives in {city}.' if faithful else f'{a} lives in {city2}.'
                elif fam==1:
                    ctx=f'{a} works as a {job}. '+ ' '.join(distract); q=f'What is {a}\'s job?'; ans=f'{a} is a {job}.' if faithful else f'{a} is a {job2}.'
                elif fam==2:
                    ctx=f'{a} is {color}. '+ ' '.join(distract); q=f'What color is {a}?'; ans=f'{a} is {color}.' if faithful else f'{a} is {color2}.'
                elif fam==3:
                    ctx=f'{a} does not own a {obj}. '+ ' '.join(distract); q=f'Does {a} own a {obj}?'; ans=f'No, {a} does not own a {obj}.' if faithful else f'Yes, {a} owns a {obj}.'
                elif fam==4:
                    ctx=f'{a} is a {job}. {a} lives in {city}. '+ ' '.join(distract); q=f'Describe {a}\'s job and city.'; ans=f'{a} is a {job} in {city}.' if faithful else f'{a} is a {job} in {city} and owns a {obj}.'
                elif fam==5:
                    ctx=f'{a} lives in {city}. {a} is {color}. '+ ' '.join(distract); q=f'What is known about {a}?'; ans=f'{a} lives in {city} and is {color}.' if faithful else f'{a} lives in {city} and is {color2}.'
                elif fam==6:
                    ctx=f'{a} works as a {job}. '+ ' '.join(distract); q=f'What is supported about {a}?'; ans=f'{a} works as a {job}.' if faithful else f'{a} works as a {job} and owns a {obj}.'
                else:
                    ctx=f'{a} lives in {city}. {a} is a {job}. '+ ' '.join(distract*2); q=f'Which statement is supported about {a}?'; ans=f'{a} lives in {city} and is a {job}.' if faithful else f'{a} lives in {city2} and is a {job}.'
            if part=='compositional_ood':
                ctx='Evidence packet. '+ctx+' No other facts are licensed by this packet.'
                q='Using only the evidence packet, '+q[0].lower()+q[1:]
            if part=='negative_oos' and j%4==0:
                ctx='The context is intentionally sparse. '+ctx
            label='Faithful' if faithful else 'Hallucinated'
            prompt=f'context:{ctx}\nquestion:{q}\nanswer:{ans}'
            cases.append({'id':f'C53-{cid:04d}','partition':part,'family':fam,'expected':label,'prompt':prompt})
            cid+=1
    assert len(cases)==n
    # exact balance by construction within every even-sized partition
    return cases

def suite_public(cases):
    counts={}
    labels={}
    for c in cases:
        counts[c['partition']]=counts.get(c['partition'],0)+1; labels[c['expected']]=labels.get(c['expected'],0)+1
    raw=json.dumps(cases,sort_keys=True,separators=(',',':')).encode()
    return {'n':len(cases),'partition_counts':counts,'label_counts':labels,'suite_sha256':sha_bytes(raw),'raw_cases_exported':False,'generator':'deterministic synthetic evidence packets; no training rows'}

def normalize_label(text):
    t=text.strip().lower()
    h='hallucinated' in t
    f='faithful' in t
    if h and not f: return 'Hallucinated'
    if f and not h: return 'Faithful'
    # tolerate common single-token variants while refusing explanations with ambiguity
    tok=re.findall(r'[a-z]+',t)
    if tok and tok[0] in ('faithful','consistent'): return 'Faithful'
    if tok and tok[0] in ('hallucinated','inconsistent'): return 'Hallucinated'
    return 'INVALID'

def bootstrap_ci(a,b,seed=31337,nboot=2000):
    # paired difference accuracy(a)-accuracy(b), arrays of bool
    rng=random.Random(seed); n=len(a); vals=[]
    for _ in range(nboot):
        s=0.0
        for __ in range(n):
            i=rng.randrange(n); s += (1 if a[i] else 0)-(1 if b[i] else 0)
        vals.append(s/n)
    vals.sort(); return [vals[int(.025*nboot)], vals[min(nboot-1,int(.975*nboot))]]

def cmd_preseal(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    cases=make_suite(); pub=suite_public(cases)
    # two independent label verifiers agree because labels derive from support construction parity.
    v2=['Faithful' if int(c['id'].split('-')[1])%2==0 else 'Hallucinated' for c in cases]
    assert all(x==c['expected'] for x,c in zip(v2,cases))
    (out/'suite_private.json').write_text(json.dumps(cases,ensure_ascii=False))
    contract={'schema':'R22569_C53_PREOUTPUT_SEAL_V1','scientific_question':'Does the adapter learn context-grounded answer faithfulness rather than label/format shortcuts?','weight_bytes_present_at_seal':0,'model_outputs_observed_at_seal':0,'adapter_repo':ADAPTER_REPO,'adapter_revision':ADAPTER_REV,'expected_adapter_sha256':ADAPTER_SHA,'runtime_reference_base':BASE_REPO,'training_time_exact_base_revision_proven':False,'suite':pub,'conditions':['BASE','FULL','RANDOM_COMPONENT_SIGN','LAYER_MODULE_SHUFFLE','DOSE_0.25','DOSE_0.5','DOSE_1.0','DOSE_1.5'],'causal_entry_requires_learned_direction_separation':True,'causal_groups':['ENCODER_SELF','DECODER_SELF','DECODER_CROSS','Q','K','V','O','EARLY','MIDDLE','LATE'],'claim_ceiling_without_training_base_provenance':'E3_SCOPED_RUNTIME_REFERENCE_ONLY'}
    (out/'R22569_C53_PREOUTPUT_SEAL.json').write_text(json.dumps(contract,indent=2))
    print(json.dumps(contract,indent=2))

def load_stack(base_dir, adapter_dir=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True)
    base=AutoModelForSeq2SeqLM.from_pretrained(base_dir,local_files_only=True,torch_dtype=torch.float32)
    if adapter_dir:
        from peft import PeftModel
        model=PeftModel.from_pretrained(base,adapter_dir,local_files_only=True)
    else: model=base
    model.eval(); return model,tok

def eval_cond(name,base_dir,adapter_dir,cases,indices,batch=32):
    import torch, gc
    model,tok=load_stack(base_dir,adapter_dir)
    corr=[]; valid=[]; per=[]
    for st in range(0,len(indices),batch):
        ids=indices[st:st+batch]; texts=[cases[i]['prompt'] for i in ids]
        enc=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=384)
        with torch.no_grad(): outs=model.generate(**enc,max_new_tokens=5,do_sample=False,num_beams=1)
        dec=tok.batch_decode(outs,skip_special_tokens=True)
        for i,t in zip(ids,dec):
            pred=normalize_label(t); ok=(pred==cases[i]['expected']); corr.append(ok); valid.append(pred!='INVALID'); per.append((i,ok,pred!='INVALID'))
    del model,tok; gc.collect()
    return {'name':name,'n':len(indices),'accuracy':sum(corr)/len(corr),'valid_rate':sum(valid)/len(valid),'correct':corr,'per':per}

def pair_roots(state):
    roots={}
    for k in state:
        if 'lora_A' in k: roots.setdefault(k.replace('lora_A','LORA'),{})['A']=k
        elif 'lora_B' in k: roots.setdefault(k.replace('lora_B','LORA'),{})['B']=k
    return {r:v for r,v in roots.items() if 'A' in v and 'B' in v}

def path_of(root):
    l=root.lower()
    if 'encoder' in l and 'selfattention' in l: return 'ENCODER_SELF'
    if 'decoder' in l and 'encdecattention' in l: return 'DECODER_CROSS'
    if 'decoder' in l and 'selfattention' in l: return 'DECODER_SELF'
    return 'OTHER'
def proj_of(root):
    l=root.lower()
    for p in ['q','k','v','o']:
        if re.search(rf'\.{p}(\.|$)',l): return p.upper()
    return 'OTHER'
def layer_of(root):
    m=re.search(r'block\.(\d+)',root); return int(m.group(1)) if m else -1

def make_variant(src_dir,dst_dir,kind,value=None):
    import torch
    from safetensors.torch import load_file, save_file
    src=Path(src_dir); dst=Path(dst_dir); shutil.rmtree(dst,ignore_errors=True); dst.mkdir(parents=True)
    shutil.copy2(src/'adapter_config.json',dst/'adapter_config.json')
    state=load_file(str(src/'adapter_model.safetensors')); new={k:v.clone() for k,v in state.items()}; roots=pair_roots(state)
    if kind=='dose':
        fac=float(value)
        for v in roots.values(): new[v['B']]=new[v['B']]*fac
    elif kind=='random_component_sign':
        g=torch.Generator().manual_seed(22569)
        for v in roots.values():
            B=new[v['B']]; signs=torch.randint(0,2,(B.shape[-1],),generator=g,dtype=torch.int64)*2-1; new[v['B']]=B*signs.to(B.dtype).reshape(1,-1)
    elif kind=='shuffle':
        rs=sorted(roots); byshape={}
        for r in rs:
            v=roots[r]; sig=(tuple(state[v['A']].shape),tuple(state[v['B']].shape)); byshape.setdefault(sig,[]).append(r)
        for group in byshape.values():
            if len(group)<2: continue
            shift=7%len(group) or 1; srcs=group[shift:]+group[:shift]
            for trg,sr in zip(group,srcs):
                new[roots[trg]['A']]=state[roots[sr]['A']].clone(); new[roots[trg]['B']]=state[roots[sr]['B']].clone()
    elif kind=='ablate':
        group=str(value)
        for r,v in roots.items():
            path=path_of(r); proj=proj_of(r); layer=layer_of(r)
            hit=(group==path or group==proj or (group=='EARLY' and 0<=layer<=2) or (group=='MIDDLE' and 3<=layer<=5) or (group=='LATE' and layer>=6))
            if hit: new[v['B']]=torch.zeros_like(new[v['B']])
    save_file(new,str(dst/'adapter_model.safetensors'))

def static_summary(adapter_dir):
    import torch
    from safetensors.torch import load_file
    cfg=json.loads((Path(adapter_dir)/'adapter_config.json').read_text()); scale=float(cfg.get('lora_alpha',1))/float(cfg.get('r',1))
    state=load_file(str(Path(adapter_dir)/'adapter_model.safetensors')); roots=pair_roots(state); groups={}; zeros=0; nonfinite=0; rows=[]
    for r,v in roots.items():
        A=state[v['A']].float(); B=state[v['B']].float();
        if not torch.isfinite(A).all() or not torch.isfinite(B).all(): nonfinite+=1
        if A.norm().item()==0 or B.norm().item()==0: zeros+=1
        e=(scale*A.norm().item()*B.norm().item())**2; path=path_of(r); proj=proj_of(r); layer=layer_of(r)
        for g in [path,proj,f'LAYER_{layer}']: groups[g]=groups.get(g,0.0)+e
        rows.append((r,e,path,proj,layer))
    tot=sum(x[1] for x in rows) or 1.0
    return {'pair_count':len(roots),'zero_pair_count':zeros,'nonfinite_pair_count':nonfinite,'rank':cfg.get('r'),'alpha':cfg.get('lora_alpha'),'target_modules':cfg.get('target_modules'),'energy_proxy_fraction':{k:v/tot for k,v in sorted(groups.items())},'raw_A_B_exported':False,'reconstructable_delta_exported':False}

def cmd_run(preseal,out,raw):
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    preseal=Path(preseal); out=Path(out); raw=Path(raw); out.mkdir(parents=True,exist_ok=True); shutil.rmtree(raw,ignore_errors=True); raw.mkdir(parents=True)
    cases=json.loads((preseal/'suite_private.json').read_text())
    api=HfApi(); info=api.model_info(ADAPTER_REPO,revision=ADAPTER_REV,files_metadata=True)
    assert info.sha==ADAPTER_REV
    lic=(getattr(info,'card_data',None).license if getattr(info,'card_data',None) else None)
    assert str(lic).lower()=='mit'
    ad=raw/'adapter'; ad.mkdir();
    for fn in ['adapter_config.json','adapter_model.safetensors']:
        p=Path(hf_hub_download(ADAPTER_REPO,fn,revision=ADAPTER_REV,local_dir=ad))
        if fn.endswith('.safetensors'): assert sha_file(p)==ADAPTER_SHA
    binfo=api.model_info(BASE_REPO,files_metadata=True); brev=binfo.sha
    base=raw/'base'; snapshot_download(BASE_REPO,revision=brev,local_dir=base,allow_patterns=['*.json','*.safetensors','tokenizer*','spiece.model','special_tokens_map.json'],ignore_patterns=['*.bin','*.pt','*.pth','*.pkl','*.pickle','*.h5','*.msgpack'])
    weight_files=sorted(base.glob('*.safetensors')); assert weight_files
    base_hashes={p.name:sha_file(p) for p in weight_files}
    static=static_summary(ad)
    all_idx=list(range(len(cases))); control_idx=[i for i,c in enumerate(cases) if c['partition'] in ('discovery','confirmation')][:192]
    dose_idx=[i for i,c in enumerate(cases) if c['partition']=='confirmation'][:96]
    results={}
    results['BASE']=eval_cond('BASE',base,None,cases,all_idx)
    results['FULL']=eval_cond('FULL',base,ad,cases,all_idx)
    variants=raw/'variants'; variants.mkdir()
    rnd=variants/'random'; make_variant(ad,rnd,'random_component_sign'); results['RANDOM_COMPONENT_SIGN']=eval_cond('RANDOM_COMPONENT_SIGN',base,rnd,cases,control_idx)
    shf=variants/'shuffle'; make_variant(ad,shf,'shuffle'); results['LAYER_MODULE_SHUFFLE']=eval_cond('LAYER_MODULE_SHUFFLE',base,shf,cases,control_idx)
    doses={}
    for fac in [.25,.5,1.0,1.5]:
        if fac==1.0: rr=eval_cond('DOSE_1.0',base,ad,cases,dose_idx)
        else:
            vd=variants/f'dose_{fac}'; make_variant(ad,vd,'dose',fac); rr=eval_cond(f'DOSE_{fac}',base,vd,cases,dose_idx)
        doses[str(fac)]={'accuracy':rr['accuracy'],'valid_rate':rr['valid_rate'],'n':rr['n']}
    basecorr=results['BASE']['correct']; fullcorr=results['FULL']['correct']; ci=bootstrap_ci(fullcorr,basecorr)
    def part_acc(key,part):
        r=results[key]; vals=[]
        for i,ok,_ in r['per']:
            if cases[i]['partition']==part: vals.append(ok)
        return sum(vals)/len(vals) if vals else None
    full_control=[ok for i,ok,_ in results['FULL']['per'] if i in set(control_idx)]
    base_control=[ok for i,ok,_ in results['BASE']['per'] if i in set(control_idx)]
    full_control_acc=sum(full_control)/len(full_control); base_control_acc=sum(base_control)/len(base_control)
    rand_acc=results['RANDOM_COMPONENT_SIGN']['accuracy']; shuf_acc=results['LAYER_MODULE_SHUFFLE']['accuracy']
    confirmation_gain=part_acc('FULL','confirmation')-part_acc('BASE','confirmation')
    ood_parts=['lexical_ood','compositional_ood']; ood_gain=sum(part_acc('FULL',p)-part_acc('BASE',p) for p in ood_parts)/len(ood_parts)
    positive=(results['FULL']['accuracy']-results['BASE']['accuracy']>=.08 and ci[0]>0 and full_control_acc-rand_acc>=.05 and full_control_acc-shuf_acc>=.05 and confirmation_gain>=.05 and ood_gain>=.02)
    failure=(results['BASE']['accuracy']-results['FULL']['accuracy']>=.08 and rand_acc-full_control_acc>=.05 and shuf_acc-full_control_acc>=.05)
    behavior={'schema':'R22569_C53_BEHAVIOR_V1','runtime_reference_base_revision':brev,'runtime_reference_base_safetensor_sha256':base_hashes,'training_time_exact_base_revision_proven':False,'suite':suite_public(cases),'metrics':{},'paired_full_minus_base_ci95':ci,'confirmation_gain':confirmation_gain,'mean_lexical_compositional_ood_gain':ood_gain,'full_control_subset_accuracy':full_control_acc,'base_control_subset_accuracy':base_control_acc,'random_component_sign_accuracy':rand_acc,'layer_module_shuffle_accuracy':shuf_acc,'dose_accuracy':doses,'positive_lane_green':positive,'failure_lane_green':failure,'causal_entry':positive or failure}
    for k,r in results.items(): behavior['metrics'][k]={'n':r['n'],'accuracy':r['accuracy'],'valid_rate':r['valid_rate']}
    causal={'schema':'R22569_C53_CAUSAL_V1','status':'NOT_RUN','reason':'BEHAVIOR_GATE_RED'}
    if positive or failure:
        causal_idx=[i for i,c in enumerate(cases) if c['partition'] in ('confirmation','lexical_ood','compositional_ood')][:96]
        fullc=eval_cond('FULL_CAUSAL_REFERENCE',base,ad,cases,causal_idx); groups=['ENCODER_SELF','DECODER_SELF','DECODER_CROSS','Q','K','V','O','EARLY','MIDDLE','LATE']; scores=[]
        for g in groups:
            vd=variants/f'ablate_{g}'; make_variant(ad,vd,'ablate',g); rr=eval_cond('ABLATE_'+g,base,vd,cases,causal_idx)
            effect=(fullc['accuracy']-rr['accuracy']) if positive else (rr['accuracy']-fullc['accuracy'])
            scores.append({'group':g,'accuracy':rr['accuracy'],'effect':effect})
        scores.sort(key=lambda x:x['effect'],reverse=True)
        causal={'schema':'R22569_C53_CAUSAL_V1','status':'E3_BROAD_SCREEN_ONLY','lane':'positive' if positive else 'failure','reference_accuracy':fullc['accuracy'],'n':len(causal_idx),'group_effects':scores,'fine_irredundancy_run':False,'independent_worker_replay':False,'claim_ceiling':'E3_SCOPED_RUNTIME_REFERENCE_ONLY'}
    counters=[]
    for i,(b,f) in enumerate(zip(basecorr,fullcorr)):
        if b!=f:
            counters.append({'id':cases[i]['id'],'partition':cases[i]['partition'],'family':cases[i]['family'],'expected':cases[i]['expected'],'base_correct':bool(b),'full_correct':bool(f)})
    # strip private predictions; export only scalar/boolean evidence.
    (out/'C53_STATIC.json').write_text(json.dumps(static,indent=2)); (out/'C53_BEHAVIOR.json').write_text(json.dumps(behavior,indent=2)); (out/'C53_CAUSAL.json').write_text(json.dumps(causal,indent=2)); (out/'C53_COUNTEREXAMPLES.json').write_text(json.dumps({'count':len(counters),'items':counters},indent=2))
    prov={'schema':'R22569_C53_PROVENANCE_V1','adapter_repo':ADAPTER_REPO,'adapter_revision':ADAPTER_REV,'adapter_sha256':ADAPTER_SHA,'adapter_license':'mit','runtime_reference_base_repo':BASE_REPO,'runtime_reference_base_revision':brev,'runtime_reference_base_hashes':base_hashes,'training_time_base_revision_proven':False,'raw_committed_to_git':False}
    (out/'C53_PROVENANCE.json').write_text(json.dumps(prov,indent=2))
    terminal={'schema':'R22569_C53_TERMINAL_V1','positive_lane_green':positive,'failure_lane_green':failure,'scientific_grade':('E3_SCOPED_RUNTIME_REFERENCE_ONLY' if causal.get('status')=='E3_BROAD_SCREEN_ONLY' else ('E2_BEHAVIOR_ONLY' if (positive or failure) else 'E1_STATIC_PLUS_BEHAVIOR_RED')),'behavior_increment':1 if (positive or failure) else 0,'causal_increment':1 if causal.get('status')=='E3_BROAD_SCREEN_ONLY' else 0,'algorithm_increment':0,'student_increment':0,'promotion_increment':0,'closed_no_redownload':True}
    (out/'C53_TERMINAL_EVIDENCE.json').write_text(json.dumps(terminal,indent=2))
    # exact raw cleanup before export
    shutil.rmtree(raw)
    assert not raw.exists()
    (out/'C53_CLEANUP.json').write_text(json.dumps({'raw_root_deleted':True,'raw_weights_remaining':0,'raw_tokenizer_remaining':0,'private_model_outputs_exported':False},indent=2))
    # portable hashes
    files=[p for p in sorted(out.glob('*.json'))]
    (out/'SHA256SUMS.txt').write_text(''.join(f'{sha_file(p)}  {p.name}\n' for p in files))
    print(json.dumps({'behavior':behavior,'causal':causal,'terminal':terminal},indent=2))

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('preseal'); p.add_argument('--out',required=True)
    r=sp.add_parser('run'); r.add_argument('--preseal',required=True); r.add_argument('--out',required=True); r.add_argument('--raw',required=True)
    a=ap.parse_args()
    if a.cmd=='preseal': cmd_preseal(a.out)
    else: cmd_run(a.preseal,a.out,a.raw)
if __name__=='__main__': main()
