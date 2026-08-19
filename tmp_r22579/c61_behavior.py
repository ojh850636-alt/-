from __future__ import annotations
import argparse,contextlib,hashlib,importlib.util,json,os,random,re,statistics,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
ADAPTER_SHA='f2cae681c0e17974108c46422a291d3b1f7ffc26960a99b5b9513b8d0a690820'
ADAPTER_REV='804fa33114143abac0df50c9ec140ce5ce628318'
BASE_REV='7ae557604adf67be50417f59c2c2f167def9a775'
BASE_SHA='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
BASE_SIZE=988097824
SUITE_SHA='22592968b52e19a1a68249d09c58e553632899ca3301ddd76d1cab5758a529ad'
def hfile(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def loadmod(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def select(rs):
    by={}
    for r in rs:by.setdefault(r['split'],[]).append(r)
    teacher=by['DISCOVERY'][:24]+by['CONFIRMATION'][:24]+by['ARITHMETIC_CROSSFIELD'][:24]+by['OCR_NOISE_OOD'][:24]
    gen=[]
    for s in ['DISCOVERY','CONFIRMATION','ARITHMETIC_CROSSFIELD','OCR_NOISE_OOD','NOOP_CLEAN','NEGATIVE_AMBIGUOUS']:gen+=by[s][:4]
    causal=by['CONFIRMATION'][:4]+by['ARITHMETIC_CROSSFIELD'][:4]
    assert (len(teacher),len(gen),len(causal))==(96,24,8);return teacher,gen,causal
def lora_modules(model,adapter='full'):
    out=[]
    for name,m in model.named_modules():
        if hasattr(m,'lora_A') and hasattr(m,'lora_B') and adapter in m.lora_A and adapter in m.lora_B and hasattr(m,'scaling') and adapter in m.scaling:out.append((name,m))
    return out
def init_controls(model,adapter_dir):
    model.load_adapter(adapter_dir,adapter_name='random');model.load_adapter(adapter_dir,adapter_name='shuffle');g=torch.Generator(device='cpu');g.manual_seed(2257901);mods=lora_modules(model,'full')
    with torch.no_grad():
        for _,m in mods:
            for side in ['lora_A','lora_B']:
                ref=getattr(m,side)['full'].weight;dst=getattr(m,side)['random'].weight;r=torch.randn(dst.shape,generator=g,dtype=dst.dtype,device=dst.device);dst.copy_(r*(torch.linalg.vector_norm(ref)/(torch.linalg.vector_norm(r)+1e-12)))
    buckets={}
    for name,m in mods:
        fam=next((z for z in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if z in name),'OTHER');key=(fam,tuple(m.lora_A['full'].weight.shape),tuple(m.lora_B['full'].weight.shape));buckets.setdefault(key,[]).append((name,m))
    rr=random.Random(2257902)
    with torch.no_grad():
        for vals in buckets.values():
            src=vals[:];rr.shuffle(src)
            if len(vals)>1 and all(a[0]==b[0] for a,b in zip(vals,src)):src=src[1:]+src[:1]
            for (_,dst),(_,sm) in zip(vals,src):dst.lora_A['shuffle'].weight.copy_(sm.lora_A['full'].weight);dst.lora_B['shuffle'].weight.copy_(sm.lora_B['full'].weight)
    return {'full_module_count':len(mods),'shuffle_buckets':len(buckets)}
def snapshot_scale(model,ad):return [(m,float(m.scaling[ad])) for _,m in lora_modules(model,ad)]
def restore_scale(s,ad):
    for m,v in s:m.scaling[ad]=v
@contextlib.contextmanager
def condition(model,name):
    if name=='BASE':
        with model.disable_adapter():yield
        return
    if name in {'FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED'}:model.set_adapter({'FULL':'full','RANDOM_RANK_MATCHED':'random','LAYER_SHUFFLED':'shuffle'}[name]);yield;return
    if name in {'DOSE_0.50','DOSE_1.50'}:
        f=.5 if name=='DOSE_0.50' else 1.5;model.set_adapter('full');s=snapshot_scale(model,'full')
        for m,v in s:m.scaling['full']=v*f
        try:yield
        finally:restore_scale(s,'full')
        return
    raise ValueError(name)
def prompt_ids(tok,r):
    msgs=[{'role':'system','content':r['instruction']},{'role':'user','content':r['ocr_text']}];pre=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True);full=tok.apply_chat_template(msgs+[{'role':'assistant','content':r['target_json']}],tokenize=True,add_generation_prompt=False);assert full[:len(pre)]==pre;return pre,full
def nll_one(model,tok,r):
    pre,full=prompt_ids(tok,r);ids=torch.tensor([full]);mask=torch.ones_like(ids);labels=ids.clone();labels[:,:len(pre)]=-100;out=model(input_ids=ids,attention_mask=mask,use_cache=False);logits=out.logits[:,:-1,:].float();y=labels[:,1:];loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='none').reshape(y.shape);v=y!=-100;val=float((loss*v).sum()/v.sum().clamp_min(1));del out,logits,loss;return val
def nll_cases(model,tok,rs,cond):
    vals=[];t=time.time();tok.padding_side='right'
    with condition(model,cond),torch.inference_mode():
        for r in rs:vals.append(nll_one(model,tok,r))
    return vals,time.time()-t
def prompt_text(tok,r):return tok.apply_chat_template([{'role':'system','content':r['instruction']},{'role':'user','content':r['ocr_text']}],tokenize=False,add_generation_prompt=True)
def generate(model,tok,rs,cond,max_new=192,batch_size=1):
    assert batch_size in (1,2);res=[];tok.padding_side='left';pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with condition(model,cond),torch.inference_mode():
        for i in range(0,len(rs),batch_size):
            bb=rs[i:i+batch_size];enc=tok([prompt_text(tok,r) for r in bb],return_tensors='pt',padding=True,add_special_tokens=False);out=model.generate(**enc,max_new_tokens=max_new,do_sample=False,num_beams=1,pad_token_id=pad,eos_token_id=tok.eos_token_id);start=enc['input_ids'].shape[1]
            for seq in out:res.append(tok.decode(seq[start:],skip_special_tokens=True).strip())
            del out
    return res
def runtime_smoke(model,tok,gen):
    probe=gen[:2];out={}
    for c in ['BASE','FULL']:
        a=generate(model,tok,probe,c,48,1);b=generate(model,tok,probe,c,48,1);bat=generate(model,tok,probe,c,48,2);ha=[hashlib.sha256(x.encode()).hexdigest() for x in a];hb=[hashlib.sha256(x.encode()).hexdigest() for x in b];hc=[hashlib.sha256(x.encode()).hexdigest() for x in bat];out[c]={'repeatability_batch1':ha==hb,'batch1_vs_batch2_diagnostic':ha==hc,'batch1_hashes':ha,'batch2_hashes':hc}
    out['canonical_batch1_repeatability_green']=all(out[c]['repeatability_batch1'] for c in ['BASE','FULL']);return out
def bootstrap_ci(vals,seed=22579,n=3000):
    rr=random.Random(seed);N=len(vals);a=[]
    for _ in range(n):a.append(sum(vals[rr.randrange(N)] for __ in range(N))/N)
    a.sort();return [a[int(.025*n)],a[int(.975*n)-1]]
def score_preds(A,B,rs,preds):
    details=[];agg={'json_valid':0,'strict':0,'field_correct':0.0,'null_discipline':0,'integrity':0}
    for r,p in zip(rs,preds):
        aa=A.score(p,r['target']);bb=B.integrity(p,r['target'])
        for k in ['json_valid','strict','field_correct','null_discipline']:agg[k]+=aa[k]
        agg['integrity']+=bb['integrity'];details.append({'case_id':r['case_id'],'split':r['split'],'prediction_sha256':hashlib.sha256(p.encode()).hexdigest(),'prediction_chars':len(p),'json_valid':aa['json_valid'],'strict':aa['strict'],'field_correct':aa['field_correct'],'null_discipline':aa['null_discipline'],'integrity':bb['integrity']})
    n=len(rs)
    for k in agg:agg[k]/=n
    return agg,details
def layer_of(name):
    m=re.search(r'\.layers\.(\d+)\.',name);return int(m.group(1)) if m else -1
def causal_match(name,h,nlayers):
    li=layer_of(name);a=nlayers//3;b=2*nlayers//3
    if h=='ATTENTION_ALL':return any(x in name for x in ['q_proj','k_proj','v_proj','o_proj'])
    if h=='MLP_ALL':return any(x in name for x in ['gate_proj','up_proj','down_proj'])
    if h=='EARLY_LAYERS':return 0<=li<a
    if h=='MIDDLE_LAYERS':return a<=li<b
    if h=='LATE_LAYERS':return li>=b
    if h=='QKV':return any(x in name for x in ['q_proj','k_proj','v_proj'])
    if h=='O_PROJ':return 'o_proj' in name
    if h=='GATE_UP':return 'gate_proj' in name or 'up_proj' in name
    if h=='DOWN_PROJ':return 'down_proj' in name
    return False
def causal_scan(model,tok,rs,full_ref,base_ref):
    hs=['ATTENTION_ALL','MLP_ALL','EARLY_LAYERS','MIDDLE_LAYERS','LATE_LAYERS','QKV','O_PROJ','GATE_UP','DOWN_PROJ'];nl=int(model.config.num_hidden_layers);out=[];model.set_adapter('full')
    for h in hs:
        snap=snapshot_scale(model,'full')
        for name,m in lora_modules(model,'full'):
            if causal_match(name,h,nl):m.scaling['full']=0.0
        abl=statistics.mean(nll_cases(model,tok,rs,'FULL')[0]);restore_scale(snap,'full');snap=snapshot_scale(model,'full')
        for name,m in lora_modules(model,'full'):
            if not causal_match(name,h,nl):m.scaling['full']=0.0
        suff=statistics.mean(nll_cases(model,tok,rs,'FULL')[0]);restore_scale(snap,'full');model.set_adapter('random');snap=snapshot_scale(model,'random')
        for name,m in lora_modules(model,'random'):
            if not causal_match(name,h,nl):m.scaling['random']=0.0
        rnd=statistics.mean(nll_cases(model,tok,rs,'RANDOM_RANK_MATCHED')[0]);restore_scale(snap,'random');model.set_adapter('full');out.append({'hypothesis':h,'necessity_delta_nll':abl-full_ref,'sufficiency_gain_vs_base':base_ref-suff,'random_group_gain_vs_base':base_ref-rnd,'ablated_nll':abl,'only_group_nll':suff})
    return {'broad':out,'best':max(out,key=lambda x:x['necessity_delta_nll'])}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--suite',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert hfile(Path(a.adapter)/'adapter_model.safetensors')==ADAPTER_SHA;assert hfile(Path(a.base)/'model.safetensors')==BASE_SHA;assert (Path(a.base)/'model.safetensors').stat().st_size==BASE_SIZE;assert hfile(a.suite)==SUITE_SHA
    A=loadmod('tmp_r22579/c61_verifier_a.py','A');B=loadmod('tmp_r22579/c61_verifier_b.py','B');rs=rows(a.suite);teacher,gen,causal=select(rs);torch.set_num_threads(max(1,int(os.environ.get('C61_THREADS','4'))));tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False);tok.pad_token=tok.eos_token if tok.pad_token_id is None else tok.pad_token;base=AutoModelForCausalLM.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,low_cpu_mem_usage=True);model=PeftModel.from_pretrained(base,a.adapter,adapter_name='full',is_trainable=False);model.eval();ctrl=init_controls(model,a.adapter);smoke=runtime_smoke(model,tok,gen);assert smoke['canonical_batch1_repeatability_green'],smoke
    nll={}
    for c in ['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0.50','DOSE_1.50']:
        vals,elapsed=nll_cases(model,tok,teacher,c);nll[c]={'mean':statistics.mean(vals),'values':vals,'elapsed_s':elapsed}
    gains=[x-y for x,y in zip(nll['BASE']['values'],nll['FULL']['values'])];ci=bootstrap_ci(gains);scores={};details={}
    for c in ['BASE','FULL']:
        t=time.time();p=generate(model,tok,gen,c,192,1);scores[c],details[c]=score_preds(A,B,gen,p);scores[c]['elapsed_s']=time.time()-t
    pg={'nll_gain_mean':statistics.mean(gains),'nll_gain_ci95':ci,'nll_margin_random':nll['RANDOM_RANK_MATCHED']['mean']-nll['FULL']['mean'],'nll_margin_shuffle':nll['LAYER_SHUFFLED']['mean']-nll['FULL']['mean'],'strict_gain':scores['FULL']['strict']-scores['BASE']['strict'],'integrity_gain':scores['FULL']['integrity']-scores['BASE']['integrity'],'json_valid_gain':scores['FULL']['json_valid']-scores['BASE']['json_valid'],'null_safety_degradation':scores['BASE']['null_discipline']-scores['FULL']['null_discipline'],'field_correct_gain':scores['FULL']['field_correct']-scores['BASE']['field_correct']};positive=(pg['nll_gain_mean']>=0.05 and ci[0]>0 and pg['nll_margin_random']>=0.02 and pg['nll_margin_shuffle']>=0.02 and pg['strict_gain']>=0.15 and pg['integrity_gain']>=0.15 and pg['json_valid_gain']>=0.10 and pg['null_safety_degradation']<=0.05);failure=((scores['FULL']['strict']-scores['BASE']['strict'])<=-0.125 or pg['null_safety_degradation']>=0.25) and (nll['FULL']['mean']>min(nll['RANDOM_RANK_MATCHED']['mean'],nll['LAYER_SHUFFLED']['mean']))
    causal_result=None
    if positive or failure:
        fref=statistics.mean(nll_cases(model,tok,causal,'FULL')[0]);bref=statistics.mean(nll_cases(model,tok,causal,'BASE')[0]);causal_result=causal_scan(model,tok,causal,fref,bref);causal_result.update({'full_ref_nll':fref,'base_ref_nll':bref})
    grade='E2_CONTROL_SEPARATED_POSITIVE' if positive else ('E2_CONTROL_SEPARATED_FAILURE' if failure else 'E1_E2_NOT_ADMITTED');export_nll={c:{'mean':v['mean'],'elapsed_s':v['elapsed_s'],'case_nll':[{'case_id':r['case_id'],'nll':x} for r,x in zip(teacher,v['values'])]} for c,v in nll.items()};result={'schema':'R22579_C61_BEHAVIOR_CAUSAL_V1','adapter_revision':ADAPTER_REV,'base_revision':BASE_REV,'base_new_source_credit':0,'training_time_base_revision_proven':False,'runtime':{'python':sys.version,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'peft':__import__('peft').__version__,'dtype':'float32','threads':torch.get_num_threads(),'controls':ctrl,'smoke':smoke},'gate':{'positive_green':positive,'failure_green':failure,'grade':grade,'metrics':pg},'scores':scores,'generation_case_receipts':details,'nll':export_nll,'causal':causal_result,'raw_output_text_exported':False,'raw_logits_exported':False,'raw_activations_exported':False,'model_outputs_observed':True};(out/'C61_BEHAVIOR_CAUSAL_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True));print(json.dumps(result['gate'],indent=2,sort_keys=True))
if __name__=='__main__':main()
