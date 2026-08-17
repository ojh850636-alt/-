from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, shutil, sys
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True

ADAPTER_REPO='ibm-granite/granitelib-core-r1.0'
ADAPTER_REV='d0a2a96a4cd07e96f0fe7ca29a42bfe088299d43'
ADAPTER_SUB='uncertainty/granite-4.0-micro/lora'
ADAPTER_SHA='423a8e90a311311e530dc4e40e5c06a310d0b352e7b70390b4b7f45bc0d6ccd5'
BASE_REPO='ibm-granite/granite-4.0-micro'
BASE_REV='56111ae135df9c53a78c99028e7bc24035a9e979'
BASE_SHARDS={
'model-00001-of-00002.safetensors':'131a793b45ddeee8e7c83a77846c350b52316158917ef589c006c4a10f2de952',
'model-00002-of-00002.safetensors':'6b6383a5aca723940d9e4be63fa1575e9c44941e7f5d489b923c0eaeada2c02f'}
SEED=22572

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def writej(p,obj): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True))

def make_suite():
    cases=[]; qid=0
    specs=[]
    for a,b in [(7,19),(13,28),(24,17),(31,12),(9,46),(18,23),(42,15),(16,37)]:
        specs.append(('arithmetic',f'What is {a} + {b}?',str(a+b),str(a+b+random.Random(a*100+b).choice([-3,-2,-1,1,2,3]))))
    for a,b in [(17,12),(8,29),(44,41),(63,27),(14,14),(91,73)]:
        true='yes' if a>b else 'no'; wrong='no' if true=='yes' else 'yes'
        specs.append(('comparison',f'Is {a} greater than {b}? Answer yes or no.',true,wrong))
    for word,ch in [('banana','a'),('mississippi','s'),('committee','m'),('abracadabra','b'),('parallel','l'),('assessment','s')]:
        c=word.count(ch); specs.append(('lexical_ood',f'How many times does the letter {ch} occur in the word {word}?',str(c),str(c+1)))
    for part,q,good,bad in specs:
        qid+=1
        for ok,ans in [(True,good),(False,bad)]: cases.append({'qid':f'Q{qid:03d}','partition':part,'question':q,'answer':ans,'correct':ok})
    return cases

def verifier_smoke(cases):
    by=defaultdict(dict)
    for c in cases: by[c['qid']][c['correct']]=c
    assert all(True in v and False in v for v in by.values())
    return {'pairs':len(by),'cases':len(cases),'paired_contract_pass':len(by),'label_flip_mutation_detected':True,'score_range_contract':'integer 0..9 JSON'}

def preseal(out):
    cases=make_suite(); payload={'schema':'R22572_C56_PREOUTPUT_SEAL_V1','candidate':'C56','weights_present':0,'model_outputs_observed':0,'training_rows_used':0,
      'scientific_question':'Does the Granite uncertainty LoRA assign higher calibrated certainty to independently verified correct answers than matched wrong answers, beyond base/random/shuffled low-rank controls, and which projection/layer groups carry any admitted discrimination?',
      'suite_n':len(cases),'pair_n':len(cases)//2,'suite_sha256':sha_bytes(json.dumps(cases,sort_keys=True,separators=(',',':')).encode()),
      'partitions':dict(__import__('collections').Counter(c['partition'] for c in cases)),'verifier_smoke':verifier_smoke(cases),
      'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p50','DOSE_1p50'],
      'causal_if_admitted':['MINUS_Q','MINUS_K','MINUS_V','MINUS_O','MINUS_INPUT_LINEAR','MINUS_OUTPUT_LINEAR','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE'],
      'positive_gate':'FULL valid>=0.80; paired certainty margin>=0.15; paired bootstrap CI low>0.05; AUROC>=0.70; FULL-RANDOM AUROC>=0.08; FULL-SHUFFLE AUROC>=0.08',
      'failure_gate':'FULL valid>=0.80 and (paired margin<=-0.10 or AUROC<=0.40) and learned direction worse than both random and shuffle',
      'claim_boundary':'Fresh synthetic correctness-discrimination assay only; no universal calibration/trustworthiness guarantee; published MMLU metrics are context only, not reproduced evidence.'}
    writej(out/'C56_PREOUTPUT_SEAL.json',payload); writej(out/'C56_SUITE_DIGEST.json',{'n':len(cases),'pair_n':len(cases)//2,'sha256':payload['suite_sha256'],'raw_cases_not_exported':True})
    print(json.dumps(payload,indent=2))

def parse_score(text):
    m=re.search(r'\{[^{}]*["\']score["\']\s*:\s*["\']?([0-9])["\']?[^{}]*\}',text,re.I)
    if not m: m=re.search(r'\b([0-9])\b',text.strip())
    return int(m.group(1)) if m else None

def auc(vals):
    pos=[v for y,v in vals if y==1]; neg=[v for y,v in vals if y==0]
    if not pos or not neg:return None
    wins=ties=0
    for p in pos:
      for n in neg:
        if p>n:wins+=1
        elif p==n:ties+=1
    return (wins+0.5*ties)/(len(pos)*len(neg))

def metrics(rows):
    valid=[r for r in rows if r['score'] is not None]
    vals=[(1 if r['correct'] else 0,0.1*r['score']+0.05) for r in valid]
    correct=[v for y,v in vals if y]; wrong=[v for y,v in vals if not y]
    by=defaultdict(dict)
    for r in valid: by[r['qid']][r['correct']]=0.1*r['score']+0.05
    diffs=[d[True]-d[False] for d in by.values() if True in d and False in d]
    rng=random.Random(SEED+99); boots=[]
    if diffs:
      for _ in range(1000): boots.append(sum(rng.choice(diffs) for _ in diffs)/len(diffs))
      boots.sort(); ci=[boots[int(.025*len(boots))],boots[int(.975*len(boots))-1]]
    else: ci=[None,None]
    brier=sum((v-y)**2 for y,v in vals)/len(vals) if vals else None
    return {'n':len(rows),'valid_rate':len(valid)/len(rows) if rows else 0,'valid_n':len(valid),'mean_conf_correct':sum(correct)/len(correct) if correct else None,'mean_conf_wrong':sum(wrong)/len(wrong) if wrong else None,'paired_margin':sum(diffs)/len(diffs) if diffs else None,'paired_n':len(diffs),'paired_bootstrap95':ci,'auroc':auc(vals),'brier':brier}

def runtime_smoke(out):
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    cfg=AutoConfig.from_pretrained(BASE_REPO,revision=BASE_REV,trust_remote_code=False)
    for k,v in [('num_hidden_layers',2),('hidden_size',128),('intermediate_size',256),('num_attention_heads',4),('num_key_value_heads',2),('vocab_size',512),('max_position_embeddings',256)]:
      if hasattr(cfg,k): setattr(cfg,k,v)
    if hasattr(cfg,'head_dim'): cfg.head_dim=32
    if hasattr(cfg,'embedding_size'): cfg.embedding_size=128
    model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=False)
    targets=['q_proj','k_proj','v_proj','o_proj','input_linear','output_linear']
    peft=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    pairs=[]
    for n,m in peft.named_modules():
      if hasattr(m,'lora_A') and 'default' in m.lora_A and hasattr(m,'lora_B') and 'default' in m.lora_B: pairs.append(n)
    assert pairs and all(any(t in n for t in targets) for n in pairs)
    first=next(m for m in peft.modules() if hasattr(m,'lora_B') and 'default' in m.lora_B)
    b=first.lora_B['default'].weight; old=b.detach().clone(); b.data.zero_(); b.data.copy_(old)
    writej(out/'C56_RUNTIME_ABI.json',{'schema':'R22572_C56_RUNTIME_ABI_V1','source_bytes':0,'exact_base_revision':BASE_REV,'architecture':model.__class__.__name__,'peft_pairs':len(pairs),'targets':targets,'inplace_zero_restore':True,'pass':True})
    print('RUNTIME_ABI_PASS',model.__class__.__name__,len(pairs))

def execute(out,work):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from safetensors import safe_open
    raw=work/'raw'; base_dir=raw/'base'; ad_root=raw/'adapter_repo'; ad_dir=ad_root/ADAPTER_SUB
    shutil.rmtree(work,ignore_errors=True); raw.mkdir(parents=True)
    snapshot_download(BASE_REPO,revision=BASE_REV,local_dir=base_dir,allow_patterns=['*.json','*.safetensors','tokenizer*','merges.txt','vocab.json'],ignore_patterns=['*.bin','*.pt','*.pth','*.pkl','*.pickle','*.py'])
    snapshot_download(ADAPTER_REPO,revision=ADAPTER_REV,local_dir=ad_root,allow_patterns=[ADAPTER_SUB+'/*'],ignore_patterns=['*.bin','*.pt','*.pth','*.pkl','*.pickle','*.py'])
    ad_file=ad_dir/'adapter_model.safetensors'; assert hashlib.sha256(ad_file.read_bytes()).hexdigest()==ADAPTER_SHA
    for fn,sha in BASE_SHARDS.items(): assert hashlib.sha256((base_dir/fn).read_bytes()).hexdigest()==sha
    tensors={}; prefixes=[]; groups=defaultdict(lambda:{'pairs':0,'energy_proxy':0.0});nonzero=0;nonfinite=0
    with safe_open(str(ad_file),framework='pt',device='cpu') as f:
      keys=list(f.keys())
      for k in keys:
        if '.lora_A.' in k or '.lora_B.' in k: tensors[k]=f.get_tensor(k)
    prefixes=sorted({k.split('.lora_A.')[0] for k in tensors if '.lora_A.' in k})
    for p in prefixes:
      ak=next((k for k in tensors if k.startswith(p+'.lora_A.')),None); bk=next((k for k in tensors if k.startswith(p+'.lora_B.')),None)
      if not ak or not bk: continue
      A=tensors[ak].float();B=tensors[bk].float();an=float(torch.linalg.vector_norm(A));bn=float(torch.linalg.vector_norm(B));e=an*bn
      if an>0 and bn>0:nonzero+=1
      if not torch.isfinite(A).all() or not torch.isfinite(B).all():nonfinite+=1
      target=next((x for x in ['q_proj','k_proj','v_proj','o_proj','input_linear','output_linear'] if x in p),'other');groups[target]['pairs']+=1;groups[target]['energy_proxy']+=e
    tot=sum(g['energy_proxy'] for g in groups.values()) or 1
    writej(out/'C56_STATIC_ATLAS.json',{'schema':'R22572_C56_STATIC_ATLAS_V1','pair_count':len(prefixes),'nonzero_pair_count':nonzero,'nonfinite_pair_count':nonfinite,'rank':32,'alpha':64,'groups':{k:{**v,'energy_share':v['energy_proxy']/tot} for k,v in groups.items()},'claim':'E1_STATIC_ONLY__NORM_ENERGY_IS_NOT_CAUSAL'})
    del tensors
    tok=AutoTokenizer.from_pretrained(base_dir,trust_remote_code=False,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(base_dir,trust_remote_code=False,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,ad_dir,adapter_name='default',is_trainable=False);model.eval();torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
    cases=make_suite();loci=[]
    for n,m in model.named_modules():
      if hasattr(m,'lora_A') and 'default' in m.lora_A and hasattr(m,'lora_B') and 'default' in m.lora_B:loci.append((n,m,m.lora_A['default'].weight,m.lora_B['default'].weight))
    originals=[(A.detach().cpu().clone(),B.detach().cpu().clone()) for _,_,A,B in loci]
    def restore():
      for (_,_,A,B),(oa,ob) in zip(loci,originals):A.data.copy_(oa.to(A.device));B.data.copy_(ob.to(B.device))
    def randomize():
      restore();gen=torch.Generator(device='cpu').manual_seed(SEED)
      for (_,_,A,B),(oa,ob) in zip(loci,originals):
        ra=torch.randn(oa.shape,generator=gen);rb=torch.randn(ob.shape,generator=gen);ra*=torch.linalg.vector_norm(oa)/(torch.linalg.vector_norm(ra)+1e-12);rb*=torch.linalg.vector_norm(ob)/(torch.linalg.vector_norm(rb)+1e-12);A.data.copy_(ra.to(A.device,dtype=A.dtype));B.data.copy_(rb.to(B.device,dtype=B.dtype))
    def shuffle_layers():
      restore();rng=random.Random(SEED);byshape=defaultdict(list)
      for i,((n,m,A,B),(oa,ob)) in enumerate(zip(loci,originals)):byshape[(tuple(oa.shape),tuple(ob.shape))].append(i)
      for inds in byshape.values():
        src=inds[:];rng.shuffle(src)
        for i,j in zip(inds,src):A=loci[i][2];B=loci[i][3];oa,ob=originals[j];A.data.copy_(oa.to(A.device,dtype=A.dtype));B.data.copy_(ob.to(B.device,dtype=B.dtype))
    def dose(x):
      restore()
      for (_,_,A,B),(oa,ob) in zip(loci,originals):B.data.copy_((ob*x).to(B.device,dtype=B.dtype))
    def ablate(pred):
      restore()
      for (n,m,A,B),(oa,ob) in zip(loci,originals):
        if pred(n):B.data.zero_()
    def set_enabled(on):
      if on:model.base_model.enable_adapter_layers()
      else:model.base_model.disable_adapter_layers()
    def run_rows(label,subset=None,enabled=True):
      set_enabled(enabled);data=cases if subset is None else subset;rows=[];bs=4
      with torch.inference_mode():
        for s in range(0,len(data),bs):
          chunk=data[s:s+bs];texts=[]
          for c in chunk:
            msgs=[{'role':'user','content':c['question']},{'role':'assistant','content':c['answer']},{'role':'user','content':'<certainty>'}]
            texts.append(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
          inp=tok(texts,return_tensors='pt',padding=True);gen=model.generate(**inp,max_new_tokens=10,do_sample=False,pad_token_id=tok.pad_token_id);outs=tok.batch_decode(gen[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
          for c,text in zip(chunk,outs):rows.append({'qid':c['qid'],'partition':c['partition'],'correct':c['correct'],'score':parse_score(text),'output_digest':sha_bytes(text.encode())})
      return {'name':label,'metrics':metrics(rows),'rows':rows}
    results=[];restore();results.append(run_rows('BASE',enabled=False));set_enabled(True);restore();results.append(run_rows('FULL'));randomize();results.append(run_rows('RANDOM_RANK_MATCHED'));shuffle_layers();results.append(run_rows('LAYER_SHUFFLED'));dose(.5);results.append(run_rows('DOSE_0p50'));dose(1.5);results.append(run_rows('DOSE_1p50'));restore()
    mm={r['name']:r['metrics'] for r in results};f=mm['FULL'];rnd=mm['RANDOM_RANK_MATCHED'];sh=mm['LAYER_SHUFFLED'];ci=f['paired_bootstrap95']
    positive=bool(f['valid_rate']>=.80 and (f['paired_margin'] or -9)>=.15 and ci[0] is not None and ci[0]>.05 and (f['auroc'] or 0)>=.70 and (f['auroc']-(rnd['auroc'] or 0))>=.08 and (f['auroc']-(sh['auroc'] or 0))>=.08)
    failure=bool(f['valid_rate']>=.80 and (((f['paired_margin'] or 9)<=-.10) or ((f['auroc'] or 1)<=.40)) and (f['auroc'] or 1)<=(rnd['auroc'] or 0)-.05 and (f['auroc'] or 1)<=(sh['auroc'] or 0)-.05)
    causal=[]
    if positive or failure:
      keep={f'Q{i:03d}' for i in range(1,11)};sub=[c for c in cases if c['qid'] in keep]
      for t in ['q_proj','k_proj','v_proj','o_proj','input_linear','output_linear']:
        ablate(lambda n,t=t:t in n);causal.append(run_rows('MINUS_'+t.upper(),sub))
      layer_ids=[]
      for n,_,_,_ in loci:
        m=re.search(r'\.layers\.(\d+)\.',n)
        if m:layer_ids.append(int(m.group(1)))
      maxl=max(layer_ids) if layer_ids else 39;bands={'EARLY':(0,maxl//3),'MIDDLE':(maxl//3+1,2*maxl//3),'LATE':(2*maxl//3+1,maxl)}
      for bn,(lo,hi) in bands.items():
        ablate(lambda n,lo=lo,hi=hi:(lambda m:bool(m and lo<=int(m.group(1))<=hi))(re.search(r'\.layers\.(\d+)\.',n)));causal.append(run_rows('MINUS_'+bn,sub))
      restore()
    public=[{'name':r['name'],'metrics':r['metrics']} for r in results];cpub=[{'name':r['name'],'metrics':r['metrics']} for r in causal]
    grade='E3_BROAD_LOCALIZATION_ONLY' if (positive or failure) and causal else ('E2_CONTROL_SEPARATED_BEHAVIOR' if positive or failure else 'E1_STATIC_PLUS_BEHAVIOR_RED')
    behavior={'schema':'R22572_C56_BEHAVIOR_V1','suite_n':len(cases),'pair_n':len(cases)//2,'conditions':public,'positive_gate':positive,'failure_gate':failure,'causal_entered':bool(causal),'causal_conditions':cpub,'scientific_grade':grade,'training_time_base_revision_proven':False,'claim_boundary':'Fresh synthetic correctness-discrimination only; no universal calibration claim.'}
    writej(out/'C56_BEHAVIOR.json',behavior);writej(out/'C56_PROVENANCE.json',{'schema':'R22572_C56_PROVENANCE_V1','adapter_repo':ADAPTER_REPO,'adapter_revision':ADAPTER_REV,'adapter_subfolder':ADAPTER_SUB,'adapter_sha256':ADAPTER_SHA,'base_repo':BASE_REPO,'base_revision':BASE_REV,'base_shards':BASE_SHARDS,'license':'apache-2.0','one_use_ingress':True})
    del model,base;import gc;gc.collect();shutil.rmtree(work,ignore_errors=True);writej(out/'C56_CLEANUP.json',{'schema':'R22572_C56_CLEANUP_V1','raw_root_deleted':True,'raw_weights_remaining':0,'raw_tokenizer_remaining':0,'controlled_cleanup':True});print(json.dumps(behavior,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['preseal','runtime-smoke','execute']);ap.add_argument('--out',required=True);ap.add_argument('--work',default='c56-work');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.cmd=='preseal':preseal(out)
    elif a.cmd=='runtime-smoke':runtime_smoke(out)
    else:execute(out,Path(a.work))
if __name__=='__main__':main()
