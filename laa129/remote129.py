"""Public RoBERTa CoLA two-seed raw-isolated CPU experiment.
Never include original weights, tensors, activations or logits in export.
No source edits/weight merges. Trained classifier and LoRA controlled separately.
Requires sealed suite.jsonl and lock129.json. Outside network sandbox ONLY.
"""
from __future__ import annotations
import json,hashlib,math,os,re,shutil,sys,tempfile,traceback,time,collections,random
from pathlib import Path

os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import HfApi,snapshot_download
from transformers import AutoTokenizer,AutoModelForSequenceClassification

ROOT=Path(__file__).resolve().parent
SUITE=ROOT/'suite129.jsonl'
LOCK=ROOT/'lock129.json'
OUT=Path(os.environ.get('LAA129_OUTPUT','result129.json')).resolve()

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def load_lock():
    lock=json.loads(LOCK.read_text(encoding='utf-8'))
    if sha_file(SUITE)!=lock['suite_file_sha256']:raise ValueError('SUITE_LOCK_MISMATCH')
    suite=[json.loads(line) for line in SUITE.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(suite)!=224 or len({r['id'] for r in suite})!=224:raise ValueError('SUITE_INVALID')
    if lock['base_repo']!='FacebookAI/roberta-base' or lock['candidate_task']!='cola':raise ValueError('BAD_SOURCE')
    return lock,suite

def checked_sources(lock,root):
    repo=lock['repo']; revision=lock['adapter_revision']
    if HfApi().model_info(repo,revision=revision).sha!=revision:raise ValueError('ADAPTER_REVISION_DRIFT')
    adapter=root/'adapter'
    safe_files=[f'cola/seed{s}/{name}' for s in (0,1) for name in ('adapter_model.safetensors','adapter_config.json','train_meta.json')]
    snapshot_download(repo,revision=revision,allow_patterns=safe_files,local_dir=adapter)
    for seed in (0,1):
        wf=adapter/f'cola/seed{seed}/adapter_model.safetensors'
        source=lock['seed_sources'][str(seed)]
        if wf.stat().st_size!=source['size'] or sha_file(wf)!=source['sha256']:raise ValueError('ADAPTER_SHA_MISMATCH')
        cfg=json.loads((wf.parent/'adapter_config.json').read_text())
        tr=json.loads((wf.parent/'train_meta.json').read_text())
        if tr['recipe']['base_revision']!=lock['base_revision']:raise ValueError('TRAIN_RECIPE_BASE_MISMATCH')
        if cfg['r']!=8 or cfg['lora_alpha']!=16 or cfg['bias']!='none':raise ValueError('PEFT_CONFIG_MISMATCH')
        if set(cfg['target_modules'])!={'query','key','value','output.dense','intermediate.dense'}:raise ValueError('TARGET_SET_MISMATCH')
        if tr['budget']['num_labels']!=2:raise ValueError('LABEL_MAP_MISMATCH')
    base=root/'base'
    allow=['model.safetensors','config.json','tokenizer.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json']
    snapshot_download(lock['base_repo'],revision=lock['base_revision'],allow_patterns=allow,local_dir=base)
    bp=base/'model.safetensors'
    if bp.stat().st_size!=lock['base_size'] or sha_file(bp)!=lock['base_sha256']:raise ValueError('BASE_SHA_MISMATCH')
    return adapter,base

PAT=re.compile(r'^(?:base_model\.model\.)?(.+?)\.lora_([AB])(?:\.default)?\.weight$')
HEAD=re.compile(r'^(?:base_model\.model\.)?classifier\.modules_to_save\.[^.]+\.(.+)$')

def read_adapter(path,expected_count=72):
    pairs=collections.defaultdict(dict);head={};tensor_count=0
    with safe_open(str(path),framework='pt',device='cpu') as f:
        for k in f.keys():
            t=f.get_tensor(k).to(torch.float32)
            if not bool(torch.isfinite(t).all()):raise ValueError('NONFINITE_TENSOR')
            a=PAT.fullmatch(k);b=HEAD.fullmatch(k)
            if a:pairs[a.group(1)][a.group(2)]=t
            elif b:head[b.group(1)]=t
            else:raise ValueError('UNSUPPORTED_ADAPTER_COMPONENT:'+k[:100])
            tensor_count+=1
    if len(pairs)!=expected_count or any(set(x)!={'A','B'} for x in pairs.values()):raise ValueError('LORA_PAIR_COUNT_OR_MISSING')
    if set(head)!={'dense.weight','dense.bias','out_proj.weight','out_proj.bias'}:raise ValueError('TRAINED_HEAD_NOT_COMPLETE:'+str(sorted(head)))
    for k,v in pairs.items():
        if v['A'].ndim!=2 or v['B'].ndim!=2 or v['A'].shape[0]!=8 or v['B'].shape[1]!=8:raise ValueError('LORA_SHAPE')
    if tensor_count!=148:raise ValueError('ADAPTER_TENSOR_COUNT')
    stats={'site_count':len(pairs),'full_head_tensors':len(head),
           'head_elements':sum(x.numel() for x in head.values()),
           'nonzero_operators':0,'exact_zero_operators':0,
           'nonfinite_values':0}
    for v in pairs.values():
        a,b=v['A'],v['B']
        energy=torch.trace((a@a.T)@(b.T@b)).item()
        if energy<1e-18:stats['exact_zero_operators']+=1
        else:stats['nonzero_operators']+=1
    return dict(pairs),head,stats

def rand_lora(pairs,seed):
    g=torch.Generator().manual_seed(seed)
    new={}
    for name,x in pairs.items():
        a,b=x['A'],x['B']
        ar=torch.randn(a.shape,generator=g);br=torch.randn(b.shape,generator=g)
        original=torch.trace((a@a.T)@(b.T@b)).item()
        random_energy=torch.trace((ar@ar.T)@(br.T@br)).item()
        if random_energy<1e-15:raise ValueError('RANDOM_ENERGY_ZERO')
        br*=math.sqrt(max(0,original)/random_energy)
        new[name]={'A':ar,'B':br}
    return new

def shuffled_lora(pairs,seed):
    g=random.Random(seed);groups=collections.defaultdict(list)
    for name,x in pairs.items():groups[(tuple(x['A'].shape),tuple(x['B'].shape))].append(name)
    out={}
    for grp,names in groups.items():
        ordered=sorted(names);perm=ordered[:];g.shuffle(perm)
        if len(perm)>1 and perm==ordered:perm=perm[1:]+perm[:1]
        for n,m in zip(ordered,perm):out[n]=pairs[m]
    return out

def head_loader(model,weights):
    own=model.classifier.state_dict()
    if set(own)!=set(weights):raise ValueError('HEAD_KEYS_DO_NOT_MATCH')
    for k in own:
        if own[k].shape!=weights[k].shape:raise ValueError('HEAD_SHAPE_MISMATCH')
    model.classifier.load_state_dict(weights,strict=True)

def attach(model,pairs,scale=1.0,skip_layers=()):
    modules=dict(model.named_modules());handles=[];found=[]
    for key,x in sorted(pairs.items()):
        target=key
        if target not in modules:raise ValueError('MODULE_NOT_FOUND:'+target)
        if any(f'.encoder.layer.{i}.' in target for i in skip_layers):continue
        module=modules[target]
        a,b=x['A'],x['B']
        if a.shape[1]!=module.in_features or b.shape[0]!=module.out_features:raise ValueError('SHAPE_NOT_LINEAR:'+target)
        def make_hook(a,b):
            def hook(m,inputs,output):
                delta=(inputs[0].float() @ a.T) @ b.T
                return output + delta.to(output.dtype)*(2.0*scale)
            return hook
        handles.append(module.register_forward_hook(make_hook(a,b)))
        found.append(target)
    return handles,found

def get_logits(model,batches):
    out=[]
    with torch.inference_mode():
        for chunk in batches:
            logits=model(**chunk).logits
            if logits.shape[-1]!=2 or not bool(torch.isfinite(logits).all()):raise ValueError('BAD_LOGITS')
            out.extend(logits.argmax(-1).cpu().tolist())
    return out

def parity_check(base,adapter_dir,model,heads,pairs,tok,texts):
    # Full PeFT reference in a separate object; manual hooks must match.
    from peft import PeftModel
    torch.manual_seed(4242)
    ref=AutoModelForSequenceClassification.from_pretrained(str(base),num_labels=2,local_files_only=True,use_safetensors=True,trust_remote_code=False)
    ref=PeftModel.from_pretrained(ref,str(adapter_dir),local_files_only=True,is_trainable=False)
    ref.eval()
    x=tok(texts[:6],padding=True,truncation=True,max_length=128,return_tensors='pt')
    head_loader(model,heads)
    h,_=attach(model,pairs)
    try:
        with torch.inference_mode():custom=model(**x).logits
    finally:
        for hook in h:hook.remove()
    with torch.inference_mode():reference=ref(**x).logits
    error=float((custom-reference).abs().max().item())
    del ref
    if not math.isfinite(error) or error>0.0003:raise ValueError('NATIVE_PEFT_PARITY_FAIL:'+str(error))
    return error

def run():
    t0=time.time();torch.set_num_threads(min(4,os.cpu_count() or 1))
    lock,rows=load_lock()
    final={'schema':'LAA129_REMOTE_EXPERIMENT_V1','source':{'repo':lock['repo'],'adapter_revision':lock['adapter_revision'],
                'base_repo':lock['base_repo'],'base_revision':lock['base_revision'],'source_sha256':lock['seed_sources']},
           'protocol_digest':sha_file(LOCK),'suite_digest':sha_file(SUITE),
           'condition_definitions':{'BASE':'random reference head+base backbone','HEAD_ONLY':'trained head+base backbone',
            'LORA_ONLY':'random reference head+learned LoRA','FULL':'trained head+learned LoRA','SHUFFLED':'trained head+site-shuffled LoRA same shapes',
            'RANDOM':'trained head+rank-matched random LoRA same norm','HALF':'trained head+LoRA x0.5','HIGH':'trained head+LoRA x1.5'},
           'created_external_source_files_in_delivery':False,'raw_logits_in_delivery':False,'raw_activations_in_delivery':False,
           'neural_forward_executed':False,'causal_circuit_supported':False,'algorithm_exported':False,'student_exported':False,'promoted':False,
           'seed_receipts':{},'prediction_rows':[],'status':'INITIALIZED'}
    with tempfile.TemporaryDirectory(prefix='laa129_src_') as tmp:
        root=Path(tmp)
        os.environ['HF_HOME']=str(root/'cache')
        os.environ['HUGGINGFACE_HUB_CACHE']=str(root/'cache'/'hub')
        adapter,base=checked_sources(lock,root)
        tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,use_fast=True)
        texts=[r['text'] for r in rows]
        batches=[]
        for i in range(0,len(rows),16):batches.append(tok(texts[i:i+16],padding=True,truncation=True,max_length=128,return_tensors='pt'))
        for seed in (0,1):
            path=adapter/f'cola/seed{seed}'
            pairs,head,structure=read_adapter(path/'adapter_model.safetensors')
            torch.manual_seed(4242)
            model=AutoModelForSequenceClassification.from_pretrained(str(base),num_labels=2,local_files_only=True,use_safetensors=True,trust_remote_code=False)
            model.eval()
            base_head={k:v.detach().clone() for k,v in model.classifier.state_dict().items()}
            error=parity_check(base,path,model,head,pairs,tok,texts)
            result={'seed':seed,'structural':structure,'native_peft_parity_max_abs_logit_error':error,'conditions':{},'elapsed_s':None}
            plan=['BASE','HEAD_ONLY','LORA_ONLY','FULL','SHUFFLED','RANDOM','HALF','HIGH']
            for cond in plan:
                head_loader(model,base_head if cond in ('BASE','LORA_ONLY') else head)
                lora=None;scale=1.0
                if cond in ('LORA_ONLY','FULL'):lora=pairs
                elif cond=='SHUFFLED':lora=shuffled_lora(pairs,20269+seed)
                elif cond=='RANDOM':lora=rand_lora(pairs,12029+seed)
                elif cond=='HALF':lora=pairs;scale=0.5
                elif cond=='HIGH':lora=pairs;scale=1.5
                handles=[]
                try:
                    if lora:handles,_=attach(model,lora,scale)
                    yhat=get_logits(model,batches)
                finally:
                    for hook in handles:hook.remove()
                acc={}
                for split in sorted({r['split'] for r in rows}):
                    ix=[i for i,r in enumerate(rows) if r['split']==split]
                    acc[split]=round(sum(yhat[i]==rows[i]['label'] for i in ix)/len(ix),6)
                result['conditions'][cond]={'accuracy_by_split':acc,'prediction_digest':sha_bytes(bytes(yhat))}
                final['prediction_rows'].extend({'seed':seed,'condition':cond,'id':case['id'],'pred':int(pred)} for case,pred in zip(rows,yhat))
            # Precommitted quarantined diagnostic: run broad ablation if material change seen,
            # independently of direction; no promotion without independent workers.
            diffs=[result['conditions']['FULL']['accuracy_by_split'][s]-result['conditions']['HEAD_ONLY']['accuracy_by_split'][s]
                   for s in ('CONFIRMATION','OOD_FAMILY')]
            if any(abs(x)>=0.10 for x in diffs):
                for label,skips in [('ABL_EARLY',range(0,4)),('ABL_MIDDLE',range(4,8)),('ABL_LATE',range(8,12))]:
                    head_loader(model,head)
                    hooks,_=attach(model,pairs,skip_layers=skips)
                    try:yp=get_logits(model,batches)
                    finally:
                        for h in hooks:h.remove()
                    result['conditions'][label]={'accuracy_by_split':{split:round(sum(yp[i]==rows[i]['label'] for i,r in enumerate(rows) if r['split']==split)/sum(r['split']==split for r in rows),6) for split in ('CONFIRMATION','OOD_FAMILY')},
                        'prediction_digest':sha_bytes(bytes(yp))}
                    final['prediction_rows'].extend({'seed':seed,'condition':label,'id':case['id'],'pred':int(pred)} for case,pred in zip(rows,yp))
                result['causal_diagnostic']='EXPLORATORY_BROAD_ABLATION_ONLY_NO_MINIMALITY'
            else:result['causal_diagnostic']='NOT_RUN_PRECOMMIT_BEHAVIOR_MAGNITUDE_GATE'
            result['elapsed_s']=round(time.time()-t0,2)
            final['seed_receipts'][str(seed)]=result
            del model,pairs,head,base_head
        final['neural_forward_executed']=True
        final['status']='BEHAVIOR_EXECUTED_CAUSAL_CIRCUIT_UNPROVEN'
    # Raw source/cache folder was destroyed BEFORE the only delivery JSON is written.
    if root.exists():raise ValueError('RAW_TEMP_DIRECTORY_NOT_REMOVED')
    final['raw_isolation_exit_absence']=True
    final['runtime_seconds']=round(time.time()-t0,2)
    OUT.write_text(json.dumps(final,sort_keys=True,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('DONE',final['status'],'seeds',list(final['seed_receipts']), 'seconds',final['runtime_seconds'],'raw_absent',not root.exists(), flush=True)

if __name__=='__main__':run()