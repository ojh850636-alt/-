"""R131 one-ingress, CPU-only TREC LoRA scientific answer-type archaeology.
E0/E1/E2/E3 exploratory only. NEVER export raw weights/A/B/DeltaW, tokenizer,
raw logits, activations, or hidden states. Scientific experiment only.
The hook controller and provenance pattern descend from sealed R129, not a
new independent neural method. Source ingress consumes TREC seed0 & seed1 once.
"""
from __future__ import annotations
import os,hashlib,json,re,math,time,tempfile,collections,random
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import torch
from huggingface_hub import HfApi,snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer,AutoModelForSequenceClassification

ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get('LAA131_OUTPUT','result131.json')).resolve()
LOCK=ROOT/'lock131.json'
SUITE=ROOT/'suite131.jsonl'
LABELS=('ABBR','ENTY','DESC','HUM','LOC','NUM')
PAT=re.compile(r'^(?:base_model\.model\.)?(.+?)\.lora_([AB])(?:\.default)?\.weight$')
HEAD=re.compile(r'^(?:base_model\.model\.)?classifier\.modules_to_save\.[^.]+\.(.+)$')
DIRECT=re.compile(r'^(?:base_model\.model\.)?classifier\.((?:dense|out_proj)\.(?:weight|bias))$')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def load_contract():
    v=json.loads(LOCK.read_text())
    assert sha(SUITE)==v['suite_sha256'],'SUITE_DIGEST_MISMATCH'
    assert sha(__file__)==v['remote_code_sha256'],'CODE_DIGEST_MISMATCH'
    assert v['adapter_repo']=='october25kim/lora-merge-prereg-roberta-base'
    assert v['adapter_revision']=='8896fca8a657e47d4d144853f0123aff77292fd4'
    assert v['base_repo']=='FacebookAI/roberta-base'
    assert v['base_revision']=='e2da8e2f811d1448a5b465c236feacd80ffbac7b'
    assert v['source_scope']=='trec/seed0+trec/seed1' and v['model_output_seen_before_seal']==0
    rows=[json.loads(s) for s in SUITE.read_text().splitlines() if s.strip()]
    assert len(rows)==480 and len({r['id'] for r in rows})==480
    assert tuple(v['label_names'])==LABELS
    assert all((r['label'] in range(6)) if r['split']!='NEGATIVE' else r['label'] is None for r in rows)
    return v,rows

def get_sources(v,root):
    repo=v['adapter_repo']; rev=v['adapter_revision']
    assert HfApi().model_info(repo,revision=rev).sha==rev
    files=[f'trec/seed{s}/{f}' for s in (0,1) for f in ('adapter_config.json','adapter_model.safetensors','train_meta.json')]
    adapter=root/'adapter'
    snapshot_download(repo,revision=rev,allow_patterns=files,local_dir=adapter)
    for seed in (0,1):
        p=adapter/f'trec/seed{seed}'
        weight=p/'adapter_model.safetensors'
        entry=v['seed_sources'][str(seed)]
        assert weight.stat().st_size==entry['size'] and sha(weight)==entry['sha256'],'ADAPTER_HASH_FAIL'
        cfg=json.loads((p/'adapter_config.json').read_text())
        meta=json.loads((p/'train_meta.json').read_text())
        assert cfg['r']==8 and cfg['lora_alpha']==16 and cfg['bias']=='none'
        assert set(cfg['target_modules'])=={'query','key','value','output.dense','intermediate.dense'}
        assert set(cfg['modules_to_save'])=={'classifier','score'}
        assert meta['task']=='trec' and meta['seed']==seed and meta['budget']['num_labels']==6
        assert meta['recipe']['base_revision']==v['base_revision']
        assert meta['spec'][6]=='coarse_label'
    base=root/'base'
    allow=['model.safetensors','config.json','tokenizer.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json']
    snapshot_download(v['base_repo'],revision=v['base_revision'],allow_patterns=allow,local_dir=base)
    weight=base/'model.safetensors'
    assert weight.stat().st_size==v['base_size'] and sha(weight)==v['base_sha256'],'BASE_HASH_FAIL'
    return adapter,base

def read_adapter(path):
    pairs=collections.defaultdict(dict);head={};count=0
    with safe_open(str(path),framework='pt',device='cpu') as f:
        for name in f.keys():
            t=f.get_tensor(name).to(torch.float32)
            assert bool(torch.isfinite(t).all()),'NONFINITE_WEIGHT'
            m=PAT.fullmatch(name);h=HEAD.fullmatch(name) or DIRECT.fullmatch(name)
            if m:pairs[m.group(1)][m.group(2)]=t
            elif h:head[h.group(1)]=t
            else:raise ValueError('UNSUPPORTED_ADAPTER_COMPONENT:'+name)
            count+=1
    assert count==148 and len(pairs)==72
    assert set(head)=={'dense.weight','dense.bias','out_proj.weight','out_proj.bias'}
    stats={'site_count':72,'head_tensors':4,'head_elements':sum(t.numel() for t in head.values()),
           'nonzero_operator_count':0,'zero_operator_count':0,'site_energy':{}}
    for key,site in pairs.items():
        a=site['A'];b=site['B']
        assert a.ndim==b.ndim==2 and a.shape[0]==b.shape[1]==8
        energy=torch.trace((a@a.T)@(b.T@b)).item()
        assert math.isfinite(energy)
        stats['nonzero_operator_count' if energy>1e-18 else 'zero_operator_count']+=1
        # site identifier & scalar energy only, never vectors
        stats['site_energy'][key]=round(float(energy),7)
    return dict(pairs),head,stats

def rand_lora(pairs,seed):
    g=torch.Generator().manual_seed(seed)
    out={}
    for name,x in pairs.items():
        a,b=x['A'],x['B'];ar=torch.randn(a.shape,generator=g);br=torch.randn(b.shape,generator=g)
        e0=torch.trace((a@a.T)@(b.T@b)).item()
        er=torch.trace((ar@ar.T)@(br.T@br)).item()
        if er<1e-18:raise ValueError('RANDOM_ZERO')
        br=br*math.sqrt(max(e0,0)/er)
        out[name]={'A':ar,'B':br}
    return out

def shuffled_lora(pairs,seed):
    rnd=random.Random(seed);groups=collections.defaultdict(list)
    for name,x in pairs.items():groups[(tuple(x['A'].shape),tuple(x['B'].shape))].append(name)
    out={}
    for names in groups.values():
        keys=sorted(names);perm=keys[:];rnd.shuffle(perm)
        if len(keys)>1 and perm==keys:perm=keys[1:]+keys[:1]
        for n,m in zip(keys,perm):out[n]=pairs[m]
    return out

def set_head(model,head):
    own=model.classifier.state_dict()
    assert set(own)==set(head)
    assert all(own[k].shape==head[k].shape for k in own)
    model.classifier.load_state_dict(head,strict=True)

def attach(model,pairs,scale=1.0,skip_layers=(),only_layers=None,skip_proj=None):
    modules=dict(model.named_modules());handles=[];names=[]
    try:
        for name,site in sorted(pairs.items()):
            mx=re.search(r'\.encoder\.layer\.(\d+)\.',name)
            if not mx:raise ValueError('UNKNOWN_LAYER:'+name)
            idx=int(mx.group(1))
            if idx in skip_layers or (only_layers is not None and idx not in only_layers):continue
            if skip_proj and skip_proj in name:continue
            if name not in modules:raise ValueError('MISSING_MODULE:'+name)
            module=modules[name];a,b=site['A'],site['B']
            if a.shape[1]!=module.in_features or b.shape[0]!=module.out_features:
                raise ValueError('SHAPE_MISMATCH:'+name)
            def make_hook(u,v):
                def hook(m,inputs,output):
                    d=(inputs[0].float()@u.T)@v.T
                    return output+d.to(output.dtype)*(2.0*scale)
                return hook
            handles.append(module.register_forward_hook(make_hook(a,b)))
            names.append(name)
    except BaseException:
        for h in handles:h.remove()
        raise
    return handles,names

def predict(model,batches):
    preds=[];confbins=[]
    with torch.inference_mode():
        for x in batches:
            logits=model(**x).logits
            assert logits.shape[-1]==6 and bool(torch.isfinite(logits).all())
            prob=logits.float().softmax(-1)
            y=logits.argmax(-1).cpu().tolist()
            p=prob.max(-1).values.cpu().tolist()
            preds.extend(y)
            # one categorical confidence bucket, NOT raw logits or probabilities
            confbins.extend(2 if v>=0.90 else 1 if v>=0.80 else 0 for v in p)
    return preds,confbins

def accuracy(y,rows,split):
    idx=[i for i,r in enumerate(rows) if r['split']==split and r['label'] is not None]
    assert idx
    return round(sum(y[i]==rows[i]['label'] for i in idx)/len(idx),6)

def parity(base,path,model,head,pairs,tokenizer,rows):
    from peft import PeftModel
    torch.manual_seed(4242)
    reference=AutoModelForSequenceClassification.from_pretrained(str(base),num_labels=6,local_files_only=True,use_safetensors=True,trust_remote_code=False)
    reference=PeftModel.from_pretrained(reference,str(path),local_files_only=True,is_trainable=False)
    reference.eval()
    batch=tokenizer([r['question'] for r in rows[:6]],padding=True,truncation=True,max_length=128,return_tensors='pt')
    set_head(model,head)
    hh,_=attach(model,pairs)
    try:
        with torch.inference_mode():a=model(**batch).logits
    finally:
        for h in hh:h.remove()
    with torch.inference_mode():b=reference(**batch).logits
    err=float((a-b).abs().max().item())
    del reference
    if not math.isfinite(err) or err>0.0003:raise ValueError('PEFT_PARITY_FAILURE:'+str(err))
    return err

def main():
    start=time.time();torch.set_num_threads(min(4,os.cpu_count() or 1))
    lock,rows=load_contract()
    rec={'schema':'LAA131_TREC_BEHAVIOR_V1','status':'INITIALIZED',
         'source':{'repo':lock['adapter_repo'],'revision':lock['adapter_revision'],
                   'base_repo':lock['base_repo'],'base_revision':lock['base_revision'],
                   'adapter_sha256':lock['seed_sources'],'base_sha256':lock['base_sha256']},
         'lock_sha256':sha(LOCK),'suite_sha256':sha(SUITE),
         'neural_forward_executed':False,'raw_logits_exported':False,'raw_activations_exported':False,
         'model_weights_exported':False,'base_revision_training_time_solely_producer_claim':True,
         'causal_supported_E4':False,'causal_supported_E5':False,'external_algorithm':False,'student':False,'promotion':False,
         'seed_receipts':{},'predictions':[],
         'protocol':{'six_label_ontology':list(LABELS),'same_object_four_way':True,
                     'main_conditions':['BASE','HEAD_ONLY','LORA_ONLY','FULL','RANDOM','SHUFFLED','HALF','HIGH','ABL_EARLY','ABL_MIDDLE','ABL_LATE'],
                     'fine_cases':['CONFIRMATION','GENERATOR_OOD','CONTRAST'],
                     'fine_conditional_gate':'abs(FULL-HEAD_ONLY)>=0.08 in any locked split; exploratory E3 max',
                     'fine_classes':'12 layers,5 projection groups,3 keep-only bands,9 random matching four-layer sites'}}
    with tempfile.TemporaryDirectory(prefix='laa131_src_') as d:
        src=Path(d);os.environ['HF_HOME']=str(src/'cache');os.environ['HUGGINGFACE_HUB_CACHE']=str(src/'cache'/'hub')
        adapter,base=get_sources(lock,src)
        tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,use_fast=True)
        batches=[]
        for i in range(0,len(rows),24):
            batches.append(tok([r['question'] for r in rows[i:i+24]],padding=True,truncation=True,max_length=128,return_tensors='pt'))
        focus=[i for i,r in enumerate(rows) if r['split'] in ('CONFIRMATION','GENERATOR_OOD','CONTRAST')]
        focus_batches=[]
        for p in range(0,len(focus),24):
            focus_batches.append(tok([rows[i]['question'] for i in focus[p:p+24]],padding=True,truncation=True,max_length=128,return_tensors='pt'))
        for seed in (0,1):
            folder=adapter/f'trec/seed{seed}'
            pairs,head,structure=read_adapter(folder/'adapter_model.safetensors')
            torch.manual_seed(4242)
            model=AutoModelForSequenceClassification.from_pretrained(str(base),num_labels=6,local_files_only=True,use_safetensors=True,trust_remote_code=False)
            model.eval()
            baseline={k:v.detach().clone() for k,v in model.classifier.state_dict().items()}
            err=parity(base,folder,model,head,pairs,tok,rows)
            out={'structural':structure,'native_peft_parity_max_abs_logit_error':err,'main':{},'fine':{},'fine_gate':None}
            combos=['BASE','HEAD_ONLY','LORA_ONLY','FULL','RANDOM','SHUFFLED','HALF','HIGH','ABL_EARLY','ABL_MIDDLE','ABL_LATE']
            for cond in combos:
                set_head(model,baseline if cond in ('BASE','LORA_ONLY') else head)
                act=None;skip=();scale=1.0
                if cond in ('FULL','LORA_ONLY'):act=pairs
                elif cond=='RANDOM':act=rand_lora(pairs,100131+seed)
                elif cond=='SHUFFLED':act=shuffled_lora(pairs,200131+seed)
                elif cond=='HALF':act=pairs;scale=0.5
                elif cond=='HIGH':act=pairs;scale=1.5
                elif cond=='ABL_EARLY':act=pairs;skip=range(0,4)
                elif cond=='ABL_MIDDLE':act=pairs;skip=range(4,8)
                elif cond=='ABL_LATE':act=pairs;skip=range(8,12)
                hh=[]
                try:
                    if act:hh,_=attach(model,act,scale=scale,skip_layers=skip)
                    ys,cs=predict(model,batches)
                finally:
                    for h in hh:h.remove()
                out['main'][cond]={'accuracy':{s:accuracy(ys,rows,s) for s in ('DISCOVERY','CONFIRMATION','GENERATOR_OOD','CONTRAST')},
                                   'prediction_digest':hashlib.sha256(bytes(ys)).hexdigest(),
                                   'negative_oos_high08':sum(cs[i]>=1 for i,r in enumerate(rows) if r['split']=='NEGATIVE'),
                                   'negative_oos_high09':sum(cs[i]>=2 for i,r in enumerate(rows) if r['split']=='NEGATIVE')}
                rec['predictions'].extend({'seed':seed,'condition':cond,'case_id':r['id'],'prediction':int(y),'confidence_bucket':c if r['split']=='NEGATIVE' else None}
                                          for r,y,c in zip(rows,ys,cs))
            focus_splits=('CONFIRMATION','GENERATOR_OOD','CONTRAST')
            deltas=[out['main']['FULL']['accuracy'][s]-out['main']['HEAD_ONLY']['accuracy'][s] for s in focus_splits]
            trigger=any(abs(d)>=0.08 for d in deltas)
            out['fine_gate']='PRESEALED_EXPLORATORY_MAGNITUDE_GATE_PASS' if trigger else 'NOT_RUN_PRESEALED_GATE_RED'
            if trigger:
                rnd=random.Random(202631+seed)
                fine=[('ABL_LAYER_'+str(i),set([i]),None,None,None) for i in range(12)]
                fine += [('ABL_QUERY',set(),None,'attention.self.query',None),('ABL_KEY',set(),None,'attention.self.key',None),
                         ('ABL_VALUE',set(),None,'attention.self.value',None),('ABL_INTER',set(),None,'intermediate.dense',None),
                         ('ABL_OUTPUT',set(),None,'output.dense',None)]
                fine += [('KEEP_EARLY',set(),set(range(0,4)),None,None),('KEEP_MIDDLE',set(),set(range(4,8)),None,None),
                         ('KEEP_LATE',set(),set(range(8,12)),None,None)]
                for band in range(3):
                    actual=set(range(band*4,band*4+4))
                    others=[i for i in range(12) if i not in actual]
                    for j in range(3):
                        # match 4 removed layers with a different random site combination
                        s=set(rnd.sample(others,4))
                        fine.append((f'RAND_SITE_{band}_{j}',s,None,None,None))
                for label,skip,only,proj,_ in fine:
                    set_head(model,head)
                    hh,_=attach(model,pairs,skip_layers=skip,only_layers=only,skip_proj=proj)
                    try:yy,cc=predict(model,focus_batches)
                    finally:
                        for h in hh:h.remove()
                    out['fine'][label]={'accuracy':{split:round(sum(yy[k]==rows[ix]['label'] for k,ix in enumerate(focus) if rows[ix]['split']==split)/sum(rows[ix]['split']==split for ix in focus),6) for split in focus_splits},
                                        'ablation_type':'SITE_SILENCED' if label.startswith(('ABL','RAND')) else 'COMPLEMENT_SILENCED'}
                    rec['predictions'].extend({'seed':seed,'condition':label,'case_id':rows[ix]['id'],'prediction':int(y),'confidence_bucket':None}
                                             for ix,y in zip(focus,yy))
            rec['seed_receipts'][str(seed)]=out
            del model,pairs,head,baseline
            print('SEED_FINISHED',seed,'main',out['main']['FULL']['accuracy'],'fine',out['fine_gate'],'elapsed',round(time.time()-start,1),flush=True)
        rec['neural_forward_executed']=True
        rec['status']='REAL_TREC_SOURCE_EXECUTED_EXPLORATORY_E3_ONLY'
    if src.exists():raise RuntimeError('RAW_ISOLATION_LEFT_BEHIND')
    rec['raw_isolation_exit_absence']=True
    rec['elapsed_seconds']=round(time.time()-start,3)
    # JSON written ONLY after temporary source directory deletion
    OUT.write_text(json.dumps(rec,sort_keys=True,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('DONE',rec['status'],len(rec['predictions']),'records','raw_absent',not src.exists(),flush=True)

if __name__=='__main__':main()
