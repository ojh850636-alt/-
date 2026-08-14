from __future__ import annotations
import argparse, hashlib, json, math, random, re
from collections import defaultdict
from pathlib import Path
SEED=22565
ADAPTER_SHA='e50ef9a7b3ab9ce41feb3b0003b777e087d04257c56311cce6c003fab225a7d1'
BASE_SHA='3db1c98132b54901976d6da56fa711756c7780a6263d5db8f336232584f5fe58'
LABELS={0:'World',1:'Sports',2:'Business',3:'Sci/Tech'}

def fsha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def mean(xs): return sum(xs)/len(xs) if xs else 0.0
def accuracy(y,p): return mean([a==b for a,b in zip(y,p)])
def macro_f1(y,p):
    vals=[]
    for c in range(4):
        tp=sum(a==c and b==c for a,b in zip(y,p)); fp=sum(a!=c and b==c for a,b in zip(y,p)); fn=sum(a==c and b!=c for a,b in zip(y,p))
        pr=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.; vals.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
    return mean(vals)
def bootstrap_acc_delta(y,a,b,n=2000):
    r=random.Random(SEED); N=len(y); vals=[]
    for _ in range(n):
        ix=[r.randrange(N) for _ in range(N)]; vals.append(mean([int(a[i]==y[i])-int(b[i]==y[i]) for i in ix]))
    vals.sort(); return [vals[int(.025*(n-1))],vals[int(.975*(n-1))]]
def static_forensics(p):
    from safetensors import safe_open
    out={'schema':'LAA_R22565_C49_STATIC_V1','adapter_sha256':fsha(p)}; assert out['adapter_sha256']==ADAPTER_SHA
    total=lora=aux=0; keys=[]; proj=defaultdict(int); layer=defaultdict(int)
    with safe_open(str(p),framework='pt',device='cpu') as f:
        for k in f.keys():
            sh=list(f.get_slice(k).get_shape()); n=math.prod(sh); total+=n; keys.append({'key':k,'shape':sh,'params':n})
            if 'lora_A' in k or 'lora_B' in k:
                lora+=n; proj['query' if 'query' in k else ('value' if 'value' in k else 'other')]+=n
                m=re.search(r'layer\.(\d+)',k); layer[(m.group(1) if m else 'na')]+=n
            else: aux+=n
    out.update(total_params=total,lora_params=lora,auxiliary_params=aux,lora_fraction=(lora/total if total else 0),auxiliary_fraction=(aux/total if total else 0),key_count=len(keys),lora_params_by_projection=dict(proj),lora_params_by_layer=dict(layer),sample_keys=keys[:20]); return out
def iter_lora(m):
    for n,x in m.named_modules():
        if hasattr(x,'lora_A') and hasattr(x,'lora_B') and hasattr(x,'scaling') and getattr(x,'lora_A',None): yield n,x
def iter_aux(m):
    for n,x in m.named_modules():
        if x.__class__.__name__=='ModulesToSaveWrapper' and hasattr(x,'modules_to_save'): yield n,x
def akey(x): return next(iter(x.scaling))
def clone_sd(module): return {k:v.detach().clone() for k,v in module.state_dict().items()}
def load_sd(module,sd):
    cur=module.state_dict()
    for k,v in sd.items(): cur[k].copy_(v)
class Controller:
    def __init__(self,m):
        self.m=m; self.l=list(iter_lora(m)); self.a=list(iter_aux(m)); self.scal={(n,akey(x)):float(x.scaling[akey(x)]) for n,x in self.l}; self.B={(n,akey(x)):x.lora_B[akey(x)].weight.detach().clone() for n,x in self.l}; self.aux={}
        for n,x in self.a:
            name=next(iter(x.modules_to_save.keys())); self.aux[n]={'wrapper':x,'name':name,'trained':clone_sd(x.modules_to_save[name]),'original':clone_sd(x.original_module)}
    def restore_lora(self):
        for n,x in self.l:
            k=akey(x); x.scaling[k]=self.scal[(n,k)]; x.lora_B[k].weight.data.copy_(self.B[(n,k)])
    def set_aux(self,trained):
        for n,d in self.aux.items(): load_sd(d['wrapper'].modules_to_save[d['name']],d['trained'] if trained else d['original'])
    def scale(self,s):
        for n,x in self.l: x.scaling[akey(x)]=self.scal[(n,akey(x))]*s
    def restore(self): self.restore_lora(); self.set_aux(True)
    def set(self,c):
        self.restore()
        if c=='BASE': self.scale(0); self.set_aux(False)
        elif c=='HEAD_ONLY': self.scale(0); self.set_aux(True)
        elif c=='LORA_ONLY': self.scale(1); self.set_aux(False)
        elif c=='FULL': pass
        elif c.startswith('DOSE_'): self.scale(float(c.split('_')[1]))
        elif c=='RANDOM_SIGN_SAME_HEAD':
            for i,(n,x) in enumerate(self.l): x.scaling[akey(x)]=self.scal[(n,akey(x))]*(-1 if (i*17+3)%5<2 else 1)
        elif c=='LAYER_SHUFFLED_SAME_HEAD':
            groups=defaultdict(list)
            for n,x in self.l: groups['query' if 'query' in n else ('value' if 'value' in n else 'other')].append((n,x))
            for pr,arr in groups.items():
                bs=[self.B[(n,akey(x))] for n,x in arr]
                if len(arr)>1:
                    bs=bs[1:]+bs[:1]
                    for (n,x),b in zip(arr,bs): x.lora_B[akey(x)].weight.data.copy_(b)
    def zero_projection(self,pr):
        self.restore()
        for n,x in self.l:
            if pr in n: x.scaling[akey(x)]=0.0
    def zero_layer(self,idx):
        self.restore(); pat=f'.layer.{idx}.'
        for n,x in self.l:
            if pat in n: x.scaling[akey(x)]=0.0
def load_rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def predict(model,tok,rows,batch=32):
    import torch
    out=[]
    for i in range(0,len(rows),batch):
        ch=rows[i:i+batch]; enc=tok([x['text'] for x in ch],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.no_grad(): lg=model(**enc).logits
        out.extend(lg.argmax(-1).cpu().tolist())
    return out
def summarize(y,p): return {'accuracy':accuracy(y,p),'macro_f1':macro_f1(y,p)}
def split_summaries(rows,p):
    out={}
    for s in sorted({x['split'] for x in rows}):
        ix=[i for i,x in enumerate(rows) if x['split']==s]; out[s]=summarize([rows[i]['label'] for i in ix],[p[i] for i in ix])
    return out
def self_test():
    import torch
    from transformers import BertConfig,BertForSequenceClassification
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(SEED); cfg=BertConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_labels=4,pad_token_id=0); b=BertForSequenceClassification(cfg); pc=LoraConfig(r=1,lora_alpha=1,target_modules=['query','value'],task_type='SEQ_CLS'); m=get_peft_model(b,pc); ctl=Controller(m)
    assert ctl.l and ctl.a, (len(ctl.l),len(ctl.a))
    for c in ['BASE','HEAD_ONLY','LORA_ONLY','FULL','RANDOM_SIGN_SAME_HEAD','LAYER_SHUFFLED_SAME_HEAD','DOSE_0.5']: ctl.set(c)
    print(json.dumps({'self_test':'PASS','lora_modules':len(ctl.l),'aux_wrappers':len(ctl.a)}))
def run(a):
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
    from peft import PeftModel
    torch.manual_seed(SEED); root=Path(a.work); b=root/'base'; ad=root/'adapter'; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    assert fsha(b/'model.safetensors')==BASE_SHA and fsha(ad/'adapter_model.safetensors')==ADAPTER_SHA
    rows=load_rows(a.suite); y=[x['label'] for x in rows]; static=static_forensics(ad/'adapter_model.safetensors'); (out/'C49_STATIC.json').write_text(json.dumps(static,indent=2,sort_keys=True))
    cfg=AutoConfig.from_pretrained(b,local_files_only=True); cfg.num_labels=4; cfg.id2label=LABELS; cfg.label2id={v:k for k,v in LABELS.items()}; tok=AutoTokenizer.from_pretrained(b,local_files_only=True); base=AutoModelForSequenceClassification.from_pretrained(b,config=cfg,ignore_mismatched_sizes=True,local_files_only=True); model=PeftModel.from_pretrained(base,str(ad),local_files_only=True,is_trainable=False); model.eval(); ctl=Controller(model)
    runtime={'lora_modules':len(ctl.l),'aux_wrappers':len(ctl.a),'base_num_labels':4,'static_aux_params':static['auxiliary_params']}
    conditions=['BASE','HEAD_ONLY','LORA_ONLY','FULL']; pred={}; summaries={}; split={}
    for c in conditions: ctl.set(c); pred[c]=predict(model,tok,rows,a.batch); summaries[c]=summarize(y,pred[c]); split[c]=split_summaries(rows,pred[c])
    delta=summaries['FULL']['accuracy']-summaries['HEAD_ONLY']['accuracy']; ci=bootstrap_acc_delta(y,pred['FULL'],pred['HEAD_ONLY']); controls={}
    for c in ['RANDOM_SIGN_SAME_HEAD','LAYER_SHUFFLED_SAME_HEAD','DOSE_0.25','DOSE_0.5','DOSE_1.5']:
        ctl.set(c); pp=predict(model,tok,rows,a.batch); controls[c]=summarize(y,pp)
    control_sep=(summaries['FULL']['accuracy']-controls['RANDOM_SIGN_SAME_HEAD']['accuracy']>=0.02 and summaries['FULL']['accuracy']-controls['LAYER_SHUFFLED_SAME_HEAD']['accuracy']>=0.02); gate=(summaries['FULL']['accuracy']>=0.65 and delta>=0.03 and ci[0]>0.01 and control_sep); causal={'projections':{},'layers':{}}
    if gate:
        for pr in ['query','value']:
            ctl.zero_projection(pr); pp=predict(model,tok,rows,a.batch); ss=summarize(y,pp); ss['drop_from_full']=summaries['FULL']['accuracy']-ss['accuracy']; causal['projections'][pr]=ss
        nl=int(getattr(model.base_model.model.config,'num_hidden_layers',4))
        for i in range(nl):
            ctl.zero_layer(i); pp=predict(model,tok,rows,a.batch); ss=summarize(y,pp); ss['drop_from_full']=summaries['FULL']['accuracy']-ss['accuracy']; causal['layers'][str(i)]=ss
    maxp=max([v['drop_from_full'] for v in causal['projections'].values()] or [0]); maxl=max([v['drop_from_full'] for v in causal['layers'].values()] or [0]); e3=bool(gate and maxp>=0.02 and maxl>=0.015); claim='E3_BROAD_LAYER_PROJECTION_LOCALIZATION_ONLY' if e3 else ('E2_CONTROL_SEPARATED_BEHAVIOR' if gate else 'E1_BEHAVIOR_ONLY_OR_FAILURE')
    summary={'schema':'LAA_R22565_C49_BEHAVIOR_V1','n':len(rows),'runtime':runtime,'conditions':summaries,'split_metrics':split,'full_minus_head_accuracy':delta,'full_minus_head_accuracy_ci95':ci,'controls':controls,'control_separation':control_sep,'behavior_gate_green':gate,'causal':causal,'e3_localization_gate':e3,'claim_level':claim,'training_data_license_limitation':'AG News repository license metadata is unknown; no AG News rows were ingested for this evaluation. Evaluation uses only procedural fresh cases.'}
    (out/'C49_BEHAVIOR_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    with (out/'C49_CASE_RESULTS.jsonl').open('w') as f:
        for i,r in enumerate(rows): f.write(json.dumps({'case_id':r['case_id'],'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),'label':r['label'],**{c:pred[c][i] for c in conditions}},sort_keys=True)+'\n')
    print(json.dumps(summary,sort_keys=True))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--run',action='store_true'); p.add_argument('--work'); p.add_argument('--suite'); p.add_argument('--out'); p.add_argument('--batch',type=int,default=32); a=p.parse_args(); self_test() if a.self_test else (run(a) if a.run else None)
if __name__=='__main__': main()
