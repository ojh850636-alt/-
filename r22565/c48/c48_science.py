from __future__ import annotations
import argparse, hashlib, json, math, random, re
from collections import defaultdict
from pathlib import Path
SEED=22565
ADAPTER_SHA='e50ef9a7b3ab9ce41feb3b0003b777e087d04257c56311cce6c003fab225a7d1'
BASE_SHA='3db1c98132b54901976d6da56fa711756c7780a6263d5db8f336232584f5fe58'
LABELS={0:'World',1:'Sports',2:'Business',3:'Sci/Tech'}

def sha_file(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def mean(xs): return sum(xs)/len(xs) if xs else 0.0

def macro_f1(y,p,k=4):
    vals=[]
    for c in range(k):
        tp=sum(a==c and b==c for a,b in zip(y,p)); fp=sum(a!=c and b==c for a,b in zip(y,p)); fn=sum(a==c and b!=c for a,b in zip(y,p))
        pr=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.; vals.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
    return mean(vals)

def bootstrap_accuracy_delta(y,pa,pb,n=2000):
    r=random.Random(SEED); N=len(y); vals=[]
    for _ in range(n):
        ix=[r.randrange(N) for __ in range(N)]
        vals.append(mean([pa[i]==y[i] for i in ix])-mean([pb[i]==y[i] for i in ix]))
    vals.sort(); return [vals[int(.025*(n-1))],vals[int(.975*(n-1))]]

def static_forensics(path:Path):
    from safetensors import safe_open
    out={'schema':'LAA_R22565_C48_STATIC_V1','adapter_sha256':sha_file(path)}; assert out['adapter_sha256']==ADAPTER_SHA
    total=lora=aux=0; by_proj=defaultdict(int); by_layer=defaultdict(int); keys=[]
    with safe_open(str(path),framework='pt',device='cpu') as f:
        for k in f.keys():
            sh=list(f.get_slice(k).get_shape()); n=math.prod(sh); total+=n; keys.append(k)
            if 'lora_A' in k or 'lora_B' in k:
                lora+=n
                proj='query' if '.query.' in k else ('value' if '.value.' in k else 'other'); by_proj[proj]+=n
                m=re.search(r'layer\.(\d+)',k); by_layer[m.group(1) if m else 'na']+=n
            else: aux+=n
    out.update(total_params=total,lora_params=lora,auxiliary_params=aux,lora_fraction=lora/total if total else 0,auxiliary_fraction=aux/total if total else 0,key_count=len(keys),lora_params_by_projection=dict(by_proj),lora_params_by_layer=dict(by_layer),sample_keys=keys[:16])
    return out

def lora_modules(m):
    for n,x in m.named_modules():
        if hasattr(x,'lora_A') and hasattr(x,'lora_B') and hasattr(x,'scaling') and x.lora_A:
            yield n,x

def aux_modules(m):
    for n,x in m.named_modules():
        if x.__class__.__name__=='ModulesToSaveWrapper': yield n,x

def ak(x): return next(iter(x.scaling))
class Controller:
    def __init__(self,m):
        self.m=m; self.l=list(lora_modules(m)); self.a=list(aux_modules(m));
        self.sc={(n,ak(x)):float(x.scaling[ak(x)]) for n,x in self.l}
        self.bs={(n,ak(x)):x.lora_B[ak(x)].weight.detach().clone() for n,x in self.l}
    def restore(self):
        for n,x in self.l:
            k=ak(x); x.scaling[k]=self.sc[(n,k)]; x.lora_B[k].weight.data.copy_(self.bs[(n,k)])
        for _,x in self.a: x.enable_adapters(enabled=True)
    def aux(self,on:bool):
        for _,x in self.a: x.enable_adapters(enabled=on)
    def dose(self,d):
        for n,x in self.l: x.scaling[ak(x)]=self.sc[(n,ak(x))]*d
    def set(self,c):
        self.restore()
        if c=='BASE': self.dose(0); self.aux(False)
        elif c=='HEAD_ONLY': self.dose(0); self.aux(True)
        elif c=='LORA_ONLY': self.dose(1); self.aux(False)
        elif c=='FULL': pass
        elif c.startswith('DOSE_'): self.dose(float(c.split('_')[1])); self.aux(True)
        elif c=='RANDOM_SIGN_SAME_HEAD':
            rr=random.Random(SEED)
            for n,x in self.l: x.scaling[ak(x)]=self.sc[(n,ak(x))]*(-1 if rr.random()<.5 else 1)
            self.aux(True)
        elif c=='LAYER_SHUFFLED_SAME_HEAD':
            groups=defaultdict(list)
            for n,x in self.l:
                proj='query' if '.query.' in n else ('value' if '.value.' in n else 'other')
                groups[(proj,tuple(x.lora_B[ak(x)].weight.shape))].append((n,x))
            rr=random.Random(SEED)
            for g,items in groups.items():
                src=[self.bs[(n,ak(x))] for n,x in items]; rr.shuffle(src)
                for (n,x),w in zip(items,src): x.lora_B[ak(x)].weight.data.copy_(w)
            self.aux(True)
    def zero_projection(self,p):
        self.restore(); self.aux(True)
        for n,x in self.l:
            if f'.{p}.' in n: x.scaling[ak(x)]=0.0
    def zero_band(self,lo,hi):
        self.restore(); self.aux(True)
        for n,x in self.l:
            m=re.search(r'layer\.(\d+)',n)
            if m and lo<=int(m.group(1))<=hi: x.scaling[ak(x)]=0.0

def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def predict(model,tok,rs,batch=32):
    import torch
    out=[]
    for i in range(0,len(rs),batch):
        ch=rs[i:i+batch]; enc=tok([x['text'] for x in ch],padding=True,truncation=True,max_length=192,return_tensors='pt')
        with torch.no_grad(): logits=model(**enc).logits
        out.extend(logits.argmax(-1).cpu().tolist())
    return out

def self_test():
    from transformers import BertConfig,BertForSequenceClassification
    from peft import LoraConfig,get_peft_model
    c=BertConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_labels=4,pad_token_id=0)
    b=BertForSequenceClassification(c); pc=LoraConfig(r=1,lora_alpha=1,target_modules=['query','value'],task_type='SEQ_CLS'); m=get_peft_model(b,pc); ctl=Controller(m)
    assert ctl.l and ctl.a
    for cond in ['BASE','HEAD_ONLY','LORA_ONLY','FULL','RANDOM_SIGN_SAME_HEAD','LAYER_SHUFFLED_SAME_HEAD','DOSE_0.5']: ctl.set(cond)
    print(json.dumps({'self_test':'PASS','lora_modules':len(ctl.l),'aux_wrappers':len(ctl.a)}))

def run(a):
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
    from peft import PeftModel
    root=Path(a.work); base=root/'base'; ad=root/'adapter'; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    assert sha_file(base/'model.safetensors')==BASE_SHA; assert sha_file(ad/'adapter_model.safetensors')==ADAPTER_SHA
    allr=rows(a.suite); scored=[x for x in allr if x['label'] is not None]; amb=[x for x in allr if x['label'] is None]; y=[x['label'] for x in scored]
    static=static_forensics(ad/'adapter_model.safetensors'); (out/'C48_STATIC.json').write_text(json.dumps(static,indent=2,sort_keys=True))
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    cfg=AutoConfig.from_pretrained(base,local_files_only=True); cfg.num_labels=4; cfg.id2label={i:LABELS[i] for i in LABELS}; cfg.label2id={v:k for k,v in cfg.id2label.items()}
    bm=AutoModelForSequenceClassification.from_pretrained(base,config=cfg,ignore_mismatched_sizes=True,local_files_only=True)
    model=PeftModel.from_pretrained(bm,str(ad),local_files_only=True,is_trainable=False); model.eval(); ctl=Controller(model)
    conditions=['BASE','HEAD_ONLY','LORA_ONLY','FULL']; preds={}; metrics={}
    for c in conditions:
        ctl.set(c); p=predict(model,tok,scored,a.batch); preds[c]=p; metrics[c]={'accuracy':mean([aa==bb for aa,bb in zip(y,p)]),'macro_f1':macro_f1(y,p)}
    delta=metrics['FULL']['accuracy']-metrics['HEAD_ONLY']['accuracy']; ci=bootstrap_accuracy_delta(y,preds['FULL'],preds['HEAD_ONLY'])
    controls={}; cp={}
    for c in ['RANDOM_SIGN_SAME_HEAD','LAYER_SHUFFLED_SAME_HEAD','DOSE_0.25','DOSE_0.5','DOSE_1.5']:
        ctl.set(c); p=predict(model,tok,scored,a.batch); cp[c]=p; controls[c]={'accuracy':mean([aa==bb for aa,bb in zip(y,p)]),'macro_f1':macro_f1(y,p)}
    sep_random=metrics['FULL']['accuracy']-controls['RANDOM_SIGN_SAME_HEAD']['accuracy']; sep_shuffle=metrics['FULL']['accuracy']-controls['LAYER_SHUFFLED_SAME_HEAD']['accuracy']
    gate=metrics['FULL']['accuracy']>=.75 and delta>=.05 and ci[0]>0 and sep_random>=.03 and sep_shuffle>=.03
    causal={}
    if gate:
        for p in ['query','value']:
            ctl.zero_projection(p); z=predict(model,tok,scored,a.batch); acc=mean([aa==bb for aa,bb in zip(y,z)]); causal[p]={'accuracy':acc,'drop_from_full':metrics['FULL']['accuracy']-acc}
        # MiniLM-L4 has four layers; if exact model differs, infer observed max layer.
        ids=[]
        for n,_ in ctl.l:
            m=re.search(r'layer\.(\d+)',n)
            if m: ids.append(int(m.group(1)))
        if ids:
            mx=max(ids); cut=(mx+1)//2-1
            for name,lo,hi in [('LOW',0,max(0,cut)),('HIGH',max(0,cut+1),mx)]:
                ctl.zero_band(lo,hi); z=predict(model,tok,scored,a.batch); acc=mean([aa==bb for aa,bb in zip(y,z)]); causal[name]={'layers':[lo,hi],'accuracy':acc,'drop_from_full':metrics['FULL']['accuracy']-acc}
    # mixed-topic diagnostics: retain only predicted label histograms, not logits/text.
    mixed={}
    for c in conditions:
        ctl.set(c); p=predict(model,tok,amb,a.batch); hist={LABELS[i]:p.count(i) for i in range(4)}; mixed[c]=hist
    e3=gate and any(v.get('drop_from_full',0)>=.03 for v in causal.values())
    summary={'schema':'LAA_R22565_C48_BEHAVIOR_V1','n_scored':len(scored),'n_ambiguous':len(amb),'metrics':metrics,'conditional_lora_accuracy_gain':delta,'conditional_lora_accuracy_gain_ci95':ci,'control_separation':{'random_sign':sep_random,'layer_shuffle':sep_shuffle},'controls':controls,'behavior_gate_green':gate,'causal_broad':causal,'e3_localization_candidate':e3,'claim_level':'E3_LOCALIZATION_ONLY' if e3 else ('E2_CONTROL_SEPARATED_BEHAVIOR' if gate else 'E1_BEHAVIOR_OR_FAILURE'),'ambiguous_prediction_histograms':mixed}
    (out/'C48_BEHAVIOR_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    with (out/'C48_CASE_RESULTS.jsonl').open('w') as f:
        for i,r in enumerate(scored):
            f.write(json.dumps({'case_id':r['case_id'],'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),'label':r['label'],'split':r['split'],'family':r['family'],'predictions':{c:preds[c][i] for c in conditions},'correct':{c:preds[c][i]==r['label'] for c in conditions}},sort_keys=True)+'\n')
    print(json.dumps(summary,sort_keys=True))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--run',action='store_true'); p.add_argument('--work'); p.add_argument('--suite'); p.add_argument('--out'); p.add_argument('--batch',type=int,default=32); a=p.parse_args()
    if a.self_test:self_test()
    elif a.run:run(a)
if __name__=='__main__':main()
