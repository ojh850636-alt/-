#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, math, random
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from portallib.evaluation import PortalEvaluator

SEED=42
POSITIVE={"natural_min":.03,"confirmation_min":.02,"generator_min":0.0,"control_min":.02}
FAILURE={"natural_max":-.03,"confirmation_max":-.02,"generator_max":0.0,"control_max":-.02}
BANDS={"EARLY_QV":set(range(0,10)),"MIDDLE_QV":set(range(10,19)),"LATE_QV":set(range(19,28))}


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def rows_proc(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def rows_natural(path):
    return json.loads(Path(path).read_text())


def clean_hash_row(row):
    return hashlib.sha256((row['prompt']+'\x00'+'\x01'.join(row['choices'])).encode()).hexdigest()


def aggregate(values):
    if not values: return {"mean":None,"n":0}
    m=sum(values)/len(values)
    return {"mean":m,"n":len(values),"se":math.sqrt(sum((x-m)**2 for x in values)/max(1,len(values)-1)/len(values)) if len(values)>1 else 0.0}


def bootstrap_delta(a,b,seed=22574,nboot=3000):
    assert len(a)==len(b) and a
    d=[x-y for x,y in zip(a,b)]; rng=random.Random(seed); vals=[]
    for _ in range(nboot):
        vals.append(sum(d[rng.randrange(len(d))] for _ in range(len(d)))/len(d))
    vals.sort(); lo=vals[int(.025*nboot)]; hi=vals[min(nboot-1,int(.975*nboot))]
    return {"mean":sum(d)/len(d),"low":lo,"high":hi,"n":len(d)}


def directional_one_sided_p(a_acc,b_acc,want_a_gt_b=True):
    gt=lt=0
    for a,b in zip(a_acc,b_acc):
        if a>b: gt+=1
        elif a<b: lt+=1
    n=gt+lt
    if not n: return 1.0,gt,lt
    k=gt if want_a_gt_b else lt
    return min(1.0,sum(math.comb(n,i) for i in range(k,n+1))/(2**n)),gt,lt


def holm(raw,alpha=.05):
    order=sorted(raw,key=lambda k:raw[k]); out={k:False for k in raw}
    for i,k in enumerate(order):
        if raw[k] <= alpha/(len(order)-i): out[k]=True
        else: break
    return out


def boundary(input_ids,choice_ids):
    seq=torch.tensor([input_ids+choice_ids],dtype=torch.long)
    shifted=seq[:,1:]
    ignore=torch.full_like(shifted,-100)
    start=max(0,len(input_ids)-1); ignore[:,start:]=shifted[:,start:]
    return seq,ignore


def score_condition(model,tok,rows,batch=16):
    model.eval(); out=[]
    pending=[]; owners=[]
    with torch.inference_mode():
        for ri,row in enumerate(rows):
            prompt=tok(row['prompt'],add_special_tokens=True).input_ids
            for ci,ch in enumerate(row['choices']):
                ids=tok(ch,add_special_tokens=False).input_ids
                seq,lab=boundary(prompt,ids); pending.append((seq[0],lab[0],max(1,len(ch)))); owners.append((ri,ci))
        scores=[[None]*len(r['choices']) for r in rows]
        for st in range(0,len(pending),batch):
            part=pending[st:st+batch]; own=owners[st:st+batch]
            maxlen=max(len(x[0]) for x in part)
            x=torch.full((len(part),maxlen),tok.pad_token_id,dtype=torch.long)
            att=torch.zeros_like(x); lab=torch.full_like(x,-100)
            for j,(seq,labels,_) in enumerate(part):
                x[j,:len(seq)]=seq; att[j,:len(seq)]=1; lab[j,:len(labels)]=labels
            logits=model(input_ids=x,attention_mask=att,use_cache=False,logits_to_keep=lab.shape[1]).logits.float()
            if logits.shape[1]!=lab.shape[1]: logits=logits[:,-lab.shape[1]:,:]
            lp=torch.log_softmax(logits,dim=-1)
            mask=lab.ne(-100); safe=lab.masked_fill(~mask,0)
            vals=lp.gather(-1,safe.unsqueeze(-1)).squeeze(-1); sums=(vals*mask).sum(-1).cpu().tolist()
            for j,(ri,ci) in enumerate(own): scores[ri][ci]=float(sums[j]/part[j][2])
        for ri,row in enumerate(rows):
            s=scores[ri]; pred=max(range(len(s)),key=lambda i:s[i]); gold=int(row['gold_idx'])
            out.append({"id":row.get('case_id') or clean_hash_row(row),"split":row.get('split','NATURAL_LOCKED'),"gold":gold,"pred":pred,"correct":int(pred==gold),"gold_score":float(s[gold]),"margin":float(s[gold]-max(v for i,v in enumerate(s) if i!=gold))})
    return out


def parity_smoke(model,tok,natural):
    ev=PortalEvaluator(); sample=natural[:4]
    portal=[{"task":"rte","prompt":r['prompt'],"choices":r['choices'],"gold_idx":r['gold_idx']} for r in sample]
    ref=ev._score_rows(portal,model,tok,torch.device('cpu'),batch_size=2,metric_name='acc_norm',row_indexes=list(range(len(portal))))
    ours=score_condition(model,tok,sample,batch=2)
    acc=sum(x['correct'] for x in ours)/len(ours)
    return {"portal_acc":float(ref.value),"ours_acc":acc,"abs_diff":abs(float(ref.value)-acc),"pass":abs(float(ref.value)-acc)<1e-12}


def metric(rows):
    return {"accuracy":sum(x['correct'] for x in rows)/len(rows),"gold_score":aggregate([x['gold_score'] for x in rows]),"margin":aggregate([x['margin'] for x in rows]),"n":len(rows)}


def split_metric(rows):
    g=defaultdict(list)
    for x in rows:g[x['split']].append(x)
    return {k:metric(v) for k,v in sorted(g.items())}


def accvec(rows): return [x['correct'] for x in rows]


def install_state(model,path,name):
    if name in model.peft_config: model.delete_adapter(name)
    model.load_adapter(path,adapter_name=name,is_trainable=False)


def randomize(model,src_name,new_name,seed,mode):
    src={k:v.detach().cpu().clone() for k,v in model.get_adapter_state_dict(src_name).items()}
    rng=torch.Generator(device='cpu'); rng.manual_seed(seed)
    dst={}
    if mode=='random_sign':
        for k,v in src.items():
            sign=torch.where(torch.rand(v.shape,generator=rng)>0.5,torch.ones_like(v),-torch.ones_like(v)); dst[k]=v*sign
    elif mode=='layer_shuffle':
        byshape=defaultdict(list)
        for k,v in src.items(): byshape[(tuple(v.shape),'A' if 'lora_A' in k else 'B')].append(k)
        for keys in byshape.values():
            perm=keys[:]
            rr=random.Random(seed+len(keys)); rr.shuffle(perm)
            for a,b in zip(keys,perm): dst[a]=src[b].clone()
    else: raise ValueError(mode)
    model.add_adapter(new_name,model.peft_config[src_name]); model.set_adapter(new_name); model.load_adapter_state_dict(dst,adapter_name=new_name)


def dose(model,src_name,new_name,scale):
    src={k:v.detach().cpu().clone() for k,v in model.get_adapter_state_dict(src_name).items()}
    dst={k:(v*scale if 'lora_B' in k else v.clone()) for k,v in src.items()}
    model.add_adapter(new_name,model.peft_config[src_name]); model.set_adapter(new_name); model.load_adapter_state_dict(dst,adapter_name=new_name)


def layer_of(name):
    toks=name.split('.')
    for i,t in enumerate(toks[:-1]):
        if t=='layers':
            try:return int(toks[i+1])
            except:pass
    return None


def proj_of(name):
    for p in ('q_proj','v_proj'):
        if p in name:return p
    return 'other'


class Controller:
    def __init__(self,model,adapter='rte'):
        self.model=model; self.adapter=adapter
        self.mods=[]
        for n,m in model.named_modules():
            if hasattr(m,'lora_A') and adapter in m.lora_A and hasattr(m,'lora_B') and adapter in m.lora_B:
                self.mods.append((n,m,layer_of(n),proj_of(n)))
    def all_names(self): return [n for n,_,_,_ in self.mods]
    def names(self,obj):
        if obj=='Q_ALL':return [n for n,_,_,p in self.mods if p=='q_proj']
        if obj=='V_ALL':return [n for n,_,_,p in self.mods if p=='v_proj']
        if obj in BANDS:return [n for n,_,l,_ in self.mods if l in BANDS[obj]]
        if obj.startswith('LAYER_'):
            k=int(obj.split('_')[1]); return [n for n,_,l,_ in self.mods if l==k]
        raise KeyError(obj)
    def snapshot(self):
        return {n:(m.lora_A[self.adapter].weight.detach().cpu().clone(),m.lora_B[self.adapter].weight.detach().cpu().clone()) for n,m,_,_ in self.mods}
    def restore(self,snap):
        for n,m,_,_ in self.mods:
            m.lora_A[self.adapter].weight.data.copy_(snap[n][0]);m.lora_B[self.adapter].weight.data.copy_(snap[n][1])
    def ablate(self,names):
        names=set(names)
        for n,m,_,_ in self.mods:
            if n in names:m.lora_B[self.adapter].weight.data.zero_()
    def only(self,names,snap):
        keep=set(names)
        for n,m,_,_ in self.mods:
            if n not in keep:m.lora_B[self.adapter].weight.data.zero_()
            else:
                m.lora_A[self.adapter].weight.data.copy_(snap[n][0]);m.lora_B[self.adapter].weight.data.copy_(snap[n][1])


def random_disjoint(all_names,target,k,seed):
    pool=sorted(set(all_names)-set(target));
    if len(pool)<k: raise RuntimeError('no disjoint same-size matched control')
    return random.Random(seed).sample(pool,k)


def gate_eval(base,full,controls):
    bs={}; fs={}
    for split in ('CONFIRMATION','GENERATOR_HOLDOUT'):
        bs[split]=[x for x in base if x['split']==split]; fs[split]=[x for x in full if x['split']==split]
    nat_b=[x for x in base if x['split']=='NATURAL_LOCKED']; nat_f=[x for x in full if x['split']=='NATURAL_LOCKED']
    if not nat_b or len(nat_b)!=len(nat_f): raise RuntimeError('natural locked rows missing or misaligned')
    nat_controls={k:[x for x in rows if x['split']=='NATURAL_LOCKED'] for k,rows in controls.items()}
    if any(len(v)!=len(nat_f) for v in nat_controls.values()): raise RuntimeError('natural control rows missing or misaligned')
    ci=bootstrap_delta(accvec(nat_f),accvec(nat_b))
    dnat=metric(nat_f)['accuracy']-metric(nat_b)['accuracy']
    dconf=metric(fs['CONFIRMATION'])['accuracy']-metric(bs['CONFIRMATION'])['accuracy']
    dgen=metric(fs['GENERATOR_HOLDOUT'])['accuracy']-metric(bs['GENERATOR_HOLDOUT'])['accuracy']
    full_nat=metric(nat_f)['accuracy']; control_nat={k:metric(v)['accuracy'] for k,v in nat_controls.items()}
    best=max(control_nat.values())
    pos=(dnat>=POSITIVE['natural_min'] and dconf>=POSITIVE['confirmation_min'] and dgen>=POSITIVE['generator_min'] and ci['low']>0 and full_nat-best>=POSITIVE['control_min'])
    neg=(dnat<=FAILURE['natural_max'] and dconf<=FAILURE['confirmation_max'] and dgen<=FAILURE['generator_max'] and ci['high']<0 and all(full_nat-x<=FAILURE['control_max'] for x in control_nat.values()))
    return {'natural_delta':dnat,'confirmation_delta':dconf,'generator_holdout_delta':dgen,'natural_ci':ci,'natural_control_accuracy':control_nat,'full_minus_best_control':full_nat-best,'positive_green':pos,'failure_green':neg}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-dir',required=True);ap.add_argument('--adapter-escrow',required=True);ap.add_argument('--procedural',required=True);ap.add_argument('--natural-json',required=True);ap.add_argument('--out',required=True);ap.add_argument('--batch',type=int,default=16)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    proc=rows_proc(a.procedural);nat=rows_natural(a.natural_json)
    for r in nat:r['split']='NATURAL_LOCKED'
    allrows=proc+nat
    tok=AutoTokenizer.from_pretrained(a.base_dir,local_files_only=True,trust_remote_code=False,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(a.base_dir,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cpu().eval()
    model=PeftModel.from_pretrained(base,Path(a.adapter_escrow)/'rte',adapter_name='rte',is_trainable=False)
    model.load_adapter(Path(a.adapter_escrow)/'copa',adapter_name='copa',is_trainable=False)
    model.disable_adapter_layers();par=parity_smoke(model,tok,nat);assert par['pass'],par
    cond={}
    model.disable_adapter_layers();cond['BASE']=score_condition(model,tok,allrows,a.batch)
    model.enable_adapter_layers();model.set_adapter('rte');cond['FULL_RTE']=score_condition(model,tok,allrows,a.batch)
    randomize(model,'rte','random_sign',224,'random_sign');cond['RANDOM_NORM_MATCHED']=score_condition(model,tok,allrows,a.batch)
    randomize(model,'rte','layer_shuffle',1337,'layer_shuffle');cond['LAYER_SHUFFLE']=score_condition(model,tok,allrows,a.batch)
    model.set_adapter('copa');cond['WRONG_TASK_COPA']=score_condition(model,tok,allrows,a.batch)
    for sc in (.25,.5,1.5):
        name=f'dose_{str(sc).replace(".","p")}';dose(model,'rte',name,sc);cond['DOSE_'+str(sc).replace('.','p')]=score_condition(model,tok,allrows,a.batch)
    model.set_adapter('rte')
    shuffled=[];rng=random.Random(2257401);prompts=[r['prompt'] for r in proc];rng.shuffle(prompts)
    for r,p in zip(proc,prompts): q=dict(r);q['prompt']=p;shuffled.append(q)
    model.disable_adapter_layers(); pbase_rows=score_condition(model,tok,shuffled,a.batch); model.enable_adapter_layers(); model.set_adapter('rte')
    pfull_rows=score_condition(model,tok,shuffled,a.batch)
    gate=gate_eval(cond['BASE'],cond['FULL_RTE'],{k:cond[k] for k in ('RANDOM_NORM_MATCHED','LAYER_SHUFFLE','WRONG_TASK_COPA')})
    pbase=metric(pbase_rows)['accuracy'];pfull=metric(pfull_rows)['accuracy'];orig=metric([x for x in cond['FULL_RTE'] if x['split']!='NATURAL_LOCKED'])['accuracy']
    retention=pfull/orig if orig>0 else 1.0;safety=retention<=.80
    lane='positive' if gate['positive_green'] else ('failure' if gate['failure_green'] else None)
    causal={'status':'NOT_RUN_NO_CONTROL_SEPARATED_REPRODUCTION','lane':lane,'broad':{},'holm':{},'selected_layers':{},'locked_validation':{}}
    if lane:
        model.set_adapter('rte');ctl=Controller(model);snap=ctl.snapshot();disc=[x for x in proc if x['split']=='DISCOVERY']+nat[:64];locked=[x for x in proc if x['split']=='GENERATOR_HOLDOUT']+nat[64:128]
        full_d=score_condition(model,tok,disc,a.batch);base_d=score_condition_disabled=model.disable_adapter_layers() or score_condition(model,tok,disc,a.batch);model.enable_adapter_layers();model.set_adapter('rte')
        fb=metric(full_d)['accuracy'];bb=metric(base_d)['accuracy'];rawp={};surv=[]
        for oi,obj in enumerate(('Q_ALL','V_ALL','EARLY_QV','MIDDLE_QV','LATE_QV')):
            names=ctl.names(obj);ctl.restore(snap);ctl.ablate(names);abl=score_condition(model,tok,disc,a.batch);ctl.restore(snap);ctl.only(names,snap);only=score_condition(model,tok,disc,a.batch);ctl.restore(snap)
            rnd=random_disjoint(ctl.all_names(),names,len(names),9000+oi);ctl.ablate(rnd);rabl=score_condition(model,tok,disc,a.batch);ctl.restore(snap);ctl.only(rnd,snap);ronly=score_condition(model,tok,disc,a.batch);ctl.restore(snap)
            if lane=='positive':necess=fb-metric(abl)['accuracy'];suff=metric(only)['accuracy']-bb;rn=fb-metric(rabl)['accuracy'];rs=metric(ronly)['accuracy']-bb
            else:necess=metric(abl)['accuracy']-fb;suff=bb-metric(only)['accuracy'];rn=metric(rabl)['accuracy']-fb;rs=bb-metric(ronly)['accuracy']
            p,rr,hh=directional_one_sided_p(accvec(full_d),accvec(abl),want_a_gt_b=(lane=='positive'))
            causal['broad'][obj]={'sites':len(names),'necessity':necess,'sufficiency':suff,'random_necessity':rn,'random_sufficiency':rs,'control_margin':min(necess-rn,suff-rs),'raw_p':p,'paired_rescue':rr,'paired_harm':hh};rawp[obj]=p
        causal['holm']=holm(rawp);surv=[o for o in rawp if causal['holm'][o] and causal['broad'][o]['necessity']>=.03 and causal['broad'][o]['sufficiency']>=.03 and causal['broad'][o]['control_margin']>=.01]
        layers=sorted({layer_of(n) for o in surv for n in ctl.names(o) if layer_of(n) is not None})
        lp={}
        for l in layers:
            obj=f'LAYER_{l}';names=ctl.names(obj);ctl.restore(snap);ctl.ablate(names);abl=score_condition(model,tok,disc,a.batch);ctl.restore(snap);ctl.only(names,snap);only=score_condition(model,tok,disc,a.batch);ctl.restore(snap)
            if lane=='positive':necess=fb-metric(abl)['accuracy'];suff=metric(only)['accuracy']-bb
            else:necess=metric(abl)['accuracy']-fb;suff=bb-metric(only)['accuracy']
            p,_,_=directional_one_sided_p(accvec(full_d),accvec(abl),want_a_gt_b=(lane=='positive'));causal['selected_layers'][str(l)]={'sites':len(names),'necessity':necess,'sufficiency':suff,'raw_p':p};lp[str(l)]=p
        lholm=holm(lp);fine=[int(l) for l in lp if lholm[l] and causal['selected_layers'][l]['necessity']>=.03 and causal['selected_layers'][l]['sufficiency']>=.03]
        if fine:
            full_l=score_condition(model,tok,locked,a.batch);model.disable_adapter_layers();base_l=score_condition(model,tok,locked,a.batch);model.enable_adapter_layers();model.set_adapter('rte');fl=metric(full_l)['accuracy'];bl=metric(base_l)['accuracy']
            for l in fine:
                names=ctl.names(f'LAYER_{l}');ctl.restore(snap);ctl.ablate(names);abl=score_condition(model,tok,locked,a.batch);ctl.restore(snap);ctl.only(names,snap);only=score_condition(model,tok,locked,a.batch);ctl.restore(snap)
                if lane=='positive':n=fl-metric(abl)['accuracy'];s=metric(only)['accuracy']-bl
                else:n=metric(abl)['accuracy']-fl;s=bl-metric(only)['accuracy']
                causal['locked_validation'][str(l)]={'necessity':n,'sufficiency':s,'pass':n>=.02 and s>=.02}
            final=[l for l in fine if causal['locked_validation'][str(l)]['pass']]
            causal['status']='E3_FINE_LAYER_LOCALIZATION_AWAIT_PROJECTION_SUBSPACE_RANDOM_AND_FRESH_PROCESS' if final else 'BROAD_ONLY_NO_LOCKED_FINE_SURVIVOR';causal['fine_survivors']=final
        elif surv:causal['status']='E3_BROAD_LOCALIZATION_NO_FINE_LAYER_SURVIVOR'
        else:causal['status']='CAUSAL_SCREEN_NO_HOLM_CONTROLLED_SURVIVOR'
        ctl.restore(snap)
    report={'schema':'R22574_C58_BEHAVIOR_CAUSAL_RAWFREE_V2','candidate':'C58_PORTAL_RTE_PORTABILITY','parity':par,'source_rows':{'procedural':len(proc),'natural_locked':len(nat)},'behavior':{'metrics':{k:metric(v) for k,v in cond.items()},'splits':{k:split_metric(v) for k,v in cond.items()},'gate':gate,'reproduction_green':gate['positive_green'],'failure_reproduction_green':gate['failure_green'],'premise_shuffle':{'base_accuracy':pbase,'full_accuracy':pfull,'retention':retention,'both_scored_on_shuffled_prompts':True},'safety_green':safety},'causal':causal,'case_digest':{k:sha_json([{'id':x['id'],'correct':x['correct'],'pred':x['pred']} for x in v]) for k,v in cond.items()},'raw_contract':{'raw_logits':False,'raw_activations':False,'hidden_states':False,'natural_text':False,'adapter_AB_values':False,'deltaW':False}}
    (out/'C58_BEHAVIOR_CAUSAL_RAWFREE.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'reproduction':gate['positive_green'],'failure':gate['failure_green'],'safety':safety,'causal':causal['status']},sort_keys=True))

if __name__=='__main__':main()
