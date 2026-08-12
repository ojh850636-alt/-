from __future__ import annotations
import argparse,hashlib,json,random,re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from c40_suite import build_wire

ALLOWED={'ignore','store_episodic','promote_semantic'}
STORE={'store_episodic','promote_semantic'}
WORD_RE=re.compile(r"[A-Za-z0-9_'-]+")
STOP={'the','a','an','and','or','to','of','in','on','for','is','are','was','were','be','as','that','this','it','i','you','my','we'}
SYSTEM_PROMPT=(
    'Act as a durable-memory admission classifier. Return exactly one JSON object and no surrounding text. '
    'Use fields in this order: reasoning, action, memory, facts, indexables. '
    'action must be ignore, store_episodic, or promote_semantic. '
    'Any stored memory or fact must be grounded in the current assistant response.'
)
USER_PREFIX=(
    'Decide whether the assistant response below contains durable information worth memory extraction. '
    'Use store_episodic for specific durable events/results, promote_semantic for stable facts/preferences, '
    'and ignore for transient or non-durable text.\n\nAssistant response:\n'
)

def tokens(s):return {x.lower() for x in WORD_RE.findall(s) if len(x)>2 and x.lower() not in STOP}
def parse(raw,response):
    try:o=json.loads(raw.strip())
    except Exception:return {'parse_valid':False,'action':None,'gate':None,'grounding_overlap':0,'grounding_pass':False}
    if not isinstance(o,dict) or o.get('action') not in ALLOWED:return {'parse_valid':False,'action':o.get('action') if isinstance(o,dict) else None,'gate':None,'grounding_overlap':0,'grounding_pass':False}
    a=o['action'];g='IGNORE' if a=='ignore' else 'STORE_LIKE';mem=o.get('memory');content=mem.get('content') if isinstance(mem,dict) and isinstance(mem.get('content'),str) else None
    if g=='IGNORE':ov=0;gp=(mem is None)
    else:ov=len(tokens(content or '')&tokens(response));gp=bool(content) and ov>=2
    return {'parse_valid':True,'action':a,'gate':g,'grounding_overlap':ov,'grounding_pass':gp}

def lora_modules(model):
    out=[]
    for name,m in model.named_modules():
        if hasattr(m,'lora_A') and hasattr(m,'lora_B') and getattr(m,'lora_A',None):
            keys=list(m.lora_A.keys())
            if keys:out.append((name,m,keys[0]))
    return out

def apply_dose(model,dose):
    for _,m,k in lora_modules(model):m.scaling[k]=float(m.scaling[k])*float(dose)

def apply_random_sign(model,seed=22543040):
    for name,m,k in lora_modules(model):
        b=m.lora_B[k].weight.data
        h=int(hashlib.sha256((name+str(seed)).encode()).hexdigest()[:16],16);rng=random.Random(h)
        signs=torch.tensor([1.0 if rng.random()>=.5 else -1.0 for _ in range(b.shape[1])],dtype=b.dtype,device=b.device)
        b.mul_(signs.unsqueeze(0))

def apply_layer_shuffle(model):
    groups={}
    for name,m,k in lora_modules(model):
        proj=next((p for p in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj') if name.endswith(p)),None)
        if proj:groups.setdefault(proj,[]).append((name,m,k))
    for proj,mods in groups.items():
        mods=sorted(mods,key=lambda x:x[0]); snaps=[(m.lora_A[k].weight.detach().clone(),m.lora_B[k].weight.detach().clone()) for _,m,k in mods]
        if len(mods)<2:continue
        for i,(_,m,k) in enumerate(mods):
            a,b=snaps[(i+1)%len(snaps)];m.lora_A[k].weight.data.copy_(a);m.lora_B[k].weight.data.copy_(b)

def render(tok,response):
    msgs=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':USER_PREFIX+response}]
    return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)

def load(condition,base,adapters):
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
    b=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':'cpu'})
    if condition=='BASE':return b,tok,0
    amap={'STORAGE_FULL':'storage','RETRIEVAL_PLAN':'retrieval_plan','CONSOLIDATION':'consolidation','STORAGE_RANDOM_SIGN':'storage','STORAGE_LAYER_SHUFFLE':'storage','STORAGE_DOSE_0.25':'storage','STORAGE_DOSE_0.5':'storage','STORAGE_DOSE_1.5':'storage'}
    role=amap[condition];m=PeftModel.from_pretrained(b,str(Path(adapters)/role),local_files_only=True,is_trainable=False)
    mods=lora_modules(m);assert len(mods)>0
    if condition.startswith('STORAGE_DOSE_'):apply_dose(m,float(condition.rsplit('_',1)[1]))
    elif condition=='STORAGE_RANDOM_SIGN':apply_random_sign(m)
    elif condition=='STORAGE_LAYER_SHUFFLE':apply_layer_shuffle(m)
    return m,tok,len(mods)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--condition',required=True);ap.add_argument('--base',required=True);ap.add_argument('--adapters',required=True);ap.add_argument('--out',required=True);ap.add_argument('--batch',type=int,default=8);a=ap.parse_args()
    suite=build_wire();m,tok,nmods=load(a.condition,a.base,a.adapters);m.eval();rows=suite['cases'];results=[]
    with torch.inference_mode():
      for st in range(0,len(rows),a.batch):
        rr=rows[st:st+a.batch];prompts=[render(tok,r['assistant_response']) for r in rr];enc=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=768);n=enc['input_ids'].shape[1]
        g=m.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
        for i,r in enumerate(rr):
            raw=tok.decode(g[i,n:],skip_special_tokens=True);p=parse(raw,r['assistant_response']);expected=r['expected_gate'];correct=None if expected=='BOUNDARY' else bool(p['parse_valid'] and p['gate']==expected)
            results.append({'case_id':r['case_id'],'split':r['split'],'semantic_class':r['semantic_class'],'expected_gate':expected,'parse_valid':p['parse_valid'],'predicted_gate':p['gate'],'action':p['action'],'correct':correct,'grounding_overlap':p['grounding_overlap'],'grounding_pass':p['grounding_pass'],'output_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_output_chars':len(raw)})
    out={'schema':'LUCIA_AA_R22543_C40_BEHAVIOR_CONDITION_V1','condition':a.condition,'suite_sha256':'b01aac567a1e6165475a018e0dfe447ef48b33ec47e6a51bb913d841fd690cec','case_count':len(results),'lora_module_count':nmods,'generation':{'do_sample':False,'max_new_tokens':160},'results':results,'raw_outputs_saved':False,'raw_weights_saved':False,'raw_logits_saved':False,'hf_model_redownloads':0}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
