from __future__ import annotations
import argparse, hashlib, importlib.util, json, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

OBJECTS = [
    'EARLY_BAND','MIDDLE_BAND','LATE_BAND','ATTENTION_ALL','MLP_ALL',
    'Q_PROJ','K_PROJ','V_PROJ','O_PROJ','GATE_PROJ','UP_PROJ','DOWN_PROJ'
]
PROJ_MAP = {
    'Q_PROJ':'q_proj','K_PROJ':'k_proj','V_PROJ':'v_proj','O_PROJ':'o_proj',
    'GATE_PROJ':'gate_proj','UP_PROJ':'up_proj','DOWN_PROJ':'down_proj',
}

def load_helper(path: str):
    spec = importlib.util.spec_from_file_location('c40_behavior_pinned', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod

def lora_modules(model):
    rows=[]
    for name,m in model.named_modules():
        if hasattr(m,'lora_A') and hasattr(m,'lora_B') and getattr(m,'lora_A',None):
            keys=list(m.lora_A.keys())
            if keys:
                k=keys[0]
                mm=re.search(r'layers\.(\d+)\.',name)
                layer=int(mm.group(1)) if mm else -1
                proj=next((p for p in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj') if name.endswith(p)),None)
                rows.append({'name':name,'module':m,'key':k,'layer':layer,'proj':proj,'scale':float(m.scaling[k])})
    rows.sort(key=lambda x:x['name'])
    return rows

def selected(rows, obj):
    layers=[r['layer'] for r in rows if r['layer']>=0]
    nl=max(layers)+1
    cut=max(1,nl//3)
    if obj=='EARLY_BAND': return [r for r in rows if 0<=r['layer']<cut]
    if obj=='MIDDLE_BAND': return [r for r in rows if cut<=r['layer']<2*cut]
    if obj=='LATE_BAND': return [r for r in rows if r['layer']>=2*cut]
    if obj=='ATTENTION_ALL': return [r for r in rows if r['proj'] in {'q_proj','k_proj','v_proj','o_proj'}]
    if obj=='MLP_ALL': return [r for r in rows if r['proj'] in {'gate_proj','up_proj','down_proj'}]
    if obj in PROJ_MAP: return [r for r in rows if r['proj']==PROJ_MAP[obj]]
    raise KeyError(obj)

def configure(rows, target, mode):
    ts={r['name'] for r in target}
    if mode=='NECESSITY':
        for r in rows: r['module'].scaling[r['key']] = 0.0 if r['name'] in ts else r['scale']
    elif mode=='SUFFICIENCY':
        for r in rows: r['module'].scaling[r['key']] = r['scale'] if r['name'] in ts else 0.0
    else: raise KeyError(mode)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--object',required=True,choices=OBJECTS)
    ap.add_argument('--mode',required=True,choices=['NECESSITY','SUFFICIENCY'])
    ap.add_argument('--base',required=True)
    ap.add_argument('--adapter',required=True)
    ap.add_argument('--helper',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--batch',type=int,default=8)
    a=ap.parse_args()
    h=load_helper(a.helper)
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,use_fast=True)
    tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'
    base=AutoModelForCausalLM.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':'cpu'})
    model=PeftModel.from_pretrained(base,a.adapter,local_files_only=True,is_trainable=False)
    rows=lora_modules(model); assert len(rows)==168, len(rows)
    target=selected(rows,a.object); assert target
    configure(rows,target,a.mode)
    suite=h.build_wire()['cases']
    cases=[r for r in suite if r['split']=='DISCOVERY']; assert len(cases)==64
    model.eval(); results=[]
    with torch.inference_mode():
        for st in range(0,len(cases),a.batch):
            rr=cases[st:st+a.batch]
            prompts=[h.render(tok,r['assistant_response']) for r in rr]
            enc=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=768)
            n=enc['input_ids'].shape[1]
            g=model.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
            for i,r in enumerate(rr):
                raw=tok.decode(g[i,n:],skip_special_tokens=True)
                p=h.parse(raw,r['assistant_response'])
                exp=r['expected_gate']
                correct=None if exp=='BOUNDARY' else bool(p['parse_valid'] and p['gate']==exp)
                results.append({'case_id':r['case_id'],'semantic_class':r['semantic_class'],'expected_gate':exp,'parse_valid':p['parse_valid'],'predicted_gate':p['gate'],'action':p['action'],'correct':correct,'grounding_pass':p['grounding_pass'],'output_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_output_chars':len(raw)})
    out={'schema':'LUCIA_AA_R22543_C40_CAUSAL_BROAD_CONFIG_V1','object':a.object,'mode':a.mode,'target_site_count':len(target),'target_site_names_sha256':hashlib.sha256('\n'.join(sorted(r['name'] for r in target)).encode()).hexdigest(),'discovery_cases':64,'scored_cases':48,'results':results,'raw_outputs_saved':False,'raw_weights_saved':False,'raw_logits_saved':False,'hf_model_redownloads':0,'claim_cap':'E3_BOUNDED_CAUSAL_LOCALIZATION_TRAINING_PROVENANCE_HOLD'}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
