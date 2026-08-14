from __future__ import annotations
import argparse, json, hashlib, gc, random
from pathlib import Path
# Independent minimal replay intentionally does not import c44_science.
# It independently reconstructs true BASE vs FULL even when PEFT auto-saved resized embedding/head state exists.
SEED=22564
ADAPTER_SHA='ca0710c04480627b50bb1350bd32fdf5545a95c17ecab38eab3cd6f92c24c0d7'
BASE_SHA='6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933'
SPECIALS=['<|json|>','</|json|>']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def rows(p):
    out=[]
    for x in Path(p).read_text().splitlines():
        if not x.strip(): continue
        r=json.loads(x)
        if not r['negative']: out.append(r)
    return out

def score(model,tok,rs):
    import torch
    out=[]
    for r in rs:
        p=r['prompt']; t=r['target']+tok.eos_token
        e=tok(p+t,return_tensors='pt',add_special_tokens=False)
        plen=len(tok(p,add_special_tokens=False)['input_ids']); flen=e['input_ids'].shape[1]
        with torch.inference_mode(): lp=torch.log_softmax(model(**e).logits[0].float(),-1)
        toks=e['input_ids'][0,plen:flen]; pred=lp[plen-1:flen-1]
        out.append(float(pred.gather(1,toks[:,None]).mean()))
    return sum(out)/len(out)

def load_full_and_base_snapshot(root,tok):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    torch.manual_seed(SEED)
    b=AutoModelForCausalLM.from_pretrained(str(root/'base'),local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    if b.get_input_embeddings().num_embeddings < len(tok): b.resize_token_embeddings(len(tok))
    base_embed=b.get_input_embeddings().weight.detach().cpu().clone()
    base_head=b.get_output_embeddings().weight.detach().cpu().clone()
    m=PeftModel.from_pretrained(b,str(root/'adapter'),local_files_only=True,is_trainable=False);m.eval()
    return m,base_embed,base_head

def true_base(model,base_embed,base_head):
    import torch
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(base_embed.to(model.get_input_embeddings().weight.device,dtype=model.get_input_embeddings().weight.dtype))
        model.get_output_embeddings().weight.copy_(base_head.to(model.get_output_embeddings().weight.device,dtype=model.get_output_embeddings().weight.dtype))
    for _,mod in model.named_modules():
        if hasattr(mod,'scaling') and getattr(mod,'scaling',None):
            for k in list(mod.scaling): mod.scaling[k]=0.0

def main():
    a=argparse.ArgumentParser();a.add_argument('--work');a.add_argument('--suite');a.add_argument('--primary-behavior');a.add_argument('--out');x=a.parse_args()
    primary=json.load(open(x.primary_behavior)); gate=bool(primary.get('positive_capability_gate') or primary.get('failure_gate'))
    out={'schema':'LAA_R22564_C44_INDEPENDENT_REPLAY_V2','entered':gate,'independent_implementation':True,'true_base_auxiliary_restoration':True}
    if not gate:
        out.update({'verdict':'NOT_RUN_PRIMARY_GATE_CLOSED'});Path(x.out).write_text(json.dumps(out,indent=2,sort_keys=True));return
    import torch
    from transformers import AutoTokenizer
    root=Path(x.work); assert sha(root/'base/model.safetensors')==BASE_SHA;assert sha(root/'adapter/adapter_model.safetensors')==ADAPTER_SHA
    tok=AutoTokenizer.from_pretrained(str(root/'base'),local_files_only=True);tok.add_special_tokens({'additional_special_tokens':SPECIALS});tok.pad_token=tok.eos_token
    rs=[r for r in rows(x.suite) if r['split'] in {'CONFIRMATION','TACTICAL_OOD','QUIET_OOD'}][:32]
    m,base_embed,base_head=load_full_and_base_snapshot(root,tok)
    full=score(m,tok,rs)
    true_base(m,base_embed,base_head)
    base=score(m,tok,rs)
    delta=full-base; pd=float(primary['full_minus_base_nats']); same=(delta==0 and pd==0) or (delta*pd>0); mag=abs(delta)>=0.05
    out.update({'full_mean':full,'base_mean':base,'delta_nats':delta,'primary_delta_nats':pd,'same_sign':same,'minimum_magnitude_pass':mag,'verdict':'REPLAY_PASS' if same and mag else 'REPLAY_RED'})
    Path(x.out).write_text(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
