import os,re,json,pathlib
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
os.environ.setdefault('SHARD','0'); os.environ.setdefault('CONDITION','FULL')
primary_src=pathlib.Path('c82_matrix_runner.py').read_text(); prefix=primary_src.split("tok=AutoTokenizer.from_pretrained")[0]; ns={}; exec(prefix,ns)
cases=ns['cases']; score=ns['score']; selected=[c for c in cases if c['split'] in {'Confirmation','OOD'}]; assert len(selected)==32
H={
 'MIDDLE':({'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'},set(range(10,20))),
 'ATTENTION_ALL':({'q_proj','k_proj','v_proj','o_proj'},set(range(30))),
 'MLP_ALL':({'gate_proj','up_proj','down_proj'},set(range(30))),
}
hyp=os.environ['HYPOTHESIS']; mode=os.environ['MODE']; mods,layers=H[hyp]
ROOT=pathlib.Path('source_bundle/src'); BASE=str(ROOT/'base'); AD=str(ROOT/'adapter')
tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True); tok.padding_side='left'; tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,dtype=torch.float32); model=PeftModel.from_pretrained(base,AD,local_files_only=True); model.eval()
rx=re.compile(r'layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.lora_([AB])\.default\.weight$'); zeroed=[]
with torch.no_grad():
 for n,p in model.named_parameters():
  if '.lora_B.' not in n: continue
  m=rx.search(n)
  if not m: continue
  layer=int(m.group(1)); mod=m.group(3); inside=(mod in mods and layer in layers)
  dozero=(mode=='ABLATE' and inside) or (mode=='ONLY' and not inside)
  if dozero: p.zero_(); zeroed.append(n)
assert zeroed and mode in {'ABLATE','ONLY'}
rows=[]
for b0 in range(0,32,4):
 batch=selected[b0:b0+4]; enc=tok([c['prompt'] for c in batch],return_tensors='pt',padding=True)
 with torch.inference_mode(): gen=model.generate(**enc,max_new_tokens=96,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
 ilen=enc['input_ids'].shape[1]
 for c,g in zip(batch,gen):
  out=tok.decode(g[ilen:],skip_special_tokens=True); rows.append({'id':c['id'],'split':c['split'],'hypothesis':hyp,'mode':mode,'output':out,'metrics':score(c,out)})
path=pathlib.Path('result'); path.mkdir(exist_ok=True); (path/f'{hyp}_{mode}.json').write_text(json.dumps({'schema':'R22622_C82_FULL_CONFIRM_OUTPUT_V1','hypothesis':hyp,'mode':mode,'zeroed_B_tensors':len(zeroed),'rows':rows},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
