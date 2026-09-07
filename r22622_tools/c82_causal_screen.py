import os,re,json,pathlib,sys
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

# Reuse the exact precommitted primary suite/scorer source without executing its model section.
os.environ.setdefault('SHARD','0')
os.environ.setdefault('CONDITION','FULL')
primary_src=pathlib.Path('c82_matrix_runner.py').read_text()
prefix=primary_src.split("tok=AutoTokenizer.from_pretrained")[0]
ns={}
exec(prefix,ns)
cases=ns['cases']; score=ns['score']
by_id={c['id']:c for c in cases}
screen_ids=['C00','C02','C04','C06','C08','C10','C12','C14','O00','O02','O04','O06','O08','O10','O12','O14']
selected=[by_id[i] for i in screen_ids]
all_secret=[c['request']['secret'].lower() for c in cases]
all_twist=[c['request']['twist'].lower() for c in cases]

HYPOTHESES={
 'ATTENTION_ALL':({'q_proj','k_proj','v_proj','o_proj'},set(range(30))),
 'MLP_ALL':({'gate_proj','up_proj','down_proj'},set(range(30))),
 'Q_PROJ':({'q_proj'},set(range(30))),
 'K_PROJ':({'k_proj'},set(range(30))),
 'V_PROJ':({'v_proj'},set(range(30))),
 'O_PROJ':({'o_proj'},set(range(30))),
 'GATE_PROJ':({'gate_proj'},set(range(30))),
 'UP_PROJ':({'up_proj'},set(range(30))),
 'DOWN_PROJ':({'down_proj'},set(range(30))),
 'EARLY':({'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'},set(range(0,10))),
 'MIDDLE':({'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'},set(range(10,20))),
 'LATE':({'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'},set(range(20,30))),
}
hyp=os.environ['HYPOTHESIS']
mods,layers=HYPOTHESES[hyp]
ROOT=pathlib.Path('source_bundle/src'); BASE=str(ROOT/'base'); AD=str(ROOT/'adapter')
tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True); tok.padding_side='left'; tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,dtype=torch.float32)
model=PeftModel.from_pretrained(base,AD,local_files_only=True); model.eval()
rx=re.compile(r'layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.lora_([AB])\.default\.weight$')
zeroed=[]
with torch.no_grad():
 for n,p in model.named_parameters():
  if '.lora_B.' not in n: continue
  m=rx.search(n)
  if not m: continue
  layer=int(m.group(1)); mod=m.group(3)
  if mod in mods and layer in layers:
   p.zero_(); zeroed.append(n)
assert zeroed, hyp
rows=[]
for b0 in range(0,len(selected),4):
 batch=selected[b0:b0+4]
 enc=tok([c['prompt'] for c in batch],return_tensors='pt',padding=True)
 with torch.inference_mode():
  gen=model.generate(**enc,max_new_tokens=96,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
 ilen=enc['input_ids'].shape[1]
 for c,g in zip(batch,gen):
  out=tok.decode(g[ilen:],skip_special_tokens=True)
  rows.append({'id':c['id'],'split':c['split'],'hypothesis':hyp,'output':out,'metrics':score(c,out)})
path=pathlib.Path('result'); path.mkdir(exist_ok=True)
(path/f'{hyp}.json').write_text(json.dumps({'schema':'R22622_C82_CAUSAL_SCREEN_OUTPUT_V1','hypothesis':hyp,'zeroed_B_tensors':len(zeroed),'rows':rows},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
