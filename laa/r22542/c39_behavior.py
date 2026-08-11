from __future__ import annotations
import argparse,contextlib,hashlib,json,random,re
from pathlib import Path

def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def mean(xs):
 xs=list(xs);return float(sum(xs)/len(xs)) if xs else 0.0

def gen_a(i,split):
 k=i%8;s=f'{i:03d}'
 if k==0: goal=f'α{s} : Type\nx{s} y{s} : α{s}\nh{s} : x{s} = y{s}\n⊢ y{s} = x{s}'; tac=f'exact h{s}.symm'; fam='EQ_SYMM'; src=f'theorem c39_a_{s} (α{s} : Type) (x{s} y{s} : α{s}) (h{s} : x{s} = y{s}) : y{s} = x{s} := by\n  {tac}'
 elif k==1: goal=f'P{s} Q{s} : Prop\nh{s} : P{s} ∧ Q{s}\n⊢ Q{s} ∧ P{s}';tac=f'exact ⟨h{s}.2, h{s}.1⟩';fam='AND_SWAP';src=f'theorem c39_a_{s} (P{s} Q{s} : Prop) (h{s} : P{s} ∧ Q{s}) : Q{s} ∧ P{s} := by\n  {tac}'
 elif k==2: goal=f'α{s} : Type\nP{s} : α{s} → Prop\na{s} : α{s}\nh{s} : ∀ x, P{s} x\n⊢ P{s} a{s}';tac=f'exact h{s} a{s}';fam='FORALL_APPLY';src=f'theorem c39_a_{s} (α{s} : Type) (P{s} : α{s} → Prop) (a{s} : α{s}) (h{s} : ∀ x, P{s} x) : P{s} a{s} := by\n  {tac}'
 elif k==3: goal=f'a{s} b{s} : Nat\n⊢ a{s} + b{s} = b{s} + a{s}';tac='omega';fam='NAT_ADD_COMM';src=f'theorem c39_a_{s} (a{s} b{s} : Nat) : a{s} + b{s} = b{s} + a{s} := by\n  omega'
 elif k==4: goal=f'a{s} b{s} : Nat\n⊢ a{s} ≤ a{s} + b{s}';tac='omega';fam='NAT_LE_ADD';src=f'theorem c39_a_{s} (a{s} b{s} : Nat) : a{s} ≤ a{s} + b{s} := by\n  omega'
 elif k==5: goal=f'x{s} y{s} : Int\n⊢ (x{s} + y{s})^2 = x{s}^2 + 2*x{s}*y{s} + y{s}^2';tac='ring';fam='INT_RING_SQUARE';src=f'theorem c39_a_{s} (x{s} y{s} : Int) : (x{s} + y{s})^2 = x{s}^2 + 2*x{s}*y{s} + y{s}^2 := by\n  ring'
 elif k==6: goal=f'α{s} : Type\nxs{s} ys{s} : List α{s}\n⊢ (xs{s} ++ ys{s}).length = xs{s}.length + ys{s}.length';tac='simp';fam='LIST_LENGTH_APPEND';src=f'theorem c39_a_{s} (α{s} : Type) (xs{s} ys{s} : List α{s}) : (xs{s} ++ ys{s}).length = xs{s}.length + ys{s}.length := by\n  simp'
 else: goal=f'P{s} Q{s} : Prop\nh{s} : P{s} → Q{s}\nhp{s} : P{s}\n⊢ Q{s}';tac=f'exact h{s} hp{s}';fam='IMPL_APPLY';src=f'theorem c39_a_{s} (P{s} Q{s} : Prop) (h{s} : P{s} → Q{s}) (hp{s} : P{s}) : Q{s} := by\n  {tac}'
 return {'case_id':f'C39-{i:05d}','split':split,'generator':'GENERATOR_A','family':fam,'goal_state':goal,'prompt':goal+':::','canonical_tactic':tac,'standalone_source':src,'positive':True}

def gen_b(i,split):
 j=1000+i;s=f'{j:04d}';k=i%8
 if k==0: goal=f'P{s} Q{s} : Prop\nh{s} : P{s} ∨ Q{s}\n⊢ Q{s} ∨ P{s}';tac=f'rcases h{s} with h | h <;> simp [h]';fam='OR_SWAP';src=f'theorem c39_b_{s} (P{s} Q{s} : Prop) (h{s} : P{s} ∨ Q{s}) : Q{s} ∨ P{s} := by\n  {tac}'
 elif k==1: goal=f'α{s} : Type\nP{s} : α{s} → Prop\na{s} : α{s}\nh{s} : P{s} a{s}\n⊢ ∃ x, P{s} x';tac=f'exact ⟨a{s}, h{s}⟩';fam='EXISTS_WITNESS';src=f'theorem c39_b_{s} (α{s} : Type) (P{s} : α{s} → Prop) (a{s} : α{s}) (h{s} : P{s} a{s}) : ∃ x, P{s} x := by\n  {tac}'
 elif k==2: goal=f'P{s} Q{s} R{s} : Prop\nh{s} : (P{s} ∧ Q{s}) ∧ R{s}\n⊢ P{s} ∧ (Q{s} ∧ R{s})';tac=f'exact ⟨h{s}.1.1, h{s}.1.2, h{s}.2⟩';fam='AND_ASSOC';src=f'theorem c39_b_{s} (P{s} Q{s} R{s} : Prop) (h{s} : (P{s} ∧ Q{s}) ∧ R{s}) : P{s} ∧ (Q{s} ∧ R{s}) := by\n  {tac}'
 elif k==3: goal=f'a{s} b{s} : Nat\n⊢ a{s} * b{s} = b{s} * a{s}';tac=f'exact Nat.mul_comm a{s} b{s}';fam='NAT_MUL_COMM';src=f'theorem c39_b_{s} (a{s} b{s} : Nat) : a{s} * b{s} = b{s} * a{s} := by\n  {tac}'
 elif k==4: goal=f'x{s} y{s} : Int\nh{s} : x{s} = y{s} + 3\n⊢ x{s} - y{s} = 3';tac='omega';fam='INT_LINEAR';src=f'theorem c39_b_{s} (x{s} y{s} : Int) (h{s} : x{s} = y{s} + 3) : x{s} - y{s} = 3 := by\n  omega'
 elif k==5: goal=f'α{s} : Type\nxs{s} : List α{s}\n⊢ xs{s}.reverse.reverse = xs{s}';tac='simp';fam='LIST_REVERSE_INV';src=f'theorem c39_b_{s} (α{s} : Type) (xs{s} : List α{s}) : xs{s}.reverse.reverse = xs{s} := by\n  simp'
 elif k==6: goal=f'α{s} β{s} : Type\nf{s} : α{s} → β{s}\nx{s} y{s} : α{s}\nh{s} : x{s} = y{s}\n⊢ f{s} x{s} = f{s} y{s}';tac=f'simpa [h{s}]';fam='CONGR_ARG';src=f'theorem c39_b_{s} (α{s} β{s} : Type) (f{s} : α{s} → β{s}) (x{s} y{s} : α{s}) (h{s} : x{s} = y{s}) : f{s} x{s} = f{s} y{s} := by\n  {tac}'
 else: goal=f'P{s} Q{s} R{s} : Prop\nh1{s} : P{s} → Q{s}\nh2{s} : Q{s} → R{s}\n⊢ P{s} → R{s}';tac=f'intro hp; exact h2{s} (h1{s} hp)';fam='IMPL_COMPOSE';src=f'theorem c39_b_{s} (P{s} Q{s} R{s} : Prop) (h1{s} : P{s} → Q{s}) (h2{s} : Q{s} → R{s}) : P{s} → R{s} := by\n  {tac}'
 return {'case_id':f'C39-{j:05d}','split':split,'generator':'GENERATOR_B','family':fam,'goal_state':goal,'prompt':goal+':::','canonical_tactic':tac,'standalone_source':src,'positive':True}

def neg(i):
 j=9000+i;s=f'{j:04d}';k=i%5
 if k==0: goal=f'u{s} : Unit\n⊢ False';src=f'theorem c39_n_{s} (u{s} : Unit) : False := by\n  __CANDIDATE_TACTIC__';fam='NEG_FALSE'
 elif k==1: goal=f'n{s} : Nat\n⊢ n{s} + 1 = n{s}';src=f'theorem c39_n_{s} (n{s} : Nat) : n{s} + 1 = n{s} := by\n  __CANDIDATE_TACTIC__';fam='NEG_NAT_SUCC_EQ'
 elif k==2: goal=f'P{s} : Prop\n⊢ P{s}';src=f'theorem c39_n_{s} (P{s} : Prop) : P{s} := by\n  __CANDIDATE_TACTIC__';fam='NEG_ARBITRARY_PROP'
 elif k==3: goal=f'a{s} b{s} : Nat\n⊢ a{s} = b{s}';src=f'theorem c39_n_{s} (a{s} b{s} : Nat) : a{s} = b{s} := by\n  __CANDIDATE_TACTIC__';fam='NEG_ARBITRARY_EQ'
 else: goal=f'α{s} : Type\nxs{s} : List α{s}\n⊢ xs{s} = []';src=f'theorem c39_n_{s} (α{s} : Type) (xs{s} : List α{s}) : xs{s} = [] := by\n  __CANDIDATE_TACTIC__';fam='NEG_ARBITRARY_LIST'
 return {'case_id':f'C39-{j:05d}','split':'NEGATIVE_UNPROVABLE','generator':'GENERATOR_NEG','family':fam,'goal_state':goal,'prompt':goal+':::','canonical_tactic':None,'standalone_source':src,'positive':False}

def build_suite():
 r=[]
 r += [gen_a(i,'DISCOVERY') for i in range(64)]
 r += [gen_a(i,'CONFIRMATION') for i in range(64,128)]
 r += [gen_b(i,'GENERATOR_HOLDOUT') for i in range(64)]
 r += [gen_b(i,'ALPHA_RENAME_OOD') for i in range(64,112)]
 for i in range(128,176):
  x=gen_a(i,'HYP_ORDER_OOD');ls=x['goal_state'].splitlines();x['goal_state']='\n'.join(list(reversed(ls[:-1]))+[ls[-1]]) if len(ls)>2 else x['goal_state'];x['prompt']=x['goal_state']+':::';r.append(x)
 r += [gen_b(i,'COMPOSITIONAL_OOD') for i in range(112,160)]
 r += [neg(i) for i in range(48)]
 assert len(r)==384 and len({x['prompt'] for x in r})==384
 return r

class Controller:
 def __init__(self,m):
  self.mods=[];layers=[]
  for n,x in m.named_modules():
   if hasattr(x,'lora_A') and hasattr(x,'lora_B') and getattr(x,'lora_A',None):
    key=next(iter(x.lora_A.keys()));mm=re.search(r'layers\.(\d+)\.',n);L=int(mm.group(1)) if mm else -1;layers += [L] if L>=0 else [];pr=next((p for p in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj') if n.endswith(p)),'OTHER');self.mods.append({'name':n,'m':x,'key':key,'scale':float(x.scaling[key]),'layer':L,'proj':pr})
  self.nl=max(layers)+1 if layers else 0
 def all(self,v):
  for x in self.mods:x['m'].scaling[x['key']]=x['scale']*float(v)
 def snap(self):return [(x,x['m'].lora_B[x['key']].weight.detach().cpu().clone()) for x in self.mods]
 def restore(self,s):
  for x,w in s:
   q=x['m'].lora_B[x['key']].weight;q.data.copy_(w.to(q.device,dtype=q.dtype))
 def random(self,seed):
  import torch;g=torch.Generator(device='cpu').manual_seed(seed)
  for x in self.mods:
   q=x['m'].lora_B[x['key']].weight;old=q.detach().float().cpu();z=torch.randn(old.shape,generator=g);z*=old.norm()/(z.norm()+1e-12);q.data.copy_(z.to(q.device,dtype=q.dtype))
 def shuffle(self,seed=22542):
  import torch;g=torch.Generator(device='cpu').manual_seed(seed)
  for x in self.mods:
   q=x['m'].lora_B[x['key']].weight;old=q.detach().flatten().cpu();p=torch.randperm(old.numel(),generator=g);q.data.copy_(old[p].reshape(q.shape).to(q.device,dtype=q.dtype))

def score(model,tok,rows,batch=2,disable=False):
 import torch
 pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;out=[];ctx=model.disable_adapter() if disable else contextlib.nullcontext();model.eval()
 with ctx,torch.inference_mode():
  for i in range(0,len(rows),batch):
   rr=rows[i:i+batch];pairs=[]
   for r in rr:
    p=tok(r['prompt'],add_special_tokens=False).input_ids;c=tok(r['canonical_tactic'],add_special_tokens=False).input_ids;pairs.append((p,c))
   mx=max(len(p)+len(c) for p,c in pairs);ids=[];masks=[];starts=[];lens=[]
   for p,c in pairs:
    q=p+c;n=len(q);ids.append(q+[pad]*(mx-n));masks.append([1]*n+[0]*(mx-n));starts.append(len(p));lens.append(len(c))
   ii=torch.tensor(ids,dtype=torch.long,device=model.device);am=torch.tensor(masks,dtype=torch.long,device=model.device);logits=model(input_ids=ii,attention_mask=am,use_cache=False).logits.float();lp=torch.log_softmax(logits[:,:-1,:],dim=-1)
   for j,r in enumerate(rr):
    st=starts[j];ln=lens[j];t=ii[j,st:st+ln];v=lp[j,st-1:st+ln-1].gather(-1,t.unsqueeze(-1)).squeeze(-1);out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'mean_logprob':float(v.mean().cpu()),'sum_logprob':float(v.sum().cpu()),'token_count':ln})
 return out

def summary(rows):
 by={}
 for x in rows:by.setdefault(x['split'],[]).append(x)
 def s(v):return {'n':len(v),'mean_logprob':mean(x['mean_logprob'] for x in v)}
 return {'overall':s(rows),'by_split':{k:s(v) for k,v in sorted(by.items())}}

def sanitize(s):
 s=s.replace('\r','\n').strip();s=s.split(':::',1)[0].split('\n',1)[0].strip();
 if s.startswith('by '):s=s[3:].strip()
 lo=s.lower();bad=('sorry','admit','axiom','unsafe','run_tac','set_option','#eval','#check','#print')
 if not s:return '','EMPTY'
 if any(x in lo for x in bad):return '','FORBIDDEN'
 if len(s)>600:return '','TOO_LONG'
 return s,'OK'

def generate(model,tok,rows,condition,seed,k=8):
 import torch
 tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token;torch.manual_seed(seed);out=[];model.eval()
 with torch.inference_mode():
  for r in rows:
   e=tok(r['prompt'],return_tensors='pt',add_special_tokens=False,truncation=True,max_length=768);e={a:b.to(model.device) for a,b in e.items()};n=e['input_ids'].shape[1]
   g=model.generate(**e,max_new_tokens=48,do_sample=True,temperature=.8,top_p=.95,top_k=50,num_return_sequences=k,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   for rank in range(k):
    raw=tok.decode(g[rank,n:],skip_special_tokens=True);t,st=sanitize(raw);out.append({'condition':condition,'seed':seed,'case_id':r['case_id'],'split':r['split'],'family':r['family'],'rank':rank,'status':st,'tactic':t,'tactic_sha256':hashlib.sha256(t.encode()).hexdigest() if t else None})
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--out',required=True);ap.add_argument('--private',required=True);a=ap.parse_args();out=Path(a.out);priv=Path(a.private);out.mkdir(parents=True,exist_ok=True);priv.mkdir(parents=True,exist_ok=True)
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 rows=build_suite();positive=[r for r in rows if r['positive']]
 tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,use_safetensors=True)
 model=PeftModel.from_pretrained(base,a.adapter,local_files_only=True,is_trainable=False);ctl=Controller(model);assert len(ctl.mods)==96,len(ctl.mods)
 # teacher-forced all 336 positives
 base_sc=score(model,tok,positive,disable=True);ctl.all(1);full_sc=score(model,tok,positive);conditions={'BASE':summary(base_sc),'FULL':summary(full_sc)};teacher={'BASE':base_sc,'FULL':full_sc}
 for d in (.25,.5,1.5):ctl.all(d);z=score(model,tok,positive);conditions[f'DOSE_{d}']=summary(z);teacher[f'DOSE_{d}']=z
 ctl.all(1);snap=ctl.snap();ctl.random(42);z=score(model,tok,positive);conditions['RANDOM_42']=summary(z);teacher['RANDOM_42']=z;ctl.restore(snap);ctl.shuffle(22542);z=score(model,tok,positive);conditions['SHUFFLED']=summary(z);teacher['SHUFFLED']=z;ctl.restore(snap);ctl.all(1)
 # deterministic locked generation subset, 12 per split
 locked=[]
 for sp in ('CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE'):
  locked += sorted([r for r in rows if r['split']==sp],key=lambda x:x['case_id'])[:12]
 private=[]
 # base
 with model.disable_adapter():private += generate(model,tok,locked,'BASE',42)
 for seed in (42,224,1337):ctl.all(1);private += generate(model,tok,locked,'FULL',seed)
 ctl.restore(snap);ctl.random(42);private += generate(model,tok,locked,'RANDOM_42',42);ctl.restore(snap);ctl.shuffle(22542);private += generate(model,tok,locked,'SHUFFLED',42);ctl.restore(snap);ctl.all(1)
 pp=priv/'C39_GENERATED_TACTICS_PRIVATE.jsonl';pp.write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in private),encoding='utf-8')
 # Public teacher evidence only; no raw tactic strings.
 pubteacher={k:[{'case_id':x['case_id'],'split':x['split'],'family':x['family'],'mean_logprob':x['mean_logprob'],'token_count':x['token_count']} for x in v] for k,v in teacher.items()}
 dump(out/'R22542_C39_TEACHER_BEHAVIOR.json',{'schema':'LUCIA_AA_R22542_C39_TEACHER_BEHAVIOR_V1','candidate_id':'R22542-C39-OPENPROOF-LEAN-TACTIC-2B','tokenizer_policy':'RUNTIME_REFERENCE_BASE_TOKENIZER__AUTHOR_SAVED_TOKENIZER_SEMANTICS_UNPROVEN','tokenizer_provenance_hold':True,'conditions':conditions,'case_evidence':pubteacher,'generation_private_records':len(private),'generation_locked_cases':len(locked),'generation_k':8,'generation_seeds':[42,224,1337],'raw_generated_tactics_exported':False,'model_outputs_observed':len(positive)*7+len(private),'training_time_exact_base_revision_proven':False})
 dump(out/'R22542_C39_MODEL_RUNTIME_RECEIPT.json',{'schema':'LUCIA_AA_R22542_C39_MODEL_RUNTIME_RECEIPT_V1','verdict':'PASS_MODEL_FORWARD_AND_PRIVATE_GENERATION','lora_module_count':len(ctl.mods),'num_layers':ctl.nl,'dtype':'bfloat16','device':'cpu','raw_tactics_path_private':str(pp),'raw_tactics_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'raw_tactics_export_allowed':False})
if __name__=='__main__':main()
