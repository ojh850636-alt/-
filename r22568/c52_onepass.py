from __future__ import annotations
import gc, hashlib, json, os, random, re, shutil, sys, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from r22568.c52_contract import generate_suite, verifier_a, verifier_b, suite_digest, contract_digest, THRESHOLDS, SCHEMA_FIELDS
CID='R22568-C52-CALENDAR-TEMPORAL-EXTRACTION'; SEED=22568
ADAPTER_REPO='waliaMuskaan011/calendar-event-extractor-smollm'; ADAPTER_REV='08d0bd53801b5bf035b44ff4ae084c94a51126ee'
BASE_REPO='HuggingFaceTB/SmolLM-360M'; BASE_REV='59f7ef243ee09a72cbc14cb054393a3e3b771d41'
EXPECTED_SUITE_SHA='34aa41d17083f277ea060459db4cf93a467d9f42494f30dfa0660e4f0b749b47'; EXPECTED_CONTRACT_SHA='9319dbc4e4ee1ad548278feb57326156ad9af0ffc3e2980c25f5181227d4e9e6'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; ROOT=Path.cwd(); PRIVATE=ROOT/'private_c52'; ADAPTER_DIR=PRIVATE/'adapter'; BASE_DIR=PRIVATE/'base'; HF_HOME=PRIVATE/'hf_home'; OUT=ROOT/'r22568_evidence'; OUT.mkdir(exist_ok=True); PRIVATE.mkdir(exist_ok=True); os.environ['HF_HOME']=str(HF_HOME); os.environ['TOKENIZERS_PARALLELISM']='false'
def sha256_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def write_json(n,o):(OUT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def prompt_for(t):return f'Extract calendar information from: "{t}"\nCalendar JSON:'
def target_text(t):return json.dumps({k:t.get(k) for k in SCHEMA_FIELDS},ensure_ascii=False,separators=(', ',': '))
def paired_bootstrap(d,seed=SEED,reps=4000):
 a=np.asarray(d,dtype=np.float64)
 if not len(a):return {'mean':None,'ci_low':None,'ci_high':None,'n':0}
 rng=np.random.default_rng(seed); n=len(a); vals=np.empty(reps)
 for i in range(reps):vals[i]=a[rng.integers(0,n,n)].mean()
 return {'mean':float(a.mean()),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),'n':n}
def signflip_pvalue(d,direction,seed,reps=12000):
 a=np.asarray(d,dtype=np.float64); obs=direction*a.mean(); rng=np.random.default_rng(seed); ext=1
 for _ in range(reps):
  s=rng.choice(np.array([-1.,1.]),size=len(a)); ext+=int(direction*(a*s).mean()>=obs-1e-15)
 return ext/(reps+1)
def holm(raw):
 items=sorted(raw.items(),key=lambda x:x[1]);m=len(items);out={};prev=0.
 for i,(k,p) in enumerate(items):adj=max(prev,min(1.,(m-i)*p));prev=adj;out[k]=adj
 return out
def layer_index(n):
 for pat in (r'layers\.(\d+)',r'layer\.(\d+)',r'h\.(\d+)'):
  m=re.search(pat,n)
  if m:return int(m.group(1))
 return None
def projection(n):
 for p in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):
  if p in n:return p
 return 'other'
def lora_layers(model):return [(n,m) for n,m in model.named_modules() if isinstance(m,BaseTunerLayer) and hasattr(m,'lora_B') and 'default' in getattr(m,'lora_B',{})]
def capture_state(model):
 out=[]
 for n,m in lora_layers(model):out.append({'name':n,'module':m,'A':m.lora_A['default'].weight.detach().cpu().clone(),'B':m.lora_B['default'].weight.detach().cpu().clone(),'scale':float(m.scaling['default']),'layer':layer_index(n),'projection':projection(n)})
 return out
def restore(rows):
 with torch.no_grad():
  for r in rows:
   r['module'].lora_A['default'].weight.copy_(r['A'].to(r['module'].lora_A['default'].weight.device)); r['module'].lora_B['default'].weight.copy_(r['B'].to(r['module'].lora_B['default'].weight.device)); r['module'].scaling['default']=r['scale']
def set_dose(rows,d):
 for r in rows:r['module'].scaling['default']=r['scale']*d
def set_random_sign(rows):
 g=torch.Generator(device='cpu');g.manual_seed(SEED+101)
 with torch.no_grad():
  for r in rows:
   b=r['B'];s=torch.randint(0,2,b.shape,generator=g,dtype=torch.int64).mul_(2).sub_(1).to(b.dtype);r['module'].lora_B['default'].weight.copy_((b*s).to(r['module'].lora_B['default'].weight.device))
def set_layer_shuffle(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['projection'],tuple(r['B'].shape))].append(r)
 with torch.no_grad():
  for g in groups.values():
   g=sorted(g,key=lambda x:(x['layer'] if x['layer'] is not None else 999,x['name']))
   if len(g)<2:continue
   vals=[x['B'] for x in g];vals=vals[1:]+vals[:1]
   for r,v in zip(g,vals):r['module'].lora_B['default'].weight.copy_(v.to(r['module'].lora_B['default'].weight.device))
def set_group(rows,pred,mode):
 for r in rows:
  hit=bool(pred(r));r['module'].scaling['default']=(0. if hit else r['scale']) if mode=='ABLATE' else (r['scale'] if hit else 0.)
def static_forensics():
 cfg=json.loads((ADAPTER_DIR/'adapter_config.json').read_text());pairs={};hidden=[]
 with safe_open(ADAPTER_DIR/'adapter_model.safetensors',framework='pt',device='cpu') as f:
  keys=list(f.keys())
  for k in keys:
   if '.lora_A.' in k or k.endswith('lora_A.weight'):pairs.setdefault(k.replace('.lora_A.default.weight','').replace('.lora_A.weight',''),{})['A']=f.get_tensor(k).float()
   elif '.lora_B.' in k or k.endswith('lora_B.weight'):pairs.setdefault(k.replace('.lora_B.default.weight','').replace('.lora_B.weight',''),{})['B']=f.get_tensor(k).float()
   else:hidden.append(k)
 rows=[];pe=defaultdict(float);le=defaultdict(float);zero=nonfinite=0
 for base,p in sorted(pairs.items()):
  if not {'A','B'}<=set(p):continue
  A,B=p['A'],p['B'];fin=bool(torch.isfinite(A).all() and torch.isfinite(B).all());nz=bool(torch.count_nonzero(A) and torch.count_nonzero(B));zero+=not nz;nonfinite+=not fin;e=float(torch.sum(A*A)*torch.sum(B*B)) if fin else 0.;pr=projection(base);li=layer_index(base);pe[pr]+=e;le[str(li)]+=e;rows.append({'module_id':base,'layer':li,'projection':pr,'a_shape':list(A.shape),'b_shape':list(B.shape),'effective_rank_a':int(torch.linalg.matrix_rank(A)) if fin else None,'effective_rank_b':int(torch.linalg.matrix_rank(B)) if fin else None,'factor_energy_proxy':e,'finite':fin,'nonzero':nz})
 total=sum(x['factor_energy_proxy'] for x in rows) or 1.
 return {'schema':'R22568_C52_STATIC_V1','adapter_config':{'r':cfg.get('r'),'lora_alpha':cfg.get('lora_alpha'),'target_modules':sorted(cfg.get('target_modules') or []),'modules_to_save':cfg.get('modules_to_save'),'task_type':cfg.get('task_type'),'base_model_name_or_path':cfg.get('base_model_name_or_path'),'revision':cfg.get('revision')},'adapter_tensor_count':len(keys),'complete_pair_count':len(rows),'zero_pair_count':int(zero),'nonfinite_pair_count':int(nonfinite),'hidden_non_lora_keys':hidden,'projection_energy_fraction':{k:v/total for k,v in sorted(pe.items())},'layer_energy_fraction':{k:v/total for k,v in sorted(le.items())},'module_scalar_inventory':rows,'raw_a_b_exported':False,'reconstructable_delta_w_exported':False,'claim_boundary':'E1_STATIC_STRUCTURE_ONLY_NOT_CAUSAL'}
def teacher_scores(model,tok,cases,batch_size=8):
 out=[];model.eval()
 for st in range(0,len(cases),batch_size):
  b=cases[st:st+batch_size];ps=[prompt_for(c['input']) for c in b];ts=[target_text(c['target']) for c in b];full=[p+t for p,t in zip(ps,ts)];enc=tok(full,return_tensors='pt',padding=True,truncation=True,max_length=512).to(DEVICE);pl=[len(tok(p,add_special_tokens=True,truncation=True,max_length=512)['input_ids']) for p in ps]
  with torch.inference_mode():logits=model(**enc).logits.float()
  ids=enc['input_ids'];mask=enc['attention_mask'];lp=torch.log_softmax(logits[:,:-1,:],-1).gather(-1,ids[:,1:].unsqueeze(-1)).squeeze(-1)
  for i,c in enumerate(b):
   n=int(mask[i].sum());p=min(pl[i],n);v=lp[i,max(0,p-1):max(0,n-1)];out.append({'id':c['id'],'split':c['split'],'mean_target_logprob':float(v.mean()) if v.numel() else -999.,'target_token_count':int(v.numel())})
  del enc,logits,ids,mask,lp
 return out
def aggregate_teacher(r):
 by=defaultdict(list)
 for x in r:by[x['split']].append(x['mean_target_logprob'])
 return {'overall':float(np.mean([x['mean_target_logprob'] for x in r])),'by_split':{k:float(np.mean(v)) for k,v in sorted(by.items())},'n':len(r)}
def generation_cases(cases):
 want={'CONFIRMATION':40,'TEMPORAL_OOD':16,'RECURRENCE_OOD':16,'NEGATIVE':8};used=defaultdict(int);out=[]
 for c in cases:
  if c['split'] in want and used[c['split']]<want[c['split']]:out.append(c);used[c['split']]+=1
 assert len(out)==80;return out
def generate_score(model,tok,cases):
 out=[];model.eval()
 for c in cases:
  enc=tok(prompt_for(c['input']),return_tensors='pt',truncation=True,max_length=384).to(DEVICE)
  with torch.inference_mode():seq=model.generate(**enc,max_new_tokens=112,do_sample=False,pad_token_id=tok.eos_token_id)
  text=tok.decode(seq[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);a=verifier_a(text,c['target']);b=verifier_b(text,c['target']);out.append({'id':c['id'],'split':c['split'],'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'output_length_chars':len(text),**a,**b})
 return out
def aggregate_generation(rows):
 def agg(rs):return {'n':len(rs),'json_valid':float(np.mean([x['json_valid'] for x in rs])),'field_exact':float(np.mean([x['field_exact'] for x in rs])),'exact':float(np.mean([x['exact'] for x in rs])),'typed_semantic':float(np.mean([x['typed_semantic'] for x in rs])),'schema':float(np.mean([x['schema'] for x in rs])),'negative_clean':float(np.mean([x['negative_clean'] for x in rs])),'false_event':float(np.mean([x['false_event'] for x in rs]))} if rs else {}
 return {'overall':agg(rows),'by_split':{s:agg([x for x in rows if x['split']==s]) for s in sorted({x['split'] for x in rows})}}
def cmap(rows):return {x['id']:x['mean_target_logprob'] for x in rows}
def behavior_gate(t,g):
 b,f,r,s=map(cmap,(t['BASE'],t['FULL'],t['RANDOM_SIGN'],t['LAYER_SHUFFLE']));ids=sorted(b);dif=[f[i]-b[i] for i in ids];boot=paired_bootstrap(dif);fr=float(np.mean([f[i]-r[i] for i in ids]));fs=float(np.mean([f[i]-s[i] for i in ids]));ag={k:aggregate_generation(v) for k,v in g.items()};fg=ag['FULL']['overall']['field_exact']-ag['BASE']['overall']['field_exact'];jg=ag['FULL']['overall']['json_valid']-ag['BASE']['overall']['json_valid'];nf=ag['FULL']['by_split']['NEGATIVE']['false_event'];nb=ag['BASE']['by_split']['NEGATIVE']['false_event'];pos=boot['ci_low']>0 and fr>.02 and fs>.02 and fg>=.10 and jg>=.05 and nf-nb<=.10;fail=boot['ci_high']<0 and fr<-.02 and fs<-.02 and (-fg>=.10 or -jg>=.05 or nf-nb>.10)
 return {'positive_reproduction_green':bool(pos),'failure_reproduction_green':bool(fail),'teacher_full_minus_base':boot,'teacher_full_minus_random':fr,'teacher_full_minus_shuffle':fs,'generation_field_exact_gain':fg,'generation_json_valid_gain':jg,'negative_false_event_increase':nf-nb,'generation_aggregates':ag,'causal_entry':bool(pos or fail)}
def causal_screen(model,tok,rows,cases,gate):
 locked=[c for c in cases if c['split']=='CONFIRMATION'];full=teacher_scores(model,tok,locked);fm=cmap(full)
 with model.disable_adapter():base=teacher_scores(model,tok,locked)
 bm=cmap(base);layers=[r['layer'] for r in rows if r['layer'] is not None];lo,hi=min(layers),max(layers);span=hi-lo+1;c1=lo+span//3;c2=lo+2*span//3;groups={p:(lambda r,p=p:r['projection']==p) for p in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')};groups.update({'early_band':lambda r:r['layer'] is not None and r['layer']<c1,'middle_band':lambda r:r['layer'] is not None and c1<=r['layer']<c2,'late_band':lambda r:r['layer'] is not None and r['layer']>=c2});direction=1 if gate['positive_reproduction_green'] else -1;outs=[];rawp={}
 for gi,(name,pred) in enumerate(groups.items()):
  restore(rows);set_group(rows,pred,'ABLATE');am=cmap(teacher_scores(model,tok,locked));restore(rows);set_group(rows,pred,'ONLY');om=cmap(teacher_scores(model,tok,locked));restore(rows);nec=[fm[i]-am[i] for i in fm] if direction==1 else [am[i]-fm[i] for i in fm];suf=[om[i]-bm[i] for i in bm] if direction==1 else [bm[i]-om[i] for i in bm];p=signflip_pvalue(nec,1,SEED+500+gi);rawp[name]=p;outs.append({'group':name,'site_count':sum(pred(r) for r in rows),'necessity_effect':float(np.mean(nec)),'necessity_ci':paired_bootstrap(nec,SEED+gi),'necessity_p_one_sided':p,'sufficiency_effect':float(np.mean(suf)),'sufficiency_ci':paired_bootstrap(suf,SEED+100+gi)})
 adj=holm(rawp)
 for x in outs:x['holm_p']=adj[x['group']];x['holm_survivor']=bool(x['holm_p']<.05 and x['necessity_effect']>0 and x['sufficiency_effect']>0)
 restore(rows);return {'schema':'R22568_C52_CAUSAL_SCREEN_V1','lane':'CAPABILITY' if direction==1 else 'FAILURE','locked_n':len(locked),'groups':outs,'survivors':[x['group'] for x in outs if x['holm_survivor']],'claim_ceiling':'E3_BROAD_LOCALIZATION_ONLY_NO_E4_WITHOUT_FINE_IRREDUNDANCY_AND_INDEPENDENT_REPLAY'}
def source_capture():
 for p in (ADAPTER_DIR,BASE_DIR):
  if p.exists():shutil.rmtree(p)
 snapshot_download(ADAPTER_REPO,revision=ADAPTER_REV,local_dir=ADAPTER_DIR,allow_patterns=['adapter_config.json','adapter_model.safetensors','README.md'],ignore_patterns=['*.bin','*.pt','*.pth','*.pkl','*.pickle','*.h5','*.msgpack','*.py','optimizer*','scheduler*','rng_state*','training_args*'])
 snapshot_download(BASE_REPO,revision=BASE_REV,local_dir=BASE_DIR,allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json'],ignore_patterns=['*.bin','*.pt','*.pth','*.pkl','*.pickle','*.h5','*.msgpack','*.py'])
 files=[]
 for role,root in [('adapter',ADAPTER_DIR),('base',BASE_DIR)]:
  for p in sorted(root.rglob('*')):
   if p.is_file() and '.cache' not in p.parts:files.append({'role':role,'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p)})
 return files
def load_runtime():
 tok=AutoTokenizer.from_pretrained(BASE_DIR,local_files_only=True,trust_remote_code=False,use_fast=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(BASE_DIR,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32).to(DEVICE);base.eval();model=PeftModel.from_pretrained(base,ADAPTER_DIR,local_files_only=True,is_trainable=False).to(DEVICE);model.eval();return model,tok
def main():
 started=time.time();terminal={'schema':'R22568_C52_EXECUTION_V1','candidate_id':CID,'status':'RUNNING','scientific_increment':0};stage='PRESEAL_CHAIN'
 try:
  cases=generate_suite();assert suite_digest(cases)==EXPECTED_SUITE_SHA and contract_digest()==EXPECTED_CONTRACT_SHA;terminal.update({'suite_n':384,'suite_sha256':EXPECTED_SUITE_SHA,'contract_sha256':EXPECTED_CONTRACT_SHA});stage='SOURCE_CAPTURE';files=source_capture();terminal['source_inventory']=files;terminal['source_bytes']=sum(x['bytes'] for x in files);terminal['adapter_weight_sha256']=sha256_file(ADAPTER_DIR/'adapter_model.safetensors');terminal['adapter_weight_bytes']=(ADAPTER_DIR/'adapter_model.safetensors').stat().st_size;terminal['base_weight_sha256']=sha256_file(BASE_DIR/'model.safetensors');terminal['base_weight_bytes']=(BASE_DIR/'model.safetensors').stat().st_size;stage='STATIC_FORENSICS';static=static_forensics();write_json('R22568_C52_STATIC.json',static)
  if static['zero_pair_count'] or static['nonfinite_pair_count']:terminal.update({'status':'TERMINAL_OPERATOR_VIABILITY_RED','stage':stage,'causal_not_run':'OPERATOR_VIABILITY_RED'});return
  stage='RUNTIME_LOAD';model,tok=load_runtime();rows=capture_state(model);assert len(rows)==static['complete_pair_count'];stage='TEACHER_BEHAVIOR';teacher={}
  with model.disable_adapter():teacher['BASE']=teacher_scores(model,tok,cases)
  restore(rows);teacher['FULL']=teacher_scores(model,tok,cases);restore(rows);set_random_sign(rows);teacher['RANDOM_SIGN']=teacher_scores(model,tok,cases);restore(rows);set_layer_shuffle(rows);teacher['LAYER_SHUFFLE']=teacher_scores(model,tok,cases);restore(rows)
  for d,l in ((.25,'DOSE_0.25'),(.5,'DOSE_0.5'),(1.5,'DOSE_1.5')):set_dose(rows,d);teacher[l]=teacher_scores(model,tok,cases);restore(rows)
  write_json('R22568_C52_TEACHER_BEHAVIOR.json',{'schema':'R22568_C52_TEACHER_V1','conditions':{k:{'aggregate':aggregate_teacher(v),'case_scores':v} for k,v in teacher.items()}});stage='LOCKED_GENERATION';gcases=generation_cases(cases);gen={}
  with model.disable_adapter():gen['BASE']=generate_score(model,tok,gcases)
  restore(rows);gen['FULL']=generate_score(model,tok,gcases);restore(rows);set_random_sign(rows);gen['RANDOM_SIGN']=generate_score(model,tok,gcases);restore(rows);set_layer_shuffle(rows);gen['LAYER_SHUFFLE']=generate_score(model,tok,gcases);restore(rows);write_json('R22568_C52_GENERATION_BEHAVIOR.json',{'schema':'R22568_C52_GENERATION_V1','locked_n':80,'conditions':{k:{'aggregate':aggregate_generation(v),'case_evidence':v} for k,v in gen.items()},'raw_generated_text_exported':False});stage='BEHAVIOR_GATE';gate=behavior_gate(teacher,gen);write_json('R22568_C52_BEHAVIOR_GATE.json',gate);stage='CAUSAL_SCREEN';causal=causal_screen(model,tok,rows,cases,gate) if gate['causal_entry'] else {'schema':'R22568_C52_CAUSAL_SCREEN_V1','status':'NOT_RUN_BEHAVIOR_GATE_RED','survivors':[]};write_json('R22568_C52_CAUSAL_SCREEN.json',causal);stage='RAWFREE_ASSETS';bg={x['id']:x for x in gen['BASE']};fg={x['id']:x for x in gen['FULL']};counter=[]
  for c in gcases:
   b,f=bg[c['id']],fg[c['id']]
   if f['field_exact']<b['field_exact'] or f['false_event'] or not f['json_valid']:counter.append({'id':c['id'],'split':c['split'],'input_sha256':hashlib.sha256(c['input'].encode()).hexdigest(),'base_field_exact':b['field_exact'],'full_field_exact':f['field_exact'],'full_json_valid':f['json_valid'],'full_false_event':f['false_event'],'full_output_sha256':f['output_sha256']})
  write_json('R22568_C52_COUNTEREXAMPLES.json',{'schema':'R22568_C52_COUNTEREXAMPLES_V1','count':len(counter),'rows':counter});write_json('R22568_C52_CAPABILITY_CARD.json',{'schema':'LUCIA_CAPABILITY_CARD_CANDIDATE_V1','candidate_id':CID,'scientific_question':'typed temporal calendar-event normalization vs JSON scaffold','gate':{k:gate[k] for k in ('positive_reproduction_green','failure_reproduction_green','causal_entry')},'claim_ceiling':'E3_BROAD_LOCALIZATION_ONLY','training_time_base_revision_proven':False,'promotion':False});write_json('R22568_C52_SKILLGENOME.json',{'schema':'LUCIA_SKILLGENOME_CANDIDATE_V1','candidate_id':CID,'trigger':'natural-language calendar-event extraction','verifier_contract':'8 typed fields; DD/MM/YYYY; 12-hour AM/PM; no-event negatives','status':'QUARANTINED_CANDIDATE','promotion':False});write_json('R22568_C52_PROGRAMDB_BOUNDARY.json',{'schema':'LUCIA_PROGRAMDB_CANDIDATE_BOUNDARY_V1','candidate_id':CID,'executable':False,'reason':'NO_CLEAN_ROOM_ALGORITHM_DERIVED_IN_C52'});write_json('R22568_C52_TRACE_NEURON.json',{'schema':'LUCIA_TRACE_NEURON_V1','candidate_id':CID,'static_projection_energy':static['projection_energy_fraction'],'static_layer_energy':static['layer_energy_fraction'],'causal_survivors':causal.get('survivors',[]),'raw_activation_included':False});write_json('R22568_C52_PROCESS_CARD.json',{'schema':'LUCIA_PROCESS_CARD_V1','candidate_id':CID,'pipeline':['source-zero 3-way audit','dependency closure','384-case dual-verifier preseal','one-use exact ingress','operator viability','behavior controls','conditional broad causal','raw-free compile','raw deletion'],'posthoc_threshold_change':False});terminal.update({'status':'CLOSED_ONE_USE_ANALYSIS_COMPLETE','stage':'COMPLETE','scientific_increment':1 if (gate['positive_reproduction_green'] or gate['failure_reproduction_green']) else 0,'evidence_class':'E2_CONTROL_SEPARATED_TASK_GAIN' if gate['positive_reproduction_green'] else ('E2_CONTROL_SEPARATED_FAILURE_DELTA' if gate['failure_reproduction_green'] else 'E1_STATIC_PLUS_BEHAVIOR_GATE_RED'),'positive_reproduction_green':gate['positive_reproduction_green'],'failure_reproduction_green':gate['failure_reproduction_green'],'causal_survivors':causal.get('survivors',[]),'claim_ceiling':'E3_BROAD_LOCALIZATION_ONLY','algorithm':0,'student':0,'promotion':0});del model,tok;gc.collect()
 except Exception as e:terminal.update({'status':'CLOSED_SOURCE_CONSUMED_TECHNICAL_HOLD_NO_REDOWNLOAD' if (ADAPTER_DIR/'adapter_model.safetensors').exists() else 'CLOSED_PREINGRESS_TECHNICAL_HOLD','stage':stage,'error_type':type(e).__name__,'error_message':str(e)[:2000],'scientific_increment':0,'algorithm':0,'student':0,'promotion':0});write_json('R22568_C52_TECHNICAL_HOLD.json',{'schema':'R22568_C52_TECHNICAL_HOLD_V1','stage':stage,'error_type':type(e).__name__,'error_message':str(e)[:2000],'traceback_sha256':hashlib.sha256(traceback.format_exc().encode()).hexdigest(),'raw_traceback_exported':False})
 finally:terminal['elapsed_seconds']=time.time()-started;write_json('R22568_C52_EXECUTION.json',terminal)
if __name__=='__main__':main()
