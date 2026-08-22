from __future__ import annotations
import collections,hashlib,json,os,random,re,shutil,sys,time,traceback,urllib.request
from pathlib import Path
R=Path('work');D=R/'download';E=R/'escrow';O=R/'out'
for p in(D,E,O):p.mkdir(parents=True,exist_ok=True)
AR='rflores3113/qwen2.5-1.5b-email-extraction-lora';AV='d2643b11367a7d1511e93723033b048e41b30af1';AS=73911112;AH='d357caa5a0310614eb53c32310ba832eaca3551d272b8a9bf33200bf96de546f'
BR='Qwen/Qwen2.5-1.5B-Instruct';BV='989aa7980e4cf806f80c7fef2b1adb7bc71aa306';UA={'User-Agent':'LUCIA-AA-R22591-C72'}
Q='CUSTOMER_SUPPORT_EMAIL_TO_SIX_FIELD_TYPED_JSON_WITH_CLOSED_ENUMS_AND_LIST_EXTRACTION';K=('customer_name','issue_category','urgency','primary_request','mentioned_products','sentiment')
CAT=('billing','technical','refund','account','general');URG=('low','medium','high');REQ=('refund','troubleshoot','reset_password','cancel_subscription','replacement','invoice_explanation');SEN=('positive','negative','neutral')
SYS='Extract the customer-support email into exactly one minified JSON object. Use exactly these keys: customer_name, issue_category, urgency, primary_request, mentioned_products, sentiment. issue_category must be one of billing|technical|refund|account|general; urgency low|medium|high; primary_request refund|troubleshoot|reset_password|cancel_subscription|replacement|invoice_explanation; sentiment positive|negative|neutral. mentioned_products must list product names in first-appearance order. No markdown and no extra keys.'
RM={'refund':('refund','negative','Please refund the duplicate charge.'),'troubleshoot':('technical','negative','Please help me troubleshoot the connection problem.'),'reset_password':('account','neutral','Please reset my account password.'),'cancel_subscription':('general','neutral','Please cancel my subscription.'),'replacement':('technical','negative','Please send a replacement unit.'),'invoice_explanation':('billing','neutral','Please explain the invoice charge.')}
US={'high':'This is urgent and I need help today.','medium':'Please handle this soon, but it is not an emergency.','low':'There is no rush; whenever convenient is fine.'};SS={'negative':'I am frustrated with the situation.','neutral':'I am simply trying to get this sorted out.','positive':'I appreciate your help and have otherwise been happy with the service.'}
N=['Alice Mercer','Bruno Park','Carla Imani','Diego Santos','Elena Novak','Farah Khan','Gavin Reed','Hana Mori','Isaac Bell','Juno Clarke','Kaito Sato','Lina Moreau','Marek Ziel','Nadia Costa','Owen Price','Priya Nair'];P=['Atlas Router','Nova Hub','Pulse Mini','Orbit Dock','Vertex Cam','Lumen Key','Echo Pad','Aero Bridge','Cobalt Node','Mira Sensor']
def jw(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def api(u):
 for i in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60) as r:return json.loads(r.read().decode())
  except Exception:
   if i==2:raise
   time.sleep(i+1)
def text(u):
 with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60) as r:return r.read(5000001).decode('utf-8','replace')
def sib(i,n):
 for x in i.get('siblings') or []:
  if x.get('rfilename')==n:
   l=x.get('lfs') or {};return {'size':l.get('size') or x.get('size'),'sha256':l.get('sha256') or l.get('oid'),'blob_id':x.get('blobId')}
 return {}
def dl(repo,rev,name,dst,t=900):
 h=hashlib.sha256();z=0;u=f'https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true'
 with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=t) as r,dst.open('wb') as f:
  while 1:
   b=r.read(1<<20)
   if not b:break
   f.write(b);h.update(b);z+=len(b)
 return {'name':name,'size':z,'sha256':h.hexdigest()}
def render(e,i):
 ps=e['mentioned_products'];ph=ps[0] if len(ps)==1 else f'{ps[0]} and {ps[1]}';name=e['customer_name'];q=e['primary_request']
 op=[f'Hi support, my name is {name}.',f'Hello — {name} here.',f'Support team, this is {name} writing about my order.',f'I am {name}; I need assistance with {ph}.'][i%4]
 body={'refund':f'I noticed the same charge twice after buying {ph}.','troubleshoot':f'My {ph} keeps dropping its connection even after a restart.','reset_password':f'I cannot sign in to manage {ph} because my password no longer works.','cancel_subscription':f'I no longer need the subscription linked to {ph}.','replacement':f'The {ph} arrived defective and will not power on reliably.','invoice_explanation':f'My invoice for {ph} contains a charge I do not recognize.'}[q]
 a=[op,body,US[e['urgency']],SS[e['sentiment']],RM[q][2]]
 if i%4==1:a=[op,US[e['urgency']],body,RM[q][2],SS[e['sentiment']]]
 if i%4==2:a=[f'Subject: help with {ps[0]}',op,body,SS[e['sentiment']],US[e['urgency']],RM[q][2]]
 if i%4==3:a=[op,body,f'For reference, the products are {ph}.',RM[q][2],US[e['urgency']],SS[e['sentiment']]]
 return '\n'.join(a)
def cases(a,b):
 out=[]
 for i in range(a,b):
  q=REQ[i%6];cat,sent,_=RM[q];sent='positive' if q in {'reset_password','invoice_explanation','cancel_subscription'} and i%5==0 else sent;u=URG[(i//2)%3];name=N[(i*5+3)%len(N)];p1=P[(i*3+1)%10];p2=P[(i*7+4)%10];ps=[p1] if i%3 else ([p1,p2] if p2!=p1 else [p1,P[(i*7+5)%10]])
  e={'customer_name':name,'issue_category':cat,'urgency':u,'primary_request':q,'mentioned_products':ps,'sentiment':sent};out.append({'id':f'em{i:03d}','email':render(e,i),'expected':e})
 return out
def ch(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def canon(c):return json.dumps({k:c['expected'][k] for k in K},ensure_ascii=False,separators=(',',':'))
def ver(t,c):
 t=t.strip()
 if not t or '```' in t:return {'pass':False,'reason':'empty_or_fence','field_pass':{}}
 try:o,e=json.JSONDecoder().raw_decode(t)
 except Exception:return {'pass':False,'reason':'json_parse','field_pass':{}}
 if t[e:].strip() or not isinstance(o,dict) or tuple(o.keys())!=K:return {'pass':False,'reason':'envelope_or_keys','field_pass':{}}
 x=c['expected'];f={'customer_name':o.get('customer_name')==x['customer_name'],'issue_category':o.get('issue_category') in CAT and o.get('issue_category')==x['issue_category'],'urgency':o.get('urgency') in URG and o.get('urgency')==x['urgency'],'primary_request':o.get('primary_request') in REQ and o.get('primary_request')==x['primary_request'],'mentioned_products':isinstance(o.get('mentioned_products'),list) and o.get('mentioned_products')==x['mentioned_products'],'sentiment':o.get('sentiment') in SEN and o.get('sentiment')==x['sentiment']};return {'pass':all(f.values()),'reason':'PASS' if all(f.values()) else 'field_mismatch','field_pass':f}
def preflight():
 import tempfile,torch,psutil,shutil
 from safetensors.torch import save_file
 from safetensors import safe_open
 from transformers import Qwen2Config,Qwen2ForCausalLM
 from peft import LoraConfig,get_peft_model
 assert shutil.disk_usage('.').free>8_000_000_000 and psutil.virtual_memory().available>8_000_000_000
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'x.safetensors';x=torch.tensor([[1.5,-2.25]],dtype=torch.bfloat16);save_file({'x':x},str(p))
  with safe_open(str(p),framework='pt',device='cpu') as f:y=f.get_tensor('x')
  assert y.dtype==torch.bfloat16 and torch.equal(x.float(),y.float())
 c=Qwen2Config(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128);m=Qwen2ForCausalLM(c).to(dtype=torch.bfloat16);m=get_peft_model(m,LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])).eval()
 with torch.inference_mode():y=m(input_ids=torch.randint(0,256,(1,8))).logits
 assert y.shape==(1,8,256) and torch.isfinite(y.float()).all();print(json.dumps({'PRE_SOURCE_PASS':True,'torch':torch.__version__,'bf16_backend':'safetensors framework=pt','numpy_backend':False}))
def static(p,cfg):
 import numpy as np,torch
 from safetensors import safe_open
 A={};B={};other=[];dc=collections.Counter()
 with safe_open(str(p),framework='pt',device='cpu') as f:
  for n in f.keys():
   s=f.get_slice(n);dc[str(s.get_dtype())]+=1
   if '.lora_A.' in n:A[n.split('.lora_A')[0]]=f.get_tensor(n).float()
   elif '.lora_B.' in n:B[n.split('.lora_B')[0]]=f.get_tensor(n).float()
   else:other.append({'name':n,'shape':list(s.get_shape()),'dtype':str(s.get_dtype())})
 ks=sorted(set(A)|set(B));assert all(k in A and k in B for k in ks);scale=cfg['lora_alpha']/cfg['r'];bm={};bl={};rows=[];tot=0.;zero=0;finite=True
 for k in ks:
  a,b=A[k],B[k];z=not(torch.count_nonzero(a) and torch.count_nonzero(b));zero+=z;finite&=bool(torch.isfinite(a).all() and torch.isfinite(b).all());aa=a.numpy().astype('float64');bb=b.numpy().astype('float64');er=min(int(np.linalg.matrix_rank(aa)),int(np.linalg.matrix_rank(bb)));e=float((scale*np.linalg.norm(aa)*np.linalg.norm(bb))**2);tot+=e;mod=k.split('.')[-1];lm=re.search(r'\.layers\.(\d+)\.',k);ly=int(lm.group(1)) if lm else -1;m=bm.setdefault(mod,{'pairs':0,'e':0.,'zero_pairs':0,'rank_min':999,'rank_max':0});m['pairs']+=1;m['e']+=e;m['zero_pairs']+=int(z);m['rank_min']=min(m['rank_min'],er);m['rank_max']=max(m['rank_max'],er);l=bl.setdefault(str(ly),{'pairs':0,'e':0.,'modules':set()});l['pairs']+=1;l['e']+=e;l['modules'].add(mod);rows.append({'component_id':k,'layer':ly,'module':mod,'a_shape':list(a.shape),'b_shape':list(b.shape),'rank_upper':er,'zero':bool(z)})
 for v in bm.values():v['energy_ratio']=v.pop('e')/tot if tot else 0
 for v in bl.values():v['energy_ratio']=v.pop('e')/tot if tot else 0;v['modules']=sorted(v['modules'])
 return {'schema':'R22591_C72_ADAPTER_OPERATOR_ARCHAEOLOGY_V1','operator':{'tensor_count':len(A)+len(B)+len(other),'complete_pairs':len(ks),'alive_pairs':len(ks)-zero,'zero_pairs':zero,'all_finite':finite,'auxiliary_tensor_count':len(other),'auxiliary_tensors':other,'dtype_counts':dict(dc),'target_modules_actual':sorted(bm),'by_module':dict(sorted(bm.items())),'by_layer':dict(sorted(bl.items(),key=lambda x:int(x[0]))),'pair_inventory':rows,'energy_proxy_definition':'(alpha/r*||A||F*||B||F)^2; static ranking only, noncausal','safetensors_backend':'torch/PT'}}
def main():
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from peft import PeftModel
 torch.set_num_threads(min(4,os.cpu_count() or 2));torch.manual_seed(22591);t0=time.time();primary=cases(0,12);assert ch(primary)=='0d8c6755ef379c4f819b1bbdb68f49304a314b2dd16689c92a877a7fb9e08b2d';access={'primary_instantiated':True,'holdout_instantiated':False,'ood_instantiated':False};jw('R22591_C72_SPLIT_ACCESS_RECEIPT.json',access)
 ai=api(f'https://huggingface.co/api/models/{AR}?blobs=true');am=sib(ai,'adapter_model.safetensors');cfg=json.loads(text(f'https://huggingface.co/{AR}/raw/{AV}/adapter_config.json'));readme=text(f'https://huggingface.co/{AR}/raw/{AV}/README.md');bi=api(f'https://huggingface.co/api/models/{BR}/revision/{BV}?blobs=true')
 if ai.get('sha')!=AV or ai.get('cardData',{}).get('license')!='apache-2.0' or int(am.get('size') or -1)!=AS or str(am.get('sha256')).replace('sha256:','')!=AH:raise RuntimeError('ADAPTER_PIN_FAIL')
 if BR not in readme or BV not in readme or bi.get('sha')!=BV or cfg.get('base_model_name_or_path')!=BR or cfg.get('peft_type')!='LORA' or cfg.get('r')!=16 or cfg.get('lora_alpha')!=32 or cfg.get('modules_to_save') not in (None,[]):raise RuntimeError('PROVENANCE_OR_CONFIG_FAIL')
 jw('R22591_C72_ATOMIC_SOURCE_PIN.json',{'schema':'R22591_C72_ATOMIC_SOURCE_PIN_V1','question':Q,'adapter_repo':AR,'adapter_revision':AV,'adapter_lfs':am,'license':'apache-2.0','base_repo':BR,'training_base_revision':BV,'training_base_revision_source':'immutable Adapter README + exact Base API','weight_gets_before_pin':0,'model_weight_bytes_before_pin':0,'publisher_metrics_credit':0})
 ap=D/'adapter_model.safetensors';ar=dl(AR,AV,'adapter_model.safetensors',ap);assert ar['size']==AS and ar['sha256']==AH;jw('R22591_C72_SOURCE_INGRESS_RECEIPT.json',{'schema':'R22591_C72_SOURCE_INGRESS_RECEIPT_V1','source_consumed':True,'adapter_weight_get_count':1,'adapter':{'repo':AR,'revision':AV,**ar},'base_weight_get_count':0,'base_bytes':0,'written_before_deep_static':True});ad=E/'adapter';ad.mkdir();os.replace(ap,ad/'adapter_model.safetensors');(ad/'adapter_config.json').write_text(json.dumps(cfg));st=static(ad/'adapter_model.safetensors',cfg);jw('R22591_C72_ADAPTER_OPERATOR_ARCHAEOLOGY.json',st);op=st['operator']
 if op['zero_pairs'] or not op['all_finite'] or op['auxiliary_tensor_count']:raise RuntimeError('OPERATOR_VIABILITY_FAIL')
 names=[x.get('rfilename') for x in bi.get('siblings') or [] if x.get('rfilename')];allow={'config.json','generation_config.json','model.safetensors','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json','chat_template.jinja'};need=sorted({x for x in names if x in allow or re.fullmatch(r'model-\d+-of-\d+\.safetensors',x or '')});bd=E/'base';bd.mkdir();mf=[];bb=0;wg=0
 for n in need:
  r=dl(BR,BV,n,bd/n);mf.append(r);bb+=r['size'];wg+=int(n.endswith('.safetensors'));jw('R22591_C72_BASE_INGRESS_RECEIPT.json',{'schema':'R22591_C72_BASE_INGRESS_RECEIPT_V1','base_source_consumed':True,'repo':BR,'revision':BV,'files':mf,'base_bytes':bb,'base_weight_get_count':wg,'complete':False})
 jw('R22591_C72_BASE_INGRESS_RECEIPT.json',{'schema':'R22591_C72_BASE_INGRESS_RECEIPT_V1','base_source_consumed':True,'repo':BR,'revision':BV,'files':mf,'base_bytes':bb,'base_weight_get_count':wg,'complete':True});x=json.loads((O/'R22591_C72_SOURCE_INGRESS_RECEIPT.json').read_text());x.update(base_weight_get_count=wg,base_bytes=bb);jw('R22591_C72_SOURCE_INGRESS_RECEIPT.json',x)
 tok=AutoTokenizer.from_pretrained(str(bd),local_files_only=True,use_fast=True);base=AutoModelForCausalLM.from_pretrained(str(bd),local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager').eval();m=PeftModel.from_pretrained(base,str(ad),is_trainable=False).eval()
 def msg(c):return [{'role':'system','content':SYS},{'role':'user','content':c['email']}]
 def gen(c,maxn=112):
  ids=tok.apply_chat_template(msg(c),tokenize=True,add_generation_prompt=True,return_tensors='pt')
  with torch.inference_mode():o=m.generate(ids,max_new_tokens=maxn,do_sample=False,pad_token_id=tok.eos_token_id)
  s=tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True);v=ver(s,c);return {'id':c['id'],'text':s,**v}
 def ev(cs):return [gen(c) for c in cs]
 def agg(z):return {'strict_pass':sum(i['pass'] for i in z),'total':len(z),'field_pass_counts':{k:sum(bool(i.get('field_pass',{}).get(k)) for i in z) for k in K},'outcomes':z}
 def nll(c):
  p=tok.apply_chat_template(msg(c),tokenize=True,add_generation_prompt=True,return_tensors='pt');q=tok(canon(c),add_special_tokens=False,return_tensors='pt').input_ids;ids=torch.cat([p,q],1);lab=torch.full_like(ids,-100);lab[:,p.shape[1]:]=q
  with torch.inference_mode():return float(m(input_ids=ids,labels=lab).loss)
 B={'schema':'R22591_C72_FRESH_EMAIL_BEHAVIOR_V1','primary':{},'controls':{},'holdout':{'status':'LOCKED_UNINSTANTIATED'},'ood':{'status':'LOCKED_UNINSTANTIATED'},'causal':{'status':'NOT_RUN'},'publisher_metrics_credit':0}
 with m.disable_adapter():bp=ev(primary);bn=[nll(c) for c in primary[:4]]
 fp=ev(primary);fn=[nll(c) for c in primary[:4]];B['primary']['BASE']={**agg(bp),'target_nll_first4':bn};B['primary']['FULL']={**agg(fp),'target_nll_first4':fn};res=[i for i,(a,b) in enumerate(zip(bp,fp)) if not a['pass'] and b['pass']];harm=[i for i,(a,b) in enumerate(zip(bp,fp)) if a['pass'] and not b['pass']];gain=len(res)>len(harm) and bool(res);B['primary']['paired']={'rescue_indices':res,'harm_indices':harm,'primary_gain':gain}
 np={n:p for n,p in m.named_parameters() if 'lora_A' in n or 'lora_B' in n};orig={n:p.detach().cpu().clone() for n,p in np.items()}
 def restore():
  with torch.no_grad():
   for n,p in np.items():p.copy_(orig[n].to(dtype=p.dtype))
 def rnd(seed):
  restore();g=torch.Generator().manual_seed(seed)
  with torch.no_grad():
   for n,p in np.items():
    a=orig[n].float();r=torch.randn(a.shape,generator=g);r*=float(a.norm())/max(float(r.norm()),1e-12);p.copy_(r.to(dtype=p.dtype))
 def shuf():
  restore();gs={}
  for n,p in np.items():
   if 'lora_B' in n:gs.setdefault(tuple(p.shape),[]).append(n)
  with torch.no_grad():
   for a in gs.values():
    a.sort();src=[orig[n] for n in a]
    for i,n in enumerate(a):np[n].copy_(src[(i+1)%len(src)].to(dtype=np[n].dtype))
 def dose(v):
  restore()
  with torch.no_grad():
   for n,p in np.items():
    if 'lora_B' in n:p.copy_((orig[n].float()*v).to(dtype=p.dtype))
 for c in ['random1','random2','shuffle','dose0.5','dose1.5']:
  {'random1':lambda:rnd(225911),'random2':lambda:rnd(225912),'shuffle':shuf,'dose0.5':lambda:dose(.5),'dose1.5':lambda:dose(1.5)}[c]();z={'target_nll_first4':[nll(x) for x in primary[:4]],'generation':'NOT_RUN_NO_RESCUE'}
  if res:z['generation']={**agg(ev([primary[i] for i in res[:3]])),'indices':res[:3]}
  B['controls'][c]=z
 restore();sep=False
 if res:
  z=min(3,len(res));sep=all(B['controls'][c]['generation']['strict_pass']<z for c in ['random1','random2','shuffle'])
 B['primary']['paired']['random_random_shuffle_separated']=sep;hg=False
 if gain and sep:
  H=cases(12,20);assert ch(H)=='88e329f2f51652d9d9395520f69ba1fe746af527561123d236c93de3bdbb6fe3';access['holdout_instantiated']=True;jw('R22591_C72_SPLIT_ACCESS_RECEIPT.json',access)
  with m.disable_adapter():hb=ev(H)
  hf=ev(H);hr=[i for i,(a,b) in enumerate(zip(hb,hf)) if not a['pass'] and b['pass']];hh=[i for i,(a,b) in enumerate(zip(hb,hf)) if a['pass'] and not b['pass']];hg=len(hr)>len(hh) and bool(hr);B['holdout']={'status':'INSTANTIATED_AFTER_PRIMARY_CONTROL_GATE','BASE':agg(hb),'FULL':agg(hf),'rescue_indices':hr,'harm_indices':hh,'same_environment_replicated_gain':hg}
 if gain and sep and hg:
  Z=cases(20,28);assert ch(Z)=='07a02168a3c84c01dd9baff253443252763d81e68977039cb7e8581c280581ff';access['ood_instantiated']=True;jw('R22591_C72_SPLIT_ACCESS_RECEIPT.json',access)
  with m.disable_adapter():ob=ev(Z)
  of=ev(Z);B['ood']={'status':'INSTANTIATED_AFTER_HOLDOUT_GATE','BASE':agg(ob),'FULL':agg(of)}
  probe=primary[res[0]];C={'probe_id':probe['id'],'module_ablation':{},'module_sufficiency':{}}
  def zero(pred):
   restore()
   with torch.no_grad():
    for n,p in np.items():
     if 'lora_B' in n and pred(n):p.zero_()
  for mod in op['target_modules_actual']:zero(lambda n,m0=mod:f'.{m0}.' in n);C['module_ablation'][mod]=gen(probe)
  for mod in op['target_modules_actual']:
   restore()
   with torch.no_grad():
    for n,p in np.items():
     if 'lora_B' in n and f'.{mod}.' not in n:p.zero_()
   C['module_sufficiency'][mod]=gen(probe)
  restore();B['causal']={'status':'E3_BOUNDED_SAME_ENVIRONMENT_ONLY','results':C,'not_e4_or_e5':True}
 else:
  if B['holdout']['status']=='LOCKED_UNINSTANTIATED':B['holdout']['status']='LOCKED_UNINSTANTIATED_PRIMARY_OR_CONTROL_GATE_FAIL'
  B['ood']={'status':'LOCKED_UNINSTANTIATED_HOLDOUT_GATE_FAIL'};B['causal']={'status':'NOT_RUN_GATE_FAIL'}
 B['claim_boundary']={'fresh_behavior_executed':True,'fresh_primary_gain':gain,'random_random_shuffle_separated':sep,'same_environment_holdout_gain':hg,'e3_increment':1 if B['causal']['status'].startswith('E3') else 0,'e4_plus_increment':0,'e5_increment':0};jw('R22591_C72_FRESH_EMAIL_BEHAVIOR.json',B);jw('R22591_C72_SPLIT_ACCESS_RECEIPT.json',access);return {'status':'PASS','operator_pairs':op['complete_pairs'],'alive_pairs':op['alive_pairs'],'base_bytes':bb,'behavior':B['claim_boundary'],'elapsed_seconds':time.time()-t0}
if '--preflight' in sys.argv:preflight();raise SystemExit
result=err=None
try:result=main()
except Exception as e:err={'type':type(e).__name__,'message':str(e),'traceback_tail':'\n'.join(traceback.format_exc().splitlines()[-14:])}
finally:
 for p in(D,E):shutil.rmtree(p,ignore_errors=True)
 raw=[str(p) for p in R.rglob('*') if p.is_file() and (p.suffix.lower() in {'.safetensors','.bin','.pt','.pth','.pkl','.pickle'} or p.name in {'tokenizer.json','vocab.json','merges.txt'})];sp=O/'R22591_C72_SOURCE_INGRESS_RECEIPT.json';bp=O/'R22591_C72_BASE_INGRESS_RECEIPT.json';sc=bool(sp.exists() and json.loads(sp.read_text()).get('source_consumed'));bc=bool(bp.exists() and json.loads(bp.read_text()).get('base_source_consumed'));jw('R22591_C72_FINAL_DELETION_RECEIPT.json',{'schema':'R22591_C72_FINAL_DELETION_RECEIPT_V1','result':result,'error':err,'raw_remaining':raw,'raw_remaining_count':len(raw),'post_delete_pass':not raw,'source_consumed':sc,'base_source_consumed':bc,'base_behavior_executed':(O/'R22591_C72_FRESH_EMAIL_BEHAVIOR.json').exists()})
if err or not result or raw:print(json.dumps({'PASS':False,'error':err,'deletion':not raw,'source_consumed':sc,'base_source_consumed':bc},indent=2));raise SystemExit(1)
print(json.dumps({'PASS':True,'result':result,'deletion':True},indent=2))
