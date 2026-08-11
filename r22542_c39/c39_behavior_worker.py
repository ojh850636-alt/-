from __future__ import annotations
import contextlib, hashlib, json, os, random, re, resource, time, gc
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from peft import PeftModel
from lean_tactic_forge import build

EXPECTED_SUITE_SHA='a081ee299a7035337bb6fa6f6c380fb6d8cbd328534b049cd45a5fbe032338f2'
CID='R22542-C39-OPENPROOF-LEAN-TACTIC-2B'
FORBIDDEN=('sorry','admit','axiom','unsafe','run_tac','set_option','#eval','#check','#print')

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def dump(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def mean(xs):
    xs=list(xs); return float(sum(xs)/len(xs)) if xs else 0.0
def bootstrap_ci(vals,seed=22542,reps=4000):
    vals=list(vals)
    if not vals:return [0.0,0.0]
    rng=random.Random(seed);n=len(vals);s=[]
    for _ in range(reps):s.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
    s.sort();return [float(s[int(.025*reps)]),float(s[min(reps-1,int(.975*reps))])]
def signflip_p(vals,seed=22542,reps=4096):
    vals=list(vals)
    if not vals:return 1.0
    obs=abs(mean(vals));rng=random.Random(seed);ge=0
    for _ in range(reps):
        x=abs(sum(v*(1 if rng.getrandbits(1) else -1) for v in vals)/len(vals));ge+=x>=obs-1e-12
    return float((ge+1)/(reps+1))

class LoraController:
    def __init__(self,model):
        self.modules=[]
        for name,mod in model.named_modules():
            if not (hasattr(mod,'lora_A') and hasattr(mod,'lora_B') and hasattr(mod,'scaling')):continue
            if not getattr(mod,'lora_A',None):continue
            key=next(iter(mod.lora_A.keys()));m=re.search(r'layers\.(\d+)\.',name);layer=int(m.group(1)) if m else -1
            proj=next((p for p in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj') if name.endswith(p)),'OTHER')
            self.modules.append({'name':name,'module':mod,'key':key,'base_scale':float(mod.scaling[key]),'layer':layer,'projection':proj,'path':'ATTENTION' if proj in {'q_proj','k_proj','v_proj','o_proj'} else 'MLP'})
    def reset(self):
        for x in self.modules:x['module'].scaling[x['key']]=x['base_scale']
    def set_all(self,mult):
        for x in self.modules:x['module'].scaling[x['key']]=x['base_scale']*float(mult)
    def snapshot_B(self):return [(x,x['module'].lora_B[x['key']].weight.detach().cpu().clone()) for x in self.modules]
    def restore_B(self,snap):
        for x,w in snap:
            d=x['module'].lora_B[x['key']].weight;d.data.copy_(w.to(d.device,dtype=d.dtype))
    def randomize_B(self,seed):
        g=torch.Generator(device='cpu').manual_seed(seed)
        for x in self.modules:
            w=x['module'].lora_B[x['key']].weight;old=w.detach().float().cpu();rnd=torch.randn(old.shape,generator=g);rnd*=old.norm()/(rnd.norm()+1e-12);w.data.copy_(rnd.to(w.device,dtype=w.dtype))
    def shuffle_B(self,seed):
        g=torch.Generator(device='cpu').manual_seed(seed)
        for x in self.modules:
            w=x['module'].lora_B[x['key']].weight;old=w.detach().flatten().cpu();perm=torch.randperm(old.numel(),generator=g);w.data.copy_(old[perm].reshape(w.shape).to(w.device,dtype=w.dtype))

def sanitize(text):
    s=text.replace('\r','\n').strip()
    if ':::' in s:s=s.split(':::',1)[0].strip()
    if '\n' in s:s=s.split('\n',1)[0].strip()
    low=s.lower()
    if not s:return '','EMPTY'
    if any(t in low for t in FORBIDDEN):return '','FORBIDDEN'
    if len(s)>600:return '','TOO_LONG'
    return s,'OK'
def suite_digest(rows):return sha_bytes(''.join(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n' for r in rows).encode())

def score_rows(model,tok,rows,disable_adapter=False,label=''):
    model.eval();out=[];ctx=model.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with ctx,torch.inference_mode():
        for i,r in enumerate(rows):
            p=tok(r['prompt'],add_special_tokens=False).input_ids;c=tok(r['canonical_tactic'],add_special_tokens=False).input_ids
            ids=torch.tensor([p+c],dtype=torch.long,device=model.device);mask=torch.ones_like(ids)
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=len(c)+1).logits.float();lp=torch.log_softmax(logits[:,:-1,:],dim=-1);tgt=ids[0,-len(c):];vals=lp[0].gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
            out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'token_count':len(c),'mean_logprob':float(vals.mean().cpu()),'sum_logprob':float(vals.sum().cpu())})
            del ids,mask,logits,lp,tgt,vals
            if (i+1)%32==0:print('TF_PROGRESS',label,i+1,'/',len(rows),'rss_kb',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    return out

def summarize(scores):
    by={}
    for x in scores:by.setdefault(x['split'],[]).append(x)
    def s(v):return {'n':len(v),'mean_logprob':mean(x['mean_logprob'] for x in v),'mean_sum_logprob':mean(x['sum_logprob'] for x in v)}
    return {'overall':s(scores),'by_split':{k:s(v) for k,v in sorted(by.items())}}
def paired(a,b,key='mean_logprob'):
    bm={x['case_id']:x for x in b};return [float(x[key]-bm[x['case_id']][key]) for x in a if x['case_id'] in bm]
def teacher_report(res):
    summ={k:summarize(v) for k,v in res.items()}
    def cmp(a,b,seed):
        vals=paired(res[a],res[b]);return {'n':len(vals),'mean_delta':mean(vals),'bootstrap95':bootstrap_ci(vals,seed),'signflip_p':signflip_p(vals,seed)}
    comps={'FULL_MINUS_BASE':cmp('FULL','BASE',22542),'FULL_MINUS_RANDOM':cmp('FULL','RANDOM_NORM_MATCHED_LORA',22543),'FULL_MINUS_SHUFFLED':cmp('FULL','SHUFFLED_LORA',22544)}
    bm={x['case_id']:x for x in res['BASE']};fm={x['case_id']:x for x in res['FULL']};splits={};pos=[r for r in build() if r['positive']]
    for j,sp in enumerate(sorted({r['split'] for r in pos})):
        vals=[fm[r['case_id']]['mean_logprob']-bm[r['case_id']]['mean_logprob'] for r in pos if r['split']==sp];splits[sp]={'n':len(vals),'mean_delta':mean(vals),'bootstrap95':bootstrap_ci(vals,22600+j)}
    return {'conditions':summ,'comparisons':comps,'full_minus_base_by_split':splits}

def generate_condition(model,tok,rows,condition,seed,disable_adapter=False,k=8):
    tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token;model.eval();case_rows=[];ctx=model.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with ctx,torch.inference_mode():
        for i,r in enumerate(rows):
            enc=tok(r['prompt'],return_tensors='pt',truncation=True,max_length=768,add_special_tokens=False);enc={kk:vv.to(model.device) for kk,vv in enc.items()};g=torch.Generator(device=model.device if str(model.device)!='cpu' else 'cpu').manual_seed(seed*100000+i)
            gen=model.generate(**enc,max_new_tokens=48,do_sample=True,temperature=.8,top_p=.95,top_k=50,num_return_sequences=k,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,generator=g);plen=enc['input_ids'].shape[1];samples=[]
            for q in range(k):
                raw=tok.decode(gen[q,plen:],skip_special_tokens=True);tac,status=sanitize(raw);exact=bool(r['positive'] and status=='OK' and tac==r['canonical_tactic']);samples.append({'rank':q,'status':status,'tactic_sha256':sha_bytes(tac.encode()) if tac else None,'exact_canonical_match':exact,'kernel_verified_success_conservative':exact})
            case_rows.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'positive':r['positive'],'success_at_k':any(x['kernel_verified_success_conservative'] for x in samples),'forbidden_count':sum(x['status']=='FORBIDDEN' for x in samples),'empty_count':sum(x['status']=='EMPTY' for x in samples),'noncanonical_unverified_count':sum(x['status']=='OK' and not x['exact_canonical_match'] for x in samples),'sample_digests':samples})
            if (i+1)%12==0:print('GEN_PROGRESS',condition,seed,i+1,'/',len(rows),flush=True)
    return case_rows
def gen_summary(rows):
    by={}
    for r in rows:by.setdefault(r['split'],[]).append(r)
    def s(v):return {'n':len(v),'success_at_k_rate':mean(1.0 if x['success_at_k'] else 0.0 for x in v),'forbidden_sample_count':sum(x['forbidden_count'] for x in v),'noncanonical_unverified_sample_count':sum(x['noncanonical_unverified_count'] for x in v)}
    return {'overall':s(rows),'by_split':{k:s(v) for k,v in sorted(by.items())}}
def success_map(rows):return {r['case_id']:1.0 if r['success_at_k'] else 0.0 for r in rows}

def main():
    root=Path(os.environ.get('WORK_ROOT','.')).resolve();out=root/'result';out.mkdir(parents=True,exist_ok=True);source=root/'source';start=time.time()
    rows=build();assert suite_digest(rows)==EXPECTED_SUITE_SHA;pos=[r for r in rows if r['positive']];locked=[]
    for sp in ['CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE']:locked.extend(sorted([r for r in rows if r['split']==sp],key=lambda x:x['case_id'])[:12])
    base_dir=source/'base';cand_dir=source/'candidate';tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False);base=Qwen3_5ForCausalLM.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True);model=PeftModel.from_pretrained(base,cand_dir,local_files_only=True,is_trainable=False);ctl=LoraController(model);snap=ctl.snapshot_B();print('LOAD_DONE',len(ctl.modules),'rss_kb',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    def restore():ctl.restore_B(snap);ctl.reset();gc.collect()
    res={};restore();res['BASE']=score_rows(model,tok,pos,True,'BASE');restore();res['FULL']=score_rows(model,tok,pos,False,'FULL');restore();ctl.randomize_B(22542);res['RANDOM_NORM_MATCHED_LORA']=score_rows(model,tok,pos,False,'RANDOM');restore();ctl.shuffle_B(22542);res['SHUFFLED_LORA']=score_rows(model,tok,pos,False,'SHUFFLED');res['DOSE_0.00']=res['BASE'];res['DOSE_1.00']=res['FULL']
    for d in [0.25,0.5,1.5]:restore();ctl.set_all(d);res[f'DOSE_{d:.2f}']=score_rows(model,tok,pos,False,f'DOSE_{d:.2f}')
    tr=teacher_report(res);dump(out/'C39_TEACHER_FORCED_BEHAVIOR.json',{'schema':'LUCIA_AA_R22542_C39_TEACHER_FORCED_BEHAVIOR_V2','candidate_id':CID,'suite_sha256':EXPECTED_SUITE_SHA,'positive_cases':len(pos),'scorer':'COMPACT_LOGITS_TO_KEEP_NUMERIC_PARITY_PROVEN','duplicate_conditions_reused':{'DOSE_0.00':'BASE','DOSE_1.00':'FULL'},'raw_logits_saved':False,'raw_text_saved':False,**tr})
    gen={};restore();gen['BASE_seed42']=generate_condition(model,tok,locked,'BASE',42,True);restore();gen['FULL_seed42']=generate_condition(model,tok,locked,'FULL',42,False);restore();ctl.randomize_B(22542);gen['RANDOM_seed42']=generate_condition(model,tok,locked,'RANDOM',42,False);restore();ctl.shuffle_B(22542);gen['SHUFFLED_seed42']=generate_condition(model,tok,locked,'SHUFFLED',42,False)
    for seed in [224,1337]:restore();gen[f'FULL_seed{seed}']=generate_condition(model,tok,locked,'FULL',seed,False)
    gs={k:gen_summary(v) for k,v in gen.items()};fm=success_map(gen['FULL_seed42']);bm=success_map(gen['BASE_seed42']);rm=success_map(gen['RANDOM_seed42']);sm=success_map(gen['SHUFFLED_seed42'])
    def splitrate(mp,sp):
        ids=[r['case_id'] for r in locked if r['split']==sp];return mean(mp[i] for i in ids)
    confirmation_delta=splitrate(fm,'CONFIRMATION')-splitrate(bm,'CONFIRMATION');hold_delta=splitrate(fm,'GENERATOR_HOLDOUT')-splitrate(bm,'GENERATOR_HOLDOUT');full_all=mean(fm.values());bestctrl=max(mean(rm.values()),mean(sm.values()));ctrl_margin=full_all-bestctrl;conf_ids=[r['case_id'] for r in locked if r['split']=='CONFIRMATION'];conf_pair=[fm[i]-bm[i] for i in conf_ids];conf_ci=bootstrap_ci(conf_pair,22542);neg_full=splitrate(fm,'NEGATIVE_UNPROVABLE')
    gate_checks={'confirmation_full_minus_base_success_pp':confirmation_delta,'full_minus_best_random_or_shuffle_pp':ctrl_margin,'generator_holdout_full_minus_base_success_pp':hold_delta,'negative_verified_success':neg_full,'confirmation_paired_bootstrap95':conf_ci};reproduction_green=confirmation_delta>=.10 and ctrl_margin>=.05 and hold_delta>=.10 and neg_full<=0.0 and conf_ci[0]>0.0;failure_green=(splitrate(bm,'CONFIRMATION')-splitrate(fm,'CONFIRMATION')>=.10 and (mean(rm.values())-full_all>=.05 or mean(sm.values())-full_all>=.05)) or neg_full>0.0
    dump(out/'C39_GENERATION_BEHAVIOR.json',{'schema':'LUCIA_AA_R22542_C39_GENERATION_BEHAVIOR_V1','verification_policy':'CONSERVATIVE_PRESEALED_CANONICAL_EXACT_MATCH_ONLY_COUNTS_KERNEL_VERIFIED; NONCANONICAL_OUTPUTS_UNVERIFIED_NOT_SUCCESS','locked_cases':len(locked),'k':8,'summaries':gs,'gate_checks':gate_checks,'reproduction_gate':'GREEN' if reproduction_green else 'RED','failure_gate':'GREEN' if failure_green else 'RED','raw_tactic_text_saved':False,'raw_model_logits_saved':False,'case_records':gen});dump(out/'C39_BEHAVIOR_GATE.json',{'schema':'LUCIA_AA_R22542_C39_BEHAVIOR_GATE_V1','reproduction_gate':'GREEN' if reproduction_green else 'RED','failure_gate':'GREEN' if failure_green else 'RED','causal_positive_entry':reproduction_green,'causal_failure_entry':failure_green,'gate_checks':gate_checks,'tokenizer_provenance_hold':True,'training_time_base_revision_proven':False,'max_positive_claim':'E3_EXPLORATORY_TOKENIZER_PROVENANCE_HOLD'})
    with (out/'C39_TEACHER_CASE_SCALARS.jsonl').open('w',encoding='utf-8') as f:
        for cond,vals in res.items():
            for x in vals:f.write(json.dumps({'condition':cond,**x},sort_keys=True)+'\n')
    dump(out/'C39_WORKER_RECEIPT.json',{'schema':'LUCIA_AA_R22542_C39_BEHAVIOR_WORKER_RECEIPT_V1','started_unix':start,'finished_unix':time.time(),'torch':torch.__version__,'maxrss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_outputs_observed':True,'hf_redownloads':0,'source':'authoritative escrow run 31484392389','suite_sha256':EXPECTED_SUITE_SHA,'result_files':sorted(p.name for p in out.iterdir())});print('FINAL_GATE',json.dumps({'reproduction':reproduction_green,'failure':failure_green,'gate_checks':gate_checks},sort_keys=True),flush=True)
if __name__=='__main__':main()
