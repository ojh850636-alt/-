from __future__ import annotations
import argparse,contextlib,hashlib,json,math,os,random,re,shutil,sys,time
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
import r22573_c57_suite as suite
SEED=22573
ADAPTER_REPO='morpheuslord/rewrite'
ADAPTER_REV='b0223f103634eb941e067996c07827c17a385a80'
ADAPTER_SHA='afd37e31d5f2da21314cbc2e6dc13879a05853d1d4fa780263571261969504d6'
ADAPTER_CONFIG_SHA='8d1cac28f0fc1322070a3ba8a94733e8ff1c6a1f0e62d39f8d8a2a9703a6fdb1'
BASE_REPO='grammarly/coedit-large'
BASE_REV='5637bcdf9d8d4419f97c8cfea36f7d35c79232b6'
BASE_WEIGHT_SHA='c692f68bca5c6899801bc9fad626fefd3359ccd95e26dccae4dd72186fd98852'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def preseal(out):
    out.mkdir(parents=True,exist_ok=True);rec=suite.preseal_record();(out/'C57_PREOUTPUT_SEAL.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'suite_n':rec['suite_n'],'suite_sha256':rec['suite_sha256'],'weights_present':0,'model_outputs_observed':0},indent=2))
def runtime_smoke(out):
    import torch
    from transformers import AutoConfig,AutoModelForSeq2SeqLM
    from peft import LoraConfig,get_peft_model
    cfg=AutoConfig.from_pretrained(BASE_REPO,revision=BASE_REV,trust_remote_code=False)
    for a,v in [('d_model',64),('d_ff',128),('num_layers',2),('num_decoder_layers',2),('num_heads',4),('d_kv',16),('vocab_size',256)]:
        if hasattr(cfg,a):setattr(cfg,a,v)
    cfg.decoder_start_token_id=0;cfg.pad_token_id=0;cfg.eos_token_id=1
    model=AutoModelForSeq2SeqLM.from_config(cfg);lc=LoraConfig(r=4,lora_alpha=8,lora_dropout=.0,target_modules=['k','q','wi_1','wo','wi_0','o','v'],task_type='SEQ_2_SEQ_LM');p=get_peft_model(model,lc);names=['control_random','control_shuffle','dose_0p25','dose_0p50','dose_1p50','minus_encoder_attn','minus_ffn']
    for n in names:p.add_adapter(n,lc);p.set_adapter(n);p.delete_adapter(n)
    p.set_adapter('default');x=torch.randint(2,256,(1,12));am=torch.ones_like(x);lab=torch.randint(2,256,(1,8))
    with torch.inference_mode():y=p(input_ids=x,attention_mask=am,labels=lab)
    rec={'schema':'R22573_C57_RUNTIME_ABI_V1','pass':True,'weights_present':0,'logits_shape':list(y.logits.shape),'identifiers':names,'target_modules':['k','q','wi_1','wo','wi_0','o','v']};out.mkdir(parents=True,exist_ok=True);(out/'C57_RUNTIME_ABI.json').write_text(json.dumps(rec,indent=2)+'\n');print(rec)
def bootstrap_ci(vals,B=1200):
    rng=random.Random(SEED+991);n=len(vals);means=[]
    for _ in range(B):means.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
    means.sort();return [means[int(.025*B)],means[min(B-1,int(.975*B))]]
def execute(out,work):
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download,snapshot_download
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    from peft import PeftModel
    from safetensors.torch import load_file
    out.mkdir(parents=True,exist_ok=True);raw=work/'raw';raw.mkdir(parents=True,exist_ok=True);ad=raw/'adapter';basep=raw/'base';ad.mkdir();basep.mkdir();started=time.time()
    for fn in ['adapter_config.json','adapter_model.safetensors']:
        src=hf_hub_download(ADAPTER_REPO,filename=fn,revision=ADAPTER_REV);shutil.copy2(src,ad/fn)
    assert sha(ad/'adapter_config.json')==ADAPTER_CONFIG_SHA and sha(ad/'adapter_model.safetensors')==ADAPTER_SHA
    snapshot_download(BASE_REPO,revision=BASE_REV,local_dir=basep,allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','spiece.model'])
    assert sha(basep/'model.safetensors')==BASE_WEIGHT_SHA
    cfg=json.loads((ad/'adapter_config.json').read_text());assert cfg['base_model_name_or_path']==BASE_REPO and cfg['r']==16 and cfg['lora_alpha']==32
    provenance={'schema':'R22573_C57_PROVENANCE_V1','adapter':{'repo':ADAPTER_REPO,'revision':ADAPTER_REV,'sha256':ADAPTER_SHA,'bytes':(ad/'adapter_model.safetensors').stat().st_size,'config_sha256':ADAPTER_CONFIG_SHA,'card_license':'mit','source_policy':'EXACT_ROOT_TWO_FILE_ONLY'},'base':{'repo':BASE_REPO,'revision':BASE_REV,'model_sha256':BASE_WEIGHT_SHA,'bytes':(basep/'model.safetensors').stat().st_size,'license':'cc-by-nc-4.0','training_time_exact_revision_proven':False},'raw_committed_to_git':False,'training_rows_used':0,'claim_boundary':'research-only synthetic GEC assay; raw weights deleted before export'};(out/'C57_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    st=load_file(str(ad/'adapter_model.safetensors'),device='cpu');pairs=defaultdict(set);energy=defaultdict(float);zero=nonfinite=0
    for k,t in st.items():
        zero+=int(float(t.float().norm())==0.0);nonfinite+=int(not torch.isfinite(t).all());root=re.sub(r'\.lora_[AB]\.weight$','',k);side='A' if '.lora_A.' in k else 'B' if '.lora_B.' in k else 'other';pairs[root].add(side);fam=next((x for x in ['wi_0','wi_1','wo','q','k','v','o'] if re.search(rf'\.{re.escape(x)}\.lora_',k)),'other');energy[fam]+=float(t.float().pow(2).sum())
    E=sum(energy.values()) or 1.0;static={'schema':'R22573_C57_STATIC_FORENSICS_V1','tensor_count':len(st),'complete_pair_count':sum(v=={'A','B'} for v in pairs.values()),'zero_tensor_count':zero,'nonfinite_tensor_count':nonfinite,'rank_config':cfg['r'],'alpha':cfg['lora_alpha'],'target_modules':sorted(cfg['target_modules']),'energy_proxy_by_family':{k:v/E for k,v in sorted(energy.items())},'evidence_grade':'E1_STATIC_ONLY','causal_warning':'Static energy is not causal importance.'};(out/'C57_STATIC_FORENSICS.json').write_text(json.dumps(static,indent=2)+'\n');del st
    tok=AutoTokenizer.from_pretrained(basep,local_files_only=True);base=AutoModelForSeq2SeqLM.from_pretrained(basep,local_files_only=True,dtype=torch.float32,device_map='cpu');model=PeftModel.from_pretrained(base,ad,is_trainable=False);model.eval();model.set_adapter('default')
    cases=suite.build_suite();byid={c['id']:c for c in cases};seal=suite.preseal_record();control=[byid[x] for x in seal['control_ids']];gen_cases=[byid[x] for x in seal['generation_ids']];causal_cases=[byid[x] for x in seal['causal_ids']];causal_gen=[byid[x] for x in seal['causal_generation_ids']]
    params={n:p for n,p in model.named_parameters() if ('.lora_A.' in n or '.lora_B.' in n) and '.default.' in n};originals={n:p.detach().clone() for n,p in params.items()}
    def restore():
        with torch.no_grad():
            for n,p in params.items():p.copy_(originals[n])
    def family(n):return next((x for x in ['wi_0','wi_1','wo','q','k','v','o'] if f'.{x}.' in n),'other')
    def mutate(cond):
        restore();model.set_adapter('default')
        with torch.no_grad():
            if cond=='FULL':return
            if cond.startswith('DOSE_'):
                f={'DOSE_0p25':.25,'DOSE_0p50':.5,'DOSE_1p50':1.5}[cond]
                for n,p in params.items():
                    if '.lora_B.' in n:p.mul_(f)
                return
            if cond=='RANDOM_RANK_MATCHED':
                g=torch.Generator().manual_seed(SEED+17)
                for n,p in params.items():
                    o=originals[n];r=torch.randn(o.shape,generator=g,dtype=o.dtype);rn=float(r.norm());on=float(o.norm());p.copy_(r*(on/rn if rn else 0))
                return
            if cond=='LAYER_SHUFFLED':
                groups=defaultdict(list)
                for n,p in params.items():groups[(family(n),'A' if '.lora_A.' in n else 'B',tuple(p.shape),'enc' if '.encoder.' in n else 'dec' if '.decoder.' in n else 'x')].append(n)
                for names in groups.values():
                    if len(names)<2:continue
                    for dst,sn in zip(names,names[1:]+names[:1]):params[dst].copy_(originals[sn])
                return
            raise ValueError(cond)
    def nll(eval_cases,disable=False,batch=4):
        vals=[];ctx=model.disable_adapter() if disable else contextlib.nullcontext()
        with ctx,torch.inference_mode():
            for i in range(0,len(eval_cases),batch):
                chunk=eval_cases[i:i+batch];inputs=tok([suite.PREFIX+c['source'] for c in chunk],padding=True,truncation=True,max_length=128,return_tensors='pt');targets=tok(text_target=[c['target'] for c in chunk],padding=True,truncation=True,max_length=128,return_tensors='pt');labels=targets['input_ids'];mask=(labels!=tok.pad_token_id);lab=labels.clone();lab[~mask]=-100;logits=model(**inputs,labels=lab).logits;loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),lab.reshape(-1),ignore_index=-100,reduction='none').reshape(lab.shape);vals.extend(((loss*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().tolist())
        return vals
    def generate(eval_cases,disable=False,batch=2):
        rows=[];ctx=model.disable_adapter() if disable else contextlib.nullcontext()
        with ctx,torch.inference_mode():
            for i in range(0,len(eval_cases),batch):
                chunk=eval_cases[i:i+batch];inputs=tok([suite.PREFIX+c['source'] for c in chunk],padding=True,truncation=True,max_length=128,return_tensors='pt');ids=model.generate(**inputs,max_new_tokens=32,num_beams=1,do_sample=False);texts=tok.batch_decode(ids,skip_special_tokens=True)
                for c,t in zip(chunk,texts):rows.append({'id':c['id'],'partition':c['partition'],'score':suite.score(t,c)})
        return rows
    def agg_gen(rows):return suite.aggregate([r['score'] for r in rows]) if rows else {}
    def part_gen(rows):return {part:suite.aggregate([x['score'] for x in rows if x['partition']==part]) for part in sorted(set(r['partition'] for r in rows))}
    restore();base_nll=nll(cases,disable=True);restore();full_nll=nll(cases);paired=[b-f for b,f in zip(base_nll,full_nll)];ci=bootstrap_ci(paired);base_mean=sum(base_nll)/len(base_nll);full_mean=sum(full_nll)/len(full_nll);idx={c['id']:i for i,c in enumerate(cases)};full_control=[full_nll[idx[c['id']]] for c in control];control_means={};control_vectors={}
    for cond in ['RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p25','DOSE_0p50','DOSE_1p50']:
        mutate(cond);v=nll(control);control_vectors[cond]=v;control_means[cond]=sum(v)/len(v)
    restore();base_gen=generate(gen_cases,disable=True);restore();full_gen=generate(gen_cases);base_ga=agg_gen(base_gen);full_ga=agg_gen(full_gen);protected_drop=full_ga['protected']-base_ga['protected'];noop_increase=full_ga['noop_false_edit']-base_ga['noop_false_edit'];gen_comp_gain=full_ga['composite']-base_ga['composite'];gen_exact_gain=full_ga['exact']-base_ga['exact'];full_locked=sum(full_control)/len(full_control);random_sep=control_means['RANDOM_RANK_MATCHED']-full_locked;shuffle_sep=control_means['LAYER_SHUFFLED']-full_locked
    positive=bool(ci[0]>.01 and random_sep>=.015 and shuffle_sep>=.015 and (gen_comp_gain>=.08 or gen_exact_gain>=.10) and protected_drop>=-.03 and noop_increase<=.05);neg_lane=(ci[1]<0 or gen_comp_gain<=-.08);control_bad=(full_locked-control_means['RANDOM_RANK_MATCHED']>=.015 and full_locked-control_means['LAYER_SHUFFLED']>=.015);failure=bool(neg_lane and (control_bad or noop_increase>=.15))
    behavior={'schema':'R22573_C57_BEHAVIOR_V1','n':len(cases),'base_mean_target_nll':base_mean,'full_mean_target_nll':full_mean,'paired_nll_gain_base_minus_full':sum(paired)/len(paired),'paired_bootstrap_ci95':ci,'locked_control_n':len(control),'full_locked_control_mean_nll':full_locked,'control_mean_nll':control_means,'full_minus_random_separation_nats':random_sep,'full_minus_shuffle_separation_nats':shuffle_sep,'generation_n':len(gen_cases),'base_generation':base_ga,'full_generation':full_ga,'base_generation_by_partition':part_gen(base_gen),'full_generation_by_partition':part_gen(full_gen),'generation_composite_gain':gen_comp_gain,'generation_exact_gain':gen_exact_gain,'protected_preservation_delta':protected_drop,'noop_false_edit_delta':noop_increase,'positive_lane_green':positive,'failure_lane_green':failure,'causal_entry':bool(positive or failure),'training_rows_used':0,'raw_model_text_exported':False};(out/'C57_BEHAVIOR.json').write_text(json.dumps(behavior,indent=2)+'\n')
    base_gen_map={x['id']:x['score'] for x in base_gen};full_gen_map={x['id']:x['score'] for x in full_gen};case_scalars=[]
    for c,b,f in zip(cases,base_nll,full_nll):case_scalars.append({'id':c['id'],'partition':c['partition'],'base_nll':b,'full_nll':f,'nll_gain':b-f,'base_generation_score':base_gen_map.get(c['id']),'full_generation_score':full_gen_map.get(c['id'])})
    (out/'C57_CASE_SCALARS.json').write_text(json.dumps({'schema':'R22573_C57_CASE_SCALARS_V1','cases':case_scalars},indent=2)+'\n')
    causal={'schema':'R22573_C57_CAUSAL_V1','executed':False,'reason':'BEHAVIOR_LANES_RED','groups':{},'e3_localized_groups':[]}
    if positive or failure:
        restore();base_cn=nll(causal_cases,disable=True);restore();full_cn=nll(causal_cases);full_cgen=generate(causal_gen);full_cga=agg_gen(full_cgen);gain=(sum(base_cn)-sum(full_cn))/len(full_cn);groups={};localized=[]
        def remove_group(g):
            restore()
            with torch.no_grad():
                for n,p in params.items():
                    if '.lora_B.' not in n:continue
                    m=re.search(r'\.block\.(\d+)\.',n);layer=int(m.group(1)) if m else -1;hit=False
                    if g=='MINUS_ENCODER_ATTN':hit=('.encoder.' in n and family(n) in ['q','k','v','o'])
                    elif g=='MINUS_DECODER_SELF_ATTN':hit=('.decoder.' in n and 'SelfAttention' in n and family(n) in ['q','k','v','o'])
                    elif g=='MINUS_CROSS_ATTN':hit=('EncDecAttention' in n and family(n) in ['q','k','v','o'])
                    elif g=='MINUS_FFN':hit=family(n) in ['wi_0','wi_1','wo']
                    elif g=='MINUS_EARLY':hit=0<=layer<8
                    elif g=='MINUS_MIDDLE':hit=8<=layer<16
                    elif g=='MINUS_LATE':hit=16<=layer<24
                    if hit:p.zero_()
        for g in seal['causal_if_admitted']:
            remove_group(g);nv=nll(causal_cases);gv=generate(causal_gen);ga=agg_gen(gv);mn=sum(nv)/len(nv);fmn=sum(full_cn)/len(full_cn);erasure=(mn-fmn)/gain if gain>1e-9 else 0.0;comp_drop=full_cga['composite']-ga['composite'];prot_improve=ga['protected']-full_cga['protected'];required=bool(positive and erasure>=.5 and (mn-fmn)>=.015 and comp_drop>=.05 and prot_improve<=.01);groups[g]={'mean_nll':mn,'nll_worsening_vs_full':mn-fmn,'full_gain_erasure_fraction':erasure,'generation':ga,'generation_composite_drop_vs_full':comp_drop,'protected_improvement_vs_full':prot_improve,'localized_required':required};localized.append(g) if required else None
        restore();causal={'schema':'R22573_C57_CAUSAL_V1','executed':True,'entry_lane':'positive' if positive else 'failure','base_causal_mean_nll':sum(base_cn)/len(base_cn),'full_causal_mean_nll':sum(full_cn)/len(full_cn),'full_gain_nats':gain,'full_causal_generation':full_cga,'groups':groups,'e3_localized_groups':localized,'e3_credit_allowed':bool(positive and localized),'e3_gate':'presealed NLL erasure + generation drop + protected-span guard'}
    (out/'C57_CAUSAL.json').write_text(json.dumps(causal,indent=2)+'\n');cex=[]
    for r in case_scalars:
        bs=r.get('base_generation_score');fs=r.get('full_generation_score')
        if bs and fs and abs(fs['composite']-bs['composite'])>=.20:cex.append({'id':r['id'],'partition':r['partition'],'base_composite':bs['composite'],'full_composite':fs['composite'],'delta':fs['composite']-bs['composite'],'base_protected':bs['protected'],'full_protected':fs['protected'],'base_noop_false':bs['noop_false_edit'],'full_noop_false':fs['noop_false_edit']})
    (out/'C57_COUNTEREXAMPLES.json').write_text(json.dumps({'schema':'R22573_C57_COUNTEREXAMPLES_V1','raw_model_text_exported':False,'cases':cex},indent=2)+'\n');(out/'C57_CLEANUP.json').write_text(json.dumps({'schema':'R22573_C57_CLEANUP_V1','controlled_cleanup_by_runner':False,'raw_root_deleted':False,'elapsed_seconds_before_workflow_cleanup':time.time()-started},indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preseal','runtime-smoke','execute']);ap.add_argument('--out',required=True);ap.add_argument('--work');a=ap.parse_args();out=Path(a.out)
    if a.mode=='preseal':preseal(out)
    elif a.mode=='runtime-smoke':runtime_smoke(out)
    else:execute(out,Path(a.work))
if __name__=='__main__':main()
