#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, os, random, statistics, time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_core(path: Path):
    spec=importlib.util.spec_from_file_location('c4_core',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-dir',type=Path,required=True);ap.add_argument('--adapter-dir',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--core-script',type=Path,required=True);args=ap.parse_args()
    c=load_core(args.core_script);out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    random.seed(c.SEED);torch.manual_seed(c.SEED);torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
    cases,negatives=c.build_dataset()
    c.write_jsonl(out/'C4_SQL_CASES.jsonl',cases)
    c.write_json(out/'C4_ATLAS_SCOPE.json',{'schema':'LUCIA_AA_C4_FULL_ATLAS_SCOPE_V1','candidate_id':c.CANDIDATE_ID,'positive_count':len(cases),'negative_count_not_scored_here':len(negatives),'split_counts':dict(__import__('collections').Counter(x['split'] for x in cases)),'dataset_sha256':c.sha256_file(out/'C4_SQL_CASES.jsonl'),'role':'BASE_FULL_MAX_BEHAVIOR_ATLAS_ONLY'})
    tok=AutoTokenizer.from_pretrained(args.base_dir,local_files_only=True,trust_remote_code=False);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.base_dir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,args.adapter_dir,local_files_only=True);model.eval();model.config.use_cache=False
    started=time.time()
    with model.disable_adapter():base_rows=c.score_pairs(model,tok,cases,batch_size=16)
    full_rows=c.score_pairs(model,tok,cases,batch_size=16)
    c.write_jsonl(out/'C4_TF_BASE.jsonl',base_rows);c.write_jsonl(out/'C4_TF_FULL.jsonl',full_rows)
    bmap={r['case_id']:r for r in base_rows};fmap={r['case_id']:r for r in full_rows};paired=[]
    for case in cases:
        b=bmap[case['case_id']];f=fmap[case['case_id']]
        paired.append({'case_id':case['case_id'],'split':case['split'],'family':case['family'],'base_preferred':b['reference_preferred'],'full_preferred':f['reference_preferred'],'base_margin':b['margin'],'full_margin':f['margin'],'margin_delta':f['margin']-b['margin']})
    c.write_jsonl(out/'C4_BEHAVIOR_ATLAS.jsonl',paired)
    split_stats={};family_stats={}
    for split in ('DISCOVERY','CONFIRMATION','OOD'):
        xs=[x for x in paired if x['split']==split];rescue=sum((not x['base_preferred']) and x['full_preferred'] for x in xs);harm=sum(x['base_preferred'] and (not x['full_preferred']) for x in xs);bacc=sum(x['base_preferred'] for x in xs)/len(xs);facc=sum(x['full_preferred'] for x in xs)/len(xs);lo,hi=c.bootstrap_delta([x['base_preferred'] for x in xs],[x['full_preferred'] for x in xs],c.SEED+len(split));split_stats[split]={'n':len(xs),'base_preference_accuracy':bacc,'full_preference_accuracy':facc,'delta':facc-bacc,'rescue':rescue,'harm':harm,'exact_one_sided_p':c.exact_one_sided_p(rescue,harm),'bootstrap_95':[lo,hi],'mean_margin_delta':statistics.fmean(x['margin_delta'] for x in xs)}
    for key in sorted({(x['split'],x['family']) for x in paired}):
        xs=[x for x in paired if (x['split'],x['family'])==key];family_stats['|'.join(key)]={'n':len(xs),'base_preference_accuracy':sum(x['base_preferred'] for x in xs)/len(xs),'full_preference_accuracy':sum(x['full_preferred'] for x in xs)/len(xs),'mean_margin_delta':statistics.fmean(x['margin_delta'] for x in xs),'rescue':sum((not x['base_preferred']) and x['full_preferred'] for x in xs),'harm':sum(x['base_preferred'] and (not x['full_preferred']) for x in xs)}
    c.write_json(out/'C4_ATLAS_SUMMARY.json',{'schema':'LUCIA_AA_C4_FULL_ATLAS_SUMMARY_V1','candidate_id':c.CANDIDATE_ID,'elapsed_seconds':time.time()-started,'split_statistics':split_stats,'split_family_statistics':family_stats,'teacher_forced_evidence_only':True,'actual_generation_not_claimed_here':True})
    counter=[x for x in paired if (not x['full_preferred']) or x['margin_delta']<0]
    c.write_jsonl(out/'C4_COUNTEREXAMPLES.jsonl',counter)
    c.write_json(out/'C4_STATIC_DELTA_REPORT.json',c.static_report(args.adapter_dir/'adapter_model.safetensors'))
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():files.append({'path':p.relative_to(out).as_posix(),'size_bytes':p.stat().st_size,'sha256':c.sha256_file(p)})
    c.write_json(out/'C4_ATLAS_EVIDENCE_INDEX.json',{'schema':'LUCIA_AA_C4_FULL_ATLAS_EVIDENCE_INDEX_V1','candidate_id':c.CANDIDATE_ID,'files':files,'raw_weight_included':False,'raw_logits_included':False,'raw_activation_included':False,'verdict':'PASS_RAW_FREE_ATLAS'})
    print(json.dumps({'candidate_id':c.CANDIDATE_ID,'positive_count':len(cases),'split_statistics':split_stats,'counterexample_count':len(counter)},indent=2))
if __name__=='__main__':main()
