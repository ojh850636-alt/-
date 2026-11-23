"""One-use external public RTE LoRA CPU archaeology; receipt-only external outputs.

Use ONLY inside isolated network worker with pinned source. No hidden network calls;
Hugging Face files are fetched explicitly in source_capture() and never exported.
This worker intentionally makes NO supported causal claims from group ablation alone.
"""
from __future__ import annotations
import argparse, json, hashlib, math, os, sys, tempfile, time, shutil, traceback
from pathlib import Path

ADAPTER_REPO='october25kim/lora-merge-prereg-bert-base-uncased'
ADAPTER_REV='6fae52636e9f2f5e8def14a88fae202dbceaba47'
ADAPTER_SUB='rte/seed0'
ADAPTER_SHA='4f9b3e87b79196ce31a0fc192c5e477794481cafb6513d50bf512a1bd6ca00d2'
BASE_REPO='google-bert/bert-base-uncased'
BASE_REV='86b5e0934494bd15c9632b12f734a8a67f723594'
BASE_SHA='68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3'
SUITE_SHA='746aab149b2844da731bec52db8101b1d0277f7e6be918cbe86cf4e7f3ff0e31'
EXPECTED_POLICY_SHA=None

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def write_atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def source_capture(isolate):
    root=Path(isolate)
    os.environ['HF_HOME']=str(root/'hfcache')
    os.environ['HF_HUB_CACHE']=str(root/'hfcache'/'hub')
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    base=root/'base';ad=root/'adapter';base.mkdir();ad.mkdir()
    get=lambda repo,rev,name: Path(hf_hub_download(repo,filename=name,revision=rev,local_dir=str(base if repo==BASE_REPO else ad)))
    files=[]
    for name in ('config.json','model.safetensors','tokenizer.json','vocab.txt','tokenizer_config.json'):
        p=get(BASE_REPO,BASE_REV,name); files.append(p)
    for name in ('adapter_config.json','adapter_model.safetensors','train_meta.json'):
        p=Path(hf_hub_download(ADAPTER_REPO,filename=f'{ADAPTER_SUB}/{name}',revision=ADAPTER_REV,local_dir=str(ad)))
        files.append(p)
    bf=base/'model.safetensors';af=ad/ADAPTER_SUB/'adapter_model.safetensors'
    if sha(bf)!=BASE_SHA or sha(af)!=ADAPTER_SHA:
        raise RuntimeError('SOURCE_SHA_MISMATCH')
    cfg=json.loads((ad/ADAPTER_SUB/'adapter_config.json').read_text())
    if cfg['peft_type']!='LORA' or cfg['r']!=8 or cfg['lora_alpha']!=16:
        raise RuntimeError('ADAPTER_ARCHITECTURE_REJECTED')
    if cfg['base_model_name_or_path'] not in ('bert-base-uncased','google-bert/bert-base-uncased'):
        raise RuntimeError('BASE_ID_CONFLICT:'+str(cfg['base_model_name_or_path']))
    if cfg.get('revision') not in (None,BASE_REV):raise RuntimeError('UNEXPECTED_BASE_REVISION')
    with safe_open(str(af),framework='pt',device='cpu') as f:ak=list(f.keys())
    with safe_open(str(bf),framework='pt',device='cpu') as f:bk=list(f.keys())
    if len(ak)<4 or len(bk)<50:raise RuntimeError('UNEXPECTED_TENSOR_INVENTORY')
    meta=json.loads((ad/ADAPTER_SUB/'train_meta.json').read_text())
    return {'base_file':str(bf),'adapter_file':str(af),'adapter_config':cfg,
            'adapter_keys':len(ak),'base_keys':len(bk),
            'provenance':{'adapter_repo':ADAPTER_REPO,'adapter_commit':ADAPTER_REV,'adapter_sha256':ADAPTER_SHA,
                          'base_repo':BASE_REPO,'base_commit':BASE_REV,'base_sha256':BASE_SHA,
                          'training_time_base_commit_proven':False,
                          'training_meta_sha256':sha(ad/ADAPTER_SUB/'train_meta.json'),
                          'tokenizer_file_shas':{n:sha(base/n) for n in ('tokenizer.json','vocab.txt','tokenizer_config.json')},
                          'source_file_count':len(files),'adapter_keys':len(ak),'base_keys':len(bk)}}

def build_model(root,cap):
    import torch
    from transformers import BertConfig,BertForSequenceClassification,BertTokenizerFast
    from safetensors.torch import load_file
    from r128.adapter import parse_lora_tensors,Controller,stat_operator
    from r128.compat import bridge_layernorm_aliases
    torch.manual_seed(224)
    torch.set_num_threads(min(4,max(1,os.cpu_count() or 2)))
    bdir=Path(root)/'base'
    cfg=BertConfig.from_json_file(str(bdir/'config.json'))
    cfg.num_labels=2;cfg.problem_type='single_label_classification'
    model=BertForSequenceClassification(cfg)
    base_sd=load_file(cap['base_file'],device='cpu')
    # PRE-OUTPUT AMEND128: byte-preserving legacy alias bridge (never silent).
    base_sd,ln_aliases=bridge_layernorm_aliases(base_sd,model.state_dict().keys())
    info=model.load_state_dict(base_sd,strict=False)
    un_loaded=[x for x in info.missing_keys if not x.startswith('classifier.')]
    if un_loaded: raise RuntimeError('BASE_UNEXPECTED_MISSING:'+str(un_loaded[:4]))
    if any(not x.startswith(('cls.','predictions.','seq_relationship.')) for x in info.unexpected_keys):
        raise RuntimeError('BASE_UNEXPECTED_EXTRA:'+str(info.unexpected_keys[:4]))
    del base_sd
    model.eval()
    tok=BertTokenizerFast.from_pretrained(str(bdir),local_files_only=True)
    sd=load_file(cap['adapter_file'],device='cpu')
    factors,aux=parse_lora_tensors(sd,expected_rank=8)
    del sd
    ctrl=Controller(model,factors,aux,alpha=16)
    return model,tok,ctrl,stat_operator(factors,alpha=16),len(factors),len(ln_aliases)

def exact_one_sided_reverse(old,new,y):
    from r128.logic import paired
    return paired(new,old,y)['one_sided_p']

def run_cpu(isolate,source,suite,policy):
    import torch
    from r128.adapter import Controller,selection
    from r128.logic import paired,paired_bootstrap,gate,digest
    model,tok,ctrl,site_stats,nsites,norm_aliases=build_model(isolate,source)
    if nsites<30: raise RuntimeError('LOW_OPERATOR_SITE_COUNT')
    y=[0 if c['expected_label']=='ENTAILS' else 1 for c in suite]
    splits=[c['split'] for c in suite]
    ids=[c['id'] for c in suite]
    sentences=[c['premise'] for c in suite]
    hypothesis=[c['hypothesis'] for c in suite]
    with torch.no_grad():
        first=tok(sentences[:4],hypothesis[:4],padding=True,truncation=True,max_length=128,return_tensors='pt')
        if 'token_type_ids' not in first:raise RuntimeError('BERT_SEGMENT_IDS_MISSING')
    conditions=['BASE','HEAD_ONLY','LORA_ONLY','FULL','RANDOM_SAME_HEAD','SHUFFLED_SAME_HEAD']
    predictions={}
    conf_hi={}
    def evaluate(name,indices=None,dose=1.,scope='ALL',seed=128):
        if indices is None:indices=list(range(len(suite)))
        ctrl.condition(name,scope=scope,dose=dose,seed=seed)
        pred=[]; confident=[]
        with torch.inference_mode():
            for lo in range(0,len(indices),8):
                ix=indices[lo:lo+8]
                tokens=tok([sentences[i] for i in ix],[hypothesis[i] for i in ix],
                           padding=True,truncation=True,max_length=128,return_tensors='pt')
                logits=model(**tokens).logits
                p=torch.softmax(logits,dim=1)
                pred+=p.argmax(dim=1).tolist()
                confident+=[float(v)>=0.9 for v in p.max(dim=1).values.tolist()]
        return pred,confident
    try:
        for c in conditions:
            predictions[c],conf_hi[c]=evaluate(c)
        # Exact dose0 ensures trained-head only. This tests operator control, not neural parity.
        full0,_=evaluate('FULL',dose=0.0)
        if full0!=predictions['HEAD_ONLY']:raise RuntimeError('DOSE_ZERO_HEAD_MISMATCH')
        # Reference negative control: Base and LORA_ONLY use identical classifier initialization.
        lo0,_=evaluate('LORA_ONLY',dose=0.0)
        if lo0!=predictions['BASE']:raise RuntimeError('DOSE_ZERO_BASE_MISMATCH')
        from r128.logic import paired
        bysplit={}
        for s in sorted(set(splits)) + ['ALL']:
            ix=[i for i in range(len(suite)) if s=='ALL' or splits[i]==s]
            ys=[y[i] for i in ix]
            local={k:[predictions[k][i] for i in ix] for k in conditions}
            bysplit[s]={'n':len(ix),'by_condition':{k:sum(p==t for p,t in zip(v,ys))/len(ys) for k,v in local.items()},
                        'full_vs_head':paired(local['HEAD_ONLY'],local['FULL'],ys),
                        'full_vs_random':paired(local['RANDOM_SAME_HEAD'],local['FULL'],ys),
                        'full_vs_shuffle':paired(local['SHUFFLED_SAME_HEAD'],local['FULL'],ys)}
        gates=gate(bysplit,policy)
        global_pair=bysplit['ALL']['full_vs_head']
        ci=paired_bootstrap(predictions['HEAD_ONLY'],predictions['FULL'],y)
        gates['paired_bootstrap_95_ci']=ci
        gates['positive_ci_low_gt_zero']=ci[0]>0
        gates['behavior_positive_screen']='GREEN' if gates['behavior_positive_screen']=='GREEN' and ci[0]>0 else 'RED'
        gates['failure_screen']='GREEN' if (global_pair['delta']<=-policy['failure_gate']['min_drop']
           and exact_one_sided_reverse(predictions['HEAD_ONLY'],predictions['FULL'],y)<=policy['failure_gate']['max_p']) else 'RED'
        # Negative audit: on written author-generated formulas. A 'NOT_ENTAILS' high confidence is not necessarily unsafe.
        gates['safety_screen']='NOT_EVALUATED_EXTERNAL_COLLATERAL_OR_NATURAL_OOD'
        # Exactly predicated broad/causal screen branch, no post-hoc changes. 
        # Broad site ablation is an intervention but not a minimal causal circuit.
        diagnostic={}
        ix=[i for i,s in enumerate(splits) if s in ('DISCOVERY','CONFIRMATION')]
        for group in policy['intervention']['groups']:
            scope='ALL'
            # Use FULL with all EXCEPT selected group: temporarily flip selection for appropriate sites.
            # Implement by patching a non-exported active_site predicate.
            ctrl.condition('FULL',scope='ALL')
            if group=='ALL':
                p,_=evaluate('HEAD_ONLY',indices=ix)
            else:
                # Hooks route to actual scope through predicate; choose complement using explicit active set.
                orig_factors=ctrl.ops
                exclude={site for site in ctrl.ops if selection(site,group)}
                # With dose=0 for excluded sites only via mask on bound controller.
                ctrl.suppressed=exclude
                try:
                    p,_=evaluate('FULL',indices=ix)
                finally:
                    ctrl.suppressed=set()
            ys=[y[i] for i in ix]
            ref=[predictions['FULL'][i] for i in ix]
            diagnostic[group]={'n':len(ix),'full_vs_removed':paired(ref,p,ys),
                 'site_count':sum(selection(site,group) for site in ctrl.ops)}
        # Accurate evidence type: behavior + broad intervention, not necessary/sufficient causal E4.
        dose_results={}
        for d in policy['dose']:
            p,_=evaluate('FULL',dose=d)
            dose_results[str(d)]={'accuracy':sum(a==t for a,t in zip(p,y))/len(y),
                        'delta_vs_head':paired(predictions['HEAD_ONLY'],p,y)['delta']}
        site_bands={}
        for site in site_stats:
            fam=site['family'];s=site_bands.setdefault(fam,{'site_count':0,'operator_norm_sq':0.})
            s['site_count']+=1;s['operator_norm_sq']+=site['operator_norm_sq']
        return {'schema':'LAA128_EXTERNAL_RTE_BEHAVIOR_V1','source':source['provenance'],
              'case_count':len(y),'case_suite_sha256':SUITE_SHA,
              'neural_forward_performed':True,
              'conditions':conditions,'by_split':bysplit,'gates':gates,
              'dose':dose_results,'broad_site_screen':diagnostic,'static_by_family':site_bands,
              'aux_trained_head_tensor_count':len(ctrl.aux),'active_operator_sites':len(ctrl.factors),
              'legacy_layernorm_alias_count':norm_aliases,
              'claim_boundary':{'paired_behavior_e2_eligible':True,'causal_e3_e4_e5_supported':False,
                'algorithm_transfer':False,'trained_student':False,'lucia_promotion':False,
                'evidence_not_independent':'ALL_240_FROM_R127_AUTHOR_GENERATOR',
                'model_vs_manual_peft_parity':'NOT_TESTED','training_time_base_revision_proven':False},
              'public_raw_bytes':0, 'user_gpu_used':False}
    finally:
        ctrl.close()

def verify_suite(suite_path):
    data=Path(suite_path).read_bytes()
    if sha_bytes(data)!=SUITE_SHA: raise RuntimeError('SUITE_SHA_MISMATCH')
    rows=[json.loads(line) for line in data.decode().splitlines()]
    if len(rows)!=240 or len({x['id'] for x in rows})!=240:raise RuntimeError('SUITE_EXPECTATION_FAIL')
    if {x['split'] for x in rows}!={'DISCOVERY','CONFIRMATION','FAMILY_OOD'}:raise RuntimeError('SPLIT_MISSING')
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',required=True)
    ap.add_argument('--policy',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Store error-only receipt if any step fails; do not build a made-up positive record.
    report={'schema':'LAA128_EXECUTION_V1','status':'INIT','phase':'PREOUTPUT','source_download_bytes':0}
    phase='PREOUTPUT'
    raw=None
    try:
        suite=verify_suite(args.suite)
        policy_bytes=Path(args.policy).read_bytes()
        policy=json.loads(policy_bytes)
        if policy.get('case_suite_sha256')!=SUITE_SHA or policy.get('source_binding')!=ADAPTER_SHA:
            raise RuntimeError('PRESEAL_POLICY_UNBOUND')
        if policy['behavior_gate']['max_p']<=0 or policy['intervention']['groups']!=['ALL','ATTENTION','MLP','POOLER','LAYER_BAND_EARLY','LAYER_BAND_MIDDLE','LAYER_BAND_LATE']:
            raise RuntimeError('POLICY_INVALID')
        phase='CAPTURE'
        raw=tempfile.mkdtemp(prefix='laa128-raw-')
        cap=source_capture(raw)
        phase='MODEL_BEHAVIOR'
        result=run_cpu(raw,cap,suite,policy)
        write_atomic(out/'RESULT128.json',result)
        report.update({'status':'COMPLETED_SCOPED_E2_AND_NONPROMOTED_GROUP_SCREEN',
                       'phase':'MODEL_BEHAVIOR','source_download_bytes':sum(Path(x).stat().st_size for x in (cap['base_file'],cap['adapter_file'])),
                       'public_behavior_receipt':True,'actual_model_outputs_observed':True})
    except Exception as e:
        report.update({'status':'HOLD_NO_CLAIM','phase':phase,'error_type':type(e).__name__,
                       'error_code':str(e)[:240]})
    finally:
        if raw is not None:
            try:shutil.rmtree(raw)
            except Exception:report['cleanup']='FAILED_TEMP_CLEANUP';report['status']='HOLD_DELETE_FAILURE'
            else:report['cleanup']='SELF_REPORTED_TEMP_RAW_REMOVED'
        report['raw_weight_in_receipt']=False
        write_atomic(out/'EXEC128.json',report)
        # Always fail the Action if not scientifically completed, but preserve minimal HOLD receipt.
        return 0 if report['status'].startswith('COMPLETED') else 1
if __name__=='__main__':sys.exit(main())
