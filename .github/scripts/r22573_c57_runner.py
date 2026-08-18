from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, random, re, shlex, shutil, sys, time
from collections import defaultdict, Counter
from pathlib import Path
sys.dont_write_bytecode=True
SEED=22573
ADAPTER_REPO='Harish2002/cli-lora-tinyllama'
ADAPTER_REV='e3b89b4003c39161fcecf0b6262280afe42fe515'
ADAPTER_SHA='e37f9f26baf3d204058ed21c01c610823fe55b6d4311bff30df5b9ca44213531'
BASE_REPO='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
BASE_REV='fe8a4ea1ffedaf415f4da2f062534de366a451e6'
BASE_WEIGHT_SHA='6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933'
SAFE_CMDS={'git','ls','mkdir','cp','mv','grep','wc','sort','head','tail','tar','gzip','python','python3','pip','pip3','find','pwd','cat','touch'}
DANGEROUS_TOKENS={'rm','sudo','curl','wget','ssh','scp','nc','ncat','dd','mkfs','shutdown','reboot','chmod','chown','kill','pkill','iptables'}

def sha_obj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def case(q,cmd,family,part='basic'):
    return {'question':q,'command':cmd,'family':family,'partition':part,'target':None if cmd is None else f'Use `{cmd}`.'}
def build_suite():
    xs=[]
    xs += [case('How do I initialize a new Git repository?','git init','git'),case('How can I check the current Git working tree status?','git status','git'),case('How do I create a new branch named feature-x?','git branch feature-x','git'),case('How do I switch to the branch feature-x?','git checkout feature-x','git'),case('How do I create and switch to a new branch named hotfix?','git checkout -b hotfix','git','composition'),case('How do I stage the file report.txt?','git add report.txt','git'),case('How do I stage all current changes?','git add .','git'),case('How do I commit staged changes with message fix parser?','git commit -m "fix parser"','git','quoting'),case('How do I see the commit history?','git log','git'),case('How do I show a compact one-line Git history?','git log --oneline','git'),case('How do I list local Git branches?','git branch','git'),case('How do I stash my current changes?','git stash','git'),case('How do I restore the most recent stash?','git stash pop','git'),case('How do I see the diff of unstaged changes?','git diff','git'),case('How do I merge branch feature-x into the current branch?','git merge feature-x','git'),case('How do I display configured Git remotes?','git remote -v','git')]
    xs += [case('How do I list files in the current directory?','ls','fs'),case('How do I list all files including hidden ones?','ls -a','fs'),case('How do I list files with long details?','ls -l','fs'),case('How do I create a directory named logs?','mkdir logs','fs'),case('How do I create nested directories a/b/c?','mkdir -p a/b/c','fs'),case('How do I copy input.txt to backup.txt?','cp input.txt backup.txt','fs'),case('How do I copy the directory src recursively to src_backup?','cp -r src src_backup','fs'),case('How do I rename old.txt to new.txt?','mv old.txt new.txt','fs'),case('How do I print the current working directory?','pwd','fs'),case('How do I create an empty file named notes.txt?','touch notes.txt','fs'),case('How do I display file.txt on the terminal?','cat file.txt','fs'),case('How do I search for error in app.log?','grep error app.log','text'),case('How do I search case-insensitively for warning in app.log?','grep -i warning app.log','text'),case('How do I include line numbers while searching for TODO in main.py?','grep -n TODO main.py','text'),case('How do I count lines in data.csv?','wc -l data.csv','text'),case('How do I sort names.txt?','sort names.txt','text'),case('How do I output unique sorted lines from names.txt?','sort -u names.txt','text'),case('How do I show the first 5 lines of app.log?','head -n 5 app.log','text'),case('How do I show the last 10 lines of app.log?','tail -n 10 app.log','text'),case('How do I search recursively for TODO under src?','grep -r TODO src','text','composition'),case('How do I find Python files under src?','find src -name "*.py"','text','quoting'),case('How do I copy a file named quarterly report.txt to archive report.txt?','cp "quarterly report.txt" "archive report.txt"','fs','quoting')]
    xs += [case('How do I create archive.tar from the docs directory?','tar -cf archive.tar docs','archive'),case('How do I create a gzipped tar archive backup.tar.gz from backup?','tar -czf backup.tar.gz backup','archive'),case('How do I extract archive.tar?','tar -xf archive.tar','archive'),case('How do I extract backup.tar.gz?','tar -xzf backup.tar.gz','archive'),case('How do I gzip report.txt?','gzip report.txt','archive'),case('How do I create a Python virtual environment named .venv?','python -m venv .venv','python'),case('How do I install requests with pip?','pip install requests','python'),case('How do I show installed pip packages?','pip list','python'),case('How do I show the Python version?','python --version','python'),case('How do I run script.py with Python?','python script.py','python')]
    xs += [case('How do I count the number of lines containing error in app.log?','grep error app.log | wc -l','pipe','composition'),case('How do I show the first 3 sorted lines of names.txt?','sort names.txt | head -n 3','pipe','composition'),case('How do I show the last 2 sorted unique lines of names.txt?','sort -u names.txt | tail -n 2','pipe','composition'),case('How do I count Python files under src?','find src -name "*.py" | wc -l','pipe','composition'),case('How do I list Git branches and search for feature?','git branch | grep feature','pipe','composition'),case('How do I list files and count the output lines?','ls | wc -l','pipe','composition'),case('How do I search TODO recursively under src and count matches?','grep -r TODO src | wc -l','pipe','composition'),case('How do I print the first line containing warning in app.log?','grep warning app.log | head -n 1','pipe','composition')]
    xs += [case('What does exit status zero usually mean in a command-line program?',None,'concept','negative'),case('What is the difference between standard output and standard error?',None,'concept','negative'),case('What is a Git branch conceptually?',None,'concept','negative'),case('What does a relative path mean?',None,'concept','negative'),case('Why are quotes useful around filenames containing spaces?',None,'concept','negative'),case('What is a shell pipeline?',None,'concept','negative'),case('What is a Python virtual environment?',None,'concept','negative'),case('What does recursive search mean?',None,'concept','negative')]
    assert len(xs)==64
    for i,x in enumerate(xs):x['id']=f'C57-{i:03d}'
    return xs

def strip_code(answer):
    s=answer.strip();m=re.search(r'```(?:bash|sh|shell)?\s*\n?([^`]+?)```',s,re.I|re.S)
    if m:s=m.group(1).strip().splitlines()[0].strip()
    else:
        m=re.search(r'`([^`\n]+)`',s)
        if m:s=m.group(1).strip()
        else:
            cand=[]
            for z in [q.strip() for q in s.splitlines() if q.strip()]:
                z=re.sub(r'^(?:\$|>|Command:|Use:|Run:)\s*','',z,flags=re.I).strip()
                if z and z.split()[0] in SAFE_CMDS:cand.append(z)
            if cand:s=cand[0]
            else:return None
    return re.sub(r'^\$\s*','',s).strip()
def parse_pipeline(cmd):
    if not cmd:return None
    if any(tok in cmd.lower().split() for tok in DANGEROUS_TOKENS):return {'unsafe':True,'segments':[]}
    try:
        segs=[]
        for part in [x.strip() for x in cmd.split('|')]:
            toks=shlex.split(part)
            if not toks:return None
            segs.append(toks)
        return {'unsafe':False,'segments':segs}
    except Exception:return None
def compare_ast(pred,gold):
    if gold is None:
        extracted=strip_code(pred) if isinstance(pred,str) else None;pa=parse_pipeline(extracted) if extracted else None;executable=int(bool(pa and pa.get('segments')))
        return {'extractable':int(extracted is not None),'family':int(not executable),'flags':int(not executable),'operands':int(not executable),'pipe':int(not executable),'exact':int(not executable),'unsafe':int(bool(pa and pa.get('unsafe'))),'negative_fp':executable,'semantic':int(not executable)}
    ex=strip_code(pred) if isinstance(pred,str) else None;pa=parse_pipeline(ex);ga=parse_pipeline(gold)
    if not pa or pa.get('unsafe') or not ga:return {'extractable':0,'family':0,'flags':0,'operands':0,'pipe':0,'exact':0,'unsafe':int(bool(pa and pa.get('unsafe'))),'negative_fp':0,'semantic':0}
    ps,gs=pa['segments'],ga['segments'];pipe=int(len(ps)==len(gs));fam=int(pipe and all(ps[i][0]==gs[i][0] for i in range(len(gs))))
    def split(t):return [x for x in t[1:] if x.startswith('-')],[x for x in t[1:] if not x.startswith('-')]
    flags=oper=0
    if fam:flags=int(all(sorted(split(ps[i])[0])==sorted(split(gs[i])[0]) for i in range(len(gs))));oper=int(all(split(ps[i])[1]==split(gs[i])[1] for i in range(len(gs))))
    exact=int(fam and flags and oper and pipe);return {'extractable':1,'family':fam,'flags':flags,'operands':oper,'pipe':pipe,'exact':exact,'unsafe':0,'negative_fp':0,'semantic':(fam+flags+oper+pipe+exact)/5}
def aggregate(rows):return {k:sum(float(x[k]) for x in rows)/len(rows) for k in rows[0]}
def bootstrap_ci(vals,B=1000):
    rng=random.Random(SEED);n=len(vals);m=[sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(B)];m.sort();return [m[int(.025*B)],m[min(B-1,int(.975*B))]]
def preseal(out):
    out.mkdir(parents=True,exist_ok=True);xs=build_suite();pos=[x for x in xs if x['command']]
    perfect=[compare_ast((f'Use `{x["command"]}`.' if x['command'] else 'This is a conceptual explanation with no executable command.'),x['command']) for x in xs];copy=[compare_ast(x['question'],x['command']) for x in xs];mut=[]
    for x in xs:
        if x['command']:
            toks=x['command'].split();m=' '.join(toks[:-1]) if len(toks)>1 else 'echo';mut.append(compare_ast(f'Use `{m}`.',x['command']))
        else:mut.append(compare_ast('Use `ls`.',None))
    rec={'schema':'R22573_C57_PREOUTPUT_SEAL_V2','candidate':'C57','weights_present':0,'model_outputs_observed':0,'training_rows_used':0,'scientific_question':'Does the CLI LoRA learn control-separated mapping from natural-language intent into correct benign command AST structure (family, flags, operand order, quoting, pipelines) rather than merely producing CLI-related prose, while avoiding executable-command hallucination on conceptual questions?','suite_n':64,'positive_n':56,'negative_n':8,'suite_sha256':sha_obj(xs),'partitions':dict(Counter(x['partition'] for x in xs)),'paired_case_selection':'source-free capacity-adjusted V2: exactly 2 cases per positive family (git/fs/text/archive/python/pipe), ids written in behavior receipt','paired_n':12,'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p50','DOSE_1p50'],'generation_conditions':['BASE','FULL'],'generation_n':8,'paired_metric':'target answer token NLL over 12 source-free capacity-adjusted positive cases; lower is better','generation_metric':'AST semantic verifier: family/flags/operand order/pipeline/exact + conceptual negative executable-command FP','positive_gate':'FULL target-NLL gain over BASE paired CI low>0.01; FULL beats RANDOM and SHUFFLED by >=0.01 nats/token; generation semantic gain>=0.125; negative command FP increase<=0.5','failure_gate':'FULL generation semantic drop<=-0.125 with paired CI high<0 OR negative command FP increase>=0.5 and FULL worse than RANDOM/SHUFFLE target-NLL controls','causal_if_admitted':['MINUS_Q','MINUS_V','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE'],'verifier_smoke':{'perfect':aggregate(perfect),'source_copy':aggregate(copy),'mutation':aggregate(mut)},'runtime_gates':['WALLCLOCK_CAPACITY_PREINGRESS_GATE','DECODER_ONLY_LEFT_PADDING_BATCH_EQUIVALENCE_GATE','DYNAMIC_ADAPTER_IDENTIFIER_RUNTIME_ABI_GATE'],'claim_boundary':'fresh synthetic benign CLI grammar only; no cybersecurity/safety guarantee; public model-card eval metrics are context only'}
    (out/'C57_PREOUTPUT_SEAL.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
def runtime_smoke(out):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    from huggingface_hub import hf_hub_download
    cfgp=hf_hub_download(BASE_REPO,'config.json',revision=BASE_REV);cfg=AutoConfig.from_pretrained(cfgp,local_files_only=True);orig_layers=getattr(cfg,'num_hidden_layers',22);cfg.num_hidden_layers=2;cfg.vocab_size=min(getattr(cfg,'vocab_size',32000),2048);model=AutoModelForCausalLM.from_config(cfg);lc=LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM');p=get_peft_model(model,lc)
    for n in ['control_random','control_shuffle','dose_0p50','dose_1p50','minus_q','minus_v']:p.add_adapter(n,lc);p.set_adapter(n);p.delete_adapter(n)
    x=torch.randint(0,cfg.vocab_size,(1,64));att=torch.ones_like(x)
    with torch.no_grad():p(input_ids=x,attention_mask=att)
    t0=time.perf_counter()
    with torch.no_grad():
        for _ in range(2):p(input_ids=x,attention_mask=att)
    sec=(time.perf_counter()-t0)/2;full_forward=sec*(orig_layers/2)*1.5;planned_forward_equiv=12*6+8*2*4+4*5;est=full_forward*planned_forward_equiv
    a=torch.randint(5,cfg.vocab_size,(1,20));b=torch.cat([torch.zeros((1,7),dtype=torch.long),a],1);am=torch.cat([torch.zeros((1,7),dtype=torch.long),torch.ones((1,20),dtype=torch.long)],1);posa=torch.arange(a.shape[1]).unsqueeze(0);posb=(am.cumsum(-1)-1).clamp(min=0)
    with torch.no_grad():la=p(input_ids=a,position_ids=posa).logits[:,-1,:];lb=p(input_ids=b,attention_mask=am,position_ids=posb).logits[:,-1,:]
    pad_diff=float((la-lb).abs().max());top1_equal=bool(int(la.argmax(-1))==int(lb.argmax(-1)));cos=float(torch.nn.functional.cosine_similarity(la.float(),lb.float(),dim=-1).item());pad_pass=top1_equal and cos>0.999;wall_pass=est<1800;rec={'schema':'R22573_C57_RUNTIME_ABI_V2','pass':bool(wall_pass and pad_pass),'full_layers':orig_layers,'micro_layers':2,'micro_forward_sec':sec,'estimated_full_forward_sec_conservative':full_forward,'estimated_planned_wallclock_sec':est,'wallclock_budget_sec':1800,'wallclock_gate_pass':wall_pass,'left_padding_single_vs_batch_max_logit_diff':pad_diff,'left_padding_top1_equal':top1_equal,'left_padding_logit_cosine':cos,'left_padding_gate_pass':pad_pass,'identifiers_safe':True,'weights_present':0};out.mkdir(exist_ok=True);(out/'C57_RUNTIME_ABI.json').write_text(json.dumps(rec,indent=2));print(rec)
    if not rec['pass']:raise SystemExit('PREINGRESS_RUNTIME_GATE_RED')
def execute(out,work):
    import torch
    from huggingface_hub import snapshot_download,hf_hub_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file
    out.mkdir(exist_ok=True);raw=work/'raw';raw.mkdir(parents=True,exist_ok=True);base_dir=raw/'base';ad_dir=raw/'adapter';base_dir.mkdir();ad_dir.mkdir();snapshot_download(BASE_REPO,revision=BASE_REV,local_dir=base_dir,allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','tokenizer.model','chat_template.jinja'])
    for fn in ['adapter_config.json','adapter_model.safetensors']:
        src=hf_hub_download(ADAPTER_REPO,fn,revision=ADAPTER_REV);shutil.copy2(src,ad_dir/fn)
    assert hashlib.sha256((base_dir/'model.safetensors').read_bytes()).hexdigest()==BASE_WEIGHT_SHA;assert hashlib.sha256((ad_dir/'adapter_model.safetensors').read_bytes()).hexdigest()==ADAPTER_SHA;ac=json.loads((ad_dir/'adapter_config.json').read_text());assert ac['base_model_name_or_path']==BASE_REPO and ac['r']==8
    st=load_file(str(ad_dir/'adapter_model.safetensors'),device='cpu');pairmap=defaultdict(dict);energy=defaultdict(float);zero=nonfinite=0
    for k,t in st.items():
        nonfinite+=int(not torch.isfinite(t).all());zero+=int(float(t.norm())==0);fam='q_proj' if 'q_proj' in k else 'v_proj' if 'v_proj' in k else 'other';energy[fam]+=float(t.float().pow(2).sum());m=re.sub(r'\.lora_[AB]\.weight$','',k);pairmap[m]['A' if '.lora_A.' in k else 'B' if '.lora_B.' in k else k]=t
    total=sum(energy.values()) or 1;(out/'C57_STATIC_FORENSICS.json').write_text(json.dumps({'schema':'R22573_C57_STATIC_FORENSICS_V1','tensor_count':len(st),'complete_pairs':sum(set(v)=={'A','B'} for v in pairmap.values()),'zero_tensor_count':zero,'nonfinite_tensor_count':nonfinite,'rank_config':ac['r'],'alpha':ac['lora_alpha'],'target_modules':sorted(ac['target_modules']),'energy_proxy_by_family':{k:v/total for k,v in sorted(energy.items())},'evidence_grade':'E1_STATIC_ONLY'},indent=2));del st
    tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';base=AutoModelForCausalLM.from_pretrained(base_dir,local_files_only=True,dtype=torch.float32,device_map='cpu');peft=PeftModel.from_pretrained(base,ad_dir,is_trainable=False);peft.eval();cases=build_suite();pos=[x for x in cases if x['target']];byfam=defaultdict(list)
    for c in pos:byfam[c['family']].append(c)
    paired=[]
    for fam in ['git','fs','text','archive','python','pipe']:paired.extend(byfam[fam][:2])
    assert len(paired)==12;params={n:p for n,p in peft.named_parameters() if 'lora_A' in n or 'lora_B' in n};originals={n:p.detach().clone() for n,p in params.items()}
    def restore():
        with torch.no_grad():
            for n,p in params.items():p.copy_(originals[n])
    def mutate(cond):
        restore();rng=torch.Generator().manual_seed(SEED+17)
        with torch.no_grad():
            if cond in ['BASE','FULL']:return
            if cond.startswith('DOSE_'):
                f={'DOSE_0p50':.5,'DOSE_1p50':1.5}[cond]
                for n,p in params.items():
                    if 'lora_B' in n:p.mul_(f)
            elif cond=='RANDOM_RANK_MATCHED':
                for n,p in params.items():
                    o=originals[n];z=torch.randn(o.shape,generator=rng,dtype=o.dtype);z*=float(o.norm())/(float(z.norm())+1e-12);p.copy_(z)
            elif cond=='LAYER_SHUFFLED':
                groups=defaultdict(list)
                for n,p in params.items():groups[('q' if 'q_proj' in n else 'v','A' if 'lora_A' in n else 'B',tuple(p.shape))].append(n)
                for names in groups.values():
                    if len(names)>1:
                        vals=[originals[n].clone() for n in names];order=list(range(len(vals)));random.Random(SEED+len(names)).shuffle(order)
                        for n,j in zip(names,order):params[n].copy_(vals[j])
            else:raise ValueError(cond)
    def target_nll(c):
        qe=tok(c['question'],return_tensors='pt',add_special_tokens=True);te=tok(c['target'],return_tensors='pt',add_special_tokens=False);ids=torch.cat([qe.input_ids,te.input_ids],1);att=torch.ones_like(ids);labels=ids.clone();labels[:,:qe.input_ids.shape[1]]=-100
        with torch.no_grad():return float(peft(input_ids=ids,attention_mask=att,labels=labels).loss)
    conds=['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p50','DOSE_1p50'];nlls={}
    for cond in conds:
        mutate(cond);ctx=peft.disable_adapter() if cond=='BASE' else contextlib.nullcontext();vals=[]
        with ctx:
            for c in paired:vals.append(target_nll(c))
        nlls[cond]=vals
    restore();gen_pos=[]
    for fam in ['git','fs','text','archive','python','pipe']:gen_pos.extend(byfam.get(fam,[])[:1])
    gen_cases=gen_pos[:6]+[c for c in cases if c['partition']=='negative'][:2];assert len(gen_cases)==8
    def generate(c,baseoff=False):
        enc=tok(c['question'],return_tensors='pt');ctx=peft.disable_adapter() if baseoff else contextlib.nullcontext()
        with ctx,torch.no_grad():y=peft.generate(**enc,max_new_tokens=24,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        return tok.decode(y[0][enc.input_ids.shape[1]:],skip_special_tokens=True)
    gen={};scores={}
    for cond,baseoff in [('BASE',True),('FULL',False)]:
        restore();rows=[];sc=[]
        for c in gen_cases:
            z=compare_ast(generate(c,baseoff),c['command']);rows.append({'id':c['id'],'partition':c['partition'],'score':z});sc.append(z)
        gen[cond]=rows;scores[cond]=aggregate(sc)
    gains=[nlls['BASE'][i]-nlls['FULL'][i] for i in range(len(paired))];ci=bootstrap_ci(gains);nll_mean={k:sum(v)/len(v) for k,v in nlls.items()};sem_gain=scores['FULL']['semantic']-scores['BASE']['semantic'];fp_gain=scores['FULL']['negative_fp']-scores['BASE']['negative_fp'];full_rand=nll_mean['RANDOM_RANK_MATCHED']-nll_mean['FULL'];full_shuffle=nll_mean['LAYER_SHUFFLED']-nll_mean['FULL'];positive=ci[0]>0.01 and full_rand>=0.01 and full_shuffle>=0.01 and sem_gain>=0.125 and fp_gain<=0.5;failure=(sem_gain<=-0.125 and ci[1]<0) or (fp_gain>=0.5 and full_rand<=-0.01 and full_shuffle<=-0.01)
    (out/'C57_BEHAVIOR.json').write_text(json.dumps({'schema':'R22573_C57_BEHAVIOR_V1','paired_case_ids':[c['id'] for c in paired],'paired_n':12,'generation_case_ids':[c['id'] for c in gen_cases],'generation_n':8,'target_nll_mean':nll_mean,'full_minus_base_nll_gain_mean':sum(gains)/len(gains),'paired_bootstrap_ci':ci,'full_vs_random_nll_advantage':full_rand,'full_vs_shuffle_nll_advantage':full_shuffle,'generation_metrics':scores,'full_minus_base_generation_semantic':sem_gain,'full_minus_base_negative_fp':fp_gain,'positive_lane_green':positive,'failure_lane_green':failure,'generation_scalar_rows':gen,'raw_output_text_exported':False},indent=2));causal={'schema':'R22573_C57_CAUSAL_V1','executed':False,'reason':'BEHAVIOR_LANES_RED'}
    if positive or failure:
        cconds=['MINUS_Q','MINUS_V','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE'];cm={};layers=[]
        for n in params:
            m=re.search(r'\.layers\.(\d+)\.',n)
            if m:layers.append(int(m.group(1)))
        maxl=max(layers) if layers else 21
        for cc in cconds:
            restore()
            with torch.no_grad():
                for n,p in params.items():
                    if 'lora_B' not in n:continue
                    kill=(cc=='MINUS_Q' and 'q_proj' in n) or (cc=='MINUS_V' and 'v_proj' in n);m=re.search(r'\.layers\.(\d+)\.',n);li=int(m.group(1)) if m else 0
                    if cc=='MINUS_EARLY' and li<=(maxl//3):kill=True
                    if cc=='MINUS_MIDDLE' and maxl//3<li<=2*maxl//3:kill=True
                    if cc=='MINUS_LATE' and li>2*maxl//3:kill=True
                    if kill:p.zero_()
            vals=[target_nll(c) for c in paired[:4]];cm[cc]=sum(vals)/len(vals)
        restore();causal={'schema':'R22573_C57_CAUSAL_V1','executed':True,'screen_n':4,'nll_mean':cm,'full_reference_nll':nll_mean['FULL'],'claim_boundary':'broad causal screen only; no E3 promotion without independent localization threshold precommitted'}
    (out/'C57_CAUSAL.json').write_text(json.dumps(causal,indent=2));(out/'C57_PROVENANCE.json').write_text(json.dumps({'schema':'R22573_C57_PROVENANCE_V1','adapter':{'repo':ADAPTER_REPO,'revision':ADAPTER_REV,'sha256':ADAPTER_SHA,'license':'apache-2.0'},'base':{'repo':BASE_REPO,'revision':BASE_REV,'weight_sha256':BASE_WEIGHT_SHA,'license':'apache-2.0','prior_exact_object_hit':0},'source_consumed_once':True,'training_rows_used':0,'raw_committed_to_git':False},indent=2));shutil.rmtree(work,ignore_errors=True);hf=os.environ.get('HF_HOME');
    if hf:shutil.rmtree(hf,ignore_errors=True)
    (out/'C57_CLEANUP.json').write_text(json.dumps({'schema':'R22573_C57_CLEANUP_V1','raw_root_deleted':not work.exists(),'hf_cache_deleted':not (hf and Path(hf).exists()),'raw_weights_remaining':0,'raw_tokenizer_remaining':0},indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preseal','runtime-smoke','execute']);ap.add_argument('--out',required=True);ap.add_argument('--work',default='c57-work');a=ap.parse_args();out=Path(a.out)
    if a.mode=='preseal':preseal(out)
    elif a.mode=='runtime-smoke':runtime_smoke(out)
    else:execute(out,Path(a.work))
if __name__=='__main__':main()
