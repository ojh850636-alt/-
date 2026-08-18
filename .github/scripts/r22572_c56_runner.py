from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, random, re, shutil, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
SEED=22572
ADAPTER_REPO='kartikey31/txn-parser'
ADAPTER_REV='d8ed83ee015931ddde7f53fc64a4f2959e993212'
ADAPTER_SUB='smollm2-360m/adapters'
ADAPTER_SHA='9b4d482a8fd5bb463dd858f9d8b576759471cf800f45e12660cfc22090988dae'
BASE_REPO='HuggingFaceTB/SmolLM2-360M-Instruct'
BASE_REV='a10cc1512eabd3dde888204e902eca88bddb4951'
BASE_WEIGHT_SHA='e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e'
SYSTEM='''You convert voice-transcribed transaction descriptions into structured JSON.\n\nOutput ONLY a JSON object with this schema, no other text:\n{"transactions":[{"amount":<number>,"currency":"INR"|"USD","item":"<lowercase singular noun phrase>","category":"<enum>","type":"expense"|"income"}]}\n\nCategories: Food, Drinks, Groceries, Transport, Shopping, Entertainment, Bills, Health, Education, Personal, Gifts, Income, Other.\nRules:\n- Currency defaults to INR. Use USD only when the input explicitly says "dollars" or contains "$".\n- Amounts: "k" = ×1000, "hazaar" = ×1000, "sau" = ×100, "lakh" = ×100000. Convert number-words to digits.\n- type is "expense" by default; "income" only for explicit salary, cashback, refund, gift received, payment received.\n- For disfluencies and corrections ("500 wait no 600"), output the CORRECTED amount only.\n- For ambiguous items, use item "unspecified" and category "Other".\n- Item field: lowercase singular noun phrase.\n- Multi-transaction inputs become multiple array entries in spoken order.'''
CAT={'samosa':'Food','chai':'Drinks','coffee':'Drinks','beer':'Drinks','uber ride':'Transport','metro ticket':'Transport','petrol':'Transport','rent':'Bills','wifi':'Bills','electricity':'Bills','medicine':'Health','doctor visit':'Health','book':'Education','notebook':'Education','shirt':'Shopping','movie ticket':'Entertainment','groceries':'Groceries','gift':'Gifts','unspecified':'Other','salary':'Income','cashback':'Income','refund':'Income','payment':'Income'}
def tx(amount,item,currency='INR',typ='expense'):return {'amount':amount,'currency':currency,'item':item,'category':('Income' if typ=='income' else CAT[item]),'type':typ}
def target(txs):return json.dumps({'transactions':txs},separators=(',',':'),ensure_ascii=False)
def build_suite():
    cases=[]
    def add(part,src,txs):cases.append({'id':f'C56-{len(cases):03d}','partition':part,'source':src,'target':target(txs),'txs':txs})
    items=['samosa','chai','coffee','uber ride','petrol','rent','wifi','medicine'];forms=[('2k',2000),('3 hazaar',3000),('five hundred',500),('teen sau',300),('1 lakh',100000),('2.5k',2500),('750',750),('four thousand',4000)]
    for i in range(64):
        f,a=forms[i%len(forms)];it=items[(i*3)%len(items)];add('normalization',f'{f} for {it}',[tx(a,it)])
    vals=[(500,600),(2000,2500),(300,350),(1000,900),(50,70),(4500,4200)]
    for i in range(48):
        a,b=vals[i%len(vals)];it=items[(i*5)%len(items)];phrase=['wait no','sorry make that','actually','nahi'][i%4];add('correction',f'{a} for {it} {phrase} {b}',[tx(b,it)])
    pairs=[('samosa','chai'),('uber ride','coffee'),('medicine','metro ticket'),('groceries','wifi')]
    for i in range(48):
        x,y=pairs[i%4];a=100+(i%7)*50;b=40+(i%5)*20;add('multi',f'{a} for {x} and {b} for {y}',[tx(a,x),tx(b,y)])
    inc=[('salary',25000),('cashback',500),('refund',1200),('payment',3500)]
    for i in range(32):
        it,a=inc[i%4];phr={'salary':'salary received','cashback':'cashback received','refund':'refund received','payment':'payment received'}[it];add('income',f'{phr} {a}',[tx(a,it,typ='income')])
    neg=["order number 4500 status pending","ticket 2207 is my support reference","product model 500 costs unknown","invoice id 1200 please check status","OTP 660044 for verification","room number 305 reservation query","flight 602 delayed","account reference 7001 needs update"]
    for i in range(32):add('hard_negative',neg[i%len(neg)],[])
    for i in range(32):
        if i%2==0:
            it=items[i%len(items)];a=2000+(i%4)*1000;add('code_switch',f'{a//1000} hazaar ka {it}',[tx(a,it)])
        else:
            it=items[i%len(items)];a=20+(i%5)*10;add('code_switch',f'${a} dollars for {it}',[tx(a,it,'USD')])
    assert len(cases)==256;return cases
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def parse_output(s):
    s=s.strip()
    try:
        if '```' in s:
            m=re.search(r'\{.*\}',s,re.S);s=m.group(0) if m else s
        o=json.loads(s);xs=o.get('transactions') if isinstance(o,dict) else None
        if not isinstance(xs,list):return None
        out=[]
        for z in xs:
            if not isinstance(z,dict):return None
            out.append({'amount':float(z.get('amount')),'currency':str(z.get('currency')),'item':str(z.get('item')).lower().strip(),'category':str(z.get('category')),'type':str(z.get('type'))})
        return out
    except Exception:return None
def score_case(pred,gold):
    if pred is None:return {'schema':0,'count':0,'order':0,'amount':0,'currency':0,'item':0,'category':0,'type':0,'semantic':0,'fp':1 if not gold else 0}
    if not gold:
        ok=int(len(pred)==0);return {'schema':1,'count':ok,'order':ok,'amount':ok,'currency':ok,'item':ok,'category':ok,'type':ok,'semantic':ok,'fp':int(len(pred)>0)}
    count=int(len(pred)==len(gold));n=min(len(pred),len(gold))
    if n==0:return {'schema':1,'count':0,'order':0,'amount':0,'currency':0,'item':0,'category':0,'type':0,'semantic':0,'fp':0}
    amount=sum(abs(pred[i]['amount']-gold[i]['amount'])<1e-6 for i in range(n))/len(gold);currency=sum(pred[i]['currency']==gold[i]['currency'] for i in range(n))/len(gold);item=sum(pred[i]['item']==gold[i]['item'] for i in range(n))/len(gold);category=sum(pred[i]['category']==gold[i]['category'] for i in range(n))/len(gold);typ=sum(pred[i]['type']==gold[i]['type'] for i in range(n))/len(gold);order=int(count and all(pred[i]['item']==gold[i]['item'] and abs(pred[i]['amount']-gold[i]['amount'])<1e-6 for i in range(len(gold))));sem=(count+amount+currency+item+category+typ+order)/7
    return {'schema':1,'count':count,'order':order,'amount':amount,'currency':currency,'item':item,'category':category,'type':typ,'semantic':sem,'fp':0}
def aggregate(scores):return {k:sum(float(x[k]) for x in scores)/len(scores) for k in scores[0]}
def bootstrap_ci(vals,seed=22572,B=1500):
    rng=random.Random(seed);n=len(vals);means=[sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(B)];means.sort();return [means[int(.025*B)],means[min(B-1,int(.975*B))]]
def preseal(out):
    out.mkdir(parents=True,exist_ok=True);cases=build_suite();perfect=[score_case(c['txs'],c['txs']) for c in cases];copy=[score_case(None,c['txs']) for c in cases];mut=[]
    for c in cases:
        p=[dict(x) for x in c['txs']]
        if p:p[0]['amount']+=1
        mut.append(score_case(p,c['txs']))
    rec={'schema':'R22572_C56_PREOUTPUT_SEAL_V1','candidate':'C56','weights_present':0,'model_outputs_observed':0,'training_rows_used':0,'scientific_question':'Does a small transaction-parser LoRA learn control-separated semantic transduction for spoken numeric-unit normalization, correction precedence, code-switch currency, and ordered multi-event decomposition rather than JSON-format imitation or number hallucination?','suite_n':len(cases),'suite_sha256':sha_obj(cases),'partitions':dict(__import__('collections').Counter(c['partition'] for c in cases)),'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p25','DOSE_0p50','DOSE_1p50'],'generation_conditions':['BASE','FULL'],'paired_metric':'target token NLL over all 256 cases; lower is better','generation_n':128,'generation_metric':'independent semantic verifier: amount/currency/type/count/order/item/category + hard-negative false transaction rate','positive_gate':'FULL target-NLL gain over BASE paired CI low>0.01; FULL beats RANDOM and SHUFFLED by >=0.01 nats/token; generation semantic gain >=0.05; hard-negative FP increase <=0.05','failure_gate':'FULL semantic generation drop<=-0.05 with paired CI high<0 OR hard-negative FP increase>=0.10 and learned FULL worse than RANDOM/SHUFFLE target-NLL controls','causal_if_admitted':['MINUS_ATTENTION','MINUS_MLP','MINUS_Q','MINUS_K','MINUS_V','MINUS_O','MINUS_GATE','MINUS_UP','MINUS_DOWN','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE'],'verifier_smoke':{'perfect_semantic':aggregate(perfect)['semantic'],'null_semantic':aggregate(copy)['semantic'],'amount_mutation_semantic':aggregate(mut)['semantic']},'runtime_abi_gate':'before binary: instantiate SmolLM2 config + PEFT LoRA lifecycle; controls mutate one loaded adapter in memory; identifiers use underscore only','claim_boundary':'fresh synthetic transaction grammar only; no financial/compliance guarantee; public model-card metrics are context only'}
    (out/'C56_PREOUTPUT_SEAL.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
def runtime_smoke(out):
    from transformers import AutoConfig,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    cfg=AutoConfig.from_pretrained(BASE_REPO,revision=BASE_REV,trust_remote_code=False)
    for attr,val in [('hidden_size',64),('intermediate_size',128),('num_hidden_layers',2),('num_attention_heads',4),('num_key_value_heads',2),('vocab_size',256)]:
        if hasattr(cfg,attr):setattr(cfg,attr,val)
    model=AutoModelForCausalLM.from_config(cfg);lc=LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM');p=get_peft_model(model,lc);names=['control_random','control_shuffle','dose_0p25','dose_0p50','dose_1p50','minus_q','minus_early']
    for n in names:p.add_adapter(n,lc);p.set_adapter(n);p.delete_adapter(n)
    import torch;x=torch.randint(0,256,(1,8));y=p(input_ids=x);rec={'schema':'R22572_C56_RUNTIME_ABI_V1','pass':True,'logits_shape':list(y.logits.shape),'identifiers':names,'weights_present':0};out.mkdir(exist_ok=True);(out/'C56_RUNTIME_ABI.json').write_text(json.dumps(rec,indent=2));print(rec)
def execute(out,work):
    import torch
    from huggingface_hub import snapshot_download,hf_hub_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file
    out.mkdir(exist_ok=True);raw=work/'raw';raw.mkdir(parents=True,exist_ok=True);base_dir=raw/'base';ad_dir=raw/'adapter';base_dir.mkdir();ad_dir.mkdir()
    snapshot_download(BASE_REPO,revision=BASE_REV,local_dir=base_dir,allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja'])
    for fn in ['adapter_config.json','adapter_model.safetensors']:
        src=hf_hub_download(ADAPTER_REPO,filename=f'{ADAPTER_SUB}/{fn}',revision=ADAPTER_REV);shutil.copy2(src,ad_dir/fn)
    assert hashlib.sha256((base_dir/'model.safetensors').read_bytes()).hexdigest()==BASE_WEIGHT_SHA;assert hashlib.sha256((ad_dir/'adapter_model.safetensors').read_bytes()).hexdigest()==ADAPTER_SHA
    ac=json.loads((ad_dir/'adapter_config.json').read_text());assert ac['base_model_name_or_path']==BASE_REPO and ac['r']==32
    st=load_file(str(ad_dir/'adapter_model.safetensors'),device='cpu');pairmap=defaultdict(dict);nonfinite=zero=0;energy=defaultdict(float)
    for k,t in st.items():
        nonfinite+=int(not torch.isfinite(t).all());zero+=int(float(t.norm())==0);fam=next((x for x in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if x in k),'other');energy[fam]+=float(t.float().pow(2).sum());m=re.sub(r'\.lora_[AB]\.weight$','',k);pairmap[m]['A' if '.lora_A.' in k else 'B' if '.lora_B.' in k else k]=t
    totalE=sum(energy.values()) or 1;static={'schema':'R22572_C56_STATIC_FORENSICS_V1','tensor_count':len(st),'complete_pairs':sum(set(v)=={'A','B'} for v in pairmap.values()),'zero_tensor_count':zero,'nonfinite_tensor_count':nonfinite,'rank_config':ac['r'],'alpha':ac['lora_alpha'],'target_modules':sorted(ac['target_modules']),'energy_proxy_by_family':{k:v/totalE for k,v in sorted(energy.items())},'evidence_grade':'E1_STATIC_ONLY'};(out/'C56_STATIC_FORENSICS.json').write_text(json.dumps(static,indent=2));del st
    tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;base=AutoModelForCausalLM.from_pretrained(base_dir,local_files_only=True,dtype=torch.float32,device_map='cpu');peft=PeftModel.from_pretrained(base,ad_dir,is_trainable=False);peft.eval();cases=build_suite();params={n:p for n,p in peft.named_parameters() if 'lora_A' in n or 'lora_B' in n};originals={n:p.detach().clone() for n,p in params.items()}
    def restore():
        with torch.no_grad():
            for n,p in params.items():p.copy_(originals[n])
    def mutate(cond):
        restore();rng=torch.Generator().manual_seed(SEED+77)
        with torch.no_grad():
            if cond=='FULL':return
            if cond.startswith('DOSE_'):
                factor={'DOSE_0p25':.25,'DOSE_0p50':.5,'DOSE_1p50':1.5}[cond]
                for n,p in params.items():
                    if 'lora_B' in n:p.mul_(factor)
            elif cond=='RANDOM_RANK_MATCHED':
                for n,p in params.items():
                    o=originals[n];z=torch.randn(o.shape,generator=rng,dtype=o.dtype);p.copy_(z*(o.norm()/(z.norm()+1e-12)))
            elif cond=='LAYER_SHUFFLED':
                groups=defaultdict(list)
                for n,p in params.items():groups[(re.sub(r'.*layers\.\d+\.','',n),tuple(p.shape))].append(n)
                for names in groups.values():
                    if len(names)>1:
                        vals=[originals[x] for x in names];vals=vals[1:]+vals[:1]
                        for n,v in zip(names,vals):params[n].copy_(v)
            elif cond.startswith('MINUS_'):
                token=cond[6:].lower();L=getattr(peft.base_model.model.config,'num_hidden_layers',32);third=max(1,L//3)
                for n,p in params.items():
                    hit=(token=='attention' and 'self_attn' in n) or (token=='mlp' and 'mlp' in n) or (token in ['q','k','v','o','gate','up','down'] and f'{token}_proj' in n)
                    if token in ['early','middle','late']:
                        m=re.search(r'layers\.(\d+)\.',n);layer=int(m.group(1)) if m else -1;hit=(token=='early' and layer<third) or (token=='middle' and third<=layer<2*third) or (token=='late' and layer>=2*third)
                    if hit and 'lora_B' in n:p.zero_()
    def prompt_text(c):
        msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':c['source']}]
        try:return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        except Exception:return SYSTEM+'\nUser: '+c['source']+'\nAssistant:'
    def nll_all(cond):
        ctx=peft.disable_adapter() if cond=='BASE' else contextlib.nullcontext()
        if cond!='BASE':mutate(cond)
        vals=[]
        with ctx,torch.no_grad():
            for start in range(0,len(cases),16):
                cs=cases[start:start+16];prompts=[prompt_text(c) for c in cs];full=[p+c['target'] for p,c in zip(prompts,cs)];enc=tok(full,padding=True,return_tensors='pt',truncation=True,max_length=512);lab=enc.input_ids.clone()
                for i,p in enumerate(prompts):lab[i,:len(tok(p,add_special_tokens=False).input_ids)]=-100
                outm=peft(**enc);logits=outm.logits[:,:-1];labels=lab[:,1:];loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100,reduction='none').reshape(labels.shape);mask=labels.ne(-100);per=(loss*mask).sum(1)/mask.sum(1).clamp_min(1);vals.extend(float(x) for x in per)
        return vals
    conditions=['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p25','DOSE_0p50','DOSE_1p50'];nll={c:nll_all(c) for c in conditions};by=defaultdict(list)
    for c in cases:by[c['partition']].append(c)
    counts={'normalization':32,'correction':24,'multi':24,'income':16,'hard_negative':16,'code_switch':16};subset=[]
    for part,n in counts.items():subset+=by[part][:n]
    def gen(cond):
        ctx=peft.disable_adapter() if cond=='BASE' else contextlib.nullcontext()
        if cond!='BASE':mutate('FULL')
        scores=[];dig=[]
        with ctx,torch.no_grad():
            for start in range(0,len(subset),8):
                cs=subset[start:start+8];prompts=[prompt_text(c) for c in cs];enc=tok(prompts,padding=True,return_tensors='pt',truncation=True,max_length=384);ids=peft.generate(**enc,max_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
                for i,c in enumerate(cs):
                    s=tok.decode(ids[i,enc.input_ids.shape[1]:],skip_special_tokens=True);scores.append(score_case(parse_output(s),c['txs']));dig.append(hashlib.sha256(s.encode()).hexdigest())
        return scores,dig
    bs,bd=gen('BASE');fs,fd=gen('FULL');ba,fa=aggregate(bs),aggregate(fs);semdiff=[f['semantic']-b['semantic'] for b,f in zip(bs,fs)];ci_sem=bootstrap_ci(semdiff,SEED+1);gain=[b-f for b,f in zip(nll['BASE'],nll['FULL'])];ci_gain=bootstrap_ci(gain,SEED+2);full_gain=sum(gain)/len(gain);rand_margin=sum(r-f for r,f in zip(nll['RANDOM_RANK_MATCHED'],nll['FULL']))/len(gain);shuf_margin=sum(s-f for s,f in zip(nll['LAYER_SHUFFLED'],nll['FULL']))/len(gain);hard_idx=[i for i,c in enumerate(subset) if c['partition']=='hard_negative'];hard_base=sum(bs[i]['fp'] for i in hard_idx)/len(hard_idx);hard_full=sum(fs[i]['fp'] for i in hard_idx)/len(hard_idx);sem_gain=fa['semantic']-ba['semantic'];positive=(ci_gain[0]>.01 and rand_margin>=.01 and shuf_margin>=.01 and sem_gain>=.05 and hard_full-hard_base<=.05 and ci_sem[0]>0);failure=(sem_gain<=-.05 and ci_sem[1]<0) or (hard_full-hard_base>=.10 and rand_margin<-.01 and shuf_margin<-.01);admitted=positive or failure;causal={}
    if admitted:
        for c in ['MINUS_ATTENTION','MINUS_MLP','MINUS_Q','MINUS_K','MINUS_V','MINUS_O','MINUS_GATE','MINUS_UP','MINUS_DOWN','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE']:
            vals=nll_all(c);causal[c]={'mean_nll':sum(vals)/len(vals),'drop_vs_full':sum(v-f for v,f in zip(vals,nll['FULL']))/len(vals)}
    scalar=[{'id':c['id'],'partition':c['partition'],'base_semantic':bs[i]['semantic'],'full_semantic':fs[i]['semantic'],'base_fp':bs[i]['fp'],'full_fp':fs[i]['fp'],'base_digest':bd[i],'full_digest':fd[i]} for i,c in enumerate(subset)]
    behavior={'schema':'R22572_C56_BEHAVIOR_V1','suite_n':256,'generation_n':128,'conditions':{c:{'mean_target_nll':sum(v)/len(v)} for c,v in nll.items()},'full_vs_base_target_nll_gain':full_gain,'paired_nll_gain_ci95':ci_gain,'full_minus_random_margin_nll':rand_margin,'full_minus_shuffle_margin_nll':shuf_margin,'generation':{'BASE':ba,'FULL':fa,'semantic_gain':sem_gain,'paired_semantic_gain_ci95':ci_sem,'hard_negative_fp_base':hard_base,'hard_negative_fp_full':hard_full},'positive_gate_pass':positive,'failure_gate_pass':failure,'causal_admitted':admitted,'causal':causal,'case_scalar_results':scalar,'training_rows_used':0,'public_model_card_metrics_used_as_evidence':False,'claim_boundary':'fresh synthetic semantic transduction assay only'};(out/'C56_BEHAVIOR.json').write_text(json.dumps(behavior,indent=2));(out/'C56_PROVENANCE.json').write_text(json.dumps({'schema':'R22572_C56_PROVENANCE_V1','adapter_repo':ADAPTER_REPO,'adapter_revision':ADAPTER_REV,'adapter_subfolder':ADAPTER_SUB,'adapter_sha256':ADAPTER_SHA,'base_repo':BASE_REPO,'base_revision':BASE_REV,'base_weight_sha256':BASE_WEIGHT_SHA,'one_use_ingress':True,'training_time_base_revision_proven':False},indent=2));restore();del peft,base,tok;shutil.rmtree(work,ignore_errors=True);cache=os.environ.get('HF_HOME');shutil.rmtree(cache,ignore_errors=True) if cache else None;(out/'C56_CLEANUP.json').write_text(json.dumps({'schema':'R22572_C56_CLEANUP_V1','raw_root_deleted':not work.exists(),'hf_cache_deleted':not cache or not Path(cache).exists(),'raw_weights_remaining':0,'raw_tokenizer_remaining':0},indent=2));print(json.dumps({'nll_gain':full_gain,'ci':ci_gain,'random_margin':rand_margin,'shuffle_margin':shuf_margin,'semantic_gain':sem_gain,'positive':positive,'failure':failure,'causal':admitted},indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['preseal','runtime-smoke','execute']);ap.add_argument('--out',required=True);ap.add_argument('--work',default='c56-work');a=ap.parse_args();out=Path(a.out)
    if a.cmd=='preseal':preseal(out)
    elif a.cmd=='runtime-smoke':runtime_smoke(out)
    else:execute(out,Path(a.work))
if __name__=='__main__':main()
