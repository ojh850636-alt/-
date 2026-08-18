from __future__ import annotations
import argparse, contextlib, hashlib, json, math, random, re, shutil, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.dont_write_bytecode=True

SEED=22573
ADAPTER_REPO='JaydeepR/SmolLM-135M-CPT-LoRA-r32'
ADAPTER_REV='d06a83bcb612671c0c854d7b9aafb0403ca6b524'
ADAPTER_SHA='fbc0b8c971b6942694172b2b3804ca12fe84032cd53cfcb24c9bea658cc31317'
RUNTIME_BASE_REPO='HuggingFaceTB/SmolLM-135M'
RUNTIME_BASE_REV='1d461723eec654e65efdc40cf49301c89c0c92f4'
RUNTIME_BASE_SHA='c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093'
TRAINING_BASE_DECLARED='unsloth/smollm-135m-bnb-4bit'

def sha_obj(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def build_pairs():
    P=[]
    def add(part,prompt,true,false):
        P.append({'id':f'C57-{len(P):03d}','partition':part,'prompt':prompt,'true':true,'false':false})
    items=[
      ("Increasing the minibatch size, all else equal,", " generally reduces the variance of the stochastic gradient estimate.", " generally increases the variance of the stochastic gradient estimate."),
      ("A standard gradient-descent parameter update moves", " in the direction opposite to the gradient.", " in the same direction as the gradient."),
      ("Gradient clipping by global norm is designed to", " cap excessively large gradient magnitudes.", " amplify excessively large gradient magnitudes."),
      ("A learning rate that is far too large can", " destabilize optimization and cause divergence.", " guarantee faster convergence without instability."),
      ("Momentum in stochastic optimization is commonly used to", " smooth updates by accumulating a velocity from past gradients.", " discard all information from past gradients at every step."),
      ("Adam's first-moment estimate tracks", " an exponential moving average of gradients.", " an exponential moving average of squared gradients only."),
      ("Adam's second-moment estimate tracks", " an exponential moving average of squared gradients.", " an exponential moving average of raw gradients only."),
      ("Decoupled weight decay tends to", " shrink parameter magnitudes toward zero over updates.", " push parameter magnitudes away from zero over updates."),
      ("For a differentiable convex objective, any local minimum is", " also a global minimum.", " necessarily a local maximum."),
      ("Gradient accumulation across several microbatches can", " approximate an update using a larger effective batch.", " force the effective batch size to become smaller."),
      ("A learning-rate warmup schedule typically", " increases the learning rate gradually during early steps.", " sets the largest learning rate before the first optimization step and only increases it afterward."),
      ("Cosine learning-rate annealing commonly", " lowers the learning rate toward the end of training.", " raises the learning rate monotonically toward the end of training."),
      ("Early stopping usually monitors", " held-out validation behavior to decide when to stop.", " only the number of parameters in the model."),
      ("A zero gradient at a differentiable point means the point is", " first-order stationary.", " guaranteed to be the unique global optimum."),
      ("Newton-style optimization uses", " curvature information from a Hessian or its approximation.", " only token-frequency counts and no curvature information."),
      ("Stochastic gradient descent without momentum updates from", " a stochastic gradient estimate computed on the current batch.", " the exact inverse Hessian at every step.")
    ]
    for a,b,c in items:add('ml_core',"Machine-learning principle: "+a,b,c)
    items=[
      ("Softmax transforms logits into probabilities that", " sum to one across the normalized dimension.", " are unconstrained and need not sum to one."),
      ("Lowering a positive softmax temperature usually makes the distribution", " sharper and more concentrated on larger logits.", " flatter and more uniform across logits."),
      ("For a fixed target distribution, cross-entropy is minimized when predictions", " match the target distribution.", " systematically move away from the target distribution."),
      ("The Kullback-Leibler divergence between valid probability distributions is", " nonnegative.", " always strictly negative."),
      ("Among categorical distributions on a fixed finite support, entropy is maximized by", " the uniform distribution.", " a point mass on one category."),
      ("A sigmoid activation maps finite real inputs to values", " between zero and one.", " outside the interval from zero to one."),
      ("Log loss gives a confident wrong prediction", " a larger penalty than a mildly wrong prediction.", " a smaller penalty than a mildly wrong prediction."),
      ("Mean squared error makes a residual twice as large contribute", " four times as much squared error.", " half as much squared error."),
      ("Binary cross-entropy corresponds naturally to", " a Bernoulli likelihood model.", " a deterministic sorting algorithm."),
      ("Minimizing negative log likelihood is equivalent to", " maximizing likelihood.", " minimizing likelihood."),
      ("Perplexity is obtained by", " exponentiating an average negative log likelihood.", " negating an average classification accuracy."),
      ("Label smoothing changes a hard one-hot target into", " a softened target distribution.", " a target with probability mass greater than one."),
      ("A calibrated classifier that predicts probability 0.8 on many comparable cases should be correct on", " roughly eighty percent of those cases.", " roughly twenty percent of those cases by definition."),
      ("Precision uses true positives divided by", " true positives plus false positives.", " true positives plus false negatives."),
      ("Recall uses true positives divided by", " true positives plus false negatives.", " true positives plus false positives."),
      ("The F1 score is the", " harmonic mean of precision and recall.", " arithmetic difference between precision and recall.")
    ]
    for a,b,c in items:add('ml_core',"Machine-learning principle: "+a,b,c)
    items=[
      ("A causal attention mask prevents a token from", " attending to future tokens.", " attending to any earlier token."),
      ("Self-attention allows a token representation to", " aggregate information from other positions according to attention weights.", " use only a fixed local convolution kernel with no content-dependent weighting."),
      ("A residual connection typically", " adds a block's input to a transformed path.", " deletes the block's input before producing the output."),
      ("Layer normalization in a transformer commonly normalizes", " hidden features within a token representation.", " the number of training examples in the dataset."),
      ("A convolutional layer commonly", " shares kernel parameters across spatial positions.", " learns unrelated kernel parameters for every spatial position by definition."),
      ("A recurrent neural network carries information through", " a hidden state updated across sequence steps.", " a separate randomly reinitialized state at every sequence step."),
      ("Positional encodings provide a transformer with", " information about token order or position.", " labels for the supervised loss function."),
      ("An embedding layer maps discrete token identifiers to", " continuous vectors.", " optimizer learning rates."),
      ("Multi-head attention permits different heads to", " attend through distinct learned projection subspaces.", " share no input tokens and operate on unrelated datasets."),
      ("The standard transformer feed-forward block is applied", " independently at each sequence position after attention mixing.", " only once to the entire training dataset before tokenization."),
      ("Bidirectional encoder self-attention without a causal mask can", " use context from both earlier and later tokens.", " use only future tokens and never earlier tokens."),
      ("Rotary positional embeddings encode position by", " rotating representation components as a function of position.", " replacing all token embeddings with one constant vector."),
      ("Max pooling retains", " the maximum activation within each pooling region.", " the arithmetic mean of all activations within each pooling region."),
      ("Average pooling computes", " an average over activations in each pooling region.", " the maximum activation only."),
      ("Depthwise convolution applies", " separate spatial filters per input channel or channel group.", " one dense fully connected matrix across the entire sequence by definition."),
      ("A causal language model predicts each next token from", " preceding allowed context.", " only tokens that appear after the predicted token.")
    ]
    for a,b,c in items:add('ml_core',"Machine-learning principle: "+a,b,c)
    items=[
      ("LoRA represents an adaptation using", " low-rank factor matrices added to a frozen or mostly frozen base transformation.", " a mandatory replacement of every base weight with an unrelated dense matrix."),
      ("For LoRA rank r, the rank of the factorized update is", " at most r.", " necessarily larger than every dimension of the original weight."),
      ("Increasing LoRA rank, with dimensions fixed, generally", " increases the maximum rank available to the update.", " decreases the maximum possible rank of the update."),
      ("Parameter-efficient fine-tuning is designed to", " train far fewer new parameters than full-model fine-tuning.", " require updating every base parameter on every step by definition."),
      ("Dropout is normally", " active during training and disabled for deterministic evaluation.", " disabled during training and activated only for deterministic evaluation."),
      ("L1 regularization tends to encourage", " sparsity more strongly than L2 regularization.", " all parameters to have exactly equal magnitude."),
      ("Data augmentation creates", " transformed or perturbed training examples intended to preserve relevant labels.", " optimizer states that replace all training examples."),
      ("Weight tying means", " reusing the same parameters in multiple roles.", " forcing every layer to have a different parameter tensor."),
      ("A frozen backbone during adapter training receives", " no optimizer updates to its frozen parameters.", " larger optimizer updates than the adapter parameters by definition."),
      ("Lower-precision quantization is often used to", " reduce model memory footprint.", " increase the number of stored bits per parameter."),
      ("Knowledge distillation trains a student using", " information or targets supplied by a teacher model.", " only random labels unrelated to a teacher."),
      ("Model pruning removes", " selected weights, channels, heads, or connections.", " the need for any input data by definition."),
      ("Batch normalization at inference commonly uses", " running statistics accumulated during training.", " fresh gradient backpropagation through the entire training set."),
      ("Mixed-precision training uses", " lower precision for some operations while retaining higher-precision paths for stability.", " only arbitrary-precision arithmetic for every operation."),
      ("Gradient checkpointing saves activation memory by", " recomputing selected activations during backward computation.", " storing every intermediate activation multiple extra times."),
      ("Parameter sharing reduces", " the number of unique parameter values needed for repeated computation.", " the number of tokens in every input sequence.")
    ]
    for a,b,c in items:add('ml_core',"Machine-learning principle: "+a,b,c)
    items=[
      ("Leakage from a test set into model selection tends to", " inflate the apparent generalization estimate.", " guarantee an unbiased generalization estimate."),
      ("An ablation study assesses a component by", " removing or altering it and measuring the resulting change.", " keeping every component unchanged and reporting no comparison."),
      ("A held-out evaluation set should", " remain outside parameter fitting and tuning decisions.", " be repeatedly optimized against during training."),
      ("K-fold cross-validation", " rotates which fold serves as validation across repeated fits.", " uses the same examples as validation and training in every fold."),
      ("A nonparametric bootstrap typically resamples observations", " with replacement.", " without replacement until every observation appears exactly once."),
      ("With comparable variance and assumptions, increasing sample size usually makes a confidence interval", " narrower.", " wider without bound."),
      ("Changing a random seed can change", " a stochastic training trajectory.", " the mathematical definition of the softmax function."),
      ("Severe class imbalance can make raw accuracy", " misleading about minority-class performance.", " identical to macro-F1 by definition."),
      ("Macro-averaged F1 gives each class", " equal weight in the final class average.", " weight proportional only to its sample frequency."),
      ("Micro-averaged F1 is computed from", " globally aggregated decision counts.", " an unweighted arithmetic mean of per-class F1 values."),
      ("Out-of-distribution evaluation tests behavior when", " the evaluation distribution differs from the training distribution.", " every evaluation example is copied from the training set."),
      ("Covariate shift refers to a change in", " the input distribution while the conditional target mechanism may remain stable.", " only the spelling of optimizer variable names."),
      ("Concept drift refers to a change over time in", " the relationship between inputs and targets or outcomes.", " the byte order of saved model files only."),
      ("Overfitting is characterized by", " fitting training data well while generalization degrades.", " high training loss together with guaranteed perfect generalization."),
      ("Regularization can trade", " increased bias for reduced variance.", " infinite variance for exactly zero bias in every problem."),
      ("A counterfactual evaluation changes", " a relevant factor while holding comparison structure as constant as practical.", " every factor simultaneously so no attribution is possible.")
    ]
    for a,b,c in items:add('ml_ood',"Research-method principle: "+a,b,c)
    items=[
      ("At standard atmospheric pressure, liquid water freezes near", " zero degrees Celsius.", " one hundred degrees Celsius."),
      ("At standard atmospheric pressure, liquid water boils near", " one hundred degrees Celsius.", " zero degrees Celsius."),
      ("The Earth orbits", " the Sun.", " the Moon."),
      ("The Moon orbits", " the Earth.", " the Sun as its primary direct orbital parent."),
      ("Copper is generally", " a good electrical conductor.", " a perfect electrical insulator."),
      ("Visible light travels faster in vacuum than", " sound travels in air.", " any electromagnetic wave travels in vacuum."),
      ("A triangle in Euclidean geometry has interior angles summing to", " one hundred eighty degrees.", " three hundred sixty degrees."),
      ("An even integer is divisible by", " two.", " three by definition."),
      ("A kilogram is a unit of", " mass.", " electric current."),
      ("A meter is a unit of", " length.", " temperature."),
      ("DNA commonly stores biological hereditary information using", " nucleotide sequences.", " acoustic pressure waves."),
      ("Photosynthesis converts light energy into", " chemical energy stored in molecules.", " gravitational potential by orbital motion."),
      ("A compass needle aligns approximately with", " the local magnetic field.", " the local sound-pressure field."),
      ("The freezing of liquid water is", " a phase transition from liquid to solid.", " a transition from solid to plasma."),
      ("In ordinary decimal notation, ten multiplied by ten equals", " one hundred.", " one thousand."),
      ("A standard hour contains", " sixty minutes.", " one hundred minutes.")
    ]
    for a,b,c in items:add('general_control',"General-knowledge principle: "+a,b,c)
    assert len(P)==96
    return P

def bootstrap_ci(vals,B=1500,seed=SEED):
    rng=random.Random(seed);n=len(vals);xs=[sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(B)];xs.sort();return [xs[int(.025*B)],xs[min(B-1,int(.975*B))]]

def tier_ids(pairs,name):
    q={'LARGE':{'ml_core':64,'ml_ood':16,'general_control':16},'MEDIUM':{'ml_core':48,'ml_ood':12,'general_control':12},'SMALL':{'ml_core':32,'ml_ood':8,'general_control':8}}[name];out=[]
    for p,n in q.items():out += [x['id'] for x in pairs if x['partition']==p][:n]
    return out

def preseal(out):
    out.mkdir(parents=True,exist_ok=True);pairs=build_pairs();tiers={}
    for name in ['LARGE','MEDIUM','SMALL']:
        ids=tier_ids(pairs,name);tiers[name]={'ids':ids,'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p50','DOSE_1p50'] if name=='LARGE' else ['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED'],'causal_ids':[x for x in ids if next(p for p in pairs if p['id']==x)['partition'].startswith('ml_')][:40 if name=='LARGE' else 24 if name=='MEDIUM' else 16]}
    rec={'schema':'R22573_C57_ML_CPT_PREOUTPUT_SEAL_V1','candidate':'C57','weights_present':0,'model_outputs_observed':0,'training_rows_used':0,'scientific_question':'Does an arXiv-ML continued-pretraining LoRA create a control-separated prior for genuine machine-learning concept relations over relation-swapped counterfactuals, rather than merely increasing ML vocabulary/register fluency?','fingerprint':'DOMAIN_CPT_RELATION_PRIOR_VS_REGISTER','pair_n':len(pairs),'suite_sha256':sha_obj(pairs),'partitions':dict(Counter(x['partition'] for x in pairs)),'paired_metric':'margin = NLL(false relation) - NLL(true relation); higher is stronger preference for the genuine relation','tiers':tiers,'positive_gate':'On selected tier: ML-core Full-vs-Base margin gain >=0.015 and paired bootstrap CI low>0.003; ML-OOD gain>=0.005; Full margin beats RANDOM and SHUFFLED by>=0.01; ML-core gain exceeds general-control gain by>=0.005.','failure_gate':'ML-core Full-vs-Base margin gain<=-0.015 with CI high<-0.003 and learned Full is worse than both RANDOM and SHUFFLED by>=0.01.','causal_if_admitted':['ATTENTION','MLP','Q','K','V','O','GATE','UP','DOWN','EARLY','MIDDLE','LATE'],'causal_E3_gate':'same group on two disjoint ML halves removes >=50% of admitted Full-vs-Base margin gain and absolute loss-of-gain>=0.01 nats/token.','workload_selection':'binary-free full 135M architecture benchmark selects largest presealed tier whose worst-case including 12 causal groups is <=65% of 45-minute budget; if none, abort before weights.','runtime_base_boundary':'Adapter card names HuggingFaceTB/SmolLM-135M, adapter_config names unsloth/smollm-135m-bnb-4bit. Full-precision HuggingFaceTB/SmolLM-135M exact revision is runtime reference Base only; exact training-time weight identity is unproven.','claim_boundary':'Fresh synthetic relation assay only; no claim that these statements were absent from pretraining or CPT corpus; no raw training rows are used.'};(out/'C57_PREOUTPUT_SEAL.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2));print(json.dumps(rec,indent=2))

def runtime_smoke(out):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    torch.set_grad_enabled(False);cfg=AutoConfig.from_pretrained(RUNTIME_BASE_REPO,revision=RUNTIME_BASE_REV,trust_remote_code=False);model=AutoModelForCausalLM.from_config(cfg);model.eval();lc=LoraConfig(r=32,lora_alpha=32,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM');p=get_peft_model(model,lc);p.eval();names=['control_random','control_shuffle','dose_0p50','dose_1p50','minus_q','minus_early']
    for n in names:p.add_adapter(n,lc);p.set_adapter(n);p.delete_adapter(n)
    x=torch.randint(3,int(cfg.vocab_size),(16,96));am=torch.ones_like(x);_=p(input_ids=x,attention_mask=am).logits;t0=time.perf_counter();_=p(input_ids=x,attention_mask=am).logits;fwd=time.perf_counter()-t0;pre=json.loads((out/'C57_PREOUTPUT_SEAL.json').read_text());est={}
    for name,t in pre['tiers'].items():
        n=len(t['ids']);conds=len(t['conditions']);ca=len(t['causal_ids']);main_batches=math.ceil((2*n)/16)*conds;causal_batches=math.ceil((2*ca)/16)*12;est[name]=180+(main_batches+causal_batches)*fwd*1.8
    budget=45*60*.65;selected=next((n for n in ['LARGE','MEDIUM','SMALL'] if est[n]<=budget),None);rec={'schema':'R22573_C57_RUNTIME_PREFLIGHT_V1','pass':bool(selected),'weights_present':0,'forward_b16_l96_seconds':fwd,'tier_estimates_seconds':est,'budget_seconds':budget,'selected_tier':selected,'identifiers':names};(out/'C57_RUNTIME_PREFLIGHT.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2));
    if not selected:raise SystemExit('RUNTIME_PREFLIGHT_RED')

def execute(out,work):
    import torch, torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file
    torch.set_grad_enabled(False);torch.manual_seed(SEED);random.seed(SEED);pre=json.loads((out/'C57_PREOUTPUT_SEAL.json').read_text());rt=json.loads((out/'C57_RUNTIME_PREFLIGHT.json').read_text());tier=rt['selected_tier'];T=pre['tiers'][tier];pairs=build_pairs();selected=[x for x in pairs if x['id'] in set(T['ids'])];raw=work/'raw';base_dir=raw/'base';ad_dir=raw/'adapter';base_dir.mkdir(parents=True);ad_dir.mkdir(parents=True)
    for fn in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json']:
        try:src=hf_hub_download(RUNTIME_BASE_REPO,filename=fn,revision=RUNTIME_BASE_REV);shutil.copy2(src,base_dir/fn)
        except Exception:
            if fn in {'config.json','model.safetensors','tokenizer.json','tokenizer_config.json'}:raise
    for fn in ['adapter_config.json','adapter_model.safetensors']:
        src=hf_hub_download(ADAPTER_REPO,filename=fn,revision=ADAPTER_REV);shutil.copy2(src,ad_dir/fn)
    assert hashlib.sha256((base_dir/'model.safetensors').read_bytes()).hexdigest()==RUNTIME_BASE_SHA;assert hashlib.sha256((ad_dir/'adapter_model.safetensors').read_bytes()).hexdigest()==ADAPTER_SHA;ac=json.loads((ad_dir/'adapter_config.json').read_text());assert int(ac['r'])==32 and int(ac['lora_alpha'])==32
    st=load_file(str(ad_dir/'adapter_model.safetensors'),device='cpu');pairmap=defaultdict(dict);energy=defaultdict(float);zero=nonfinite=0
    for k,t in st.items():
        zero+=int(float(t.float().norm())==0);nonfinite+=int(not torch.isfinite(t).all());fam=next((x for x in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if x in k),'other');energy[fam]+=float(t.float().pow(2).sum());m=re.sub(r'\.lora_[AB]\.weight$','',k);side='A' if '.lora_A.' in k else 'B' if '.lora_B.' in k else None
        if side:pairmap[m][side]=t
    total=sum(energy.values()) or 1;static={'schema':'R22573_C57_STATIC_FORENSICS_V1','tensor_count':len(st),'complete_pairs':sum(set(v)=={'A','B'} for v in pairmap.values()),'zero_tensor_count':zero,'nonfinite_tensor_count':nonfinite,'rank_config':ac['r'],'alpha':ac['lora_alpha'],'target_modules':sorted(ac['target_modules']),'energy_proxy_by_family':{k:v/total for k,v in sorted(energy.items())},'evidence_grade':'E1_STATIC_ONLY'};(out/'C57_STATIC_FORENSICS.json').write_text(json.dumps(static,indent=2));del st
    tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;base=AutoModelForCausalLM.from_pretrained(base_dir,local_files_only=True,dtype=torch.float32,device_map='cpu');base.eval();peft=PeftModel.from_pretrained(base,ad_dir,is_trainable=False);peft.eval();params={n:p for n,p in peft.named_parameters() if 'lora_A' in n or 'lora_B' in n};originals={n:p.detach().clone() for n,p in params.items()}
    def restore():
        with torch.no_grad():
            for n,p in params.items():p.copy_(originals[n])
    def mutate(cond):
        restore();g=torch.Generator().manual_seed(SEED+31)
        with torch.no_grad():
            if cond=='FULL':return
            if cond.startswith('DOSE_'):
                fac={'DOSE_0p50':.5,'DOSE_1p50':1.5}[cond]
                for n,p in params.items():
                    if 'lora_B' in n:p.mul_(fac)
            elif cond=='RANDOM_RANK_MATCHED':
                for n,p in params.items():
                    o=originals[n];z=torch.randn(o.shape,generator=g,dtype=o.dtype);zn=float(z.norm());on=float(o.norm());p.copy_(z*(on/zn if zn else 0))
            elif cond=='LAYER_SHUFFLED':
                groups=defaultdict(list)
                for n,o in originals.items():groups[(tuple(o.shape),'A' if 'lora_A' in n else 'B')].append(n)
                rng=random.Random(SEED+32)
                for names in groups.values():
                    src=names[:];rng.shuffle(src)
                    for dst,ss in zip(names,src):params[dst].copy_(originals[ss])
            else:raise KeyError(cond)
    def ctx(cond):
        if cond=='BASE':return peft.disable_adapter()
        mutate(cond);return contextlib.nullcontext()
    def statement_nll_rows_current(cases,batch_size=16):
        seq=[];meta=[]
        for c in cases:
            for lab,comp in [('true',c['true']),('false',c['false'])]:
                pids=tok(c['prompt'],add_special_tokens=True)['input_ids'];tids=tok(comp,add_special_tokens=False)['input_ids'];seq.append((pids+tids,[-100]*len(pids)+tids));meta.append((c['id'],c['partition'],lab))
        outrows=[]
        for bi in range(0,len(seq),batch_size):
            chunk=seq[bi:bi+batch_size];mm=meta[bi:bi+batch_size];mx=max(len(s) for s,l in chunk);ids=[];am=[];labs=[]
            for s,l in chunk:
                padn=mx-len(s);ids.append(s+[tok.pad_token_id]*padn);am.append([1]*len(s)+[0]*padn);labs.append(l+[-100]*padn)
            ids=torch.tensor(ids);am=torch.tensor(am);labs=torch.tensor(labs);log=peft(input_ids=ids,attention_mask=am).logits;sl=log[:,:-1,:].contiguous();yl=labs[:,1:].contiguous();loss=F.cross_entropy(sl.view(-1,sl.size(-1)),yl.view(-1),ignore_index=-100,reduction='none').view(yl.shape);mask=yl.ne(-100)
            for j,(cid,part,lab) in enumerate(mm):outrows.append({'id':cid,'partition':part,'label':lab,'nll':float(loss[j][mask[j]].mean()),'tokens':int(mask[j].sum())})
        pairrows=[];by=defaultdict(dict)
        for r in outrows:by[r['id']][r['label']]=r
        for c in cases:
            tr=by[c['id']]['true'];fr=by[c['id']]['false'];pairrows.append({'id':c['id'],'partition':c['partition'],'true_nll':tr['nll'],'false_nll':fr['nll'],'margin':fr['nll']-tr['nll']})
        return pairrows
    def rows(cond,cases):
        with ctx(cond):return statement_nll_rows_current(cases)
    allrows={cond:rows(cond,selected) for cond in T['conditions']};maps={c:{r['id']:r for r in rs} for c,rs in allrows.items()}
    def partmean(cond,part):
        xs=[r['margin'] for r in allrows[cond] if r['partition']==part];return sum(xs)/len(xs)
    core_ids=[r['id'] for r in allrows['BASE'] if r['partition']=='ml_core'];ood_ids=[r['id'] for r in allrows['BASE'] if r['partition']=='ml_ood'];gen_ids=[r['id'] for r in allrows['BASE'] if r['partition']=='general_control'];gains=[maps['FULL'][i]['margin']-maps['BASE'][i]['margin'] for i in core_ids];oodg=[maps['FULL'][i]['margin']-maps['BASE'][i]['margin'] for i in ood_ids];gg=[maps['FULL'][i]['margin']-maps['BASE'][i]['margin'] for i in gen_ids];core_gain=statistics.mean(gains);ood_gain=statistics.mean(oodg);general_gain=statistics.mean(gg);ci=bootstrap_ci(gains);full_core=partmean('FULL','ml_core');random_core=partmean('RANDOM_RANK_MATCHED','ml_core');shuffle_core=partmean('LAYER_SHUFFLED','ml_core');fr=full_core-random_core;fs=full_core-shuffle_core;positive=bool(core_gain>=.015 and ci[0]>.003 and ood_gain>=.005 and fr>=.01 and fs>=.01 and core_gain-general_gain>=.005);failure=bool(core_gain<=-.015 and ci[1]<-.003 and random_core-full_core>=.01 and shuffle_core-full_core>=.01)
    behavior={'schema':'R22573_C57_BEHAVIOR_V1','tier':tier,'condition_partition_margin_means':{c:{p:partmean(c,p) for p in ['ml_core','ml_ood','general_control']} for c in allrows},'ml_core_full_minus_base_margin_gain':core_gain,'ml_core_gain_ci95':ci,'ml_ood_full_minus_base_margin_gain':ood_gain,'general_control_full_minus_base_margin_gain':general_gain,'domain_specificity_gap':core_gain-general_gain,'full_vs_random_ml_core_margin_gap':fr,'full_vs_shuffle_ml_core_margin_gap':fs,'positive_behavior_gate':positive,'failure_gate':failure,'raw_output_text_exported':False};(out/'C57_BEHAVIOR.json').write_text(json.dumps(behavior,indent=2));causal={'schema':'R22573_C57_CAUSAL_V1','executed':False,'results':[],'E3_localization_admitted':False,'admitted_groups':[]}
    if positive:
        cset=[x for x in selected if x['id'] in set(T['causal_ids'])];base_rows=rows('BASE',cset);full_rows=rows('FULL',cset);bm={r['id']:r['margin'] for r in base_rows};fm={r['id']:r['margin'] for r in full_rows};ids=[x['id'] for x in cset];basegain=statistics.mean([fm[i]-bm[i] for i in ids]);layernos=[]
        for n in params:
            m=re.search(r'layers\.(\d+)\.',n)
            if m:layernos.append(int(m.group(1)))
        maxlayer=max(layernos) if layernos else 0
        def ablate(group):
            restore()
            with torch.no_grad():
                for n,p in params.items():
                    hit=False
                    if group=='ATTENTION':hit=any(x in n for x in ['q_proj','k_proj','v_proj','o_proj'])
                    elif group=='MLP':hit=any(x in n for x in ['gate_proj','up_proj','down_proj'])
                    elif group in ['Q','K','V','O','GATE','UP','DOWN']:hit=(group.lower()+'_proj') in n
                    else:
                        m=re.search(r'layers\.(\d+)\.',n);li=int(m.group(1)) if m else -1;third=max(1,math.ceil((maxlayer+1)/3));band='EARLY' if li<third else 'MIDDLE' if li<2*third else 'LATE';hit=(band==group)
                    if hit and 'lora_B' in n:p.zero_()
        halves=[ids[:len(ids)//2],ids[len(ids)//2:]]
        for group in ['ATTENTION','MLP','Q','K','V','O','GATE','UP','DOWN','EARLY','MIDDLE','LATE']:
            ablate(group);ar=statement_nll_rows_current(cset);am={r['id']:r['margin'] for r in ar};hs=[];passes=[]
            for h in halves:
                if not h:continue
                bg=statistics.mean([fm[i]-bm[i] for i in h]);loss=statistics.mean([fm[i]-am[i] for i in h]);frac=(loss/bg) if bg>1e-9 else 0;hp=bool(loss>=.01 and frac>=.5);hs.append({'base_full_gain':bg,'ablation_loss_of_gain':loss,'fraction_removed':frac,'pass':hp});passes.append(hp)
            loss=statistics.mean([fm[i]-am[i] for i in ids]);frac=(loss/basegain) if basegain>1e-9 else 0;gp=bool(len(passes)==2 and all(passes));causal['results'].append({'group':group,'base_full_gain':basegain,'loss_of_gain':loss,'fraction_removed':frac,'half_checks':hs,'E3_gate':gp})
            if gp:causal['admitted_groups'].append(group)
        causal['executed']=True;causal['E3_localization_admitted']=bool(causal['admitted_groups'])
    (out/'C57_CAUSAL.json').write_text(json.dumps(causal,indent=2));prov={'schema':'R22573_C57_PROVENANCE_V1','adapter_repo':ADAPTER_REPO,'adapter_revision':ADAPTER_REV,'adapter_sha256':ADAPTER_SHA,'adapter_license':'apache-2.0','runtime_base_repo':RUNTIME_BASE_REPO,'runtime_base_revision':RUNTIME_BASE_REV,'runtime_base_sha256':RUNTIME_BASE_SHA,'runtime_base_license':'apache-2.0','adapter_config_training_base':ac.get('base_model_name_or_path'),'training_time_exact_base_identity_proven':False,'source_consumed_once':True,'raw_committed_to_git':False,'training_rows_used':0,'selected_tier':tier};(out/'C57_PROVENANCE.json').write_text(json.dumps(prov,indent=2));shutil.rmtree(raw,ignore_errors=True);cleanup={'schema':'R22573_C57_CLEANUP_V1','raw_root_deleted':not raw.exists(),'raw_weights_remaining':0,'raw_tokenizer_remaining':0,'hf_cache_external_cleanup_required_by_workflow':True};(out/'C57_CLEANUP.json').write_text(json.dumps(cleanup,indent=2));print(json.dumps({'static':static,'behavior':behavior,'causal':causal},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preseal','runtime-smoke','execute']);ap.add_argument('--out',required=True);ap.add_argument('--work',default='c57-work');a=ap.parse_args();out=Path(a.out)
    if a.mode=='preseal':preseal(out)
    elif a.mode=='runtime-smoke':runtime_smoke(out)
    else:execute(out,Path(a.work))
if __name__=='__main__':main()
