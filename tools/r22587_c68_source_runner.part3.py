    card=info.get('cardData') or {}; meta=sibling_meta(info,'adapter_model.safetensors')
    readme=text_url(f'https://huggingface.co/{ADAPTER_REPO}/raw/{arev}/README.md'); cfg_txt=text_url(f'https://huggingface.co/{ADAPTER_REPO}/raw/{arev}/adapter_config.json')
    cfg=json.loads(cfg_txt)
    exacts=sorted(set(re.findall(r'Qwen/Qwen2\.5-Coder-1\.5B-Instruct`?\s*@\s*`?([0-9a-f]{40})',readme,re.I)+re.findall(r'revision=["\']([0-9a-f]{40})["\']',readme,re.I)))
    if BASE_REV not in exacts: raise RuntimeError('EXACT_TRAINING_BASE_NOT_IN_IMMUTABLE_CARD')
    if card.get('license')!='cc-by-4.0': raise RuntimeError('ADAPTER_LICENSE_DRIFT:'+str(card.get('license')))
    if cfg.get('base_model_name_or_path')!=BASE_REPO or cfg.get('peft_type')!='LORA' or cfg.get('r')!=16 or cfg.get('lora_alpha')!=32: raise RuntimeError('ADAPTER_CONFIG_DRIFT')
    expected_targets={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    if set(cfg.get('target_modules') or [])!=expected_targets or cfg.get('modules_to_save') not in (None,[]): raise RuntimeError('TARGET_OR_AUX_DRIFT')
    warning=('Licence verification is OUTSTANDING' in readme or 'License verification is OUTSTANDING' in readme)
    provenance={'schema':'R22587_C68_ATOMIC_SOURCE_PIN_V1','adapter_repo':ADAPTER_REPO,'adapter_revision':arev,'api_weight_meta':meta,'adapter_license':card.get('license'),'upstream_license_verification_warning':warning,'base_repo':BASE_REPO,'training_base_revision':BASE_REV,'training_base_revision_source':'immutable adapter README generated from run artifacts','adapter_config_revision':cfg.get('revision'),'question':QUESTION,'public_card_proximity_calibration':PUBLIC_CAL,'weight_gets_before_pin':0,'model_weight_bytes_before_pin':0,'primary_prompt_hashes':[sha_bytes(x.encode()) for x in PRIMARY_PROMPTS],'holdout_prompt_hashes':[sha_bytes(x.encode()) for x in HOLDOUT_PROMPTS]}
    jwrite('R22587_C68_ATOMIC_SOURCE_PIN.json',provenance)
    # Download exact source into download staging then atomic rename to run-scoped escrow.
    adld=DL/'adapter_model.safetensors'; ad=direct_download(ADAPTER_REPO,arev,'adapter_model.safetensors',adld); source_consumed=True; (ROOT/'SOURCE_CONSUMED.marker').write_text(json.dumps({'adapter_repo':ADAPTER_REPO,'adapter_revision':arev,'sha256':ad['sha256'],'size':ad['size']}))
    adesc=ESC/'adapter'; adesc.mkdir(); os.replace(adld,adesc/'adapter_model.safetensors'); (adesc/'adapter_config.json').write_text(cfg_txt); (adesc/'README.md').write_text(readme)
    raw_files += [adesc/'adapter_model.safetensors',adesc/'adapter_config.json',adesc/'README.md']
    if meta.get('size') and int(meta['size'])!=ad['size']: raise RuntimeError('ADAPTER_API_SIZE_MISMATCH')
    if meta.get('sha256') and str(meta['sha256']).replace('sha256:','')!=ad['sha256']: raise RuntimeError('ADAPTER_API_SHA_MISMATCH')
    # Static archaeology before Base ingress.
    A={};B={};other=[]
    with safe_open(adesc/'adapter_model.safetensors',framework='numpy') as f:
        names=list(f.keys())
        for n in names:
            if '.lora_A.' in n: A[n.replace('.lora_A.default.weight','').replace('.lora_A.weight','')]=np.asarray(f.get_tensor(n),dtype=np.float32)
            elif '.lora_B.' in n: B[n.replace('.lora_B.default.weight','').replace('.lora_B.weight','')]=np.asarray(f.get_tensor(n),dtype=np.float32)
            else: other.append({'name':n,'shape':list(f.get_slice(n).get_shape()),'dtype':str(f.get_slice(n).get_dtype())})
    keys=sorted(set(A)|set(B)); incomplete=[k for k in keys if k not in A or k not in B]
    if incomplete: raise RuntimeError('INCOMPLETE_LORA_PAIRS')
    bymod={}; bylayer={}; pairs=[]; total_proxy=0.0; zero=0; finite=True; scale=cfg['lora_alpha']/cfg['r']
    for k in keys:
        a=A[k]; b=B[k]; z=not (np.count_nonzero(a) and np.count_nonzero(b)); zero+=int(z); finite &= bool(np.isfinite(a).all() and np.isfinite(b).all())
        er=min(int(np.linalg.matrix_rank(a.astype(np.float64))),int(np.linalg.matrix_rank(b.astype(np.float64))))
        proxy=float((scale*np.linalg.norm(a.astype(np.float64))*np.linalg.norm(b.astype(np.float64)))**2); total_proxy+=proxy
        mod=k.split('.')[-1]; lm=re.search(r'\.layers\.(\d+)\.',k); layer=int(lm.group(1)) if lm else -1
        bm=bymod.setdefault(mod,{'pairs':0,'proxy':0.0,'zero_pairs':0,'rank_min':999999,'rank_max':0}); bm['pairs']+=1; bm['proxy']+=proxy; bm['zero_pairs']+=int(z); bm['rank_min']=min(bm['rank_min'],er); bm['rank_max']=max(bm['rank_max'],er)
        bl=bylayer.setdefault(str(layer),{'pairs':0,'proxy':0.0,'modules':set()}); bl['pairs']+=1; bl['proxy']+=proxy; bl['modules'].add(mod)
        pairs.append({'component_id':k,'layer':layer,'module':mod,'a_shape':list(a.shape),'b_shape':list(b.shape),'effective_rank_upper_bound':er,'zero_pair':z})
    for v in bymod.values(): v['energy_ratio']=v.pop('proxy')/total_proxy if total_proxy else 0.0
    for v in bylayer.values(): v['energy_ratio']=v.pop('proxy')/total_proxy if total_proxy else 0.0; v['modules']=sorted(v['modules'])
    static={'schema':'R22587_C68_ADAPTER_OPERATOR_ARCHAEOLOGY_V1','source':{'repo':ADAPTER_REPO,'revision':arev,'sha256':ad['sha256'],'size':ad['size']},'operator':{'tensor_count':len(A)+len(B)+len(other),'complete_pairs':len(keys),'alive_pairs':len(keys)-zero,'zero_pairs':zero,'all_finite':finite,'auxiliary_tensor_count':len(other),'target_modules_actual':sorted(bymod),'by_module':dict(sorted(bymod.items())),'by_layer':dict(sorted(bylayer.items(),key=lambda x:int(x[0]))),'pair_shape_rank_inventory':pairs,'energy_proxy_definition':'(alpha/r*||A||F*||B||F)^2 ranking only; not Delta-W norm or causal importance'},'claim_boundary':'STATIC_ONLY_UNTIL_FRESH_EXECUTABLE_BEHAVIOR'}
    jwrite('R22587_C68_ADAPTER_OPERATOR_ARCHAEOLOGY.json',static)
    if zero or not finite or other: raise RuntimeError('ADAPTER_OPERATOR_VIABILITY_FAIL')
    # Exact Base one-use ingress only after provenance + live Adapter are closed.
    besc=ESC/'base'; besc.mkdir()
    base_files=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']
    base_manifest=[]
    for name in base_files:
        p=DL/name
        try: rec=direct_download(BASE_REPO,BASE_REV,name,p)
        except Exception:
            if name in ('generation_config.json',): continue
            raise
        os.replace(p,besc/name); raw_files.append(besc/name); base_manifest.append({'name':name,'size':rec['size'],'sha256':rec['sha256']})
    provenance['base_files']=base_manifest; provenance['adapter_actual']={'size':ad['size'],'sha256':ad['sha256']}; jwrite('R22587_C68_ATOMIC_SOURCE_PIN.json',provenance)
    # Load Base exactly once.
    tok=AutoTokenizer.from_pretrained(str(besc),local_files_only=True,use_fast=True); 
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    base=AutoModelForCausalLM.from_pretrained(str(besc),local_files_only=True,dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager'); base.eval()
    prior_fx=build_prior_fixtures(); asof_fx=build_asof_fixtures()
    fixture_hashes={'prior72':sha_bytes(json.dumps(prior_fx,sort_keys=True,separators=(',',':')).encode()),'asof64':sha_bytes(json.dumps(asof_fx,sort_keys=True,separators=(',',':')).encode())}
    def gen(m,user,max_new=300):
        p=prompt_text(tok,user); inp=tok(p,return_tensors='pt');
        with torch.inference_mode(): out=m.generate(**inp,max_new_tokens=max_new,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
        return tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
    def nll(m,user,target):
        p=prompt_text(tok,user); a=tok(p,add_special_tokens=False)['input_ids']; b=tok(target,add_special_tokens=False)['input_ids']; ids=torch.tensor([a+b],dtype=torch.long); lab=torch.tensor([[-100]*len(a)+b],dtype=torch.long)
        with torch.inference_mode(): return float(m(input_ids=ids,labels=lab,use_cache=False).loss.float().item())
    def eval_prompt(m,user,fam):
        text=gen(m,user); func='prior_rolling_mean' if fam=='prior' else 'asof_join'; fx=prior_fx if fam=='prior' else asof_fx; r=outcome_from_text(text,func,fx); del text; return r
    behavior={'schema':'R22587_C68_FRESH_EXECUTABLE_BEHAVIOR_V1','question':QUESTION,'fixture_hashes':fixture_hashes,'public_card_proximity_calibration_not_claim_bearing':True,'primary':{},'holdout':{'status':'LOCKED_NOT_READ_UNLESS_GATE'},'controls':{},'causal':{'status':'NOT_RUN'},'claim_boundary':{}}
    # Base primary + public proximity calibration.
    base_prior=[eval_prompt(base,p,'prior') for p in PRIMARY_PROMPTS]; base_cal=[eval_prompt(base,p,'asof') for p in CAL_PROMPTS]
    base_nll=[nll(base,p,PRIOR_REF) for p in PRIMARY_PROMPTS[:4]]
    behavior['primary']['BASE']={'prior':base_prior,'prior_strict_pass':sum(x['pass'] for x in base_prior),'prior_fixture_pass_total':sum(x['passed'] for x in base_prior),'cal_asof':base_cal,'cal_strict_pass':sum(x['pass'] for x in base_cal),'target_nll':base_nll}
    # Attach Adapter without reloading Base.
    model=PeftModel.from_pretrained(base,str(adesc),is_trainable=False); model.eval()
    original={n:p.detach().clone() for n,p in model.named_parameters() if 'lora_' in n}
    def restore():
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in original: p.copy_(original[n])
    def apply(cond):
        restore()
        with torch.no_grad():
            if cond.startswith('dose'):
                d=float(cond[4:]);
                for n,p in model.named_parameters():
                    if n in original and 'lora_B' in n: p.mul_(d)
            elif cond.startswith('random'):
                seed=int(cond.replace('random',''))+22587; g=torch.Generator(device='cpu'); g.manual_seed(seed)
                for n,p in model.named_parameters():
                    if n in original:
                        x=torch.randn(tuple(p.shape),generator=g,dtype=torch.float32); norm=float(original[n].float().norm()); x=x/(x.norm()+1e-12)*norm; p.copy_(x.to(p.dtype))
            elif cond=='shuffle':
                groups={}
                for n in original:
                    m=re.search(r'\.layers\.(\d+)\.',n)
                    if m: groups.setdefault(re.sub(r'\.layers\.\d+\.', '.layers.*.',n),[]).append((int(m.group(1)),n))
                for arr in groups.values():
                    arr.sort(); src=[original[n] for _,n in arr]
