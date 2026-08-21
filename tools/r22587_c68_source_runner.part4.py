                    for i,(_,n) in enumerate(arr): dict(model.named_parameters())[n].copy_(src[(i+1)%len(src)])
    restore(); full_prior=[eval_prompt(model,p,'prior') for p in PRIMARY_PROMPTS]; full_cal=[eval_prompt(model,p,'asof') for p in CAL_PROMPTS]; full_nll=[nll(model,p,PRIOR_REF) for p in PRIMARY_PROMPTS[:4]]
    behavior['primary']['FULL']={'prior':full_prior,'prior_strict_pass':sum(x['pass'] for x in full_prior),'prior_fixture_pass_total':sum(x['passed'] for x in full_prior),'cal_asof':full_cal,'cal_strict_pass':sum(x['pass'] for x in full_cal),'target_nll':full_nll}
    rescue=[i for i,(b,f) in enumerate(zip(base_prior,full_prior)) if (not b['pass']) and f['pass']]; harm=[i for i,(b,f) in enumerate(zip(base_prior,full_prior)) if b['pass'] and (not f['pass'])]
    primary_gain=len(rescue)>len(harm) and len(rescue)>0
    behavior['primary']['paired']={'rescue_prompt_indices':rescue,'harm_prompt_indices':harm,'primary_gain':primary_gain}
    # Cheap NLL controls on all pre-registered control arms. Generation controls only on rescued prompts.
    for cond in ['random1','random2','shuffle','dose0.5','dose1.5']:
        apply(cond); losses=[nll(model,p,PRIOR_REF) for p in PRIMARY_PROMPTS[:4]]; rec={'target_nll':losses,'generation':'NOT_RUN_NO_PRIMARY_RESCUE'}
        if rescue:
            inds=rescue[:min(3,len(rescue))]; outs=[eval_prompt(model,PRIMARY_PROMPTS[i],'prior') for i in inds]; rec['generation']={'prompt_indices':inds,'outcomes':outs,'strict_pass':sum(x['pass'] for x in outs)}
        behavior['controls'][cond]=rec
    restore()
    # Control separation: learned FULL rescues must not all be reproduced by both random and shuffle.
    sep=False
    if rescue:
        n=min(3,len(rescue)); rp=behavior['controls']['random1']['generation']['strict_pass']; sp=behavior['controls']['shuffle']['generation']['strict_pass']; sep=(rp<n and sp<n)
    behavior['primary']['paired']['random_shuffle_separated']=sep
    # Same-environment locked prompt replay only after primary+control gate.
    holdout_gain=False; hold_rescue=[]; hold_harm=[]
    if primary_gain and sep:
        restore(); hb=[]
        # Disable adapter on same Peft model to avoid a second Base object.
        with model.disable_adapter(): hb=[eval_prompt(model,p,'prior') for p in HOLDOUT_PROMPTS]
        restore(); hf=[eval_prompt(model,p,'prior') for p in HOLDOUT_PROMPTS]
        hold_rescue=[i for i,(b,f) in enumerate(zip(hb,hf)) if (not b['pass']) and f['pass']]; hold_harm=[i for i,(b,f) in enumerate(zip(hb,hf)) if b['pass'] and (not f['pass'])]
        holdout_gain=len(hold_rescue)>len(hold_harm) and len(hold_rescue)>0
        behavior['holdout']={'status':'READ_AFTER_PRIMARY_CONTROL_GATE','BASE':hb,'FULL':hf,'rescue_prompt_indices':hold_rescue,'harm_prompt_indices':hold_harm,'same_environment_replicated_gain':holdout_gain}
    # Bounded E3 necessity/sufficiency only after same-env positive replication.
    if primary_gain and sep and holdout_gain:
        probe_idx=rescue[0]; mods=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']; caus={'probe_prompt_index':probe_idx,'module_ablation':{},'layer_quartile_ablation':{},'module_sufficiency':{}}
        def zero_where(pred):
            restore()
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if n in original and 'lora_B' in n and pred(n): p.zero_()
        for mod in mods:
            zero_where(lambda n,m=mod: f'.{m}.' in n); caus['module_ablation'][mod]=eval_prompt(model,PRIMARY_PROMPTS[probe_idx],'prior')
        for lo,hi in [(0,6),(7,13),(14,20),(21,27)]:
            zero_where(lambda n,lo=lo,hi=hi: (lambda m: bool(m) and lo<=int(m.group(1))<=hi)(re.search(r'\.layers\.(\d+)\.',n))); caus['layer_quartile_ablation'][f'{lo}_{hi}']=eval_prompt(model,PRIMARY_PROMPTS[probe_idx],'prior')
        for mod in mods:
            restore()
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if n in original and 'lora_B' in n and f'.{mod}.' not in n: p.zero_()
            caus['module_sufficiency'][mod]=eval_prompt(model,PRIMARY_PROMPTS[probe_idx],'prior')
        restore(); behavior['causal']={'status':'E3_BOUNDED_SAME_ENVIRONMENT_ONLY','results':caus,'not_e4_or_e5':True}
    behavior['claim_boundary']={'fresh_primary_gain':primary_gain,'random_shuffle_separated':sep,'same_environment_holdout_gain':holdout_gain,'external_positive_capability_e4_plus_increment':0,'e5_increment':0,'reason_e4e5_zero':'single physical GitHub runner; same-env holdout is at most scoped replication, not independent E5; public-card asof calibration excluded from claim'}
    jwrite('R22587_C68_FRESH_EXECUTABLE_BEHAVIOR.json',behavior)
    # Raw-free brain summary.
    brain={'schema':'R22587_C68_RAWFREE_BRAIN_MATERIAL_V1','question':QUESTION,'observed':{'static_operator':True,'fresh_primary_gain':primary_gain,'control_separated':sep,'same_env_holdout_gain':holdout_gain,'bounded_causal_ran':behavior['causal']['status'].startswith('E3')},'method_laws':['PUBLIC_CARD_PROXIMITY_TASK_MUST_BE_QUARANTINED_FROM_CLAIM','ATOMIC_DISCOVERY_TO_IMMUTABLE_PIN_BEFORE_WEIGHT_GET','FRESH_EXECUTABLE_FUNCTION_BEFORE_CAUSAL','LOCKED_SAME_ENVIRONMENT_REPLAY_IS_NOT_E5'],'promotion':'QUARANTINED_RESEARCH_ONLY_UNTIL_INDEPENDENT_E5_AND_LICENSE_PROVENANCE_CLOSURE','upstream_license_warning':warning}
    jwrite('R22587_C68_RAWFREE_BRAIN_MATERIAL.json',brain)
    return {'status':'PASS','source_consumed':source_consumed,'adapter_revision':arev,'behavior':behavior['claim_boundary'],'elapsed_seconds':time.time()-start}

result=None; err=None; source_consumed=False
try:
    result=main(); source_consumed=bool(result.get('source_consumed'))
except Exception as e:
    err={'type':type(e).__name__,'message':str(e),'traceback_tail':'\n'.join(traceback.format_exc().splitlines()[-12:])}
finally:
    # Delete all raw external source and model outputs before artifact upload.
    for p in (DL,ESC):
        if p.exists(): shutil.rmtree(p,ignore_errors=True)
    # Clean caches potentially created by libraries under work only; runner-global pip cache is not source material.
    raw_remaining=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and (p.suffix.lower() in {'.safetensors','.bin','.pt','.pth','.pkl','.pickle'} or p.name in {'tokenizer.json','vocab.json','merges.txt'}): raw_remaining.append(str(p))
    source_consumed = source_consumed or (ROOT/'SOURCE_CONSUMED.marker').exists()
    receipt={'schema':'R22587_C68_FINAL_DELETION_RECEIPT_V1','result':result,'error':err,'source_consumed':source_consumed,'raw_remaining':raw_remaining,'raw_remaining_count':len(raw_remaining),'model_weight_files_in_out':0,'post_delete_pass':len(raw_remaining)==0}
    jwrite('R22587_C68_FINAL_DELETION_RECEIPT.json',receipt)
if err or not result or not receipt['post_delete_pass']:
    print(json.dumps({'PASS':False,'error':err,'deletion':receipt['post_delete_pass']},indent=2)); sys.exit(1)
print(json.dumps({'PASS':True,'result':result,'deletion':True},indent=2))
