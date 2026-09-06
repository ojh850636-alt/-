import json, pathlib, sys

def load(p): return json.loads(pathlib.Path(p).read_text())
def fail(msg): print('FAIL',msg); raise SystemExit(1)

def walk_no_raw_text(x,path='root'):
    if isinstance(x,dict):
        for k,v in x.items():
            lk=k.lower()
            if lk in {'raw_output','output_text','generated_text','logits','hidden_states','activations','lora_a','lora_b','delta_w'}: fail(f'raw key {path}.{k}')
            walk_no_raw_text(v,path+'.'+k)
    elif isinstance(x,list):
        for i,v in enumerate(x): walk_no_raw_text(v,f'{path}[{i}]')

def verify_behavior(p):
    o=load(p)
    if o.get('source',{}).get('adapter_sha256')!='414f0d5db8d6560a12f5841eb4a7fdc8ce558db37f4fe4e1c3c19a249223f918': fail('adapter sha')
    if o.get('source',{}).get('base_sha256')!='d3ae9628a0ed2cfa21b5b8345be4e1d4126794e734f3483dcc275b4735bf5a15': fail('base sha')
    if o.get('claim_suffix')!='_BASE_REVISION_HOLD': fail('base hold missing')
    if o.get('raw_model_outputs_retained')!=0: fail('raw outputs retained')
    g=bool(o.get('control_separated_positive_green'))
    if g:
        for k in ('FULL_MINUS_BASE_LOCKED','FULL_MINUS_RANDOM_LOCKED','FULL_MINUS_SHUFFLE_LOCKED'):
            ci=o['condition_summaries'][k]['ci95']
            if ci[0] is None or ci[0] <= 0: fail('green inconsistent '+k)
    walk_no_raw_text(o)
    return g

def verify_causal(p):
    o=load(p)
    if o.get('claim_suffix')!='_BASE_REVISION_HOLD': fail('causal base hold missing')
    for k in ('e4_claim','e5_claim','algorithm_claim','student_claim','promotion_claim'):
        if o.get(k) is not False: fail(k+' must be false')
    if not str(o.get('max_supported_claim','')).endswith('BASE_REVISION_HOLD'): fail('causal claim suffix')
    if o.get('raw_activations_retained')!=0 or o.get('raw_logits_retained')!=0: fail('causal raw retained')
    walk_no_raw_text(o)

def verify_delete(p):
    o=load(p)
    if o.get('remaining_targets') not in ([],None): fail('raw deletion remaining')
    if o.get('raw_git_commit') not in (0,None): fail('raw git commit')

if __name__=='__main__':
    if len(sys.argv)<3: raise SystemExit('usage: behavior.json deletion.json [causal.json causal_delete.json]')
    g=verify_behavior(sys.argv[1]); verify_delete(sys.argv[2])
    if len(sys.argv)>=5: verify_causal(sys.argv[3]); verify_delete(sys.argv[4])
    print('PASS C78 raw-free claim-boundary verifier; behavior_green=',g)
