from __future__ import annotations
import argparse,json,random,statistics
from pathlib import Path

PRIMARY=['CONFIRMATION','GENERATOR_HOLDOUT']
CONTROLS=['RETRIEVAL_PLAN','CONSOLIDATION','STORAGE_RANDOM_SIGN','STORAGE_LAYER_SHUFFLE']
CONDITIONS=['BASE','STORAGE_FULL','RETRIEVAL_PLAN','CONSOLIDATION','STORAGE_RANDOM_SIGN','STORAGE_LAYER_SHUFFLE','STORAGE_DOSE_0.25','STORAGE_DOSE_0.5','STORAGE_DOSE_1.5']

def metrics(rows,split):
    x=[r for r in rows if r['split']==split and r['expected_gate']!='BOUNDARY'];assert len(x)==48
    good=[bool(r['correct']) for r in x];parse=sum(r['parse_valid'] for r in x)/len(x)
    store=[r for r in x if r['expected_gate']=='STORE_LIKE'];ign=[r for r in x if r['expected_gate']=='IGNORE']
    return {'n':len(x),'accuracy':sum(good)/len(x),'parse_valid':parse,'store_like_recall':sum(bool(r['correct']) for r in store)/len(store),'ignore_recall':sum(bool(r['correct']) for r in ign)/len(ign),'grounding_pass_store_like':sum(bool(r['grounding_pass']) for r in store)/len(store)}

def paired_ci(a,b,reps=10000,seed=22543040):
    # a,b correctness dicts by case id; return Full-Base bootstrap CI
    ids=sorted(set(a)&set(b));rng=random.Random(seed);vals=[];n=len(ids)
    for _ in range(reps):
        s=0.0
        for __ in range(n):
            k=ids[rng.randrange(n)];s+=float(a[k])-float(b[k])
        vals.append(s/n)
    vals.sort();return [vals[int(.025*(reps-1))],vals[int(.975*(reps-1))]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.root)
    by={}
    for p in root.rglob('RESULT.json'):
        d=json.load(open(p));c=d['condition'];assert c not in by;by[c]=d
    assert set(by)==set(CONDITIONS),(set(CONDITIONS)-set(by),set(by)-set(CONDITIONS))
    stats={c:{s:metrics(d['results'],s) for s in PRIMARY} for c,d in by.items()}
    cis={}
    for s in PRIMARY:
        f={r['case_id']:bool(r['correct']) for r in by['STORAGE_FULL']['results'] if r['split']==s and r['expected_gate']!='BOUNDARY'}
        b={r['case_id']:bool(r['correct']) for r in by['BASE']['results'] if r['split']==s and r['expected_gate']!='BOUNDARY'}
        cis[s]=paired_ci(f,b)
    checks=[]
    for s in PRIMARY:
        fs=stats['STORAGE_FULL'][s];bs=stats['BASE'][s];best=max(stats[c][s]['accuracy'] for c in CONTROLS)
        checks += [
          {'id':f'{s}_full_accuracy','pass':fs['accuracy']>=.75,'value':fs['accuracy'],'threshold':.75},
          {'id':f'{s}_full_minus_base','pass':fs['accuracy']-bs['accuracy']>=.10,'value':fs['accuracy']-bs['accuracy'],'threshold':.10},
          {'id':f'{s}_full_minus_best_control','pass':fs['accuracy']-best>=.05,'value':fs['accuracy']-best,'threshold':.05},
          {'id':f'{s}_parse','pass':fs['parse_valid']>=.95,'value':fs['parse_valid'],'threshold':.95},
          {'id':f'{s}_store_recall','pass':fs['store_like_recall']>=.75,'value':fs['store_like_recall'],'threshold':.75},
          {'id':f'{s}_ignore_recall','pass':fs['ignore_recall']>=.75,'value':fs['ignore_recall'],'threshold':.75},
          {'id':f'{s}_paired_ci_lower','pass':cis[s][0]>0.0,'value':cis[s][0],'threshold':'>0'},
        ]
    boundary={c:{} for c in CONDITIONS}
    for c,d in by.items():
      for s in PRIMARY:
        x=[r for r in d['results'] if r['split']==s and r['expected_gate']=='BOUNDARY'];boundary[c][s]={'n':len(x),'predicted_store_like':sum(r['predicted_gate']=='STORE_LIKE' for r in x),'predicted_ignore':sum(r['predicted_gate']=='IGNORE' for r in x),'parse_valid':sum(r['parse_valid'] for r in x)}
    verdict='GREEN' if all(x['pass'] for x in checks) else 'RED'
    out={'schema':'LUCIA_AA_R22543_C40_BEHAVIOR_GATE_V1','verdict':verdict,'conditions_complete':len(by),'condition_stats':stats,'paired_bootstrap_ci_full_minus_base':cis,'checks':checks,'boundary_duplicate_report':boundary,'dose_accuracy':{c:{s:stats[c][s]['accuracy'] for s in PRIMARY} for c in ['BASE','STORAGE_DOSE_0.25','STORAGE_DOSE_0.5','STORAGE_FULL','STORAGE_DOSE_1.5']},'causal_lane':'OPEN' if verdict=='GREEN' else 'CLOSED_NOT_RUN','raw_outputs_saved':False,'raw_weights_saved':False,'hf_model_redownloads':0}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'verdict':verdict,'stats':{c:stats[c] for c in ['BASE','STORAGE_FULL']},'cis':cis},indent=2))
if __name__=='__main__':main()
