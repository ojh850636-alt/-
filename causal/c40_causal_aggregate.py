from __future__ import annotations
import argparse, json, math
from pathlib import Path
OBJECTS=['EARLY_BAND','MIDDLE_BAND','LATE_BAND','ATTENTION_ALL','MLP_ALL','Q_PROJ','K_PROJ','V_PROJ','O_PROJ','GATE_PROJ','UP_PROJ','DOWN_PROJ']

def scored(rows): return {r['case_id']:bool(r['correct']) for r in rows if r['expected_gate']!='BOUNDARY'}
def acc(d): return sum(d.values())/len(d)
def exact_sign_p(diff):
    # one-sided exact sign/binomial on positive vs negative paired differences; ties ignored
    pos=sum(x>0 for x in diff); neg=sum(x<0 for x in diff); n=pos+neg
    if n==0:return 1.0
    return sum(math.comb(n,k) for k in range(pos,n+1))/(2**n)
def holm(rows,alpha=.05):
    ordered=sorted(rows,key=lambda x:(x['p_joint'],x['object']))
    reject={}
    active=True
    for i,r in enumerate(ordered):
        thr=alpha/(len(ordered)-i)
        ok=active and r['p_joint']<=thr
        reject[r['object']]={'reject':ok,'threshold':thr,'rank':i+1}
        if not ok:active=False
    return reject

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--baseline-root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    configs={}
    for p in Path(a.root).rglob('RESULT.json'):
        d=json.load(open(p));k=(d['object'],d['mode']);assert k not in configs;k=k;configs[k]=d
    expect={(o,m) for o in OBJECTS for m in ('NECESSITY','SUFFICIENCY')};assert set(configs)==expect,(expect-set(configs),set(configs)-expect)
    bdocs={}
    for p in Path(a.baseline_root).rglob('RESULT.json'):
        d=json.load(open(p));bdocs[d['condition']]=d
    assert {'BASE','STORAGE_FULL'}<=set(bdocs)
    base=scored([r for r in bdocs['BASE']['results'] if r['split']=='DISCOVERY'])
    full=scored([r for r in bdocs['STORAGE_FULL']['results'] if r['split']=='DISCOVERY'])
    assert len(base)==len(full)==48
    rows=[]
    for o in OBJECTS:
        ab=scored(configs[(o,'NECESSITY')]['results']);only=scored(configs[(o,'SUFFICIENCY')]['results']);assert len(ab)==len(only)==48
        ids=sorted(full)
        nec=[int(full[i])-int(ab[i]) for i in ids]
        suf=[int(only[i])-int(base[i]) for i in ids]
        ng=acc(full)-acc(ab);sg=acc(only)-acc(base)
        pn=exact_sign_p(nec);ps=exact_sign_p(suf);pj=max(pn,ps)
        rows.append({'object':o,'site_count':configs[(o,'NECESSITY')]['target_site_count'],'site_names_sha256':configs[(o,'NECESSITY')]['target_site_names_sha256'],'full_accuracy':acc(full),'base_accuracy':acc(base),'ablated_accuracy':acc(ab),'only_accuracy':acc(only),'necessity_gain':ng,'sufficiency_gain':sg,'p_necessity':pn,'p_sufficiency':ps,'p_joint':pj})
    corr=holm(rows)
    for r in rows:r['holm']=corr[r['object']]
    survivors=[r['object'] for r in rows if r['necessity_gain']>0 and r['sufficiency_gain']>0 and r['holm']['reject']]
    out={'schema':'LUCIA_AA_R22543_C40_CAUSAL_STAGE1_AGGREGATE_V1','objects_complete':12,'config_count':24,'discovery_scored_cases':48,'baseline_base_accuracy':acc(base),'baseline_full_accuracy':acc(full),'rows':rows,'survivors':survivors,'survivor_count':len(survivors),'stage2':'OPEN' if survivors else 'CLOSED_NO_SURVIVOR','statistic':'paired correctness exact one-sided sign/binomial; joint p=max(necessity,sufficiency); Holm alpha=.05 across 12 objects','claim_cap':'E3_BOUNDED_CAUSAL_LOCALIZATION_TRAINING_PROVENANCE_HOLD','raw_outputs_saved':False,'raw_weights_saved':False,'hf_model_redownloads':0}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'survivors':survivors,'rows':[{k:r[k] for k in ('object','necessity_gain','sufficiency_gain','p_joint')} for r in rows]},indent=2))
if __name__=='__main__':main()
