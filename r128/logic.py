"""Pure CPU statistical gates for paired evidence; no neural weights or raw logits.

The exact tests are hypothesis-driven; these functions cannot promote a circuit.
"""
from __future__ import annotations
from collections import Counter
from math import comb
import hashlib
import json
import random

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()).hexdigest()

def paired(a,b,y):
    """Accuracy of b relative to a, exact discordant sign test; 0/1 labels."""
    if len(a)!=len(b) or len(a)!=len(y) or not a:
        raise ValueError('PAIRED_LENGTH_OR_EMPTY')
    if any(v not in (0,1) for v in a+b+y):
        raise ValueError('BINARY_LABEL_REQUIRED')
    rescue=sum(x!=t and z==t for x,z,t in zip(a,b,y))
    harm=sum(x==t and z!=t for x,z,t in zip(a,b,y))
    n=rescue+harm
    p=(sum(comb(n,k) for k in range(rescue,n+1)) / (2**n)) if n else 1.0
    return {'n':len(y),'old_accuracy':sum(x==t for x,t in zip(a,y))/len(y),
            'new_accuracy':sum(z==t for z,t in zip(b,y))/len(y),
            'delta':(rescue-harm)/len(y), 'rescue':rescue,'harm':harm,
            'one_sided_p':p,'discordant_pairs':n}

def paired_bootstrap(a,b,y,seed=12823,draws=4000):
    if len(a)!=len(b) or len(a)!=len(y) or not y:
        raise ValueError('PAIRED_LENGTH_OR_EMPTY')
    v=[int(z==t)-int(x==t) for x,z,t in zip(a,b,y)]
    rng=random.Random(seed); n=len(v)
    samples=sorted(sum(v[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return [samples[int(draws*.025)],samples[min(draws-1,int(draws*.975))]]

def split_summary(records,labels,split_ids,conditions):
    if len(records)!=len(labels) or len(records)!=len(split_ids):
        raise ValueError('SUMMARY_LENGTH')
    if not set(conditions).issubset(set(records[0])):
        raise ValueError('SUMMARY_CONDITION_MISSING')
    out={}
    for split in sorted(set(split_ids)):
        ix=[i for i,s in enumerate(split_ids) if s==split]
        ys=[labels[i] for i in ix]
        one={k:[records[i][k] for i in ix] for k in conditions}
        out[split]={'n':len(ix),'by_condition':{k:sum(v==t for v,t in zip(a,ys))/len(ys) for k,a in one.items()},
                    'full_vs_head':paired(one['HEAD_ONLY'],one['FULL'],ys)}
    return out

def gate(summary, policy):
    """Do not infer causal support from this precommitted behavioral screening."""
    global_s=summary['ALL']['full_vs_head']
    conf=summary['CONFIRMATION']['full_vs_head']
    ood=summary['FAMILY_OOD']['full_vs_head']
    thresholds=policy['behavior_gate']
    flags={
        'conditional_positive':global_s['delta']>=thresholds['min_delta'],
        'paired_p_positive':global_s['one_sided_p']<=thresholds['max_p'],
        'confirmation_positive':conf['delta']>0,
        'family_ood_positive':ood['delta']>0,
        'random_control_margin': summary['ALL']['by_condition']['FULL']-summary['ALL']['by_condition']['RANDOM_SAME_HEAD']>=thresholds['min_control_margin'],
        'shuffled_control_margin': summary['ALL']['by_condition']['FULL']-summary['ALL']['by_condition']['SHUFFLED_SAME_HEAD']>=thresholds['min_control_margin'],
    }
    return {'behavior_positive_screen':'GREEN' if all(flags.values()) else 'RED',
            'flags':flags, 'causal_claim':False, 'promotion':False}
