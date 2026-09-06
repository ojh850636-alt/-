#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path

NOUNS = [
    'glim','zorp','tavin','merek','sular','kefin','brann','vorel','nemi','doran',
    'pelun','rask','tomer','vess','lurin','cavel','sorin','bex','navel','qorin'
]
NAMES = ['Mina','Taro','Lena','Oren','Vika','Sami','Nora','Ilan','Rhea','Davi','Kira','Miro']

def render_a(recipe):
    s=[]
    for a,b in recipe['rules']:
        s.append(f"Every {a} is a {b}.")
    for n,c in recipe['facts']:
        s.append(f"{n} is a {c}.")
    for n,c in recipe.get('distractors',[]):
        s.append(f"{n} is a {c}.")
    premise=' '.join(s)
    hname,hcls=recipe['hypothesis']
    hyp=f"{hname} is a {hcls}."
    return f"Premise: {premise}\nHypothesis: {hyp}\nEntailment?"

def render_b(recipe):
    s=[]
    for a,b in recipe['rules']:
        s.append(f"Anything classified as {a} also counts as {b}.")
    for n,c in recipe['facts']:
        s.append(f"{n} belongs to the {c} category.")
    for n,c in recipe.get('distractors',[]):
        s.append(f"Separately, {n} belongs to {c}.")
    premise=' '.join(s)
    hname,hcls=recipe['hypothesis']
    hyp=f"{hname} belongs to {hcls}."
    return f"Premise: {premise}\nHypothesis: {hyp}\nEntailment?"

def closure_entails(recipe):
    edges={}
    for a,b in recipe['rules']:
        edges.setdefault(a,set()).add(b)
    known=set(recipe['facts']) | set(recipe.get('distractors',[]))
    changed=True
    while changed:
        changed=False
        for n,c in list(known):
            for d in edges.get(c,()):
                if (n,d) not in known:
                    known.add((n,d)); changed=True
    return tuple(recipe['hypothesis']) in known

def models_entail(recipe):
    names=sorted({n for n,_ in recipe['facts']} | {n for n,_ in recipe.get('distractors',[])} | {recipe['hypothesis'][0]})
    classes=sorted({x for r in recipe['rules'] for x in r} | {c for _,c in recipe['facts']} | {c for _,c in recipe.get('distractors',[])} | {recipe['hypothesis'][1]})
    atoms=[(n,c) for n in names for c in classes]
    required=set(recipe['facts']) | set(recipe.get('distractors',[]))
    free=[a for a in atoms if a not in required]
    if len(free)>14:
        target=tuple(recipe['hypothesis'])
        memo={}
        parents={b:set() for a,b in recipe['rules']}
        for a,b in recipe['rules']: parents.setdefault(b,set()).add(a)
        def provable(atom, trail=frozenset()):
            if atom in required: return True
            if atom in trail: return False
            if atom in memo: return memo[atom]
            n,c=atom
            ans=any(provable((n,p), trail|{atom}) for p in parents.get(c,()))
            memo[atom]=ans
            return ans
        return provable(target)
    target=tuple(recipe['hypothesis'])
    for mask in range(1<<len(free)):
        m=set(required)
        for i,a in enumerate(free):
            if mask>>i & 1: m.add(a)
        ok=True
        for n in names:
            for a,b in recipe['rules']:
                if (n,a) in m and (n,b) not in m:
                    ok=False; break
            if not ok: break
        if ok and target not in m:
            return False
    return True

def make_recipe(rng, depth, label, renderer, idx):
    classes=rng.sample(NOUNS, min(depth+4, len(NOUNS)))
    chain=classes[:depth+1]
    rules=list(zip(chain[:-1], chain[1:]))
    name=rng.choice(NAMES)
    facts=[(name,chain[0])]
    other=rng.choice([n for n in NAMES if n!=name])
    distract_cls=classes[depth+1]
    distractors=[(other,distract_cls)]
    hyp=(name,chain[-1]) if label else (name,classes[-1])
    return {
        'case_id': f'C58-{idx:04d}', 'renderer': renderer, 'name': name,
        'classes': classes, 'rules': rules, 'facts': facts, 'distractors': distractors,
        'hypothesis': hyp, 'gold_idx': 0 if label else 1,
        'choices': [' yes',' no']
    }

def build(seed=22574):
    rng=random.Random(seed)
    specs=[('DISCOVERY',64,'A',(1,2)),('CONFIRMATION',64,'A',(1,3)),('GENERATOR_HOLDOUT',64,'B',(1,3)),('COMPOSITIONAL_OOD',48,'B',(3,4)),('HARD_NEGATIVE',16,'B',(2,4))]
    rows=[]; idx=0
    for split,count,renderer,depths in specs:
        for j in range(count):
            label=(j%2==0)
            r=make_recipe(rng,rng.randint(*depths),label,renderer,idx); idx+=1
            r['split']=split
            r['prompt']=render_a(r) if renderer=='A' else render_b(r)
            a=closure_entails(r); b=models_entail(r)
            assert a==b==label, (r,a,b,label)
            public={k:r[k] for k in ('case_id','split','renderer','prompt','choices','gold_idx')}
            semantic=json.dumps({k:r[k] for k in ('rules','facts','distractors','hypothesis')},sort_keys=True,separators=(',',':')).encode()
            public['semantic_digest']=hashlib.sha256(semantic).hexdigest()
            rows.append(public)
    assert len(rows)==256
    assert sum(x['gold_idx']==0 for x in rows)==128
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True)
    args=ap.parse_args(); rows=build()
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    payload='\n'.join(json.dumps(x,sort_keys=True,separators=(',',':')) for x in rows)+'\n'
    p.write_text(payload,encoding='utf-8')
    print(json.dumps({'cases':len(rows),'sha256':hashlib.sha256(payload.encode()).hexdigest(),'yes':sum(x['gold_idx']==0 for x in rows),'no':sum(x['gold_idx']==1 for x in rows)},sort_keys=True))
if __name__=='__main__': main()
