from __future__ import annotations
import hashlib, json, random

SEED = 22583
WORK_START = 9*60
WORK_END = 17*60
LUNCH = (12*60, 13*60)
BUFFER = 10
PRIORITY_COST = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
SYSTEM = """You are a deterministic schedule-adjustment engine. Return JSON only, exactly with keys start, end, deferred_tasks. Times must be HH:MM. Rules: workday 09:00-17:00; lunch 12:00-13:00 is immovable; fixed meetings are immovable and require a 10-minute buffer before and after; flexible tasks should remain unless deferral is necessary; if deferral is necessary, defer the fewest tasks, then the lowest total priority cost (LOW<MEDIUM<HIGH), then choose lexicographically smallest task-id set; schedule the new task at the earliest feasible 5-minute start; preserve its exact duration. deferred_tasks must be a JSON list of task ids."""

def hhmm(m:int)->str:
    return f"{m//60:02d}:{m%60:02d}"

def intervals_blocked(items, deferred=()):
    d=set(deferred); out=[LUNCH]
    for x in items:
        if x['id'] in d: continue
        s,e=x['start'],x['end']
        if x['fixed']:
            s=max(WORK_START,s-BUFFER); e=min(WORK_END,e+BUFFER)
        out.append((s,e))
    return sorted(out)

def feasible_start(items,duration,deferred=()):
    blocked=intervals_blocked(items,deferred)
    for s in range(WORK_START, WORK_END-duration+1, 5):
        e=s+duration
        if all(e<=a or s>=b for a,b in blocked):
            return s
    return None

def solve(items,duration):
    flex=[x for x in items if not x['fixed']]
    import itertools
    best=None
    for k in range(len(flex)+1):
        cands=[]
        for comb in itertools.combinations(flex,k):
            ids=tuple(sorted(x['id'] for x in comb))
            s=feasible_start(items,duration,ids)
            if s is None: continue
            cost=sum(PRIORITY_COST[x['priority']] for x in comb)
            cands.append((cost,ids,s))
        if cands:
            cost,ids,s=min(cands, key=lambda z:(z[0],z[1],z[2]))
            return {'start':hhmm(s),'end':hhmm(s+duration),'deferred_tasks':list(ids)}
    raise RuntimeError('unschedulable')

def _build_base(r,i,dense):
    duration=[30,45,60,75][i%4]
    fixed=[]
    starts=[9*60+30,10*60+40,13*60+20,14*60+45,15*60+50]
    r.shuffle(starts)
    take=2 if not dense else 3
    for j,s in enumerate(sorted(starts[:take])):
        ln=[25,30,35,40][(i+j)%4]
        if s < 12*60 and s+ln > 12*60: continue
        fixed.append({'id':f'F{j+1}','start':s,'end':s+ln,'fixed':True,'priority':'HIGH'})
    flex=[]
    flex_starts=[9*60,10*60+10,11*60+10,13*60,14*60,15*60,16*60]
    r.shuffle(flex_starts)
    nflex=2 if not dense else 4
    for j,s in enumerate(sorted(flex_starts[:nflex])):
        ln=[25,30,40,45][(i+2*j)%4]
        if s < 13*60 and s+ln > 12*60: continue
        pri=['LOW','MEDIUM','LOW','HIGH'][(i+j)%4]
        flex.append({'id':f'T{j+1}','start':s,'end':min(s+ln,WORK_END),'fixed':False,'priority':pri})
    return fixed+flex,duration

def make_cases():
    r=random.Random(SEED); cases=[]; pair_i=0; attempts=0; need_deferral=4
    while pair_i<12 and attempts<500:
        attempts+=1; dense=(pair_i>=8)
        items,duration=_build_base(r,attempts,dense)
        try: exp1=solve(items,duration)
        except RuntimeError: continue
        s1=int(exp1['start'][:2])*60+int(exp1['start'][3:])
        block_len=min(30,duration)
        blocker={'id':'FX','start':s1,'end':min(s1+block_len,WORK_END),'fixed':True,'priority':'HIGH'}
        try: exp2=solve(items+[blocker],duration)
        except RuntimeError: continue
        if exp2['start']==exp1['start']: continue
        has_def=bool(exp1['deferred_tasks'] or exp2['deferred_tasks'])
        if pair_i>=8 and not has_def: continue
        pair_i+=1
        for v,(its,exp) in enumerate([(items,exp1),(items+[blocker],exp2)],1):
            inp={'existing_items':[{**x,'start':hhmm(x['start']),'end':hhmm(x['end'])} for x in its],'new_task':{'id':'NEW','duration_minutes':duration,'priority':'HIGH'},'workday':'09:00-17:00','lunch':'12:00-13:00','fixed_buffer_minutes':10}
            user="Adjust this schedule. Input JSON:\n"+json.dumps(inp,sort_keys=True,separators=(',',':'))
            prompt=SYSTEM+'\nUSER\n'+user
            cases.append({'case_id':f'C65-P{pair_i:02d}-V{v}','pair_id':f'C65-P{pair_i:02d}','variant':v,'input':inp,'expected':exp,'prompt':prompt,'prompt_hash':hashlib.sha256(prompt.encode()).hexdigest(),'input_hash':hashlib.sha256(json.dumps(inp,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'expected_hash':hashlib.sha256(json.dumps(exp,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
    if len(cases)!=24: raise RuntimeError(f'generated {len(cases)}')
    assert sum(bool(c['expected']['deferred_tasks']) for c in cases)>=4
    return cases

def metadata(cases=None):
    cases=cases or make_cases()
    return [{k:c[k] for k in ['case_id','pair_id','variant','prompt_hash','input_hash','expected_hash']}|{'expected':c['expected']} for c in cases]

if __name__=='__main__':
    cs=make_cases(); print(json.dumps({'case_count':len(cs),'deferral_cases':sum(bool(c['expected']['deferred_tasks']) for c in cs),'metadata':metadata(cs)},indent=2))
