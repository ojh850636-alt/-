from __future__ import annotations
import ast, contextlib, hashlib, json, math, os, random, re, shutil, subprocess, sys, tempfile, time, traceback, urllib.request
from pathlib import Path

ADAPTER_REPO='Hak5/qwen2.5-coder-1.5b-ml-engineer'
BASE_REPO='Qwen/Qwen2.5-Coder-1.5B-Instruct'
BASE_REV='2e1fd397ee46e1388853d2af2c993145b0f1098a'
SYSTEM_PROMPT='You are a rigorous ML engineer. Produce executable Python, inspect data before modeling, prevent leakage, justify validation, and return reusable artifacts.'
QUESTION='PRIOR_ONLY_ENTITY_ROLLING_FEATURE_WITH_WINDOW_STALENESS_TIE_ORDER_AND_NO_CURRENT_OR_FUTURE_LEAKAGE'
PUBLIC_CAL='PUBLIC_CARD_PROXIMITY_ASOF_JOIN_CALIBRATION_ONLY_NOT_CLAIM_BEARING'
ROOT=Path('work'); DL=ROOT/'download'; ESC=ROOT/'escrow'; OUT=ROOT/'out'
for p in (DL,ESC,OUT): p.mkdir(parents=True,exist_ok=True)
UA={'User-Agent':'LUCIA-AA-R22587-C68-canonical-one-use'}

PRIMARY_PROMPTS=[
"Implement only standard-library Python function prior_rolling_mean(rows, max_points, max_age_minutes). Each row has row_id, entity, integer ts minutes, optional seq default 0, and numeric value. For each row, compute the mean of at most max_points strictly earlier rows in the same entity order defined by (ts, seq, row_id), keeping only history no older than max_age_minutes inclusive. A row must never use itself or a later row. Return records in the original input order as {'row_id':..., 'feature': mean_or_None}. Equal timestamps are ordered by seq then row_id. Return executable Python, not pseudocode.",
"Write a dependency-free Python prior_rolling_mean(rows, max_points, max_age_minutes) for leakage-safe online features. Sort separately per entity by ts, then seq (missing seq is zero), then row_id. The feature at a row may see only items before it in that order, must respect an inclusive age limit and the last max_points eligible observations, and must not leak the current value. Preserve the caller's row order in the returned list of row_id/feature dicts. If no history is eligible use None.",
"Create executable stdlib-only code defining prior_rolling_mean. Input rows are shuffled dictionaries with row_id/entity/ts/seq/value. Build a point-in-time rolling mean: same entity only; causal order is (ts, seq, row_id); use no current/future value; age <= max_age_minutes; use at most the most recent max_points candidates; max_points=0 means no candidates. Output one {'row_id','feature'} per input row in input order, None when empty.",
"Produce Python function prior_rolling_mean(rows,max_points,max_age_minutes) with strict temporal leakage prevention. Rows sharing a timestamp are sequenced by seq and then row_id, so lower sequence peers may precede higher ones. History is entity-local. Exclude stale history older than the inclusive minute threshold, truncate to max_points newest eligible rows, average their numeric value, and never include the current row. The output must preserve input order and contain exactly row_id and feature.",
"Implement the following hidden-test contract using only Python's standard library: prior_rolling_mean(rows, max_points, max_age_minutes). Treat (entity, ts, seq default 0, row_id) as deterministic processing order. Before appending the current row to history, average the newest max_points same-entity prior rows whose timestamp age is <= max_age_minutes. Return in original order. max_points can be zero; missing history gives None. No pandas, files, network, globals, or examples are needed.",
"Return executable Python defining prior_rolling_mean for an online feature store. It receives an unsorted list of row dicts. Prevent target/future leakage by calculating each feature only from previously ordered rows of that entity; tie order is timestamp, optional sequence, row_id. Apply an inclusive freshness horizon and finite point window, then restore original order. Output list elements are {'row_id': original id, 'feature': float mean or None}.",
"Write a robust pure-Python prior_rolling_mean. The semantic rules are entity isolation, deterministic chronological order (ts, seq=0 when absent, row_id), no self/current-row contribution, no future contribution, inclusive max_age_minutes freshness, last max_points only, and original-order output. Empty history or max_points==0 produces None. Input values may be negative or floats.",
"Define prior_rolling_mean(rows, max_points, max_age_minutes) as a reusable standard-library function. Rows can arrive out of order and multiple entities/timestamp ties are possible. At every row's causal position, use only earlier same-entity rows, limited both by inclusive age and by most recent max_points; then emit the mean or None. Do not mutate caller rows. Return exactly row_id/feature dicts in incoming order."
]
HOLDOUT_PROMPTS=[
"Implement pure Python prior_rolling_mean for a streaming feature snapshot. The input order is arbitrary. Causal order within an entity is ts, then seq (default zero), then row_id. A row's feature is the mean of up to max_points eligible predecessors with age not exceeding max_age_minutes; the row itself and all successors are forbidden. Return row_id plus feature in original order, or None without predecessors.",
"Write only executable stdlib Python exposing prior_rolling_mean(rows,max_points,max_age_minutes). Enforce same-entity causal history, tie ordering by (ts,seq,row_id), an inclusive freshness cutoff, a newest-N window, and zero leakage from the current/future rows. Preserve incoming output order and represent no feature as None.",
"Create prior_rolling_mean for leakage-safe feature engineering on shuffled row dictionaries. Before each row is admitted to its entity history, average the newest max_points earlier items no more than max_age_minutes old. Earlier means ordered by timestamp, seq default 0, then row_id. Keep output order identical to input and return {'row_id','feature'}; max_points zero yields None.",
"Produce standard-library Python function prior_rolling_mean. It must handle multiple entities, equal timestamp rows, negative values, unsorted input, stale history, and max_points=0. Use only causal predecessors ordered by ts/seq/row_id, never the current row, and return exact row_id/feature records in original order."
]
CAL_PROMPTS=[
"Define stdlib-only asof_join(events, readings, max_age_minutes). For each event, select the newest reading from the same entity with reading ts <= event ts and age <= max_age_minutes inclusive; equal reading timestamps choose highest seq (missing seq 0). Preserve event order and return {'event_id','value'}, None if none. Return executable Python only.",
"Implement leakage-safe point-in-time lookup asof_join using plain Python lists of dicts. Match same entity, never a future reading, enforce inclusive freshness, break timestamp ties by seq, and emit event_id/value in the events' original order."
]

PRIOR_REF="""def prior_rolling_mean(rows, max_points, max_age_minutes):
    ordered=sorted(rows,key=lambda r:(r['entity'],r['ts'],r.get('seq',0),r['row_id']))
    history={}
    value_by_id={}
    for r in ordered:
        h=history.setdefault(r['entity'],[])
        eligible=[x for x in h if x['ts']<=r['ts'] and r['ts']-x['ts']<=max_age_minutes]
        eligible=eligible[-max_points:] if max_points>0 else []
        value_by_id[r['row_id']]=(sum(x['value'] for x in eligible)/len(eligible)) if eligible else None
        h.append(r)
    return [{'row_id':r['row_id'],'feature':value_by_id[r['row_id']]} for r in rows]
"""
ASOF_REF="""def asof_join(events, readings, max_age_minutes):
    out=[]
    for e in events:
        candidates=[r for r in readings if r['entity']==e['entity'] and r['ts']<=e['ts'] and e['ts']-r['ts']<=max_age_minutes]
        candidates.sort(key=lambda r:(r['ts'],r.get('seq',0)))
        out.append({'event_id':e['event_id'],'value':candidates[-1]['value'] if candidates else None})
    return out
"""

def sha_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def jwrite(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def api_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read().decode())
def text_url(url,limit=3_000_000):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:
        b=r.read(limit+1)
        if len(b)>limit: raise RuntimeError('metadata response too large')
        return b.decode('utf-8','replace')
def direct_download(repo,rev,name,dst:Path):
    url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true'
    req=urllib.request.Request(url,headers=UA)
    h=hashlib.sha256(); n=0
    with urllib.request.urlopen(req,timeout=180) as r, dst.open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b: break
            f.write(b); h.update(b); n+=len(b)
    return {'path':str(dst),'size':n,'sha256':h.hexdigest()}
def sibling_meta(info,name):
    for x in info.get('siblings') or []:
        if x.get('rfilename')==name:
            l=x.get('lfs') or {}
            return {'size':l.get('size') or x.get('size'),'sha256':l.get('sha256') or l.get('oid'),'blob_id':x.get('blobId')}
    return {}

def build_prior_fixtures():
    def ref(rows,k,age):
        ns={}; hist={}; feat={}
        for r in sorted(rows,key=lambda r:(r['entity'],r['ts'],r.get('seq',0),r['row_id'])):
            h=hist.setdefault(r['entity'],[]); ok=[x for x in h if r['ts']-x['ts']<=age]; ok=ok[-k:] if k>0 else []
            feat[r['row_id']]=sum(x['value'] for x in ok)/len(ok) if ok else None; h.append(r)
        return [{'row_id':r['row_id'],'feature':feat[r['row_id']]} for r in rows]
    f=[]; manual=[
      ([{'row_id':'r0','entity':'a','ts':10,'seq':0,'value':100.}],3,30),
      ([{'row_id':'r1','entity':'a','ts':11,'seq':0,'value':3.},{'row_id':'r0','entity':'a','ts':10,'seq':0,'value':1.}],3,30),
