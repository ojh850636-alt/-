      ([{'row_id':'a0','entity':'a','ts':1,'seq':0,'value':2.},{'row_id':'b0','entity':'b','ts':1,'seq':0,'value':100.},{'row_id':'a1','entity':'a','ts':2,'seq':0,'value':4.}],5,30),
      ([{'row_id':'old','entity':'a','ts':0,'seq':0,'value':2.},{'row_id':'edge','entity':'a','ts':15,'seq':0,'value':4.}],5,15),
      ([{'row_id':'old','entity':'a','ts':0,'seq':0,'value':2.},{'row_id':'late','entity':'a','ts':16,'seq':0,'value':4.}],5,15),
      ([{'row_id':'s2','entity':'a','ts':5,'seq':2,'value':30.},{'row_id':'s0','entity':'a','ts':5,'seq':0,'value':10.},{'row_id':'s1','entity':'a','ts':5,'seq':1,'value':20.}],2,10),
      ([{'row_id':'x0','entity':'x','ts':1,'seq':0,'value':1.},{'row_id':'x1','entity':'x','ts':2,'seq':0,'value':2.},{'row_id':'x2','entity':'x','ts':3,'seq':0,'value':9.}],1,100),
      ([{'row_id':'x0','entity':'x','ts':1,'seq':0,'value':1.},{'row_id':'x1','entity':'x','ts':2,'seq':0,'value':2.}],0,100),
    ]
    i=0
    for rows,k,a in manual: f.append({'id':f'pr{i:03d}','rows':rows,'max_points':k,'max_age_minutes':a,'expected':ref(rows,k,a)}); i+=1
    rng=random.Random(22587)
    while len(f)<72:
        ents=['a','b','c'][:rng.randint(1,3)]; rows=[]; rid=0
        for ent in ents:
            t=rng.randint(0,5)
            for _ in range(rng.randint(2,6)):
                t+=rng.choice([0,1,2,5,16]); rows.append({'row_id':f'r{rid}','entity':ent,'ts':t,'seq':rng.randint(0,3),'value':float(rng.randint(-9,20))}); rid+=1
        rng.shuffle(rows); k=rng.randint(0,4); age=rng.choice([0,1,3,15,30])
        f.append({'id':f'pr{i:03d}','rows':rows,'max_points':k,'max_age_minutes':age,'expected':ref(rows,k,age)}); i+=1
    return f

def build_asof_fixtures():
    def ref(events,reads,age):
        out=[]
        for e in events:
            c=[r for r in reads if r['entity']==e['entity'] and r['ts']<=e['ts'] and e['ts']-r['ts']<=age]; c.sort(key=lambda r:(r['ts'],r.get('seq',0)))
            out.append({'event_id':e['event_id'],'value':c[-1]['value'] if c else None})
        return out
    f=[]; rng=random.Random(22584); i=0
    manual=[
      ([{'event_id':'e','entity':'a','ts':10}],[{'entity':'a','ts':11,'seq':0,'value':9},{'entity':'a','ts':9,'seq':0,'value':3}],15),
      ([{'event_id':'e','entity':'a','ts':10}],[{'entity':'b','ts':10,'seq':0,'value':9}],15),
      ([{'event_id':'e','entity':'a','ts':15}],[{'entity':'a','ts':0,'seq':0,'value':2}],15),
      ([{'event_id':'e','entity':'a','ts':16}],[{'entity':'a','ts':0,'seq':0,'value':2}],15),
      ([{'event_id':'e','entity':'a','ts':5}],[{'entity':'a','ts':5,'seq':2,'value':20},{'entity':'a','ts':5,'seq':0,'value':10}],10),
    ]
    for ev,rd,a in manual: f.append({'id':f'as{i:03d}','events':ev,'readings':rd,'max_age_minutes':a,'expected':ref(ev,rd,a)}); i+=1
    while len(f)<64:
        ents=['a','b','c'][:rng.randint(1,3)]; reads=[]
        for ent in ents:
            for _ in range(rng.randint(1,5)): reads.append({'entity':ent,'ts':rng.randint(0,40),'seq':rng.randint(0,2),'value':rng.randint(-5,20)})
        ev=[{'event_id':f'e{j}','entity':rng.choice(ents),'ts':rng.randint(0,40)} for j in range(rng.randint(1,5))]
        rng.shuffle(reads); age=rng.choice([0,1,5,15,30]); f.append({'id':f'as{i:03d}','events':ev,'readings':reads,'max_age_minutes':age,'expected':ref(ev,reads,age)}); i+=1
    return f

ALLOWED={'bisect','datetime','typing','math','collections','statistics'}; BANNED_CALLS={'open','eval','exec','compile','__import__','input','breakpoint','help'}; BANNED_ROOTS={'os','sys','subprocess','socket','requests','urllib','pathlib','shutil','glob','pickle','marshal','ctypes'}
def extract_code(text):
    t=text.strip()
    if '```' in t:
        parts=t.split('```'); bs=[]
        for i in range(1,len(parts),2):
            b=parts[i]
            if b.lstrip().startswith('python'): b=b.lstrip()[6:].lstrip('\r\n')
            bs.append(b)
        if bs: t=max(bs,key=len)
    return t.strip()
def static_code(code,func):
    tree=ast.parse(code); funcs=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if func not in funcs: raise ValueError('MISSING_FUNC')
    for top in tree.body:
        if isinstance(top,ast.Expr) and isinstance(getattr(top,'value',None),(ast.Constant,)) and isinstance(top.value.value,str): continue
        if not isinstance(top,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.AsyncFunctionDef,ast.Assign,ast.AnnAssign)): raise ValueError('TOPLEVEL_EXEC')
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            names=[a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else [(n.module or '').split('.')[0]]
            if any(x and x not in ALLOWED for x in names): raise ValueError('IMPORT_BANNED')
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in BANNED_CALLS: raise ValueError('CALL_BANNED')
        if isinstance(n,ast.Name) and n.id in BANNED_ROOTS: raise ValueError('NAME_BANNED')
    return sum(1 for _ in ast.walk(tree))
def verify_code(code,func,fixtures,timeout=8):
    try: nodes=static_code(code,func)
    except Exception as e: return {'pass':False,'passed':0,'total':len(fixtures),'static_error':str(e),'ast_nodes':None}
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); (td/'c.py').write_text(code); (td/'f.json').write_text(json.dumps(fixtures,separators=(',',':')))
        if func=='prior_rolling_mean': call="m.prior_rolling_mean(f['rows'],f['max_points'],f['max_age_minutes'])"
        else: call="m.asof_join(f['events'],f['readings'],f['max_age_minutes'])"
        harness=f'''import importlib.util,json,sys\ns=importlib.util.spec_from_file_location("c",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nfx=json.load(open(sys.argv[2]));out=[]\ndef norm(x):\n  if isinstance(x,float): return round(x,12)\n  if isinstance(x,list): return [norm(v) for v in x]\n  if isinstance(x,dict): return {{k:norm(v) for k,v in x.items()}}\n  return x\nfor f in fx:\n  try:\n    got={call};out.append(norm(got)==norm(f['expected']))\n  except Exception: out.append(False)\nprint(json.dumps(out))\n'''
        (td/'h.py').write_text(harness)
        try: cp=subprocess.run([sys.executable,'-I',str(td/'h.py'),str(td/'c.py'),str(td/'f.json')],capture_output=True,text=True,timeout=timeout,cwd=td)
        except subprocess.TimeoutExpired: return {'pass':False,'passed':0,'total':len(fixtures),'runtime_error':'TIMEOUT','ast_nodes':nodes}
        if cp.returncode!=0: return {'pass':False,'passed':0,'total':len(fixtures),'runtime_error':'NONZERO','ast_nodes':nodes}
        try: vals=json.loads(cp.stdout.strip().splitlines()[-1]); passed=sum(bool(x) for x in vals)
        except Exception: passed=0
        return {'pass':passed==len(fixtures),'passed':passed,'total':len(fixtures),'ast_nodes':nodes}

def prompt_text(tok,user):
    return tok.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
def outcome_from_text(text,func,fixtures):
    code=extract_code(text); r=verify_code(code,func,fixtures); r.update({'completion_sha256':sha_bytes(text.encode()),'code_sha256':sha_bytes(code.encode()),'completion_chars':len(text),'code_chars':len(code)}); return r

def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from safetensors import safe_open
    torch.set_num_threads(min(4,os.cpu_count() or 2)); torch.manual_seed(22587)
    start=time.time(); provenance={}; source_consumed=False; raw_files=[]
    # Atomic metadata discovery -> immutable pin before any weight GET.
    info=api_json('https://huggingface.co/api/models/'+ADAPTER_REPO+'?blobs=true'); arev=info.get('sha')
    if not re.fullmatch(r'[0-9a-f]{40}',str(arev)): raise RuntimeError('NO_IMMUTABLE_ADAPTER_SHA')
