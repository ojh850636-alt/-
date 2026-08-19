from __future__ import annotations
import json,re
FIELDS={"company_name","address","date","total_amount","line_items"}
def parse_output(text):
    if isinstance(text,dict): return text
    if not isinstance(text,str): return None
    s=text.strip()
    try: return json.loads(s)
    except Exception:
        m=re.search(r"\{.*\}",s,re.S)
        if not m: return None
        try: return json.loads(m.group(0))
        except Exception: return None
def score(text,target):
    obj=parse_output(text)
    out={"json_valid":0,"schema_exact":0,"field_correct":0.0,"line_items_exact":0,"null_discipline":0,"strict":0}
    if not isinstance(obj,dict): return out
    out["json_valid"]=1; out["schema_exact"]=int(set(obj)==FIELDS)
    correct=0
    for k in ["company_name","address","date"]: correct+=int(obj.get(k)==target.get(k))
    tv=target.get("total_amount"); ov=obj.get("total_amount")
    total_ok=(tv is None and ov is None) or (isinstance(tv,(int,float)) and isinstance(ov,(int,float)) and abs(float(tv)-float(ov))<0.005)
    correct+=int(total_ok)
    li_ok=obj.get("line_items")==target.get("line_items"); correct+=int(li_ok)
    out["field_correct"]=correct/5; out["line_items_exact"]=int(li_ok)
    out["null_discipline"]=int(all((target.get(k) is not None) or (obj.get(k) is None) for k in ["company_name","address","date","total_amount"]))
    out["strict"]=int(out["schema_exact"] and correct==5 and out["null_discipline"])
    return out
