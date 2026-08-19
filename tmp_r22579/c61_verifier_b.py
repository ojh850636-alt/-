from __future__ import annotations
import json,decimal
D=decimal.Decimal
def _obj(text):
    if isinstance(text,dict): return text
    try: return json.loads(str(text).strip())
    except Exception:
        a=str(text).find("{"); b=str(text).rfind("}")
        if a<0 or b<a: return None
        try: return json.loads(str(text)[a:b+1])
        except Exception: return None
def integrity(text,target):
    x=_obj(text)
    r={"parse":0,"item_arithmetic":0,"target_total":0,"no_extra_top_fields":0,"null_safety":0,"integrity":0}
    if not isinstance(x,dict): return r
    r["parse"]=1
    allowed={"company_name","address","date","total_amount","line_items"}; r["no_extra_top_fields"]=int(set(x).issubset(allowed))
    items=x.get("line_items"); arithmetic=False
    if isinstance(items,list):
        try:
            s=D("0")
            for it in items:
                if not isinstance(it,dict) or set(it)!={"item_name","quantity","price"}: raise ValueError
                s += D(str(it["quantity"]))*D(str(it["price"]))
            ov=x.get("total_amount"); arithmetic=(ov is None) or abs(s-D(str(ov)))<=D("0.01")
        except Exception: arithmetic=False
    r["item_arithmetic"]=int(arithmetic)
    tv=target.get("total_amount"); ov=x.get("total_amount")
    r["target_total"]=int((tv is None and ov is None) or (tv is not None and isinstance(ov,(int,float)) and abs(float(tv)-float(ov))<0.005))
    r["null_safety"]=int(all(target.get(k) is not None or x.get(k) is None for k in ["company_name","address","date","total_amount"]))
    r["integrity"]=int(all(r[k] for k in ["parse","item_arithmetic","target_total","no_extra_top_fields","null_safety"]))
    return r
