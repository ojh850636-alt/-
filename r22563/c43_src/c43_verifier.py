from __future__ import annotations
import re,shlex

def canon(s:str)->str:
    s=s.strip().replace('\r','')
    s=re.sub(r'\s+',' ',s)
    return s

def lexical_tokens(s):
    try: return shlex.split(s,posix=True)
    except Exception: return []

def syntax_ok(s):
    if not s.strip(): return False
    try: shlex.split(s,posix=True); return True
    except Exception: return False

def exact_or_token_equiv(pred,target):
    if canon(pred)==canon(target): return True
    return lexical_tokens(pred)==lexical_tokens(target) and bool(lexical_tokens(target))

def semantic_signature(s):
    # independent shallow shell structure verifier; does not execute commands.
    x=canon(s)
    toks=lexical_tokens(x)
    if not toks: return {'ok':False}
    return {
      'ok':True,'tool':toks[0],'tokens':toks,
      'pipeline_count':x.count('|'), 'and_count':x.count('&&'),
      'redirect_out': bool(re.search(r'(^|\s)>\s*\S+',x)),
      'redirect_err': bool(re.search(r'2>\s*\S+',x)),
    }

def semantic_equiv(pred,target):
    a,b=semantic_signature(pred),semantic_signature(target)
    if not a.get('ok') or not b.get('ok'): return False
    return a==b
