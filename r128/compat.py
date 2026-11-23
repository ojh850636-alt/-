"""Explicit audited legacy BERT LayerNorm gamma/beta alias; no value edits.

Only takes .LayerNorm.gamma/.beta keys absent from destination model, never changes
weights or injects state. Original tensor byte hashes remain unchanged.
"""
def bridge_layernorm_aliases(state,destination_keys):
    dst=set(destination_keys)
    out={};changes=[]
    for k,v in state.items():
        replacement=None
        if '.LayerNorm.' in k and k.endswith('.gamma'):
            replacement=k[:-5]+'weight'
        elif '.LayerNorm.' in k and k.endswith('.beta'):
            replacement=k[:-4]+'bias'
        name=k
        if replacement is not None and replacement in dst and k not in dst:
            name=replacement;changes.append({'old':k,'new':name})
        if name in out:
            raise ValueError('STATE_ALIAS_COLLISION:'+name)
        out[name]=v
    return out, changes
