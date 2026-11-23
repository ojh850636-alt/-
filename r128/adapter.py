"""Non-merging LoRA + trained-head controller; raw tensors are confined to isolate.

Operator geometry and intervention masks stay in memory only. DO NOT EXPORT matrices.
"""
from __future__ import annotations
from collections import defaultdict
import torch
import torch.nn.functional as F

PREFIX='base_model.model.'

def parse_lora_tensors(sd, expected_rank=8, alpha=16):
    if not sd:
        raise ValueError('EMPTY_ADAPTER')
    factors={}
    aux={}
    extra=[]
    for key,value in sd.items():
        if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('INVALID_TENSOR')
        if '.lora_A.' in key or '.lora_B.' in key:
            side='A' if '.lora_A.' in key else 'B'
            stem=key.split('.lora_'+side+'.')[0]
            if stem.startswith(PREFIX):stem=stem[len(PREFIX):]
            if stem in factors and side in factors[stem]:raise ValueError('DUPLICATE_FACTOR')
            factors.setdefault(stem,{})[side]=value.detach().cpu().to(torch.float32)
        elif '.modules_to_save.' in key:
            stem,tail=key.split('.modules_to_save.',1)
            if stem.startswith(PREFIX):stem=stem[len(PREFIX):]
            parts=tail.split('.')
            if len(parts)<2 or parts[-1] not in ('weight','bias'):
                raise ValueError('UNKNOWN_AUX_COMPONENT')
            target=stem+'.'+parts[-1]
            aux[target]=value.detach().cpu().to(torch.float32)
        else:
            extra.append(key)
    if extra:raise ValueError('UNSUPPORTED_TENSORS:'+str(len(extra)))
    if not factors:raise ValueError('NO_LORA_PAIR')
    for name,parts in factors.items():
        if set(parts)!={'A','B'}:raise ValueError('INCOMPLETE_PAIR:'+name)
        a,b=parts['A'],parts['B']
        if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=expected_rank or b.shape[1]!=expected_rank or b.shape[0]<=0 or a.shape[1]<=0:
            raise ValueError('RANK_SHAPE_MISMATCH:'+name)
    if 'classifier.weight' not in aux or 'classifier.bias' not in aux:
        raise ValueError('TRAINED_HEAD_NOT_FOUND')
    return factors,aux

def stat_operator(factors,alpha=16):
    """Small-Gram, no dense delta construction; scalar-only return."""
    out=[]
    for site,parts in sorted(factors.items()):
        a,b=parts['A'],parts['B']
        scale=alpha/a.shape[0]
        gb=b.T@b; ga=a@a.T
        sq=float((scale**2)*(gb*ga.T).sum())
        out.append({'site':site, 'rank':int(a.shape[0]), 'operator_norm_sq':sq,
                    'zero':bool(sq<1e-14),'family':('ATTENTION' if '.attention.' in site else 'MLP' if '.intermediate.' in site or '.output.' in site else 'POOLER')})
    return out

def randomized_b(factors,mode,seed=128):
    if mode not in ('RANDOM','SHUFFLED','ACTUAL'):raise ValueError('UNSUPPORTED_MODE')
    if mode=='ACTUAL':return {k:(v['A'],v['B']) for k,v in factors.items()}
    g=torch.Generator(device='cpu');g.manual_seed(seed)
    grouped=defaultdict(list)
    for k,v in factors.items():grouped[tuple(v['B'].shape)].append(k)
    out={}
    for shape,sites in grouped.items():
        sites=sorted(sites)
        order=torch.randperm(len(sites),generator=g).tolist()
        if len(sites)>1 and all(i==j for i,j in enumerate(order)):
            order=order[1:]+order[:1]
        for i,k in enumerate(sites):
            a,b=factors[k]['A'],factors[k]['B']
            if mode=='SHUFFLED':
                replacement=factors[sites[order[i]]]['B'].clone()
                if sites[order[i]]==k:
                    replacement=replacement[torch.randperm(replacement.shape[0],generator=g)]
            else:
                replacement=torch.randn(b.shape,generator=g,dtype=b.dtype)
            replacement*=torch.linalg.norm(b)/torch.linalg.norm(replacement).clamp(min=1e-30)
            out[k]=(a,replacement)
    return out

def selection(site,scope):
    if scope in ('ALL','NONE'):return scope=='ALL'
    if scope=='ATTENTION':return '.attention.' in site
    if scope=='MLP':return ('.intermediate.' in site or '.output.' in site) and '.attention.' not in site
    if scope=='POOLER':return '.pooler.' in site
    if scope.startswith('LAYER_BAND_'):
        import re
        m=re.search(r'encoder\.layer\.(\d+)\.',site)
        if not m:return False
        i=int(m.group(1))
        return (i<4 if scope=='LAYER_BAND_EARLY' else 4<=i<8 if scope=='LAYER_BAND_MIDDLE' else i>=8 if scope=='LAYER_BAND_LATE' else False)
    raise ValueError('UNKNOWN_SCOPE:'+scope)

class Controller:
    def __init__(self,model,factors,aux,alpha=16):
        self.model=model
        self.factors=factors
        self.aux=aux
        self.alpha=alpha
        self.handles=[]
        self.randomization='ACTUAL';self.scope='NONE';self.dose=0
        self.suppressed=set()
        mods=dict(model.named_modules())
        self.base_head={k:v.detach().clone() for k,v in model.named_parameters() if k.startswith('classifier.')}
        if set(self.base_head)!={'classifier.weight','classifier.bias'}:raise ValueError('MODEL_HEAD_UNKNOWN')
        for k,v in aux.items():
            if k not in self.base_head or self.base_head[k].shape!=v.shape:
                raise ValueError('AUX_HEAD_SHAPE_MISMATCH:'+k)
        self.ops={}
        for site,parts in factors.items():
            if site not in mods or not isinstance(mods[site],torch.nn.Linear):raise ValueError('SITE_NOT_LINEAR:'+site)
            m=mods[site]
            a,b=parts['A'],parts['B']
            if tuple(m.weight.shape)!=(b.shape[0],a.shape[1]):raise ValueError('SITE_WEIGHT_SHAPE:'+site)
            def make_hook(s):
                def hook(mod,args,output):
                    if self.dose==0 or s in self.suppressed or not selection(s,self.scope):return output
                    a_,b_=self.ops[s]
                    x=args[0]
                    return output+(self.alpha/a_.shape[0])*self.dose*F.linear(F.linear(x,a_.to(x.device,x.dtype)),b_.to(x.device,x.dtype))
                return hook
            self.handles.append(m.register_forward_hook(make_hook(site)))
    def condition(self,which,scope='ALL',dose=1.0,seed=128):
        if which not in ('BASE','HEAD_ONLY','LORA_ONLY','FULL','RANDOM_SAME_HEAD','SHUFFLED_SAME_HEAD'):
            raise ValueError('BAD_CONDITION')
        trained=which in ('HEAD_ONLY','FULL','RANDOM_SAME_HEAD','SHUFFLED_SAME_HEAD')
        with torch.no_grad():
            params=dict(self.model.named_parameters())
            for k,v in self.base_head.items():
                params[k].copy_(self.aux[k] if trained else v)
        self.dose=(0.0 if which in ('BASE','HEAD_ONLY') else float(dose))
        self.scope=('NONE' if not self.dose else scope)
        mode='RANDOM' if which=='RANDOM_SAME_HEAD' else 'SHUFFLED' if which=='SHUFFLED_SAME_HEAD' else 'ACTUAL'
        self.ops=randomized_b(self.factors,mode,seed)
    def close(self):
        self.condition('BASE')
        for h in self.handles:h.remove()
        self.handles=[]
