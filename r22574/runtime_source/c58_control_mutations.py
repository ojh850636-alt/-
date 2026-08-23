from __future__ import annotations
from collections import defaultdict
import random
import torch


def snapshot_adapter(model, adapter_name: str):
    snap={}
    for n,m in model.named_modules():
        if hasattr(m,'lora_A') and adapter_name in m.lora_A and hasattr(m,'lora_B') and adapter_name in m.lora_B:
            snap[n]=(m.lora_A[adapter_name].weight.detach().cpu().clone(),m.lora_B[adapter_name].weight.detach().cpu().clone())
    if not snap:
        raise RuntimeError(f'no live LoRA modules for {adapter_name}')
    return snap


def restore_adapter(model, adapter_name: str, snap):
    seen=set()
    for n,m in model.named_modules():
        if n in snap:
            m.lora_A[adapter_name].weight.data.copy_(snap[n][0].to(m.lora_A[adapter_name].weight.device,dtype=m.lora_A[adapter_name].weight.dtype))
            m.lora_B[adapter_name].weight.data.copy_(snap[n][1].to(m.lora_B[adapter_name].weight.device,dtype=m.lora_B[adapter_name].weight.dtype))
            seen.add(n)
    if seen != set(snap):
        raise RuntimeError(f'adapter restore mismatch missing={sorted(set(snap)-seen)[:5]}')


def randomize_inplace(model, adapter_name: str, snap, seed: int, mode: str):
    restore_adapter(model,adapter_name,snap)
    model.set_adapter(adapter_name)
    rng=torch.Generator(device='cpu'); rng.manual_seed(seed)
    if mode=='random_sign':
        for n,m in model.named_modules():
            if n not in snap: continue
            for container in (m.lora_A,m.lora_B):
                w=container[adapter_name].weight
                sign=torch.where(torch.rand(tuple(w.shape),generator=rng)>0.5,torch.ones(tuple(w.shape)),-torch.ones(tuple(w.shape))).to(w.device,dtype=w.dtype)
                w.data.mul_(sign)
    elif mode=='layer_shuffle':
        byshape=defaultdict(list)
        for n,(a,b) in snap.items():
            byshape[(tuple(a.shape),'A')].append((n,a))
            byshape[(tuple(b.shape),'B')].append((n,b))
        target={}
        for (_shape,side),items in byshape.items():
            names=[n for n,_ in items]; srcs=[x for _,x in items]
            order=list(range(len(items))); random.Random(seed+len(items)).shuffle(order)
            for dst_i,src_i in enumerate(order): target[(names[dst_i],side)]=srcs[src_i]
        for n,m in model.named_modules():
            if n not in snap: continue
            m.lora_A[adapter_name].weight.data.copy_(target[(n,'A')].to(m.lora_A[adapter_name].weight.device,dtype=m.lora_A[adapter_name].weight.dtype))
            m.lora_B[adapter_name].weight.data.copy_(target[(n,'B')].to(m.lora_B[adapter_name].weight.device,dtype=m.lora_B[adapter_name].weight.dtype))
    else:
        raise ValueError(mode)


def dose_inplace(model, adapter_name: str, snap, scale: float):
    restore_adapter(model,adapter_name,snap)
    model.set_adapter(adapter_name)
    for n,m in model.named_modules():
        if n in snap:
            m.lora_B[adapter_name].weight.data.copy_((snap[n][1]*scale).to(m.lora_B[adapter_name].weight.device,dtype=m.lora_B[adapter_name].weight.dtype))
