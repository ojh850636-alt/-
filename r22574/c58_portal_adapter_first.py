#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, shutil, tempfile
from pathlib import Path
from collections import defaultdict
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file
from portallib import PortalModel

REPO='RampPublic/portal-qwen3-1.7b'
TAG='v0.2.0'
EXPECTED_COMMIT='be09be533b5c0418ad20269f19ebb63e9efbc330'
EXPECTED_SHA='732286e119c396c62b8c1d6b115f3ae6eec951a47eaa9bcdf0fee65564ff9688'
EXPECTED_BASE='Qwen/Qwen3-1.7B'
EXPECTED_BASE_REV='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
EXPECTED_TASKS=['truthfulqa','rte','cb','copa','wic','wsc','boolq','arc_easy','arc_challenge','hellaswag','openbookqa','winogrande','commonsense_qa','sciq']
OUT=Path(os.environ.get('C58_OUT','c58_out'))
ESCROW=Path(os.environ.get('C58_ESCROW','c58_escrow'))
OUT.mkdir(parents=True,exist_ok=True)

def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def tensor_hash(t:torch.Tensor):
    a=t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(a).hexdigest()

def state_digest(state):
    h=hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(b'\0'); h.update(bytes.fromhex(tensor_hash(state[k])))
    return h.hexdigest()

def pair_key(k):
    if '.lora_A.' in k: return k.replace('.lora_A.','.lora_X.'), 'A'
    if '.lora_B.' in k: return k.replace('.lora_B.','.lora_X.'), 'B'
    if 'lora_A' in k: return k.replace('lora_A','lora_X'), 'A'
    if 'lora_B' in k: return k.replace('lora_B','lora_X'), 'B'
    return None,None

def build_pairs(state):
    pairs=defaultdict(dict); extra=[]
    for k,v in state.items():
        pk,side=pair_key(k)
        if side: pairs[pk][side]=v.float().cpu()
        else: extra.append(k)
    good={k:v for k,v in pairs.items() if set(v)=={'A','B'}}
    return good, sorted(extra), sorted(k for k,v in pairs.items() if set(v)!={'A','B'})

def op_norm_sq(A,B):
    BtB=B.T@B; AAt=A@A.T
    return float(torch.sum(BtB*AAt.T).item())

def op_inner(A1,B1,A2,B2):
    x=B1.T@B2; y=A2@A1.T
    return float(torch.sum(x*y.T).item())

def site_info(pk):
    proj='unknown'
    for p in ('q_proj','v_proj','k_proj','o_proj','gate_proj','up_proj','down_proj'):
        if p in pk: proj=p; break
    layer=None
    toks=pk.split('.')
    for i,t in enumerate(toks[:-1]):
        if t in ('layers','layer','h'):
            try: layer=int(toks[i+1]); break
            except Exception: pass
    return layer,proj

def summarize_adapter(path:Path):
    st=load_file(str(path))
    pairs,extra,incomplete=build_pairs(st)
    items=[]; total=0.0; zero=0; nonfinite=0; ranks=[]; proj_energy=defaultdict(float); layer_energy=defaultdict(float)
    for pk,ab in sorted(pairs.items()):
        A,B=ab['A'],ab['B']
        if not torch.isfinite(A).all() or not torch.isfinite(B).all(): nonfinite+=1
        na=float(torch.linalg.vector_norm(A).item()); nb=float(torch.linalg.vector_norm(B).item())
        eff=na*nb
        if eff==0.0: zero+=1
        rA=int(torch.linalg.matrix_rank(A).item()); rB=int(torch.linalg.matrix_rank(B).item())
        ranks.append([rA,rB])
        en=op_norm_sq(A,B); total+=en
        layer,proj=site_info(pk); proj_energy[proj]+=en; layer_energy[str(layer)]+=en
        items.append({'site':pk,'A_shape':list(A.shape),'B_shape':list(B.shape),'effective_strength':eff,'operator_energy':en,'rank_A':rA,'rank_B':rB,'layer':layer,'projection':proj})
    for x in items:
        x['operator_energy_ratio']=x['operator_energy']/total if total else 0.0
    return {
        'tensor_count':len(st),'state_digest':state_digest(st),'pair_count':len(pairs),'extra_tensor_keys':extra,
        'incomplete_pair_keys':incomplete,'zero_effective_pairs':zero,'nonfinite_pairs':nonfinite,
        'effective_rank_min':min(min(x) for x in ranks) if ranks else None,
        'effective_rank_max':max(max(x) for x in ranks) if ranks else None,
        'operator_energy_total':total,
        'projection_energy_ratio':{k:v/total for k,v in sorted(proj_energy.items())} if total else {},
        'layer_energy_ratio':{k:v/total for k,v in sorted(layer_energy.items(),key=lambda kv:int(kv[0]) if kv[0]!='None' else -1)} if total else {},
        'sites':items,
        '_pairs':pairs,
    }

def cosine_matrix(vectors):
    names=list(vectors); out={}
    for a in names:
        row={}; va=vectors[a].float().flatten(); na=float(torch.linalg.vector_norm(va).item())
        for b in names:
            vb=vectors[b].float().flatten(); nb=float(torch.linalg.vector_norm(vb).item())
            row[b]=float(torch.dot(va,vb).item()/(na*nb)) if na and nb else None
        out[a]=row
    return out

def operator_cosine(summaries):
    names=list(summaries); norms={t:summaries[t]['operator_energy_total'] for t in names}; out={}
    for a in names:
        row={}; pa=summaries[a]['_pairs']
        for b in names:
            pb=summaries[b]['_pairs']; common=sorted(set(pa)&set(pb)); inner=0.0
            for site in common:
                inner += op_inner(pa[site]['A'],pa[site]['B'],pb[site]['A'],pb[site]['B'])
            den=math.sqrt(norms[a]*norms[b]) if norms[a] and norms[b] else 0.0
            row[b]=inner/den if den else None
        out[a]=row
    return out

def escrow_exports(exports:Path, viable:bool):
    if ESCROW.exists(): shutil.rmtree(ESCROW)
    manifest={'schema':'R22574_C58_DERIVATIVE_ESCROW_V1','created':False,'files':[],'native_portal_source_included':False,'base_included':False}
    if not viable:
        return manifest
    ESCROW.mkdir(parents=True,exist_ok=True)
    for task in ('rte','copa'):
        src=exports/task; dst=ESCROW/task
        shutil.copytree(src,dst)
    for p in sorted(x for x in ESCROW.rglob('*') if x.is_file()):
        manifest['files'].append({'path':p.relative_to(ESCROW).as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    manifest['created']=True
    manifest['total_bytes']=sum(x['bytes'] for x in manifest['files'])
    (ESCROW/'ESCROW_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest

def main():
    api=HfApi(); info=api.model_info(REPO, revision=TAG, files_metadata=True)
    resolved=info.sha
    if resolved!=EXPECTED_COMMIT: raise SystemExit(f'commit drift {resolved}')
    with tempfile.TemporaryDirectory(prefix='c58-portal-') as td:
        td=Path(td); src=td/'portal'; exports=td/'exports'; exports.mkdir()
        snapshot_download(repo_id=REPO,revision=resolved,allow_patterns=['config.json','model.safetensors','README.md','metrics.json'],local_dir=src)
        sf=src/'model.safetensors'
        if sha256_file(sf)!=EXPECTED_SHA: raise SystemExit('native safetensors hash mismatch')
        cfg=json.loads((src/'config.json').read_text())
        tasks=cfg.get('tasks') or EXPECTED_TASKS
        if list(tasks)!=EXPECTED_TASKS: raise SystemExit(f'task list drift {tasks}')
        base_id=cfg.get('base_model_name_or_path') or cfg.get('base_model')
        base_rev=cfg.get('base_model_revision')
        if base_id!=EXPECTED_BASE or base_rev!=EXPECTED_BASE_REV: raise SystemExit(f'base drift {base_id}@{base_rev}')
        portal=PortalModel.from_pretrained(src)
        native=load_file(str(sf))
        latent_candidates=[(k,v) for k,v in native.items() if 'task_latent' in k]
        if len(latent_candidates)!=1: raise SystemExit(f'expected one task latent tensor, got {[k for k,_ in latent_candidates]}')
        latent_key,latent=latent_candidates[0]
        if latent.ndim!=2 or latent.shape[0]!=len(tasks): raise SystemExit(f'latent shape {tuple(latent.shape)}')
        latent_vectors={task:latent[i].float().cpu() for i,task in enumerate(tasks)}
        group_summary={}
        for prefix in ('core.','alignment.'):
            vals=[(k,v) for k,v in native.items() if k.startswith(prefix)]
            group_summary[prefix[:-1]]={
                'tensor_count':len(vals),'parameter_count':sum(v.numel() for _,v in vals),
                'l2_norm':math.sqrt(sum(float(torch.sum(v.float()**2).item()) for _,v in vals)),
                'group_digest':hashlib.sha256(''.join(k+tensor_hash(v) for k,v in sorted(vals)).encode()).hexdigest()
            }
        summaries={}
        for task in tasks:
            out=exports/task; portal.export_peft(task,out)
            summaries[task]=summarize_adapter(out/'adapter_model.safetensors')
        out2=exports/'rte_repeat'; portal.export_peft('rte',out2)
        repeat=summarize_adapter(out2/'adapter_model.safetensors')
        export_deterministic=(repeat['state_digest']==summaries['rte']['state_digest'])
        opcos=operator_cosine(summaries); latcos=cosine_matrix(latent_vectors)
        clean_summaries={task:{k:v for k,v in s.items() if k!='_pairs'} for task,s in summaries.items()}
        rte=clean_summaries['rte']
        viable=(export_deterministic and rte['pair_count']==56 and rte['zero_effective_pairs']==0 and rte['nonfinite_pairs']==0 and rte['effective_rank_min']==8 and rte['effective_rank_max']==8 and not rte['extra_tensor_keys'] and not rte['incomplete_pair_keys'])
        escrow=escrow_exports(exports,viable)
        report={
          'schema':'R22574_C58_PORTAL_ADAPTER_FIRST_RAWFREE_V2',
          'source':{'repo':REPO,'tag':TAG,'resolved_commit':resolved,'native_sha256':EXPECTED_SHA,'base':base_id,'base_revision':base_rev,'tasks':tasks},
          'native':{'tensor_count':len(native),'task_latent_key':latent_key,'task_latent_shape':list(latent.shape),'task_latent_cosine':latcos,'groups':group_summary},
          'exports':clean_summaries,'operator_cosine':opcos,'rte_export_repeat_state_digest':repeat['state_digest'],'export_deterministic':export_deterministic,
          'base_ingress_authorized':viable,
          'verdict':'PASS_ADAPTER_FIRST_VIABLE_AUTHORIZE_BASE' if viable else 'CLOSED_E1_OPERATOR_DEGENERATE_BASE_BYTES_ZERO',
          'derivative_escrow':{k:v for k,v in escrow.items() if k!='files'} | {'file_manifest_digest':hashlib.sha256(json.dumps(escrow.get('files',[]),sort_keys=True,separators=(',',':')).encode()).hexdigest()},
          'raw_export_contract':{'native_weights_in_rawfree_artifact':False,'exported_lora_weights_in_rawfree_artifact':False,'A_B_values_exported_in_rawfree_artifact':False,'reconstructable_deltaW_exported':False,'temporary_derivative_escrow_separate':bool(escrow.get('created'))}
        }
        (OUT/'C58_PORTAL_ADAPTER_FIRST_RAWFREE.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        shutil.rmtree(src,ignore_errors=True); shutil.rmtree(exports,ignore_errors=True)
        (OUT/'C58_ADAPTER_FIRST_CLEANUP.json').write_text(json.dumps({'native_source_removed':not src.exists(),'temporary_export_tree_removed':not exports.exists(),'derivative_escrow_created':bool(escrow.get('created')),'base_downloaded':False},indent=2)+'\n')
        print(report['verdict'])
if __name__=='__main__': main()
