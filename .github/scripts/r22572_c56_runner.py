from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, shutil, sys
from collections import defaultdict
from pathlib import Path

SEED=22572
ADAPTER_REPO='Hanno-Labs/bosun-xs'
ADAPTER_REV='31129f876462c8b0ec3f360b4c82ab03c430f43e'
BASE_REPO='Qwen/Qwen3-Reranker-0.6B'
EXPECTED_YES_ID=9693
EXPECTED_NO_ID=2152
EXPECTED_MAX_LEN=3072

RULE_TEXT={
 'same_account':'Connected only if both findings refer to the same account identifier.',
 'same_city':'Connected only if both findings concern the same city.',
 'not_same_entity':'Connected only if the two findings do NOT concern the same named entity.',
 'same_entity':'Connected only if both findings concern the same named entity.',
 'same_status':'Connected only if both findings report the same status.',
 'same_account_diff_city':'Connected only if both findings refer to the same account identifier AND they concern different cities.',
 'a_supersedes_b':'Connected only if Finding A explicitly supersedes the release described by Finding B.',
}
QUERY='These two findings share the specified relationship.'

def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canon(obj)->bytes:return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()

def fact_text(which,entity,account,city,status,amount,sector,version=None,supersedes=None):
    s=(f"Finding {which} reports entity {entity}, account {account}, city {city}, "
       f"status {status}, amount {amount:.1f} billion dollars, sector {sector}.")
    if version is not None:
        s += f" It describes release {version}."
    if supersedes is not None:
        s += f" It explicitly says release {version} supersedes release {supersedes}."
    return s

def parse_fact(s:str):
    pat=(r"entity ([A-Za-z0-9_-]+), account ([A-Za-z0-9_-]+), city ([A-Za-z0-9_-]+), "
         r"status ([A-Za-z0-9_-]+), amount ([0-9.]+) billion dollars, sector ([A-Za-z0-9_-]+)\.")
    m=re.search(pat,s); assert m,s
    d={'entity':m.group(1),'account':m.group(2),'city':m.group(3),'status':m.group(4),'amount':float(m.group(5)),'sector':m.group(6)}
    vm=re.search(r"It describes release ([A-Za-z0-9_.-]+)\.",s)
    if vm:d['version']=vm.group(1)
    sm=re.search(r"explicitly says release ([A-Za-z0-9_.-]+) supersedes release ([A-Za-z0-9_.-]+)\.",s)
    if sm:d['supersedes']=sm.group(2)
    return d

def verifier_b(case):
    a=parse_fact(case['a']); b=parse_fact(case['b']); r=case['rule_key']
    if r=='same_account':return int(a['account']==b['account'])
    if r=='same_city':return int(a['city']==b['city'])
    if r=='same_entity':return int(a['entity']==b['entity'])
    if r=='not_same_entity':return int(a['entity']!=b['entity'])
    if r=='same_status':return int(a['status']==b['status'])
    if r=='same_account_diff_city':return int(a['account']==b['account'] and a['city']!=b['city'])
    if r=='a_supersedes_b':return int(a.get('supersedes')==b.get('version'))
    raise KeyError(r)

def build_suite():
    rng=random.Random(SEED)
    entities=[f'Entity{c}' for c in 'ABCDEFGHJKLMNPQRSTUVWXYZ']
    accounts=[f'AC{100+i}' for i in range(80)]
    cities=['Lagos','Kano','Accra','Nairobi','Dakar','Tunis','Riga','Oslo']
    statuses=['OPEN','CLOSED','PAUSED']
    sectors=['banking','energy','logistics','health','telecom']
    cases=[]; cid=0
    def attrs(i):
        return dict(entity=entities[i%len(entities)],account=accounts[i%len(accounts)],city=cities[i%len(cities)],status=statuses[i%len(statuses)],amount=0.4+0.3*(i%8),sector=sectors[i%len(sectors)])
    def add(part,rule,a,b,label,group=None):
        nonlocal cid
        c={'id':f'C56-{cid:03d}','partition':part,'rule_key':rule,'instruction':RULE_TEXT[rule],'query':QUERY,'a':a,'b':b,'label':int(label),'pair_group':group};cid+=1
        assert verifier_b(c)==c['label'],(c,verifier_b(c));cases.append(c)
    # 64 rule-flip cases = 32 pairs, each same docs with one positive and one negative rule.
    for i in range(32):
        x=attrs(i); y=attrs(i+17)
        if i%2==0:
            y['account']=x['account']
            if y['city']==x['city']: y['city']=cities[(cities.index(x['city'])+1)%len(cities)]
            pos='same_account'; neg='same_city'
        else:
            y['city']=x['city']
            if y['account']==x['account']: y['account']=accounts[(i+33)%len(accounts)]
            pos='same_city'; neg='same_account'
        a=fact_text('A',**x); b=fact_text('B',**y); g=f'flip-{i:02d}'
        add('rule_flip',pos,a,b,1,g); add('rule_flip',neg,a,b,0,g)
    # 48 negation cases = 24 same docs evaluated under same_entity and not_same_entity.
    for i in range(24):
        x=attrs(40+i); y=attrs(64+i)
        same=(i%2==0)
        y['entity']=x['entity'] if same else entities[(entities.index(x['entity'])+3)%len(entities)]
        a=fact_text('A',**x);b=fact_text('B',**y);g=f'neg-{i:02d}'
        add('negation','same_entity',a,b,int(same),g)
        add('negation','not_same_entity',a,b,int(not same),g)
    # 48 conjunction cases, 24 positive and 24 near-miss negatives.
    for i in range(48):
        x=attrs(90+i); y=attrs(140+i)
        if i<24:
            y['account']=x['account']; y['city']=cities[(cities.index(x['city'])+1)%len(cities)]; lab=1
        elif i%2==0:
            y['account']=accounts[(i+51)%len(accounts)]; y['city']=cities[(cities.index(x['city'])+1)%len(cities)]; lab=0
        else:
            y['account']=x['account']; y['city']=x['city']; lab=0
        add('conjunction','same_account_diff_city',fact_text('A',**x),fact_text('B',**y),lab)
    # 48 directional cases: positive A supersedes B; negative is swapped order.
    for i in range(24):
        x=attrs(170+i); y=attrs(200+i); old=f'R{i%9+1}.0'; new=f'R{i%9+2}.0'
        xa=dict(x,version=new,supersedes=old); yb=dict(y,version=old,supersedes=None)
        a=fact_text('A',**xa);b=fact_text('B',**yb);g=f'dir-{i:02d}'
        add('directional','a_supersedes_b',a,b,1,g)
        # swap visible findings but preserve labels from rule direction
        add('directional','a_supersedes_b',fact_text('A',**yb),fact_text('B',**xa),0,g)
    # 48 confirmation: same-status rule, balanced.
    for i in range(48):
        x=attrs(230+i);y=attrs(280+i);same=i<24
        y['status']=x['status'] if same else statuses[(statuses.index(x['status'])+1)%len(statuses)]
        add('confirmation','same_status',fact_text('A',**x),fact_text('B',**y),int(same))
    assert len(cases)==256
    assert sum(c['label'] for c in cases)==128
    assert all(verifier_b(c)==c['label'] for c in cases)
    return cases

def mutation_smoke(cases):
    # independent verifier must reject label flips and directional swaps.
    flip_detect=sum(verifier_b(dict(c,label=1-c['label']))!=1-c['label'] for c in cases[:64])
    dirs=[c for c in cases if c['partition']=='directional']
    swap_detect=0
    for c in dirs[:24]:
        m=dict(c,a=c['b'].replace('Finding B','Finding A',1),b=c['a'].replace('Finding A','Finding B',1),label=1-c['label'])
        if verifier_b(m)==m['label']: swap_detect+=1
    return {'verifier_a_contract_pass':len(cases),'verifier_b_contract_pass':sum(verifier_b(c)==c['label'] for c in cases),'label_flip_mutation_detected':flip_detect,'direction_swap_mutation_pass':swap_detect}

def preseal(out:Path):
    out.mkdir(parents=True,exist_ok=True);cases=build_suite(); smoke=mutation_smoke(cases)
    public_cases=[{k:c[k] for k in ['id','partition','rule_key','instruction','query','a','b','label','pair_group']} for c in cases]
    suite_sha=sha_bytes(canon(public_cases))
    d={
      'schema':'R22572_C56_PREOUTPUT_SEAL_V1','candidate':'C56','weights_present':0,'model_outputs_observed':0,'training_rows_used':0,
      'scientific_question':'Does a programmable relation-judgment LoRA learn rule-conditioned relational steering beyond the generic reranker base, rather than static similarity, and which projection/layer low-rank groups carry any admitted steering effect?',
      'suite_n':len(cases),'suite_sha256':suite_sha,
      'partitions':{p:sum(c['partition']==p for c in cases) for p in sorted({c['partition'] for c in cases})},
      'conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p25','DOSE_0p50','DOSE_1p50'],
      'causal_if_admitted':['MINUS_Q','MINUS_K','MINUS_V','MINUS_O','MINUS_GATE','MINUS_UP','MINUS_DOWN','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE'],
      "positive_gate":"composite gain FULL-BASE>=0.06; signed-margin paired bootstrap CI low>0.01; FULL-RANDOM>=0.04; FULL-SHUFFLE>=0.04; rule-flip order gain>=0.08; hard-negative/negation false-positive collateral increase<=0.05; independent verifier B same direction",
      "failure_gate":"composite FULL-BASE<=-0.06 with CI high<-0.01 and FULL materially worse than RANDOM/SHUFFLE, or negation/directional failure control-separated by>=0.08; verifier B same direction",
      "claim_boundary":"Synthetic fresh relational-judgment assay only; no external benchmark/ranking claim; training-time exact Base revision unproven; model-card metrics are not evidence; E4/E5 forbidden without independent replay/fine irredundancy.",
      "runtime_abi_gate":"Exact Qwen3 causal-LM class, logits_to_keep=1, serving yes/no IDs, PEFT lifecycle, safe adapter identifiers and control mutations must pass before binary ingress.",
      'verifier_smoke':smoke,'causal_locked_partitions':['conjunction','directional']
    }
    (out/'C56_PREOUTPUT_SEAL.json').write_text(json.dumps(d,indent=2,ensure_ascii=False))
    (out/'C56_SUITE_DIGEST.json').write_text(json.dumps({'schema':'R22572_C56_SUITE_DIGEST_V1','suite_n':len(cases),'suite_sha256':suite_sha,'raw_cases_exported':False,'reproduce':'python src/lucia_aa_r22572/reproduce_r22572_c56_suite.py'},indent=2))
    print(json.dumps(d,indent=2))

def auc(labels,scores):
    pos=[s for y,s in zip(labels,scores) if y==1];neg=[s for y,s in zip(labels,scores) if y==0]
    if not pos or not neg:return float('nan')
    wins=0.0
    for p in pos:
      for n in neg:wins += 1.0 if p>n else (0.5 if p==n else 0.0)
    return wins/(len(pos)*len(neg))

def metrics(cases,scores):
    labs=[c['label'] for c in cases]; pred=[int(x>=0.5) for x in scores]
    tp=sum(y==1 and p==1 for y,p in zip(labs,pred));tn=sum(y==0 and p==0 for y,p in zip(labs,pred));fp=sum(y==0 and p==1 for y,p in zip(labs,pred));fn=sum(y==1 and p==0 for y,p in zip(labs,pred))
    tpr=tp/max(1,tp+fn);tnr=tn/max(1,tn+fp);bal=(tpr+tnr)/2
    signed=[(1 if y els`-1)*(s-0.5) for y,s in zip(labs,scores)]
    groups=defaultdict(list)
    for c,s in zip(cases,scores):
      if c.get('pair_group'):groups[c['pair_group']].append((c['label'],s,c['partition']))
    orders=[]
    for g,vals in groups.items():
      Zs=[s for y,s,_ in vals if y==1];ns=[s for y,s,_ in vals if y==0]
      if ps and ns:orders.append(int(sum(ps)/len(ps)>sum(ns)/len(ns)))
    part={}
    for p in sorted({c['partition'] for c in cases}):
      idx=[i for i,c in enumerate(cases) if c['partition']==p]; ll=[labs[i] for i in idx]; ss=[scores[i] for i in idx]; pp=[pred[i] for i in idx]
      part[p]={'n':len(idx),'accuracy':sum(a==b for a,b in zip(ll,pp))/len(idx),'auc':auc(ll,ss),'mean_score':sum(ss)/len(ss),'positive_rate':sum(pp)/len(pp)}
    return {'n':len(cases),'accuracy':sum(a==b for a,b in zip(labs,pred))/len(labs),'balanced_accuracy':bal,'auc':auc(labs,scores),'mean_signed_margin':sum(signed)/len(signed),"pair_order_accuracy":sum(orders)/len(orders) if orders else None,'false_positive_rate':fp/max(1,fp+tn),'partitions':part}

def bootstrap_delta(cases,a_scores,b_scores,iters=1200):
    rng=random.Random(SEED+99); vals=[]; n=len(cases); labs=[c['label'] for c in cases]
    dif=[(1 if y else -1)*((a-0.5)-(b-0.5)) for y,a,b in zip(labs,a_scores,b_scores)]
    for _ in range(iters):
      vals.append(sum(dif[rng.randrange(n)] for _ in range(n))/n)
    vals.sort();return {'mean':sum(dif)/n,'low':vals[int(.025*iters)],'high':vals[int(.975*iters)-1]}

def execute(out:Path,work:Path):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from safetensors import safe_open
    out.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True)
    raw=work/'raw'; variants=work/'variants'; raw.mkdir(exist_ok=True);variants.mkdir(exist_ok=True)
    cases=build_suite(); seal=json.loads((out/'C56_PREOUTPUT_SEAL.json').read_text()); assert sha_bytes(canon(cases))==seal['suite_sha256']
    api=HfApi(); ai=api.model_info(ADAPTER_REPO,revision=ADAPTER_REV,files_metadata=True); assert ai.sha==ADAPTER_REV
    card=(ai.card_data.to_dict() if ai.card_data else {}); assert card.get('license')=='apache-2.0'
    adfiles={s.rfilename:s for s in ai.siblings}; assert 'adapter_model.safetensors' in adfiles and 'adapter_config.json' in adfiles and 'serving.json' in adfiles
    assert not any(re.search(r'^\\.(bin|pt|pth|pkl|pickle)$',x,re.I) for x in adfiles)
    base_info=api.model_info(BASE_REPO,files_metadata=True); base_rev=base_info.sha; base_card=(base_info.card_data.to_dict() if base_info.card_data else {})
    # Require safe Base model. Tokenizer is taken from adapter exact source.
    bfiles={s.rfilename:s for s in base_info.siblings}; assert any(x.endswith('.safetensors') for x in bfiles); assert not any(re.search(r'\\.(bin|pt|pth|pkl|pickle)$',x,re.I) for x in bfiles)
    ad_dir=Path(snapshot_download(ADAPTER_REPO,revision=ADAPTER_REV,local_dir=raw/'adapter',allow_patterns=['adapter_model.safetensors','adapter_config.json','serving.json','tokenizer/*']))
    base_dir=Path(snapshot_download(BASE_REPO,revision=base_rev,local_dir=raw/'base',allow_patterns=['config.json','generation_config.json','*.safetensors','*.safetensors.index.json']))
    serving=json.loads((ad_dir/'serving.json').read_text()); assert serving['base_model']==BASE_REPO; assert serving['yes_id']==EXPECTED_YES_ID and serving['no_id']==EXPECTED_NO_ID and serving['max_len']==EXPECTED_MAX_LEN
    ad_sha=sha_bytes((ad_dir/'adapter_model.safetensors').read_bytes())
    # model source identity manifest, without shipping bytes
    base_manifest=[]
    for p in sorted(base_dir.rglob('*')):
      if p.is_file():base_manifest.append({'path':p.relative_to(base_dir).as_posix(),'bytes':p.stat().st_size,'sha256':sha_bytes(p.read_bytes())})
    # static anatomy before loading
    with safe_open(ad_dir/'adapter_model.safetensors',framework='pt',device='cpu') as sf:
      keys=list(sf.keys()); tensors={k:sf.get_tensor(k) for k in keys}
    pair_roots={}
    for k in keys:
      if '.lora_A.' in k:pair_roots.setdefault(k.split('.lora_A.')[0],{})['A']=k
      if '.lora_B.' in k:pair_roots.setdefault(k.split('.lora_B.')[0],{})['B']=k
    cfg=json.loads((ad_dir/'adapter_config.json').read_text()); scale=float(cfg['lora_alpha'])/float(cfg['r'])
    static=[]
    for root,ab in pair_roots.items():
      if set(ab)!={'A','B'}:continue
      A=tensors[ab['A']].float();B=tensors[ab['B']].float(); fam=next((f for f in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if f in root),İ\‰ÊBˆO\™KœÙX\˜Ú
‰×—›^Y\œ×Š
ÊW‰Ë›Ûİ
NÈ^Y\Z[
K™Ü›İ\
JJHYˆH[ÙHLBˆİ]XË˜\[™
ÉÜ›ÛİÙYÙ\İ	ÎœÚWØ]\Ê›Ûİ™[˜ÛÙJ
JK	Ù˜[Z[IÎ™˜[K	Û^Y\‰Î›^Y\‹	Ü˜[šÉÎš[
KœÚ\VÌJK	ØWÜÚ\IÎ›\İ
KœÚ\JK	Ø—ÜÚ\IÎ›\İ
‹œÚ\JK	Ù[™\™ŞWÜ›ŞIÎ™›Ø]
Ü˜Ú›[˜[Ë™XİÜ—Û›Ü›JJJÜ˜Ú›[˜[Ë™XİÜ—Û›Ü›JŠJœØØ[JK	ØWÙš[š]IÎ˜›ÛÛ
Ü˜Úš\Ùš[š]JJK˜[

JK	Ø—Ùš[š]IÎ˜›ÛÛ
Ü˜Úš\Ùš[š]JŠK˜[

J_JBˆİ[ÙO\İ[JÉÙ[™\™ŞWÜ›ŞI×H›Üˆ[ˆİ]XÊHÜˆKŒÈ˜[WÙ[™\™ŞOYY˜][Xİ
›Ø]
NÈ^Y\—Ù[™\™ŞOYY˜][Xİ
›Ø]
Bˆ›Üˆ[ˆİ]XÎ™˜[WÙ[™\™ŞVŞÉÙ˜[Z[I×WJÏ^ÉÙ[™\™ŞWÜ›ŞI×NÛ^Y\—Ù[™\™ŞVŞÉÛ^Y\‰×WJÏ^ÉÙ[™\™ŞWÜ›ŞI×Bˆİ]X×ÜX›XÏ^ÉÜØÚ[XIÎ‰ÔŒŒMÌ—ĞÍM—ÔÕUP×ĞUT×ÕŒIË	ÜZ\—ØÛİ[	Î›[Šİ]XÊK	Ü˜[š×ØÛÛ™šYÉÎ˜Ù™ÖÉÜ‰×K	Ø[IÎ˜Ù™ÖÉÛÜ˜WØ[I×K	Ş™\›×Ù[™\™ŞWÜZ\œÉÎœİ[JÉÙ[™\™ŞWÜ›ŞI×OOL›Üˆ[ˆİ]XÊK	Û›Û™š[š]WÜZ\œÉÎœİ[J›İ
ÉØWÙš[š]I×H[™ÉØ—Ùš[š]I×JH›Üˆ[ˆİ]XÊK	Ù˜[Z[WÙ[™\™ŞWÜÚ\™IÎÚÎ‹İİ[ÙH›ÜˆËˆ[ˆÛÜY
˜[WÙ[™\™ŞKš][\Ê
J_K	İÜÛ^Y\—Ù[™\™ŞWÜÚ\™IÎœÛÜY

ÉÛ^Y\‰ÎšË	ÜÚ\™IÎ‹İİ[Ù_H›ÜˆËˆ[ˆ^Y\—Ù[™\™ŞKš][\Ê
KÙ^O[[X™H‹^ÉÜÚ\™I×JVÎK	ØÛZ[WØ›İ[™\IÎ‰ÔÕUP×ĞĞS‘QUWÓÓ“W×Ó“Ô“WÑS‘T‘ÖWÒT×Ó“ÕĞĞUTĞS	ßBˆÈØY^Xİ[[YHÚ]İ]™[[İHÛÙBˆÚÏP]]ÕÚÙ[š^™\‹™œ›ÛWÜ™]˜Z[™Y
YÙ\‹ÉİÚÙ[š^™\‰ËY[™×ÜÚYOIÛY	Ë\İÜ™[[İWØÛÙOQ˜[ÙJBˆ˜\ÙOP]]Ó[Ù[›ÜØ]\Ø[K™œ›ÛWÜ™]˜Z[™Y
˜\ÙWÙ\‹Ü˜ÚÙ\O]Ü˜Ú™›Ø]Ì‹]—Ú[\[Y[][ÛIÜÙIË\İÜ™[[İWØÛÙOQ˜[ÙJK™]˜[

BˆYTY[Ù[™œ›ÛWÜ™]˜Z[™Y
˜\ÙKYÙ\‹Y\\—Û˜[YOIÙY˜][	Ë\×İ˜Z[˜X›OQ˜[ÙJK™]˜[

Bˆ\ÜÙ\ÚË›ØØX—ÜÚ^™O›X^
Ù\š[™ÖÉŞY\×ÚY	×KÙ\š[™ÖÉÛ›×ÚY	×JBˆ™Yš^\Ù\š[™ÖÉÜ™Yš^	×NÜİY™š^\Ù\š[™ÖÉÜİY™š^	×NÈ\ÜÙ\\Ú[œİ[˜ÙJ™Yš^İŠH[™\Ú[œİ[˜ÙJİY™š^İŠBˆYˆ›Û\
ÊNœ™]\›ˆ™Yš^
Ùˆ[œİXİˆØÖÉÚ[œİXİ[Û‰×_W—LØÔ]Y\WLÙNˆØÖÉÜ]Y\I×_W—LØÑØİ[Y[LÙNˆ’S‘S‘ÈN—ØÖÉØI×_W—‘’S‘S‘È—ØÖÉØ‰×_HŠÜİY™š^ˆÜ˜Úš[™™\™[˜ÙWÛ[ÙJ
BˆYˆØÛÜ™WØØ\Ù\ÊİX˜Ø\Ù\Ë˜]ÚLM‹\ØX›OQ˜[ÚN‚ˆÜÏV×Bˆİ\Y™\ØX›WØY\\Š
HYˆ\ØX›H[ÙH›Û™BˆYˆİ˜İ—×Ù[\—×Ê
BˆN‚ˆ›ÜˆH[ˆ˜[™ÙJ[ŠİX˜Ø\Ù\ÊK˜]Ú
N‚ˆ^ÏVÜ›Û\
ÊH›ÜˆÈ[ˆİX˜Ø\Ù\ÖÚNšJØ˜]ÚWNÈ[˜Ï]ÚÊ^Ë™]\›—İ[œÛÜœÏIÜ	ËY[™ÏUYK[˜Ø][ÛUYKX^Û[™İ[Z[ŠLL‹Ù\š[™ÖÉÛX^Û[‰×JKYÜÜXÚX[İÚÙ[œÏQ˜[ÙJBˆÏ\Y

Š™[˜ËÙÚ]×İ×ÚÙY\LJK›ÙÚ]ÖÎ‹LK—Bˆ›Ø]Ü˜ÚœÚYÛ[ÚY
ÖÎ‹Ù\š[™ÖÉŞY\×ÚY	×WK[ÖÎ‹Ù\š[™ÖÉÛ›×ÚY	×WJNÜÜË™^[™
›Ø]

H›Üˆ[ˆ›Ø‹˜ÜJ
JBˆš[˜[N‚ˆYˆİ˜İ—×Ù^]×Ê›Û™K›Û™K›Û™JBˆ™]\›ˆÜÂˆÈØ\\™HÜšYÚ[˜[İË\˜[šÈİ]HÛ›H
›È˜\ÙHÛÛš[™ÊK‚ˆ\˜[\Ï^Ûœ›Üˆ‹[ˆY›˜[YYÜ\˜[Y]\œÊ
HYˆ	Ë›Ü˜WĞK™Y˜][ÙZYÚ	È[ˆˆÜˆ	Ë›Ü˜WĞ‹™Y˜][ÙZYÚ	È[ˆŸBˆÜšYÚ[˜[Ï^Ûœ™]XÚ

K˜ÜJ
K˜ÛÛ™J
H›Üˆ‹[ˆ\˜[\Ëš][\Ê
_BˆYˆ™\İÜ™J
N‚ˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆ›Üˆ‹[ˆ\˜[\Ëš][\Ê
Nœ˜ÛÜWÊÜšYÚ[˜[ÖÛ—KÊ™]šXÙK™\JJBˆYˆ˜[™ÛZ^™J
N‚ˆ™\İÜ™J
NÙÏ]Ü˜Ú‘Ù[™\˜]ÜŠ]šXÙOIØÜIÊNÙË›X[X[ÜÙYY
ÑQQ
ÍÊBˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆ›Üˆ‹[ˆ\˜[\Ëš][\Ê
N‚ˆÏ[ÜšYÚ[˜[ÖÛ—K™›Ø]

NÜ]Ü˜Úœ˜[™ŠËœÚ\KÙ[™\˜]ÜYÊNÜ\‹ÊÜ˜Ú›[˜[Ë™XİÜ—Û›Ü›JŠJÌYKLLŠJŠÜ˜Ú›[˜[Ë™XİÜ—Û›Ü›JÊJÌYKLLŠNÜ˜ÛÜWÊ‹Ê™]šXÙK™\JJBˆYˆÚY™›WÜZ\œÊ
N‚ˆ™\İÜ™J
NÈ›ÛİÏYY˜][Xİ
Xİ
Bˆ›Üˆˆ[ˆ\˜[\Î‚ˆYˆ	Ë›Ü˜WĞK™Y˜][ÙZYÚ	È[ˆœ›ÛİÖÛ‹œÜ]
	Ë›Ü˜WĞK™Y˜][ÙZYÚ	ÊVÌWVÉĞI×O[‚ˆ[Yˆ	Ë›Ü˜WĞ‹™Y˜][ÙZYÚ	È[ˆœ›ÛİÖÛ‹œÜ]
	Ë›Ü˜WĞ‹™Y˜][ÙZYÚ	ÊVÌWVÉĞ‰×O[‚ˆÜ›İ\ÏYY˜][Xİ
\İ
Bˆ›Üˆ›ÛİXˆ[ˆ›ÛİËš][\Ê
N‚ˆYˆÙ]
XŠOO^ÉĞIË	Ğ‰ßN™Ü›İ\ÖÊ\JÜšYÚ[˜[ÖØX–ÉĞI×WKœÚ\JK\JÜšYÚ[˜[ÖØX–ÉĞ‰×WKœÚ\JJWK˜\[™
›Ûİ
Bˆ›™Ï\˜[™ÛK”˜[™ÛJÑQQ
Î
BˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆ›ÜˆËœÈ[ˆÜ›İ\Ëš][\Ê
N‚ˆÜ˜Ï\œÖÎ—NÜ›™ËœÚY™›JÜ˜ÊBˆ›ÜˆİÜˆ[ˆš\
œËÜ˜ÊN‚ˆ›ÜˆÚYH[ˆÉĞIË	Ğ‰×N‚ˆ\›ÛİÖÙİVÜÚYWNÜÛ\›ÛİÖÜÜ—VÜÚYWNÜ\˜[\ÖÙ—K˜ÛÜWÊÜšYÚ[˜[ÖÜÛ—KÊ\˜[\ÖÙ—K™]šXÙK\˜[\ÖÙ—K™\JJBˆYˆÜÙJ
N‚ˆ™\İÜ™J
BˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆ›Üˆ‹[ˆ\˜[\Ëš][\Ê
N‚ˆYˆ	Ë›Ü˜WĞ‹™Y˜][ÙZYÚ	È[ˆœ›][Ê
BˆYˆX›]WÙ˜[Z[J˜[JN‚ˆ™\İÜ™J
BˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆ›Üˆ‹[ˆ\˜[\Ëš][\Ê
N‚ˆYˆ	Ë›Ü˜WĞ‹™Y˜][ÙZYÚ	È[ˆˆ[™‰ËÙ˜[_K‰È[ˆœ™\›×Ê
BˆYˆX›]WØ˜[™
˜[™
N‚ˆ™\İÜ™J
NÛ^Y\œÏV×Bˆ›Üˆˆ[ˆ\˜[\Î‚ˆO\™KœÙX\˜Ú
‰×›^Y\œ×Š
ÊW‰ËŠBˆYˆN›^Y\œË˜\[™
[
K™Ü›İ\
JJJBˆ^[X^
^Y\œÊNØİ]OJ^
ÌJKËÌÎØİ]LŠŠ^
ÌJKËÌÂˆYˆ]

Nœ™]\›ˆ
İ]HYˆ˜[™OIÙX\›IÈ[ÙH
İ]O[İ]ˆYˆ˜[™OIÛZYIÈ[ÙHXİ]ŠJBˆÚ]Ü˜Ú››×ÙÜ˜Y

N‚ˆ›Üˆ‹[ˆ\˜[\Ëš][\Ê
N‚ˆO\™KœÙX\˜Ú
‰×›^Y\œ×Š
ÊW‰ËŠBˆYˆ	Ë›Ü˜WĞ‹™Y˜][ÙZYÚ	È[ˆˆ[™H[™]
[
K™Ü›İ\
JJJNœ™\›×Ê
BˆÈ™Z]š[ÜˆÛÛ™][ÛœË‚ˆ™\İÜ™J
NØ˜\ÙWÜØÛÜ™\Ï\ØÛÜ™WØØ\Ù\ÊØ\Ù\Ë\ØX›OUYJBˆ™\İÜ™J
NÙ[ÜØÛÜ™\Ï\ØÛÜ™WØØ\Ù\ÊØ\Ù\ÊBˆ˜[™ÛZ^™J
NÜ˜[™ÜØÛÜ™\Ï\ØÛÜ™WØØ\Ù\ÊØ\Ù\ÊBˆÚY™›WÜZ\œÊ
NÜÚY—ÜØÛÜ™\Ï\ØÛÜ™WØØ\Ù\ÊØ\Ù\ÊBˆÜÙWÜØÛÜ™\Ï^ßBˆ›Üˆ[ˆÌŒKKKWN™ÜÙJ
NÙÜÙWÜØÛÜ™\ÖÜİŠ
WO\ØÛÜ™WØØ\Ù\ÊØ\Ù\ÊBˆ™\İÜ™J
BˆÛÛ™^ÉĞTÑIÎ›Y]šXÜÊØ\Ù\Ë˜\ÙWÜØÛÜ™\ÊK	Ñ•S	Î›Y]šXÜÊØ\Ù\Ë[ÜØÛÜ™\ÊK	ÔS‘ÓWÔS’×ÓPUÒQ	Î›Y]šXÜÊØ\Ù\Ë˜[™ÜØÛÜ™\ÊK	ÓVQT—ÔÒQ‘“Q	Î›Y]šXÜÊØ\Ù\ËÚY—ÜØÛÜ™\Ê_Bˆ›ÜˆÈ[ˆÜÙWÜØÛÜ™\Ëš][\Ê
N˜ÛÛ™ÉÑÔÑWÉÊÙœ™\XÙJ	Ë‰Ë	Ü	ÊWO[Y]šXÜÊØ\Ù\ËÊBˆYˆÛÛ\ÜÚ]JJNœ™]\›ˆMJ›VÉØ]XÉ×JÌJŠVÉÜZ\—ÛÜ™\—ØXØİ\˜XŞI×HYˆVÉÜZ\—ÛÜ™\—ØXØİ\˜XŞI×H\È›İ›Û™H[ÙHVÉØ˜[[˜ÙYØXØİ\˜XŞI×JBˆÛÛ\^ÚÎ˜ÛÛ\ÜÚ]JŠH›ÜˆËˆ[ˆÛÛ™š][\Ê
_NØÚOX›Ûİİ˜\Ù[JØ\Ù\Ë[ÜØÛÜ™\Ë˜\ÙWÜØÛÜ™\ÊBˆ™Y×ÙœÙ[XÛÛ™ÉÑ•S	×VÉÜ\][ÛœÉ×VÉÛ™YØ][Û‰×VÉÜÜÚ]]™WÜ˜]I×NÛ™Y×ÙœØ˜\ÙOXÛÛ™ÉĞTÑI×VÉÜ\][ÛœÉ×VÉÛ™YØ][Û‰×VÉÜÜÚ]]™WÜ˜]I×BˆÜÏJÛÛ\ÉÑ•S	×KXÛÛ\ÉĞTÑI×OLŒˆ[™ÚVÉÛİÉ×OŒŒH[™ÛÛ\ÉÑ•S	×KXÛÛ\ÉÔS‘ÓWÔS’×ÓPUÒQ	×OLŒ[™ÛÛ\ÉÑ•S	×KXÛÛ\ÉÓVQT—ÔÒQ‘“Q	×OLŒ[™ÛÛ™ÉÑ•S	×VÉÜZ\—ÛÜ™\—ØXØİ\˜XŞI×KXÛÛ™ÉĞTÑI×VÉÜZ\—ÛÜ™\—ØXØİ\˜XŞI×OLŒ[™™Y×ÙœÙ[[™Y×ÙœØ˜\ÙOLŒJBˆ˜Z[JÛÛ\ÉÑ•S	×KXÛÛ\ÉĞTÑI×OKLŒˆ[™ÚVÉÚYÚ	×OLŒH[™ÛÛ\ÉÑ•S	×OZ[ŠÛÛ\ÉÔS‘ÓWÔS’×ÓPUÒQ	×KÛÛ\ÉÓVQT—ÔÒQ‘“Q	×JKLŒ
BˆØ]\Ø[V×BˆØÚÙYVØÈ›ÜˆÈ[ˆØ\Ù\ÈYˆÖÉÜ\][Û‰×H[ˆÉØÛÛš[˜İ[Û‰Ë	Ù\™Xİ[Û˜[	ßWBˆYˆÜÈÜˆ˜Z[‚ˆ™\İÜ™J
NÙ[ÛØÚÙY\ØÛÜ™WØØ\Ù\ÊØÚÙY
NÙ›O[Y]šXÜÊØÚÙY[ÛØÚÙY
NÙ˜ÏXÛÛ\ÜÚ]J›JBˆ›Üˆ˜[H[ˆÉÜWÜ›Ú‰Ë	Ú×Ü›Ú‰Ë	İ—Ü›Ú‰Ë	Û×Ü›Ú‰Ë	ÙØ]WÜ›Ú‰Ë	İ\Ü›Ú‰Ë	ÙİÛ—Ü›Ú‰×N‚ˆX›]WÙ˜[Z[J˜[JNÜØÏ\ØÛÜ™WØØ\Ù\ÊØÚÙY
NÛ[O[Y]šXÜÊØÚÙYØÊNØØ]\Ø[˜\[™
ÉÚ[\™[[Û‰Î‰ÓRS•T×ÉÊÙ˜[KœÜ]
	×ÉÊVÌK\\Š
K	ØÛÛ\ÜÚ]IÎ˜ÛÛ\ÜÚ]J[JK	Ù›ÜÙœ›ÛWÙ[ÛØÚÙY	Î™˜ËXÛÛ\ÜÚ]J[JK	Û‰Î›[ŠØÚÙY
_JBˆ›Üˆ˜[™[ˆÉÙX\›IË	ÛZYIË	Û]I×N‚ˆX›]WØ˜[™
˜[™
NÜØÏ\ØÛÜ™WØØ\Ù\ÊØÚÙY
NÛ[O[Y]šXÜÊØÚÙYØÊNØØ]\Ø[˜\[™
ÉÚ[\™[[Û‰Î‰ÓRS•T×ÉÊØ˜[™\\Š
K	ØÛÛ\ÜÚ]IÎ˜ÛÛ\ÜÚ]J[JK	Ù›ÜÙœ›ÛWÙ[ÛØÚÙY	Î™˜ËXÛÛ\ÜÚ]J[JK	Û‰Î›[ŠØÚÙY
_JBˆ™\İÜ™J
BˆÈ›È˜]È™YXİ[ÛœÈ^ÜYÈØØ[\ˆÛİ[\™^[\\ÈÛ›K‚ˆÛİ[\V×Bˆ›ÜˆË‹‹›‹Ú[ˆš\
Ø\Ù\Ë˜\ÙWÜØÛÜ™\Ë[ÜØÛÜ™\Ë˜[™ÜØÛÜ™\ËÚY—ÜØÛÜ™\ÊN‚ˆYˆXœÊ‹XŠOLŒŒÜˆ
[
KJHOXÖÉÛX™[	×H[™[
KJOOXÖÉÛX™[	×JN‚ˆÛİ[\‹˜\[™
ÉÚY	Î˜ÖÉÚY	×K	Ü\][Û‰Î˜ÖÉÜ\][Û‰×K	Ü[WÚÙ^IÎ˜ÖÉÜ[WÚÙ^I×K	ÛX™[	Î˜ÖÉÛX™[	×K	Ø˜\ÙWÜØÛÜ™IÎœ›İ[™
‹ŠK	Ù[ÜØÛÜ™IÎœ›İ[™
‹ŠK	Ü˜[™ÛWÜØÛÜ™IÎœ›İ[™
›‹ŠK	ÜÚY™›WÜØÛÜ™IÎœ›İ[™
ÚŠ_JBˆ™Z]š[Ü^ÉÜØÚ[XIÎ‰ÔŒŒMÌ—ĞÍM—Ğ‘RU’SÔ—ÕŒIË	ÜİZ]WÛ‰Î›[ŠØ\Ù\ÊK	ÜİZ]WÜÚLM‰ÎœÙX[ÉÜİZ]WÜÚLM‰×K	ØÛÛ™][ÛœÉÎ˜ÛÛ™	ØÛÛ\ÜÚ]IÎ˜ÛÛ\	Ù[ÛZ[\×Ø˜\ÙWÜÚYÛ™YÛX\™Ú[—Ø›Ûİİ˜\MIÎ˜ÚK	ÜÜÚ]]™WÙØ]IÎ˜›ÛÛ
ÜÊK	Ù˜Z[\™WÙØ]IÎ˜›ÛÛ
˜Z[
K	ØØ]\Ø[Ü˜[‰Î˜›ÛÛ
Ø]\Ø[
K	ØØ]\Ø[ÜØÜ™Y[‰Î˜Ø]\Ø[	ØÛİ[\™^[\WÜØØ[\—ØÛİ[	Î›[ŠÛİ[\ŠK	ØÛZ[WØ›İ[™\IÎ‰Ñœ™\ÚŞ[]XÈ™[][Û˜[\İY\š[™È\ÜØ^NÈ›È^\›˜[™[˜ÚX\šÈÛZ[KˆÜÚ]]™KÙ˜Z[\™HØ]\ÈÛÛ›ÛØ]\Ø[[NÈ˜]È™YXİ[ÛœÈ›İ^ÜY‰ßBˆ›İ^ÉÜØÚ[XIÎ‰ÔŒŒMÌ—ĞÍM—Ô“Õ‘SSÑWÕŒIË	ØY\\—Ü™\ÉÎQTT—Ô‘TË	ØY\\—Ü™]š\Ú[Û‰ÎQTT—Ô‘U‹	ØY\\—ÜØY™][œÛÜœ×ÜÚLM‰Î˜YÜÚK	ØY\\—Ø]\ÉÎŠYÙ\‹ÉØY\\—Û[Ù[œØY™][œÛÜœÉÊKœİ]

KœİÜÚ^™K	ØY\\—ÛXÙ[œÙIÎ‰Ø\XÚKL‹Œ	Ë	Ø˜\ÙWÜ™\ÉÎTÑWÔ‘TË	Ü[[YWÜ™Y™\™[˜ÙWØ˜\ÙWÜ™]š\Ú[Û‰Î˜˜\ÙWÜ™]‹	İ˜Z[š[™×İ[YWØ˜\ÙWÜ™]š\Ú[Û—Ü›İ™[‰Î‘˜[ÙK	Ø˜\ÙWÛXÙ[œÙIÎ˜˜\ÙWØØ\™™Ù]
	ÛXÙ[œÙIÊK	Ø˜\ÙWÛX[šY™\İ	Î˜˜\ÙWÛX[šY™\İ	ÜÙ\š[™×ØÛÛ˜Xİ	ÎÉŞY\×ÚY	ÎœÙ\š[™ÖÉŞY\×ÚY	×K	Û›×ÚY	ÎœÙ\š[™ÖÉÛ›×ÚY	×K	ÛX^Û[‰ÎœÙ\š[™ÖÉÛX^Û[‰×K	Ü™Yš^ÜÚLM‰ÎœÚWØ]\Ê™Yš^™[˜ÛÙJ
JK	ÜİY™š^ÜÚLM‰ÎœÚWØ]\ÊİY™š^™[˜ÛÙJ
J_K	İ\İÜ™[[İWØÛÙIÎ‘˜[ÙK	ÜXÚÛWØÛÛœİ[YY	Î‘˜[Ù_Bˆ
İ]ÉĞÍM—ÔÕUP×ĞUTËšœÛÛ‰ÊKÜš]Wİ^
œÛÛ‹™[\Êİ]X×ÜX›XË[™[LŠJBˆ
İ]ÉĞÍM—Ğ‘RU’SÔ‹šœÛÛ‰ÊKÜš]Wİ^
œÛÛ‹™[\Ê™Z]š[Ü‹[™[LŠJBˆ
İ]ÉĞÍM—ĞÓÕS•T‘VSTT×ÔĞĞST‹šœÛÛ‰ÊKÜš]Wİ^
œÛÛ‹™[\ÊÉÜØÚ[XIÎ‰ÔŒŒMÌ—ĞÍM—ĞÓÕS•T‘VSTT×ÔĞĞST—ÕŒIË	Ú][\ÉÎ˜Ûİ[\ŸK[™[LŠJBˆ
İ]ÉĞÍM—Ô“Õ‘SSÑKšœÛÛ‰ÊKÜš]Wİ^
œÛÛ‹™[\Ê›İ‹[™[LŠJBˆÈ\İXİ]™HÛX[\ˆ[Y˜\ÙKÚË[œÛÜœËÜšYÚ[˜[Ë\˜[\Âˆ[\ÜØÎÙØË˜ÛÛXİ

BˆÚ][œ›]™YJÛÜšËYÛ›Ü™WÙ\œ›ÜœÏUYJBˆšÛYO[ÜË™[š\›Û‹™Ù]
	Ò—ÒÓQIÊNÂˆYˆšÛYNœÚ][œ›]™YJšÛYKYÛ›Ü™WÙ\œ›ÜœÏUYJBˆ
İ]ÉĞÍM—ĞÓPS•TšœÛÛ‰ÊKÜš]Wİ^
œÛÛ‹™[\ÊÉÜØÚ[XIÎ‰ÔŒŒMÌ—ĞÍM—ĞÓPS•TÕŒIË	ØÛÛ›ÛYØÛX[\	Î•YK	Ü˜]×Ü›ÛİÙ[]Y	Î››İÛÜšË™^\İÊ
K	Ú—ØØXÚWÙ[]Y	Î››İ
šÛYH[™]
šÛYJK™^\İÊ
JK	Ü˜]×İÙZYÚ×Ü™[XZ[š[™ÉÎŒ	Ü˜]×İÚÙ[š^™\—Ü™[XZ[š[™ÉÎŒK[™[LŠJBˆš[
œÛÛ‹™[\ÊÉØ™Z]š[Ü‰Î˜™Z]š[Ü‹	Üİ]XÉÎœİ]X×ÜX›XßK[™[LŠVÎŒLŒJB‚™YˆXZ[Š
N‚ˆ\X\™Ü\œÙK\™İ[Y[\œÙ\Š
NØ\˜YØ\™İ[Y[
	Û[ÙIËÚÚXÙ\ÏVÉÜ™\ÙX[	Ë	Ù^Xİ]I×JNØ\˜YØ\™İ[Y[
	ËK[İ]	Ë™\]Z\™YUYJNØ\˜YØ\™İ[Y[
	ËK]ÛÜšÉËY˜][IØÍM‹]ÛÜšÉÊNØOX\œ\œÙWØ\™ÜÊ
NÛİ]T]
K›İ]
BˆYˆK›[ÙOOIÜ™\ÙX[	Îœ™\ÙX[
İ]
Bˆ[ÙN™^Xİ]Jİ]]
KÛÜšÊJBšYˆ×Û˜[YW×ÏOI××ÛXZ[—×ÉÎ›XZ[Š
B