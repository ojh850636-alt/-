from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path
from difflib import SequenceMatcher
sys.dont_write_bytecode=True
SEED=22573
PREFIX='Fix grammatical errors in this sentence: '

def build_suite():
    cases=[]
    def add(part,source,target,protected=(),error_kind=''):
        cases.append({'id':f'C57-{len(cases):03d}','partition':part,'source':source,'target':target,'protected':list(protected),'error_kind':error_kind})
    nouns=[('report','contains'),('study','includes'),('device','requires'),('method','reduces'),('system','supports'),('trial','measures'),('paper','describes'),('model','predicts')]
    for i,(n,v) in enumerate(nouns):
        add('single_error',f'The {n} {v[:-1]} three sections.',f'The {n} {v} three sections.',error_kind='subject_verb')
        add('single_error',f'These {n}s {v} a clear trend.',f'These {n}s {v[:-1]} a clear trend.',error_kind='subject_verb')
    roles=['engineer','analyst','editor','architect','researcher','auditor','designer','advisor']
    for role in roles:add('single_error',f'She is {role} at Delta Labs.',f'She is a {role} at Delta Labs.',protected=('Delta Labs',),error_kind='article')
    names=['Mira','Lina','Owen','Ravi','Nora','Evan','Priya','Jonah']
    for name in names:add('single_error',f'Yesterday, {name} submit the form.',f'Yesterday, {name} submitted the form.',protected=(name,),error_kind='tense')
    prep=[('depends of','depends on'),('arrived to','arrived at'),('interested on','interested in'),('responsible of','responsible for'),('consists in','consists of'),('similar with','similar to'),('applied on','applied to'),('focused in','focused on')]
    for a,b in prep:add('single_error',f'The protocol {a} the sample.',f'The protocol {b} the sample.',error_kind='preposition')
    spells=[('analysys','analysis'),('approch','approach'),('enviroment','environment'),('recieve','receive'),('seperate','separate'),('occurence','occurrence'),('relevent','relevant'),('accomodate','accommodate')]
    for a,b in spells:add('single_error',f'This {a} is documented clearly.',f'This {b} is documented clearly.',error_kind='spelling')
    plural=[('student','students'),('device','devices'),('sample','samples'),('result','results'),('method','methods'),('article','articles'),('region','regions'),('sensor','sensors')]
    for a,b in plural:add('single_error',f'Many {a} were included.',f'Many {b} were included.',error_kind='number_agreement')
    each_nouns=['sample','report','method','device','trial','paper','model','protocol']
    for n in each_nouns:add('single_error',f'Each {n} were reviewed.',f'Each {n} was reviewed.',error_kind='agreement_each')
    protected_rows=[
      ('Mira submit report 17 on March 5, 2026.','Mira submitted report 17 on March 5, 2026.',('Mira','17','March 5, 2026')),
      ('Dr. Park review batch 42 on June 8, 2025.','Dr. Park reviewed batch 42 on June 8, 2025.',('Dr. Park','42','June 8, 2025')),
      ('Nora send file 903 to Acme Research.','Nora sent file 903 to Acme Research.',('Nora','903','Acme Research')),
      ('Ravi record 12 trials at Delta Labs.','Ravi recorded 12 trials at Delta Labs.',('Ravi','12','Delta Labs')),
      ('Lina write section 4 for Project Orion.','Lina wrote section 4 for Project Orion.',('Lina','4','Project Orion')),
      ('Owen submit version 2.7 on April 11, 2026.','Owen submitted version 2.7 on April 11, 2026.',('Owen','2.7','April 11, 2026')),
      ('Priya check sensor A-19 in Lab 3.','Priya checked sensor A-19 in Lab 3.',('Priya','A-19','Lab 3')),
      ('Jonah attach invoice 771 for Northstar Ltd.','Jonah attached invoice 771 for Northstar Ltd.',('Jonah','771','Northstar Ltd.')),
    ]
    for rep in range(4):
        for src,tgt,prot in protected_rows:add('protected_span',src,tgt,prot,error_kind='protected_tense')
    noop=['The report contains three sections.','These results show a clear trend.','Mira submitted report 17 on March 5, 2026.','Dr. Park reviewed batch 42 on June 8, 2025.','The method depends on the sample size.','Many students were included.','She is an engineer at Delta Labs.','The analysis is documented clearly.']
    for rep in range(4):
        for s in noop:
            prots=tuple(x for x in ['Mira','17','March 5, 2026','Dr. Park','42','June 8, 2025','Delta Labs'] if x in s)
            add('noop',s,s,prots,error_kind='none')
    multi=[
      ('Yesterday, Lina submit a reports to Dr. Park.','Yesterday, Lina submitted a report to Dr. Park.',('Lina','Dr. Park')),
      ('These method was tested in 3 labs.','These methods were tested in 3 labs.',('3',)),
      ('Mira have wrote section 7 for Orion.','Mira has written section 7 for Orion.',('Mira','7','Orion')),
      ('The two device was install on May 9, 2026.','The two devices were installed on May 9, 2026.',('May 9, 2026',)),
      ('Dr. Lee are responsible of batch 51.','Dr. Lee is responsible for batch 51.',('Dr. Lee','51')),
      ('The analysys show that 4 sample is missing.','The analysis shows that 4 samples are missing.',('4',)),
      ('Nora recieve a files from Delta Labs.','Nora received a file from Delta Labs.',('Nora','Delta Labs')),
      ('Project Vega were complete in 2 phase.','Project Vega was completed in 2 phases.',('Project Vega','2')),
    ]
    for rep in range(4):
        for src,tgt,prot in multi:add('multi_error',src,tgt,prot,error_kind='multi')
    ood=[
      ('The spectrometer measure two signals.','The spectrometer measures two signals.',()),
      ('These chromatographs detects trace compounds.','These chromatographs detect trace compounds.',()),
      ('Aisha submit the memorandum yesterday.','Aisha submitted the memorandum yesterday.',('Aisha',)),
      ('The manuscript depends of external evidence.','The manuscript depends on external evidence.',()),
      ('Many observatory was included.','Many observatories were included.',()),
      ('This calibrtion is documented clearly.','This calibration is documented clearly.',()),
      ('Professor Imani review sample 88.','Professor Imani reviewed sample 88.',('Professor Imani','88')),
      ('The two prototype was deploy in Sector 9.','The two prototypes were deployed in Sector 9.',('Sector 9',)),
    ]
    for rep in range(4):
        for src,tgt,prot in ood:add('lexical_ood',src,tgt,prot,error_kind='ood')
    assert len(cases)==192, len(cases)
    return cases

def suite_sha(cases):return hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def norm(s): return re.sub(r'\s+',' ',s.strip())
def tok(s): return re.findall(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*|[^\w\s]",norm(s),flags=re.UNICODE)
def edit_distance(a,b):
    a,b=tok(a),tok(b);m=len(b);prev=list(range(m+1))
    for i,x in enumerate(a,1):
        cur=[i]+[0]*m
        for j,y in enumerate(b,1):cur[j]=min(cur[j-1]+1,prev[j]+1,prev[j-1]+(x!=y))
        prev=cur
    return prev[m]
def score(pred,case):
    pred=norm(pred);src=norm(case['source']);tgt=norm(case['target']);exact=float(pred==tgt);source_copy=float(pred==src);ratio=SequenceMatcher(None,tok(pred),tok(tgt),autojunk=False).ratio();prot=float(all(p in pred for p in case['protected'])) if case['protected'] else 1.0;required=max(1,edit_distance(src,tgt));pt=edit_distance(pred,tgt);sp=edit_distance(src,pred);repair=max(0.0,1.0-pt/required);excess=max(0,sp-required)/max(1,len(tok(src)));noop_false=float(case['partition']=='noop' and pred!=src);composite=max(0.0,min(1.0,0.40*exact+0.25*repair+0.20*prot+0.15*ratio-0.25*excess-0.35*noop_false));return {'exact':exact,'repair':repair,'protected':prot,'target_ratio':ratio,'source_copy':source_copy,'excess_edit':excess,'noop_false_edit':noop_false,'composite':composite}
def aggregate(rows):return {k:sum(float(r[k]) for r in rows)/len(rows) for k in rows[0]}
def preseal_record():
    cases=build_suite();perfect=aggregate([score(c['target'],c) for c in cases]);copy=aggregate([score(c['source'],c) for c in cases]);drift=[]
    for c in cases:
        p=c['target'].replace('the ','the very ',1) if 'the ' in c['target'].lower() else c['target']+' Indeed.';drift.append(score(p,c))
    drift=aggregate(drift)
    pick=lambda spec:[c['id'] for part,n in spec for c in [x for x in cases if x['partition']==part][:n]]
    return {'schema':'R22573_C57_PREOUTPUT_SEAL_V1','candidate':'C57','weights_present':0,'model_outputs_observed':0,'training_rows_used':0,'scientific_question':'Does a GEC LoRA learn localized minimal-edit repair beyond an already editing-specialized Base while preserving correct text and protected names/numbers/facts, rather than broad paraphrase or style drift?','suite_n':len(cases),'suite_sha256':suite_sha(cases),'partitions':{p:sum(c['partition']==p for c in cases) for p in ['single_error','protected_span','noop','multi_error','lexical_ood']},'input_contract':PREFIX+'<source sentence>','conditions':['BASE','FULL','RANDOM_RANK_MATCHED','LAYER_SHUFFLED','DOSE_0p25','DOSE_0p50','DOSE_1p50'],'control_ids':pick([('single_error',24),('protected_span',18),('noop',18),('multi_error',18),('lexical_ood',18)]),'generation_ids':pick([('single_error',16),('protected_span',12),('noop',12),('multi_error',12),('lexical_ood',12)]),'causal_ids':pick([('single_error',16),('protected_span',12),('noop',12),('multi_error',12),('lexical_ood',12)]),'causal_generation_ids':pick([('single_error',6),('protected_span',5),('noop',5),('multi_error',4),('lexical_ood',4)]),'causal_localization_gate':'only if positive or failure behavior lane admitted: a broad ablation is localized-required when it erases >=50% of FULL-vs-BASE paired NLL gain on causal_ids, worsens NLL vs FULL by >=0.015 nats/token, and lowers generation composite vs FULL by >=0.05 on causal_generation_ids; E3 credit requires both NLL and generation directions and no protected-span preservation improvement explaining the drop','generation_conditions':['BASE','FULL'],'paired_metric':'target token NLL; lower is better','generation_n':64,'generation_metric':'exact target + required repair + protected-span preservation + no-op false edit + excess edit + target token similarity','positive_gate':'FULL-BASE target-NLL paired CI low>0.01; FULL-RANDOM>=0.015 and FULL-SHUFFLE>=0.015 nats/token on locked controls; generation composite gain>=0.08 OR exact-target gain>=0.10; protected preservation drop<=0.03; no-op false-edit increase<=0.05','failure_gate':'FULL target-NLL paired CI high<0 OR generation composite drop<=-0.08, with learned FULL worse than RANDOM/SHUFFLE by>=0.015 nats/token or no-op false-edit increase>=0.15; protected-span verifier same direction','causal_if_admitted':['MINUS_ENCODER_ATTN','MINUS_DECODER_SELF_ATTN','MINUS_CROSS_ATTN','MINUS_FFN','MINUS_EARLY','MINUS_MIDDLE','MINUS_LATE'],'verifier_smoke':{'perfect':perfect,'source_copy':copy,'overedit_mutation':drift},'wallclock_gate':'exact full random CoEdIT architecture + synthetic LoRA microbenchmark must predict planned workload <=70% of hard timeout before binary','license_boundary':'Adapter repo card MIT; Base CoEdIT-large CC-BY-NC-4.0. Research-only ingress; no raw redistribution; derived raw-free evidence only.','claim_boundary':'fresh synthetic GEC assay only; not writing-quality, authorship, educational, or general language competence proof.'}
