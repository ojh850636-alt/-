from __future__ import annotations
import json, hashlib, random
from pathlib import Path
SEED=22565
LABELS={0:'World',1:'Sports',2:'Business',3:'Sci/Tech'}

def h(b:bytes): return hashlib.sha256(b).hexdigest()

def build(out:Path):
    r=random.Random(SEED); rows=[]
    countries=['Norland','Estavia','Meridia','Valoria','Caledon','Ardell']
    orgs=['Aster Corp','Nova Dynamics','Helix Systems','Orion Labs','Pioneer Group','VectorWorks']
    teams=['Falcons','Tigers','Mariners','Comets','Rovers','Titans']
    cities=['Riverton','Lakeside','Hillcrest','Stonebridge','Northport','Fairview']
    sports=['football','basketball','tennis','cricket','hockey','baseball']
    tech=['quantum processor','AI accelerator','satellite sensor','battery chemistry','robotics platform','cybersecurity patch']
    # 120 clean per class = 480
    for i in range(120):
        c=countries[i%len(countries)]; c2=countries[(i+2)%len(countries)]; city=cities[i%len(cities)]
        variants=[
          f'{c} and {c2} sign a border treaty after three days of diplomatic talks in {city}.',
          f'The parliament of {c} approves an election reform bill after a nationwide vote.',
          f'Foreign ministers from {c} and {c2} meet at the United Nations to discuss a ceasefire.',
          f'{c} announces new visa rules for citizens arriving from {c2}.',
        ]
        rows.append({'case_id':f'C48W{i:03d}','label':0,'label_name':'World','split':'DISCOVERY' if i<64 else 'CONFIRMATION','family':['diplomacy','politics','international','policy'][i%4],'text':variants[i%4]})
    for i in range(120):
        t=teams[i%len(teams)]; t2=teams[(i+3)%len(teams)]; sp=sports[i%len(sports)]; city=cities[i%len(cities)]
        variants=[
          f'The {t} defeated the {t2} 3-1 in the {sp} championship final in {city}.',
          f'Coach Mira Solano named a new starting lineup before the {t} playoff match.',
          f'A late goal carried the {t} into the semifinals of the national {sp} tournament.',
          f'Rookie striker Eli Voss scored twice as the {t} won their league opener.',
        ]
        rows.append({'case_id':f'C48S{i:03d}','label':1,'label_name':'Sports','split':'DISCOVERY' if i<64 else 'CONFIRMATION','family':['match','team','tournament','player'][i%4],'text':variants[i%4]})
    for i in range(120):
        o=orgs[i%len(orgs)]; o2=orgs[(i+2)%len(orgs)]; city=cities[i%len(cities)]
        variants=[
          f'{o} reported quarterly revenue of ${120+i} million and raised its profit forecast.',
          f'Shares of {o} rose after the company agreed to acquire {o2} for ${2+i%7} billion.',
          f'{o} will open a new manufacturing plant in {city} and hire 800 workers.',
          f'The central bank rate decision pushed banking and retail stocks higher on Monday.',
        ]
        rows.append({'case_id':f'C48B{i:03d}','label':2,'label_name':'Business','split':'DISCOVERY' if i<64 else 'CONFIRMATION','family':['earnings','merger','industry','markets'][i%4],'text':variants[i%4]})
    for i in range(120):
        o=orgs[i%len(orgs)]; x=tech[i%len(tech)]; city=cities[i%len(cities)]
        variants=[
          f'Researchers in {city} demonstrated a new {x} that cuts energy use by 30 percent.',
          f'{o} released an open-source software update that fixes a critical encryption flaw.',
          f'A space telescope transmitted high-resolution images using a redesigned optical sensor.',
          f'Engineers trained a compact machine-learning model to detect faults in industrial robots.',
        ]
        rows.append({'case_id':f'C48T{i:03d}','label':3,'label_name':'Sci/Tech','split':'DISCOVERY' if i<64 else 'CONFIRMATION','family':['research','software','space','ai'][i%4],'text':variants[i%4]})
    # 32 lexical/OOD, still unambiguous (8 per class)
    ood=[
      (0,'The ambassador was recalled after two governments failed to renew a maritime accord.'),
      (1,'The seeded doubles pair advanced after winning a five-set match at the open.'),
      (2,'The insurer cut its annual earnings outlook after claims expenses rose sharply.'),
      (3,'A new compiler optimization reduces memory traffic on heterogeneous accelerators.'),
    ]
    for j in range(32):
        lab,txt=ood[j%4]
        rows.append({'case_id':f'C48O{j:03d}','label':lab,'label_name':LABELS[lab],'split':'TOOL_OOD','family':'lexical_ood','text':txt.replace('.',f' ({j+1}).')})
    # 64 mixed/ambiguous diagnostics, excluded from accuracy gate
    mixed=[
      'A technology company rose 12 percent after unveiling a new AI chip for data centers.',
      'A national football federation signed a sponsorship agreement with a major bank.',
      'Government regulators opened an antitrust inquiry into a cloud-computing acquisition.',
      'A sports broadcaster launched an AI system to generate real-time match statistics.',
    ]
    for i in range(64):
        rows.append({'case_id':f'C48M{i:03d}','label':None,'label_name':'AMBIGUOUS','split':'AMBIGUOUS_MIXED','family':'mixed_topic','text':mixed[i%4]})
    assert len(rows)==576 and len({x['case_id'] for x in rows})==576
    r.shuffle(rows)
    data=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows).encode()
    out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(data)
    meta={'schema':'LAA_R22565_C48_SUITE_V1','cases':576,'scored_cases':512,'ambiguous_diagnostic':64,'labels':LABELS,'sha256':h(data),'generation':'procedural_fresh_no_ag_news_rows','seed':SEED}
    print(json.dumps(meta,sort_keys=True)); return meta
if __name__=='__main__': build(Path('C48_AGNEWS_FRESH_SUITE.jsonl'))
