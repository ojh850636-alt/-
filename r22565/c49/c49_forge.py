from __future__ import annotations
import hashlib, json, random
from pathlib import Path
SEED=22565
LABELS={0:'World',1:'Sports',2:'Business',3:'Sci/Tech'}

def H(b:bytes): return hashlib.sha256(b).hexdigest()

def build(out_path: str):
    r=random.Random(SEED); rows=[]
    countries=['Japan','Brazil','Kenya','Poland','Vietnam','Canada','Chile','Egypt','Norway','India','Mexico','Greece']
    cities=['Tokyo','Brasilia','Nairobi','Warsaw','Hanoi','Ottawa','Santiago','Cairo','Oslo','Delhi','Mexico City','Athens']
    teams=['Falcons','Tigers','Rovers','United','Warriors','Cyclones','Mariners','Rockets','Lions','Eagles','Giants','Comets']
    sports=['football','basketball','tennis','cricket','rugby','baseball','hockey','volleyball']
    companies=['Northstar','BluePeak','Orion Motors','Cedar Bank','Nova Retail','Harbor Foods','Summit Air','Vertex Energy','Atlas Media','Pioneer Labs','Metro Telecom','Crescent Steel']
    tech=['quantum sensor','AI accelerator','battery chemistry','robotics platform','open-source compiler','satellite modem','gene-editing tool','fusion magnet','photonic chip','cybersecurity protocol','space telescope','medical imaging system']
    for i in range(64):
        c=countries[i%len(countries)]; d=countries[(i+3)%len(countries)]; city=cities[i%len(cities)]
        rows.append({'case_id':f'C49-W-D-{i:03d}','label':0,'split':'DISCOVERY','family':'world_core','text':f'{c} and {d} signed a new diplomatic accord after talks in {city}, officials announced.'})
        team=teams[i%len(teams)]; opp=teams[(i+5)%len(teams)]; sport=sports[i%len(sports)]
        rows.append({'case_id':f'C49-S-D-{i:03d}','label':1,'split':'DISCOVERY','family':'sports_core','text':f'{team} defeated {opp} in the {sport} final after a late comeback, securing the championship.'})
        co=companies[i%len(companies)]; n=(i%17)+2
        rows.append({'case_id':f'C49-B-D-{i:03d}','label':2,'split':'DISCOVERY','family':'business_core','text':f'{co} reported quarterly revenue growth of {n}% and raised its profit forecast for the year.'})
        item=tech[i%len(tech)]; org=['university researchers','an engineering team','a national laboratory','software developers'][i%4]
        rows.append({'case_id':f'C49-T-D-{i:03d}','label':3,'split':'DISCOVERY','family':'scitech_core','text':f'{org} unveiled a new {item} that improves performance in controlled tests.'})
    for i in range(40):
        c=countries[(i+2)%len(countries)]; city=cities[(i+4)%len(cities)]
        rows.append({'case_id':f'C49-W-C-{i:03d}','label':0,'split':'CONFIRMATION','family':'world_policy','text':f'Parliament in {c} approved a national election reform bill after a week of debate in {city}.'})
        sport=sports[(i+2)%len(sports)]; team=teams[(i+1)%len(teams)]
        rows.append({'case_id':f'C49-S-C-{i:03d}','label':1,'split':'CONFIRMATION','family':'sports_result','text':f'{team} advanced to the semifinals of the international {sport} tournament with a record-breaking performance.'})
        co=companies[(i+2)%len(companies)]; co2=companies[(i+7)%len(companies)]
        rows.append({'case_id':f'C49-B-C-{i:03d}','label':2,'split':'CONFIRMATION','family':'business_deal','text':f'{co} agreed to acquire {co2} in a cash-and-stock deal valued at ${(i%9)+2} billion.'})
        item=tech[(i+3)%len(tech)]
        rows.append({'case_id':f'C49-T-C-{i:03d}','label':3,'split':'CONFIRMATION','family':'scitech_research','text':f'A peer-reviewed study describes a {item} prototype that reduced error rates in repeated laboratory trials.'})
    for i in range(20):
        c=countries[(i+6)%len(countries)]; d=countries[(i+9)%len(countries)]
        rows.append({'case_id':f'C49-W-O-{i:03d}','label':0,'split':'LEXICAL_OOD','family':'world_ood','text':f'Delegates from {c} and {d} reopened border negotiations under a newly appointed mediator.'})
        team=teams[(i+8)%len(teams)]; sport=sports[(i+5)%len(sports)]
        rows.append({'case_id':f'C49-S-O-{i:03d}','label':1,'split':'LEXICAL_OOD','family':'sports_ood','text':f'An underdog {team} squad clinched a playoff berth in {sport} after winning in overtime.'})
        co=companies[(i+9)%len(companies)]
        rows.append({'case_id':f'C49-B-O-{i:03d}','label':2,'split':'LEXICAL_OOD','family':'business_ood','text':f'Shares of {co} fell after auditors warned that operating margins could narrow next quarter.'})
        item=tech[(i+8)%len(tech)]
        rows.append({'case_id':f'C49-T-O-{i:03d}','label':3,'split':'LEXICAL_OOD','family':'scitech_ood','text':f'Engineers published benchmark results for an experimental {item}, including energy and latency measurements.'})
    for i in range(20):
        c=countries[i%len(countries)]; co=companies[(i+1)%len(companies)]
        rows.append({'case_id':f'C49-W-M-{i:03d}','label':0,'split':'MIXED_CONTEXT','family':'world_mixed','text':f'Although markets reacted sharply, the main development was {c}\'s government calling a snap national election for next month; {co} shares later recovered.'})
        team=teams[(i+2)%len(teams)]; co=companies[(i+3)%len(companies)]
        rows.append({'case_id':f'C49-S-M-{i:03d}','label':1,'split':'MIXED_CONTEXT','family':'sports_mixed','text':f'{co} sponsored the event, but the headline result was {team} winning the championship on penalties after a tied final.'})
        co=companies[(i+4)%len(companies)]; item=tech[(i+5)%len(tech)]
        rows.append({'case_id':f'C49-B-M-{i:03d}','label':2,'split':'MIXED_CONTEXT','family':'business_mixed','text':f'{co} announced a {item}, but investors focused on its surprise quarterly loss and a cut to full-year revenue guidance.'})
        co=companies[(i+6)%len(companies)]; item=tech[(i+7)%len(tech)]
        rows.append({'case_id':f'C49-T-M-{i:03d}','label':3,'split':'MIXED_CONTEXT','family':'scitech_mixed','text':f'{co} plans to commercialize the product later, but researchers first demonstrated that the new {item} solves a previously measured technical bottleneck.'})
    assert len(rows)==576 and len({x['case_id'] for x in rows})==576
    counts={k:sum(x['label']==k for x in rows) for k in LABELS}; assert counts=={0:144,1:144,2:144,3:144}
    p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True)
    data=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows).encode(); p.write_bytes(data)
    meta={'schema':'LAA_R22565_C49_FRESH_SUITE_V1','cases':576,'labels':LABELS,'counts':counts,'sha256':H(data),'generation':'procedural_fresh_no_ag_news_rows','seed':SEED}
    print(json.dumps(meta,sort_keys=True)); return meta
if __name__=='__main__': build('datasets/r22565_c49/C49_FRESH_NEWS_TOPIC_SUITE.jsonl')
