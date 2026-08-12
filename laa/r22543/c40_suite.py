from __future__ import annotations
import hashlib,json

NAMES=['Mira','Noah','Lena','Omar','Priya','Joon','Sara','Iris','Theo','Aya','Nico','Zara']
PROJECTS=['Atlas','Comet','Orchid','Harbor','Lucent','Juniper','Nimbus','Quartz']
FOODS=['mango','ramen','pears','lentil soup','dark chocolate','sushi','oatmeal','kimchi']
COLORS=['teal','amber','violet','navy','coral','silver','olive','maroon']
CITIES=['Da Nang','Hanoi','Busan','Osaka','Lisbon','Toronto','Berlin','Taipei']
EXPECTED_SHA='b01aac567a1e6165475a018e0dfe447ef48b33ec47e6a51bb913d841fd690cec'
PARENT_SHA='ff47f83a79c1afa9830fc1e943790054626c99d8cff668e60a39ff3d07c41198'

def _abstract():
    rows=[]
    def add(gen,cls,i,existing,message,slot,old,new,action):
        cid=f'C40-{gen}-{cls}-{i:03d}'
        o={'case_id':cid,'generator':gen,'class':cls,'existing_memory':existing,'message':message,'slot':slot,'old_value':old,'new_value':new,'expected_action':action}
        o['semantic_sha256']=hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False).encode()).hexdigest(); rows.append(o)
    for gen in ['A','B']:
      for i in range(24):
        n=NAMES[(i+(0 if gen=='A' else 5))%len(NAMES)];p=PROJECTS[(i*3+(1 if gen=='B' else 0))%len(PROJECTS)]
        f=FOODS[(i*5+(2 if gen=='B' else 0))%len(FOODS)];c=COLORS[(i*7+(3 if gen=='B' else 0))%len(COLORS)];city=CITIES[(i*2+(4 if gen=='B' else 0))%len(CITIES)]
        msg=f'{n} says: "For future meal planning, remember that I prefer {f}."' if gen=='A' else f'Persistent profile update from {n}: favorite color is {c}; use this in later personalization.'
        add(gen,'STORE',i,[],msg,'preference',None,f if gen=='A' else c,'STORE')
        msg=f'{n} says: "I just opened the {p} dashboard and will close it after this check."' if gen=='A' else f'One-off status from {n}: "The terminal is currently on screen while I inspect {p}."'
        add(gen,'IGNORE',i,[{'slot':'home_city','value':city}],msg,'transient',None,None,'IGNORE')
        old=CITIES[(i+1)%len(CITIES)];new=CITIES[(i+3)%len(CITIES)];existing=[{'slot':'home_city','value':old}]
        msg=f'{n} says: "I moved permanently; my home city is now {new}, not {old}."' if gen=='A' else f'Durable correction for {n}: replace the saved home_city value {old} with {new}.'
        add(gen,'UPDATE',i,existing,msg,'home_city',old,new,'UPDATE')
        existing=[{'slot':'favorite_color','value':c}];msg=f'{n} repeats: "My favorite color is {c}."' if gen=='A' else f'Profile note repeats an already saved value for {n}: favorite_color={c}.'
        add(gen,'DUPLICATE',i,existing,msg,'favorite_color',c,c,'NOOP')
    for r in rows:
        i=int(r['case_id'].split('-')[-1]);r['split']='DISCOVERY' if i<8 else ('CONFIRMATION' if i<16 else 'GENERATOR_HOLDOUT')
    return rows

def build_wire():
    wire=[]
    for c in _abstract():
        expected='BOUNDARY' if c['class']=='DUPLICATE' else ('IGNORE' if c['class']=='IGNORE' else 'STORE_LIKE')
        wire.append({'case_id':c['case_id'],'split':c['split'],'generator':c['generator'],'semantic_class':c['class'],'assistant_response':c['message'],'expected_gate':expected,'semantic_sha256':c['semantic_sha256']})
    out={'schema':'LUCIA_AA_R22543_C40_STORAGE_WIRE_SUITE_V1','parent_abstract_suite_sha256':PARENT_SHA,'abi_receipt':'receipts/r22543_c40/C40_PRIVATE_ABI_DELETION_RECEIPT.json','counts':{'total':192,'scored':144,'store_like':96,'ignore':48,'boundary_duplicate':48},'duplicate_policy':'boundary-only; no primary gate credit or penalty because storage ABI lacks existing-memory state','exact_training_prompt_text_exported':False,'model_outputs_observed_at_render':0,'cases':wire}
    raw=json.dumps(out,indent=2,sort_keys=True,ensure_ascii=False)+'\n'
    assert hashlib.sha256(raw.encode()).hexdigest()==EXPECTED_SHA
    return out
