from __future__ import annotations
from pathlib import Path
import argparse,json,random,hashlib
INSTRUCTION=("Extract the following fields from the OCR text as JSON: company_name, address, date, total_amount, line_items (each with item_name, quantity, price). Use null for any field that cannot be determined.")
companies=['NORTH STAR MARKET','GREEN ORCHARD FOODS','RIVER MART','SUNRISE PANTRY','CITY CORNER SHOP','BLUE HARBOR STORE','MAPLE GROCERS','CEDAR MINI MART']
streets=['12 PARK ROAD','88 LAKE AVENUE','31 MARKET STREET','7 CEDAR LANE','105 HILL ROAD','44 KING STREET','9 RIVER DRIVE','63 GARDEN WAY']
items=['RICE','MILK','TEA','BREAD','APPLES','SOAP','COFFEE','NOODLES','JUICE','EGGS','FLOUR','YOGURT']
noise_map={'O':'0','I':'1','E':'3','A':'4','S':'5','G':'6','T':'7','B':'8','Z':'2'}
SPLITS=[('DISCOVERY',48),('CONFIRMATION',48),('ARITHMETIC_CROSSFIELD',48),('OCR_NOISE_OOD',48),('NOOP_CLEAN',32),('NEGATIVE_AMBIGUOUS',32)]
def noisy(s,rng,level):
    if level=='clean': return s
    prob={'light':0.045,'medium':0.09,'heavy':0.15}[level]; out=[]
    for ch in s:
        up=ch.upper()
        if up in noise_map and rng.random()<prob: out.append(noise_map[up])
        elif ch==' ' and rng.random()<prob/5: out.append('_')
        else: out.append(ch)
    return ''.join(out)
def make_case(i,split,rng):
    company=companies[i%len(companies)]; address=streets[(i*3)%len(streets)]
    y=2023+(i%4); m=1+((i*5)%12); d=1+((i*7)%27); date=f'{y:04d}-{m:02d}-{d:02d}'
    n=1+(i%4); rows=[]; line_items=[]; total=0.0
    for j in range(n):
        name=items[(i*2+j*3)%len(items)]; qty=1+((i+j)%3); price=round(1.25+(((i*17+j*11)%1800)/100),2); line=round(qty*price,2); total=round(total+line,2)
        rows.append(f'{name}  QTY {qty}  {price:.2f}  {line:.2f}'); line_items.append({'item_name':name,'quantity':qty,'price':price})
    missing=set()
    if split=='NEGATIVE_AMBIGUOUS':
        mode=i%4; missing.add(['address','company_name','date','total_amount'][mode])
    target={'company_name':None if 'company_name' in missing else company,'address':None if 'address' in missing else address,'date':None if 'date' in missing else date,'total_amount':None if 'total_amount' in missing else total,'line_items':line_items}
    lines=[]
    if 'company_name' not in missing: lines.append(company)
    if 'address' not in missing: lines.append(address)
    if 'date' not in missing: lines.append('DATE: '+date)
    lines.extend(rows)
    if 'total_amount' not in missing: lines.append(f'NET PAYABLE: {total:.2f}')
    if split=='NOOP_CLEAN': level='clean'
    elif split=='OCR_NOISE_OOD': level='heavy'
    elif split=='ARITHMETIC_CROSSFIELD': level='medium'
    else: level=['light','medium'][i%2]
    text='\n'.join(noisy(x,rng,level) for x in lines)
    if split=='OCR_NOISE_OOD' and i%3==0: text=text.replace('\n','\n---\n')
    return {'case_id':f'C61-{i:04d}','split':split,'instruction':INSTRUCTION,'ocr_text':text,'target':target,'target_json':json.dumps(target,ensure_ascii=False,separators=(',',':')),'noise_level':level,'source':'PROCEDURAL_NO_THIRD_PARTY_ROWS','has_crossfield_sum_invariant':True}
def build(path):
    rng=random.Random(22579); rows=[]; idx=0
    for split,count in SPLITS:
        for _ in range(count): rows.append(make_case(idx,split,rng)); idx+=1
    b=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows).encode(); Path(path).write_bytes(b); return len(rows),hashlib.sha256(b).hexdigest()
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); n,h=build(a.out); print(json.dumps({'count':n,'sha256':h},sort_keys=True))
