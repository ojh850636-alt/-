import hashlib,json,sys
from pathlib import Path
OUT=Path(sys.argv[1])
eng_nouns=['flight','ticket','meeting','phone','hotel','train','doctor','market','office','payment','order','account','package','project','server','code']
eng_verbs=['book','check','open','send','call','cancel','confirm','update','start','finish','review','save','change','bring','share','print']
hi=['mera','mujhe','aaj','kal','ghar','dost','chai','paani','bahut','acha','nahi','hai','tha','karna','jana','chahiye','abhi','phir','jaldi','thoda','sahi','wala','wali','ke','liye','se','ko','aur','par','mein','kya','kaise']
nums=['zero','one','two','three','four','five','six','seven','eight','nine']
acronyms=['pee-en-aar','oh-tee-pee','eye-dee','you-pee-eye','pee-em','ay-em','ess-em-ess','you-ar-el']
rows=[];seen=set()
def uniq_words(n): return f'{nums[(n//100)%10]} {nums[(n//10)%10]} {nums[n%10]}'
def add(split,family,text,gate=True,note=''):
    text=' '.join(text.split()); text=f'{text} marker {uniq_words(len(rows))}'
    if text in seen:return False
    seen.add(text);rows.append({'case_id':f'C33-{len(rows):04d}','split':split,'family':family,'input_text':text,'prompt':f'Input: {text}\\nOutput:','capability_gate_eligible':gate,'reference_status':'NOT_MATERIALIZED_PRE_MODEL','note':note});return True
for i in range(80): add('DISCOVERY','ENGLISH_CORE',f"please {eng_verbs[i%16]} the {eng_nouns[(i*3)%16]} {nums[(i//16)%10]}")
for i in range(112): add('DISCOVERY','HINGLISH_CORE',f"{hi[i%32]} {eng_nouns[(i*5)%16]} {hi[(i*7+3)%32]} {eng_verbs[(i*11)%16]} {hi[(i*13+5)%32]}")
for i in range(96): add('CONFIRMATION','HINGLISH_CONFIRM',f"{eng_verbs[(i*7)%16]} {hi[(i*11)%32]} {eng_nouns[(i*13)%16]} {hi[(i*17+9)%32]} {nums[(i*3)%10]}")
for i in range(64): add('CONFIRMATION','ACRONYM_DIGIT',f"mera {eng_nouns[i%16]} {acronyms[i%8]} {nums[(i+1)%10]} {nums[(i+4)%10]} {hi[(i*3)%32]} hai")
loans=['quartz','xylophone','router','cache','kernel','docker','python','cinema','guitar','sushi','croissant','metro','quantum','neuron','satellite','volcano']
for i in range(64): add('LEXICAL_OOD','LOAN_TECH_OOD',f"{hi[(i*5)%32]} {loans[i%16]} {eng_verbs[(i*9)%16]} {loans[(i*7+3)%16]} {hi[(i*11+1)%32]}")
for i in range(64): add('STRUCTURAL_OOD','LONG_COMPOSITION',f"{hi[i%32]} {eng_nouns[i%16]} {hi[(i+5)%32]} {acronyms[i%8]} {nums[i%10]} {nums[(i+2)%10]} {eng_verbs[(i+4)%16]} {hi[(i+9)%32]} {eng_nouns[(i+6)%16]}")
dev=['मेरा टिकट तैयार है','मुझे आज घर जाना है','कल मीटिंग कितने बजे है','यह फोन नंबर सही है','पानी और चाय ले आओ','मेरा दोस्त ऑफिस में है','आज ट्रेन जल्दी आएगी','मुझे प्रोजेक्ट भेजना है']
for i in range(32): add('BOUNDARY','DEVANAGARI_HELDOUT',dev[i%8]+f' {i+1}',False,'Model card states Devanagari-carrier lines were held out; boundary only, never primary capability gate.')
assert len(rows)==512 and len(seen)==512
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
sha=hashlib.sha256(OUT.read_bytes()).hexdigest(); assert sha=='463d80e249bf75fc2d0daf7b2e8b6eac33b64b4a9f0b64e88204dc33c3ae5367',sha
print(json.dumps({'cases':512,'sha256':sha}))
