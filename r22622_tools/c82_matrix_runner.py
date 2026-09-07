import os,re,json,random,pathlib,sys
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

ROOT=pathlib.Path('source_bundle/src'); BASE=str(ROOT/'base'); AD=str(ROOT/'adapter')
shard=int(os.environ['SHARD']); cond=os.environ['CONDITION']; start=shard*8; end=start+8

weather1=['violet rain inside a calculator','fog that counts backward','snowflakes shaped like commas','sunlight trapped under a saucer','a breeze carrying paper stars','mist humming in lowercase','warm drizzle above a chessboard','thunder hiding behind curtains','a frost line drawing spirals','rain that smells like pencils','a cloud balancing on thread','wind folding tiny envelopes','dewdrops ringing like bells','moonlight leaking from a drawer','a sunbeam wearing dust','hail tapping in Morse code']
prop1=['brass thimble','glass key','wooden whistle','silver spoon','red marble','paper lantern','tin compass','blue ribbon','ceramic bell','green domino','copper button','folded fan','clockwork pear','velvet glove','tiny umbrella','porcelain star']
mood1=['patient','skeptical','relieved','curious','solemn','mischievous','hopeful','puzzled','calm','brave','wistful','focused','gentle','stubborn','amused','watchful']
secret1=['a cedar map folded into a coin','a cobalt accordion under the floor','a lemon clock that ticks sideways','a velvet comet inside a jar','a copper moth sleeping in a book','a marble ladder drawn on silk','an amber compass pointing inward','a paper moon stamped with seven dots','a silver acorn full of rain','a blue telescope hiding one feather','a brass envelope sealed with mint','a glass pebble that remembers winter','a red kite folded into a teaspoon','a green bell buried in sugar','a porcelain key wrapped in moss','a tin crown carrying one raindrop']
twist1=['gravity pauses beside the prop','the shadow arrives one minute early','the floor remembers a different room','the prop becomes lighter than its echo','every reflection blinks out of order','the ceiling borrows the weather','the secret briefly becomes audible','time folds once around the prop','a footprint appears before the step','the weather changes places with its shadow','the prop casts two contradictory shadows','the stage grows one impossible corner','all echoes move toward the light','the smallest object becomes the horizon','the final sound happens before the first','the prop forgets which way is down']
end1=['cinder','maple','orbit','ripple','quartz','moss','ember','harbor','cobalt','acorn','velvet','thimble','lantern','pebble','willow','comet']
weather2=['gravity-colored drizzle','a silent aurora in a shoebox','steam spelling prime numbers','a sideways rainbow under glass','cold sunlight with square edges','a pocket storm of feathers','static snow above a paper bridge','a midnight breeze counting sevens','heat shimmer shaped like ladders','a tiny eclipse beneath the table','rainfall moving upward in rows','a pale cyclone made of receipts','golden fog around a metronome','a dawn cloud with mirrored edges','soft thunder inside a bottle','a spiral breeze beneath the floor']
prop2=['ivory gear','linen cube','obsidian bead','wax trumpet','paper hinge','bronze needle','crystal cork','rubber moon','chalk wheel','woolen ring','steel petal','wooden prism','glass feather','copper dice','silk magnet','stone zipper']
mood2=['analytical','uncertain','deliberate','defiant','tranquil','alert','tender','reserved','playful','earnest','restless','confident','suspicious','cheerful','careful','resolute']
secret2=['a saffron ruler measuring echoes','a black violin folded into rain','a nickel staircase with no top','a linen planet sewn to a ticket','a quartz fish hiding in chalk','a bronze sentence missing one verb','a glass orchard inside a matchbox','a wool compass pointing to silence','a paper engine powered by snow','a ceramic ladder smelling of mint','an iron cloud inside a pocket','a violet coin that refuses circles','a wooden eclipse under a stamp','a silver thread tying two mornings','a stone bell containing blue smoke','a copper mirror reflecting next week']
twist2=['all distances become equal for a breath','the prop answers before it is spoken to','the weather briefly becomes a direction','the secret casts a brighter shadow than the prop','one second repeats without repeating','the stage swaps near and far','every edge turns inward at once','the prop weighs exactly one whisper','light travels backward across the floor','the weather becomes a temporary doorway','the final line changes the first shadow','the prop is older than the stage for one beat','four corners become five then return','silence moves from left to right','the secret appears only in reflections','the stage forgets the meaning of below']
end2=['saffron','hinge','aurora','needle','prism','echo','violet','cork','metronome','feather','cipher','orchard','ruler','magnet','spiral','horizon']

def prompt(q):
 return ('Write only the scene text for a tiny strange joyful stage play in 55 words or fewer. The first sentence must name the exact Weather and exact Prop below. Use vivid concrete details, one impossible event, and one short quoted line spoken by the prop. Make the weather and prop the only characters; do not invent named humans. Avoid generic towns, fantasy summaries, questions, and meta-commentary. No preface. No bullet points. No explanation.\n'+f"Weather: {q['weather']}\nProp: {q['prop']}\nMood: {q['mood']}\nSecret: {q['secret']}\nWeirdness level: {q['weirdness']}/5\nRequired twist: {q['twist']}\nDirector challenge: end with the exact lowercase word {q['endword']}\nScene:")

def mk(i,split,w,p,m,s,t,e):
 q={'weather':w,'prop':p,'mood':m,'secret':s,'weirdness':1+(i%5),'twist':t,'endword':e}; return {'id':f'{split[:1]}{i:02d}','split':split,'request':q,'prompt':prompt(q)}
cases=[]
for i in range(16): cases.append(mk(i,'Discovery',weather1[i],prop1[i],mood1[i],secret1[i],twist1[i],end1[i]))
for i in range(16): cases.append(mk(i,'Confirmation',weather1[i],prop1[(i+5)%16],mood1[(i+7)%16],secret1[(i+9)%16],twist1[(i+11)%16],end1[(i+13)%16]))
for i in range(16): cases.append(mk(i,'OOD',weather2[i],prop2[i],mood2[i],secret2[i],twist2[i],end2[i]))
all_secret=[c['request']['secret'].lower() for c in cases]; all_twist=[c['request']['twist'].lower() for c in cases]

def score(c,out):
 q=c['request']; lo=out.lower(); m=re.search(r'[.!?\n]',out.strip()); fs=(out.strip()[:m.start()+1] if m else out.strip()).lower(); words=re.findall(r"\b[\w'-]+\b",out); last=words[-1].lower() if words else ''
 p=re.escape(q['prop'].lower()); quote=bool(re.search(r'[\"“”][^\"“”]{1,140}[\"“”]',out)); speaker=bool(re.search(rf'(said|says|whispered|whispers|cried|murmured|calls)\s+(?:the\s+)?{p}\b',lo) or re.search(rf'\b{p}\b\s+(said|says|whispered|whispers|cried|murmured)',lo))
 d={'weather_exact':q['weather'].lower() in lo,'prop_exact':q['prop'].lower() in lo,'first_sentence_weather_prop':q['weather'].lower() in fs and q['prop'].lower() in fs,'mood_exact':re.search(rf'(?<!\w){re.escape(q["mood"].lower())}(?!\w)',lo) is not None,'secret_exact':q['secret'].lower() in lo,'twist_exact':q['twist'].lower() in lo,'quoted_line':quote,'prop_speaker':speaker,'word_limit_55':len(words)<=55,'endword_exact':last==q['endword'].lower()}
 d['cross_case_intrusions']=sum(1 for x in all_secret if x!=q['secret'].lower() and x in lo)+sum(1 for x in all_twist if x!=q['twist'].lower() and x in lo); d['word_count']=len(words); d['last_word']=last; d['total_binding_score']=sum(int(d[k]) for k in ['weather_exact','prop_exact','first_sentence_weather_prop','mood_exact','secret_exact','twist_exact','quoted_line','prop_speaker','word_limit_55','endword_exact'])/10.0
 return d

tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True); tok.padding_side='left'; tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,dtype=torch.float32); model=PeftModel.from_pretrained(base,AD,local_files_only=True); model.eval()
lora={n:p for n,p in model.named_parameters() if '.lora_A.' in n or '.lora_B.' in n}; orig={n:p.detach().clone() for n,p in lora.items()}; rx=re.compile(r'layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.lora_([AB])\.default\.weight$'); pairs={}
for n,p in lora.items():
 m=rx.search(n)
 if m: pairs.setdefault((m.group(2),m.group(3),int(m.group(1))),{})[m.group(4)]=n
with torch.no_grad():
 if cond=='BASE':
  for n,p in lora.items():
   if '.lora_B.' in n: p.zero_()
 elif cond=='RANDOM_NORM_MATCHED':
  g=torch.Generator(device='cpu'); g.manual_seed(2262201)
  for n,p in lora.items():
   src=orig[n]; r=torch.randn(src.shape,generator=g,dtype=src.dtype); r=r/(torch.linalg.vector_norm(r)+1e-12)*torch.linalg.vector_norm(src); p.copy_(r)
 elif cond=='LAYER_SHUFFLED':
  groups={}
  for (block,mod,layer),d in pairs.items():
   if set(d)=={'A','B'}: groups.setdefault((block,mod),[]).append(layer)
  for gi,((block,mod),layers) in enumerate(sorted(groups.items())):
   layers=sorted(layers); rr=random.Random(2262202+gi); perm=layers[:]; rr.shuffle(perm); perm=perm if perm!=layers else perm[1:]+perm[:1]
   for tgt,src in zip(layers,perm):
    td=pairs[(block,mod,tgt)]; sd=pairs[(block,mod,src)]; lora[td['A']].copy_(orig[sd['A']]); lora[td['B']].copy_(orig[sd['B']])
 elif cond!='FULL': raise ValueError(cond)
rows=[]
for b0 in range(start,end,4):
 batch=cases[b0:min(b0+4,end)]; enc=tok([c['prompt'] for c in batch],return_tensors='pt',padding=True)
 with torch.inference_mode(): gen=model.generate(**enc,max_new_tokens=96,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
 ilen=enc['input_ids'].shape[1]
 for c,g in zip(batch,gen):
  out=tok.decode(g[ilen:],skip_special_tokens=True); rows.append({'id':c['id'],'split':c['split'],'condition':cond,'output':out,'metrics':score(c,out)})
path=pathlib.Path('result'); path.mkdir(exist_ok=True); (path/f'{cond}_{start:02d}_{end:02d}.json').write_text(json.dumps({'schema':'R22622_C82_BINDING_OUTPUT_V1','condition':cond,'start':start,'end':end,'rows':rows},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
