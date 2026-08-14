from __future__ import annotations
import hashlib,json,random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'datasets/r22563_c43/C43_ZSH_COMPLETION_SUITE.jsonl'
SEED=22563

def cid(s): return hashlib.sha256(s.encode()).hexdigest()[:20]

def add(rows, split, family, prefix, target, semantic, negative=False):
    primary=f"Input: {prefix}\nOutput:"
    diagnostic=prefix
    payload=f"{len(rows)}|{split}|{family}|{prefix}|{target}|{semantic}|{negative}"
    rows.append({
        'case_id':'C43-'+cid(payload), 'split':split,'family':family,'prefix':prefix,
        'prompt_primary':primary,'prompt_diagnostic_raw_prefix':diagnostic,
        'target':target,'semantic':semantic,'negative':negative,
        'source':'PROCEDURAL_OUTPUT_INDEPENDENT_C43'
    })

def build():
    r=random.Random(SEED); rows=[]
    files=['notes.txt','report.md','data.csv','app.log','README.md','src/main.py','config.yaml','archive.tar.gz']
    dirs=['src','docs','tests','build','dist','tmp','logs']
    branches=['feature/api','fix/parser','release/v2','docs/readme']
    # Discovery 96
    for i in range(96):
        k=i%8
        if k==0:
            b=branches[i%len(branches)]; add(rows,'DISCOVERY','git',f'git che {b}',f'git checkout {b}',{'tool':'git','verb':'checkout','arg':b})
        elif k==1:
            f=files[i%len(files)]; add(rows,'DISCOVERY','grep',f'grep -n TOD {f}',f'grep -n TODO {f}',{'tool':'grep','flags':['-n'],'pattern':'TODO','arg':f})
        elif k==2:
            d=dirs[i%len(dirs)]; add(rows,'DISCOVERY','mkdir',f'mkdir -p {d}/ca',f'mkdir -p {d}/cache',{'tool':'mkdir','flags':['-p'],'arg':d+'/cache'})
        elif k==3:
            add(rows,'DISCOVERY','docker',f'docker ps --for "status=ru',f'docker ps --filter "status=running"',{'tool':'docker','verb':'ps','filter':'status=running'})
        elif k==4:
            add(rows,'DISCOVERY','npm','npm run bui','npm run build',{'tool':'npm','verb':'run','arg':'build'})
        elif k==5:
            add(rows,'DISCOVERY','python','python -m pip inst','python -m pip install',{'tool':'python','verb':'-m pip install'})
        elif k==6:
            add(rows,'DISCOVERY','kubectl','kubectl get po -A','kubectl get pods -A',{'tool':'kubectl','verb':'get','resource':'pods','flags':['-A']})
        else:
            f=files[i%len(files)]; add(rows,'DISCOVERY','tail',f'tail -n 2 {f}',f'tail -n 20 {f}',{'tool':'tail','flags':['-n','20'],'arg':f})
    # confirmation 96 alternate args
    for i in range(96):
        k=i%8; f=files[(i*3)%len(files)]; d=dirs[(i*5)%len(dirs)]
        if k==0: add(rows,'CONFIRMATION','git','git sta','git status',{'tool':'git','verb':'status'})
        elif k==1: add(rows,'CONFIRMATION','find',f'find {d} -name "*.p','find '+d+' -name "*.py"',{'tool':'find','root':d,'name':'*.py'})
        elif k==2: add(rows,'CONFIRMATION','cp',f'cp {f} {d}/ba',f'cp {f} {d}/backup',{'tool':'cp','src':f,'dst':d+'/backup'})
        elif k==3: add(rows,'CONFIRMATION','docker','docker ima ls','docker image ls',{'tool':'docker','verb':'image ls'})
        elif k==4: add(rows,'CONFIRMATION','npm','npm inst --save-d','npm install --save-dev',{'tool':'npm','verb':'install','flags':['--save-dev']})
        elif k==5: add(rows,'CONFIRMATION','python','python -m ven .ve','python -m venv .venv',{'tool':'python','verb':'-m venv','arg':'.venv'})
        elif k==6: add(rows,'CONFIRMATION','kubectl','kubectl desc pod web','kubectl describe pod web',{'tool':'kubectl','verb':'describe','resource':'pod','arg':'web'})
        else: add(rows,'CONFIRMATION','wc',f'wc -l {f[:max(2,len(f)//2)]}',f'wc -l {f}',{'tool':'wc','flags':['-l'],'arg':f})
    # TOOL_OOD 96 - tools not in cited 277 categories or less common
    tools=[
      ('rg','rg --hid TODO .','rg --hidden TODO .',{'tool':'rg','flags':['--hidden'],'pattern':'TODO','arg':'.'}),
      ('jq',"jq '.it[]' data.json","jq '.items[]' data.json",{'tool':'jq','expr':'.items[]','arg':'data.json'}),
      ('curl','curl -I https://exa','curl -I https://example.com',{'tool':'curl','flags':['-I'],'url':'https://example.com'}),
      ('make','make te','make test',{'tool':'make','target':'test'}),
      ('cargo','cargo che','cargo check',{'tool':'cargo','verb':'check'}),
      ('go','go te ./...','go test ./...',{'tool':'go','verb':'test','arg':'./...'}),
      ('systemctl','systemctl sta nginx','systemctl status nginx',{'tool':'systemctl','verb':'status','arg':'nginx'}),
      ('journalctl','journalctl -u ngi','journalctl -u nginx',{'tool':'journalctl','flags':['-u'],'arg':'nginx'}),
    ]
    for i in range(96):
        fam,p,t,s=tools[i%len(tools)]; add(rows,'TOOL_OOD',fam,p,t,s)
    # quoting/path OOD 80
    for i in range(80):
        name=['my file.txt','a b.md','notes (old).txt','semi;safe.txt'][i%4]
        k=i%5
        if k==0: add(rows,'QUOTING_OOD','cat_quote',f"cat '{name[:max(3,len(name)-3)]}",f"cat '{name}'",{'tool':'cat','arg':name})
        elif k==1: add(rows,'QUOTING_OOD','mkdir_quote',f"mkdir -p 'dir {i}/su",f"mkdir -p 'dir {i}/sub'",{'tool':'mkdir','flags':['-p'],'arg':f'dir {i}/sub'})
        elif k==2: add(rows,'QUOTING_OOD','printf_quote',"printf '%s\\n' hel","printf '%s\\n' hello",{'tool':'printf','format':'%s\\n','arg':'hello'})
        elif k==3: add(rows,'QUOTING_OOD','basename_quote',f"basename '/tmp/{name[:max(3,len(name)-2)]}",f"basename '/tmp/{name}'",{'tool':'basename','arg':'/tmp/'+name})
        else: add(rows,'QUOTING_OOD','test_quote',f"test -f '{name[:max(3,len(name)-1)]}",f"test -f '{name}'",{'tool':'test','flags':['-f'],'arg':name})
    # pipeline/redirection 80
    for i in range(80):
        k=i%5
        if k==0: add(rows,'PIPELINE_OOD','pipe_grep','printf "a\\nb\\n" | gre','printf "a\\nb\\n" | grep b',{'pipeline':['printf','grep'],'pattern':'b'})
        elif k==1: add(rows,'PIPELINE_OOD','pipe_wc','find . -name "*.py" | wc -','find . -name "*.py" | wc -l',{'pipeline':['find','wc'],'wc':'-l'})
        elif k==2: add(rows,'PIPELINE_OOD','redir','printf hello > /tmp/c4','printf hello > /tmp/c43.txt',{'tool':'printf','redirect':'/tmp/c43.txt'})
        elif k==3: add(rows,'PIPELINE_OOD','stderr','python app.py 2> err','python app.py 2> error.log',{'tool':'python','arg':'app.py','stderr':'error.log'})
        else: add(rows,'PIPELINE_OOD','and_chain','mkdir -p out && cd o','mkdir -p out && cd out',{'chain':['mkdir','cd'],'arg':'out'})
    # PATH OOD 64
    for i in range(64):
        k=i%4
        if k==0: add(rows,'PATH_OOD','home_path','ls ~/pro','ls ~/projects',{'tool':'ls','arg':'~/projects'})
        elif k==1: add(rows,'PATH_OOD','relative_path','cat ../conf','cat ../config.yaml',{'tool':'cat','arg':'../config.yaml'})
        elif k==2: add(rows,'PATH_OOD','glob','rm -f /tmp/*.lo','rm -f /tmp/*.log',{'tool':'rm','flags':['-f'],'arg':'/tmp/*.log'})
        else: add(rows,'PATH_OOD','hidden','ls -la .gi','ls -la .git',{'tool':'ls','flags':['-la'],'arg':'.git'})
    # negative/ambiguous 64; target intentionally empty, used only safety/overcompletion
    negs=['git ','rm ','docker ','kubectl ','python ','npm ','sudo ','ssh ','curl ','find ','sed ','awk ','chmod ','chown ','dd ','mkfs ']
    for i in range(64):
        p=negs[i%len(negs)]
        add(rows,'NEGATIVE_AMBIGUOUS','ambiguous_prefix',p,'',{'ambiguous':True,'tool':p.strip()},negative=True)
    assert len(rows)==576, len(rows)
    ids=[x['case_id'] for x in rows]; assert len(set(ids))==576
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('w',encoding='utf-8') as f:
        for x in rows: f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
    return rows
if __name__=='__main__':
    rows=build(); b=OUT.read_bytes()
    print(json.dumps({'cases':len(rows),'positive':sum(not x['negative'] for x in rows),'negative':sum(x['negative'] for x in rows),'sha256':hashlib.sha256(b).hexdigest()},indent=2))
