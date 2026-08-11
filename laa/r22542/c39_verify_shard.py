from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path

def loadmod(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private',required=True);ap.add_argument('--lean-root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 v=loadmod('laa/r22542/c39_lean_verify.py','c39_verify_mod');rows={x['case_id']:x for x in v.suite()};rec=[json.loads(x) for x in Path(a.private).read_text(encoding='utf-8').splitlines() if x.strip()];tmp=Path(a.lean_root)/'C39Shard.lean';results=[]
 for x in rec:
  if x['status']!='OK':results.append({'condition':x['condition'],'seed':x['seed'],'case_id':x['case_id'],'split':x['split'],'rank':x['rank'],'verified':False,'error_class':x['status']});continue
  tmp.write_text(v.render(rows[x['case_id']],x['tactic']),encoding='utf-8');p=subprocess.run(['lake','env','lean',tmp.name],cwd=a.lean_root,text=True,capture_output=True,timeout=45);log=(p.stdout or '')+'\n'+(p.stderr or '');results.append({'condition':x['condition'],'seed':x['seed'],'case_id':x['case_id'],'split':x['split'],'rank':x['rank'],'verified':p.returncode==0,'error_class':'PASS' if p.returncode==0 else v.classify(log),'lean_log_sha256':hashlib.sha256(log.encode()).hexdigest()})
 tmp.unlink(missing_ok=True);groups={}
 for x in results:groups.setdefault((x['condition'],x['seed'],x['split'],x['case_id']),[]).append(x)
 cases=[]
 for (c,s,sp,cid),rr in groups.items():cases.append({'condition':c,'seed':s,'split':sp,'case_id':cid,'success':any(y['verified'] for y in rr),'success_count':sum(y['verified'] for y in rr),'best_rank':min((y['rank'] for y in rr if y['verified']),default=None),'error_counts':{e:sum(y['error_class']==e for y in rr) for e in sorted({y['error_class'] for y in rr})}})
 Path(a.out).write_text(json.dumps({'schema':'LUCIA_AA_R22542_C39_LEAN_GENERATION_SHARD_V1','records_verified':len(results),'case_results':cases,'raw_tactics_exported':False},indent=2,sort_keys=True)+'\n');Path(a.private).unlink(missing_ok=True)
if __name__=='__main__':main()
