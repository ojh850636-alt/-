from __future__ import annotations
import argparse,hashlib,json,random,re,subprocess,tempfile
from pathlib import Path

def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def mean(xs):
 xs=list(xs);return float(sum(xs)/len(xs)) if xs else 0.0

def gen_a(i,split):
 k=i%8;s=f'{i:03d}'
 if k==0: goal=f'α{s} : Type\nx{s} y{s} : α{s}\nh{s} : x{s} = y{s}\n⊢ y{s} = x{s}';tac=f'exact h{s}.symm';src=f'theorem c39_a_{s} (α{s} : Type) (x{s} y{s} : α{s}) (h{s} : x{s} = y{s}) : y{s} = x{s} := by\n  {tac}'
 elif k==1: goal=f'P{s} Q{s} : Prop\nh{s} : P{s} ∧ Q{s}\n⊢ Q{s} ∧ P{s}';tac=f'exact ⟨h{s}.2, h{s}.1⟩';src=f'theorem c39_a_{s} (P{s} Q{s} : Prop) (h{s} : P{s} ∧ Q{s}) : Q{s} ∧ P{s} := by\n  {tac}'
 elif k==2: goal=f'α{s} : Type\nP{s} : α{s} → Prop\na{s} : α{s}\nh{s} : ∀ x, P{s} x\n⊢ P{s} a{s}';tac=f'exact h{s} a{s}';src=f'theorem c39_a_{s} (α{s} : Type) (P{s} : α{s} → Prop) (a{s} : α{s}) (h{s} : ∀ x, P{s} x) : P{s} a{s} := by\n  {tac}'
 elif k==3: goal=f'a{s} b{s} : Nat\n⊢ a{s} + b{s} = b{s} + a{s}';tac='omega';src=f'theorem c39_a_{s} (a{s} b{s} : Nat) : a{s} + b{s} = b{s} + a{s} := by\n  omega'
 elif k==4: goal=f'a{s} b{s} : Nat\n⊢ a{s} ≤ a{s} + b{s}';tac='omega';src=f'theorem c39_a_{s} (a{s} b{s} : Nat) : a{s} ≤ a{s} + b{s} := by\n  omega'
 elif k==5: goal=f'x{s} y{s} : Int\n⊢ (x{s} + y{s})^2 = x{s}^2 + 2*x{s}*y{s} + y{s}^2';tac='ring';src=f'theorem c39_a_{s} (x{s} y{s} : Int) : (x{s} + y{s})^2 = x{s}^2 + 2*x{s}*y{s} + y{s}^2 := by\n  ring'
 elif k==6: goal=f'α{s} : Type\nxs{s} ys{s} : List α{s}\n⊢ (xs{s} ++ ys{s}).length = xs{s}.length + ys{s}.length';tac='simp';src=f'theorem c39_a_{s} (α{s} : Type) (xs{s} ys{s} : List α{s}) : (xs{s} ++ ys{s}).length = xs{s}.length + ys{s}.length := by\n  simp'
 else: goal=f'P{s} Q{s} : Prop\nh{s} : P{s} → Q{s}\nhp{s} : P{s}\n⊢ Q{s}';tac=f'exact h{s} hp{s}';src=f'theorem c39_a_{s} (P{s} Q{s} : Prop) (h{s} : P{s} → Q{s}) (hp{s} : P{s}) : Q{s} := by\n  {tac}'
 return {'case_id':f'C39-{i:05d}','split':split,'goal_state':goal,'canonical_tactic':tac,'standalone_source':src,'positive':True}

def gen_b(i,split):
 j=1000+i;s=f'{j:04d}';k=i%8
 if k==0:goal=f'P{s} Q{s} : Prop\nh{s} : P{s} ∨ Q{s}\n⊢ Q{s} ∨ P{s}';tac=f'rcases h{s} with h | h <;> simp [h]';src=f'theorem c39_b_{s} (P{s} Q{s} : Prop) (h{s} : P{s} ∨ Q{s}) : Q{s} ∨ P{s} := by\n  {tac}'
 elif k==1:goal=f'α{s} : Type\nP{s} : α{s} → Prop\na{s} : α{s}\nh{s} : P{s} a{s}\n⊢ ∃ x, P{s} x';tac=f'exact ⟨a{s}, h{s}⟩';src=f'theorem c39_b_{s} (α{s} : Type) (P{s} : α{s} → Prop) (a{s} : α{s}) (h{s} : P{s} a{s}) : ∃ x, P{s} x := by\n  {tac}'
 elif k==2:goal=f'P{s} Q{s} R{s} : Prop\nh{s} : (P{s} ∧ Q{s}) ∧ R{s}\n⊢ P{s} ∧ (Q{s} ∧ R{s})';tac=f'exact ⟨h{s}.1.1, h{s}.1.2, h{s}.2⟩';src=f'theorem c39_b_{s} (P{s} Q{s} R{s} : Prop) (h{s} : (P{s} ∧ Q{s}) ∧ R{s}) : P{s} ∧ (Q{s} ∧ R{s}) := by\n  {tac}'
 elif k==3:goal=f'a{s} b{s} : Nat\n⊢ a{s} * b{s} = b{s} * a{s}';tac=f'exact Nat.mul_comm a{s} b{s}';src=f'theorem c39_b_{s} (a{s} b{s} : Nat) : a{s} * b{s} = b{s} * a{s} := by\n  {tac}'
 elif k==4:goal=f'x{s} y{s} : Int\nh{s} : x{s} = y{s} + 3\n⊢ x{s} - y{s} = 3';tac='omega';src=f'theorem c39_b_{s} (x{s} y{s} : Int) (h{s} : x{s} = y{s} + 3) : x{s} - y{s} = 3 := by\n  omega'
 elif k==5:goal=f'α{s} : Type\nxs{s} : List α{s}\n⊢ xs{s}.reverse.reverse = xs{s}';tac='simp';src=f'theorem c39_b_{s} (α{s} : Type) (xs{s} : List α{s}) : xs{s}.reverse.reverse = xs{s} := by\n  simp'
 elif k==6:goal=f'α{s} β{s} : Type\nf{s} : α{s} → β{s}\nx{s} y{s} : α{s}\nh{s} : x{s} = y{s}\n⊢ f{s} x{s} = f{s} y{s}';tac=f'simpa [h{s}]';src=f'theorem c39_b_{s} (α{s} β{s} : Type) (f{s} : α{s} → β{s}) (x{s} y{s} : α{s}) (h{s} : x{s} = y{s}) : f{s} x{s} = f{s} y{s} := by\n  {tac}'
 else:goal=f'P{s} Q{s} R{s} : Prop\nh1{s} : P{s} → Q{s}\nh2{s} : Q{s} → R{s}\n⊢ P{s} → R{s}';tac=f'intro hp; exact h2{s} (h1{s} hp)';src=f'theorem c39_b_{s} (P{s} Q{s} R{s} : Prop) (h1{s} : P{s} → Q{s}) (h2{s} : Q{s} → R{s}) : P{s} → R{s} := by\n  {tac}'
 return {'case_id':f'C39-{j:05d}','split':split,'goal_state':goal,'canonical_tactic':tac,'standalone_source':src,'positive':True}

def neg(i):
 j=9000+i;s=f'{j:04d}';k=i%5
 if k==0:goal=f'u{s} : Unit\n⊢ False';src=f'theorem c39_n_{s} (u{s} : Unit) : False := by\n  __CANDIDATE_TACTIC__'
 elif k==1:goal=f'n{s} : Nat\n⊢ n{s} + 1 = n{s}';src=f'theorem c39_n_{s} (n{s} : Nat) : n{s} + 1 = n{s} := by\n  __CANDIDATE_TACTIC__'
 elif k==2:goal=f'P{s} : Prop\n⊢ P{s}';src=f'theorem c39_n_{s} (P{s} : Prop) : P{s} := by\n  __CANDIDATE_TACTIC__'
 elif k==3:goal=f'a{s} b{s} : Nat\n⊢ a{s} = b{s}';src=f'theorem c39_n_{s} (a{s} b{s} : Nat) : a{s} = b{s} := by\n  __CANDIDATE_TACTIC__'
 else:goal=f'α{s} : Type\nxs{s} : List α{s}\n⊢ xs{s} = []';src=f'theorem c39_n_{s} (α{s} : Type) (xs{s} : List α{s}) : xs{s} = [] := by\n  __CANDIDATE_TACTIC__'
 return {'case_id':f'C39-{j:05d}','split':'NEGATIVE_UNPROVABLE','goal_state':goal,'canonical_tactic':None,'standalone_source':src,'positive':False}

def suite():
 r=[gen_a(i,'DISCOVERY') for i in range(64)]+[gen_a(i,'CONFIRMATION') for i in range(64,128)]+[gen_b(i,'GENERATOR_HOLDOUT') for i in range(64)]+[gen_b(i,'ALPHA_RENAME_OOD') for i in range(64,112)]
 for i in range(128,176):
  x=gen_a(i,'HYP_ORDER_OOD');ls=x['goal_state'].splitlines();x['goal_state']='\n'.join(list(reversed(ls[:-1]))+[ls[-1]]) if len(ls)>2 else x['goal_state'];r.append(x)
 r += [gen_b(i,'COMPOSITIONAL_OOD') for i in range(112,160)]+[neg(i) for i in range(48)];return r

def render(row,tactic):
 src=row['standalone_source']
 if '__CANDIDATE_TACTIC__' in src:return 'import Mathlib\n\n'+src.replace('__CANDIDATE_TACTIC__',tactic)+'\n'
 prefix=src.rsplit(':= by\n',1)[0];return 'import Mathlib\n\n'+prefix+':= by\n  '+tactic+'\n'

def classify(txt):
 t=txt.lower()
 if 'unknown identifier' in t:return 'UNKNOWN_IDENTIFIER'
 if 'unsolved goals' in t:return 'UNSOLVED_GOALS'
 if 'type mismatch' in t:return 'TYPE_MISMATCH'
 if 'unexpected token' in t or 'parser' in t:return 'SYNTAX'
 if 'declaration has metavariables' in t:return 'METAVARIABLE'
 return 'LEAN_REJECT'

def bootstrap(vals,seed=22542,reps=4000):
 if not vals:return (0,0)
 rng=random.Random(seed);n=len(vals);z=[]
 for _ in range(reps):z.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
 z.sort();return (z[int(.025*reps)],z[min(reps-1,int(.975*reps))])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private',required=True);ap.add_argument('--teacher',required=True);ap.add_argument('--lean-root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 rows={x['case_id']:x for x in suite()}; rec=[json.loads(x) for x in Path(a.private).read_text(encoding='utf-8').splitlines() if x.strip()];results=[];tmp=Path(a.lean_root)/'C39Probe.lean'
 for i,x in enumerate(rec):
  if x['status']!='OK': results.append({**{k:x[k] for k in ('condition','seed','case_id','split','rank','tactic_sha256')},'verified':False,'error_class':x['status'],'lean_log_sha256':None});continue
  text=render(rows[x['case_id']],x['tactic']);tmp.write_text(text,encoding='utf-8');p=subprocess.run(['lake','env','lean',tmp.name],cwd=a.lean_root,text=True,capture_output=True,timeout=45);log=(p.stdout or '')+'\n'+(p.stderr or '');results.append({**{k:x[k] for k in ('condition','seed','case_id','split','rank','tactic_sha256')},'verified':p.returncode==0,'error_class':'PASS' if p.returncode==0 else classify(log),'lean_log_sha256':hashlib.sha256(log.encode()).hexdigest()})
 tmp.unlink(missing_ok=True)
 groups={}
 for x in results:groups.setdefault((x['condition'],x['seed'],x['split'],x['case_id']),[]).append(x)
 case=[]
 for (c,s,sp,cid),v in groups.items():case.append({'condition':c,'seed':s,'split':sp,'case_id':cid,'success':any(y['verified'] for y in v),'success_count':sum(y['verified'] for y in v),'best_rank':min((y['rank'] for y in v if y['verified']),default=None),'forbidden_or_empty_count':sum(y['error_class'] in {'FORBIDDEN','EMPTY','TOO_LONG'} for y in v),'unknown_identifier_count':sum(y['error_class']=='UNKNOWN_IDENTIFIER' for y in v)})
 def rate(cond,seed,sp):
  v=[x for x in case if x['condition']==cond and x['seed']==seed and x['split']==sp];return mean(x['success'] for x in v)
 splits=['CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE'];rates={}
 for c,s in [('BASE',42),('FULL',42),('FULL',224),('FULL',1337),('RANDOM_42',42),('SHUFFLED',42)]:rates[f'{c}:{s}']={sp:rate(c,s,sp) for sp in splits}
 # primary gates use seed42
 def paired(sp):
  b={x['case_id']:x['success'] for x in case if x['condition']=='BASE' and x['seed']==42 and x['split']==sp};f={x['case_id']:x['success'] for x in case if x['condition']=='FULL' and x['seed']==42 and x['split']==sp};return [float(f[k])-float(b[k]) for k in sorted(set(b)&set(f))]
 dc=paired('CONFIRMATION');dg=paired('GENERATOR_HOLDOUT');ci=bootstrap(dc);conf_gain=mean(dc);gen_gain=mean(dg)
 control_split=['CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD'];
 def combined(c,s):
  v=[x for x in case if x['condition']==c and x['seed']==s and x['split'] in control_split];return mean(x['success'] for x in v)
 full=combined('FULL',42);rand=combined('RANDOM_42',42);shuf=combined('SHUFFLED',42);neg_rate=rate('FULL',42,'NEGATIVE_UNPROVABLE')
 reproduction=conf_gain>=.10 and gen_gain>=.10 and ci[0]>0 and full-max(rand,shuf)>=.05 and neg_rate==0
 base_comb=combined('BASE',42);failure=(full-base_comb)<=-.10 and (max(rand,shuf)-full)>=.05
 safety=neg_rate==0 and sum(x['forbidden_or_empty_count'] for x in case if x['condition']=='FULL' and x['seed']==42)==0
 teacher=json.load(open(a.teacher,encoding='utf-8'));tb=teacher['conditions']['BASE']['overall']['mean_logprob'];tf=teacher['conditions']['FULL']['overall']['mean_logprob']
 public_results=[{k:v for k,v in x.items() if k not in ('tactic_sha256',)} for x in results]
 dump(out/'R22542_C39_LEAN_GENERATION_VERIFICATION.json',{'schema':'LUCIA_AA_R22542_C39_LEAN_GENERATION_VERIFICATION_V1','verified_candidate_records':len(results),'case_results':case,'rates':rates,'raw_tactics_exported':False,'lean_runtime':'4.28.0 / Mathlib v4.28.0'})
 dump(out/'R22542_C39_BEHAVIOR.json',{'schema':'LUCIA_AA_R22542_C39_BEHAVIOR_V1','candidate_id':'R22542-C39-OPENPROOF-LEAN-TACTIC-2B','primary_metric':'LEAN_KERNEL_VERIFIED_SUCCESS_AT_8','confirmation_gain':conf_gain,'confirmation_bootstrap95':[ci[0],ci[1]],'generator_holdout_gain':gen_gain,'full_combined_success':full,'base_combined_success':base_comb,'random_combined_success':rand,'shuffled_combined_success':shuf,'full_minus_best_random_shuffle':full-max(rand,shuf),'negative_false_success_rate':neg_rate,'teacher_full_minus_base_mean_logprob':tf-tb,'reproduction_gate':'GREEN' if reproduction else 'RED','failure_gate':'GREEN' if failure else 'RED','safety_gate':'GREEN' if safety else 'RED','causal_entry':bool(reproduction or failure),'tokenizer_provenance_hold':True,'training_time_exact_base_revision_proven':False,'max_claim':'E3_EXPLORATORY_TOKENIZER_PROVENANCE_HOLD','raw_tactics_exported':False})
 # raw-free counterexample ids only
 b={(x['case_id'],x['split']):x['success'] for x in case if x['condition']=='BASE' and x['seed']==42};f={(x['case_id'],x['split']):x['success'] for x in case if x['condition']=='FULL' and x['seed']==42};rows_out=[]
 for k in sorted(set(b)&set(f)):
  if b[k]!=f[k]:rows_out.append({'case_id':k[0],'split':k[1],'kind':'RESCUE' if f[k] else 'HARM','base_success':b[k],'full_success':f[k]})
 (out/'R22542_C39_COUNTEREXAMPLES.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows_out))
if __name__=='__main__':main()
