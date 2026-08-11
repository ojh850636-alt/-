from __future__ import annotations
import hashlib, json, random
from pathlib import Path

SEED = 225420039
SCHEMA = "LUCIA_AA_R22542_C39_FRESH_LEAN_TACTIC_SUITE_V1"

# Two deliberately separate generator implementations. Neither consumes OpenProof rows.
def _gen_a(i: int, split: str) -> dict:
    k=i%8; s=f"{i:03d}"
    fam=[]
    if k==0:
        goal=f"α{s} : Type\nx{s} y{s} : α{s}\nh{s} : x{s} = y{s}\n⊢ y{s} = x{s}"
        src=f"theorem c39_a_{s} (α{s} : Type) (x{s} y{s} : α{s}) (h{s} : x{s} = y{s}) : y{s} = x{s} := by\n  exact h{s}.symm"
        tac=f"exact h{s}.symm"; family="EQ_SYMM"
    elif k==1:
        goal=f"P{s} Q{s} : Prop\nh{s} : P{s} ∧ Q{s}\n⊢ Q{s} ∧ P{s}"
        src=f"theorem c39_a_{s} (P{s} Q{s} : Prop) (h{s} : P{s} ∧ Q{s}) : Q{s} ∧ P{s} := by\n  exact ⟨h{s}.2, h{s}.1⟩"
        tac=f"exact ⟨h{s}.2, h{s}.1⟩"; family="AND_SWAP"
    elif k==2:
        goal=f"α{s} : Type\nP{s} : α{s} → Prop\na{s} : α{s}\nh{s} : ∀ x, P{s} x\n⊢ P{s} a{s}"
        src=f"theorem c39_a_{s} (α{s} : Type) (P{s} : α{s} → Prop) (a{s} : α{s}) (h{s} : ∀ x, P{s} x) : P{s} a{s} := by\n  exact h{s} a{s}"
        tac=f"exact h{s} a{s}"; family="FORALL_APPLY"
    elif k==3:
        goal=f"a{s} b{s} : Nat\n⊢ a{s} + b{s} = b{s} + a{s}"
        src=f"theorem c39_a_{s} (a{s} b{s} : Nat) : a{s} + b{s} = b{s} + a{s} := by\n  omega"
        tac="omega"; family="NAT_ADD_COMM"
    elif k==4:
        goal=f"a{s} b{s} : Nat\n⊢ a{s} ≤ a{s} + b{s}"
        src=f"theorem c39_a_{s} (a{s} b{s} : Nat) : a{s} ≤ a{s} + b{s} := by\n  omega"
        tac="omega"; family="NAT_LE_ADD"
    elif k==5:
        goal=f"x{s} y{s} : Int\n⊢ (x{s} + y{s})^2 = x{s}^2 + 2*x{s}*y{s} + y{s}^2"
        src=f"theorem c39_a_{s} (x{s} y{s} : Int) : (x{s} + y{s})^2 = x{s}^2 + 2*x{s}*y{s} + y{s}^2 := by\n  ring"
        tac="ring"; family="INT_RING_SQUARE"
    elif k==6:
        goal=f"α{s} : Type\nxs{s} ys{s} : List α{s}\n⊢ (xs{s} ++ ys{s}).length = xs{s}.length + ys{s}.length"
        src=f"theorem c39_a_{s} (α{s} : Type) (xs{s} ys{s} : List α{s}) : (xs{s} ++ ys{s}).length = xs{s}.length + ys{s}.length := by\n  simp"
        tac="simp"; family="LIST_LENGTH_APPEND"
    else:
        goal=f"P{s} Q{s} : Prop\nh{s} : P{s} → Q{s}\nhp{s} : P{s}\n⊢ Q{s}"
        src=f"theorem c39_a_{s} (P{s} Q{s} : Prop) (h{s} : P{s} → Q{s}) (hp{s} : P{s}) : Q{s} := by\n  exact h{s} hp{s}"
        tac=f"exact h{s} hp{s}"; family="IMPL_APPLY"
    return _row(i,split,"GENERATOR_A",family,goal,src,tac,True)

def _gen_b(i: int, split: str) -> dict:
    j=1000+i; s=f"{j:04d}"; k=i%8
    if k==0:
        goal=f"P{s} Q{s} : Prop\nh{s} : P{s} ∨ Q{s}\n⊢ Q{s} ∨ P{s}"
        src=f"theorem c39_b_{s} (P{s} Q{s} : Prop) (h{s} : P{s} ∨ Q{s}) : Q{s} ∨ P{s} := by\n  rcases h{s} with h | h <;> simp [h]"
        tac=f"rcases h{s} with h | h <;> simp [h]"; family="OR_SWAP"
    elif k==1:
        goal=f"α{s} : Type\nP{s} : α{s} → Prop\na{s} : α{s}\nh{s} : P{s} a{s}\n⊢ ∃ x, P{s} x"
        src=f"theorem c39_b_{s} (α{s} : Type) (P{s} : α{s} → Prop) (a{s} : α{s}) (h{s} : P{s} a{s}) : ∃ x, P{s} x := by\n  exact ⟨a{s}, h{s}⟩"
        tac=f"exact ⟨a{s}, h{s}⟩"; family="EXISTS_WITNESS"
    elif k==2:
        goal=f"P{s} Q{s} R{s} : Prop\nh{s} : (P{s} ∧ Q{s}) ∧ R{s}\n⊢ P{s} ∧ (Q{s} ∧ R{s})"
        src=f"theorem c39_b_{s} (P{s} Q{s} R{s} : Prop) (h{s} : (P{s} ∧ Q{s}) ∧ R{s}) : P{s} ∧ (Q{s} ∧ R{s}) := by\n  exact ⟨h{s}.1.1, h{s}.1.2, h{s}.2⟩"
        tac=f"exact ⟨h{s}.1.1, h{s}.1.2, h{s}.2⟩"; family="AND_ASSOC"
    elif k==3:
        goal=f"a{s} b{s} : Nat\n⊢ a{s} * b{s} = b{s} * a{s}"
        src=f"theorem c39_b_{s} (a{s} b{s} : Nat) : a{s} * b{s} = b{s} * a{s} := by\n  exact Nat.mul_comm a{s} b{s}"
        tac=f"exact Nat.mul_comm a{s} b{s}"; family="NAT_MUL_COMM"
    elif k==4:
        goal=f"x{s} y{s} : Int\nh{s} : x{s} = y{s} + 3\n⊢ x{s} - y{s} = 3"
        src=f"theorem c39_b_{s} (x{s} y{s} : Int) (h{s} : x{s} = y{s} + 3) : x{s} - y{s} = 3 := by\n  omega"
        tac="omega"; family="INT_LINEAR"
    elif k==5:
        goal=f"α{s} : Type\nxs{s} : List α{s}\n⊢ xs{s}.reverse.reverse = xs{s}"
        src=f"theorem c39_b_{s} (α{s} : Type) (xs{s} : List α{s}) : xs{s}.reverse.reverse = xs{s} := by\n  simp"
        tac="simp"; family="LIST_REVERSE_INV"
    elif k==6:
        goal=f"α{s} β{s} : Type\nf{s} : α{s} → β{s}\nx{s} y{s} : α{s}\nh{s} : x{s} = y{s}\n⊢ f{s} x{s} = f{s} y{s}"
        src=f"theorem c39_b_{s} (α{s} β{s} : Type) (f{s} : α{s} → β{s}) (x{s} y{s} : α{s}) (h{s} : x{s} = y{s}) : f{s} x{s} = f{s} y{s} := by\n  simpa [h{s}]"
        tac=f"simpa [h{s}]"; family="CONGR_ARG"
    else:
        goal=f"P{s} Q{s} R{s} : Prop\nh1{s} : P{s} → Q{s}\nh2{s} : Q{s} → R{s}\n⊢ P{s} → R{s}"
        src=f"theorem c39_b_{s} (P{s} Q{s} R{s} : Prop) (h1{s} : P{s} → Q{s}) (h2{s} : Q{s} → R{s}) : P{s} → R{s} := by\n  intro hp; exact h2{s} (h1{s} hp)"
        tac=f"intro hp; exact h2{s} (h1{s} hp)"; family="IMPL_COMPOSE"
    return _row(j,split,"GENERATOR_B",family,goal,src,tac,True)

def _neg(i: int) -> dict:
    j=9000+i; s=f"{j:04d}"; k=i%5
    if k==0:
        goal=f"u{s} : Unit\n⊢ False"; src=f"theorem c39_n_{s} (u{s} : Unit) : False := by\n  __CANDIDATE_TACTIC__"; fam="NEG_FALSE"
    elif k==1:
        goal=f"n{s} : Nat\n⊢ n{s} + 1 = n{s}"; src=f"theorem c39_n_{s} (n{s} : Nat) : n{s} + 1 = n{s} := by\n  __CANDIDATE_TACTIC__"; fam="NEG_NAT_SUCC_EQ"
    elif k==2:
        goal=f"P{s} : Prop\n⊢ P{s}"; src=f"theorem c39_n_{s} (P{s} : Prop) : P{s} := by\n  __CANDIDATE_TACTIC__"; fam="NEG_ARBITRARY_PROP"
    elif k==3:
        goal=f"a{s} b{s} : Nat\n⊢ a{s} = b{s}"; src=f"theorem c39_n_{s} (a{s} b{s} : Nat) : a{s} = b{s} := by\n  __CANDIDATE_TACTIC__"; fam="NEG_ARBITRARY_EQ"
    else:
        goal=f"α{s} : Type\nxs{s} : List α{s}\n⊢ xs{s} = []"; src=f"theorem c39_n_{s} (α{s} : Type) (xs{s} : List α{s}) : xs{s} = [] := by\n  __CANDIDATE_TACTIC__"; fam="NEG_ARBITRARY_LIST"
    return _row(j,"NEGATIVE_UNPROVABLE","GENERATOR_NEG",fam,goal,src,None,False)

def _row(i,split,generator,family,goal,source,tactic,positive):
    return {"case_id":f"C39-{i:05d}","split":split,"generator":generator,"family":family,"goal_state":goal,"prompt":goal+":::","standalone_source":source,"canonical_tactic":tactic,"positive":bool(positive),"model_output_observed":False,"training_dataset_row_used":False}

def build() -> list[dict]:
    rows=[]
    for i in range(64): rows.append(_gen_a(i,"DISCOVERY"))
    for i in range(64,128): rows.append(_gen_a(i,"CONFIRMATION"))
    for i in range(64): rows.append(_gen_b(i,"GENERATOR_HOLDOUT"))
    for i in range(64,112): rows.append(_gen_b(i,"ALPHA_RENAME_OOD"))
    for i in range(128,176):
        r=_gen_a(i,"HYP_ORDER_OOD"); lines=r["goal_state"].splitlines(); ctx=lines[:-1]; target=lines[-1]
        if len(ctx)>1: ctx=list(reversed(ctx))
        r["goal_state"]="\n".join(ctx+[target]); r["prompt"]=r["goal_state"]+":::"; rows.append(r)
    for i in range(112,160): rows.append(_gen_b(i,"COMPOSITIONAL_OOD"))
    for i in range(48): rows.append(_neg(i))
    assert len(rows)==384
    ids=[r['case_id'] for r in rows]; assert len(ids)==len(set(ids))
    prompts=[r['prompt'] for r in rows]; assert len(prompts)==len(set(prompts))
    return rows

def write_jsonl(path: str|Path) -> dict:
    rows=build(); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    split_counts={s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})}
    return {"schema":SCHEMA,"cases":len(rows),"positive_cases":sum(r['positive'] for r in rows),"negative_cases":sum(not r['positive'] for r in rows),"split_counts":split_counts,"generators":sorted({r['generator'] for r in rows}),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"model_outputs_observed":0,"training_dataset_rows_used":0}

if __name__=='__main__':
    import sys; print(json.dumps(write_jsonl(sys.argv[1]),indent=2,sort_keys=True))
