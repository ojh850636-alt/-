#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 2251
CANDIDATE_ID = "R225-C4-QWEN05-TEXT2SQL"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def result_digest(rows: list[tuple[Any, ...]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    return sha256_bytes(payload)


def sqlite_result(setup_sql: list[str], query: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(":memory:")
    try:
        for stmt in setup_sql:
            conn.execute(stmt)
        conn.commit()
        q = query.strip().rstrip(";").strip()
        if not re.match(r"(?is)^(select|with)\b", q):
            raise ValueError("only SELECT/WITH is allowed")
        if ";" in q:
            raise ValueError("multiple SQL statements are forbidden")
        return conn.execute(q).fetchall()
    finally:
        conn.close()


def mk_rows(rng: random.Random, n: int, cats: list[str]) -> list[tuple[int, str, int, str]]:
    rows = []
    for i in range(1, n + 1):
        cat = cats[(i + rng.randrange(len(cats))) % len(cats)]
        value = 5 + ((i * 17 + rng.randrange(31)) % 91)
        name = f"item_{rng.randrange(1000, 9999)}_{i}"
        rows.append((i, cat, value, name))
    return rows


def sql_quote(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def make_case(split: str, family: str, idx: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    suffix = f"{split.lower()}_{idx}_{rng.randrange(10000,99999)}"
    table = f"records_{suffix}"
    cats = [f"group_{rng.randrange(10,99)}", f"group_{rng.randrange(100,199)}", f"group_{rng.randrange(200,299)}"]
    rows = mk_rows(rng, 9 + idx % 4, cats)
    setup = [f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER, name TEXT)"]
    setup += [f"INSERT INTO {table} VALUES ({r[0]}, {sql_quote(r[1])}, {r[2]}, {sql_quote(r[3])})" for r in rows]
    schema = f"{table}(id INTEGER PRIMARY KEY, category TEXT, amount INTEGER, name TEXT)"
    db = f"db_{suffix}"

    if family == "COUNT_ALL":
        question = f"How many rows are stored in {table}?"
        ref = f"SELECT COUNT(*) FROM {table}"
        bad = f"SELECT COUNT(*) FROM {table} WHERE amount > 1000"
    elif family == "FILTER_EQ":
        target, wrong = cats[0], cats[1]
        question = f"List the names in {table} whose category is {target}."
        ref = f"SELECT name FROM {table} WHERE category = {sql_quote(target)} ORDER BY name"
        bad = f"SELECT name FROM {table} WHERE category = {sql_quote(wrong)} ORDER BY name"
    elif family == "FILTER_GT":
        threshold = sorted(r[2] for r in rows)[len(rows)//2]
        question = f"Return the ids in {table} with amount strictly greater than {threshold}."
        ref = f"SELECT id FROM {table} WHERE amount > {threshold} ORDER BY id"
        bad = f"SELECT id FROM {table} WHERE amount >= {threshold} ORDER BY id"
    elif family == "ORDER_LIMIT":
        question = f"Which name in {table} has the largest amount?"
        ref = f"SELECT name FROM {table} ORDER BY amount DESC, id ASC LIMIT 1"
        bad = f"SELECT name FROM {table} ORDER BY amount ASC, id ASC LIMIT 1"
    elif family == "AVERAGE":
        question = f"What is the average amount in {table}?"
        ref = f"SELECT AVG(amount) FROM {table}"
        bad = f"SELECT SUM(amount) FROM {table}"
    elif family == "GROUP_COUNT":
        question = f"For each category in {table}, return the category and number of rows."
        ref = f"SELECT category, COUNT(*) FROM {table} GROUP BY category ORDER BY category"
        bad = f"SELECT category, SUM(amount) FROM {table} GROUP BY category ORDER BY category"
    elif family == "MINIMUM":
        question = f"Return the smallest amount stored in {table}."
        ref = f"SELECT MIN(amount) FROM {table}"
        bad = f"SELECT MAX(amount) FROM {table}"
    elif family == "DISTINCT":
        question = f"Return every distinct category in {table} in alphabetical order."
        ref = f"SELECT DISTINCT category FROM {table} ORDER BY category"
        bad = f"SELECT category FROM {table} ORDER BY category"
    elif family == "JOIN_LOOKUP":
        parent = f"owners_{suffix}"
        child = f"orders_{suffix}"
        owners = [(i, f"owner_{rng.randrange(1000,9999)}_{i}") for i in range(1, 6)]
        orders = [(i, 1 + (i % 5), 10 + ((i * 23 + rng.randrange(17)) % 120)) for i in range(1, 13)]
        threshold = sorted(x[2] for x in orders)[7]
        setup = [f"CREATE TABLE {parent} (owner_id INTEGER PRIMARY KEY, owner_name TEXT)", f"CREATE TABLE {child} (order_id INTEGER PRIMARY KEY, owner_id INTEGER, total INTEGER)"]
        setup += [f"INSERT INTO {parent} VALUES ({a},{sql_quote(b)})" for a,b in owners]
        setup += [f"INSERT INTO {child} VALUES ({a},{b},{c})" for a,b,c in orders]
        schema = f"{parent}(owner_id INTEGER PRIMARY KEY, owner_name TEXT); {child}(order_id INTEGER PRIMARY KEY, owner_id INTEGER, total INTEGER)"
        question = f"List owner names that have an order total greater than {threshold}."
        ref = f"SELECT DISTINCT p.owner_name FROM {parent} p JOIN {child} c ON p.owner_id = c.owner_id WHERE c.total > {threshold} ORDER BY p.owner_name"
        bad = f"SELECT DISTINCT p.owner_name FROM {parent} p JOIN {child} c ON p.owner_id = c.order_id WHERE c.total > {threshold} ORDER BY p.owner_name"
    elif family == "JOIN_SUM":
        parent = f"clients_{suffix}"
        child = f"payments_{suffix}"
        owners = [(i, f"client_{rng.randrange(1000,9999)}_{i}") for i in range(1, 6)]
        orders = [(i, 1 + (i % 5), 10 + ((i * 19 + rng.randrange(23)) % 100)) for i in range(1, 16)]
        setup = [f"CREATE TABLE {parent} (client_id INTEGER PRIMARY KEY, client_name TEXT)", f"CREATE TABLE {child} (payment_id INTEGER PRIMARY KEY, client_id INTEGER, amount INTEGER)"]
        setup += [f"INSERT INTO {parent} VALUES ({a},{sql_quote(b)})" for a,b in owners]
        setup += [f"INSERT INTO {child} VALUES ({a},{b},{c})" for a,b,c in orders]
        schema = f"{parent}(client_id INTEGER PRIMARY KEY, client_name TEXT); {child}(payment_id INTEGER PRIMARY KEY, client_id INTEGER, amount INTEGER)"
        question = "For each client, return the client name and total payment amount."
        ref = f"SELECT p.client_name, SUM(c.amount) FROM {parent} p JOIN {child} c ON p.client_id = c.client_id GROUP BY p.client_id, p.client_name ORDER BY p.client_name"
        bad = f"SELECT p.client_name, AVG(c.amount) FROM {parent} p JOIN {child} c ON p.client_id = c.client_id GROUP BY p.client_id, p.client_name ORDER BY p.client_name"
    else:
        raise ValueError(family)

    expected = sqlite_result(setup, ref)
    wrong = sqlite_result(setup, bad)
    if expected == wrong:
        raise AssertionError(f"distractor collision: {family} {split} {idx}")
    prompt = (
        "Convert the following natural language question to SQL. Return only one read-only SQL statement.\n\n"
        f"Database: {db}\nSchema:\n{schema}\nQuestion: {question}\n\nSQL:\n"
    )
    return {
        "case_id": f"C4-{split}-{family}-{idx:04d}", "split": split, "family": family,
        "database": db, "schema": schema, "question": question, "prompt": prompt,
        "reference_sql": ref, "distractor_sql": bad, "setup_sql": setup,
        "expected_result_sha256": result_digest(expected), "distractor_result_sha256": result_digest(wrong),
    }


def build_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = ["COUNT_ALL","FILTER_EQ","FILTER_GT","ORDER_LIMIT","AVERAGE","GROUP_COUNT","MINIMUM","DISTINCT","JOIN_LOOKUP","JOIN_SUM"]
    plan = [("DISCOVERY", 32), ("CONFIRMATION", 20), ("OOD", 16)]
    cases=[]
    n=0
    for split,count in plan:
        for family in families:
            for i in range(count):
                cases.append(make_case(split, family, i, SEED + n * 9973 + len(family)))
                n += 1
    negative_prompts = [
        "Write a friendly greeting for a new teammate.", "Explain why rainbows appear.",
        "Summarize the benefits of regular sleep.", "Give three ideas for a weekend walk.",
        "Translate 'good morning' into Spanish.", "Describe how photosynthesis works.",
        "Write a two-line poem about the moon.", "What should someone pack for a short hike?",
        "Explain the difference between weather and climate.", "Give a safe recipe for fruit salad.",
    ]
    negatives=[]
    for i in range(40):
        text=negative_prompts[i % len(negative_prompts)] + f" Variation {i}."
        negatives.append({"case_id":f"C4-NEGATIVE-{i:04d}","split":"NEGATIVE","prompt":text,"sql_completion":"SELECT * FROM users;","natural_completion":"I cannot produce SQL because no database question or schema was provided."})
    return cases, negatives


def exact_one_sided_p(rescue: int, harm: int) -> float:
    n=rescue+harm
    if n == 0: return 1.0
    return sum(math.comb(n,k) for k in range(rescue,n+1)) / (2**n)


def bootstrap_delta(base: list[bool], full: list[bool], seed: int=SEED, rounds: int=2000) -> tuple[float,float]:
    rng=random.Random(seed); n=len(base); vals=[]
    for _ in range(rounds):
        idx=[rng.randrange(n) for _ in range(n)]
        vals.append(sum(full[i]-base[i] for i in idx)/n)
    vals.sort()
    return vals[int(.025*rounds)], vals[int(.975*rounds)-1]


def adapter_layers(model: PeftModel) -> list[LoraLayer]:
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def adapter_name(layer: LoraLayer) -> str:
    keys=list(layer.lora_A.keys())
    if not keys: raise RuntimeError("LoRA layer has no adapters")
    return keys[0]


def save_lora_state(model: PeftModel) -> dict[str, torch.Tensor]:
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if "lora_A" in n or "lora_B" in n}


def restore_lora_state(model: PeftModel, state: dict[str, torch.Tensor]) -> None:
    named=dict(model.named_parameters())
    with torch.no_grad():
        for n,t in state.items(): named[n].copy_(t.to(named[n].device, dtype=named[n].dtype))


def set_scale(model: PeftModel, factor: float, originals: dict[int,float]) -> None:
    for layer in adapter_layers(model):
        name=adapter_name(layer)
        layer.scaling[name]=originals[id(layer)] * factor


def shuffled_state(model: PeftModel, seed: int) -> None:
    rng=torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for layer in adapter_layers(model):
            name=adapter_name(layer); b=layer.lora_B[name].weight
            perm=torch.randperm(b.shape[1], generator=rng)
            b.copy_(b[:,perm].clone())


def random_state(model: PeftModel, seed: int) -> None:
    gen=torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for layer in adapter_layers(model):
            name=adapter_name(layer)
            for p in (layer.lora_A[name].weight, layer.lora_B[name].weight):
                target=float(torch.linalg.vector_norm(p.float()).item())
                r=torch.randn(p.shape, generator=gen, dtype=torch.float32)
                r *= target / max(float(torch.linalg.vector_norm(r).item()), 1e-12)
                p.copy_(r.to(p.dtype))


def score_texts(model: PeftModel, tok, prompts: list[str], completions: list[str], batch_size: int=8) -> list[dict[str,float]]:
    assert len(prompts)==len(completions)
    out=[]
    for start in range(0,len(prompts),batch_size):
        ps=prompts[start:start+batch_size]; cs=completions[start:start+batch_size]
        full=[]; comp_lens=[]
        for p,c in zip(ps,cs):
            p_ids=tok(p, add_special_tokens=True)["input_ids"]
            f_ids=tok(p+c, add_special_tokens=True)["input_ids"]
            if f_ids[:len(p_ids)] != p_ids:
                raise RuntimeError("prompt is not a token prefix of prompt+completion")
            full.append(p+c); comp_lens.append(len(f_ids)-len(p_ids))
        enc=tok(full, return_tensors="pt", padding=True, add_special_tokens=True)
        keep=max(comp_lens)+1
        with torch.inference_mode():
            outputs=model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False, logits_to_keep=keep)
        logits=outputs.logits
        if logits.shape[1] != keep:
            raise RuntimeError(f"logits_to_keep was not honored: {logits.shape} keep={keep}")
        for i,c in enumerate(comp_lens):
            begin=keep-c-1
            selected=logits[i,begin:begin+c].float()
            labels=enc.input_ids[i,-c:]
            token_lp=selected.gather(-1, labels[:,None]).squeeze(-1) - torch.logsumexp(selected,dim=-1)
            out.append({"sum_logprob":float(token_lp.sum().item()),"mean_logprob":float(token_lp.mean().item()),"token_count":int(c)})
        del enc, outputs, logits
    return out


def score_pairs(model: PeftModel, tok, cases: list[dict[str,Any]], batch_size: int=8) -> list[dict[str,Any]]:
    prompts=[]; comps=[]
    for c in cases:
        prompts.extend([c["prompt"],c["prompt"]]); comps.extend([c["reference_sql"]+";",c["distractor_sql"]+";"])
    scores=score_texts(model,tok,prompts,comps,batch_size)
    rows=[]
    for i,c in enumerate(cases):
        ref=scores[2*i]; bad=scores[2*i+1]; margin=ref["mean_logprob"]-bad["mean_logprob"]
        rows.append({"case_id":c["case_id"],"split":c["split"],"family":c["family"],"reference_mean_logprob":ref["mean_logprob"],"distractor_mean_logprob":bad["mean_logprob"],"margin":margin,"reference_preferred":margin>0})
    return rows


def aggregate(rows: list[dict[str,Any]]) -> dict[str,Any]:
    by=defaultdict(list)
    for r in rows: by[(r["split"],r["family"])].append(r)
    def agg(xs):
        return {"n":len(xs),"preference_accuracy":sum(x["reference_preferred"] for x in xs)/len(xs),"mean_margin":statistics.fmean(x["margin"] for x in xs)}
    return {"overall":agg(rows),"by_split_family":{"|".join(k):agg(v) for k,v in sorted(by.items())}}


def extract_sql(text: str) -> str | None:
    text=text.replace("```sql","").replace("```","").strip()
    match=re.search(r"(?is)\b(select|with)\b",text)
    if not match: return None
    q=text[match.start():].strip()
    q=q.split("\n\n",1)[0].strip()
    q=q.split(";",1)[0].strip()
    return q or None


def generation_eval(model: PeftModel,tok,cases:list[dict[str,Any]],negatives:list[dict[str,Any]],condition:str) -> dict[str,Any]:
    rows=[]
    for case in cases:
        enc=tok(case["prompt"],return_tensors="pt")
        with torch.inference_mode():
            ids=model.generate(**enc,max_new_tokens=64,do_sample=False,pad_token_id=tok.eos_token_id)
        continuation=tok.decode(ids[0,enc.input_ids.shape[1]:],skip_special_tokens=True)
        sql=extract_sql(continuation); ok=False; reason=None
        if sql:
            try: ok=result_digest(sqlite_result(case["setup_sql"],sql))==case["expected_result_sha256"]
            except Exception as e: reason=type(e).__name__
        rows.append({"case_id":case["case_id"],"condition":condition,"generated_text":continuation[:500],"extracted_sql":sql,"execution_exact":ok,"failure_reason":reason})
    neg_rows=[]
    for case in negatives:
        enc=tok(case["prompt"],return_tensors="pt")
        with torch.inference_mode(): ids=model.generate(**enc,max_new_tokens=40,do_sample=False,pad_token_id=tok.eos_token_id)
        continuation=tok.decode(ids[0,enc.input_ids.shape[1]:],skip_special_tokens=True)
        neg_rows.append({"case_id":case["case_id"],"condition":condition,"generated_text":continuation[:500],"sql_activated":extract_sql(continuation) is not None})
    return {"positive":rows,"negative":neg_rows,"execution_accuracy":sum(x["execution_exact"] for x in rows)/len(rows),"negative_sql_activation":sum(x["sql_activated"] for x in neg_rows)/len(neg_rows)}


def static_report(adapter_path: Path) -> dict[str,Any]:
    module_rows=[]; totals=defaultdict(float); total_params=0
    with safe_open(str(adapter_path),framework="pt",device="cpu") as f:
        keys=list(f.keys()); tensors={k:f.get_tensor(k) for k in keys}
    pairs={}
    for k,t in tensors.items():
        total_params += t.numel()
        if ".lora_A." in k: pairs.setdefault(k.replace(".lora_A.",".PAIR."),{})["A"]=(k,t)
        elif ".lora_B." in k: pairs.setdefault(k.replace(".lora_B.",".PAIR."),{})["B"]=(k,t)
    for pair,ab in sorted(pairs.items()):
        if set(ab)!={"A","B"}: continue
        a=ab["A"][1].float(); b=ab["B"][1].float()
        gram_a=a@a.T; gram_b=b.T@b
        energy=float(torch.sum(gram_a*gram_b).item())
        name=pair.split(".PAIR.")[0]
        m=re.search(r"layers\.(\d+)\.",name); layer=int(m.group(1)) if m else None
        module=name.split(".")[-1]
        totals[module]+=energy
        module_rows.append({"module_id":name,"layer":layer,"module_type":module,"rank":int(a.shape[0]),"a_shape":list(a.shape),"b_shape":list(b.shape),"parameter_count":a.numel()+b.numel(),"delta_frobenius_sq":energy})
    total_energy=sum(totals.values())
    for r in module_rows: r["energy_ratio"]=r["delta_frobenius_sq"]/total_energy if total_energy else 0.0
    return {"schema":"LUCIA_AA_C4_STATIC_NONRECONSTRUCTIVE_V1","tensor_count":len(tensors),"total_parameter_count":total_params,"lora_pair_count":len(module_rows),"module_type_energy":{k:{"energy":v,"ratio":v/total_energy if total_energy else 0.0} for k,v in sorted(totals.items())},"modules":module_rows,"raw_tensor_values_included":False,"reconstructable_delta_w_included":False}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--base-dir",type=Path,required=True); ap.add_argument("--adapter-dir",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); args=ap.parse_args()
    out=args.out_dir; out.mkdir(parents=True,exist_ok=True)
    random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
    cases,negatives=build_dataset()
    write_jsonl(out/"C4_SQL_CASES.jsonl",cases); write_jsonl(out/"C4_NEGATIVE_CASES.jsonl",negatives)
    manifest={"schema":"LUCIA_AA_C4_SQL_DATASET_MANIFEST_V1","candidate_id":CANDIDATE_ID,"positive_count":len(cases),"negative_count":len(negatives),"split_counts":dict(Counter(c["split"] for c in cases)),"family_counts":dict(Counter(c["family"] for c in cases)),"cases_sha256":sha256_file(out/"C4_SQL_CASES.jsonl"),"negative_sha256":sha256_file(out/"C4_NEGATIVE_CASES.jsonl"),"public_benchmark_rows_copied":0,"verifier":"SQLite execution result equivalence"}
    write_json(out/"C4_SQL_DATASET_MANIFEST.json",manifest)

    tok=AutoTokenizer.from_pretrained(args.base_dir,local_files_only=True,trust_remote_code=False)
    tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.base_dir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,args.adapter_dir,local_files_only=True)
    model.eval(); model.config.use_cache=False
    originals=save_lora_state(model); original_scales={id(l):float(l.scaling[adapter_name(l)]) for l in adapter_layers(model)}
    write_json(out/"C4_RUNTIME_RECEIPT.json",{"schema":"LUCIA_AA_C4_RUNTIME_RECEIPT_V1","torch":torch.__version__,"transformers":__import__("transformers").__version__,"peft":__import__("peft").__version__,"dtype":str(next(model.parameters()).dtype),"lora_layer_count":len(adapter_layers(model)),"trust_remote_code":False,"device":"cpu"})

    prereg=[c for c in cases if c["split"]=="CONFIRMATION"][:80]+[c for c in cases if c["split"]=="OOD"][:64]
    condition_rows={}
    def run_condition(name:str, subset:list[dict[str,Any]], base_disabled=False, scale=1.0, transform=None):
        restore_lora_state(model,originals); set_scale(model,scale,original_scales)
        if transform: transform(model)
        started=time.time()
        if base_disabled:
            with model.disable_adapter(): rows=score_pairs(model,tok,subset)
        else: rows=score_pairs(model,tok,subset)
        condition_rows[name]=rows
        write_jsonl(out/f"C4_TF_{name}.jsonl",rows)
        write_json(out/f"C4_TF_{name}_SUMMARY.json",{"condition":name,"elapsed_seconds":time.time()-started,**aggregate(rows)})
        return rows

    base_rows=run_condition("BASE",cases,base_disabled=True)
    full_rows=run_condition("FULL",cases)
    shuffle_rows=run_condition("SHUFFLED",prereg,transform=lambda m:shuffled_state(m,SEED+11))
    random_rows=run_condition("RANDOM_RANK_MATCHED",prereg,transform=lambda m:random_state(m,SEED+29))
    dose_rows={}
    for factor in (0.25,0.5,1.5): dose_rows[str(factor)]=run_condition(f"DOSE_{factor}",prereg,scale=factor)
    restore_lora_state(model,originals); set_scale(model,1.0,original_scales)

    # Non-SQL bias and general-language collateral.
    neg_prompts=[]; neg_comps=[]
    for n in negatives:
        neg_prompts.extend([n["prompt"],n["prompt"]]); neg_comps.extend([n["sql_completion"],n["natural_completion"]])
    with model.disable_adapter(): neg_base=score_texts(model,tok,neg_prompts,neg_comps)
    neg_full=score_texts(model,tok,neg_prompts,neg_comps)
    neg_rows=[]
    for i,n in enumerate(negatives):
        bm=neg_base[2*i]["mean_logprob"]-neg_base[2*i+1]["mean_logprob"]
        fm=neg_full[2*i]["mean_logprob"]-neg_full[2*i+1]["mean_logprob"]
        neg_rows.append({"case_id":n["case_id"],"base_sql_bias_margin":bm,"full_sql_bias_margin":fm,"increase":fm-bm})
    write_jsonl(out/"C4_NEGATIVE_SQL_BIAS.jsonl",neg_rows)

    collateral_prompts=[
        "The sun rises in the", "Water freezes at", "A healthy routine often includes", "The capital of France is",
        "Plants use sunlight to", "A triangle has three", "Good software tests help", "A careful explanation should",
        "The opposite of hot is", "A library is a place with",
    ]
    collateral=[]
    for i in range(60):
        p=collateral_prompts[i%len(collateral_prompts)] + f" example {i}: "
        true=["east.","zero degrees Celsius.","regular sleep and exercise.","Paris.","make energy through photosynthesis.","sides.","catch regressions.","state assumptions clearly.","cold.","books and readers."][i%10]
        false=["database table.","SQL query.","random schema.","SELECT statement.","foreign key.","join clause.","aggregation.","drop table.","integer column.","query planner."][i%10]
        collateral.append((p,true,false))
    cp=[]; cc=[]
    for p,t,f in collateral: cp.extend([p,p]); cc.extend([t,f])
    with model.disable_adapter(): cb=score_texts(model,tok,cp,cc)
    cf=score_texts(model,tok,cp,cc)
    col_rows=[]
    for i,(p,t,f) in enumerate(collateral):
        b=cb[2*i]["mean_logprob"]-cb[2*i+1]["mean_logprob"]
        x=cf[2*i]["mean_logprob"]-cf[2*i+1]["mean_logprob"]
        col_rows.append({"case_id":f"C4-COLLATERAL-{i:04d}","base_margin":b,"full_margin":x,"base_preferred":b>0,"full_preferred":x>0,"margin_delta":x-b})
    write_jsonl(out/"C4_LANGUAGE_COLLATERAL.jsonl",col_rows)

    # Representative free generation: 8 positive and 4 negative prompts per condition.
    reps=[c for c in cases if c["split"]=="CONFIRMATION"][:4]+[c for c in cases if c["split"]=="OOD"][:4]
    neg_reps=negatives[:4]
    with model.disable_adapter(): gen_base=generation_eval(model,tok,reps,neg_reps,"BASE")
    gen_full=generation_eval(model,tok,reps,neg_reps,"FULL")
    write_json(out/"C4_FREE_GENERATION_BASE.json",gen_base); write_json(out/"C4_FREE_GENERATION_FULL.json",gen_full)

    # Paired evidence and gate.
    base_map={r["case_id"]:r for r in base_rows}; full_map={r["case_id"]:r for r in full_rows}
    paired=[]
    for c in cases:
        b=base_map[c["case_id"]]; f=full_map[c["case_id"]]
        paired.append({"case_id":c["case_id"],"split":c["split"],"family":c["family"],"base_preferred":b["reference_preferred"],"full_preferred":f["reference_preferred"],"base_margin":b["margin"],"full_margin":f["margin"],"margin_delta":f["margin"]-b["margin"]})
    write_jsonl(out/"C4_BEHAVIOR_ATLAS.jsonl",paired)
    split_stats={}
    for split in ("DISCOVERY","CONFIRMATION","OOD"):
        xs=[x for x in paired if x["split"]==split]
        rescue=sum((not x["base_preferred"]) and x["full_preferred"] for x in xs)
        harm=sum(x["base_preferred"] and (not x["full_preferred"]) for x in xs)
        bacc=sum(x["base_preferred"] for x in xs)/len(xs); facc=sum(x["full_preferred"] for x in xs)/len(xs)
        lo,hi=bootstrap_delta([x["base_preferred"] for x in xs],[x["full_preferred"] for x in xs],SEED+len(split))
        split_stats[split]={"n":len(xs),"base_preference_accuracy":bacc,"full_preference_accuracy":facc,"delta":facc-bacc,"rescue":rescue,"harm":harm,"exact_one_sided_p":exact_one_sided_p(rescue,harm),"bootstrap_95":[lo,hi],"mean_margin_delta":statistics.fmean(x["margin_delta"] for x in xs)}

    pre_ids={c["case_id"] for c in prereg}
    def pref(rows): return sum(r["reference_preferred"] for r in rows)/len(rows)
    base_pre=[r for r in base_rows if r["case_id"] in pre_ids]; full_pre=[r for r in full_rows if r["case_id"] in pre_ids]
    control={"BASE":pref(base_pre),"FULL":pref(full_pre),"SHUFFLED":pref(shuffle_rows),"RANDOM_RANK_MATCHED":pref(random_rows),**{f"DOSE_{k}":pref(v) for k,v in dose_rows.items()}}
    dose_sequence=[control["BASE"],control["DOSE_0.25"],control["DOSE_0.5"],control["FULL"],control["DOSE_1.5"]]
    dose_directional=all(dose_sequence[i+1]+0.02>=dose_sequence[i] for i in range(3)) and dose_sequence[4]+0.03>=dose_sequence[3]
    collateral_base=sum(x["base_preferred"] for x in col_rows)/len(col_rows); collateral_full=sum(x["full_preferred"] for x in col_rows)/len(col_rows)
    negative_increase=statistics.fmean(x["increase"] for x in neg_rows)
    failures=[]
    if split_stats["CONFIRMATION"]["delta"] < .05 or split_stats["CONFIRMATION"]["bootstrap_95"][0] <= 0: failures.append("CONFIRMATION_DELTA_NOT_SUPPORTED")
    if split_stats["OOD"]["delta"] < .02: failures.append("OOD_DELTA_TOO_SMALL")
    if control["FULL"]-control["SHUFFLED"] < .05: failures.append("SHUFFLED_CONTROL_NOT_BEATEN")
    if control["FULL"]-control["RANDOM_RANK_MATCHED"] < .05: failures.append("RANDOM_CONTROL_NOT_BEATEN")
    if not dose_directional: failures.append("DOSE_RESPONSE_NOT_DIRECTIONAL")
    if collateral_full-collateral_base < -.02: failures.append("LANGUAGE_COLLATERAL_REGRESSION")
    if negative_increase > .50: failures.append("NON_SQL_SQL_BIAS_INCREASE")
    if gen_full["execution_accuracy"]-gen_base["execution_accuracy"] < .10: failures.append("FREE_GENERATION_EXECUTION_DELTA_TOO_SMALL")
    if gen_full["negative_sql_activation"]-gen_base["negative_sql_activation"] > .10: failures.append("FREE_GENERATION_NON_SQL_ACTIVATION")
    gate={"schema":"LUCIA_AA_C4_BEHAVIOR_GATE_V1","candidate_id":CANDIDATE_ID,"split_statistics":split_stats,"controls":control,"dose_sequence":dose_sequence,"dose_directional":dose_directional,"collateral":{"base_preference_accuracy":collateral_base,"full_preference_accuracy":collateral_full,"delta":collateral_full-collateral_base},"negative_sql_bias":{"mean_increase":negative_increase},"free_generation":{"base_execution_accuracy":gen_base["execution_accuracy"],"full_execution_accuracy":gen_full["execution_accuracy"],"base_negative_sql_activation":gen_base["negative_sql_activation"],"full_negative_sql_activation":gen_full["negative_sql_activation"]},"failures":failures,"verdict":"GREEN_BEHAVIOR_GATE" if not failures else "RED_BEHAVIOR_GATE","causal_entry_allowed":not failures}
    write_json(out/"C4_BEHAVIOR_GATE.json",gate)
    write_json(out/"C4_STATIC_DELTA_REPORT.json",static_report(args.adapter_dir/"adapter_model.safetensors"))

    counterexamples=[x for x in paired if x["full_preferred"] is False or x["margin_delta"]<0]
    counterexamples += [{"case_id":x["case_id"],"type":"NON_SQL_BIAS","bias_increase":x["increase"]} for x in neg_rows if x["increase"]>0]
    write_jsonl(out/"C4_COUNTEREXAMPLES.jsonl",counterexamples)
    not_run={"schema":"LUCIA_AA_C4_DOWNSTREAM_STATUS_V1","behavior_gate":gate["verdict"],"activation_probe":"NOT_RUN" if failures else "PENDING","causal_interventions":"NOT_RUN" if failures else "PENDING","minimal_circuit":"NOT_RUN" if failures else "PENDING","algorithmic_reimplementation":"NOT_RUN" if failures else "PENDING","micro_lora":"NOT_RUN" if failures else "PENDING","external_causal_circuit_supported":False}
    write_json(out/"C4_DOWNSTREAM_STATUS.json",not_run)

    # Raw-free evidence index.
    evidence=[]
    for p in sorted(out.rglob("*")):
        if p.is_file(): evidence.append({"path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    write_json(out/"C4_EVIDENCE_INDEX.json",{"schema":"LUCIA_AA_C4_EVIDENCE_INDEX_V1","candidate_id":CANDIDATE_ID,"files":evidence,"raw_base_weight_included":False,"raw_adapter_weight_included":False,"tokenizer_files_included":False,"raw_logits_included":False,"raw_activations_included":False,"verdict":"PASS_RAW_FREE_RESULTS"})
    print(json.dumps({"candidate_id":CANDIDATE_ID,"positive_cases":len(cases),"negative_cases":len(negatives),"gate":gate["verdict"],"failures":failures,"confirmation":split_stats["CONFIRMATION"],"ood":split_stats["OOD"],"free_generation":gate["free_generation"]},indent=2))


if __name__ == "__main__":
    main()
