#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("c4_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load C4 core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subset_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split, count in (("DISCOVERY", 40), ("CONFIRMATION", 80), ("OOD", 80)):
        selected.extend([case for case in cases if case["split"] == split][:count])
    if len(selected) != 200:
        raise RuntimeError(f"parallel scope mismatch: {len(selected)}")
    return selected


def load_model(core, base_dir: Path, adapter_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
    model.eval()
    model.config.use_cache = False
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("BASE_FULL", "CONTROLS", "DOSE", "GENERATION"), required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--core-script", type=Path, required=True)
    args = parser.parse_args()

    core = load_core(args.core_script)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(core.SEED)
    torch.manual_seed(core.SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    all_cases, negatives = core.build_dataset()
    cases = subset_cases(all_cases)
    core.write_jsonl(args.out_dir / "C4_PARALLEL_SCOPE_CASES.jsonl", cases)
    core.write_json(args.out_dir / "C4_PARALLEL_SCOPE.json", {
        "schema": "LUCIA_AA_C4_PARALLEL_SCOPE_V1",
        "candidate_id": core.CANDIDATE_ID,
        "mode": args.mode,
        "positive_count": len(cases),
        "split_counts": {
            split: sum(case["split"] == split for case in cases)
            for split in ("DISCOVERY", "CONFIRMATION", "OOD")
        },
        "negative_total_available": len(negatives),
        "dataset_sha256": core.sha256_file(args.out_dir / "C4_PARALLEL_SCOPE_CASES.jsonl"),
        "evidence_contract": "same source, dataset generator, controls and thresholds as sequential fast Gate",
    })

    tokenizer, model = load_model(core, args.base_dir, args.adapter_dir)
    originals = core.save_lora_state(model)
    original_scales = {
        id(layer): float(layer.scaling[core.adapter_name(layer)])
        for layer in core.adapter_layers(model)
    }
    core.write_json(args.out_dir / "C4_PARALLEL_RUNTIME.json", {
        "schema": "LUCIA_AA_C4_PARALLEL_RUNTIME_V1",
        "mode": args.mode,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "dtype": str(next(model.parameters()).dtype),
        "lora_layer_count": len(core.adapter_layers(model)),
        "trust_remote_code": False,
        "device": "cpu",
    })

    started = time.time()
    if args.mode == "BASE_FULL":
        with model.disable_adapter():
            base_rows = core.score_pairs(model, tokenizer, cases, batch_size=16)
        full_rows = core.score_pairs(model, tokenizer, cases, batch_size=16)
        core.write_jsonl(args.out_dir / "C4_TF_BASE.jsonl", base_rows)
        core.write_jsonl(args.out_dir / "C4_TF_FULL.jsonl", full_rows)
        core.write_json(args.out_dir / "C4_BASE_FULL_SUMMARY.json", {
            "schema": "LUCIA_AA_C4_PARALLEL_BASE_FULL_V1",
            "base": core.aggregate(base_rows),
            "full": core.aggregate(full_rows),
        })

    elif args.mode == "CONTROLS":
        prereg = [case for case in cases if case["split"] in ("CONFIRMATION", "OOD")]
        core.restore_lora_state(model, originals)
        core.set_scale(model, 1.0, original_scales)
        core.shuffled_state(model, core.SEED + 11)
        shuffled = core.score_pairs(model, tokenizer, prereg, batch_size=16)
        core.restore_lora_state(model, originals)
        core.set_scale(model, 1.0, original_scales)
        core.random_state(model, core.SEED + 29)
        random_rows = core.score_pairs(model, tokenizer, prereg, batch_size=16)
        core.write_jsonl(args.out_dir / "C4_TF_SHUFFLED.jsonl", shuffled)
        core.write_jsonl(args.out_dir / "C4_TF_RANDOM_RANK_MATCHED.jsonl", random_rows)
        core.write_json(args.out_dir / "C4_CONTROLS_SUMMARY.json", {
            "schema": "LUCIA_AA_C4_PARALLEL_CONTROLS_V1",
            "shuffled": core.aggregate(shuffled),
            "random_rank_matched": core.aggregate(random_rows),
        })

    elif args.mode == "DOSE":
        prereg = [case for case in cases if case["split"] in ("CONFIRMATION", "OOD")]
        summaries = {}
        for factor in (0.25, 0.5, 1.5):
            core.restore_lora_state(model, originals)
            core.set_scale(model, factor, original_scales)
            rows = core.score_pairs(model, tokenizer, prereg, batch_size=16)
            core.write_jsonl(args.out_dir / f"C4_TF_DOSE_{factor}.jsonl", rows)
            summaries[str(factor)] = core.aggregate(rows)
        core.write_json(args.out_dir / "C4_DOSE_SUMMARY.json", {
            "schema": "LUCIA_AA_C4_PARALLEL_DOSE_V1",
            "doses": summaries,
        })

    elif args.mode == "GENERATION":
        confirmation = [case for case in cases if case["split"] == "CONFIRMATION"][:2]
        ood = [case for case in cases if case["split"] == "OOD"][:2]
        representative = confirmation + ood
        negative_representative = negatives[:2]
        with model.disable_adapter():
            base_generation = core.generation_eval(
                model, tokenizer, representative, negative_representative, "BASE"
            )
        full_generation = core.generation_eval(
            model, tokenizer, representative, negative_representative, "FULL"
        )
        core.write_json(args.out_dir / "C4_FREE_GENERATION_BASE.json", base_generation)
        core.write_json(args.out_dir / "C4_FREE_GENERATION_FULL.json", full_generation)

    core.write_json(args.out_dir / "C4_PARALLEL_MODE_COMPLETION.json", {
        "schema": "LUCIA_AA_C4_PARALLEL_MODE_COMPLETION_V1",
        "mode": args.mode,
        "elapsed_seconds": time.time() - started,
        "verdict": "PASS_MODE_COMPLETE",
    })
    files = []
    for path in sorted(args.out_dir.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(args.out_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": core.sha256_file(path),
            })
    core.write_json(args.out_dir / "C4_PARALLEL_MODE_EVIDENCE_INDEX.json", {
        "schema": "LUCIA_AA_C4_PARALLEL_MODE_EVIDENCE_INDEX_V1",
        "mode": args.mode,
        "files": files,
        "raw_weight_included": False,
        "tokenizer_files_included": False,
        "raw_logits_included": False,
        "raw_activation_included": False,
        "verdict": "PASS_RAW_FREE_MODE_RESULTS",
    })
    print(json.dumps({"mode": args.mode, "elapsed_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
