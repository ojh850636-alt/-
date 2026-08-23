#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

SEED = 2251


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {matches}")
    return matches[0]


def exact_one_sided_p(rescue: int, harm: int) -> float:
    total = rescue + harm
    if total == 0:
        return 1.0
    return sum(math.comb(total, value) for value in range(rescue, total + 1)) / (2 ** total)


def bootstrap_delta(base: list[bool], full: list[bool], seed: int, rounds: int = 2000) -> list[float]:
    rng = random.Random(seed)
    count = len(base)
    values = []
    for _ in range(rounds):
        indices = [rng.randrange(count) for _ in range(count)]
        values.append(sum(full[index] - base[index] for index in indices) / count)
    values.sort()
    return [values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1]]


def preference_accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(row["reference_preferred"] for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--safety-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_jsonl(find_one(args.evidence_root, "C4_TF_BASE.jsonl"))
    full_rows = read_jsonl(find_one(args.evidence_root, "C4_TF_FULL.jsonl"))
    shuffled_rows = read_jsonl(find_one(args.evidence_root, "C4_TF_SHUFFLED.jsonl"))
    random_rows = read_jsonl(find_one(args.evidence_root, "C4_TF_RANDOM_RANK_MATCHED.jsonl"))
    dose_025 = read_jsonl(find_one(args.evidence_root, "C4_TF_DOSE_0.25.jsonl"))
    dose_05 = read_jsonl(find_one(args.evidence_root, "C4_TF_DOSE_0.5.jsonl"))
    dose_15 = read_jsonl(find_one(args.evidence_root, "C4_TF_DOSE_1.5.jsonl"))
    generation_base = read_json(find_one(args.evidence_root, "C4_FREE_GENERATION_BASE.json"))
    generation_full = read_json(find_one(args.evidence_root, "C4_FREE_GENERATION_FULL.json"))
    safety = read_json(find_one(args.safety_root, "C4_EARLY_SAFETY_GATE.json"))

    base_map = {row["case_id"]: row for row in base_rows}
    full_map = {row["case_id"]: row for row in full_rows}
    if set(base_map) != set(full_map) or len(base_map) != 200:
        raise RuntimeError("BASE/FULL case identity mismatch")

    split_statistics = {}
    for split in ("DISCOVERY", "CONFIRMATION", "OOD"):
        identifiers = [case_id for case_id, row in base_map.items() if row["split"] == split]
        base_values = [bool(base_map[case_id]["reference_preferred"]) for case_id in identifiers]
        full_values = [bool(full_map[case_id]["reference_preferred"]) for case_id in identifiers]
        rescue = sum((not base_value) and full_value for base_value, full_value in zip(base_values, full_values))
        harm = sum(base_value and (not full_value) for base_value, full_value in zip(base_values, full_values))
        base_accuracy = sum(base_values) / len(base_values)
        full_accuracy = sum(full_values) / len(full_values)
        split_statistics[split] = {
            "n": len(identifiers),
            "base_preference_accuracy": base_accuracy,
            "full_preference_accuracy": full_accuracy,
            "delta": full_accuracy - base_accuracy,
            "rescue": rescue,
            "harm": harm,
            "exact_one_sided_p": exact_one_sided_p(rescue, harm),
            "bootstrap_95": bootstrap_delta(base_values, full_values, SEED + len(split)),
            "mean_margin_delta": statistics.fmean(
                full_map[case_id]["margin"] - base_map[case_id]["margin"]
                for case_id in identifiers
            ),
        }

    prereg_ids = {
        case_id for case_id, row in base_map.items()
        if row["split"] in ("CONFIRMATION", "OOD")
    }
    base_prereg = [row for row in base_rows if row["case_id"] in prereg_ids]
    full_prereg = [row for row in full_rows if row["case_id"] in prereg_ids]
    controls = {
        "BASE": preference_accuracy(base_prereg),
        "FULL": preference_accuracy(full_prereg),
        "SHUFFLED": preference_accuracy(shuffled_rows),
        "RANDOM_RANK_MATCHED": preference_accuracy(random_rows),
        "DOSE_0.25": preference_accuracy(dose_025),
        "DOSE_0.5": preference_accuracy(dose_05),
        "DOSE_1.5": preference_accuracy(dose_15),
    }
    dose_sequence = [
        controls["BASE"], controls["DOSE_0.25"], controls["DOSE_0.5"],
        controls["FULL"], controls["DOSE_1.5"],
    ]
    dose_directional = (
        all(dose_sequence[index + 1] + 0.02 >= dose_sequence[index] for index in range(3))
        and dose_sequence[4] + 0.03 >= dose_sequence[3]
    )

    failures = list(safety.get("failures", []))
    confirmation = split_statistics["CONFIRMATION"]
    ood = split_statistics["OOD"]
    if confirmation["delta"] < 0.05 or confirmation["bootstrap_95"][0] <= 0:
        failures.append("CONFIRMATION_DELTA_NOT_SUPPORTED")
    if ood["delta"] < 0.02:
        failures.append("OOD_DELTA_TOO_SMALL")
    if controls["FULL"] - controls["SHUFFLED"] < 0.05:
        failures.append("SHUFFLED_CONTROL_NOT_BEATEN")
    if controls["FULL"] - controls["RANDOM_RANK_MATCHED"] < 0.05:
        failures.append("RANDOM_CONTROL_NOT_BEATEN")
    if not dose_directional:
        failures.append("DOSE_RESPONSE_NOT_DIRECTIONAL")

    base_generation_accuracy = generation_base["execution_accuracy"]
    full_generation_accuracy = generation_full["execution_accuracy"]
    base_negative_activation = generation_base["negative_sql_activation"]
    full_negative_activation = generation_full["negative_sql_activation"]
    if full_generation_accuracy - base_generation_accuracy < 0.10:
        failures.append("FREE_GENERATION_EXECUTION_DELTA_TOO_SMALL")
    if full_negative_activation - base_negative_activation > 0.10:
        failures.append("FREE_GENERATION_NON_SQL_ACTIVATION")

    failures = sorted(set(failures))
    gate = {
        "schema": "LUCIA_AA_C4_PARALLEL_BEHAVIOR_GATE_V1",
        "candidate_id": "R225-C4-QWEN05-TEXT2SQL",
        "source_scope": "exact source artifacts from run 29771037886",
        "parallel_mode_scope": {
            "positive_cases": 200,
            "discovery": 40,
            "confirmation": 80,
            "ood": 80,
            "actual_generation_positive_per_condition": len(generation_base["positive"]),
            "actual_generation_negative_per_condition": len(generation_base["negative"]),
        },
        "early_safety_gate": safety,
        "split_statistics": split_statistics,
        "controls": controls,
        "dose_sequence": dose_sequence,
        "dose_directional": dose_directional,
        "free_generation": {
            "base_execution_accuracy": base_generation_accuracy,
            "full_execution_accuracy": full_generation_accuracy,
            "base_negative_sql_activation": base_negative_activation,
            "full_negative_sql_activation": full_negative_activation,
        },
        "failures": failures,
        "verdict": "GREEN_BEHAVIOR_GATE" if not failures else "RED_BEHAVIOR_GATE",
        "causal_entry_allowed": not failures,
        "claim_boundary": "Teacher-forced preference and actual SQLite execution are separate evidence layers. Causal entry requires every hard Gate to pass.",
    }
    write_json(args.out_dir / "C4_PARALLEL_BEHAVIOR_GATE.json", gate)

    files = []
    for root in (args.evidence_root, args.safety_root, args.out_dir):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append({
                    "scope": root.name,
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
    write_json(args.out_dir / "C4_PARALLEL_GATE_EVIDENCE_INDEX.json", {
        "schema": "LUCIA_AA_C4_PARALLEL_GATE_EVIDENCE_INDEX_V1",
        "files": files,
        "raw_weight_included": False,
        "tokenizer_files_included": False,
        "raw_logits_included": False,
        "raw_activation_included": False,
        "verdict": "PASS_RAW_FREE_PARALLEL_GATE_EVIDENCE",
    })
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
