# Dedup & Consolidation Playbook

## Phases
- Inventory & stack detection
- Clone detection (hash/name/AST similarity)
- Define canonical modules and adapters
- Replace references and remove shadows
- Continuous verification (build/test/lint/typecheck)

## Commands
- tools/hash-inventory.sh: content hashes and name collisions
- tools/stack-detect.sh: languages, package managers, entry points
- tools/diff-apply.sh: apply batch edits safely

## Handoff Procedure (120k threshold)
- Snapshot CONTEXT/WORKLOG
- Write handoff summary
- Split tasks and assign

## Reporting
- artifacts/hash_inventory.tsv
- artifacts/name_collisions.tsv
- artifacts/stack_report.txt
