# Rules authority consolidation — working plan

## Accepted outcome and constraints

MTG-177 is draft work for review: account for every pinned Comprehensive Rule
with honest evidence, deliver bounded improvements, and assess CI cost without
weakening coverage. Jack Heart approved the draft and a later pursue run,
prioritizing competing definitions that execute the same rule differently;
those approvals have no supplied calendar dates. No ready publication, merge,
auto-merge or Task completion is authorized.

Jack Heart's 2026-10-21 direction in AGENTS.md requires complete shared human/bot
choices, viewer-safe observations and one native authority. Preserve Command
binding, stale rejection, actual pending decisions on reconnect, replay/fork
identity and identifiable priority-stop automation. Changed rules meaning needs
new world/conformance identity; frozen evidence remains unchanged.

## Current implementation

- The original audit is committed at `2b6e2b1b7`. The index accounts for 3,166
  numbered rules/subrules and 156 headings: 3 implemented, 25 partial, 2 missing,
  322 not applicable and 2,970 unknown. Unknown means triaged without proof.
- `c514d11b0` removes the independent stack-target minimum-mana-value comparison
  from offers; enumeration, acceptance and resolution use `target_is_legal`.
  Candidate order and distinct game-stage responsibilities remain.
- `202416a8` centralizes permanent battlefield presence and opposing hexproof
  before wording restrictions. It removes `hexproof_blocks`, four branch-local
  guards and repeated lookups. Controller, type, predicate and threshold checks
  remain; nontargeted presence helpers still have consumers. The CR 702.11b
  witness is enabled. No offer-time legality cache or stage merge is introduced.
- Live world is w12; tensor shapes, authored w11 content and learning schema 7
  are unchanged. CI/tests select fresh `semantic-kernel-w12-v1` receipts and
  Academy selects newly recorded w12 demo prefixes. Old worlds, checkpoints,
  protocol fixtures and demo bytes remain frozen.
- The Academy regression changes the selected legal offer and matching played
  Command while preserving its pre-command address. Existing ed2 frame/landmark
  checks, erd1 Command/digest rejection and optional-import boundaries remain.
- Inventory tests use a normal `scripts.audit_rules` import and rule-ID lookups;
  source text is cached only within one anchor-validation call. The CLI now
  selects rows, validates them, then compares/writes output, so invalid evidence
  cannot replace the index.

Rules input is the September 25, 2026 Wizards TXT at
`scratch/MagicCompRules-20260925.txt`, SHA-256
`8d860e451f20f38865b725b42d82feb714c725373dd8f3b32b8652b3eeb070ca`.
Detailed findings, anchors and receipts live in `docs/rules/audit-2026-10/`.

## Delete — do not maintain

Removed `scratch/profile_collector.py`: the one-off two-update monkeypatch served
its diagnosis and has no runtime/test consumers. Its complete source is retained
at `0c640a2e:scratch/profile_collector.py`; the profile log and validation receipt
remain. No production deletion targets or exclusive fixtures remain in this cut.

## Validation and remaining acceptance

`0c640a2e` retains the passing canonical diagnostic and reconciles the earlier
collector failure. Full result: `docs/rules/audit-2026-10/learnability-w12-canonical.json`.
The run used the draft later committed as `202416a8`; its source digest/native
SHA identify the tested bytes. It passed in 99.6s (86.5 CPU seconds), with 144
capacity assertions and production PPO/checkpoint reload: 65,536 transitions,
1,541 games, after 1.0 versus no-update 0.640625. The 480s total/300s PPO
deadlines, sample counts and assertions were unchanged. Canonical dependency-
blocked play/Study protocol also passed all 17 tests.

The earlier `.runs/learnability/rules-targeting-w12/worker.log` records a collector
deadline failure after 384.3s (76.2 CPU seconds). Inspection of
`manabot/verify/learning_control.py` and `manabot/sim/net_opponent.py`, plus the
unchanged two-update profile (9.047s), reproduced no collector defect. Host
contention is plausible, not proved. No production training repair was warranted.
Keep both outcomes; do not replay the full diagnostic for documentation/tooling.

Retained pre-packaging evidence: 8 debug targeting witnesses (plus 1 ignored
separate trigger-order proposal), 6 debug conformance checks over 14 cases /
3,491 Commands / 15 Phase rows, 8 Command/recovery checks, 4 repaired decision-
surface checks and 44 tooling/package-boundary checks. All 11 arena admission
cases passed across bounded chunks. These are historical receipts, not current-
head suite passes or full conformance/strength certification. The canonical
native wheel build and atomic install/alias-identity check passed after upstream's
`fd8cfb92` packaging merge (58.4s / 0.7s).

Gate/CI owns unfinished Study semantic retry, all 7 Study branch cases, and four
monitor cases (random/explicit-matrix outcomes, admission-failure persistence,
saved dashboard). Study runtime passed 3/4 and monitor 16/20 before 120s timeouts;
partial suites are not passes. Preserve failures, including the original mixed-
target fixture mistake and invalid pytest selector. The timeout helper now kills
its process group; its earlier owned orphans were stopped. No broad rerun or
budget increase is justified without changed code or new evidence.

## Separate findings and scope

CR 603.3b same-controller trigger ordering lacks a shared choice: at 2 life,
gain-2/damage-2 ordering distinguishes survival from loss. CR 616 replacement
ordering also covers a narrower deterministic subset than affected-player
choice. Collection, placement and movement have legitimate separate roles.
These missing-choice designs remain separate from targeting consolidation.

Prism/Treasure payment still has a retained counterexample despite bounded
arithmetic evidence. Payment repair stays with its owners. The last refresh
found MTG-174 identifies a training Task, so replacement payment ownership is
unresolved; optional-cost/continuation Task `4ba5dd` has delivery in history.
These historical observations authorize no lifecycle changes or new coordination.

No paid compute, training/deck expansion, frozen-evidence edits or Wave-memory
writes. Rules' GOAL was read; `.lf/wave/rules/MEMORY.md` is absent. Proposed memory
lesson: a shared predicate can still contain contradictory branch semantics;
consumer agreement and a rules-correct repair need distinct evidence.

Validation ceilings: two Cargo jobs, 900s per cold Rust command, 180s per warm
Rust command and 120s per Python check, with debug required. Full learnability's
explicit acceptance uses its existing 480s watchdog. No stress loops or random
load searches. Approximate combined draft ceiling is 10k lines excluding report/
index artifacts; prior conservative charged churn was 3,953 lines. Historical
step detail and accounting are retained at `0c640a2e:scratch/plan.md` and in the
report receipts. Realign owns reconciliation of the final accounting.

Check: `uv run --no-sync python scratch/run_bounded.py compress-inventory-cli 120 uv run --no-sync pytest tests/infra/test_rules_inventory.py tests/infra/test_ci_selection.py -q` — 43 passed (0.65s); unchanged runtime results reused; unfinished Study/monitor checks remain with gate/CI.
