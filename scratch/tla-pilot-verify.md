# TLA pilot continuation — 2026-11-04 UTC

MTG-213 is incomplete: v3/256 failed with 270 distinct facts before its first
GPU learner update. There is no trained endpoint and 256 scheduled evaluation
games are missing. The Task's rental ended; no unchanged retry or automatic
capacity increase is selected. The next capacity/representation decision belongs
to review; broader representation/history/memory work has Task 5da5f4.

## Surviving contract and evidence

PR #416 merged as `3b122de9078c985f846a0f3c5a3a42ebca3b1182`; its tree
`3e696c396defb93716b3a428a1bef66b9abe71ab` matches checkout HEAD `7b8bb3273`.
CI run 41752523278 and 38 focused corpus/pilot/discovery tests passed. Etude froze
that exact merged source for run `c216c6be7037`, cohort `mtg213-pilot-v3`.

Keep the v3 protocol: w17/card digest
`dcb9d47c674a657176c2a9d56e016288d1e175729aa09cd60f0dc333633330ed`,
v6/256, unchanged 134 trophy entries, learning recipe, seeds and eight-pair/four-leg
schedule. Jack Heart approved the larger-capacity hotfix; the operator selected
256 after his question. V1/v2/v3 inputs and evidence remain frozen.

`experiments/tla-trophy-pilot.md` owns the result narrative and accounting.
Its linked `experiments/data/mtg213/gpu-v3-20261104/` receipts retain the CUDA
checks, 171 completed-but-unoptimized games, 86 active stream prefixes, the
270-row scalar/vector/Python replay, fresh initialization and 32 valid baseline
games (3 wins / 29 losses, all replay-passing). Diagnostic parity at 512 proves
neither a new pilot capacity nor universal coverage. All 120 evidence files are
artifact-published in `experiments/data.manifest.json`; the advertised
`etude-result.json` was absent. Local scripts and intermediate receipts remain
in `.runs/mtg213/v3-continuation/`.

## Operational limits for continuation

Etude 0.2.13 via isolated `uv tool run` restored inventory decoding of
`input_undelivered`; the frozen worker retained 0.2.12. Before launch there were
no provider pods, and only an unrelated owned mini session was unsettled.
L4 `l4-762887e71be1` / pod `z84dbb8r0fw52v` cost $0.59/hour with a $0.7375
maximum reservation and $1 cohort ceiling within the existing $10 cap.

A network outage stalled cloud evaluation discovery before any match and delayed
cleanup. The reader was interrupted; the canonical CLI evaluated a byte copy of
this run's tracking prefix without retrying a failed match. Cancellation intent
was retained at 02:26:01 UTC. After connectivity recovered, the existing L4 owner
settled the failed run under its supply lock and confirmed deletion at
02:26:33 UTC. A global poll stuck on unrelated mini supply was stopped. No
database rewrite, package patch or action on another person's work was used.

Prior provider bills total $0.07092813; the full v3 rental estimate is $0.09958535,
including idle cleanup. Cumulative billed-plus-estimated compute is $0.17051348;
the smaller v3 per-run allocation estimate ($0.03342127) omits some rental time.
Retain all three reservations ($2.2125) against $10 until final billing. The
remaining differently named provider pod was absent from this launcher database
and left untouched. All local processes started by the continuation exited or
were stopped.

## Delete — do not maintain

No product path was replaced. Removed inherited MTG-239 notes from
`scratch/questions.md` and superseded pending-launch instructions/validation
chronology from `experiments/tla-trophy-pilot.md`; their committed detail remains
at `7b8bb32731205981b6be09097eb4ca4b39b3864f` under those same paths. Keep the
report as the result owner and this note as the continuation boundary. No
remaining deletion target, alternate trainer, dependency pin or schema is needed.

Validation: `git diff --check` passed; all 120 added manifest entries match local
hashes/sizes, prior entries are unchanged and report links resolve. Reuse the
passing `uv run --extra dev pytest tests/infra/test_experiment_data_untracked.py -q`
(1 test); source behavior is unchanged. Gate owns publication checks;
realign/review owns the next prospective contract before dependent GPU execution.
