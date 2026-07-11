# 04: The 2×2 Diagnosis (cycle D1) — launch spec

Self-contained brief for running the intelligence wave's first diagnostic
cycle. Written 2026-07-10 for dispatch from a fresh context; assumes no
conversation history — everything needed is in this file and the paths it
cites. World: **w2** (see `/WORLDS.md`).

## The question

The project's search-based player fails delayed-control decisions
(exp-09: holds neither board-wipes nor counterspells; max 0.39 correct on
known-correct-line scenarios, flat in simulation count;
`experiments/exp-09-control-competency.md`). At least three distinct
mechanisms could cause this, each with a different remedy. No experiment has
yet isolated which is operative. D1 does exactly that, by crossing two
factors:

- **Information**: {hidden (normal play — opponent hand/libraries unknown,
  search determinizes), full (oracle — search runs on the true game state,
  no determinization)}.
- **Continuation** (what happens below the first decision in evaluation):
  {random playouts (uniform-random legal actions to terminal — the current
  evaluator), strong/oracle continuation (the best available approximation
  of correct follow-up play — see Design decisions)}.

## Interpretation table (pre-committed)

| outcome pattern | diagnosis | indicated treatment |
|---|---|---|
| fails even at full-info + strong continuation | scenario or search implementation bug | fix before anything else |
| fails at full-info + random continuation, recovers with strong continuation | planning/rollout deficit (value lives in the plan, random playouts can't execute plans) | tree search / stronger rollout policy |
| full-info succeeds (both continuations), hidden-info fails | information-set inconsistency — strategy fusion proper (future actions conditioning on hidden info) | CFR-style / public-belief solving (`wave/intelligence/02-beliefs-design.md`, incl. its Sokota-conditions caveat) |
| hidden-info recovers when beliefs are oracle-informed | belief-estimation deficit | likelihood-weighted determinization |

Terminology note (per the 2026-07-10 advisor review): do NOT call a failure
"strategy fusion" unless the hidden-information column isolates it. exp-09's
failures are so far only "delayed-planning failures."

## Test beds (all exist, pointers)

1. **Competency scenarios S1–S5** (`manabot/verify/competency.py`;
   engine state-injection surface `Env.scenario_*`): constructed positions
   with documented correct lines (counter-the-bomb, hold-the-wipe,
   bolt-the-threat, race-vs-block, hold-up-quench). Primary bed — each cell
   of the 2×2 is a scenario sweep.
2. **Micro-format** (MICRO_AGGRO / MICRO_CONTROL deck constants, same
   module): mirrors ≈50% by construction; control-vs-aggro cross matchup
   with behavioral probes (instant-holding rate, proactive-counter rate).
   Secondary bed for aggregate corroboration.

## Design decisions the runner must make (and document)

- **Full information operationally**: search on a clone of the TRUE state
  (skip `determinize`; the primitives are `Game::clone` /
  `Env.flat_mc_scores` — see `manabot/sim/flat_mc.py` and
  `managym/src/flow/` for the Rust surface). Verify the searcher in
  full-info mode genuinely sees the opponent hand (assert on a constructed
  scenario where the correct line depends on it).
- **Strong continuation operationally**: exact solving is not available in
  the full engine; the honest approximations, in preference order:
  (a) high-N full-info search as the continuation policy for the first K
  plies below the decision (expensive — budget per cell), (b) the strongest
  distilled policy checkpoint as rollout policy (w2-compatible checkpoint:
  see `experiments/exp-10-value-gate.md` provenance for the fresh BC
  student), (c) scripted correct-line continuations for scenarios where the
  correct line is short and known (S2's "wait one turn, then wipe" is 2-3
  forced choices — scriptable). Document which is used per cell; (c) is
  acceptable and cheapest for scenario beds.
- **Oracle beliefs cell** (row 2, col 2 refinement): determinized search
  whose sampled worlds are replaced by the true hidden state — i.e., hidden
  information formally present but belief error zeroed. This isolates
  belief-estimation from information-set inconsistency. One extra column if
  budget allows; pre-registered in the interpretation table above.

## Non-negotiables (project discipline, `experiments/README.md`)

- **Deal diversity**: per-game engine seeds everywhere. The single-deal bug
  (F1, `paper/understanding.md`) is fixed on main but the canary is
  `experiments/repro/repro_06_seat_parity.py` — if in doubt, run it.
- **Predictions committed before results** (numbers + kill criteria + cost
  cap in git first). Suggested cost cap: 6 wall-clock hours, ≤4 workers.
- **Timeless writing**: absolute dates, world tags, source paths, terms
  defined. No conversation references.
- **Seeds are the unit** where training is involved (this cycle involves
  none — scenario sweeps are per-run stochastic; report per-scenario Wilson
  CIs over ≥100 runs/cell).
- Report: `experiments/exp-12-2x2-diagnosis.md` + data JSON; RESULT line in
  `wave/intelligence/01-experiment-loop.md`; integration into
  `paper/understanding.md` per the Pacing rule before the next cycle.

## Prior predictions on record (context, not binding)

exp-09 and exp-10 jointly suggest the planning-deficit and belief rows are
both live: random playouts demonstrably cannot execute "wait, then act"
plans (S2 mechanism, exp-09), and scalar value ordering collapses in
undecided mid-game states (exp-10 P1). A pure strategy-fusion diagnosis
(full-info clean, hidden-info broken) would be the surprising outcome. The
runner should register its own cell-level predictions before running.

## What this cycle is NOT

No training, no new engine mechanics (state injection and full-info cloning
exist), no treatment implementation — the deliverable is the diagnosis table
filled with measured numbers and the indicated treatment named. Treatment is
the NEXT cycle.
