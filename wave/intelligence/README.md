# Intelligence (search + beliefs, one wave)

> Redesigned 2026-07-10 after external advisor review and the exp-00..11 arc.
> Predecessor strategy/history: [03-search-era-notes.md](03-search-era-notes.md);
> beliefs design: [02-beliefs-design.md](02-beliefs-design.md);
> cycle log: [01-experiment-loop.md](01-experiment-loop.md).

## Thesis question (provisional — owner edits welcome)

> **Can information-set-consistent decision-time planning handle long-horizon,
> variable-action card games with no natural subgame boundaries, at practical
> inference cost?**

Magic's specific hardness — priority and the stack weave decisions
continuously; there are no poker streets to root solves on — is the potential
contribution. Public-belief search itself is not novel (ReBeL); reducing
strategy fusion in determinization is active work (EPIMC,
arXiv:2408.02380). The boundary problem is ours.

## Standing constraints

- **Worlds** (`/WORLDS.md`): shape changes are versioned; measurements are
  never ported across worlds; baselines re-run per freeze.
- **Pacing**: a cycle closes when its result is integrated into
  `paper/understanding.md` at whiteboard-defensible level. One cycle in
  flight; un-integrated results queue.
- **Reproduction** (`experiments/repro/`): headline claims have end-to-end
  scripts; the suite re-runs on world freezes. A claim without a repro script
  is provisional.
- **Discipline** (`experiments/README.md`): predict-first-in-git, name the
  confound, mechanism over aggregates, numbers trace, protect the instrument.
- **Statistics**: the experimental unit is the *training seed*, not the
  evaluation game. Game-level CIs quantify eval noise of one checkpoint;
  claims about a *method* need independent seeds and cross-seed uncertainty.

## Metrics, in order of authority

1. **Exploitability** (approximate best response, standing per-checkpoint —
   promoted from one-off probe): the only strength number that cannot be
   gamed by opponent choice. In the microgame (below), exact NashConv.
2. **Competency scenarios** (`manabot/verify/competency.py`): mechanism-level
   capability. Capability claims cite these, never win rates alone.
3. **Matched wall-clock comparisons**: any "X beats Y" is at equal
   wall-clock or the imbalance is stated.
4. **The search ladder** — *demoted to a compute-calibrated baseline.* It
   anchors cost comparisons; it is not the objective. Measured reason: the
   ladder's own anchor dominates random opponents while scoring worse than
   random on delayed-control scenarios (exp-09).

## The plan: diagnose, then treat

The exp-09/exp-10 failures have at least three candidate causes with three
different treatments. No experiment has yet isolated which we have. The wave
leads with two diagnostic instruments; their results choose the algorithm.

### D1 — the 2×2 ablation (information × continuation)

On competency scenarios and micro-format matchups, cross:
{hidden information, full information} × {random continuation, oracle/exact
continuation}.

| outcome pattern | diagnosis | treatment |
|---|---|---|
| fails even at full-info + exact search | scenario/search implementation bug | fix it first |
| fails at full-info + random continuation | planning/rollout deficit | value or tree search over rollouts |
| full-info exact succeeds, hidden-info fails | information-set inconsistency (true fusion) | CFR-style / public-belief solving |
| hidden-info improves with oracle beliefs | belief estimation deficit | likelihood-weighted determinization |

This also settles the terminology debt: exp-09's S2-class failures are only
"strategy fusion" if the hidden-information column isolates them.

### D2 — the exactly solvable microgame

A Magic subgame small enough to solve: counterspell bait, held removal, one
represent/bluff decision; tiny decks, capped turns. Compute the equilibrium
and exact best responses (NashConv). Every planner we own gets an exact
exploitability number there before we trust its approximate one at scale.
Ten thousand approximate games against random are worth less than one game
where the equilibrium is known. The full engine is for scalability, not
ground truth.

### Then: one treatment, chosen by D1/D2 — not before

Value/tree search for a planning deficit; likelihood-weighted determinization
for a belief deficit; CFR/public-belief solving (per
[02-beliefs-design.md](02-beliefs-design.md), with its Sokota-conditions
caveat) for genuine fusion. Implement the *simplest* indicated baseline,
compare per the metrics hierarchy, write the causal conclusion into
`paper/understanding.md`. That conclusion — not a ladder rung — is the unit
of progress.

## What is already known (one line each, repro-gated)

Search-256 with random rollouts is the unbeaten teacher (exp-02); distillation
beats matched-cost PPO (exp-03/07); pay-per-event shaping was harmful
(exp-04); the crank does not compound by default — label economics (exp-07);
the pilot cannot play delayed control (exp-09); scalar V is blind exactly in
interaction states (exp-10); the opponent installs the strategy, self-play
wins, the student is robust to a matched-budget exploiter (exp-11); per-seat
claims before the deal fix are suspect (exp-06).

## Exits (inherited, still armed)

Exit 2 (model-free game-theoretic pivot) remains live if the chosen treatment
fails its own gate; Exit 4 (wrap the platform paper) unchanged. Exit 1's
tripwire is superseded by D1 — diagnosis replaces the plateau proxy.

## Not here

Rules expansion (parked wave), GUI product work (own wave), cube optimization
(north star, not now), multiplayer/general-sum, hidden decklists.
