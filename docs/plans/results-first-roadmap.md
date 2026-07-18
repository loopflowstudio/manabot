# The results-first roadmap

Plan date: 2026-07-18. Status: adopted — supersedes the I1–I6 build ordering
in the Intelligence wave; per-rung status lives in the project KRs.

## Why this plan exists

The 2026-07-17/18 convergence push landed the belief-conditioned substrate end
to end: viewer-relative possible worlds and the typed query grammar in managym,
the exact-Bayes range tracker and `ExactRangePlayer` (INT-9), conditional
determinized PUCT with paired inverse-CDF world sampling (INT-12/INT-13), the
versioned `advice-v1` surface, the conditional shard contract (INT-14), the
visit-teacher loop (INT-4), and the world-pinned skill arena (INT-6).

Every one of these is currently an instrument without a committed production
result. Advice serves checked static fixtures; the belief tracker runs on no
live path; the arena has anchors and one challenger but zero rating runs; the
shipped belief comparisons move the distribution (max policy delta 0.125)
without ever flipping the recommendation (`top_action_changed` is 0.0 in every
retained condition); the INT-4 production harness has never executed.

The leverage has flipped. **The unit of progress is no longer a new
instrument; it is the first frozen result on each existing one.** Each rung
below closes only with a committed, replayable artifact.

The first agent worth being proud of is defined by three observables:

1. it starts from reasonable beliefs;
2. it updates beliefs in a reasonable, inspectable way;
3. its recommended actions visibly change in reasonable ways when specific
   cards are assumed in hand.

The exact-Bayes tracker over the compatible-deal prior already supplies the
normalized machinery for 1 and 2; the ladder determines whether its prior and
updates are reasonable in practice and makes all three observable, live, and
measured.

## The results ladder

### R1 — Freeze the first recommendation flip

Author a small suite of decision positions where a typed condition *should*
flip the argmax — the canonical case is the architecture's own product proof:
casting into open mana under `Has(Counterspell)` versus `Lacks(Counterspell)`.
Run conditional determinized PUCT offline at a real budget (hundreds of
traversals, 16+ paired worlds; latency is irrelevant to a frozen fixture).

Done when one checked fixture with `top_action_changed = 1.0` under paired
seeds is frozen, replays exactly, and is served through the `advice-v1`
comparison. If no curated position flips at ~64× the INT-12 budget with
uniform-random leaves, retain that as evidence — it is the measured case for a
stronger leaf evaluator (R4) and sharper positions, not a failure.

### R2 — Put the tracked belief on the live path

`BeliefTracker` runs during real play from the compatible-deal prior, and
advice at a live decision resolves the actual tracked posterior instead of a
static authored payload. The live decision address is Game's (the GAM-4
deferral — today the canonical replay address exists only at game close);
Intelligence's half is a belief resolver bound to the live match with the same
fail-closed identity discipline as the fixture path. Serve at a declared
compute class; cache where identity permits.

Done when one full game through `./scripts/play` can request advice mid-game
and the response's belief receipt hashes the live posterior.

### R3 — Commit the first belief-calibration curves

The "updates beliefs reasonably" observable: per game, posterior mass on the
opponent's actual hidden hand versus the compatible-deal prior's mass, over
the decision sequence. Requires closing the `RulesProviderGap` cases (managym
exposes public-commitment identity at every admitted boundary on the selected
matchup's exercised path) and binding a byte-locked checkpoint as the
action-likelihood model — `FrozenPolicyLikelihood` off its
`test-only-likelihood/v1` fallback. This is also the intelligence ladder
research's prescribed explanation of the belief player's arena result.

Done when a repro script emits the curves for seeded games and the result is
committed with standard receipts.

### R4 — First committed strength results

Two halves, either order:

- **First arena run.** Run INT-6 for real: frozen anchors, the dPUCT-32
  challenger, and `ExactRangePlayer` against a uniform-determinization control
  at matched compute — the measured answer to whether beliefs improve play.
  Commit ratings, the payoff matrix, and paired-deal uncertainty.
- **First production teacher iteration.** Unblock the INT-4 production harness
  (recover or re-freeze the absent Teacher-0 control bytes under a versioned
  contract note — never edit the frozen contract in place) and run one
  production visit-teacher iteration with multi-seed students.

The INT-7/INT-8 evidence stands: learned value heads and learned priors both
weakened one-seed smoke players inside search. The near-term strongest-player
hypothesis is belief-weighted, uniform-prior determinized PUCT at a larger
declared budget; learned components re-enter only through arena admission.

## Explicitly deferred until R1–R4 artifacts exist

- **The supervised belief head.** No rung needs it; exact-Bayes tracked
  beliefs cover the first product proof. Begin it only when it has a
  calibration baseline (R3) to beat, keeping the accepted contract: supervise
  from retained hidden worlds, never feed hidden truth at inference, adapt
  INT-9 rather than landing a parallel hand ontology.
- Content-pool widening beyond the interactive mirror, and the factorized
  marginal belief degradation — not needed while exact support fits.
- New planner families — the D1/D2 diagnostics still choose them, after the
  R results exist.
- Completing the `PlanningProblem` convergence (retiring the compatibility
  adapter) continues as background hygiene; it gates no rung.

## Cross-wave dependencies

- **Game** owns the live decision address and the player-authored belief
  input surface (R2's product half) and the fork/Retry/return UI sequenced
  behind it.
- **Rules** owns closing the `RulesProviderGap` public-commitment identity
  cases on the exercised path (R3's provider half).
- **Intelligence** owns everything else on the ladder.
