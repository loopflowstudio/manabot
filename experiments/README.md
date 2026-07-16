# experiments/

One experiment = one report (`exp-NN-<name>.md`). Raw data in `data/`.
Run provenance lives in the verify store (`.runs/verify.sqlite`).
**Runners:** the exp-specific driver scripts live in [runners/](runners/) —
`manabot/` keeps only reusable instruments.

## Discipline

1. **Predict first, in git.** Question, numeric prediction, kill criteria,
   and cost cap are committed *before* the run. No pre-run commit → the
   result is exploratory, not evidence.
2. **Name the strongest confound.** Every report says how its result could
   be wrong and what would discriminate.
3. **Mechanism over aggregates.** "Can/cannot" claims need a behavioral
   probe (competency scenario, action-level stat, per-bucket metric) — win
   rates alone are strength claims only.
4. **Numbers trace.** Every number in any doc traces to a report; every
   report traces to store rows or data files. Refutations stay on the
   record, dated, never silently edited.
5. **Seeds are the unit.** Game-level CIs quantify one checkpoint's eval
   noise; claims about a *method* need independent training seeds and
   cross-seed uncertainty. Three seeds are three data points.
6. **Protect the instrument.** Engine determinism, throughput, state
   injection, and search primitives are what make experiments cheap —
   changes that break them are failing changes regardless of green tests.

## Index

| exp | question | verdict |
|---|---|---|
| [00](exp-00-decision-profile.md) / [00-cost](exp-00-cost-basis.md) / [00c](exp-00c-seat-balanced-baselines.md) | calibrate the instrument | 194 decisions/game; single-init baselines meaningless; $0.44/1M steps. ~~Per-seat findings (94%/23.1% on-play)~~ deal artifacts per exp-06 — deal-averaged random mirrors are at parity (repro_06) |
| [01](exp-01-c1-training.md) | does the shaped recipe survive a real deck? | no — 0/3 seeds; one seat-parasitic; `cast_when_able` flipped sign |
| [02](exp-02-flat-mc.md) | how much intelligence is free? | search-256 = 99% vs random at $0; every trained policy below N=16 |
| [03](exp-03-distillation.md) | is distillation cheaper than RL? | yes — 90.5% vs matched-cost PPO's 52.7%; ladder ≈ N=8 at 1ms |
| [04](exp-04-potential-shaping.md) | is bias-free dense signal possible? | shaping was the disease — terminal-only wins; pass-collapse fails replication |
| [07](exp-07-expert-iteration.md) | does the expert-iteration crank compound? | no — R1 loses 74–26 to R0 (label economics); batched inference 12× landed |
| [06](exp-06-newworld-training.md) | did observation growth hurt training? | benign — seeds in/above the historical band |
| [08](exp-08-two-deck-matchup.md) / [08b](exp-08b-ancestral-dose.md) | what decides a matchup table? | pilot + card quality — 4× Recall swung it 55 points; UR-22% was a pilot artifact |
| [09](exp-09-control-competency.md) | can the pilot play control? | no — ≤0.39 correct at any N; random beats search on 3/5 scenarios; win rate masks incapacity |
| [10](exp-10-value-gate.md) | does search-with-V beat V-greedy? | gate passes deal-diverse 60.25% [55.4, 64.9], but V loses to cheaper random-rollout search (P3 refuted); V's ordering near-noise in undecided positions; first battery + training corpus (3 deals/225g) were deal-narrow |
| [11](exp-11-curriculum-exploitability.md) | does a stronger opponent teach better? | self-play quietly wins; the opponent installs the strategy; student robust (exploiter ≤26%) |
| [W2-214](w2-214-structural-semantic-katas.md) | can a bounded structural encoder break the bag encoder's exact symmetry on five static semantic relations? | instrument valid; bag exactly 50%; relational arm learned order/hierarchy but failed the pre-registered trainability and cost gates — redesign optimization/capacity before interpreting the remaining katas |

### Teacher program (W2-234)

| report | question | verdict |
|---|---|---|
| [teacher0](w2-234-teacher0.md) | can a search-supervised policy/value student learn from Teacher-0? | proceed to scale-up — joint policy/value matches policy-only CE/top-1, improves held-out value Brier 0.2536→0.2043, 71%→73% vs random |
| [teacher1](w2-234-teacher1.md) | does the determinized PUCT substrate work end-to-end? | substrate pass — tree search, node reuse, alternating backup, self-play shards all function; priors still uniform, playouts random |
| [teacher1-pilot](w2-234-teacher1-pilot.md) | do budgets 8/32/128 give strong, legal, affordable targets? | **preregistered, unrun** — fail-closed until the terminal Teacher-0 manifest and control lock exist |

### Platform evidence (W2)

| report | claim proven |
|---|---|
| [card-conformance-audit](card-conformance-audit.md) | 55 registered cards audited vs Scryfall; 20 shell mismatches fixed; fixture + tripwire test |
| [milestone-1](milestone-1-two-deck-slice.md) | all four rules stages landed for the UR-Lessons-vs-GW-Allies slice |
| [opcode-alignment-v1](opcode-alignment-v1.md) | semantic-program-only encoder: 91.7% zero-shot held-out cluster accuracy vs 0.0% for opaque card-id arms |
| [structured-policy-decoder](structured-policy-decoder.md) | structured decoder matches legacy adapter exactly (6435/6435 actions) at comparable latency — migration evidence |
| [w2-179](w2-179-content-pack-local-diagnostic.md) | immutable ContentPack / mutable Card seam with deterministic CardDefId hashing |
| [w2-182](w2-182-search-branching-v1.md) | full-clone branching baseline: 2,194 sims/s, clone p50 4.5µs |
| [w2-198](w2-198-compact-clone-undo-v1.md) | compact clone+undo at parity with full clone (2,097 sims/s) with journal accounting |
| [w2-208](w2-208-content-pack-clone-allocations.md) | exact clones share the immutable ContentPack by Arc pointer |
| [w2-215](w2-215-semantic-projection.md) | viewer projection baseline: ~53 tokens/object p50, ~9.9µs hot projection, zero failures |
| [w2-223](w2-223-typed-ir-interpreter.md) | generic fail-closed interpreter executes all 31 admitted programs by opcode — no card-name dispatch |

(Older platform docs: [first-light-run-1](first-light-run-1.md),
[sps-closeout](sps-closeout.md).)
