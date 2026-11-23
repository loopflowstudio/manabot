# experiments/

One experiment = one module and one report: `<name>.py` here, `reports/<name>.md`
for what it showed. Raw data in `data/`.

The [Haiku versus MTG-183 evaluation](reports/haiku-mtg183.md) reports a completed
100-game hosted-model comparison, with full evidence in the experiment cache.

Each experiment people run today is one file: its recipe, built from
[`manabot.recipes.builders`](../manabot/recipes/builders.py), its jobs, and
anything only it needs to run and report. `manabot/` keeps the reusable
instruments. Only a few experiment files are kept, each as the baseline to copy
for the next experiment of its kind; a finished experiment's code is removed
and its report under [`reports/`](reports/) names the last commit with it. The
regime files retired experiments trained stay under `regimes/` as records;
nothing on main reads one as a regime
([saved regimes](../docs/training.md#saved-regimes)). `tla_frontier.py` reads
the deck corpus (`deck_source`) of its record and states every other setting
itself.

## Etude experiment files

```sh
uv run etude report experiments/ --source r2://manabot/jobs --serve   # every experiment, with an index
uv run etude launch experiments/elemental.py --revision COMMIT --updates 3200 \
    --hours 11.9 --usd-per-run 9 --rent l4                            # launch what has not been run
uv run etude train experiments/belief.py --job policy                 # train one declared job here
uv run etude eval experiments/elemental.py --source r2://manabot/jobs # play what is new, here
```

A file defining `training_jobs()`, `eval_jobs(state)` or `report()` is an Etude
experiment, named for its stem. Its functions are explicit: `train(config)` is
what a run does (its own, or a maintained recipe's imported from
`manabot.recipes`); `training_jobs()` returns every `etude.jobs.TrainingJob`
that should have been run by now, each with a required `name`;
`eval_jobs(state)` returns the `etude.jobs.EvalJob`s whose matches the stored
checkpoints should play; `report()` draws them. A run's address is always
`EXPERIMENT/JOB@SEED`, such as `elemental/elemental@18101`. `etude launch FILE`
and `etude queue FILE` read both declarations and get machines only for what
has not been run, so typing the command again launches what is new.

| file | a baseline for | `training_jobs()` | `eval_jobs(state)` | `report()` |
| --- | --- | --- | --- | --- |
| [elemental](elemental.py) | the current recipe, with a checkpoint ladder | `elemental`: fresh seeds of the ten-deck recipe | 10×10 heuristic grid and checkpoint pairs, played by [`elemental_play`](elemental_play.py) | grid score, deck matrix and rankings, crosstable of pairs |
| [seat_comparison](seat_comparison.py) | comparing method arms | `seat-comparison-one` and `seat-comparison-both`, four seeds each | — | score by update, one-seat dotted |
| [tla_trophies](tla_trophies.py) | monitoring combat decisions on the full TLA trophy population | packed w64 seeds on 7,859 training pools | fixed four-leg deals from 873 held-out pools against the capped heuristic | strength, play behaviour and combat probes |
| [tla_frontier](tla_frontier.py) | comparing model sizes on a deck corpus | one wall-bounded job per size, named for the regime record whose settings it restates | each size against its earlier self, the heuristic and the other size | progress curves and head to head |
| [batched_baseline](batched_baseline.py) | one GPU recipe ([size and batching study](reports/longer-learning-017.md)) | `batched_baseline` | — | self-play performance and learning curves |
| [belief](belief.py) | a job that starts from another run | `policy`, then `sampler` from its final checkpoint once state shows it finished ([guide](../docs/training.md#a-recipe-that-starts-from-another-run)) | — | self-play curves and sampler metrics |

[`llm_claude`](llm_claude.py) and [`llm_qwen`](llm_qwen.py) are eval-only files,
a language model against a frozen checkpoint over the Elemental decks; they are
not runnable yet (Etude EXP-70), and each file's docstring says what is missing.

[`elemental_grid`](elemental_grid.py) and [`elemental_long`](elemental_long.py)
hold the frozen evaluation schedules `elemental` plays, and
[`tla_corpus`](tla_corpus.py) and [`tla_pools`](tla_pools.py) build the TLA
deck corpus; they declare no jobs.

Perf and ML details draw self-play's own keys ([`manabot.training.panels`](../manabot/training/panels.py)),
including a flame graph of each update's `time/` seconds. The two-deck
benchmark (`manabot.eval.benchmark`) is still bound to the retired executor's
training record, so `seat_comparison` has no `eval_jobs(state)` yet.

The MTG-183 follower and its report, the MTG-186 pairings and card tables, the
curriculum grid readers, the inference baselines and the LLM campaign drivers
were removed; the last commit with them is `b49df79b49`. The compound-decision
study is at commit `ba3cd58de25bcc5d0b32e0fa3c546ee3d27be52b`.

Reports of finished studies stay as the record. The runners, contracts, plans
and fixtures they cite were removed once nothing current used them; git
history holds them at the commit each report names.

## Data

```bash
uv run --extra artifacts python -m manabot.data fetch                # everything
uv run --extra artifacts python -m manabot.data fetch experiments/data/etu126
uv run --extra artifacts python -m manabot.data publish experiments/data/<new-run>
```

`data/` is not in git. [`data.manifest.json`](data.manifest.json) records each
file's path, sha256 and size; the bytes live once per digest at
`r2://manabot/sha256/<sha256>` and need the team's R2
credentials (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, in Doppler `manabot/prd`). `fetch` restores files to their original paths, verifies every
digest, skips files already present and refuses to overwrite a file whose bytes
differ. Runs still write results under `data/`; `publish` uploads them (storage
checks each SHA-256; nothing is read back) and updates the manifest, which you
commit with the report. Every worktree on a machine shares one read-only cache,
`$MANABOT_DATA_CACHE` (default `~/.cache/manabot/sha256/<sha256>`), filled by
`publish` and by downloads, so a machine reads each digest from the store at most once. Tests that read data are marked
`experiment_data(...)`: they fetch what they name, or skip with this command
when the store is unreachable. Data committed before the manifest remains in git
history.

## Protocols and regimes

New experiments add nothing under `protocols/` or `regimes/`: an experiment is
its module at a commit and its one report
([work boundaries](../docs/workflows.md#experiments)). The protocol files
still there are read by an experiment module or are the only written record of
an earlier experiment; the regime files are records only. Job, identity and superseded protocol files that nothing reads were
removed from the tree; they remain at
[revision ac601834d](https://github.com/loopflowstudio/manabot/tree/ac601834d/experiments/protocols),
where older documents that name them by path can find them.

Main carries one policy input, v9, so the modules of finished experiments bound
to earlier inputs were removed with their regimes: the TLA trophy pilots, the
state-representation studies, the growing-pool league, the curriculum cohorts
and the throughput C1 readings. Their reports and protocols name those modules
at the source each run pinned; ac601834d is the last revision carrying them.

## Discipline

1. **Build before diagnosing.** Primary experiments exercise a runnable agent,
   teacher, or training loop in the real engine. Use katas and ablations to
   resolve an observed ambiguity or choose between concrete next builds, not
   as admission tests before integration.
2. **Exploratory by default; a finding asks first and decides after.**
   Calibration, throughput, capacity and pilot runs are labelled exploratory.
   A result that will be cited as a finding commits one paragraph *before*
   its runs: question, what is measured against what, the strongest confound
   and a cost cap. No threshold set before the data decides the outcome; the
   report reads the result against the incumbent and recommends, with its
   reasoning. The freeze is the commit and the run ID; a change that keeps
   the question is a dated line in the report
   ([from a result to a decision](../docs/experiments.md#from-a-result-to-a-decision)).
3. **Name the strongest confound.** Every report says how its result could
   be wrong and what would discriminate.
4. **Mechanism over aggregates.** "Can/cannot" claims need a behavioral
   probe (competency scenario, action-level stat, per-bucket metric) — win
   rates alone are strength claims only.
5. **Numbers trace.** Every number in any doc traces to a report; every
   report traces to store rows or data files. Refutations stay on the
   record, dated, never silently edited.
6. **Seeds are the unit.** Game-level CIs quantify one checkpoint's eval
   noise; claims about a *method* need independent training seeds and
   cross-seed uncertainty. Three seeds are three data points.
7. **Protect the instrument.** Engine determinism, throughput, state
   injection, and search primitives are what make experiments cheap —
   changes that break them are failing changes regardless of green tests.

## Index

[One token per game thing, MTG-344](reports/one_token_layout.md): **completed
2026-11-14**, exploratory (USD 2.65). Input v10 rows and input v11's one token
per game thing are indistinguishable against the heuristic at matched updates,
with matched throughput per learner transition. Evidence is under
`experiments/data/one_token_layout/`.

[Live-token attention, MTG-319](reports/live_token_attention.md): **completed
2026-11-12**, exploratory pilot on one L4 (USD 0.24). Attending over valid objects
only, on top of bucketed shapes, takes the w384l8 learner step from 386 ms to
202 ms (1.91×, bar 1.5×). Outputs match the padded path as closely as bucketed
shapes do. Build-out recommended. Evidence is under
`experiments/data/live_token_attention/`.

[Mirror search on MTG-183](reports/search-mirror.md): **completed 2026-11-12**,
exploratory — 3,000 games (six settings × 500 deals) plus an 8,000-game earlier
sweep on the frozen w13 policy. With playable uniform hand sampling, mirror
search beats the policy's sampled play by +12.5 to +17.4 points from 4 to 16
shuffles and beats PUCT-1024 at lower cost; the true hand adds at most +5.2.
The HTML report and evidence are in the data manifest under
`experiments/data/search-mirror-v1/` and `search-mirror-v2/`; source is tag
`mirror-w13-study-frozen`.

[Critic readouts and attention depth, MTG-252](reports/value-readout-252-results.md):
**completed 2026-11-09** — five arms, three seeds each, 300 updates and 28,200
valid evaluation games. Depth-4 masked mean/token score 61.0%/61.7% against the
heuristic, with no clear readout winner. The equal-update depth advantage largely
disappears at matched measured training hours; no default promotion. Graphs,
uncertainty, cost limits and the offline HTML report are retained in the data
manifest under `experiments/data/value-readout-252/`.

[V7 state reduction, GPU horizon](reports/state-representation-gpu.md): **completed
2026-11-05** — four paired seeds, 256 updates (524,288 training games), 11,264
replayed evaluation games. No qualified plateau (one of six final-window
intervals within ±5). Held-out GRU minus pooling: −1.95 points [−5.88, +1.98],
neither advantage nor equivalence; no promotion. GRU and pooling are two
within-decision reducers, not agent memory.

[V7 state reduction, CPU](reports/state-representation-v7.md): **preliminary,
2026-11-04** — four seeds at 128 updates, 32,768 training games and 1,024
replayed evaluation games. GRU minus pooling: −0.98 points [−10.13, +8.18], at
1.06× training time. Too short a horizon to choose an architecture; the GPU
study above is the answer.

[Growing card pools, MTG-245](https://github.com/loopflowstudio/manabot/blob/ac601834d/experiments/protocols/growing-pool-245-v1.md): **removed
unexecuted, 2026-11-11** — authored reference league, frozen mixtures and
separate retention, expansion and generalization panels. No training or
strength results; the protocol and modules remain at that revision.

[Heuristic strength](reports/heuristic-strength.md): **completed 2026-10-24** — the
`heuristic` anchor as UR against MTG-183's update-1,750 GW rises from 18.3% to
62.0% (300 held-out games), and from 10.8% to 34.3% across all ten decks, with
faster decisions. Flat Monte Carlo over it adds +6.3 points [0.0, 12.7] at 16
worlds, for about 3,000 times the decision time.

[MTG-165 two-seat scale ladder](reports/two-seat-ladder.md): **running, interim report** —
20.9M learner/environment decisions retained in a verified 704-update snapshot.
Early heuristic score at 250 updates: 51.5% [48.5, 54.5], versus one-seat 47.5%;
paired delta +4 points [−1, +8.5], with unequal training cost. The $50 endpoint
is pending; $4.01 estimated incurred at the retained cost observation.

[ETU-105 mini filter scope](reports/ataraxos-technique-screen.md#mini-result--completed-2026-10-07):
actor-only gained +3.33 [−4.33, 11.33] points against held-out greedy at 45% extra
training cost; unpromoted. All 3,600 monitoring/final games replayed exactly.

[ETU-106 pooling × advantage floor](reports/pooling-filter-followup.md#result--completed-2026-10-06):
2,400 replay-verified games; no demonstrated token or floor-removal strength
benefit, no equivalence claim, and both floor changes remain unresolved.

ETU-99's [remaining Ataraxos transfer protocols](reports/ataraxos-transfer.md) map
delivered instruments to unexecuted sampler, search, distillation and robustness
studies. Proposed budgets are separate from ETU-91 and grant no execution authority.

| exp | question | verdict |
|---|---|---|
| [00](reports/exp-00-decision-profile.md) / [00-cost](reports/exp-00-cost-basis.md) / [00c](reports/exp-00c-seat-balanced-baselines.md) | calibrate the instrument | 194 decisions/game; single-init baselines meaningless; $0.44/1M steps. ~~Per-seat findings (94%/23.1% on-play)~~ deal artifacts per exp-06 — deal-averaged random mirrors are at parity (repro_06) |
| [01](reports/exp-01-c1-training.md) | does the shaped recipe survive a real deck? | no — 0/3 seeds; one seat-parasitic; `cast_when_able` flipped sign |
| [02](reports/exp-02-flat-mc.md) | how much intelligence is free? | search-256 = 99% vs random at $0; every trained policy below N=16 |
| [03](reports/exp-03-distillation.md) | is distillation cheaper than RL? | yes — 90.5% vs matched-cost PPO's 52.7%; ladder ≈ N=8 at 1ms |
| [04](reports/exp-04-potential-shaping.md) | is bias-free dense signal possible? | shaping was the disease — terminal-only wins; pass-collapse fails replication |
| [07](reports/exp-07-expert-iteration.md) | does the expert-iteration crank compound? | no — R1 loses 74–26 to R0 (label economics); batched inference 12× landed |
| [06](reports/exp-06-newworld-training.md) | did observation growth hurt training? | benign — seeds in/above the historical band |
| [08](reports/exp-08-two-deck-matchup.md) / [08b](reports/exp-08b-ancestral-dose.md) | what decides a matchup table? | pilot + card quality — 4× Recall swung it 55 points; UR-22% was a pilot artifact |
| [09](reports/exp-09-control-competency.md) | can the pilot play control? | no — ≤0.39 correct at any N; random beats search on 3/5 scenarios; win rate masks incapacity |
| [10](reports/exp-10-value-gate.md) | does search-with-V beat V-greedy? | gate passes deal-diverse 60.25% [55.4, 64.9], but V loses to cheaper random-rollout search (P3 refuted); V's ordering near-noise in undecided positions; first battery + training corpus (3 deals/225g) were deal-narrow |
| [11](reports/exp-11-curriculum-exploitability.md) | does a stronger opponent teach better? | self-play quietly wins; the opponent installs the strategy; student robust (exploiter ≤26%) |
| [W2-214](reports/w2-214-structural-semantic-katas.md) | can a bounded structural encoder break the bag encoder's exact symmetry on five static semantic relations? | instrument valid; bag exactly 50%; relational arm learned order/hierarchy but failed the pre-registered trainability and cost gates — redesign optimization/capacity before interpreting the remaining katas |
| [ETU-87](reports/etu-87-allies-lessons-balance.md) | which cards decide Allies versus Lessons? | exploratory — the 70–30 gap is in the lists (Random mirrors reproduce it); cutting White Lotus Reinforcements ×2 and Suki for lands gives UR Lessons 47.7% [45, 51] on w3; the revised Lessons list is worth ~13 points on w4 |

### Teacher program (W2-234)

| report | question | verdict |
|---|---|---|
| [teacher0](reports/w2-234-teacher0.md) | can a search-supervised policy/value student learn from Teacher-0? | proceed to scale-up — joint policy/value matches policy-only CE/top-1, improves held-out value Brier 0.2536→0.2043, 71%→73% vs random |
| [teacher1](reports/w2-234-teacher1.md) | does the determinized PUCT substrate work end-to-end? | substrate pass — tree search, node reuse, alternating backup, self-play shards all function; priors still uniform, playouts random |
| [teacher1-pilot](reports/w2-234-teacher1-pilot.md) | do budgets 8/32/128 give strong, legal, affordable targets? | **preregistered, unrun** — fail-closed until the terminal Teacher-0 manifest and control lock exist |
| [INT-4 smoke](reports/int-4-visit-teacher-smoke.md) | does the visit teacher → 2×2 students → four-agent arena → Study pipeline execute on real states? | engineering substrate pass, **not admission evidence** — 507 labels, exact 175-decision replay, four matched checkpoints, complete smoke arena, and cross-language Study validation; production controls and competencies remain |
| [INT-4 production harness](reports/int-4-visit-teacher-production.md) | can the registered multi-seed iteration fail closed on exact controls and produce independently verified admission evidence? | harness ready, **production not run** — exact frozen Teacher-0 bytes are absent; no substitution or production claim |
| [INT-8 student signal guidance](reports/int-8-student-signal-guidance.md) | should the recovered chosen-action or visit-distribution policy guide bounded PUCT? | `kill_retained_smoke_policy_guidance` — both arms lost paired score to uniform, added inference cost, and improved uniform-128 agreement by <0.05; one-seed smoke evidence only |

### Platform evidence (W2)

| report | claim proven |
|---|---|
| [card-conformance-audit](reports/card-conformance-audit.md) | 55 registered cards audited vs Scryfall; 20 shell mismatches fixed; fixture + tripwire test |
| [milestone-1](reports/milestone-1-two-deck-slice.md) | all four rules stages landed for the UR-Lessons-vs-GW-Allies slice |
| [opcode-alignment-v1](reports/opcode-alignment-v1.md) | semantic-program-only encoder: 91.7% zero-shot held-out cluster accuracy vs 0.0% for opaque card-id arms |
| [structured-policy-decoder](reports/structured-policy-decoder.md) | structured decoder matches legacy adapter exactly (6435/6435 actions) at comparable latency — migration evidence |
| [INT-11 semantic runtime policy](reports/int-11-semantic-runtime-policy.md) | learned priority/target/combat policy executes 144 legal Commands and 36/36 terminal paired games across three seeds; identity transfer beats identity-only but matches structure-shuffled, so structure evidence is null/ambiguous |
| [INT-12 belief strategy advisor](reports/int-12-belief-strategy-advisor.md) | one pinned exact-range comparison returns byte-identical live/Study advice with a 0.125 policy delta; cached p50 0.984 ms and fresh p50 16.981 s for the declared 16-simulation engineering profile, with no strength or service-SLO claim |
| [INT-17 belief calibration](reports/int-17-belief-calibration.md) | exact frozen run retained a fail-closed systems result: per-row materialization re-enumerates the full support, making likelihood updates O(S²); optimistic full-trace projection 11.55h exceeded the 6h cap, so no curves or calibration claim exist pending RUL-13 and a new execution contract that preserves v1's scientific inputs while binding the repaired runtime |
| [w2-179](reports/w2-179-content-pack-local-diagnostic.md) | immutable ContentPack / mutable Card seam with deterministic CardDefId hashing |
| [w2-182](reports/w2-182-search-branching-v1.md) | full-clone branching baseline: 2,194 sims/s, clone p50 4.5µs |
| [w2-198](reports/w2-198-compact-clone-undo-v1.md) | compact clone+undo at parity with full clone (2,097 sims/s) with journal accounting |
| [w2-208](reports/w2-208-content-pack-clone-allocations.md) | exact clones share the immutable ContentPack by Arc pointer |
| [w2-215](reports/w2-215-semantic-projection.md) | viewer projection baseline: ~53 tokens/object p50, ~9.9µs hot projection, zero failures |
| [w2-223](reports/w2-223-typed-ir-interpreter.md) | generic fail-closed interpreter executes all 31 admitted programs by opcode — no card-name dispatch |

(Older platform docs: [first-light-run-1](reports/first-light-run-1.md),
[sps-closeout](reports/sps-closeout.md).)

Distributed-RL research and the unexecuted bounded workload plan live in
[the ETU-108 report](../docs/training.md#distributed-training); its benchmark supervisor was
removed unexecuted and remains in git history.

The [GPU representation horizon study](protocols/state-representation-5da5f4-gpu.md)
retains four paired CUDA seeds, initialization and periodic evaluation, a
sustained-plateau criterion, and separate held-out scoring. Its first cohort
was cancelled before scoring when measured concurrent GPU throughput invalidated
the allocation budget. `state_representation_gpu.py` declared one recovery unit
per frozen allocation file; the throughput admission and CUDA operator profiler
that qualified it ended with the cohort and remain at source `d57c5c9e7`,
reachable through PR 421's head ref (`refs/pull/421/head`). Wider
batches change optimization; the prospective recipe retains the original
complete-game horizon. Scientific Run `e51f61b92931` ended `lost / machine_lost`
on 2026-11-05 after three of eight learners completed, with no curve or held-out
endpoint; that record is unchanged. The recovery amendment completed the cohort
the same day: [the GPU report](reports/state-representation-gpu.md) holds its result.
The preliminary CPU study remains frozen.
