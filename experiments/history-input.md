# ETU-106: bounded identity-aware recent-history screen

Prepared 2026-10-06 for Jack Heart. The original prospective protocol is retained
below. Jack Heart authorized review and autonomous software delivery; no scientific
campaign has launched. PR241 and all completed campaign artifacts remain unchanged. The six-hour allocation described
here is an execution contract, not evidence that calibration or admission passed.

## Software preparation and next commands

`history_input.py` derives the two scalar/token recipes and admits cost-only
calibration. `run_history_input.py` supervises preflight, two calibration jobs,
six training jobs, the existing twelve-cell arena path and saved analysis.
`history_input_analysis.py` reports paired on-minus-off effects, common-cost and
exposure selections, inference accounting and incomplete-cohort suppression.
Older study protocols retain their original restrictions and serialized fields.

The parent owns process-group deadlines and disk stops. Internal phase commands
require its live PID; they are not independent execution or retry entry points.
A new output directory and checkout lock retain every attempt. Calibration is
charged once; training and evaluation cannot resume. Report-only recovery is not
a public standalone command: a failed campaign stops and retains its evidence.
The current implementation takes the simpler no-recovery path allowed by the
protocol; a later recovery requires a separately reviewed remaining-time receipt.

Combined fixture validation passed 50 checks with one unsupported-configuration
skip; CI invocation collection passed all 376 tests without module-name collisions.
Lint, formatting and diff checks passed. No optimizer or scientific cohort ran:

```bash
uv run pytest tests/training/test_history_input.py tests/training/test_pooling_filter.py tests/training/test_value_screen.py -q
uv run pytest tests/model/test_recent_events.py tests/training/test_history_arena.py -q
```

The source-matched native wheel was built under `.runs/history-native-build`.
The previous extension remains at
`.runs/history-native-build/retained-extension.cpython-312-darwin.so`; the installed
extension SHA-256 is `9e3be349db2e65780e5ee32061f0677e255357d5776bef77fcb234b6d90da677`.
Recent-event fixtures passed nine checks with one unsupported-configuration skip;
the ordinary arena fixture passed all eight fixed-weight replay-verified games.
Model construction confirms totals 138,498 off / 161,410 on and the 22,912 increment.
These are software checks, not calibration or throughput measurements. Retained
campaign bytes and their runtime receipts are unchanged.

After source delivery and the exclusive-host/external-cohort audit below, the
complete entry point performs prospective cost admission before scientific work:

```bash
uv run python -m experiments.runners.run_history_input --campaign <exact-clean-delivered-commit> --out .runs/etu106-history-input
```

This command **starts calibration and, if admitted, scientific work**; it was not
run in this contribution. It checks local frozen plans/attempts for seed collisions,
not other machines or checkouts. The launch owner must establish exclusive host
use and check external proposed/retained cohorts before admission. Current-runtime
calibration rate, admitted N, disk projection and finalized source/runtime/host
identities remain unknown. Preflight runs native history fixtures (including eight
fixed-weight arena games on fixture deal 971811) inside the charged 1,800 seconds;
these are separate from the 1,200 scientific games. Host resource receipts report
the inherited unenforced 32 GiB declaration explicitly.

## Question, scope and retained baseline

Does the shipped identity-aware public recent-event input improve early learning over the same retained value-token policy without history, at fixed collected transitions and within overlapping observed training cost?

Use two arms, `history-off` and `history-on`, each at three independent paired training seeds. This is the first small-capacity history contrast. A second capacity point, scalar/WDL comparison, normalization change, compound decoder, recurrence, sampler work or masked-history capacity control would require a separate allocation. Six runs cannot also estimate a capacity × history interaction. Do not describe this screen as that factorial or as completion of ETU-106.

Retain **value_token, scalar, width 64, one attention layer, four heads, existing post-normalization, min_advantage=0.01**. Derive the complete baseline from `experiments/regimes/value-model-baseline-v1.json`, applying the existing `with_value_aggregation(..., aggregation="value_token")`; set the bounded workload and watchdogs below, then use `Experiment`/`Case`/`Model(AgentSpec(recent_events=...))` or `with_recent_events` for the sole arm contrast. Freeze the complete resolved settings, not just this list.

This is the token/.01 arm retained from the previous studies and Jack Heart's selected implementation direction. It is not a promoted winner. The scalar token screen and pooling/floor follow-up left both architecture and floor effects unresolved. Endpoint scores and common-cost orderings differed; neither fixed-update endpoint ordering establishes equal-cost superiority. Keeping the original floor avoids selecting another learning rule from those inconclusive results. Filtering remains an explicit possible mediator of any history effect. Do not silently substitute the two-layer WDL `ataraxos-mtg-v1` preset: it changes depth, output and other recipe provenance.

Controls, identical in both arms:

- Ordinary semantic Allies/Lessons policy over authored decks and sideboards; current-self opponent, no frozen-opponent or search training; unchanged legal action coverage and semantic catalog.
- `ataraxos_move`, Adam, policy/value jointly trainable, scalar target/loss dispatch; inclusive 0.75 advantage quantile, minimum 0.01, `actor_critic` filtering, policy lambda 0.5, value lambda 0.8, clip 0.2, collection KL 0.1, action-type-uniform reference, max gradient norm 0.267.
- Learning-rate max/min 0.0001/0.000005, scale 0.5, power 1.1; tau scale 0.05/power 0.3; EMA 0.999 maintained and exported but **raw only evaluated**. Retain the move rule's iteration schedules and continuation clock, including empty-filter updates; do not tune schedules to the admitted update count. Preserve the baseline's serialized schedule setting and every resolved default.
- Four streams × 64 learner transitions per stream = 256 learner transitions per update. Two linked equal-update stages; `policy-1.initial="policy-0"` carries learner, Adam, EMA, collector and iteration. No reset/retraining at midpoint. No process training resume is used for this screen.
- CPU float32, one worker, one Torch thread, OMP/MKL threads one, no paid compute or concurrent campaign. Read actual laptop RAM; retain the inherited resource declaration only with its limitation explicit (the 32 GiB field is not an enforced memory cap). Monitor RSS and free disk.
- Observation capacity: 60 cards/player, 40 permanents/player, 64 actions, two focus objects, 32 event rows. No belief inputs or compound decisions. A capacity/legality failure fails the cohort instead of clipping offers or silently expanding one arm.

The arm-diff validator must allow only recipe ID, `agent.recent_events` and its derived `observation.policy_history_version` to differ. Calibrated updates and watchdogs are common controls. All other fields must compare equal after full validation.

## What history changes

The current shipped contract is `docs/recent-events.md`, `manabot/model/recent_events.py`, `Agent` and native policy history. It supersedes the older section-6 sketch in `docs/plans/modular-architecture-recipes.md`: history is not a stack of raw allocation IDs or historical controller categories appended as attention tokens.

History-on selects native public suffix v1: at most 32 admitted rows, shape `[32,12]` plus validity, versus historical `[32,7]` on history-off. The old event tensor is not consumed by the off model. Rows encode arrival, damage, life change, spell cast/resolved/countered, amount, relative ordinal age, and source/target public definition, owner, zone and exact current-context availability. Definition IDs route into the existing semantic catalog; physical/context IDs route visible references and are not learned numbers. Zone changes expire old incarnation links. Departed public identities can survive without current-object links. Native emission, fork/undo and reset own the suffix; no private world or reconstructed historical truth enters inference.

The encoder projects each row, mean-pools valid rows, and adds context to shared object representations; policy and value both receive it. It is stateless learned encoding of a bounded history window, not learned recurrent game memory or a complete history. Passes, private draws/reveals, historical controller changes, exact time gaps and identity across departed duplicate copies remain absent. Do not explain poor control play by these omissions without evidence.

Two inseparable treatment changes must accompany every result:

1. Additional public information and a different observation ABI: policy-history version 0 versus 1, with different event shape/meaning. World/content/setup and action ABI must match, while observation identities are intentionally distinct.
2. Additional trainable capacity/computation: at width 64, the shipped `Linear(292,64) → Tanh → Linear(64,64)` adds **22,912 parameters** (source-derived, to verify against ordinary architecture receipts before launch). Record total/trainable/component counts for both full models, native history/encoding cost, collection and inference cost. There is no capacity-matched or information-scrambled third arm.

The native bounded history cache is maintained on both arms. Timing compares model input/encoding and downstream behavior costs on the same runtime, not the cost of adding the native cache itself. Common model parameters are constructed before optional history parameters; verify their initial bytes match under each paired seed. Added initialization consumes RNG draws, so equal initial shared weights do not promise identical action trajectories or globally identical RNG state. Preserve existing RNG ownership; log this limitation instead of inventing a new initialization framework.

Prospective prediction: the absolute mean endpoint history effect will be less than five percentage points at this bounded workload, while history-on will cost more per collected update. This is an uncertain numerical hypothesis, not a no-effect assumption or equivalence margin supported by power analysis.

## Six-hour total allocation

One laptop allocation, **21,600 seconds maximum for everything from campaign preflight through final report/recovery**. Calibration is inside it. Phase caps are ceilings, not promises or independent allocations; unused time is not borrowed to add updates or games.

| Phase | Maximum seconds | Included work |
| --- | ---: | --- |
| Preflight and cost-only calibration | 1,800 | Identity/storage checks, two calibration jobs, construction, collection/update, both exports/reloads, calibration receipts and frozen plan |
| Six scientific training runs | 14,400 | All setup, training, two raw/EMA exports per run, failures and orchestration |
| Twelve evaluation cells | 4,500 | 100 games each, registration/loading, command traces, exact replay and row persistence |
| Report and permitted recovery | 900 | Offline analysis/notebook, hash verification, failure accounting and report-only recovery |
| **Total** | **21,600** | No six-hour training allowance plus additional evaluation |

Enforce a parent monotonic deadline starting before preflight, with whole child process-group termination outside Python/native calls. Each phase is also capped by the original total remaining time. Training children: at most 2,400 seconds each; stages at most 1,190 seconds each. Six run caps exhaust the training envelope, so parent orchestration must fit within that same 14,400 seconds. Do not treat watchdog truncation as successful fixed-count training. Evaluation games retain a 120-second/10,000-command ceiling; the parent 4,500-second evaluation ceiling overrides it. Slow games may therefore fail the planned cohort; they never reduce cell size. Reporting and shutdown must start with their reserve intact.

The old scalar screen spent 3,224.35 seconds on 1,800 evaluation games. Scaling to 1,200 games with 50% headroom gives about 3,224 seconds, motivating the 4,500-second reserve. This is prior-world/implementation evidence for a conservative proposal, **not measured history-on throughput or admission calibration**. Do not calibrate by looking at scores, add pilot scored games, or claim the reserve guarantees completion.

Require conservative projected evidence storage and 4 GiB free reserve. Start with a 3 GiB evidence allowance plus the 4 GiB reserve, then use observed calibration artifact sizes and retained prior trace sizes to check that 24 scientific raw/EMA checkpoint files, calibration artifacts, 1,200 command traces, SQLite, logs and reports fit. Increase the projection before admission if needed; do not discard artifacts to fit. Preserve original absolute paths and a hash manifest. Native/Torch peak memory is not established by Python allocation measurements.

## Prospective update admission: costs only

Calibration has **two separate non-scientific timing jobs**, one per arm at seed 10630, each exactly 40 updates in linked 20/20 stages, a 400-second run cap and 190-second stage caps. These are additional calibration attempts charged to the same six-hour total, not extra scientific seeds. Neither calibration checkpoint is evaluated or continued into a scientific run. No games against greedy occur in calibration. All calibration attempts/failures remain retained; failure stops admission without retry.

Measure complete calibration elapsed time C, per-job full process seconds (including startup, construction, exports/reloads), per-stage costs, actual update counts, collected transitions, optimizer exposures, empty-filter skips, memory and artifacts. Use the slowest full-process seconds/update across both arms; conservatively also include any slower stage seconds/update:

`r = max(off_process_seconds/40, on_process_seconds/40, each_stage_seconds/20)`.

Freeze `N = min(800, 100 * floor(14400 / (6 * 1.25 * r * 100)))` total updates per scientific run. The 1.25 factor supplies 25% timing headroom; use full-process rates rather than gradient-only speed or historical token rates. Require C <= 1,800 and N >= 400, otherwise stop as infeasible. Thus N is one of 400/500/600/700/800, with checkpoint counts N/2 and N. The 400-update minimum is an explicit minimum learning exposure for this sparse screen (half the original 800-update study), not a power guarantee. At N=800 each run collects 204,800 transitions; at N=400, 102,400.

Freeze the source/identities, calibration receipt hashes, recipes, seeds/order, evaluation schedule, N and deadlines before scientific training or scoring. Do not increase N if later runs are fast. Check remaining-work feasibility after each run and at a two-hour campaign progress checkpoint using elapsed time and the slowest available seconds/update with 25% headroom, against the **remaining training envelope**, preserving evaluation/report reserves. No old hard-coded 21,600-second training threshold applies. Timing projection, disk exhaustion, invalid identities, nonfinite parameters, legality or replay failures may stop the entire cohort. Scores may never stop or extend it. A stopped or partial cohort has no complete comparative result.

## Seeds, assignments and complete evaluation cohort

Scientific training seeds: **10631, 10632, 10633**, paired across off/on. Order by seed: **off/on, on/off, off/on**. Three pairs cannot perfectly balance order; record timestamps and host load and acknowledge the remaining order confound. Exactly six scientific runs, no replacement seeds or best-checkpoint selection.

Retain existing executor stream derivation: initialization = seed; collection = seed+10000; minibatches = seed+20000; evaluation stream receipt = seed+30000 (not the arena's deal identity). The seat-routed collector seeds the native vector environment from collection and its self-action generator from collection+1; native stream/reset RNG derivation must be pinned by source/runtime. Calibration uses its separate seed/streams. These are deterministic PRNG families, not a claim of mathematical independence or identical self-play deals across diverging policies. Native auto-reset deals are generated from its RNG, not `base_seed+episode_number`; record the actual native derivation and assert family separation at preflight. Numeric training seeds below 100000 and reserved arena seeds alone cannot prove no physical deal coincidence.

Evaluation deals: **961260–961284 inclusive**, 25 fixed fresh blocks, reused across both arms, all three seeds and both checkpoints. No direct arm-vs-arm games, additional anchors or EMA evaluations. Every raw checkpoint plays all four deck/seat legs against the one source-pinned `scripted-greedy-fixed-anchor`: each policy gets both decks on play and draw. Total = 2 arms × 3 seeds × 2 checkpoints × 25 deals × 4 legs = **12 cells / 1,200 games**.

Use existing `arena-pair-deal-player-v1` derivation and candidate/reference aliases consistently across all cells. Keep the ArenaKey and anchor registration fixed: it enters the RNG hash. Do not regenerate an anchor cohort separately for history-on, which would change action seed families. Candidate registrations retain their own actual observation ABI and checkpoint identity; the code-only greedy anchor retains its actual fixed input declaration. Full semantic Commands and trace replay remain authoritative. A history-on checkpoint must reload its own history observation contract through the ordinary player, not inherit the off arena observation tensor.

Before freezing, check chosen IDs against retained local campaign plans, calibration families and new proposed studies. An existing collision discovered before any run requires a documented prospective amendment, not a replacement seed after seeing outcomes. Keep the paired-deal set out of all training, tuning and timing work. It is held out from optimization, but reused at both evaluation checkpoints and therefore not an untouched endpoint confirmation cohort. No human-play or general-strength claim follows.

Run all training before arena scoring. Midpoint/endpoint mean scores are measurements at fixed collected transitions, not cost-matched checkpoints. A trained history treatment also changes self-play opponents and visited states; the target is the complete coupled learning recipe, not solely inference under a frozen opponent distribution.

## Analysis fixed before scoring

For each seed, checkpoint and deal, average the four leg scores (win=1, draw=.5, loss=0). Subtract **history-on minus history-off within the same seed/deal** before aggregation. Retain all per-seed scores/differences. Primary endpoint is the mean paired difference across three training seeds and 25 common deal blocks at N updates. Midpoint is diagnostic, never selection. Report 95% percentile intervals with 10,000 NumPy default_rng bootstrap draws, seed **10634**, resampling the three seed indexes and 25 deal indexes with replacement and applying the same indexes to both arms; keep all four legs together. The experimental unit for method uncertainty is the training seed. These small-cohort intervals are descriptive; they do not turn 1,200 games into 1,200 training replicates or establish equivalence.

Use the existing cost comparison on the full completed six-run cohort, named greedy anchor, common observed support `[max(first checkpoint costs), min(last checkpoint costs)]`. At its upper endpoint, select the last available checkpoint for each run, then compute the paired effect from the selected saved game rows. Select by cost only; no interpolation, extrapolation, nearest-future checkpoint or score-dependent cutoff. Report both endpoints, selected checkpoint hashes and the common-cost qualification next to the primary result. If support is empty, say unavailable. Two points per run cannot resolve detailed learning-speed dynamics; step curves are a last-observed convention, not measurements between checkpoints.

Retain per-update diagnostic curves for losses, signed/absolute advantages, selection counts, empty skips and learning settings; per-checkpoint score curves versus updates, learner transitions, native decisions, cumulative training hours and optimizer exposures. Use the same overlapping-support/last-available rule for exposure comparisons, labelling them descriptive post-treatment accounting, not exposure-matched causal estimates. Report actor/critic exposures separately, collection/learning/export time, failed attempt cost, inference seconds/decision, decision counts, observed RSS and contention. Recorded value loss is not automatically a full-update MSE; terminal-distance-censored rows are not known distant outcomes. Do not explain a history effect causally by lower exposure alone.

Predeclared disposition: history merits a larger confirmatory allocation only if endpoint mean gain >=5 percentage points, every seed difference >=0, the overlapping-cost mean difference is positive, all scheduled games terminate/replay and pooled history-on inference seconds/decision <=1.25×off. If all three endpoint differences are negative and their mean <=−5 points, reject history for this recipe/budget. Otherwise unresolved, including unavailable common-cost comparison. These are follow-up priorities, not automatic model defaults, product admission or a claim of significant general superiority. One CPU thread does not match inference FLOPs; changed decision distributions confound pooled serving latency.

If a run/game fails, retain partial diagnostics and all attempted rows, and suppress the complete paired effect and promotion/rejection decision. Missing games are not draws or losses; report missingness and bounds separately. Never drop an inconvenient seed, replay a failed cell as a replacement, or recompute a different cohort after scores. Report-only recovery may regenerate analysis from immutable complete evidence inside the remaining 900-second/overall allowance. No training or evaluation resume/retry is planned. After timeout only preserve failure evidence; further compute requires another explicit allocation.

## Source, native and artifact freeze

This planning inspection used clean HEAD `60e2f8977dfee95979c42c1a776a1472ca8ebdf8`; it is **not** the future launch source freeze. The local file `managym/_managym.cpython-312-darwin.so` hashes to `808c1da98358e592767fad34046a9ac8c355a5772fa4e899691a71a39684f1c0`. This is a file hash only; no native capability or source/build match was exercised in preparation. Baseline file SHA-256: `fe4a658a0fa32df6cff1cb587564c27de5370bacc6e96b6e032b4a34b74a74bc`. Original screen-plan file SHA-256: `7b4e0d1d19b9ae2e89721dda9d077a3979f69d95771877228def9cf2309a24eb`. Do not relabel old checkpoints or reuse their calibration as current-source measurements.

Future admission must freeze an exact clean, delivered source commit containing the minimum plumbing below, after existing evidence delivery. Use the actual imported extension path/hash, native world and rules schemas, engine source bundle, content manifest/catalog, authored setup/sideboards/Lesson pool, action ABI and **each arm's observation ABI and input schema/history version**. Resolve fingerprints with the actual recipe match and ObservationSpace, never the helper's default Interactive mirror. Bind all manabot Python sources, study runner/analysis/notebook sources, lockfile and pyproject, Torch/NumPy/Python versions, platform/CPU/RAM, device/thread settings, and experiment provenance plus resolved regime digests. Record architecture identity/component counts and ordinary checkpoint world binding for both arms. Different history ABIs must be explicit in the plan, not hidden behind one run's identity.

Before launch, native history presence and meaning must agree with the source through focused fixture checks. The verified rebuild above precedes calibration. Do not modify source or the imported extension during calibration/training. Any implementation/native change after calibration invalidates count admission; there is no automatic recalibration/restart within this proposal. Preserve all costs and request a separately reviewed continuation if needed.

Evidence owners stay TrainingRun, VerifyStore, ordinary raw/EMA artifacts, PlayerRegistration and arena traces. Freeze source/data/action RNG provenance, all attempts, hash-bound calibration and plans, run JSON, checkpoint bytes, arena rows, notebook, machine-readable paired effects and human-readable report. Reproducibility means the bounded recipe and evidence can be replayed/audited on pinned identities; stochastic training is not promised byte-identical.

## Delivered scope and remaining admission

The scoped `history-input` branch of EvaluationProtocol/ResolvedStudy fixes the
two recipes, paired seeds/deals, raw two-checkpoint cohort and six-hour budget.
Typed per-arm bindings preserve distinct observation contracts while shared
world/content/action/source identities must match. The supervisor owns calibration,
count admission, sequential execution, deadlines and failed-attempt retention;
existing TrainingRun/VerifyStore and arena/reporting remain the evidence owners.
Saved-row analysis and the notebook include paired effects and common-cost/exposure
selections. No model or trainer changes are introduced.

Focused fixtures cover recipe/identity drift, count admission, phase accounting,
process-group cleanup, incomplete evidence, native history/reset/information safety,
ordinary checkpoint reload and full-game replay. Prospective calibration rate,
admitted N, actual launch-host resource envelope, disk projection and finalized
source/runtime identities remain unknown. Exclusive-host and external-cohort seed
checks belong to the launch owner. This contribution delivers software only;
calibration and the scientific campaign have not run.

## Sources inspected for this proposal

Repository-relative paths refer to the planning HEAD above: `docs/recent-events.md`; relevant history/initialization and accepted-delivery sections of `docs/plans/modular-architecture-recipes.md`; `docs/training-experiments.md`; `manabot/README.md`; `experiments/value-token-screen.md`; `experiments/pooling-filter-followup.md`; `experiments/regimes/value-model-baseline-v1.json`; `manabot/model/recent_events.py`; history construction in `manabot/model/agent.py`; recipe/history admission in `manabot/training/models.py` and `recipes.py`; runtime/seed/self-play construction in `manabot/training/execution.py` and `sim/net_opponent.py`; `runtime_fingerprints` in `sim/teacher1_evidence.py`; complete `experiments/runners/training_protocol.py`, `run_training_regimes.py`, `pooling_filter.py`, `run_pooling_filter.py`, `run_value_models.py`, `run_value_screen.py`, `value_screen.py`, `pooling_filter_analysis.py`; cost/uncertainty/report paths in `manabot/training/analysis.py`; `experiments/study/training-regimes.ipynb`; selected-match and seed/leg paths in `manabot/arena/match.py`; and relevant `tests/model/test_recent_events.py` fixtures. Published-paper architectural motivation remains the dated source review in the existing design/fidelity documents; no new external research or exact-reproduction claim is made here.
