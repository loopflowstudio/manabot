# Evaluating manabots

Evaluation measures strength from outside the training loop, and its scores
never reach the loss. The unit of measurement is a deal, two players are
compared on the same deals from both seats with both decks, and only finished
games that replay exactly count. This guide covers that contract first, then
the anchors, the bounded ways we probe for weaknesses, and the tools.
Training-time monitoring that calls this machinery is in
[training.md](training.md#monitoring); reading reports is in
[experiment-metrics.md](experiment-metrics.md).

- [One deal](#one-deal)
- [What counts](#what-counts)
- [Anchors and ratings](#anchors-and-ratings)
- [Beyond win rate](#beyond-win-rate)
- [What we know](#what-we-know)
- [Quick local evaluation](#quick-local-evaluation)
- [Etude agents](#etude-agents)
- [Continuing an evaluation](#continuing-an-evaluation)
- [Performance evidence and calibration](#performance-evidence-and-calibration)
- [Card data](#card-data)
- [Checkpoint scenarios](#checkpoint-scenarios)
- [Frozen-policy attacks](#frozen-policy-attacks)
- [Open direction](#open-direction)
- [Ownership](#ownership)

## One deal

A deal is one seeded shuffle. Two players meet on it four times, swapping decks
and seats, so the luck of the draw and the advantage of going first cancel in
the comparison. With mirrors, each player also pilots each deck against itself,
for eight legs.

![A seeded deal is played four times between two players, swapping decks and seats, and only finished, replay-verified games are scored, with uncertainty from resampling whole deals.](images/evaluation-deal.svg)

The seed of every game is derived from the arena, the pair, the deal and the
player, so the same deal can be replayed or continued exactly.

From `derive_seed` in [`manabot/eval/match.py`](../manabot/eval/match.py):

```python
identity = canonical_sha256(
    {
        "arena_key": seed_key,
        "pair": sorted(aliases.get(member, member) for member in pair),
        "deal_seed": deal_seed,
        "player_id": aliases.get(player_id, player_id),
    }
)
return int(identity[:16], 16)
```

- Legs of one deal are not independent samples. Uncertainty is a paired-deal
  bootstrap over whole deals, and a single deal has none.
- Repeating a deal seed is not a new sample.
- Monitoring plays reserved development deals that scientific cohorts exclude,
  and held-out deals never reach training. A run's selection
  rule is `last-complete-raw`, never the best evaluation, so no result is
  picked by looking at the test.

## What counts

Only terminal, replay-verified games score, and a terminal draw scores one
half. Crashes, timeouts and failed replays are retained by outcome and never
earn credit, and a cohort with an unscored game is rejected rather than quietly
trimmed.

From `fit_population` in [`manabot/eval/rating.py`](../manabot/eval/rating.py):

```python
retained = [dict(row) for row in rows]
if any(row.get("score_a") is None or row.get("truncated") for row in retained):
    raise ValueError("unscored failures reject the cohort; do not drop attempts")
```

Uncertainty resamples deals, not games. From `bootstrap_population` in the
same module:

```python
for _ in range(replicates):
    selected = rng.choice(blocks, size=len(blocks), replace=True)
    counts = {block: int(np.count_nonzero(selected == block)) for block in blocks}
    weights = np.asarray(
        [counts[int(block)] for block in row_blocks], dtype=np.float64
    )
```

- A replay that doesn't finish within the deadline is unavailable, not a
  mismatch. Replay verifies Commands, not that two devices compute the same
  policy.
- An infrastructure failure requests an Etude retry. It is never a player's
  forfeit; invalid native games keep null scores and their failure evidence.
- Game-level intervals describe one checkpoint's evaluation noise. A claim
  about a training method needs independent training seeds
  ([experiments](experiments.md#how-a-claim-gets-made)).

## Anchors and ratings

Working anchors are the rule-based `heuristic` player and frozen checkpoints.
Select `--opponent heuristic` for local evaluation or
`MonitorProtocol(opponent="heuristic")` for checkpoint monitoring. The scalar
adapter and batched opponent share the
[native heuristic rules](engine.md#built-in-heuristic).
New arena registrations use `version="native-v2"` and bind the native rule
sources and adapter as their source identity. The `managym` wheel includes engine
sources at their original package paths, so installed registration and validation hash
the same paths and bytes as a checkout. Results recorded against the Python
heuristic retain their original identity.
A changed anchor identity needs a new version, never a relabeling of old games.
The code anchors `scripted_greedy` and `random`
(`manabot.sim.anchors`, frozen v1 sources) remain admissible for compatibility
and diagnostics, not as working anchors; `fit_population` still places
`random-v1` at the origin by default. Head-to-head
[tournaments](../experiments/reports/head-to-head.md) play chosen checkpoints
against each other on shared deals.

Ratings are a seat-aware Bradley-Terry fit expressed on the Elo scale
(`eval.rating`). A rating belongs to one tuple of world, content suite,
information boundary, compute class and arena version, and never crosses those
boundaries; one anchor fixes the origin, so differences carry meaning and the
absolute number doesn't. The full contract, including why the payoff matrix
stays visible and why search players sit in their own compute class, is the
[skill rating contract](../manabot/RESEARCH.md#skill-rating-contract).

## Beyond win rate

A win rate can rise while a specific skill is missing, so two bounded probes
sit beside it:

- **[Checkpoint scenarios](#checkpoint-scenarios)** score retained tactical
  positions S1–S5, such as holding a counterspell for the bomb or waiting a
  turn to wipe more creatures. They are local behaviors against a scripted
  continuation, not strength, and must not be pooled with it.
- **[Frozen-policy attacks](#frozen-policy-attacks)** train an attacker against
  a frozen checkpoint under a declared budget and prediction. A failed attack
  never certifies equilibrium; the result is a lower bound on exploitability
  from the attacks we can afford.

## What we know

- **The anchor itself moves.** Revising the `heuristic` player on 2026-10-24
  raised it from 18.3% to 62.0% as UR against MTG-183's update-1,750 GW over
  300 held-out games, and from 10.8% to 34.3% across all ten Elemental decks
  over 400 games, without slower decisions
  ([heuristic strength](../experiments/reports/heuristic-strength.md)). Ratings are
  pinned to an anchor version.
- **Matchup tables measure the pilot and the cards.** A four-card swap moved a
  matchup by 55 points ([exp-08b](../experiments/reports/exp-08b-ancestral-dose.md)).
- **A matched-budget exploiter reached 23.5–26.0%** against the distilled
  student in an earlier world, so it was not farmable at that budget; stronger
  attackers were untested ([exp-11](../experiments/reports/exp-11-curriculum-exploitability.md)).
- **An N-game evaluation once played one deal.** A seed bug made every game
  replay the same shuffle, and the seat-advantage claims built on it were
  retracted ([F1 and F3](../paper/understanding.md)). Derived seeds and replay
  verification are the response.

## Quick local evaluation

Train the two-update local recipe, then score its final EMA policy against the
heuristic on the selected Allies/Lessons setup. With the default Etude home:

```sh
uv run etude train manabot/recipes/local.py
uv run python -m manabot.eval.local \
  --checkpoint ~/.etude/track/local/out/policies/00000002-ema.safetensors --opponent heuristic \
  --deal-seed 101 --out .runs/local-check --seconds 300
```

The local recipe writes under `$ETUDE_HOME/track/local` when `ETUDE_HOME` is
set. If `local` already exists, use `--experiment NAME` to train another run
and replace `local` in the checkpoint path with that name.

- `--opponent` also accepts another local policy file, `scripted_greedy` or
  `random`. Policy files are `.safetensors`, or `.pt` from an earlier revision.
- Repeat `--deal-seed` for more blocks; `--include-mirrors` plays all eight legs;
  `--matchup` takes an `AuthoredMatchup` JSON for another admitted roster. The
  checkpoint must already admit that setup; nothing is converted.
- `--game-seconds` and `--max-commands` bound each game. `--seconds` bounds
  checkpoint preparation, scalar play and replay, including child startup.
- `--request request.json --out NEW` repeats a saved request in full. Repeated
  seeds are not new samples.

It needs no training records, credentials or the `evaluation` extra, uses CPU
with one Torch thread per process, and never rents hardware, enters an Etude
queue or starts a callback endpoint. Exit 0 means a complete replayed cohort;
1 means an incomplete attempt. An admission timeout retains
`admission-failure.json`. Results are labelled `local-check-not-admission`:
a quick check, not promotion or held-out evidence.

Typed callers pass a `Backend` to
`manabot.eval.local.checkpoint_request(..., backend=backend)` and use
`manabot.eval.evaluation.evaluate_local(request, deadline=..., out_dir=...)`.
An unknown training seed stays `None`, never inferred from deals.

`evaluate(request, out_dir=..., agents=...)` is the single execution entry
point shared by the local CLI, training monitoring and Etude benchmark/Duel
callbacks. Both public evaluators reject an existing output directory, empty or
not, before writing anything. `evaluation.json` retains the request, retained
execution when present, rows, status, elapsed time and paired-deal intervals; incomplete
cohorts emit no interval. Journals, Commands, replay and performance records
use arena formats under `<out>/panel`. Lower down, `eval.match.play_cell` plays
explicit leg subsets.

Etude entry points are `manabot.eval.benchmark` (one Etude match is one deal
with all eight legs; Etude owns match records, resume and uncertainty) and
`manabot.eval.duel`.

## Etude agents

Managed evaluations declare the same `manabot.eval.agents.AGENTS` on every
`EvalJob`, and pass those agents to `active.play`. Etude owns game processes,
checkpoint grouping, batching, model residency, retries and performance records.
The callback is `play(match, agents)`, with objects in seat order; it runs the
native game and Command replay in that Etude process.

`manabot.eval.agent_loader.load(checkpoint)` reads `checkpoint.path`, chooses
Torch and CPU or CUDA, and returns a deterministic forward:
`list[Prepared] -> list[masked logits]`, preserving length and order. Host
preparation and admission happen once in the game-side sampler. The loader
merges admitted rows, rebasing their object, frame, fact and effect routes;
it never selects an action. `build(player, leaves)` creates the heuristic or
checkpoint sampler without importing Torch. Each decision's recorded seed
controls NumPy PCG64 sampling in the game process.

A checkpoint with a decision-memory encoder has its log prepared by the same
sampler, and the loader forwards each such row as its own batch. A seat whose
player config holds `blank_decision_memory: True` plays the same bytes with
the log zeroed at every decision, registered as a distinct player
([reliance comparison](observation-and-model.md#whole-game-decision-memory));
`manabot.eval.pairing.play_pairing` plays an authored paired deal with either.

Managed checkpoint registrations name `etude-policy-f32-auto-v1` and
`numpy-pcg64-v1`. New tournaments have fresh `agents-v1` names and one native
game group per Etude match. Frozen tournament keys, compute contracts and
receipts are unchanged. `manabot.sim.execution` reads retained
`shared-eval-v1` and `shared-eval-v2` records; it no longer allocates execution.
Standalone local evaluation and training monitoring retain scalar checkpoint
play through bounded local children.

**Failure.** The game loop checks `game_seconds` between decisions and enforces
`max_commands`. Before native execution it fsyncs the pending Command. Native
`AgentError` and PyO3 `PanicException` produce invalid, score-null rows and retain
the journal; later games in the group still run. Replay verifies Commands,
not cross-device policy equivalence. Invalid rows never become forfeits.
Other Python exceptions fail the match. Elemental also fails a match with no
valid replayed games, preserving its evidence, so an all-invalid smoke cannot
report success.

Cooperative clock checks cannot interrupt a stuck native call. At the pinned
Etude revision, a true hang or process exit retries the grouped evaluation job;
Etude Task 2d1933 owns per-match hard deadlines and isolated dead-process
failure. The callback does not create nested game processes to replace that
upstream behavior.

## Continuing an evaluation

```sh
uv run python -m manabot.eval.local --status .runs/local-check/panel
uv run python -m manabot.eval.local --continue .runs/local-check/panel --seconds 300
```

Continuation plays only the missing deals, rerunning any partial deal whole;
only the deadline may change locally. Typed callers use
`evaluation_status(directory)` and `continue_evaluation(directory,
deadline=...)`. A changed request needs a fresh evaluation. Retained served
attempts name the original revision to run them from. Relocated
checkpoints may be passed explicitly and are rechecked by digest and size.

The first `evaluation.json` freezes identity and request; later attempts write
under `attempts/<sequence>/` without replacing earlier traces, and completed
blocks are immutable. Filesystem claims admit one writer (not distributed
leases), so move an evaluation with its whole directory. Re-entering an Etude
benchmark or Duel continues a failed match; a cached success adds no sample.

## Performance evidence

Cells write `performance.json`, `rows.json`, journals and traces. Etude records
managed execution performance. Old allocation and serving receipts remain
readable by the report viewer; they are historical evidence, not live settings.
Elapsed intervals are kept separate and carry no bill; Etude cache reuse adds
no measurement.

Elemental keeps each match invocation in its own `arena/<match-key>/attempt-*`
directory under `ETUDE_OUTPUT_DIR`, or `.runs/elemental` for local evaluation
and smoke runs. Retrying a match preserves earlier failure journals and traces.

```sh
uv run python -m experiments.report_prototype.build \
  --eval-receipt .runs/eval/panel --out .runs/eval/report
```

The report viewer never sums overlapping clocks.
`./scripts/verify-eval-performance-report` is its headless proof (needs locked
`academy/ui` dependencies and Playwright Chromium).

## Card data

17Lands-style card statistics from evaluation games. A card evaluation is an
Etude eval run naming an **experiment**; all runs of an experiment pool in one
kept Etude database (`ETUDE_HOME/statsdb`). Each match plays one deal's eight
legs. The native recorder runs during Command replay, outside observations,
rule hashes, RNG and search clones.

```sh
uv run --extra evaluation python -m manabot.eval.cards evaluate --spec evaluation.json --out .runs/cards
uv run --extra evaluation python -m manabot.eval.cards data --experiment EXP --source .runs/cards [--html page.html | --json]
uv run --extra evaluation python -m manabot.eval.cards audit --experiment EXP --source .runs/cards --replay
```

`CardEvaluationSpec` (`manabot/eval/card_evaluation.py`) names experiment and
run, registrations, checkpoint paths, unique deal seeds, limits and an optional
`AuthoredMatchup` (default Lessons/Allies). Resuming requires identical players
and conditions. `data` filters with `--subject`/`--opponent` (name, Etude
identity or 8+ characters of checkpoint digest), `--position`, `--deck`,
`--opponent-deck`, `--run`, `--commit`; `--rebuild` rereads all blocks.

Any evaluation calling `play_cell` can deliver card data: pass
`card_evidence_dir`, merge `trace_card_tables(...)` tables and blobs into its
Etude `Result`, and declare `rollups=CARD_ROLLUPS` (`manabot.eval.card_metrics`).

**Columns.** GP (games with the card in deck), OH (opening-hand copies), GD
(copies drawn later), GIH = OH + GD with GIH WR = winning qualifying copies /
qualifying copies, GNS (in deck, never in hand), IWD = GIH WR − GNS WR in
points. Counts are copy-weighted; tutors, bounce and generated cards are not
draws. Only won/lost games with a passed replay are eligible; there is no
implicit minimum sample. These are associations, not causal card values. The
GIH interval is a deterministic 95% percentile bootstrap over whole deal seeds,
one draw of deal weights per filter selection.

**Tables.** Each match delivers `card_plays` (per game and player),
`card_counts`, `cards` and the shared `plays`; `seat` indexes `Match.players`,
`game_seat` is the native seat (0 on the play). Queries read only `CARD_ROLLUPS`
(`card_deals`, `game_deals`) through `manabot.eval.card_query.card_db`, never
card rows; a changed rollup takes a new name. Games pool across source commits,
which are provenance and a filter. A crashed or unreplayed game is an
ineligible row; its deal's other games still count. Evidence is a private
blob per game read with Etude's `read_blobs`; `etude convert-eval` makes older
per-file runs readable.

**Arena logs.** `cards export-log --experiment EXP --source .runs/cards --run RUN
--game GAME --viewer 0|1 --out game.log` writes the game's
[Arena-style log](engine-python.md#export-an-arena-style-log) through managym's
exporter, which any recorded game can use without manabot. Arena ids come from
the table shipped with managym, joined by Scryfall id and never by name; cards
without one are written with id `0` and counted in the coverage report. Logs
omit opponent private cards and library order, are synthetic, and must never be
submitted to 17Lands; `tests/fixtures/seventeenlands` pins the upstream client.

**Scale.** MTG-206 (2026-10-27): 5,000 player-games and 90 cards rendered 363
filter combinations in 16.01 s wall; a $0.03 L4 smoke audited 96/96 games
clean. The `Card evaluation scale proof` workflow runs
`manabot.eval.cards verify --games 1024` on CPU as an engineering proof.
Details: [card-evaluation.md at a74f31828](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/card-evaluation.md).

## Checkpoint scenarios

Score retained tactical positions S1–S5 with the ordinary checkpoint player,
without training:

```sh
uv run python -m manabot.verify.checkpoint_scenarios --checkpoints checkpoints.json --out .runs/tactics-001
uv run python -m manabot.verify.checkpoint_scenarios --out .runs/tactics-001 --report-only
```

`checkpoints.json` maps scenario names (`s1_counter_the_bomb`,
`s2_hold_the_wipe`, `s3_bolt_the_threat`, `s4_race_vs_block`,
`s5_hold_up_quench`) to checkpoint paths; omitted scenarios are unsupported.
Each checkpoint must bind that scenario's exact deck pair from
`manabot.verify.scenarios.SCENARIOS`, so an Allies/Lessons checkpoint cannot
score them; a mismatch is a failed admission, not a tactical loss. Belief
checkpoints are unsupported because injected roots lack prior history. The
runner uses CPU, one thread, deterministic actions, `--seed 2` and
`--max-steps 1000` by default, and updates no weights.

Each premise is revalidated first (failure: unsupported). A score requires the
reference behavior and its resolved effect; technical failures record no
score. These are local behaviors against a scripted continuation, not
strength, and must not be pooled. `results.json` and `traces.jsonl.gz` retain premises, attempts, digests
and Commands; `--report-only` regenerates the report by replay without
weights. Tests: `tests/verify/test_checkpoint_scenarios.py`.

## Frozen-policy attacks

Bounded adversarial evaluation, not exact exploitability: train fresh
attackers against one frozen target checkpoint and measure how far they get.
An attack plan fixed the target's bytes and SHA-256, producer seeds, at least
two independent attacker seeds, a strictly increasing ladder of cumulative
updates, shared final deal seeds, time budgets and a prediction; each
target/attacker pair trained as a one-stage self-play regime against the
frozen target, and rungs continued one attacker.

Main cannot run or read a plan: the current loop has no frozen opponent, and
the plan schema went with the [saved regimes](training.md#saved-regimes) it
produced. Revision `f4e1822faf46bbc9cdc43909293f8cbc8de4a300` has both. In an
executed plan every attacker faced its target before training and at every
rung on all four legs; only terminal, replay-verified games counted. A failed
cell stopped the cohort, and results kept `plan.json`, `result.json`,
`report.md`, `training.sqlite`, run directories and arena evidence. Final paired improvement bootstraps over
attacker seeds and is exploratory. A failed bounded attacker never certifies equilibrium. The
real-PPO positive control raised score against a deliberately passive toy
target from .59375 to 1.0 in 64 updates. History:
[frozen-policy-attacks.md at a74f31828](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/frozen-policy-attacks.md).

## LLM players

`manabot.llm` asks an unmodified language model to pick one action from the
viewer-safe text observation and its complete legal-reference list, and keeps
a receipt of the choice. It runs locally through Transformers, behind a
loopback vLLM server, or hosted through OpenRouter; the native game runner
(`manabot.verify.llm_game`) owns capture, stale rejection, submission and
replay, and importing `manabot.llm` loads no model runtime.

The protocol is strict so that scores mean something. The prompt is never
truncated: one that doesn't fit is a failure. An answer must be exactly one
listed reference. One invalid answer gets one repair prompt; a second produces
a seeded uniform draw, recorded as `fallback` in the receipt, and playing
scores must include those games. Runtime errors, provider errors and exhausted
deadlines stop with evidence and never fall back. Local configs
(`configs/llm/qwen3.5-9b*.json`) pin model and tokenizer revisions;
`OpenRouterConfig` reads `OPENROUTER_API_KEY` from the environment, never from
configuration or receipts, and `BudgetLedger` reserves a justified maximum
charge in SQLite before every paid call.

```sh
uv run python -m manabot.verify.llm_game --random-control --out .runs/llm-control
uv run --extra openrouter python -m manabot.verify.llm_game \
  --openrouter configs/llm/openrouter-haiku-high.json --out .runs/openrouter/game
```

`GameConfig.opponent_checkpoint` substitutes a frozen policy for the random
opponent, which is how Haiku 5.5 played MTG-183's final checkpoint to 51–49
over 100 games at $0.37 a game
([Haiku versus MTG-183](../experiments/reports/haiku-mtg183.md)). Qwen3.5-9B on an L4
won one functional game in 308 decisions with every selection valid first time
(2026-10-25). `manabot/reports/inference.py` reads receipts offline without the
model runtime. Full operating detail, including vLLM serving and the campaign
drivers, is in [llm.md at 055dddab1](https://github.com/loopflowstudio/manabot/blob/055dddab1/docs/llm.md).

## Open direction

Checkpoint-versus-checkpoint evaluation at growing model sizes is the primary
capacity workload; small local
checkpoint-versus-heuristic checks keep the same semantics without training,
cloud, queue or callbacks. The cloud startup policy targets at least
two hours of estimated eligible useful work on the chosen host/model mix before
a server starts, releasing idle compute when work finishes. That gate belongs
in Etude's launch scheduler and is not implemented. Compute policy, a durable
Etude priority queue, adaptive allocation and further serving optimizations
remain unselected designs; see
[shared-evaluation.md at a74f31828](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/shared-evaluation.md).

## Ownership

Simulation (`manabot.sim`) owns native games, Commands and bounded local replay.
Evaluation (`manabot.eval`) owns admission (`eval.admission`), requests and
results (`eval.evaluation`, `eval.results`), paired cells (`eval.match`),
ratings (`eval.rating`, seat-aware Bradley-Terry with paired-block bootstrap)
and Etude player loading/building (`eval.agent_loader`, `eval.agents`). Training monitoring (`manabot.eval.monitor`) holds
the request and result of one measured checkpoint; local execution does not
import the managed player loader. `manabot.eval.models` holds only the identities
`EvaluationKey` (the arena a game was played in) and `MatchRow` (one played
leg); wire field names keep their recorded meanings because frozen evidence
carries them. Tests load the INT-6 fixture's key and anchors through
`tests/eval/arena_fixture.py`.

Ordinary evaluation games are retained as canonical managym records in
`games/<cell-id>.jsonl`, with finalized per-deal shards in `games/deals/`. Evaluation
setup, participant and runtime identities stay in private record context;
viewer-frame hashes, player ordinals, policy seeds, latencies, inference ids and
local-update receipts stay in decision annotations. Card eligibility and Etude
admission still require the separate successful replay audit. Its `trace_sha256`
identifies the canonical record, including that metadata. Existing rollup names
and `gih-copies-v1` counting rules are unchanged.

### Custom statistics for every evaluation game

Pass named managym reducers to `play_cell`:

```python
from managym.statistics import Behaviour, CardPerformance

rows, trace, replay = play_cell(
    # Supply the usual key, player_a, player_b, deal_seeds and out_dir.
    **cell_arguments,
    reducers={"cards": CardPerformance(), "behaviour": Behaviour()},
)
print(trace["statistics_path"])
```

The private `statistics.json` contains `tables` under those names, the number
of games, `complete_verified_games`, and replay status/errors. Custom reducers
use the same interface as [record analysis](engine-python.md#ask-questions-of-recorded-games).
They receive decisions and card facts from the existing verification replay;
there is no additional replay or policy inference for statistics. When
`card_evidence_dir` is also set, its audited card evidence comes from that
same traversal. Each reducer's `finish` runs once across all deal blocks, not
once per worker. The card table uses native seats; participant mapping,
eligibility and bootstrap intervals remain evaluation responsibilities.

Only complete, successfully audited games contribute reducer rows. Rejected
games remain in the evaluation's original records and outcome diagnostics.
A catchable reducer exception stops custom analysis and publishes no custom
tables. Its separate `analysis_error` does not invalidate successfully completed
verification, audited card evidence or game scores. Callback, aggregation and
question/result transport failures are analysis failures. An experiment that
requires the table must check this status and can reject the evaluation without
relabeling verified games. It never retries or rewrites the played game.
A worker death or deadline still leaves unfinished verification unavailable;
a hung callback shares that supervised deadline. Tables are mode 0600 and are not attached to Academy viewer records.
Ordinary canonical game records support these hooks; legacy traces and
constructed scenario roots retain their existing verification reader.

Per-game callbacks run in the bounded replay child, so expensive Python analysis
counts toward its replay deadline. Final `finish` aggregation runs once in the
coordinator. Script-local reducer classes and dataclass rows
are supported. Imported reducer modules must be available to that child, just
as they are to the evaluation runner. Keeping all questions in this one call
shares the replay; calling `reduce()` later is a deliberate new analysis.

A failed game's `failures/*.jsonl` file is a settled capture snapshot. A pending
call is retained as unknown and is never attempted while viewing it. Diagnose
it with `uv run --no-project python -m managym.replay <file>` on the recorded
runtime (or explicitly allow a changed runtime for diagnostic replay). Old
Command traces and event journals remain readable; constructed scenario roots
continue to use Command traces.

An audit-view dictionary has the retained Command-trace shape and carries its
canonical `record`; it is a derived view for reports, not another stored
trajectory. Completion and winner come from the record ending. Runner failures
stay in private evaluation context; only a captured native exception marks the
record as failed. Interrupted endings carry a reason token such as `command_cap`,
`truncated` or `crash`, without exposing exception text in Academy.
