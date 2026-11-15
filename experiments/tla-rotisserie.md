# TLA Rotisserie: game rewards versus opening values

Implementation status (2026-11-14): native preparation, typed public draft
inputs, a random draft/construction policy, seat-owned terminal-return updates,
`TrainRotisserie` executor integration, supervised opening scoring and fixed
reference evaluation plumbing are implemented. `compare --fixtures` runs a
controlled software proof. No competent player has passed supported current
source admission, so there is no measured reward comparison or strength result.

```sh
uv run --extra artifacts --extra evaluation python experiments/tla_rotisserie.py prepare \
    --out .runs/tla-rotisserie-prepare --seed 370
```

This prepares one public eight-seat draft and eight native-legal 40-card decks,
without loading a game player or running reward games. The retained transcript
contains 360 picks, 45 cards per seat, and a shared supply falling from 664 to
304 copies. Use it to inspect supply, snake order, drafted pools and final decks
before running the reward comparison.

## Preparation and admission result, 2026-11-14

The final local preparation retained 360 draft picks, eight 45-card pools,
320 construction additions and eight admitted 40-card main decks with unused
drafted cards in sideboards. Shared supply fell from 664 to 304. The 680 intent
rows reproduce the final pools and decks. One local software-validation execution
took 0.119 seconds wall and 0.085 seconds parent CPU; there were no game workers.
These costs describe preparation alone, not game throughput or learning cost.

Local receipts are `.runs/tla-rotisserie-prepare-recorded/{setup.json,choices.jsonl,
catalog.json,printings.json,receipt.json}`. Setup binds the catalog, printing
inventory, native bytes, world, seats, picks and build cap before the first pick; an interrupted journal
has no completion receipt. Existing output directories are refused to preserve
prior attempts. Source is base `83406110db5f04dbe9267fd53a7160ef908b5c6a` plus
the uncommitted implementation, recorded as dirty. This is not a committed Etude
experiment identity. No Etude run, learner update, reward sample or evaluation
game was created.

The referenced frontier Small `final.pt`, SHA256
`c95c61aa298528e5c2782430d5e3f90929d3e9b7ac8b1fc45f560ed6ab4bbaf8`,
was copied read-only from the existing MTG-349 inputs and passed to
`manabot.backends.torch.checkpoint.load_checkpoint` locally. The maintained
loader refused it:

> checkpoint world/setup/input binding differs from this runtime in naming_vocabulary, world_digest; it records no source revision

The refusal is retained in `.runs/tla-rotisserie-player-admission-final.json`.
The exact missing dependency is a competent game player with its opening critic
that passes supported current-world/input admission, whether supplied by
MTG-353 or an already compatible artifact. The MTG-349 runs were left running.
The implementation continuation adds Prepared draft inputs, draft artifact
save/reload, a shared Learner update, supervised opening scoring, durable reward
schedules and fixed-reference tables. The compatible competent player/critic
remains necessary for measured execution. Controlled fixtures establish software
behavior, not a completed comparison.

Validation: eight focused native debug tests passed; 12 Python binding,
preparation and existing Limited-foundation tests passed. After rebuilding and
atomically installing the final CPython 3.13 extension, five focused Python
and experiment-discovery checks passed; both preparation tests passed again after
making the receipt's episode dimensions explicit. Workspace formatting, all-targets /
all-features clippy and Python lint/format checks passed. The broader debug
engine suite and remaining design acceptance checks belong to gate; no game
rules, policy input layout or game checkpoint loader was changed.

## Running the comparison

Run from a source checkout with its matching native extension built and the
locked dependencies installed. The first implementation uses the Torch backend.
For environment setup, see [development](../docs/development.md).
Supply a competent frozen game-player checkpoint that passes the maintained
loader's current world, setup and policy-input admission:

```sh
uv run --extra artifacts --extra evaluation python experiments/tla_rotisserie.py compare \
    --player /path/to/compatible-player.pt \
    --out .runs/tla-rotisserie --seed 370 --updates 1 --seconds 1800
```

`--player` is the game player and opening critic, not the drafter being trained.
Both arms create the same random drafter/builder initialization. No human draft
data, imitation weights or human-policy KL anchor is required. The game player
stays frozen for training rewards and evaluation.

`--out` selects the evidence directory; `--seed` selects the reproducible sampling
schedule; `--updates` selects updates per arm; `--seconds` caps whole-run elapsed
time. The bounded exploratory comparison uses the values above: one seed, one
update per arm and 30 minutes. It runs locally, with one numerical thread and
one supervised game at a time. Check available capacity first; this workflow
does not authorize a rental or additional work on the occupied MTG-349 host.

2026-11-14 local software check with a barely trained width-64 world-bound
TLA checkpoint (not a competent player): complete games took 8–20 seconds each
on a heavily loaded 16-CPU laptop, mostly scalar policy inference. One
game-reward update (32 games) took 280 seconds; one value-reward update
(32 opening scores) took 45 seconds. The two-update schedule plays 384
complete games (320 evaluation, 64 game rewards) plus 64 opening scores, so 30
minutes would leave most of that earlier schedule missing at that rate. The
prospectively revised default uses one draft/update per arm, 16 training reward
samples per arm, and 96 evaluation games: 112 complete games and 16 opening
scores total. The observed range suggests roughly 16–38 minutes; 30 minutes
still does not guarantee completion. This test player establishes no strength.

2026-11-14 software check of that default (`--updates 1 --seconds 1800`), with
a fresh non-competent world-bound test checkpoint trained on current source
(load average 160–240 on 16 CPUs): it completed in 996 seconds wall time. The
value update took 19 seconds (16 openings) and the game update 130 seconds
(16 games). All 96 evaluation games completed, with no failures or missing
games, at about 8 seconds per game (15 seconds at most). Checkpoints that
predate the 2026-11-14 record-vocabulary additions on main fail the maintained
loader's state-dict shape check. This is not a playing-strength result.

Value runs first with at most half the wall time remaining after admission;
game receives the remaining time, including unused value time. An arm timeout
retains its interrupted record and permits the next arm to start. Infrastructure
or integrity failures stop the comparison. Each arm completes its training
before evaluating saved checkpoints, so initial evaluation cannot prevent that
arm's first update. Training interrupted mid-bundle still supplies no gradient.
The time reservation guarantees an opportunity, not successful learning.

The output retains draft transcripts, eight admitted decks per episode,
initialization receipts, policy artifacts, per-arm exposure and CPU/wall costs,
and complete-game evaluation alongside opening values. Source commit and Etude
run ID identify the execution. Keep those identities with the report.

## Shared supply and construction

The TLA supply uses actual TLA printing membership and rarity: four copies of
each common, two of each uncommon, and one of each rare or mythic. Bonus-sheet
printings, tokens and basic lands are excluded from the draft supply. The 281
nonbasic identities comprise 91 commons, 110 uncommons, 60 rares and 20 mythics:
664 copies shared by all eight seats, not 664 per seat.

Printing membership includes Corrupt Court Official, Epic Downfall and Lightning
Strike, whose engine definitions live under earlier sets. A definition's
first-printing key does not exclude its TLA printing. Supply metadata is bundled;
preparation does not fetch it from the network or infer rarity from names.

Seats pick in public snake rounds: 0 through 7, then 7 through 0, repeated until
each has 45 cards. Everyone sees the remaining supply and all drafted pools.
Each pick transfers one copy from the shared supply into the chooser's pool.
Unavailable choices, a wrong chooser or a stale revision reject without changing
the draft.

Each seat then makes exactly 40 construction additions using its drafted cards
and freely available ordinary basics. Native Limited legality controls every
addition and final admission. Unused drafted cards form the sideboard; there is
no sideboarding between games in this comparison. The 40-addition cap is this
experiment's episode length, not a change to Limited's general minimum-deck rule.
All-basic decks remain legal. Partial builds are local to their seat; all eight
finished decks are revealed together.

## What the rewards measure

Both arms use one randomly initialized width-64 policy with two attention layers,
four heads and zero dropout for drafting and construction. Each update collects
one draft with four learner seats and four frozen opponent seats. Seat roles
rotate. Opponents include initialization and previous learner snapshots, with at
most two historical snapshots retained, including initialization permanently.
Initialization, draft, reward and evaluation randomness use separate streams so
different reward work cannot change the draft or evaluation schedule.

Rewards are queried only after complete drafting and construction. Each learner
deck meets a frozen opponent deck from the same draft on two deals crossed with
both game seats, giving four reward samples:

| Arm | Reward for the learner's deck |
| --- | --- |
| Game | Mean signed outcome of four completed games: win +1, draw 0, loss −1. |
| Value | Mean signed opening value at that seat's first published nontrivial decision on the same four deal/seat samples. |

Opening scoring advances the frozen player's actions until the evaluated seat's
first decision. A game ending before that point has no opening value; its outcome
does not substitute for a critic observation. Categorical WDL critics use signed
expectation. Scalar critics must declare `signed_outcome` or `win_logit` semantics;
logits convert with `2 * sigmoid(logit) - 1`. Unspecified scalar units are refused.
A signed-outcome regression head can extrapolate outside ±1; preserve those
finite predictions without clipping and show the range as a critic limitation.

Each seat's reward credits only its own draft and build choices. Both arms use
episode policy gradient with a detached running baseline from earlier completed
batches, one Adam step per update, learning rate 1e-4, epsilon 1e-5 and gradient
clip norm 1. There is no advantage normalization, entropy term or KL term.
Frozen opponents and the game player receive no gradients. Each learner needs
its complete four-sample bundle before contributing a gradient.

## Read the comparison

The default one-update schedule contains 16 training reward samples per arm and
340 learner draft/build decisions per arm. Drafted opponent decisions are recorded
separately from gradient exposure. Equal updates do not imply equal execution
cost: complete games and opening scoring do different amounts of work.

Evaluation plays complete games at shared initialization and each trained endpoint.
Each candidate occupies every draft seat against two fixed reference tables:
seven initialization policies, and seven policies sampling uniformly from legal
definitions during drafting and native legal additions during construction.
The candidate plays the opposite draft seat, `(candidate + 4) mod 8`, on
one deal crossed with both game seats. This gives 32 games per checkpoint and
96 scheduled evaluation games across shared initialization and the two trained
endpoints. Training retains its four-sample (two-deal) bundles.

Compare wins, draws and losses with their valid denominators, opening-value
means/ranges and value-versus-outcome divergence by checkpoint, reference and
seat. Read curves against updates, draft decisions, complete-game exposure and
measured cost separately. Parent and worker CPU costs are retained separately;
shared-host elapsed time is not isolated throughput.

Changing the candidate changes the draft and therefore both final decks. Matching
sampling seeds across candidates is not an exact paired final-deck deal; that
pairing holds only within a retained deck pairing and its seat-swapped legs.
Aggregate symmetric self-play's 50% win rate cannot establish improvement.

This is an exploratory, one-seed feasibility comparison. The frozen player's
competence and critic may degrade on Rotisserie decks outside its training
distribution. Opening values can rise while actual wins fall. Report that
divergence, poor decks and incomplete results; games from one training seed do
not establish replicated playing strength.

## Refusals, failures and recovery

Missing, ambiguous or contradictory printing metadata stops preparation with
the affected rows. A player admission mismatch stops comparison before reward
work. Use a supported compatible competent checkpoint; retain the refusal and
checkpoint identity. Do not bypass world binding, initialize missing vocabulary
weights, or substitute a random player for a measured comparison.

Games have a 60-second and 4,000-deliberate-Command bound. Engine errors,
truncations and incomplete reward bundles remain excluded failures, never draws,
and receive no gradient. The ordinary game-error ledger uses a 10% threshold
over a 100-settlement window after 100 warmup settlements; exact counts still
matter below warmup. Capacity recovery is disabled. Admission, integrity and
infrastructure failures stop immediately. A worker dying settles its dispatched
samples as interrupted with their last journals; never-dispatched samples remain
missing. Failed samples are not silently retried or replaced.

At the whole-run deadline, keep the paused or incomplete schedule and all
completed evidence. Do not shrink the schedule after seeing scores. Inspect the
retained failure and fix its cause before a separately identified attempt;
neither repairing code nor starting another attempt completes missing historical
samples. Exclusions can bias results toward games that fit the runtime limits.

Run the controlled proof without a player:

```sh
uv run --extra artifacts --extra evaluation python experiments/tla_rotisserie.py compare \
    --fixtures --out .runs/tla-rotisserie-fixtures --seed 370
```

This uses two seats, one draft pick each and native 40-card construction, then
one update with a declared synthetic return. It is controlled software
validation, not measured reward evidence. Keep it separate from the comparison. Preserve transcripts and completed replays,
planned/attempted/completed/failed/missing counts, identities and phase costs.
Publish retained experiment data through the [artifact manifest](README.md#data)
before committing the report's results.


## Retained implementation outputs

Each arm runs through `execute_regime` and retains its `TrainingRun` and Etude
identity. Draft artifacts live under `<arm>/<stage>/drafts/update-NNNN.pt`;
they carry a draft-specific schema, configuration, catalog/printing binding and
source revision. They are not game checkpoints. `catalog.json` and
`printings.json` retain the bound inputs beside them. Initialization is permanent;
the other opponent snapshot is the previous update and stays frozen throughout
collection. The two arms use the same initialization seed and independent
per-seat action generators, with separate reward and evaluation seeds.

Preparation, training and evaluation use one collector in `manabot/drafting.py`
and one `DraftEpisode` completion schema. New preparation uses the same per-seat
NumPy sampler as the uniform evaluation reference; its transcript for a given
seed differs from the earlier Python-random preparation above. Historical
receipts retain their original bytes. Every collected episode now retains source,
native, catalog and printing identities plus its journal digest and timing.

Per-episode setup and choice journals precede native mutation. Reward schedules
precede dispatch; dispatch and settlement rows are fsynced. Game traces, opening
observations and failed journals remain in the reward directory. Completed games
also require a supervised replay; a replay failure stops the attempt. A critic
observation is a successful score, never a completed game. A terminal game
without the evaluated seat's first decision has no opening score.

Fixed-reference evaluations retain table assignments, per-game outcomes and
opening values, plus summaries with planned, attempted, completed, failed and
missing denominators, win/draw/loss counts and value/outcome divergence. Etude
stores these as immutable tables. Update receipts retain collected and gradient
decisions, prior-batch baseline, collection/learning time and parent CPU/wall
cost; game receipts keep worker CPU separately (unknown after an abrupt exit).
Different rewards do not consume draft/evaluation schedule randomness.

The initial evaluation is shared between arms and retained in the value arm; each trained checkpoint is
scored separately. `comparison.json` keeps expected checkpoint endpoints even
when a deadline leaves an endpoint unscored. It also totals planned, attempted,
completed, failed and missing games for every expected endpoint, including
wholly unstarted evaluations. Unreached schedules are incomplete;
there is no automatic resume or replacement of failed evaluation samples.


## Controlled implementation validation, 2026-11-14

The `compare --fixtures` command completed one synthetic-return update with 41
learner choices and two native-legal decks. Its receipt is
`.runs/tla-rotisserie-fixtures-implementation/receipt.json` and explicitly records
zero reward games and no strength result. This proof ran from the implementation
checkout with dirty source; it is not a committed measured experiment identity.

Focused tests cover native public projection and masks, owned terminal gradients,
failed-bundle exclusion, prior-batch baselines, artifact reload/transcript identity,
regime execution, fixed-reference draft/game seats, signed scalar/logit/WDL units,
actual supervised checkpoint opening inference, completed-game replay, persistent
error allowance, worker exit, truncation, deadline and missing-slot accounting.
Successful opening scores are recorded as `opening_scored` in the shared ledger,
separately from completed games. Scalar extrapolations remain visible.

Validation commands passed: the draft/reward test files (21 checks); existing game preparation,
policy catalog, package-boundary and supervision checks (89 checks); observation,
Torch optimizer-contract and restart checks (55 checks); and focused existing
ledger threshold and acknowledged-worker-exit cases. Backend conformance included
two skips for unavailable optional backend coverage. Ruff and diff checks passed.
The new tests are included in the training CI group. Broader gate/CI validation
and the admitted competent-player dependency remain; no measured arm was launched.

## Bounded schedule implementation check, 2026-11-14

The prospective default now uses one draft/update per arm and one evaluation
deal crossed with both game seats. Both fixed reference tables and every draft
seat remain required. Value runs first within half the post-admission remaining
time; game can consume the remainder. Each arm trains before evaluating its
saved checkpoints. An arm timeout preserves artifacts and an interrupted record,
then permits the other arm to start; other fatal errors stop the comparison.
Missing evaluations remain in `comparison.json` even when never started.

Focused draft-training and reward checks: 24 passed in 7.51 seconds. Controlled
checks cover retained optimizer exposure/artifacts before evaluation timeout,
missing denominators before drafting, raised and settled deadlines, and fixed
reference/game-seat crossings. An injected-clock composition check gave each arm
900 seconds after a simulated value timeout and retained both run identities,
64 missing and 32 completed simulated evaluation slots. No games ran in that
composition check. These are software checks with dirty source, not strength,
throughput or a supported competent-player comparison. Competent compatible
player admission remains required before measured execution.
