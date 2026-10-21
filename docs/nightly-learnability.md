# Learnability diagnostic

This suite catches action representation regressions with three shared tiny
CPU policies. It is a capacity diagnostic plus one existing reward-driven
learning control, not a measure of full-game strength or MTG-118 completion.

```bash
uv sync --frozen --extra dev --extra play --no-default-groups --group play-runtime
uv run --no-sync maturin develop --release --manifest-path managym/Cargo.toml --features python
uv run --no-sync python -m manabot.verify.learnability --out .runs/learnability/check-2026-10-15
# Fast PR subset: same capacity protocol, one seed; explicitly skips PPO.
uv run --no-sync python -m manabot.verify.learnability --pr --out .runs/learnability/pr-check
```

Use a new directory for every attempt. Existing output is never overwritten,
models are never cached, and failed families are never retried automatically.
The parent watchdog kills the worker process group after 480 seconds and leaves
`result.json`, `report.md`, and `worker.log`, with a nonzero exit status. Reports
are checkpointed after each seed; partial results never imply skipped checks
passed. Native build/setup is measured separately by the workflow. Runtime
receipts include platform, Torch version, native binary hash, source commit and
source-content hash, case-input hash, configuration, seeds and artifact hashes.
Dirty development measurements are identified by their content hash, not claimed
to be identical to the base commit.

Local qualification on macOS arm64 completed the full suite in **39.6 seconds
wall / 39.4 seconds CPU**, including 25.9 seconds of production PPO training.
All 144 per-case scores passed; PPO won 64/64 held-out games versus 41/64 for
no update. The original qualification's measured diagnostic/test execution was
about 113 seconds of the 900-second calibration allowance; subsequent compression
checks added 4.83 seconds of tests and a 4.92-second one-seed run. Native release
builds were 20.4s (w6), 17.2s (w4), and 15.0s (w5); the focused debug build was 9.9s.
The [qualification JSON](evidence/learnability/qualification.json) retains failures,
per-case probabilities, costs and hashes. Its checkpoint paths refer to local
`.runs/learnability/final/` artifacts; public artifact hosting is still pending.

## Frozen protocol

`Config` in `manabot/verify/learnability/schema.py` owns the diagnostic limits.
Every seed (17, 29, 43) trains **one generic model** on all six families at once:
attack/decline, Healer/Rallier target, gain-life/draw mode, Island/Mountain,
Learn Lesson identity, and ability selector projection. No semantic-program
cohort is necessary to isolate these generic-input defects. Existing focused
semantic target tests remain in `tests/model/test_action_identity.py`.

The model has width 32, two attention heads and feed-forward width 64, with the
FiLM-conditioned policy (`AgentSpec(policy_film=True)`, protocol
`learnability-2`; see
[starting-weight sensitivity](#starting-weight-sensitivity)). Each of 240 Adam
updates (learning rate .003) uses the same 24 fit cases, for 5,760 supervised
sample exposures per seed. No early stopping or model selection.
All 24 held-out cases are evaluated only after training and weight reload.

Both preferences occur in each family: public life 5/7 selects candidate zero,
35/33 selects candidate one. Held-out life 6/8 and 34/32, with different native
shuffle seeds, tests interpolation over changed states. Held-out cases change
state only: case creation asserts that every card definition a held-out case
shows also appears in its family's fit cases. These labels are an
artificial visible context, **not a claim that low life strategically favors
attacking or any particular Lesson**. No identical observable state has opposite
labels. Case creation asserts native/scalar encoding equality.
Structural tests and diagnostic scoring share `DecisionCase` and its observation
transforms in `manabot.verify.action_cases`.

Every preference/variant must reach probability .90; family averages cannot hide
failure. Each frozen policy also undergoes independently seeded offer shuffling
and object-slot reversal with correct focus/identity remapping. Their probability
difference must be at most .0001. Those transformations are invariance checks,
not held-out states. The no-update control is the same frozen initialization,
reported alongside trained probabilities. Removing the repaired identity channel
must reproduce a candidate-logit tie to .0001; its full policy probability is
reported too (Learn has additional legal choices).

Native branches resolve both selected choices and retain their life, hand and
permanent outcomes in `cases.json`. Attack reaches combat damage; target branches
resolve the spell and assert the chosen permanent disappears; modal branches
check +3 life versus drawing one card. Land and Learn branches retain distinct
resolved definitions. **Ability rows are synthetic equal-focus projections**;
the separate debug Rust test
`cr_602_policy_preserves_ability_index_and_resolution` proves two abilities on
one source are legal and resolve differently. Python does not expose that Rust
scenario's definition mutation, so no native Python ability claim is made.

Capacity weights span several setups and intentionally omit semantic programs.
They are diagnostic artifacts, not playable checkpoints with an invented match
binding. The separate PPO layer below tests ordinary production checkpoint
admission and reload.

## Starting-weight sensitivity

Protocol `learnability-1` trained the additive policy. There, offer and state
meet only by addition inside the ReLU `action_layer` and `policy_head`. In the
attack, mode and ability cases the two offers differ in one selector column,
and public life chooses the label. If no hidden unit changes activation across
those offers and life levels, the offers' logit gap cannot depend on life.
Equal logits are then stationary, and the loss stays at ln 2. Whether a start
has such a unit is luck. Over seeds 1–300 at the frozen budget, 55 additive
starts stuck on at least one of those families. Changing
`CARD_DEFINITION_COUNT` reshuffles every later initialization, which is how
MTG-122's three vanilla cards turned seed 17 red.

`learnability-2` keeps every seed, budget and threshold. It adds one
multiplicative path: the head reads each action embedding scaled by
`1 + W·state`, where `state` is the masked mean of the post-attention objects
(FiLM, Perez et al. 2018, scale only). The product has a gradient at the
symmetric point from any start. `W` starts at zero, so the initial function and
all historical parameters are unchanged. The gate is still as sensitive:
`identity_removed` inputs still produce exact ties, and zeroing the
attack/mode/ability selector columns in training fails all 30 of 30 seeds.

It also uses case set `action-cases-2`. In `action-cases-1`, Learn's held-out
variant 1 (native seed 112) has Pop Quiz draw a Mountain. No Learn fit case shows
that definition, and about one FiLM seed in ten missed that case. Variant 1 now
uses seed 116, which draws an Island, at the same life totals. Case creation
rejects any held-out definition missing from the family's fit cases; no other
family had one.

| Seeds 1–300 | Seeds failing | Stuck fit (attack / mode / ability) | Other misses |
|---|---|---|---|
| additive, `action-cases-1` (`learnability-1`) | 78 | 34 / 25 / 25 | held-out Learn 28; Learn fit 1 |
| FiLM, `action-cases-1` | 31 | 0 / 0 / 0 | held-out Learn 29, target 2 |
| additive, `action-cases-2` | 57 | 34 / 25 / 25 | held-out Learn 1; Learn fit 1 |
| FiLM, `action-cases-2` (`learnability-2`) | 3 | 0 / 0 / 0 | held-out Learn 1, target 2 |

All remaining held-out misses involve only definitions their fit cases show:
they vary only life or native state, so they are generalization misses.

The weakness is not confined to the probe's one-column contrasts. On 96 real
authored decisions (choice-corpus rounds 0–2, ten families, each at life 5 and
35) at the production baseline's model size, 1 of 12 additive seeds stayed
at chance on 85 of the 96 decisions; all 12 FiLM seeds fit every decision.
New training runs therefore train the FiLM policy too (`new_agent_spec()`,
baseline recipe `baseline-v2`), so this gate tests the policy that is trained.

## Broader choice separability

`tests/model/test_choice_separability.py` probes nearby ways information could
disappear between native legal offers, both encoders and the ordinary policy.
The question is whether either preference can be learned; none of these labels
asserts a strategically correct move. The land/pass case reaches first-turn
second main with an unused land drop, but passing is also deliberately fitted.

| Choice pair | Information deliberately removed to reproduce a tie |
|---|---|
| Land / pass; cast / pass | Action type and object focus |
| Block / decline | Declaration bit and attacker focus |
| Block one of two different attackers | Permanent definition identity |
| Scry keep / bottom; pay kicker / decline | Action type |
| Target self / opponent | Distinct player focus |
| Select different revealed cards; discard different cards for Learn | Card definition identity |
| Tap different creatures for waterbend | Permanent definition identity |

Two shared generic models fit all ten pairs, one for each opposite preference,
using seed 17, width 32, two attention heads, feed-forward width 64, 180 Adam
updates and learning rate .003. Both fits use the same fixed native states.
Each chosen alternative must receive at least .95 probability after weight
reload. Identical copies of a card share probability and must retain equal
scores: arbitrary physical IDs are not learning targets. These fixed-state
reversal fits prove separability, not held-out generalization or PPO learning.

The tests also check native/scalar encoding equality, preservation of every
offered focus reference, distinct native successor state digests, deterministic
action selection, zero probability for padding, reversed native offers and
object-slot permutation. Distinct digests establish different transitions,
not better outcomes. The targeted ablations are information-loss hypotheses,
not historical-version qualifications. All twelve tests passed locally in
3.6 seconds; initial fixture failures and the correction for interchangeable
card copies are retained in local work notes. No production aliasing defect
was found in these ten pairs.

These tests run in the existing Linux/macOS integration jobs and in the learnability
workflow's preflight step. The two small shared fits are additional to the
three capacity policies and one production PPO control; they do not add a model
for each assertion or change the frozen qualification receipts.

## Reward-driven control

`manabot.verify.learning_control` is the single implementation formerly embedded
in `tests/training/test_exploiter_positive_control.py`; that test calls it too.
It uses the existing `SeatRoutedCollector`, `NetOpponentTrainer` and
`update_iteration`, with terminal game rewards and no alternative PPO loss.

Seed 197 trains width 8 against a frozen passing opponent in the existing
Mountain/Gray Ogre toy world: 64 updates × 8 environments × 128 transitions =
65,536 learner samples. Each update uses four epochs/four minibatches, learning
rate .01, clip .2, full sample retention, no KL penalty, and the existing default
critic/gradient clipping settings. A 150-second collection deadline is nested
inside the overall watchdog. Ordinary world-bound save/reload precedes scoring.

There are 64 evaluation games before training (deal seed 100197), then 64 each
for the trained/reloaded and no-update policies on the same held-out seed
200197. Acceptance requires at least .80 wins and at least .20 improvement over
both controls. This narrow exploitability check proves reward-driven updates
through the production path; it does not establish general strength. This one
additional model is justified by the distinct reinforcement-learning claim;
there is no extra model per capacity assertion.

## Historical qualification

Pinned source archives were compiled into their own native Python extensions
and imported with their matching Python models. No checkpoint was relabeled.
The [retained receipts](evidence/learnability/qualification.json) establish:

| World | Exact source | Attack | Other audited pairs |
|---|---|---|---|
| w4 | `9576dc40ef486966246e20d9fdf58e5d741f1c40` | forced tie | forced ties |
| w5 / PR288 | `78bf9429a628d1f4a2a0487199ccce1aa5ac7684` | learns, P=1.0 | forced ties |
| w6 / PR289 | `3b3e35fde9f7cc7819f391f657f47c7d0aec6576` plus diagnostic source hash | passes | passes |

The historical probe uses one shared generic fit (120 updates, seed 17), with
initial logit gaps and trained probabilities per family. Old Python bindings
lack mode/ability selectors and permanent definition identity. The fixture's
explicit `legacy_selectors` option is only for this pinned qualification; native
consequence checks belong to w6. Historical ability coverage is projection-only.
Daily information-removal controls reproduce the tested alias behavior; they are
**ablations**, not historical training evidence or proof of tensor equivalence
across every old world state.

To reproduce, export each exact pin with `git archive` into a disposable directory,
build its own extension there with the pinned Python 3.12 interpreter, and set
`PYTHONPATH` to that archive when invoking the current
`manabot/verify/learnability/historical.py` with `--cases` pointing to the current
`action_cases.py`, `--commit` set to the pin and a fresh `--out` JSON path. Keep
all Python and maturin commands under `uv run`; do not import an old model with
the current extension. The receipt records native/probe/case-builder hashes.

## Scheduling and publication

`.github/workflows/learnability.yml` runs the full suite on relevant PR updates
and manual dispatch. Its path filter covers manabot, managym, content, relevant
model/verification tests, diagnostic runners, dependencies, and the workflow itself. There is no
nightly schedule. The measured cost supports full PR coverage; `--pr` remains
available for quick local diagnosis. It uses ephemeral
GitHub-hosted CPU runners, overlap protection, a 15-minute job timeout, a step
summary, and 30-day evidence artifacts even on failure. No Etude/cloud launcher
credentials or Discord webhook are supplied to GitHub. The workflow also checks
watchdog failure reporting and mocked publisher receipts before the diagnostic.
The first [published-head GitHub run](https://github.com/loopflowstudio/manabot/actions/runs/38773826552)
at `e94b0e10643d430f88aa2bff3b61beaeef505d49` passed in 122.7 seconds for the
diagnostic, with 39 seconds of dependency/native setup and 3m17s for the whole
job. Capacity fits took 7.62/7.66/7.69 seconds; the PPO control took 92.12
seconds including evaluation. This measurement predates the broader
separability preflight above. Relevant PR updates are practical at this measured cost; the workflow does not
run on every local commit or every branch push. Repeating this same short suite
nightly adds little evidence. The separate [broader search prototype](../experiments/learnability-search.md)
explores native state diversity and several input configurations on isolated
GitHub CPU runners; `.github/workflows/learnability-search.yml` is manual only,
with a maximum 30-minute search budget. It is not a blocking PR gate.

Discord activation still requires a manabot server/channel webhook and the
choice of a genuinely public evidence location. Nothing has been posted and no
public report URL is invented. Keep the webhook in the mini's macOS Keychain as
service `manabot-nightly-discord`, provisioned through Keychain Access. The local
publisher accepts a completed report and its actual accessible evidence URL:

```bash
uv run --no-sync python -m manabot.verify.learnability.publish \
  .runs/learnability/check-2026-10-15/result.json \
  --state "$HOME/Library/Application Support/manabot-nightly/receipts" \
  --evidence-url "$MANABOT_REPORT_URL"
```

A local publisher can download a completed Actions artifact and invoke that
interface. Destination activation and evidence hosting await provisioning; an
automatic publication cadence has not been selected. No existing mini
experiment or launch job is changed. Retain failure reports as well as
successes. The webhook is read directly from Keychain and never written to a
receipt or log. A local lock and destination/day receipt suppress duplicate
messages. A network ambiguity leaves a pending receipt: inspect Discord and
reconcile it manually before retrying, since Discord has no idempotency key.
A confirmed message ID is the publication receipt.

## Verification

```bash
uv run --no-sync pytest tests/verify/test_learnability.py tests/verify/test_learnability_search.py tests/model/test_attack_choice.py tests/model/test_action_identity.py tests/model/test_choice_separability.py -q
cargo test --locked --manifest-path managym/Cargo.toml --test rules_tests cr_602_policy_preserves_ability_index_and_resolution
```

The full diagnostic exercises the extracted production PPO test as part of its
normal run. Reporting tests exercise actual watchdog termination, immutable
attempt directories, native consequences, permutation remapping, failed Discord
payloads, confirmed-message deduplication and ambiguous-delivery suppression
without posting to Discord.

## Ordinary baseline nightly

The longer nightly runs the same public recipe as:

```sh
uv run manabot train --recipe baseline
```

Bare `uv run manabot train` still selects the tiny local smoke run. A recipe is
above a regime: it supplies the regime, declared horizon, checkpoint cadence,
and evaluation plan. The baseline's existing Ataraxos learning-rate and reference
regularization curves are mapped to the requested number of updates. First and
last rates are stable across horizons; short runs are compressed schedules, not
prefixes of long runs. Omitted mapping preserves old regimes exactly.

The nightly uses the shared recipe default of 128 updates, with three seeds on separate GitHub CPUs and ordinary self-play
and terminal outcomes. No capacity labels, action bonuses or diagnostic states
enter training. Initialization and final checkpoints each play the same 24 deals
in all four seat/deck assignments against random. The learning decision uses
paired score gain and a hierarchical interval over seeds and complete deal
blocks. A positive lower bound and at least five percentage points of mean gain
count as learning. Negative evidence is regression; uncertain evidence is
inconclusive, never a learning pass. Missing or failed attempts are errors.

All exported policies also face the same frozen native choice corpus. Reports
show per-family relative-action and value changes, and summarize actual
collected/retained training rows by action type. Choice movement is descriptive:
we do not require arbitrary preferences, every family to improve, or nonzero
logit differences. Ordinary setup/world admission, encoder parity, masks,
reordering, batching and reload remain hard software checks.

The original short suite stays on relevant PRs and runs as the nightly's sanity
job. The generated supervised-capacity exploration remains manual. The nightly
spends its main budget on ordinary learning and independent evaluation. Its
[protocol and qualification](../experiments/baseline-nightly.md) retains attempts,
limits and results. No Discord publication or paid compute is configured.


The first full qualification completed in 22–23 minutes per seed, with all
software checks passing. Its combined paired outcome gain was 0 percentage
points (95% interval −9.4 to +8.3 points): **inconclusive**, so the learning gate
remains red. This is a measured pipeline, not yet a reliably passing learning
regression. The qualification record preserves that result and all prior failed
attempts; no threshold was weakened.
