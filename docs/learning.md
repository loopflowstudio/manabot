# How a manabot learns

A manabot starts knowing only the legal actions at each decision. It plays
Magic against copies of itself, and the only feedback is who won. This guide
follows one training iteration from the first decision to the weight update,
and explains how a win at the end of a game becomes a change to the network.

It describes `DampedPolicyLearning`, the rule every current recipe selects
(the `TrainSelfPlay` field itself still defaults to the PPO `Learning` control),
with worked numbers, the code behind each stage, the questions a batch boundary
raises, and the exact [departures from the source recipe](#fidelity-to-the-sources).
The [README](../README.md#how-a-manabot-learns) has the loop in four sentences;
for running training, see [training.md](training.md).

![One training iteration: play, assign credit, filter and update, then repeat with new weights. Weights also feed an EMA, checkpoints and evaluation, which never feed back into the loss.](images/learning-loop.svg)

- [Words used here](#words-used-here)
- [1. Play](#1-play)
- [Can a batch contain unfinished games?](#can-a-batch-contain-unfinished-games)
- [2. Assign credit](#2-assign-credit)
- [3. Filter](#3-filter)
- [4. Update](#4-update)
- [Weights, checkpoints and evaluation](#weights-checkpoints-and-evaluation)
- [Reading a run](#reading-a-run)
- [Common questions](#common-questions)
- [Open questions](#open-questions)
- [What the evidence says](#what-the-evidence-says)
- [Influences](#influences)
- [Fidelity to the sources](#fidelity-to-the-sources)
- [Where the code lives](#where-the-code-lives)

## Words used here

| Word | Meaning |
| --- | --- |
| **Decision** | One point where a seat chooses among two or more legal actions. Forced decisions are played by the engine and never appear. |
| **Row** or **transition** | One recorded decision by a learning seat: observation, action, probabilities, value, reward and end flag. |
| **Stream** | One native game slot. It plays games one after another. |
| **Trajectory** | The rows of one seat in order. Credit never crosses from one trajectory to another. |
| **Batch** | All rows collected for one iteration. |
| **Iteration** or **update** | One collect-and-learn cycle. It contains many optimizer steps. |
| **Behavior policy** | The frozen weights that played the batch. |

## 1. Play

`SeatRoutedCollector` steps every stream in lockstep, so one forward pass
chooses actions for all of them. The network sees only the acting seat's
[viewer-safe observation](observation-and-model.md). Hidden cards and game
outcomes never enter it.

Each decision stores what the update will need later: the observation, the
chosen action, the behavior probabilities over every legal action, and the
behavior value estimate. The weights do not change during collection.

By default both seats learn. Each seat of each stream keeps its own trajectory,
so one game yields two trajectories with opposite rewards. With
`learning_seats="one"`, the other seat is played but its decisions are not
trained on. Random and frozen opponents are never trained on.

A row's reward is not known when it is recorded. It stays pending until that
same seat decides again (reward 0) or the game ends (the final result). The
next row in a trajectory is therefore this seat's next decision, however many
opponent decisions came between them.

![An interleaved game of decisions by seats A and B, with one forced decision, splits into one row sequence per seat. Seat A's last row has reward +1 and seat B's has -1; all other rows have reward 0.](images/game-to-trajectories.svg)

From `SeatRoutedCollector` in
[`manabot/training/rollout/collection.py`](../manabot/training/rollout/collection.py):

```python
slot = (
    2 * int(row) + int(acting[row])
    if self.learning_seats == "both"
    else int(row)
)
self._pending[slot] = _Pending(...)
```

Each stream and seat has its own slot, and a row stays pending in it until that
seat decides again or the game ends.

### Value

The expected game outcome from the acting seat's observation, under the
policies currently playing: $V(o) \approx \mathbb{E}[\text{outcome} \mid o]$.
From `ActorCritic.forward` in
[`manabot/model/actor_critic.py`](../manabot/model/actor_critic.py):

```python
logits, raw_value = self.forward_distribution(obs)
if self.config.value_kind == "categorical_wdl":
    probabilities = raw_value.softmax(dim=-1)
    value = probabilities[:, 2] - probabilities[:, 0]
else:
    value = raw_value.squeeze(-1)
```

`categorical_wdl` predicts loss, draw and win probabilities, and its value is
P(win) − P(loss), so a draw counts as 0; `scalar`, used by the baseline recipe,
predicts the outcome directly. The critic sees the same viewer-safe observation
as the policy, never the opponent's hidden cards, and its values are recorded
during collection and stay constant through the update.

## Can a batch contain unfinished games?

It depends on how the batch is closed, and there are two ways.

![Complete-game batches let each stream finish its game and idle until every stream is done. Exact-quota batches stop every stream at the same decision count, pausing unfinished games and bootstrapping them from a value estimate.](images/batch-boundaries.svg)

`batch_closure` declares which, independently of who learns.

**Complete games: no.** This is the default for new stages, and two-seat learning
always uses it. `transitions` is a floor on a stream's retained decisions, the
total across seats when both learn, unless `games_per_stream` is set, which
replaces the floor with a game count. A stream that has passed the floor stops at the end of its current game.
Rows are packed into one `[rows, 1]` column with an end flag after each
trajectory, and row counts vary by batch. The collector asserts that the last
row is an end, and `bootstrapped_tail_fraction` is 0.

**Exact quota: yes.** With `batch_closure="exact-quota"`, which needs one learning
seat and neither `games_per_stream` nor a game error policy, each stream collects exactly `transitions` learner decisions and the batch has
shape `[transitions, streams]`. The scale ladder v1 run trained this way. One
column can hold the end of a game begun in the last batch, some whole games,
and the start of a game that hasn't finished.

For an unfinished game, the stream pauses at the learning seat's next decision.
That observation becomes `next_obs`, and no action is sampled for it. Three
things follow:

1. **The value network stands in for the outcome.** `_targets` evaluates
   `V(next_obs)` with the collecting weights, before any optimizer step, and
   uses it as the last row's successor. No reward is invented, and the row is
   not treated as a game end.
2. **These rows never see the real result directly.** They learn about the
   outcome only through that value estimate. The rest of the game lands in the
   next batch, whose rows do receive the result.
3. **No sampled action crosses the update.** The next batch chooses the paused
   decision again with the new weights, so all rows in a batch share one
   behavior policy. A single game can still be played by several versions of
   the policy.

A column can also hit a true game end in the middle of a batch. The end flag
cuts both the value lookup and the running sum at that row, so one game's
credit never reaches the game before it. When a stream's last row is itself a
game end, `next_obs` belongs to a fresh game and is ignored.

An unfinished game always comes from a deliberate pause. A game that the engine
truncates raises an error instead of counting as a draw, and a run that hits
its deadline during collection raises without training on the partial batch.
With a game error policy, a failed or truncated game's rows are deleted from
the batch and its receipt keeps the replay inputs. See [game failures](training.md#game-failures).

## 2. Assign credit

Credit is computed once for the whole batch, before any weights change.

The **reward** is +1 for a win, −1 for a loss and 0 for a draw, on the last row
of a trajectory. Every other row has reward 0, and there is no shaping.

The **advantage** says how much better things went than this decision's value
estimate predicted. It uses generalized advantage estimation with no
discounting (γ = 1):

$$\delta_t = r_t + V_{t+1} - V_t \qquad A_t = \delta_t + \lambda A_{t+1}$$

At an end, $V_{t+1}$ and $A_{t+1}$ are zero. At a paused tail, $V_{t+1}$ is the
bootstrap and $A_{t+1}$ is zero. Advantages use λ = 0.5, which leans on the
value network and gives low-variance credit near the decision. With a
categorical value head, $V$ is P(win) − P(loss).

The **value target** is what the value output should have said. It is the same
recursion with λ = 0.8, which leans more on what actually happened. A
categorical value head uses a three-way loss/draw/win mixture with the same λ
and a one-hot terminal. A scalar head uses $A_t + V_t$.

From `transition_gae` in
[`manabot/training/learning/returns.py`](../manabot/training/learning/returns.py):

```python
for step in reversed(range(len(rewards))):
    live = (~ends[step].bool()).to(rewards.dtype)
    delta = rewards[step] + gamma * live * following - values[step]
    tail = delta + gamma * lam * live * tail
    advantages[step] = tail
    following = values[step]
return advantages, advantages + values
```

Time advances per decision of the same seat, so opponent decisions add no
steps. `live` cuts both the bootstrap and the trace at a game end, so credit
never crosses into another game. `gamma` is always 1 here. Advantages are never
normalized: they stay in outcome units, which gives the filter's 0.01 floor a
fixed meaning.

### A worked example

A seat makes three decisions with values 0.1, 0.3 and 0.6, then wins.

| Decision | $V_t$ | $\delta_t$ | Advantage (λ=.5) | Value target (λ=.8) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.1 | 0.3 − 0.1 = 0.2 | 0.45 | 0.796 |
| 2 | 0.3 | 0.6 − 0.3 = 0.3 | 0.50 | 0.92 |
| 3 | 0.6 | 1 − 0.6 = 0.4 | 0.40 | 1.0 |

Now suppose an exact-quota batch closes after decision 2, with the game paused
at decision 3. The 0.6 is a bootstrap, and the win isn't known yet.

| Decision | $V_t$ | $\delta_t$ | Advantage (λ=.5) | Value target (λ=.8) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.1 | 0.2 | 0.35 | 0.54 |
| 2 | 0.3 | 0.3 | 0.30 | 0.60 |

Decision 2's advantage drops from 0.50 to 0.30 because it no longer sees
decision 3's surprise. A wrong value estimate at the pause passes straight into
these rows, which is the cost of an exact quota. The tables are real outputs of
`transition_gae`.

The bootstrap isn't a special case. Every row's $\delta_t$ already uses the next
value estimate, and the real result fades by λ per decision. Twenty decisions
from the end of a game, the result carries weight $0.5^{20} \approx 10^{-6}$ in
the advantage and $0.8^{20} \approx 0.01$ in the value target. In both batch
modes, most decisions learn about the outcome through the value network, so
advantage quality depends mainly on critic quality. A pause only cuts the last
few rows' trace short: about 2 rows' worth for advantages and 5 for value targets.

![Three curves show lambda to the power of decisions before game end. At lambda .5 the weight is near zero by ten decisions, at .8 by about thirty, and at .95 it is still about .13 at forty.](images/credit-fade.svg)

| Decisions before game end | λ = .5 | λ = .8 | λ = .95 |
| ---: | ---: | ---: | ---: |
| 1 | .5 | .8 | .95 |
| 5 | .03 | .33 | .77 |
| 10 | .001 | .11 | .60 |
| 20 | .000001 | .012 | .36 |

## 3. Filter

Most decisions teach little: their outcome matched the estimate. The filter
keeps rows whose |advantage| is at or above both the batch's 0.75 quantile and
0.01. That keeps about a quarter of rows, and all ties survive.

Selection uses magnitude, so strongly bad decisions stay alongside strongly good
ones. By default, a dropped row leaves both the policy and value losses;
`filter_scope="actor"` keeps every row for the value loss. Advantages are not
renormalized after filtering. If nothing survives, the iteration still counts
but takes no optimizer step.

These are the last five decisions of a real lost game, from a batch that kept
81 of 324 rows on an untrained network. `uv run python scripts/inspect_batch.py`
prints this view in a few seconds:

| To end | Value | Advantage | Kept |
| ---: | ---: | ---: | --- |
| 4 | −.132 | −.054 | |
| 3 | −.155 | −.063 | |
| 2 | −.160 | −.118 | Yes |
| 1 | −.056 | −.443 | Yes |
| 0 | .003 | −1.003 | Yes |

From `selected_moves` in
[`manabot/training/learning/selection.py`](../manabot/training/learning/selection.py):

```python
magnitude = advantages.detach().abs()
threshold = max(float(torch.quantile(magnitude, quantile)), minimum)
return magnitude >= threshold
```

When most advantages fall below 0.01, the floor sets the cut and far fewer than
a quarter of rows survive.

## 4. Update

The update makes one pass over the batch, in batch order, with no shuffling.
Each minibatch is a window of the flattened rows as wide as the stream count.
Filtered rows are removed within the window, and an empty window is skipped.

Per retained row, with $r$ the ratio of current to behavior probability for the
chosen action:

$$L_\pi = -\min\big(rA,\ \mathrm{clip}(r, 0.8, 1.2)\,A\big) + 0.1\,\mathrm{KL}(\pi \,\|\, \pi_{\text{behavior}}) + \tau_i\,\mathrm{KL}(\pi \,\|\, \pi_{\text{magnet}})$$

The total loss is the mean of $L_\pi$ over retained rows plus the value loss:
cross-entropy for a categorical head, squared error for a scalar one. Each term
has a separate job:

- **The clipped ratio** pushes up actions with positive advantage and pushes
  down the rest, and stops pushing once the policy has moved 20%. Ratios matter
  even in one pass, because earlier minibatches have already changed the weights.
- **KL to behavior** limits how far one iteration moves the whole distribution,
  including actions that weren't chosen.
- **KL to the magnet** pulls toward a fixed reference: uniform over action
  types, then uniform within a type. A spell with dozens of target choices then
  gets no more prior weight than a single attack. Its strength τ decays.
  [Influences](#influences) says where each term comes from.

The **behavior policy** is the frozen weights that played the batch, and the
ratio measures how far the learner has moved from them. From
`damped_policy_loss` in
[`manabot/training/learning/damped_policy.py`](../manabot/training/learning/damped_policy.py):

```python
ratio_logs = (
    (logs - behavior_logs.detach()).gather(-1, actions[:, None]).squeeze(-1)
)
surrogate = clipped_surrogate(ratio_logs, advantages, clip)
```

Collection stores log-probabilities for every legal action, because the KL term
needs the full distribution and not only the sampled action. This is a
per-action correction; there is no trajectory-level importance weighting.

The **magnet** is a fixed reference policy built for each row from its legal
actions. From `reference_distribution` in
[`manabot/training/learning/references.py`](../manabot/training/learning/references.py):

```python
counts.scatter_add_(-1, types, valid.to(counts.dtype))
weights = valid / counts.gather(-1, types).clamp_min(1)
return weights / weights.sum(-1, keepdim=True)
```

Each action type present gets equal mass, split evenly among that type's legal
actions, and the penalty is a reverse KL summed over all legal actions.

![Constants from collection feed the policy, damping and value terms. Gradients reach only the live network, through one shared encoder.](images/gradient-flow.svg)

Rewards are not backpropagated. Advantages, value targets, behavior
probabilities and the magnet are constants by this point. Each row is an
independent forward pass on its stored observation, with no backpropagation
through game time. The game's result reaches the weights only as those numbers.

Both losses train one shared encoder. The gradient norm is clipped to 0.267
before each Adam step.
Step sizes depend on the absolute iteration $i$:

$$\text{lr}_i = \mathrm{clip}(0.5\,/\,i^{1.1},\ 5\times10^{-6},\ 10^{-4}) \qquad \tau_i = 0.05\,/\,i^{0.3}$$

An `IterationSchedule` can rescale $i$ so a short run covers the same curve:
update $u$ of a run with horizon $S$ sits at $i = 1 + 9{,}999\,(u-1)/(S-1)$.
Without one, a run of a few hundred updates never leaves the learning-rate
clamp, which holds until $i \approx 2{,}305$. New sampled-roster recipes
(`manabot.recipes.builders.sampled_self_play`) take the measured seconds per
update and declare $S$ as the updates the wall reaches at that cost
(`fitted_horizon`), capped at 10,000; regimes that omit a schedule keep the
absolute iteration.

The PPO-style `Learning` rule remains as a control. It normalizes advantages
and shuffles minibatches, and may take several epochs.

## Weights, checkpoints and evaluation

Training keeps two sets of weights. The **raw** weights are the learner, and
they play the next batch. The **EMA** is an average updated once per iteration
at 0.999, including iterations the filter emptied. A recipe can collect with the
EMA by setting `behavior="ema-self"`; otherwise it is only an evaluation output.

Checkpoints are bound to their [world](../WORLDS.md): a frozen version of the
rules, cards and observation encoding. From `validate_checkpoint_world` in
[`manabot/model/world.py`](../manabot/model/world.py):

```python
if _digest(current_binding_form(binding)) != _digest(expected):
    raise ValueError(
        "checkpoint world/setup/input binding differs from this runtime"
    )
```

The check compares a digest of the world, setup and input binding, so a
matching world number alone isn't enough. Evaluation runs outside
the training clock and plays on held-out deals with both decks and seats
([evaluation.md](evaluation.md)). Its scores don't affect the loss. A run's
selection rule is `last-complete-raw`, never the best seed or best evaluation,
so a result can't be picked by looking at the test.

## Reading a run

| Question | Metric | Reading it |
| --- | --- | --- |
| Is the agent improving? | Evaluation scores | The only strength signal. Self-play results can't show it, because both seats are the same agent. |
| Did batches contain unfinished games? | `bootstrapped_tail_fraction` | Share of trajectory columns paused mid-game. Always 0 for complete games. |
| Is the filter working as designed? | `retained` / `rows` | Should sit near a quarter. |
| Is the policy moving too fast? | `collection_kl`, `numerical/clip_fraction` | Rising values mean updates are pushing against their limits. |
| Is exploration collapsing? | `entropy` | A fast fall means the policy is committing early. |
| Where does credit concentrate? | `selection_groups/<action type>/<terminal distance>/rows` | Shows which decisions, and how far from game end, survive the filter. |

Loss values describe the last minibatch of an iteration, and policy loss is not
log loss. Full definitions are in [monitoring](training.md#monitoring) and
[experiment-metrics.md](experiment-metrics.md).

## Common questions

**Why doesn't λ = 1 use the actual result everywhere?** It would, at the price
of high variance: a game has many decisions, and one result says little about
any single one. Lower λ trusts the value network more. The two different λs
let advantages stay stable while value targets stay close to real outcomes.

**Does the opponent's play affect credit?** Only through what happens next. The
opponent's decisions add no rows and no discount steps to this seat's trajectory.

**Are both seats of one game trained together?** Yes, as separate trajectories
in the same batch with opposite rewards. Each seat's rows come from its own
viewer-safe observations.

**Is there a replay buffer?** No. Each batch is used for one pass and discarded,
and no trajectory-level importance correction is applied.

**Why does a complete-game batch wait for slow games?** A stream past its floor idles
until the last stream finishes, so the batch can be built from complete games.
That idle time is the price of having real outcomes on every trajectory.

## Open questions

- **How far back should a result reach?** In one baseline, Allies reached
  87–88% against a scripted opponent while Lessons climbed slowly from 5/150
  to 32/150 ([current baseline](../experiments/current-baseline.md)). Credit
  reach might be part of the reason, but that's untested.
- **What does the advantage filter do?** Only about the top quarter of rows
  by |A| train, and both KL terms apply only to those rows. Filtering on a
  sampled advantage may change where the update converges. A game with exact
  exploitability could test that directly.
- **Which decisions actually lost the game?** Strong players can often point
  to one in a replay. Those judgments could become fixed test positions that
  track what a policy has learned.
- **Is the policy improving, or circling?** A payoff matrix across checkpoints
  from different points in training can separate the two
  ([durable progress](../manabot/RESEARCH.md#durable-progress)).

## What the evidence says

Most of this recipe is inherited from the paper and is being tested on Magic
one switch at a time. These are the results the guide leans on:

- **Terminal reward only.** Potential shaping matched it and pay-per-event
  shaping lost to it by about 14 points
  ([exp-04](../experiments/exp-04-potential-shaping.md)), in an earlier world.
- **Technique screens.** Each Ataraxos component has its own status in the
  [technique screen](../experiments/ataraxos-technique-screen.md) and
  [ablations](../experiments/ataraxos-mtg-ablations.md). The recipe is not a
  demonstrated strength result as a whole.
- **Scale.** The largest run so far is MTG-183, one seed on ten decks for
  331M learner decisions ([protocol](../experiments/protocols/elemental-183.md));
  the one-seat [scale ladder](../experiments/scale-ladder.md) is the largest
  exact-quota run. Both are summarized in the [README](../README.md#scale).

## Influences

None of these methods is new here. Each row says what we took and what changed
for Magic.

| Source | What we use | What changed |
| --- | --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347) | The clipped ratio | One pass in batch order, with no advantage normalization |
| [GAE](https://arxiv.org/abs/1506.02438) | The advantage recursion | No discounting, and steps are same-seat decisions |
| [Magnetic mirror descent](https://arxiv.org/abs/2206.05825) | The KL pull toward a fixed reference | The reference is uniform over action types, then over actions |
| [Ataraxos](https://doi.org/10.1038/s41586-026-11036-y) | The combination: both KL terms, filter, schedules, categorical value, EMA | No learned setup, no ordered past moves by default, and a much smaller network |

The rules engine, viewer-safe observations, scoring of legal actions over game
objects, and paired evaluation were built for this project. The PPO-style
`Learning` rule remains as a control.

## Fidelity to the sources

`manabot/training/learning/damped_policy.py` owns the objective; recipes select
`DampedPolicyLearning(gradient="damped_policy")`. Historical `ataraxos_move`
regimes are rejected and need their pinned source; frozen evidence keeps the
old tag. Per retained observation the policy loss is the negative clipped-ratio
advantage surrogate plus `collection_kl * KL(current || behavior)` and
`tau * KL(current || magnet)`, summed over all legal offers. Advantages use
signed expected outcomes with their own lambda. Outcome targets are three-way
lambda mixtures stored in **loss, draw, win** order with one-hot terminals; the
critic gets cross-entropy, equally weighted with the policy loss. The scalar
ablation uses squared-error lambda returns instead. Only current logits receive
gradients: behavior, references, estimates and targets are detached; the
shared encoder receives both losses.

Collection freezes learner weights within each batch; complete-game closure
finishes games before updating, while an exact-quota game may span updates. One epoch still changes parameters between
minibatches, so per-action ratios remain necessary. No replay buffer,
trajectory ratio product or state-occupancy correction is implemented.
Filtering changes the training-state measure, as in the source, and
advantages are not renormalized after filtering. Truncation fails rather than
becoming a draw; opponent actions are traversed, not trained on.

Optional controls: `filter_scope="actor"` keeps all critic rows;
`behavior="ema-self"` collects and bootstraps with averaged weights
([protocol](../experiments/ataraxos-omitted-controls.md)). EMA evaluation
outputs are selected explicitly at arena registration.

### Fidelity table

| Technique / source | Implementation | Boundary |
| --- | --- | --- |
| S3.4 eq. (6), clipped surrogate | `damped_policy_loss`; both clipping branches tested | Same loss |
| Both reverse KLs, S3.4 | Exact legal-support sums against saved behavior and magnet | Finite legal log support required; rounded zeros counted, no smoothing |
| Magnet, S3.4 | `reference_distribution`, action-type uniform by default; uniform-offer control | MTG action types replace piece-then-move groups |
| Scalar advantage, S3.4 | Transition-end GAE, gamma=1, lambda=.5; detached collection values | Learner decisions replace same-player positions |
| Outcome estimator, eq. (5) | Vector lambda=.8, terminal one-hot, paused-tail bootstrap | Loss/draw/win storage order is a permutation |
| Value representation, S3.4 | Categorical head with CE | Scalar MSE is an explicit ablation |
| Filtering, S3.4/Table S7 | Inclusive .75 quantile and .01 minimum; ties included | Torch linear quantile; source interpolation unspecified |
| Optimizer damping, Table S7 | Adam, clip=.2, collection KL=.1, max norm=.267, value weight=1 | Adam betas .9/.999, epsilon 1e-5 (source unspecified) |
| Schedules, Table S7 | LR clip(.5 / iteration^1.1, 5e-6, 1e-4); tau=.05 / iteration^.3 | One-based iterations; no elapsed-budget substitution |
| Batch reuse, S3.4 | One epoch, increasing timestep batches, no shuffle | CPU streams replace 202-step GPU batches of 1536 |
| EMA, S3.4 | .999 per iteration including empty filter; parameters averaged, buffers copied | Initialized from learner; evaluation output by default, `behavior="ema-self"` opts in |
| S3.3 setup learning | Not implemented | MTG has no learned setup |
| S3.5 belief, S3.7 search | Outside this trainer | See [beside the loop](../README.md#beside-the-loop) |
| S3.6 network scale | Semantic Agent, small width, float32 on CPU or one GPU | The paper uses an 8-layer width-384 network on GPUs |

**Unresolved source details:** Adam epsilon/betas, iteration origin, EMA
initialization, quantile interpolation and exact bootstrap pseudocode at a
collection boundary. The choices above are explicit implementations, not
recovered details. Supplement S5–S7 recipes are not blended in; DeepNash's
R-NaD is not implemented. Departures also include post-normalization
attention (the paper specifies pre-normalization), the value token as a local
choice, and missing ordered past-move inputs.

Before any scientific comparison of PPO, categorical and scalar move learning,
freeze a protocol, independent seeds, budgets, world and code identities,
held-out paired deals with both seat/deck assignments and the raw/EMA choice;
keep all failures and compare overlapping cost intervals. The technique
inventory is [ataraxos-technique-screen](../experiments/ataraxos-technique-screen.md);
ablations are in [ataraxos-mtg-ablations](../experiments/ataraxos-mtg-ablations.md).

## Where the code lives

| Step | Code |
| --- | --- |
| Iteration loop, EMA, checkpoints | `execute_regime` in [`training/execution.py`](../manabot/training/execution.py) |
| Play and batch boundaries | `SeatRoutedCollector` in [`training/rollout/collection.py`](../manabot/training/rollout/collection.py) |
| Advantages | `transition_gae` in [`training/learning/returns.py`](../manabot/training/learning/returns.py) |
| Value targets, loss, update | `_targets`, `damped_policy_loss`, `update_damped_iteration` in [`training/learning/damped_policy.py`](../manabot/training/learning/damped_policy.py) |
| Filter | [`training/learning/selection.py`](../manabot/training/learning/selection.py) |
| Magnet | [`training/learning/references.py`](../manabot/training/learning/references.py) |
| Defaults | `DampedPolicyLearning`, `TrainSelfPlay` in [`training/regime.py`](../manabot/training/regime.py) |
| Behavior pinned by tests | [`tests/training/test_credit_assignment.py`](../tests/training/test_credit_assignment.py) |
