# How a manabot learns

A manabot starts knowing only the legal actions at each decision. It plays
Magic against copies of itself, and the only feedback is who won. This guide
follows one training iteration from the first decision to the weight update,
and explains how a win at the end of a game becomes a change to the network.

It describes the default recipe, `DampedPolicyLearning`, which adapts
[Ataraxos](https://doi.org/10.1038/s41586-026-11036-y) to Magic. For running
training, see [training.md](training.md). For the line-by-line comparison with
the paper, see its [fidelity table](training.md#fidelity-table).

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
- [What the evidence says](#what-the-evidence-says)
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

## Can a batch contain unfinished games?

It depends on how the batch is closed, and there are two ways.

![Complete-game batches let each stream finish its game and idle until every stream is done. Exact-quota batches stop every stream at the same decision count, pausing unfinished games and bootstrapping them from a value estimate.](images/batch-boundaries.svg)

**Complete games: no.** This is the default, because two-seat learning always
uses it. One-seat runs use it too when they set `games_per_stream` or a game
error policy. `transitions` is a floor on a stream's decisions, counting both
seats. A stream that has passed the floor stops at the end of its current game.
Rows are packed into one `[rows, 1]` column with an end flag after each
trajectory, and row counts vary by batch. The collector asserts that the last
row is an end, and `bootstrapped_tail_fraction` is 0.

**Exact quota: yes.** With one learning seat and neither of those settings,
each stream collects exactly `transitions` learner decisions and the batch has
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

![Constants from collection feed the policy, damping and value terms. Gradients reach only the live network, through one shared encoder.](images/gradient-flow.svg)

Rewards are not backpropagated. Advantages, value targets, behavior
probabilities and the magnet are constants by this point. Each row is an
independent forward pass on its stored observation, with no backpropagation
through game time. The game's result reaches the weights only as those numbers.

Both losses train one shared encoder. The gradient norm is clipped to 0.267
before each Adam step.
Step sizes depend on the absolute iteration $i$:

$$\text{lr}_i = \mathrm{clip}(0.5\,/\,i^{1.1},\ 5\times10^{-6},\ 10^{-4}) \qquad \tau_i = 0.05\,/\,i^{0.3}$$

An `IterationSchedule` can rescale $i$ so a short run covers the same curve.

The PPO-style `Learning` rule remains as a control. It normalizes advantages
and shuffles minibatches, and may take several epochs.

## Weights, checkpoints and evaluation

Training keeps two sets of weights. The **raw** weights are the learner, and
they play the next batch. The **EMA** is an average updated once per iteration
at 0.999, including iterations the filter emptied. A recipe can collect with the
EMA by setting `behavior="ema-self"`; otherwise it is only an evaluation output.

Checkpoints are bound to their [world](../WORLDS.md). Evaluation runs outside
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

**Why does a two-seat batch wait for slow games?** A stream past its floor idles
until the last stream finishes, so the batch can be built from complete games.
That idle time is the price of having real outcomes on every trajectory.

## What the evidence says

Most of this recipe is inherited from the paper and is being tested on Magic
one switch at a time. These are the results the guide leans on:

- **Terminal reward only.** Our shaped rewards lost to it by about 15 points
  ([exp-04](../experiments/exp-04-potential-shaping.md)), in an earlier world.
- **Technique screens.** Each Ataraxos component has its own status in the
  [technique screen](../experiments/ataraxos-technique-screen.md) and
  [ablations](../experiments/ataraxos-mtg-ablations.md). The recipe is not a
  demonstrated strength result as a whole.
- **Scale.** The largest run so far is in the
  [scale ladder](../experiments/scale-ladder.md), summarized in the
  [README](../README.md#scale).

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
