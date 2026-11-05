# Start here

A manabot learns Magic by playing itself. It sees only what a player would see,
chooses among the legal actions, and finds out who won when the game ends.
Nobody tells it which plays were good.

This guide starts with the whole system and zooms in, stage by stage, to the
code. Earlier two-deck measurements put a game at about
[40 to 120 decisions per seat](../experiments/exp-08-two-deck-matchup.md#game-length-and-decision-profile),
and one result has to teach all of them. Three more things make the problem
hard:

- **The opponent learns too.** The best play depends on a policy that keeps
  changing, so a good win rate against the latest self says little.
- **Actions aren't a fixed list.** A legal action can name any object in play,
  so the network scores what's offered rather than filling a fixed output layer.
- **Cards change the rules.** Each pairing of decks is close to its own game,
  and a new card can add a rule.

- [The system](#the-system)
- [One iteration](#one-iteration)
- [Play](#play)
- [Assign credit](#assign-credit)
- [Filter](#filter)
- [Update](#update)
- [Outside the loop](#outside-the-loop)
- [What win rate hides](#what-win-rate-hides)
- [Open questions](#open-questions)
- [Where to go next](#where-to-go-next)

## The system

Four parts matter. The engine runs games, the network plays them, the learner
improves the network, and evaluation measures the result from outside the loop.

![The rules engine sends each seat's view to the policy network, which returns a chosen action. Decisions and results go to the learner, which sends new weights to the network. Checkpoints go to evaluation, which never feeds back into training.](images/system.svg)

Each part has its own guide: [the rules engine](engine.md),
[observations and the policy network](observation-and-model.md),
[how a manabot learns](learning.md) and [evaluation](evaluation.md). This page
follows the learner, because that's where the other three meet.

## One iteration

Zooming in on the loop, each iteration collects a fresh batch, computes its
targets once, makes one pass over it, and throws it away.

![One training iteration: play, assign credit, filter and update, then repeat with new weights. Weights also feed an EMA, checkpoints and evaluation, which never feed back into the loss.](images/learning-loop.svg)

The next four sections take these stages in order. Each one goes from the idea
to the code. If reinforcement learning is new to you, value and advantage are
the terms to learn first.

## Play

Every stream steps in lockstep, so one forward pass chooses actions for all of
them. Both seats learn by default, so one game produces two trajectories with
opposite results.

![An interleaved game of decisions by seats A and B, with one forced decision, splits into one row sequence per seat. Seat A's last row has reward +1 and seat B's has -1.](images/game-to-trajectories.svg)

### Decision and trajectory

A decision is a choice with at least two legal actions. A trajectory is one
seat's recorded decisions in one game, in order.

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

- The engine plays forced choices (`skip_trivial`), so every count here
  excludes them.
- Each stream and seat has its own slot. A row's successor is that seat's next
  decision, however many opponent decisions came between.
- A row stays pending until that seat decides again (reward 0) or the game
  ends (the result).

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

- There are two head types. `categorical_wdl` predicts loss, draw and win
  probabilities, and its value is P(win) − P(loss), so a draw counts as 0.
  `scalar`, used by the baseline recipe, predicts the outcome directly.
- The critic sees the same viewer-safe observation as the policy, not the
  opponent's hidden cards.
- Values are recorded during collection and stay constant through the update.

## Assign credit

The only reward is the result on a trajectory's last row. Credit is computed
once for the whole batch, before any weights change. Each row compares its
value with what came next and adds a fraction λ of the next row's advantage, so
the result's direct weight fades quickly.

![Three curves show lambda to the power of decisions before game end. At lambda .5 the weight is near zero by ten decisions, at .8 by about thirty, and at .95 it is still about .13 at forty.](images/credit-fade.svg)

Here's that recursion on a short game. A seat's estimates rise and it wins,
with λ = .5:

| Decision | Value | Next | δ | Advantage |
| --- | ---: | ---: | ---: | ---: |
| 1 | .1 | .3 | +.2 | .45 |
| 2 | .3 | .6 | +.3 | .50 |
| 3 | .6 | win | +.4 | .40 |

Ten decisions before the end, the result's weight is about one part in a
thousand. Priority passes count as decisions, which makes ten a short stretch
of a game. An early land drop or attack learns almost entirely from how the
value moved afterward, so the value network does most of the teaching.

### Advantage

A generalized advantage estimate with no discounting:
$A_t = \delta_t + \lambda A_{t+1}$, where $\delta_t = r_t + V_{t+1} - V_t$.

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

- Time advances per decision of the same seat. Opponent decisions add no steps.
- `live` cuts both the bootstrap and the trace at a game end, so credit never
  crosses into another game.
- The policy uses λ = .5, and value targets use λ = .8.
- Advantages are never normalized. They stay in outcome units, which gives the
  filter's .01 floor a fixed meaning.

## Filter

Most rows matched what the value predicted, so only about a quarter go on to
the update. These are the last five decisions of a real lost game, from a batch
that kept 81 of 324 rows on an untrained network.
`uv run python scripts/inspect_batch.py` prints this view in a few seconds:

| To end | Value | Advantage | Kept |
| ---: | ---: | ---: | --- |
| 4 | −.132 | −.054 | |
| 3 | −.155 | −.063 | |
| 2 | −.160 | −.118 | Yes |
| 1 | −.056 | −.443 | Yes |
| 0 | .003 | −1.003 | Yes |

The rule is |A| at or above both the batch's .75 quantile and .01.

From `selected_moves` in
[`manabot/training/learning/selection.py`](../manabot/training/learning/selection.py):

```python
magnitude = advantages.detach().abs()
threshold = max(float(torch.quantile(magnitude, quantile)), minimum)
return magnitude >= threshold
```

- The threshold is computed across the whole batch, and all ties survive.
- By default, a dropped row leaves the policy loss, both KL terms and the value
  loss. `filter_scope="actor"` keeps every row for the value loss.
- When most advantages fall below .01, the floor sets the cut and far fewer
  than a quarter of rows survive.

## Update

The update makes one pass over the retained rows, in batch order. Everything
computed so far is now a constant. Gradients reach only the live network, and
no gradient travels through game time.

![Constants from collection feed the policy, damping and value terms. Gradients reach only the live network, through one shared encoder.](images/gradient-flow.svg)

The loss combines established methods, listed under
[influences](#influences) below. For each retained row, with $r$ the ratio of
current to behavior probability for the chosen action:

$$L = -\min\big(rA,\ \mathrm{clip}(r, .8, 1.2)\,A\big) + .1\,\mathrm{KL}(\pi \,\|\, \pi_{\text{behavior}}) + \tau_i\,\mathrm{KL}(\pi \,\|\, \pi_{\text{magnet}})$$

### Behavior policy

The frozen weights that played the batch. The update measures how far the
learner has moved from them.

From `damped_policy_loss` in
[`manabot/training/learning/damped_policy.py`](../manabot/training/learning/damped_policy.py):

```python
ratio_logs = (
    (logs - behavior_logs.detach()).gather(-1, actions[:, None]).squeeze(-1)
)
surrogate = clipped_surrogate(ratio_logs, advantages, clip)
```

- Collection stores log-probabilities for every legal action, because the KL
  term needs the full distribution and not only the sampled action.
- Ratios matter even in a single pass, because earlier minibatches have
  already moved the learner.
- This is a per-action correction. There's no trajectory-level importance
  weighting.

### Magnet

A fixed reference policy, built for each row from its legal actions. The KL
term pulling toward it has strength $\tau_i = .05 / i^{.3}$.

From `reference_distribution` in
[`manabot/training/learning/references.py`](../manabot/training/learning/references.py):

```python
counts.scatter_add_(-1, types, valid.to(counts.dtype))
weights = valid / counts.gather(-1, types).clamp_min(1)
return weights / weights.sum(-1, keepdim=True)
```

- Each action type present gets equal mass, split evenly among that type's
  legal actions.
- The penalty is a reverse KL, KL(π ‖ magnet), summed over all legal actions.

### Influences

None of these methods is new here. Each row says what we took and what changed
for Magic.

| Source | What we use | What changed |
| --- | --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347) | The clipped ratio | One pass in batch order, with no advantage normalization |
| [GAE](https://arxiv.org/abs/1506.02438) | The advantage recursion | No discounting, and steps are same-seat decisions |
| [Magnetic mirror descent](https://arxiv.org/abs/2206.05825) | The KL pull toward a fixed reference | The reference is uniform over action types, then over actions |
| [Ataraxos](https://doi.org/10.1038/s41586-026-11036-y) | The combination: both KL terms, filter, schedules, categorical value, EMA | No learned setup, no ordered past moves by default, and a much smaller network |

The rules engine, viewer-safe observations, scoring of legal actions over game
objects, and paired evaluation were built for this project.

Each technique is a switch we test on Magic, not an established result here.
Actor-only filtering, for example, measured +3.33 points with an interval of
[−4.33, +11.33], at 1.45× the wall time
([technique screen](../experiments/ataraxos-technique-screen.md)). Nothing yet
shows that this recipe approaches equilibrium in Magic. The complete comparison
with the source is the [fidelity table](learning.md#fidelity-table).

## Outside the loop

Checkpoints leave the loop bound to a world: a frozen version of the rules,
cards and observation encoding. They load only where that binding matches.

From `validate_checkpoint_world` in
[`manabot/model/world.py`](../manabot/model/world.py):

```python
if _digest(current_binding_form(binding)) != _digest(expected):
    raise ValueError(
        "checkpoint world/setup/input binding differs from this runtime"
    )
```

- The check compares a digest of the world, setup and input binding, so a
  matching world number alone isn't enough.
- Results don't carry across worlds, and a run on an open world makes no
  comparability claim ([WORLDS.md](../WORLDS.md)).

## What win rate hides

An earlier search pilot used flat Monte Carlo over sampled hidden cards. It
beat random in 91.7% to 99% of games, and more simulations made it stronger
([exp-02](../experiments/exp-02-flat-mc.md)).

Then a test gave it Pyroclasm against two Gray Ogres, with two more in the
opponent's hand. Waiting a turn wipes four creatures rather than two, for 4
damage at 20 life. Search cast it on turn 1 in 300 of 300 runs at every
strength. Uniform random waited by accident and passed 23% of the time
([exp-09](../experiments/exp-09-control-competency.md)).

The likely cause is strategy fusion: each sampled world is judged as if the
player could see it, so a line that needs patience under uncertainty never
looks best. One replay shows the misplay, while the aggregate number doesn't.
That's why strength here is measured against held-out opponents and specific
test positions, and why exploitability is reported only as a lower bound from
attacks we can afford ([evaluation](evaluation.md)).

## Open questions

- **How far back should a result reach?** In one baseline, Allies reached
  87–88% against a scripted opponent while Lessons climbed slowly from 5/150
  to 32/150 ([current baseline](../experiments/current-baseline.md)). Credit
  reach might be part of the reason, but that's untested. A study of batch
  closure and λ is under way.
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

## Where to go next

| To understand | Read |
| --- | --- |
| What the engine guarantees: legality, hidden information, forks, replay | [The rules engine](engine.md) |
| What the network sees and how it scores actions | [Observations and the policy network](observation-and-model.md) |
| The loop in full, including batches that cut games off | [How a manabot learns](learning.md) |
| Running, monitoring and recovering a training run | [Training](training.md) |
| How strength is measured | [Evaluation](evaluation.md) |
| How a question becomes a report | [Experiments](experiments.md) |
| Search and beliefs over hidden cards | [Search and beliefs](search-and-beliefs.md) |
| What we are doing now and what we know | [Research program](../manabot/RESEARCH.md) |
