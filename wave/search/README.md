# Search

## Vision

Make search the teacher.

managym sustains 183k SPS with zero-copy observations. That is not a
data-collection loop — it is a search substrate, and it has never been used as
one. This wave turns lookahead on, uses it to generate training targets, and
replaces every hand-tuned strategic prior with something the search derived
itself.

Three things fall out, and they are the reason to do this before anything else:

1. **Search fixes credit assignment properly.** Reward shaping was a bribe paid
   to avoid doing lookahead. A search that plans to the end of the game *finds*
   the win signal; the policy distills it. Shaping gets deleted, not tuned.
2. **Search is a strength ratchet.** Search-at-N simulations is a continuously
   tunable opponent. `policy-alone beats search-at-N` is a metric that cannot
   saturate and cannot be gamed by picking a weak opponent.
3. **Known decklists make determinization valid.** Both players' lists are
   fixed and mutually known, so hidden information is (a) library order, which
   is stochastic and samplable, and (b) opponent hand, which is tightly
   constrained by public history plus card counting. This is bridge/skat
   territory, not poker.

### Not here

- Self-play, opponent pools, league play (needs a strength metric first — that
  is what this wave builds)
- Auxiliary prediction heads (a good answer to a problem search solves better;
  revisit only if search fails)
- Belief-conditioned / safe search, exploitability-minimizing methods
- Cube design, deck construction, drafting
- Cardset expansion beyond what a strategically non-degenerate deck requires
- GUI, publishing

## Strategy

### The training deck cannot express strategy

`manabot/infra/hypers.py:13-20` defines the deck both players use:

```
Mountain 12, Forest 12, Llanowar Elves 18, Grey Ogre 18
```

No instants. No removal. No counterspells. No card draw. No mass removal. The
only legal decisions are play-land, play-creature, attack, block.

The engine does not have this limitation. `cardsets/alpha.rs` implements
Lightning Bolt (`:115`, instant-speed `DealDamage` at `CreatureOrPlayer`) and
Counterspell (`:127`, `CounterSpell` at `TargetSpec::Spell`). `agent/action.rs:8`
carries `DeclareBlocker` and `ChooseTarget`. `flow/resolution.rs:13` resolves a
real stack.

Every first-light finding — including "reward shaping is required" — is a
finding about the vanilla-creature game, not about Magic. Fix the deck first, or
search will be searching a game with nothing in it.

Minimum mechanic set for the game to *contain* the phenomena worth searching:

| Mechanic | Why | Have it? |
| --- | --- | --- |
| Instant-speed removal | Attacking becomes a decision under uncertainty | Lightning Bolt |
| Counterspell | Holding up mana means something; bluffing exists | Counterspell |
| Card draw | Attrition becomes a strategy; control decks exist | **missing** |
| Mass removal | Overextension is punished; trap cards exist | **missing** |
| Blocking | Combat math is non-trivial | `DeclareBlocker` |

Card draw and mass removal are the only genuinely absent pieces, and they are
the two that make non-aggro strategy possible. Scope: add the minimum, not a
cardset.

### Inference, not the engine, is the bottleneck

`reports/sps-closeout.md`: 183k SPS env-only, **2.0k SPS with inference, 97% of
step time in torch** — on a 64-hidden-dim network that is a rounding error in
FLOPs. This is per-step kernel-launch and sync overhead, not compute. It gates
model scale *and* search throughput simultaneously. Nothing downstream moves
until it does.

### MCTS in Magic is only tractable after forced-move collapse

~200-step episodes with a priority system where the overwhelming majority of
decision points have exactly one sensible action. A naive tree drowns in passes.
No general auto-pass exists in the engine today (only a narrow
`combat_actions.rs:264` special case).

Never surface a state with one legal action. Expectation: real agent decisions
per game drop from ~200 into the 30-80 range, which is squarely tractable for
determinized search on 1990s hardware, let alone this engine.

Do the collapse before writing a line of MCTS.

### Search does not need a good value function. It needs an unbiased one.

The instinct is that a strong value estimator is a prerequisite for search. It
is not — not here. AlphaZero needed one because Go has no meaningful cheap
terminal rollout. Magic, post-collapse, is 30-80 real decisions on a 183k SPS
engine: **roll out to the true terminal reward.** The rollout policy is the leaf
evaluator. GIB did this in bridge in 1999.

What V actually buys is throughput. A terminal rollout costs ~40 forward passes;
a value net at a leaf costs 1. That is ~40x more simulations per unit compute,
and it splits search into two products with different requirements:

- **Search-as-opponent** (the strength ratchet, goal 5). Needs no V. Terminal
  rollouts, true reward, unbiased by construction. Ship first.
- **Search-as-teacher** (goal 4). Needs V, because distillation needs millions of
  searched decisions and 40x matters.

The rollout search is therefore also **the measuring instrument for V** — it
yields an unbiased estimate of `P(win | s)` at any state, for free, from work
already scheduled.

The condition to gate goal 4 on is not "V is accurate." It is:

> **search-with-V beats V-greedy.**

Search is a policy improvement operator. AlphaZero bootstraps from a noise-valued
net because search-plus-noisy-V still improves on noisy-V alone. Accuracy is not
required; *improvement* is.

#### Assessing V

1. **N-scaling curve** (one afternoon, and it answers the question). Win rate of
   search-at-N vs. a fixed reference, N in {1, 10, 100, 1k, 10k}.
   - *monotone increasing* — search works, V is sufficient, ratchet acquired
   - *flat* — search adds nothing; V uninformative, or branching too wide for N
   - *rises then falls* — the money plot. V is biased and deeper search is
     finding and exploiting V's own errors. Tells you where to look.
2. **V vs. rollout ground truth**, in increasing order of importance:
   - *calibration* — is `V(s) ~ P(win|s)`? reliability diagram. nice to have.
   - *ordering* — Spearman vs. rollout ranking. **Matters far more**: MCTS only
     compares states, so uniform overconfidence is harmless and misordering is
     fatal.
   - *bias by strategic bucket* — error conditioned on state type ("ahead on
     board, behind on cards" vs. "behind on board, holding removal"). This is
     the one that catches the pathology below.
3. **Fix what explained variance measures.** Today V is trained to predict
   *shaped returns*, so the logged EV (0.434, `reports/first-light-run-1.md`)
   partly reflects the net predicting its own land-play bonuses. That number
   cannot assess fitness for search, because it is not estimating `P(win)`.
   Deleting shaping (goal 4) makes the question well-posed for the first time.

Caveat: **distribution shift.** V is fit on states the current policy visits;
search visits states the policy would not — that is the point of searching. Fit
V on search-visited states, and expect off-policy V to be worse than on-policy
EV suggests.

#### The aggro bias is structural, not incidental

Random rollouts fire the Counterspell at a random moment and get nothing. They
Wrath away their own board. So rollout-derived value **systematically
undervalues holding interaction** — it undervalues control.

That is the third appearance of the same bias in this project:

| Vector | Mechanism |
| --- | --- |
| `opponent_life_loss_reward` | pays for damage, so race |
| Weak-policy deck evaluation | rates decks it can pilot; aggro is easy to pilot |
| Random rollouts | cannot represent a plan; holding a card is a plan |

Every cheap approximation available to this project has an aggro bias, because
cheap methods cannot represent plans. Mitigation for rollouts: roll out with the
trained policy, not a random one, and compare V-from-random against
V-from-policy on identical states. The disagreement will concentrate in exactly
the control states.

### Strategy fusion is expected, and is the finding

Perfect-information Monte Carlo assumes it will know the hidden state at future
decision points (Frank & Basin, ~1998: *strategy fusion*, *non-locality*). The
consequence in Magic is exact: **the search will never play around a trick and
never represent one.** It cannot bluff, because in every determinized world it
already knows whether the bluff works.

That is not a bug to fix in this wave. It is a quantity to measure. The gap
between determinized search and anything belief-conditioned is the gap between
competent Magic and artful Magic, and it should be on a chart before anyone
tries to close it.

## Goals

0. **Real training deck.** Add card draw and mass removal to the engine; build a
   deck with removal, counterspells, and interaction. One-line change to
   `_default_deck()` plus a small amount of Rust. Expect every first-light
   metric to regress. That is the point — it restores dynamic range.
1. **Batched inference.** 2.0k → 50k+ SPS with inference. Batch across envs,
   eliminate per-step syncs, trace/compile the actor.
2. **Forced-move collapse in the engine.** Single-legal-action states never
   reach the agent or the tree. Report real decisions per game.
3. **Determinized search, rollouts first.** Flat Monte Carlo to terminal — no
   value function needed, a day's work, and it may go further than expected.
   Then a tree. This is both the reference opponent and the value oracle.
4. **Gate: assess V.** Plot the N-scaling curve. Score the value head against
   rollout ground truth (ordering first, calibration second, bias-by-bucket
   third). Do not proceed to distillation until **search-with-V beats
   V-greedy** — that condition, not V's accuracy, is what makes search a policy
   improvement operator.
5. **Search as teacher.** Distill search targets into the policy. Set every
   shaping coefficient to zero. Confirm pass-collapse does not return. If it
   does, search is not finding the win either — which is far more useful than
   the shaping was.
6. **`policy-alone vs search-at-N` as the only headline metric.** Exploitability
   probe as the sanity check that the policy has not merely overfit to its own
   search.

## Risks

- **Deck change invalidates first-light's conclusions.** Intended. "Shaping is
  required" must be re-derived on a deck that has strategy in it, and may not
  survive contact with search.
- **Branching factor after adding interaction.** `ChooseTarget` and
  instant-speed responses deepen the tree exactly when we start searching it.
  Forced-move collapse buys headroom; measure before assuming it is enough.
- **Search too slow even after the inference fix.** Mitigation: flat rollouts
  before a tree; measure sims/decision achievable at target wall-clock before
  committing to MCTS.
- **Determinization validity rests on known decklists.** Documented assumption,
  not a permanent one. If decklists ever become hidden, this entire approach
  needs revisiting.
- **Distillation overfits to search's blind spots.** A policy that perfectly
  imitates a strategy-fused search inherits the inability to bluff. The
  exploitability probe is the tripwire.
- **Weak-policy archetype bias.** A policy too weak to pilot control will rate
  control decks as bad. Relevant the moment anything downstream reads deck
  strength off this policy. `opponent_life_loss_reward` installs exactly this
  bias by hand and must be gone before then.

## Metrics

- **Real agent decisions per game** after forced-move collapse (baseline ~200;
  target < 80)
- **SPS with inference enabled** (baseline 2.0k; target ≥ 50k)
- **Policy-alone win rate vs. search-at-N**, N ∈ {1, 10, 100, 1000} (target:
  policy with no search at inference beats search-at-100)
- **N-scaling curve shape** for search-at-N vs. a fixed reference (target:
  monotone increasing; a rise-then-fall means V is biased and search is
  exploiting its errors)
- **Search-with-V vs. V-greedy** win rate (gate for goal 5; must exceed 50%, or
  search is not improving on the value head and distillation is pointless)
- **Spearman correlation of V against rollout ground truth**, reported overall
  *and per strategic bucket* (aggro-favoring vs. control-favoring states); a
  large per-bucket gap is the aggro-bias tripwire
- **Explained variance against terminal outcome only**, not against shaped
  return (the current 0.434 measures the wrong object)
- **Exploitability**: win rate of a from-scratch agent trained against the
  frozen checkpoint (target: < 65%; ≥ 80% means the policy is farmable and the
  headline metric is lying)
- **Shaping coefficients all zero**, with no return of pass-collapse
  (`passed_when_able` stays low, `landed_when_able` stays high)
- **Bluff rate**: frequency of holding up mana with no instant in hand.
  Expected ~0 for determinized search. Measured, not fixed.

## References

- [AlphaZero](https://arxiv.org/abs/1712.01815) — search generates targets no
  bootstrap can
- [KataGo](https://arxiv.org/abs/1902.10565) — efficiency tricks for the same loop
- Frank & Basin, *Search in games with incomplete information: a case study
  using Bridge card play* (Artif. Intell. 1998) — strategy fusion, non-locality
- [Information Set Monte Carlo Tree Search](https://ieeexplore.ieee.org/document/6203567)
  (Cowling, Powley, Whitehouse, 2012)
- Ginsberg, *GIB: Steps toward an expert-level bridge-playing program* (IJCAI
  1999) — determinization working embarrassingly well in practice
- [reports/sps-closeout.md](../../reports/sps-closeout.md) — where the 2.0k
  number comes from
- [reports/first-light-run-1.md](../../reports/first-light-run-1.md) — what was
  measured on the vanilla-creature deck
