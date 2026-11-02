# Search and beliefs

Search plans one decision by playing out hidden-world samples on exact copies of
the match; beliefs supply those samples. managym owns the hidden-hand domain and
exact branching; manabot owns probability, sampling policy and search. Nothing
here fits policy parameters; training consumes search targets.

- [Forking](#forking)
- [Search players](#search-players)
- [Local policy search](#local-policy-search)
- [Possible worlds and tracked beliefs](#possible-worlds-and-tracked-beliefs)
- [Belief sampler](#belief-sampler)
- [Sampler quality reports](#sampler-quality-reports)
- [Unsupported and unmeasured](#unsupported-and-unmeasured)

## Forking

Search forks by exact full clone: `Env.selected_fork` (`managym/src/agent/env.rs`)
clones the whole `Game`. `manabot.sim.search.branch` binds it as driver
`full_clone/current_game_v1` with one structured Command mutation seam;
`legacy_env_clone/reference_v1` remains a named differential oracle, never a
fallback. `managym/src/search_state.rs` defines the logical search-state witness
(schema 2) that conformance checks compare: rules facts only, no pointers,
layout, timings or RSS.

The managym undo journal and the clone-plus-undo and event-page copy-on-write
branching drivers were removed because undo was incomplete (the structured
Command path was unjournaled) and forking is the strategy (Jack Heart,
2026-10-30). Before removal, both failed their pre-registered thresholds against
full clone; see [search branching decision v1](benchmarks/search-branching-decision-v1.md).

## Search players

`manabot.sim.search.flat_mc.make_player` builds every search player from a spec
`kind`: `search` (flat Monte Carlo), `determinized_puct`, `agent_puct`,
`policy_search`, `local_update`, `value_greedy`, `value_search`, `exact_range`,
`uniform_range`, `checkpoint` and `random`. Arena registrations live in
`manabot.sim.registration`.

- **Flat Monte Carlo** (`flat.py`, `flat_mc.py`). `Env.flat_mc_scores` scores
  each legal action over W determinized worlds by R random playouts and plays
  the argmax; no network. A determinized world redraws the opponent's hand from
  its unseen pool and reshuffles both libraries, keeping public state.
  `FlatSearchPolicy` imports neither NumPy nor Torch; Academy's search opponent
  and the arena use the same class.
- **PUCT** (`puct.py`). Separate trees per determinized world (each receives at
  least one of the total `sims`); not information-set-consistent. Play chooses
  maximal visits, then root-perspective Q, then offer order. Values are win
  scores (loss 0, draw .5, win 1): signed scalar, categorical and compound
  critics convert by `(value + 1) / 2` and invert for the other seat; only
  checkpoints declaring `bc.value_semantic="win_logit"` use a sigmoid. Untagged
  scalar heads are rejected for learned values; `value_mode="neutral"`
  substitutes .5 and keeps policy priors. Compound priors marginalize the
  retained decoder prefix onto next canonical Commands (`compound.py`); call
  `start_game(env, seat)` at game boundaries. `agent_puct` arena registrations
  bind `puct_source_sha256()` and echo `branch_audit`. Unvisited Q=.5 is a
  placeholder; `forward_calls` counts flat forwards only.
- **Conditional search** (`conditional.py`) compares one root under paired
  world conditions (`True`, has/lacks a card, a query and its negation).
- **Value players** (`value.py`) score successors or rollout leaves with a
  frozen value head.

## Local policy search

`manabot.sim.search.local_update.LocalUpdateTeacher` admits one checkpoint by
exact SHA-256. Compound checkpoints use the decoder's signed prefix critic;
ordinary scalar artifacts must declare `bc.value_semantic = signed_outcome`
(TrainingRegime writes it for self-play); untagged historical scalar critics are
rejected.

For legal actions a it maximizes
`sum p(a) Q(a) - alpha KL(p || reference) - beta KL(p || base)`, giving logits
`(Q + alpha log(reference) + beta log(base))/(alpha+beta)` with a uniform
reference over semantic offers. This follows Ataraxos supplement S3.7
equations (7)–(8), checked on 2026-10-04
([supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf));
budget, horizon, reference and value representation are MTG adaptations, and
model mismatch and truncated rollouts preclude an equilibrium guarantee.

Each sampled world rolls out every positive-support root action with
frozen-policy moves to the declared depth, encoding only the acting player's
Observation. Terminals are -1/0/+1; nonterminal leaves are the actor's signed
estimate, reversed for the other seat, unclipped. Because beta > 0, the target
is restricted to positive base support: zero-base actions get zero target, no
rollouts and `null` Q (`regularized-local-update/v3`; v1/v2 receipts still
read). `full_probability`, `worlds` and `cheap_worlds` give reproducible
per-root full/cheap allocation, inspired by
[KataGo's playout-cap randomization](https://arxiv.org/abs/1902.10565).

`sampling` selects the world source:

- `compatible_prior` (default): the physical compatible-deal measure.
- `learned`: a [belief sampler](#belief-sampler) fitted on exactly the same
  policy and raw/EMA weights; admission checks sampler bytes, dataset, schema,
  world and generating-policy digest. Receipts are labeled
  `approximate-learned-hand`; values average samples without importance
  weighting.
- `belief`: the tiny exact-reference tracker; enumerates (its `max_support` cap
  applies after enumeration) and is the only mode supporting conditional
  queries.

`manabot.sim.search.sampling.SearchHandSampler` is the injectable boundary:
`PreparedHands` binds distribution, viewer history, vocabulary and native
constraint snapshot; `HandBatch` holds counts and joint log probabilities and
is validated before `Env.materialize_sampled_hand` (native
`HiddenHandConstraints::materialize_hand`) places the hand directly, without
enumeration. Seeds fix hidden library order; counts do not model a posterior
over it. Collection and search share `public_sampler_input`.

`LocalUpdateReceipt` retains offers, base/Q/reference/target, coefficients,
sampling evidence, identities, seeds and exact branch Command tapes;
`verify_replay` re-executes from an exact root. Intermediate compound prefixes
also need the original-root cursor. Arena admission binds
`local_search_source_sha256()`. Deadlines are cooperative and charge
preparation, sampling and rollouts: an overrun yields no target, never a
policy-only substitute.

Compound search (`compound.py`) supports attacker include/exclude, single-target
casts, independent blockers, action-aligned fallback offers and ordered
waterbend payment subsets, without enumerating subsets; a legal tap outside the
retained ordered prefix has exact zero mass. Menace, effectful payments, kicker
and ward use the native sequential fallback.

## Possible worlds and tracked beliefs

`managym/src/possible_worlds.rs` owns the viewer-relative hidden-hand domain:
the opponent's unseen pool is Hand ∪ Library ∪ face-down foretold cards
(matching `Game::determinize`), a world fixes only the hand, and weights are
exact compatible-deal counts, never probabilities. It also defines the typed
`WorldQuery` grammar, deterministic materialization into exact branches, and
`HiddenHandConstraints` (public pool, known hand minima, hand size).

`manabot/belief` holds probability over that domain: `range.BeliefState`
normalizes over one `PossibleWorldSpace`; `tracker` updates it from viewer
observations and semantic transitions; `likelihood` pins policy likelihoods for
public commitments; `player`, `agent` and `runtime` wrap belief and
compatible-prior players; `encoding` projects beliefs for a policy; `audit`
scores against known truth outside the acting path. The tracker supports only
hand-multiset and chance transport, not an exact posterior over every history
or library order.

Academy no longer tracks a posterior per interactive command: that live belief
runtime was removed because Jack Heart is not confident the belief code works.
The code in `manabot/belief` remains.

## Belief sampler

The sampler (`manabot.belief.sampling.AutoregressiveBeliefSampler`) learns a
joint distribution over opponent hand counts without listing compatible hands.
Its input (`SamplerInput`) is the viewer's public pool, known hand minima, hand
size and encoded public history; actual hands are supervision only. Library
order and other hidden state are outside its domain.

```bash
uv run manabot train --regime-name belief_sampler --seed 421 --out .runs/belief-pilot
```

`experiments/belief_sampler.py` is a bounded pilot: a tiny self-play policy,
three collected games, four sampler steps. Use a fresh output directory each
time.

- **Stages.** `collect_belief` names an earlier policy stage, chooses `raw` or
  `ema`, and freezes those checkpoint bytes for both seats. `train_belief` fits
  only the sampler on that immutable dataset, so there is no circular
  policy/belief dependency. Flat observation-only policies and compound
  policies (raw only) are supported; belief-conditioned behavior policies are
  rejected. Collection uses the checkpoint's saved observation capacity and
  fails explicitly if a game exceeds it.
- **Dataset** (`sampling_data`). Complete games with disjoint
  train/validation/test membership, alternating deck assignments, policy and
  world identities, public inputs captured before acting, and private labels.
  These contain real hidden hands and belong in private training storage,
  never viewer-facing responses.
- **Estimator.** The decoder samples one definition count at a time in
  vocabulary order, masking by known minima, remaining slots and pool capacity;
  training minimizes the joint negative log probability of the actual hand,
  preserving count correlations. Its base measure is the physical compatible
  deal (remove known cards, deal unknown slots without replacement), matching
  managym's possible-world weights, not uniform over count vectors; learned
  terms correct its conditional log odds. History features summarize typed
  public commitments only. History dropout is a training treatment, not
  immunity to baiting.
- **Checkpoints** (`sampling_fit`, `checkpoint.load_belief_sampler`) bind exact
  dataset, behavior policy, schema and world; they are not playable policy
  checkpoints.

The 2026-10-04 pilot completed three games (303 decisions) in 17.41 s; its 840
held-out samples had zero support violations. That is pipeline evidence only.

## Sampler quality reports

Offline, from a completed run's immutable dataset and sampler:

```bash
uv run python -m manabot.reports.belief \
  --dataset /path/to/evaluation-dataset.json \
  --training-dataset /path/to/sampler-training-dataset.json \
  --checkpoint /path/to/sampler.pt \
  --checkpoint-sha256 "$SAMPLER_SHA256" \
  --samples 16 64 256 --seed 421 \
  --out .runs/sampler-report.json
```

Omit the three learned-artifact arguments for a physical-prior-only report.
`report_saved_sampler` returns a `SamplerQualityReport` per split and sample
count: observed-label joint NLL (nats), card-presence Brier and ten-bin ECE,
adjacent-pair conjunction Brier, support violations, sampling seconds and
allocation. It retains per-game evidence; sampling streams derive from
`(sampling seed, game ID)`. Evaluation data overlapping sampler fitting games is
rejected; malformed inputs fail rather than being dropped. NLL is a proper
score on observed labels, not KL to an enumerated posterior.

Across independently fitted samplers, `manabot.belief.sampling_cohort` analyzes
saved reports declared in a `CohortSpec` manifest:

```bash
uv run python -m manabot.belief.sampling_cohort \
  --manifest /path/to/cohort/manifest.json \
  --out .runs/sampler-cohort.json
```

Metrics are game means averaged over fits. It gives separate 95% fit and game
bootstrap intervals, never a combined interval, conditional on one producer
policy and training dataset. Failed or missing fits stay in the output and
suppress cohort means. The retained `experiments/data/sampler-uncertainty-synthetic`
report is fabricated software evidence.

## Unsupported and unmeasured

The exact arm stops at opponent choices without a public likelihood (including
combat and target choices) and at discard prompts the materializer cannot
refresh, so full-game exact-versus-prior comparison is blocked. Compound
exact-history likelihood is unavailable. No Academy advice provider is
registered to the local-update teacher. Calibration, opponent transfer, search
strength and human-play results remain unmeasured; fixture tests establish
software behavior only.

History: [belief-sampler](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/belief-sampler.md),
[local-policy-search](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/local-policy-search.md).
