# Manabot Python RL Architecture Review (Additional Perspective)

Date: 2026-03-02
Scope: `manabot/` Python training stack in this repo, plus ecosystem comparison.

---

## 1) Executive summary

Your original diagnosis was directionally right. The strongest additional conclusions from this pass are:

1. **There are still a few P0 correctness bugs in reward/termination plumbing** that can fully mask learning progress even if architecture/hyperparams were perfect.
2. **The architecture is not fundamentally doomed**: typed object encoders + action focus grounding are good ideas.
3. **The main blocker is not “attention is too fancy”**; it is a combination of training-loop correctness gaps + low signal/noisy observations + hot-loop overhead.
4. **Best redesign path is baseline-first**: make a CleanRL-style simple PPO loop verifiably learn a tiny task, then reintroduce MTG-specific structure.

---

## 2) What is working well in current setup

### 2.1 Core RL scaffolding exists and is coherent

- PPO rollout/update structure is present and readable (`manabot/model/train.py`).
- GAE recursion itself is implemented in a standard reverse-time form (`PPOBuffer.compute_advantages`).
- Clipped policy loss, optional clipped value loss, entropy bonus, grad clipping are all present.

### 2.2 Action masking foundation is correct in spirit

- Invalid actions are masked to large negative logits before sampling (`agent.py`, masked_fill with `-1e8`).
- There are tests for mask behavior (`tests/model/test_train.py`, `tests/model/test_agent.py`).

### 2.3 Architecture has useful game-structured ideas

- Separate encoders for players/cards/permanents are a sound inductive bias.
- Focus-object mapping from action metadata to object embeddings is exactly the right style for targeted game actions.
- Perspective vector (+/- by ownership) is clever and compact.

### 2.4 Good engineering instincts are visible

- Attention mask leakage check exists.
- Profiler exists.
- Gradient logging exists.
- Simulation/evaluation path exists.

So this is not a “start over from zero” situation.

---

## 3) What needs to change (by severity)

## P0 — correctness issues to fix before trusting any learning curve

### P0.1 Reward bug in Python reward wrapper (critical)

In `Reward.compute`:

- `new_obs.won` is a boolean (won from current observing player perspective).
- It is compared against `last_obs.agent.player_index` (0/1 int).

```py
if new_obs.won == last_obs.agent.player_index:
```

This is semantically wrong and can invert terminal reward depending on player index.

Evidence:
- Python typing says `won: bool` (`managym/__init__.pyi`).
- C++ observation sets `won` as bool (`managym/agent/observation.cpp`).

### P0.2 Truncation is dropped in trainer

`_rollout_step` receives `(obs, reward, terminated, truncated, info)` from vector env, but stores only `done=terminated` and ignores truncation.

Implication: bootstrapping/termination logic is wrong whenever truncation happens.

### P0.3 `norm_adv` configured but never applied

`TrainHypers.norm_adv=True` exists, but minibatch advantages are never normalized in `_optimize_step`.

This is a known PPO stability-critical detail (see CleanRL docs + PPO implementation detail references).

### P0.4 Action-space hard truncation risk

Observation encoder truncates legal actions to `max_actions` (default 10):

```py
for action in obs.action_space.actions[: self.max_actions]
```

If real legal action count exceeds 10, policy cannot represent full legal action space.

### P0.5 Terminal credit assignment asymmetry with turn-based actor storage

Transitions are only stored for acting player each step (multi-buffer keyed by `(env, player)`). In many terminal transitions, only one side gets direct terminal reward signal at final action step.

This may not be a formal bug, but is a high-risk credit-assignment distortion in self-play.

---

## P1 — learning quality blockers (high impact)

### P1.1 Missing phase/step features despite engine exposing them

`managym` provides turn phase/step in observation. Python encoder defines enums but does not encode phase/step features into model input.

For MTG, legal/optimal action heavily depends on phase/step.

### P1.2 Raw IDs as numeric features

`player.id`, `card.owner_id`, `perm.controller_id` are inserted as floats into network inputs.

IDs are relational/categorical and often non-transferable across games; this injects noisy pseudo-ordinal signal.

### P1.3 Feature scale mismatch

Continuous features with very different magnitudes (life, zone counts, mana values, booleans) are mixed without explicit normalization.

### P1.4 Extremely sparse terminal-heavy reward for complex horizon

Current default ±100 terminal reward (plus mostly 0) can work eventually, but is brittle and slow for early policy shaping in MTG-sized horizons.

### P1.5 Shared-policy self-play is currently under-scaffolded

Shared policy for both sides can work in some settings, but here combined with sparse reward + actor-as-perspective storage + missing phase features makes optimization unnecessarily hard.

---

## P2 — throughput/perf and operability risks

### P2.1 AsyncVectorEnv mode likely adds IPC overhead

You use `AsyncVectorEnv(..., shared_memory=False)` with dict observations.

Gymnasium docs explicitly note async envs use multiprocessing + pipes and that shared memory affects observation transfer efficiency.

### P2.2 Python hot-loop overhead is high

- Dict-of-arrays observation structure throughout rollout and batching.
- Per-env Python loops in buffer routing.
- Repeated tensor allocations/conversions.

### P2.3 Very heavy logging in hot path

`agent.forward` and related methods emit many debug shape/stat logs. Even if debug level is off, formatting overhead and accidental verbosity can still hurt throughput and signal-to-noise during iteration.

### P2.4 Padding footprint is large

Defaults: 100 cards/player + 50 permanents/player produce large mostly-empty attention inputs.

---

## 4) Standard vs creative in your architecture

### Standard parts

- PPO with clipped objective, GAE, minibatch epochs.
- Invalid-action masking.
- Vectorized env stepping.

### Creative/useful parts

- Object-type-specific encoders.
- Focus-object action grounding.
- Perspective vector to represent ownership.

### Recommendation

Keep the creative parts, but only after a minimal baseline loop reliably learns simple tasks.

---

## 5) External inspiration: what to actually borrow

This section is intentionally concrete: **source -> pattern -> exact adaptation for manabot**.

### 5.1 PPO / implementation fidelity

#### Source: PPO paper
- https://arxiv.org/abs/1707.06347

Takeaway:
- PPO is intended for multiple epochs of minibatch updates on collected on-policy data.

Adaptation:
- Keep PPO; do not switch algorithms yet.
- Fix implementation fidelity gaps first (adv norm, termination handling).

#### Source: CleanRL PPO docs + linked details
- https://docs.cleanrl.dev/rl-algorithms/ppo/
- https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Takeaway:
- 13 “core” implementation details are repeatedly highlighted (adv normalization, orthogonal init choices, vectorized arch, etc).

Adaptation:
- Build a “PPO conformance checklist” against your code and gate merges on it.
- Use this as your regression harness when refactoring.

---

### 5.2 High-throughput single-node architecture

#### Source: PufferLib docs
- https://puffer.ai/docs.html

Takeaway:
- Emulation layer flattens structured spaces for buffers/vec env and unflattens at model boundary.
- Shared-memory vectorization and zero-copy emphasis.
- Native handling for multiagent via padding/fixed-agent counts.

Adaptation:
- Flatten observation at env boundary (single tensor + sidecar metadata).
- Keep model-side unflatten logic only where needed.
- Replace dict/list trajectory storage with preallocated contiguous tensors.

#### Source: Sample Factory architecture docs
- https://www.samplefactory.dev/06-architecture/overview/

Takeaway:
- Decouple rollout, inference, batching, learning components.
- Shared-memory buffers with signaling between processes.
- Double-buffered sampling and clear component boundaries.

Adaptation:
- Even on one machine, separate “sampler” and “learner” responsibilities in code organization.
- Add explicit policy-versioning / policy-lag metrics.

---

### 5.3 Multi-agent policy-training patterns

#### Source: MAPPO repo + paper
- https://github.com/marlbenchmark/on-policy
- https://arxiv.org/abs/2103.01955

Takeaway:
- PPO can be very strong in MARL when implementation/hyperparameters are right.
- MAPPO codebase explicitly discusses shared-policy assumptions and tuned rollout/epoch settings.

Adaptation:
- Keep PPO baseline for MARL.
- Be explicit about parameter-sharing mode as a first-class config.
- Add “which policies are trainable now” concept.

#### Source: RLlib multi-agent docs
- https://docs.ray.io/en/master/rllib/multi-agent-envs.html

Takeaway:
- Explicit `policy_mapping_fn` and `policies_to_train` are powerful abstractions.

Adaptation:
- Mirror this in manabot: allow hero policy trainable while villain frozen/random.
- Avoid implicit “all actor transitions always update same policy” behavior.

#### Source: PettingZoo docs
- https://pettingzoo.farama.org/

Takeaway:
- AEC API cleanly models turn-based MARL; Parallel API handles simultaneous actions.

Adaptation:
- Treat MTG as explicitly turn-based multi-agent process with clean actor indexing semantics and tested transition ownership.

---

### 5.4 Imperfect-information game learning + evaluation

#### Source: OpenSpiel
- https://github.com/google-deepmind/open_spiel
- https://arxiv.org/abs/1908.09453

Takeaway:
- Strong benchmark suite and algorithm zoo for imperfect-information settings.
- Emphasis on evaluation dynamics, not just training reward.

Adaptation:
- Import OpenSpiel-style evaluation discipline:
  - exploitability proxies where feasible,
  - cross-play matrices,
  - fixed-opponent and population evaluation.

#### Source: NFSP paper
- https://arxiv.org/abs/1603.01121

Takeaway:
- In imperfect-information self-play, pure RL can diverge/cycle; NFSP-style average-policy mechanisms can stabilize toward equilibrium.

Adaptation:
- Not a v0 requirement, but keep NFSP/policy-averaging as medium-term option if PPO self-play cycles.

---

### 5.5 Self-play scale lessons

#### Source: OpenAI Five
- https://arxiv.org/abs/1912.06680

Takeaway:
- Existing methods + massive scale + robust distributed system can solve very hard games.

Adaptation:
- For manabot, immediate lesson is **system reliability and continuous training infrastructure matter** as much as algorithm novelty.

#### Source: IMPALA / SEED RL
- https://arxiv.org/abs/1802.01561
- https://arxiv.org/abs/1910.06591

Takeaway:
- Decoupled actor/learner architectures and centralized/efficient inference dramatically improve throughput and wall-clock learning.

Adaptation:
- Medium-term: separate acting and learning loops, log policy lag, and use shared-memory transport.

---

### 5.6 KataGo-style signal shaping inspiration

#### Source: KataGo paper + methods doc
- https://arxiv.org/abs/1902.10565
- https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md

Takeaway:
- Big gains came from richer targets/training mechanics and efficiency improvements.
- Methods doc highlights additional techniques discovered post-paper.

Adaptation (inference from source to MTG context):
- Add auxiliary targets (e.g., life-delta-to-end, turns-to-terminal, board-advantage proxy).
- Consider value target beyond pure win/loss when possible.

**Inference note:** KataGo uses search-supervised targets in Go; MTG hidden info makes direct transfer nontrivial. Use principle (denser supervision), not exact algorithm.

---

### 5.7 About `metta-public`

#### Source
- https://github.com/Metta-AI/metta-public

Observation:
- As of **Jan 22, 2026**, repo is archived/read-only.
- Positioning is broad multi-agent cooperation/alignment platform.

Adaptation:
- Treat as potential source of experiment ideas, not as your primary reference for PPO training-system architecture.

---

## 6) Prioritized change plan (current stack)

## Phase A: Make training loop trustworthy (no architecture changes yet)

1. Fix reward bug (`won` bool handling).
2. Correct termination/truncation propagation end-to-end.
3. Implement minibatch advantage normalization when `norm_adv=True`.
4. Add assertion/metrics when legal actions exceed `max_actions`.
5. Add deterministic tiny-task regression test (single env, fixed seed, tiny deck).

Success gate:
- Value/advantage stats sane, no NaNs, reproducible curves across seeds.

## Phase B: Improve signal and observability

6. Encode phase + step in observation features.
7. Remove raw IDs; replace with relational booleans (`is_mine`, etc.).
8. Normalize numeric features.
9. Add hero-only training mode vs frozen/random villain mode.
10. Add explicit win-rate-vs-random eval every N updates.

Success gate:
- Win rate vs random > 60% in simplified deck/task and trending upward.

## Phase C: Throughput refactor

11. Flatten rollout buffer storage (preallocated tensors).
12. Reduce logging overhead in hot path.
13. Re-evaluate Sync vs Async vector env using measured SPS.
14. Reduce padding maxima to realistic values.
15. Move more encoding onto engine side (Rust migration path).

Success gate:
- 2-5x SPS improvement without regression in learning quality.

---

## 7) Final recommendation on redesign philosophy

- **Do not throw away the object/action architecture yet.**
- First, get a tiny baseline to learn obvious behavior with robust instrumentation.
- Then layer back complexity one piece at a time with A/B gates.

Concretely:
- Baseline policy can be simple MLP over minimal features.
- Attention and focus heads are reintroduced only after baseline learning is stable and faster.

---

## 8) Sources

- PPO paper: https://arxiv.org/abs/1707.06347
- CleanRL PPO docs: https://docs.cleanrl.dev/rl-algorithms/ppo/
- PPO implementation details (ICLR blog): https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- PufferLib docs: https://puffer.ai/docs.html
- Sample Factory architecture: https://www.samplefactory.dev/06-architecture/overview/
- MAPPO repo: https://github.com/marlbenchmark/on-policy
- MAPPO paper: https://arxiv.org/abs/2103.01955
- RLlib multi-agent docs: https://docs.ray.io/en/master/rllib/multi-agent-envs.html
- PettingZoo docs: https://pettingzoo.farama.org/
- OpenSpiel repo: https://github.com/google-deepmind/open_spiel
- OpenSpiel paper: https://arxiv.org/abs/1908.09453
- KataGo paper: https://arxiv.org/abs/1902.10565
- KataGo methods: https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md
- OpenAI Five paper: https://arxiv.org/abs/1912.06680
- IMPALA: https://arxiv.org/abs/1802.01561
- SEED RL: https://arxiv.org/abs/1910.06591
- NFSP: https://arxiv.org/abs/1603.01121
- Gymnasium AsyncVectorEnv docs: https://gymnasium.farama.org/api/vector/async_vector_env/
- Metta public repo: https://github.com/Metta-AI/metta-public

