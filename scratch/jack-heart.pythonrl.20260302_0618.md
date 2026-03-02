# Stage 01: Fix PPO Correctness Bugs

Spec: `wave/first-light/01-fix-ppo-bugs.md`

## Context

Full architecture audit completed. Five confirmed bugs in the PPO
training loop, all deviating from CleanRL's known-correct reference.
This stage fixes them without changing the architecture.

## The fixes

1. **Advantage normalization** — `norm_adv=True` exists, never applied.
   Add 3 lines in `_optimize_step` after line 335.

2. **Batch size from actual data** — `inds = np.arange(batch_size)` uses
   config-derived size, but flattened multi-agent buffer may differ.
   Derive from `logprobs.shape[0]`.

3. **Policy head init gain** — `gain=1` → `gain=0.01` on final linear
   layer of policy_head. Near-uniform initial policy.

4. **Categorical(logits=)** — Replace manual softmax + `Categorical(probs=)`
   with `Categorical(logits=)`.

5. **Remove weight_decay** — `weight_decay=0.01` in Adam is non-standard
   for PPO. CleanRL uses 0. Remove it.

## Files to change

- `manabot/model/train.py` — bugs 1, 2, 5
- `manabot/model/agent.py` — bugs 3, 4
- `tests/test_train.py` — add tests for adv norm and batch size

## Notes from audit

The full architecture review is in git history (commit `372ab4a`). Key
findings that inform later stages:

- Self-play contradictory gradients → fixed in stage 02 (single-agent env)
- Sparse reward +100/-100 → fixed in stage 02 (+1/-1)
- Missing phase/step in obs → fixed in stage 03
- Raw IDs and unnormalized features → fixed in stage 03
- Performance issues (dict obs, pickle IPC, padding) → deferred past this wave
