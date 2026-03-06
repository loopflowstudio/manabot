# Gate: jack-heart.first-light.20260305_1906

## Verdict: PASS

## Test Results

- **Branch-relevant tests**: 23/23 passed (test_match.py, test_util.py)
- **Full suite**: 151 passed, 3 failed, 6 errors

### Pre-existing failures (not introduced by this branch)

All failures stem from managym's card_dim changing from 18→29 after keyword abilities were added to the Rust engine. Confirmed by stashing branch changes and re-running — same failures on main code with updated managym bindings.

- `test_observation_parity` — hardcoded card_dim=18 in test fixture
- `test_encode_observation_into` — same hardcoded dimension
- `test_train.py` errors — `Card` object missing `keywords` attr in older managym stubs

## Rust checks

No managym changes on this branch. Skipped.

## Code Review

### Correctness verified

- **Card feature indices** (`rust_vector_env.py:50-51`): `_card_land_index = num_zones + 4`, `_card_creature_index = num_zones + 5`. Matches `observation.py:256-258` encoding order exactly.
- **Validity check** (`rust_vector_env.py:233`): `cards[..., -1] > 0.5` matches `observation.py:290` where `arr[-1] = 1.0` is set for valid cards.
- **Progress shaping** (`match.py:101-148`): Correctly counts battlefield lands/creatures via zone and card_type checks. Clamps deltas to non-negative.
- **Vectorized shaping** (`rust_vector_env.py:179-226`): Mirrors scalar `match.py` logic in tensor form. Masks out terminal steps. `_prev_obs` is set at top of `step()` before `_apply_reward_policy()` — no uninitialized state risk.
- **Value head change** (`agent.py:79-86`): MaxPooling→MeanPooling with added ReLU before pooling. Clean change.
- **Eval metrics** (`verify/util.py`): Comprehensive action-type tracking with probability logging. `_policy_action_type_probabilities` runs a forward pass per decision — acceptable for eval (not training).

### No issues found

No security concerns, no correctness bugs, no regressions from this branch.
