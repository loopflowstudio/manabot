# First Light Experiment Harness

2026-03-06
Branch: `jack-heart.first-light.20260305_1906`
Status: design ready for implementation

## Goal

Replace the rigid verification ladder with a reusable research harness that:

- keeps a couple of cheap sanity checks
- records first-light experiment runs in SQLite
- generates lightweight markdown reports
- answers the move-on question for first-light

The move-on question is:

> Does the current shaped PPO recipe produce real, repeatable learning against a random opponent, especially on the causal chain metrics that matter?

Those metrics are primarily:

- play lands when available
- play creatures / cast spells when available
- reduce pass-collapse in non-trivial choice states
- improve win rate against random relative to untrained baseline

## Non-goals

Not in this implementation:

- multi-seed sweep orchestration
- formal CI integration
- DuckDB
- attention/LSTM/aux-head experiments themselves
- self-play or scale work
- a polished long-term experiment tracking product

This is a research harness first. Fast iteration matters more than permanence.

## High-level shape

## 1. Keep two lightweight sanity scripts

Retain the existing small checks, with minimal semantic change:

- `manabot.verify.step0_env_sanity`
- `manabot.verify.step1_trivial_reward`

These are the closest thing to regression checks.

### Intended role

- `step0_env_sanity`: environment health, truncation/abort sanity, basic matchup sanity
- `step1_trivial_reward`: PPO/value-path sanity under trivial reward

These can stay mostly as they are for now.

## 2. Replace old steps 2-4 with one main harness

Introduce:

- `python -m manabot.verify.run_first_light`

This is the main experiment driver.

It replaces the conceptual role of:

- `step2_memorization`
- `step3_beat_passive`
- `step4_beat_random`

Those scripts are no longer the center of gravity. They can either become thin wrappers or be retired after migration.

## 3. Add report command

Introduce:

- `python -m manabot.verify.report_first_light`

This reads one recorded run from SQLite and emits a lightweight markdown report with:

- recommendation first
- key metrics / deltas second
- experiment config and notes after that

## Why this structure

The current ladder is too step-shaped for the actual problem.

What first-light has revealed is more specific:

- the main failure mode is pass-collapse
- shaped rewards are currently required
- stochastic eval and behavioral metrics are necessary to interpret learning
- passive-opponent success is useful, but the real move-on gate is progress vs random

So the harness should optimize for:

- recording runs
- comparing to untrained baseline
- inspecting causal-chain metrics
- producing a clear recommendation

## Baseline recipe

The harness should treat shaped reward as the current first-light baseline recipe.

Reason:

- terminal-only PPO did not work
- reward shaping is not a preference here; it is what the current system requires

### Baseline shaping values

Initial canonical recipe:

- `win_reward = +1.0`
- `lose_reward = -1.0`
- `land_play_reward = 0.03`
- `creature_play_reward = 0.06`
- `opponent_life_loss_reward = 0.01`

These may become configurable, but this should be the default first-light recipe for now.

## Storage choice

Use one shared SQLite database:

- `.runs/verify.sqlite`

Why:

- simplest mental model
- single-run-first workflow
- easy append/query story
- no file explosion

Use WAL mode so reads and one writer can coexist comfortably.

## Data model

Use an explicit, normalized schema. Do not try to be clever.

## Core tables

### `runs`

One row per first-light experiment run.

Columns:

- `id` INTEGER PRIMARY KEY
- `created_at` TEXT NOT NULL
- `label` TEXT
- `status` TEXT NOT NULL            -- planned/running/completed/failed
- `mode` TEXT NOT NULL              -- dev/decision
- `seed` INTEGER NOT NULL
- `git_commit` TEXT
- `git_branch` TEXT
- `tree_dirty` INTEGER NOT NULL     -- 0/1
- `recipe_name` TEXT NOT NULL       -- e.g. first_light_shaped_v1
- `opponent_policy` TEXT NOT NULL   -- usually random
- `num_envs` INTEGER NOT NULL
- `num_steps` INTEGER NOT NULL
- `total_timesteps` INTEGER NOT NULL
- `eval_interval` INTEGER
- `eval_num_games` INTEGER NOT NULL
- `baseline_auto` INTEGER NOT NULL  -- 0/1
- `notes` TEXT

### `run_configs`

One row per run storing the canonical config blobs used.

Columns:

- `run_id` INTEGER PRIMARY KEY REFERENCES runs(id)
- `experiment_json` TEXT NOT NULL
- `train_json` TEXT NOT NULL
- `reward_json` TEXT NOT NULL
- `agent_json` TEXT NOT NULL
- `match_json` TEXT NOT NULL
- `observation_json` TEXT NOT NULL

This keeps provenance simple without over-normalizing every hyperparameter.

### `evaluations`

One row per evaluation event.

This includes:

- untrained baseline evals
- checkpoint evals
- final evals
- optional passive-opponent side evals

Columns:

- `id` INTEGER PRIMARY KEY
- `run_id` INTEGER NOT NULL REFERENCES runs(id)
- `created_at` TEXT NOT NULL
- `kind` TEXT NOT NULL              -- baseline/checkpoint/final/side
- `step` INTEGER NOT NULL
- `update_index` INTEGER
- `opponent_policy` TEXT NOT NULL
- `num_games` INTEGER NOT NULL
- `deterministic` INTEGER NOT NULL
-
- `win_rate` REAL NOT NULL
- `win_ci_lower` REAL NOT NULL
- `mean_steps` REAL NOT NULL
- `attack_rate` REAL NOT NULL
- `attacked_when_able` REAL NOT NULL
- `landed_when_able` REAL NOT NULL
- `cast_when_able` REAL NOT NULL
- `passed_when_able` REAL NOT NULL
-
- `could_attack` REAL NOT NULL
- `could_land` REAL NOT NULL
- `could_spell` REAL NOT NULL
- `could_pass` REAL NOT NULL
-
- `land_plays` REAL NOT NULL
- `spell_casts` REAL NOT NULL
- `pass_count` REAL NOT NULL
-
- `single_valid_decisions` REAL NOT NULL
- `multi_valid_decisions` REAL NOT NULL
-
- `pass_land_decisions` REAL NOT NULL
- `pass_land_pass_rate` REAL NOT NULL
- `pass_land_land_rate` REAL NOT NULL
- `pass_land_spell_rate` REAL NOT NULL
-
- `mean_pass_prob` REAL
- `mean_land_prob` REAL
- `mean_spell_prob` REAL
- `mean_attack_prob` REAL
- `mean_pass_prob_when_land_available` REAL
- `mean_land_prob_when_land_available` REAL
- `mean_pass_prob_when_pass_land` REAL
- `mean_land_prob_when_pass_land` REAL
-
- `action_space_truncations` REAL NOT NULL
- `card_space_truncations` REAL NOT NULL
- `permanent_space_truncations` REAL NOT NULL
-
- `explained_variance` REAL

This mirrors the current diagnostic surface and makes report generation simple.

### `evaluation_choice_sets`

Aggregated counts of choice-set shapes for one evaluation.

Columns:

- `evaluation_id` INTEGER NOT NULL REFERENCES evaluations(id)
- `scope` TEXT NOT NULL             -- all/priority
- `choice_set` TEXT NOT NULL        -- e.g. pass+land, land+spell+pass
- `count` REAL NOT NULL
- PRIMARY KEY (`evaluation_id`, `scope`, `choice_set`)

### `evaluation_actions`

Detailed per-action rows, but only for eval/sampled runs.

Columns:

- `id` INTEGER PRIMARY KEY
- `evaluation_id` INTEGER NOT NULL REFERENCES evaluations(id)
- `game_index` INTEGER NOT NULL
- `step` INTEGER NOT NULL
- `player` INTEGER NOT NULL
- `action_type` TEXT NOT NULL
- `choice_set` TEXT NOT NULL
- `is_trivial` INTEGER NOT NULL
- `num_valid_actions` INTEGER NOT NULL
- `attack_available` INTEGER NOT NULL
- `land_available` INTEGER NOT NULL
- `spell_available` INTEGER NOT NULL
- `pass_available` INTEGER NOT NULL
- `pass_prob` REAL
- `land_prob` REAL
- `spell_prob` REAL
- `attack_prob` REAL

### `reports`

Generated report metadata.

Columns:

- `id` INTEGER PRIMARY KEY
- `run_id` INTEGER NOT NULL REFERENCES runs(id)
- `created_at` TEXT NOT NULL
- `report_kind` TEXT NOT NULL       -- summary/decision
- `markdown_path` TEXT
- `recommendation` TEXT NOT NULL
- `summary_json` TEXT

## Detail retention policy

Store detailed per-action rows only for:

- final evals
- baseline evals
- explicit sampled/debug evals

Do not log per-action rows for every periodic checkpoint eval by default.

Reason:

- preserves the useful diagnostic surface
- avoids unbounded data growth
- keeps dev runs lightweight

## Command surface

## A. One-shot path

Primary ergonomic entry point:

```bash
python -m manabot.verify.run_first_light \
  --mode dev \
  --seed 42 \
  --label shaped-v1
```

This command should:

1. create a run row
2. record full config
3. run an untrained baseline eval against random
4. train the shaped recipe
5. run periodic checkpoint evals
6. run final eval(s)
7. write summary to stdout
8. optionally trigger report generation

## B. Decomposed path

Support subcommands inside the same module, or separate internal helpers.

Minimum requirement:

```bash
python -m manabot.verify.run_first_light train ...
python -m manabot.verify.run_first_light eval ...
python -m manabot.verify.report_first_light --run-id 17
```

The one-shot path can call the same internal functions.

## Recommended CLI

### `run_first_light`

Common flags:

- `--db .runs/verify.sqlite`
- `--seed`
- `--label`
- `--mode dev|decision`
- `--opponent random|passive`
- `--num-envs`
- `--num-steps`
- `--total-timesteps`
- `--eval-interval`
- `--eval-num-games`
- `--baseline/--no-baseline`
- `--report`

Mode presets:

- `dev`: short run, fewer games, enough to see direction quickly
- `decision`: longer run, credible enough for move-on recommendation

### `report_first_light`

```bash
python -m manabot.verify.report_first_light --run-id 17
```

Flags:

- `--db .runs/verify.sqlite`
- `--run-id`
- `--output path.md`

## Report shape

Reports should be lightweight markdown.

Top sections, in order:

1. **Recommendation**
2. **Key metrics and deltas vs untrained baseline**
3. **Interpretation of causal-chain metrics**
4. **Experiment config**
5. **Optional detailed notes / tables**

### Recommendation language

Use heuristic recommendations, not fake certainty.

Examples:

- `Move on to auxiliary-head experiments`
- `Stay in first-light: random-opponent win signal is too weak`
- `Stay in first-light: pass-collapse improved, but creature-play signal remains weak`

## Recommendation heuristic

The recommendation should be based primarily on random-opponent outcomes.

### Inputs that matter most

- win rate vs random, and delta vs untrained baseline
- `landed_when_able`
- `cast_when_able`
- `pass_land_pass_rate`
- `mean_land_prob_when_pass_land` vs `mean_pass_prob_when_pass_land`

### Inputs that matter less

- passive-opponent win rate (useful sanity, not the move-on gate)
- cross-seed stability (useful debugging, not required to express first recommendation)

### Initial heuristic rule

Recommend “move on” only if the final random-opponent eval shows all of:

1. win rate meaningfully improved over untrained baseline
2. land-vs-pass collapse corrected in the right direction
3. creature/spell-play metrics no longer suggest the chain is broken

Exact thresholds can be refined during implementation.

## Reuse vs replace

## Keep / reuse

### Keep as-is or nearly so

- `manabot.verify.step0_env_sanity`
- `manabot.verify.step1_trivial_reward`
- reward shaping in `manabot/env/match.py`
- reward shaping in `manabot/env/vector_env.py`
- evaluation metric logic in `manabot.verify.util.run_evaluation()`
- helper functions in `manabot.verify.util` for action-type / choice-set analysis

### Reuse but refactor

- current `print_result()` style reporting
  - replace with report/store-oriented helpers
- current periodic eval support in `Trainer`
  - keep the capability, but route persistence through SQLite
- `step3_beat_passive` checkpoint-loop pattern
  - useful reference for decision-mode checkpoint evaluations

## Replace

### Replace outright

- DuckDB-backed metrics logging (`manabot.infra.metrics.MetricsDB`)
- direct DuckDB usage in step 4 / training eval plumbing
- old step2/3/4 as the primary first-light interface

### Replace with

- SQLite-backed `VerifyStore` or similarly named module under `manabot.verify`
- first-light run recorder and report generator

## Migration plan

### Phase 1 — storage and reporting foundation

1. Add SQLite store module
2. Add schema creation / migration bootstrap
3. Replace DuckDB dependency in code paths used by verification
4. Remove `duckdb` dependency from `pyproject.toml` and lockfile

### Phase 2 — first-light harness

1. Implement `run_first_light`
2. Add dev/decision presets
3. Add automatic untrained baseline
4. Write checkpoint and final evals into SQLite

### Phase 3 — reporting

1. Implement `report_first_light`
2. Generate lightweight markdown
3. Add heuristic recommendation section

### Phase 4 — cleanup

1. Demote or remove old `step2_memorization`, `step3_beat_passive`, `step4_beat_random`
2. Rewrite `wave/first-light/04-verification-ladder.md` to match the new harness
3. Remove stale references to DuckDB metrics logging

## Minimal first implementation

If implementation needs to be staged, the smallest useful slice is:

1. SQLite store
2. one-shot `run_first_light`
3. automatic untrained baseline + final eval
4. markdown report from one run

Periodic checkpoint evals and detailed per-action storage can come right after that, but they are not required to make the harness real.

## Notes for the implementing agent

- Prioritize a clear, boring schema over elegance.
- Keep the one-shot path easy to use.
- Preserve the current behavioral metrics; they are the whole point.
- Do not rebuild a general experiment platform here.
- Passive-opponent success is useful, but random-opponent learning is the real decision surface.
- This harness is meant to support the next wave decisions, not become a product.

