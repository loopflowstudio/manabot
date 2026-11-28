Read when the change touches the rules engine (`managym/src`, `managym/state`, `managym/cards`): rules, engine performance, Commands or the policy vocabulary.

- Engine changes that must not alter play (performance work in `managym/src`):
  build `main`'s release extension from a detached worktree with its own
  `CARGO_TARGET_DIR`, unzip the wheel's `.so` into that worktree's `managym/`,
  and play the same seeded random games on both engines by switching
  `PYTHONPATH` (print `managym.__file__` to confirm which loaded). Per game,
  hash `obs.toJSON()` at every decision and step `rng.randrange(len(
  obs.action_space.actions))`; equal digests mean equal offers, taps and
  events. Use the authored setups (`managym.authored_deck_keys` of
  `elemental-world-v2` and `ur-lessons-vs-gw-allies`; pack directory names are
  refused) plus the TLA pool corpus human builds
  (`experiments/data/mtg282/pools-v2/corpus.json`, fetched with `manabot.data
  fetch` when absent; `PlayerConfig(name, pool["human_build"])`), which bring waterbend, Treasure
  and any-color sources the curated benchmark decks lack; 2,400 games took
  about two minutes per engine. A few games end in `PyAgentError` on `main`
  too; the error must match, not be absent. `maturin build` without
  `--release` gives an extension with `debug_assert!` live, about 10x slower;
  unzip it to its own directory, never over `managym/`.
- Native Command QA: use an ordinary seeded game and its current semantic frame;
  submit one legal Command, then another command id with the old revision and
  an absent offer id with the current revision. Native rejection raises
  `managym.AgentError` (not only `ValueError`). Compare both `state_digest()`
  and `semantic_decision_frame_json()` before/after each rejection.
- Engine cost on large boards (`docs/development.md#stress-states`): build
  once with `cargo build --release --features benchmark --example bench_states
  --manifest-path managym/Cargo.toml`, then call
  `managym/target/release/examples/bench_states` directly (all 17 states ran
  in about 11 s on an M-series laptop). Forks carry the payment cache, so
  check that a layer's `work` (`calls`, `cache_hits`) still shows the search
  it claims to time rather than a hit warmed by preparation. Compare two commits by building the
  other one in a `/private/tmp` worktree with its own `CARGO_TARGET_DIR` and
  diffing each row's `work` field; the closing digest also moves whenever
  cards are added, so it is not a behavior check across a merge from main.
  `scripts/bench_states.py` uses the installed extension.
- Engine rules changes (zones, tokens, copies, SBAs): play many games through
  `managym.Env` with `PlayerConfig(name, {card: count}, world=managym.WORLD_SETUP)`
  decks built around the cards involved, and a scripted policy over
  `obs.action_space.actions`. Actions carry no `description`; match
  `action.focus` ids against `env.observation_for_player(env.current_agent_index())`
  card ids to pick a card. Read `observation_for_player(p).agent_cards`
  (`zone`, `is_token`) at every decision. A token in the graveyard during a
  mana ability's color choice is correct: SBAs wait for priority. Academy's
  game end is a "Game Over" dialog with "Rematch".
  Battlefield permanents are `agent_permanents`, with their own ids (the ones
  activation offers focus) and an integer `registry_key`; join them to a name
  through `agent_cards` rows that have the same `registry_key`.
- Policy vocabulary/admission changes (`record_vocabulary.rs`, token kinds,
  `manabot/policy/admission.py`): the consumer is training's catalog policy.
  Script a directed `managym.Env` game that reaches the runtime program,
  `encode_observation(space, obs)` each decision at `ObservationConfig(policy_input_version=10)`
  and pass it through `manabot.verify.record_coverage._embed` with a tiny
  catalog `ActorCritic`, as `coverage_sweep` does. Find decisions carrying a
  token by scanning encoded buffers for its u32 as adjacent 16-bit halves
  (hi, lo). For Koh's `HaveAbilitiesOf`: Swamps plus Koh against Forests plus
  Giant Spider, Koh cast once, always targeting with its ETB, the villain
  playing randomly, and one Pay-1-life activation per turn. The choice
  is a Modal decision only with two or more cards exiled with Koh (one is
  skipped as forced). 3 of 6 games reached it (8 decisions, about 7 minutes at load ~110).
  With a debug wheel (`uv run --extra play maturin build` without `--release`, about 6 min,
  unzipped to `.runs/<dir>/site`, `PYTHONPATH` pointing there) the
  `TokenWriter::into_rules_tokens` guard panics natively on any unregistered
  program token. From `.runs/<dir>`, run `uv run --project ../.. python -m
  manabot.verify.record_coverage --games 40 --seed 501` to stop at the first
  bad game (about 20 s in debug). From the checkout root, the current directory
  shadows `PYTHONPATH`, so the stale `managym/*.so` loads and the guard is
  silently absent. A sweep that catches `BaseException` per game and keeps
  going ran 200 world games (41k decisions, one decision per `prepare`) in
  about 28 min. MTG-501 found Text `"all"` (`CreatureTypes::All`, a departed
  changeling's LKI) in game 38 of seed 501; `RUST_BACKTRACE=1` names the
  emitter. To name a pair, blake3 every string literal under `managym/` and
  mask the first four little-endian bytes with `0x7fffffff`.
  For the negative, build a mutated release wheel with
  `-o /tmp/<dir>/wheels`, restore the source at once, unzip the wheel into
  `/tmp/<dir>/site` and run with `PYTHONPATH=/tmp/<dir>/site` (print
  `managym.__file__`); admission must then fail at the first such decision.
- "No behavior change" engine claims: `git worktree add --detach /tmp/<x>
  <before>`, build it with `uv run --extra play maturin build --release -i .venv/bin/python -m
  /tmp/<x>/managym/Cargo.toml` (about 4-5 min), drop its `.so` into
  `/tmp/<x>/managym/`, and run one seeded script against each tree
  (`sys.path.insert(0, root)` before `import managym`). Per decision, write a
  hash of both seats' `toJSON()` plus `env.state_digest()`; `cmp` the files.
  Use `random.Random(seed).randrange(env.action_count())` for actions.
  `managym.authored_deck_keys(pack)` / `authored_deck_setup(pack, key)` give
  the authored decks. Remove the worktree afterwards. Include the TLA pool
  human builds too: in MTG-320, 138 authored and Azula games matched while 33
  of 120 pool games exposed a regression (Kyoshi Battle Fan's token attach).
  When games diverge, replay one to the first differing decision on both
  engines and diff the parsed `toJSON()`; permanents' `attached_to`, `power`
  and the cast action's `focus` names locate the card.
  Catch `BaseException` per game: `PanicException` is not an `Exception`.
  Python action types print as `ActionEnum.PriorityPlayLand` (no int
  constructor); compare `str(a.action_type)`. When a change deliberately alters
  the default action space, play a canonical policy on both engines. That policy
  drops the intended offers, draws `rng` only when two or more choices remain and
  hashes `(chooser, space kind, [(type, focus, ability_index)], turn, step)`.
  The remaining divergence is what the change did not mean to touch (MTG-385:
  324/328 matched, and the rest were the same intended correction).
- Decision memory is opt-in on raw engines: `managym.Env(...)` and
  `VectorEnv(...)` record no viewer decision log unless built with
  `decision_memory=True` (the training wrappers derive it from
  `max_decision_records > 0`). A script, fixture or product path that builds a
  raw engine and reads `decision_memory_json()`, decision-record buffers or the
  LLM text history sees an empty log otherwise, with no error. Grep new raw
  constructions when memory consumers change.
- Buffered slot lifecycle: install the candidate engine wheel in an isolated
  venv and run the [batched slots example](../../docs/engine-python.md#batched-slot-loading-and-snapshots)
  outside the checkout. With its decks and seed 42, stepping only slot 0 with
  `Random(42).randrange(current_action_summaries()[0][0])` ended in 39
  decisions. Use `RecordedVectorEnv` with `games_path` and `statistics_path`
  to verify the winner and final digest are already readable before any load.
  Compare live `rows()` from `CardPerformance` and `Behaviour` with offline
  `reduce` on the completed record to check terminal exposure.
  In an isolated step, action 999 poisons one slot while a legal action
  advances its neighbor. With `journal_path` enabled, submit a load for that
  poisoned slot and one out-of-range slot. Reopen the cursor with
  `read_record` before retrying. Its phase should be `ready`, with both
  pre-call prefixes intact. Correcting the batch should spend the same fresh
  seed as a first successful load.
- Trigger-resolution fixes (what an ability does to its own source or a
  target): play seeded random games with decks built around the card, take
  `DeclineChoice` about half the time at target choices, and count zone
  transitions per step from both seats' `agent_cards` (for example, Eagles
  entering exile not from a graveyard). Run the identical script against the
  base wheel unzipped to `.runs/base-site` with `PYTHONPATH` as the control:
  in MTG-486, 100 games took about 2 s per engine, with 625 self-exiles on base
  and 0 after the fix.
- `tests/env/test_skip_forced_decisions.py::test_no_runner_turns_skipping_off`
  greps every `.rs`/`.py` line under `managym/src`, `manabot`, `academy`,
  `scripts` and `experiments` for `skip_trivial=false`/`False`, comments
  included. Describe lockstep or debug paths in prose ("with and without
  `skip_trivial`") rather than writing the assignment in a doc comment.
- Folded offers and the decision APIs without a browser: play seeded games
  between token-heavy decks (`22 Mountain / 30 Raging Goblin / 8 Burst
  Lightning`; `12 Mountain / 10 Island / 14 Dragon Fodder / 8 Geyser Leaper /
  8 Lightning Bolt`) through `Env.decision` / `Env.preview` / `Env.submit`,
  drafting attacks from the *last* declarable copies so plans name creatures
  the engine is not asking. An attacker with empty `can_attack` attacks the
  single default defender (`defender=None`); `PlannedAttack` needs
  `incarnation`. For the deprecated compound surface, use
  `env.compound_offers()` (`structured_offers()` only covers priority and
  raises elsewhere) with `RaggedPolicyDecoder` + `SeededSemanticScorer` and
  `step_structured`. 32 games take about a second. Repeated candidate labels
  are usually real distinctions the observation hides: haste masks differing
  summoning sickness (control since the turn began, CR 302.6), the creature
  activating a waterbend ability is named by the payment, and a spell on the
  stack names its target. To explain one, cast Bolt on an `env.clone_env()` at
  each step and watch when the copy stops sharing a target offer's aliases.
- Manual mana payment (`mana_payment=("manual", "manual")`): a fresh
  `uv venv --python 3.13 .runs/venv`, `uv pip install maturin numpy`, then a
  debug `maturin build -m managym/Cargo.toml -o .runs/wheels` with
  `CARGO_TARGET_DIR=.runs/target` (about a minute warm) installed into that
  venv; run scripts from `.runs/` with `-I` so the checkout's `managym/` (no
  `.so`) does not shadow the wheel. Build constructed positions as
  `scratch/repro_mana_payment.py` does and walk every payment order on
  `env.clone_env()` children until `Priority` with a stack object or game over.
  Space kinds print as `ActionSpaceEnum.Priority` / `PayOrNot` / `Modal`
  (CamelCase): an upper-case comparison never matches and the walk plays the
  whole game. With skipping on, a finished payment can run through forced
  passes into the opponent's turn, so read the payer's result through
  `env.observation_for_player(0)`. For collection, `VectorEnv` with
  `set_observation_layout({"rewards", "terminated", "truncated"} buffers,
  managym.ObservationLayout())`, `reset_all_into_buffers(setup)`, random
  actions from `current_action_summaries()`, `step_isolated_into_buffers`,
  slot errors from `get_last_info()[i]["game_error_class"]`, and
  `load_slots_into_buffers([SlotLoad.fresh(i)])` for finished or errored
  slots (plain `VectorEnv.step` does not reset and raises on a terminal slot).
  Run it in a child process whose parent `sample`s and kills it if no step
  finishes within a minute. The authored `ur-lessons-vs-gw-allies` decks with two
  basics swapped for Tarnished Citadel, Sunbaked Canyon, Fabled Passage and
  Airship Engine Room or Treetop Village, plus Heartbeat of Spring and The
  Great Henge, gave about 30 games a minute at 32 slots in debug.
  Snapshots from the 0.2.6 wheel are refused with `card_digest mismatch` on
  any later main (card definitions moved); a cross-version replay check needs
  a replay written by the base revision.
- Counterfactual state edits (`GameStateEdit`, `Env.edit_game_state`,
  `RestoreOptions(game_state_edits=)`): play an authored game
  (`authored_deck_setup("elemental-world-v2", "elemental_gr")` vs `_gw`, seed 7,
  80% non-pass random) to a turn-6+ `Priority` with two opposing creatures
  (38 steps), then run each edit on `env.clone_env()` and step 30 random
  actions after it; the original `state_digest()` must not move. Gray Ogre /
  Hill Giant mirrors (seed 11, always attack) reach `DeclareBlocker` by turn 7.
  At `DeclareAttacker`, batch `create_token("token.spirit", seat)` with a
  tapped creature from hand and an unready creature from the opposing library
  under your control. Decline each attack to see the Spirit offered in the
  remaining prompts. The tapped and unready creatures should stay out of those
  prompts. If an Aura is in hand, add it with `attached_to` set to a surviving
  host's `Permanent.id`. Restore the same batch with `RestoreOptions.determinize`
  in both a scalar fork and `SlotLoad.snapshot`; compare digests before and
  after the same action.
  For Academy, a snapshot replay that ends at the edit shows the edited board
  but no branch note (the note rides the next recorded choice): step 25
  actions past the edit before `to_replay_json()`, then the History panel
  shows **Branch · game state edited**. Only the branch kind reaches the
  browser, never edit ids or destinations.
