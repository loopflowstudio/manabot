# MTG-277 native heuristic

Directive: port the Python heuristic to one native owner, used by scalar arena
players and `VectorEnv(opponent_policy="heuristic")`. Preserve viewer visibility,
chooser semantics, rules, deterministic ties, optional prefix capacity and rewards.
Shipping behavior is specified in docs/engine.md, docs/evaluation.md and
manabot/README.md; retain their edits.

## Plan and deletion cut

Compress cut complete: `heuristic/knowledge.rs` reads the same public definition
schema with borrowed values, without temporary legacy effect names or cloned JSON
bodies. Modal bodies stay opaque to ratings. `Board::best_card` shares discard/
selection ranking; `fight_value` removes the redundant attacker/defender branch
from symmetric fight scoring. Registration groups native source/version/compute
identity in one branch. These reversible choices preserve strategy and ties.

- Delete — do not maintain: Python heuristic rules, derived card knowledge,
  combat memory and synthetic namespace-only tests exclusive to those helpers.
  Preserve their rule coverage on native observations/debug tests and keep the
  frozen anchors.py bytes unchanged. Python HeuristicPlayer becomes an adapter.
- Native Heuristic reads only Observation; cached card knowledge derives from
  public definitions. Per-slot state resets with the game; scalar owns one
  selector per player. Retain first-cast/last-discard ties and combat ID ordering.
- Expose native selector through Python, wire batch advancement, version fresh
  arena registrations and include Rust sources in heuristic source identity.
- Capture bounded parity against the old source (temporary reference at
  a temporary file, original retained in git at 757bb6257),
  including Elemental. Build/reinstall extension atomically and run focused
  debug and installed-API tests. Gate owns full affected suites/Engine Budget.

MTG-272 / PR #458 was OPEN on 2026-11-11 during implementation. Leave its
checkout alone. Preserve Passive here until the prerequisite lands and sync
then; gate must check that integration before publication.

Reversible choices: use the existing public definition projection for card
knowledge (the same schema Python inspected); compile it once into typed native
ratings. Keep current approximate combat rules rather than changing strategy.
Fresh heuristic anchor uses native-v2 identity; historical registrations/results
are untouched and source validation continues to reject drift.

Inherited b015d5 study notes are unrelated to this Task and remain at
757bb6257:scratch/questions.md and its referenced published evidence.

Implemented: the native selector owns all heuristic rules and fallback; the
Python module is an adapter. Lazy per-Env memory is cloned on forks and cleared
on reset. The selector receives emitted observations, retaining combat events;
it never reads Game state. No rules/world/checkpoint/input layout changed.
Frozen anchors.py is unchanged. Fresh registration is native-v2.

Checks (2026-11-11):
- `cargo test --manifest-path managym/Cargo.toml -p managym --lib heuristic -j 2 -- --test-threads 2` — 11 passed in debug.
- Required release wheel build passed; cp312 extension replaced atomically.
- Original Python owner at 757bb6257 matched native on all 862 decisions from four bounded fixtures (256-decision cap; basic red/green and Elemental UR/GW, BG/UG; explicit prefix 2 included). One fixture finished; three reached the cap. Golden decisions and world provenance are in tests/fixtures/native_heuristic_parity.json. This is parity evidence, not a strength or throughput claim.
- `uv run --extra dev pytest tests/sim/test_heuristic.py -q` — original 7 installed-API tests passed (parity, scalar/batch reward/reset agreement, constructor rejection, registration).
- Added wrapper/closed-decklist/chooser-control checks plus `test_heuristic_scores_only_retained_legal_offers` — 5 passed after correcting the wrapper fixture to use its tuple reset and tensor step API. No behavior fix was needed.
- Focused Ruff lint/format and `git diff --check` passed. docs/engine.md's two-slot example executed successfully.

The first debug run caught an actual translation error: Python treats empty
predicate lists as absent when recognizing Lesson damage; native now does too.
Other initial failures were invalid test offers/names, corrected to the existing
Python fixtures (Pop Quiz, Fabled Passage, creature-only pump targets).

Compress verification (2026-11-11):
- Temporary debug comparison against the pre-compress Rust derivation matched every catalog card's complete ratings; the comparison source/test was then removed.
- `cargo test --manifest-path managym/Cargo.toml -p managym --lib heuristic -j 2 -- --test-threads 2` — 13 passed, including restored native combat-keyword and counterspell coverage.
- Required release wheel build and atomic cp312 extension replacement passed. `uv run --extra dev pytest tests/sim/test_heuristic.py tests/semantic/test_player_control.py tests/training/test_action_cutoff.py::test_heuristic_scores_only_retained_legal_offers -q` — 23 passed; all 862 retained decisions and terminal flags matched. Registration-capacity rejection coverage is retained on the adapter path.
- Rust formatting, focused Ruff lint/format and `git diff --check` passed. Scratch remains below both context limits.

Gate remains: complete affected suites, Rust clippy, Engine Budget on its normal
runner, and sync MTG-272/PR #458 after it lands (still OPEN at the final read).
The existing prospective docs specify none/random/heuristic; Passive remains
only pending that prerequisite's cut. Do not publish until the integration agrees.
No paid execution, server or watcher was started. Temporary reference scripts
were removed; their source is recoverable from the commit above.
