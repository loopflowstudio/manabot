# Card tests

Directed behavior tests for individual cards, in the `managym-cards` crate
beside the sets. This folder mirrors `managym/cards/src/`: `tla.rs` holds
tests about cards filed in `src/tla.rs`, `lea.rs` those in `src/lea.rs`, and
so on. The test binary is `tests/cards.rs`; run it from `managym/` with
`cargo test -p managym-cards --test cards` (or `... --test cards tla::` for
one set).

- File a test with the set of the card it is about, even when its scenario uses
  cards from other sets (basics, removal, filler creatures).
- Engine rules that are not about one card — injected abilities, fixture cards
  such as Crossroads of Destiny, token mechanics — stay in the engine's
  `managym/tests/rules/`.
- A set file is flat and in the card order of its set file's `cards()`. A
  setup used by one card's tests sits just before them; tests that exercise a
  mechanic across many cards follow at the end.
- Scenario helpers shared with the rules tests live in the engine's
  `managym/tests/support/`, mounted here by path: `game.rs` builds a game,
  `play.rs` drives it and `inspect.rs` reads it.

Tests and coverage never qualify or gate a card. A card is supported when its
set file defines it (see `managym/src/world.rs`); a missing test is a gap to
fill, not a support decision.

## Coverage

```sh
uv run python managym/cards/tests/cards/coverage.py            # every set
uv run python managym/cards/tests/cards/coverage.py --set tla --untested
```

The report lists each defined card with the tests in its set file that name it
(by string literal) and how many other tests mention it. Rebuild the extension
after card changes so the pool is current.

## Effect vocabulary checks

In debug builds (`cfg(debug_assertions)`), every runtime program finalized
through `TokenWriter::into_rules_tokens` on the default content pack checks
each non-numeric token against the `decision_memory=False` policy vocabulary
and panics listing every missing `(kind, 0x…)` pair in the stream. Any debug
Rust test run therefore enforces runtime program-token membership at the
emission point, including suspended resolution frames and
last-known-characteristics projections.

When adding or changing an `Effect` or runtime projection, run both the
exhaustive witness check and the debug Rust suite from the repository root,
using the development setup in the
[development guide](../../../../docs/development.md#environment-and-commands):

```sh
CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci --test-threads 2 -p managym-cards --test cards -E 'test(observability::every_effect_variant_reaches_the_learning_tokens)'
```

The witness check tests every effect witness in `observability.rs`, both as a
spell program and as a runtime rules program. It defers membership validation
until it has collected missing pairs across all variants. Other witnesses use
the emission guard. A missing vocabulary entry fails
with the effect variant, serialization path, token kind and decimal/hex token
value. For example, a runtime `Effect::HaveAbilitiesOf` needs its variant
token, closing structure `/Effect::HaveAbilitiesOf` and `definition` field
even when no printed card program contains that effect.

Keep new witnesses in the existing `effect_witnesses` inventory and its
exhaustive variant map. Allowed tokens come from one shared vocabulary: the
production catalog (including tokens and back faces) plus non-`DecisionMemory::*`
`record_vocabulary` registrations; the synthetic witness definition does not
supply allowed tokens, and `DecisionMemory::*` entries are excluded because
default (`decision_memory=False`) policies reject them. Only `Integer` and
`Boolean` values bypass exact vocabulary lookup. `DefinitionRef`, `Null` and
every other non-numeric kind require their exact `(kind, value)` pair.

For a failure, inspect the emitted stream or named effect and every missing
`(kind, 0x…)` key. If a raw storage ID struct (such as `PermanentId` or
`PublicId`) or non-visible provenance field leaked into a runtime program,
project it as a visible reference or omit the non-visible provenance rather
than registering raw ID structs. Register genuine runtime-only symbols in
`managym/src/agent/record_vocabulary.rs`; for an incidental fixture value, use
an existing valid catalog value. Do not register synthetic Witness content or
exempt another token kind to make the check pass. Rerun the check after
correcting the omission.

After rebuilding and atomically installing the matching native extension as
described in [AGENTS.md](../../../../AGENTS.md#commands), check the Python
admission boundary from the repository root:

```sh
uv run --extra dev pytest tests/model/test_effect_vocabulary.py tests/model/test_admission.py -q
```

These checks use the production host tables to verify that non-`DecisionMemory`
`record_vocabulary()` tokens and TLA printed spans are admitted on both the
dense and v11 paths, and reject each missing Koh runtime registration
independently. The shared-effect check admits registered modification tokens
and rejects unknown and disabled decision-memory pairs. A policy admission
failure lists the missing `(kind, 0x…)` keys in the same form as the Rust guard's
report. The bounded record-coverage sweep in the development guide provides
additional integration coverage.
