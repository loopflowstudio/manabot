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

On the default content pack, debug builds check every stream leaving
`TokenWriter::into_rules_tokens` against the default policy vocabulary.
A failure lists each missing `(kind, 0x…)` pair. This covers live effects,
suspended frames and last-known characteristics. The exact admission rules
are in the [engine guide](../../../../docs/engine.md#program-token-vocabulary).

Changes to an effect or runtime projection require the debug Rust suite and
the exhaustive effect witness. Run the witness from the repository root.

```sh
CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci --test-threads 2 -p managym-cards --test cards -E 'test(observability::every_effect_variant_reaches_the_learning_tokens)'
```

Add each new variant to `effect_witnesses` and its exhaustive variant map in
`observability.rs`. This test collects missing tokens across all variants
and reports the spell or runtime serialization path for each one. For example,
`Effect::HaveAbilitiesOf` needs its variant, its closing structure and its
`definition` field even when no printed card contains the effect.

Project storage IDs as viewer-visible references and omit private provenance.
Register genuine runtime-only symbols in `managym/src/agent/record_vocabulary.rs`.
Use catalog values for incidental witness data.

After [rebuilding the native extension](../../../../AGENTS.md#commands), run
the admission checks for dense facts, v11 facts and shared effects.

```sh
uv run --extra dev pytest tests/model/test_effect_vocabulary.py tests/model/test_admission.py -q
```
