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

When adding or changing an `Effect`, run the exhaustive learning-token guard
from the repository root, using the development setup in the
[development guide](../../../../docs/development.md#environment-and-commands):

```sh
CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci --test-threads 2 -p managym-cards --test cards -E 'test(observability::every_effect_variant_reaches_the_learning_tokens)'
```

The guard checks every effect witness in `observability.rs`, both as a spell
program and as a runtime rules program. A missing vocabulary entry fails with
the effect variant, serialization path, token kind and decimal/hex token value.
For example, a runtime `Effect::HaveAbilitiesOf` needs its variant token, closing
structure `/Effect::HaveAbilitiesOf` and `definition` field even when no printed
card program contains that effect.

Keep new witnesses in the existing `effect_witnesses` inventory and its
exhaustive variant map. Allowed tokens come from the production catalog
(including tokens and back faces) plus native `record_vocabulary` registrations;
the synthetic witness definition does not supply allowed tokens. Only Integer
and Boolean values bypass exact vocabulary lookup. DefinitionRef, Null and
every other nonnumeric kind require their exact kind/value pair.

For a failure, inspect the named effect and missing key. Register genuine
runtime-only symbols in `managym/src/agent/record_vocabulary.rs`; for an
incidental fixture value, use an existing valid catalog value. Do not register
synthetic Witness content or exempt another token kind to make the check pass.
Rerun the guard after correcting the omission.

After rebuilding and atomically installing the matching native extension as
described in [AGENTS.md](../../../../AGENTS.md#commands), check the Python
admission boundary from the repository root:

```sh
uv run --extra dev pytest tests/model/test_effect_vocabulary.py tests/model/test_admission.py -q
```

These checks use the production encoder's tables, verify exact lookup semantics
and reject each missing Koh runtime registration independently. Unknown tokens
remain errors; a policy admission failure lists the missing `(kind, 0x…)` keys,
in the same kind and hex form as the guard's report. The bounded record-coverage sweep in the development guide is
additional integration coverage; it does not establish that every effect ran.
