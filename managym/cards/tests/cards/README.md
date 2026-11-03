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
