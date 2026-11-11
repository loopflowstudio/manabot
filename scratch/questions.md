# MTG-282 choices

Inherited MTG-271 notes are unrelated background, recoverable at
`1659b0e290f55daa632ec8f869adff4a3059d79f:scratch/questions.md`.

The documentation step chose the reversible public API
`managym.LimitedRules(pool=..., max_cards=60).choices(picks=...)`, returning
`cards` and `can_stop`. Keep the cap separate from Limited's 40-card minimum.

Headless implementation assumption (2026-11-11): freeze 100 of the 1,000
source events for evaluation, selected by ascending SHA256 of
`mtg-282-held-out-v1:<event_id>`, ties by event ID. Save explicit membership,
selection rule and source provenance with the derived corpus before any
training. The Task did not specify a fraction or seed. Historical pilot
corpora and regimes remain untouched.

## Native format placement

Jack Heart, comment c17bdb50 (2026-11-11): "This probably needs a /format/
instead of cards?" Applied the full steer: `format/mod.rs` and
`format/limited.rs`, declared from `lib.rs`; bindings use `format::limited`.
No re-export from `cards`, which keeps definitions, packs, decklists and
world admission.

## Corpus admission counterexample — 2026-11-11

Re-admitting the retained source corpus (SHA256
`22f9b2b9411a69a85ff1270605dd68fba5941601859b752d2a4a0ecf5e4a6043`)
against this checkout's rebuilt w17 rejects 78/1,000 events, with 41 distinct
unsupported names. Examples: Scout's Warning (7 occurrences), Volcanic Torrent
(5), Blasphemous Act (5), Agent of Treachery (3), Noxious Gearhulk (3).
Historical Decklist issues must be removed by reparsing the unchanged source
text before explicit re-admission; the counts above use that path.

This contradicts the Task's premise that the current world admits all 1,000.
Corpus publication is stopped: no smaller population, replacement events,
removed cards, or new card mechanics are authorized by this implementation.
Resolve card support or revise the population contract before deriving and
publishing the corpus. No split has been published or used for training.
