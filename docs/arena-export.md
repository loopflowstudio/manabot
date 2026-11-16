# Arena-style recording export

`managym.export_arena.export_log(record, viewer=0)` reconstructs a completed
record with the shared walker and projects its native card capture. The
`manabot cards export-log` command passes its replay-verified native capture
to that same exporter. Neither path submits anything to a service.

## Coverage and provenance

The offline table `managym/arena_ids.json` was refreshed from Scryfall's
[collection endpoint](https://scryfall.com/docs/api/cards/collection) on
2026-11-16. It retains each authored printing's Arena id, name, set and collector
number. `uv run python scripts/cards/arena_ids.py` refreshes it explicitly;
export itself needs no network.

World `w17`, card digest
`b9344ef332cc07e1475e6a8860907ea84401ea427cec04a5cd23781e78c5818c`:
470 definition rows, 460 authored Scryfall printings, 355 mapped definition rows
and 115 unmapped rows. This is printing-specific coverage; another printing of
an unmapped card may exist on Arena. Tokens, back faces without a printing
identity and cards without an Arena id use `0`. Their movements remain present,
but an outside card database cannot identify them from that placeholder.

## Reader and limits

The conformance reader is the repository's pinned 17Lands client
`tests/fixtures/seventeenlands/mtga_follower.py`, SHA-256
`71117bdd8b1ef75cc019bbddfaeab9a69a02c5ef9efd741ec05a2e660918f77d`.
`tests/eval/arena_log_harness.py` replaces its service client with an in-memory
sink and disables network access. Tests check opening hands, deck ids, draws,
turns, seats, results and absence of opponent hand/library identities. Passing
this reader establishes that subset, not general Arena protocol conformance.

The output starts with an explicit synthetic marker and retains own deck and
hand, public card-zone objects, movement turn numbers, opening phase/step and
result messages. It contains neither the opponent's private cards nor hidden
object identities. No mulligan prompt exists in this recorded native setup;
the reader reports its unavailable count, `-1`. It requires more than five
state messages; shorter captures are refused instead of padded.

Against the private engine record, the export loses seeds and replay inputs,
policy annotations, unchosen legal options, hidden cards from the other seat,
precise phase/step and active-player changes between captured movements,
non-card effects and resolutions, and the identities of unmapped cards.
A transformed face without its own printing identity uses the placeholder.
The export remains text-readable without executing an engine; Academy does not
import it. Keep the engine record when exact replay or custom statistics matter.
