# MTG-206 card evaluation demo — 2026-10-27

Experience reviewed: generate card statistics from retained Etude blocks, change
cohort filters without losing exclusions, and verify/replay delivered evidence.
This is a headless autonomous demo under the caller's explicit delegation to
stand in for Jack Heart. No human observed this run or gave new feedback.

Design: [current plan](mtg206.md), the MTG-206 Task directive returned by
`lf task status MTG-206 --json`, and [usage contract](../docs/card-evaluation.md).
No product or implementation redesign is proposed by this demo.

## Original pair recovered and demonstrated

The earlier conclusion that the original converted data was unavailable is
**superseded**. The old worktree is gone, but the original converted run remains
in a temporary scratch directory. The retained Etude database's `kept.statsdb`
store identity led to its location. [Source receipt](mtg206-demo/sources.json)
records that path and the earlier comparison/timing logs.

The final code generated the [u005950-versus-heuristic page](mtg206-demo/u005950-vs-heuristic.html)
through the actual `cards data --html` CLI, using the original source and the
existing kept database. The selection contains 5,000 eligible player-games,
5,000 deal seeds, 90 cards, one run, 80.12% wins, and no exclusions.

There are 363 deck/position combinations, expanded into 1,452 embedded keys
because each fixed subject/opponent also has an all-values key. Every embedded
variant's counts, rates and summary equals the retained pre-batched page after
excluding the two interval endpoints. See [comparison](mtg206-demo/pair-comparison.json).
The older raw-row comparison also reported zero differing cards; its exact
output is copied into the source receipt. It was not rerun in this demo.

The real CLI completed in **44.67 seconds wall** (13.23 seconds user CPU,
3.64 seconds system CPU): [CLI receipt](mtg206-demo/pair-cli.log).
A second run separated database catch-up from page generation through the same
public `card_db` and `write_page` path:

- Database catch-up: 30.48 seconds wall / 10.21 seconds CPU.
- Page generation: **16.01 seconds wall / 5.28 seconds CPU**.

[Timing receipt](mtg206-demo/pair-timing.json) and
[reproduction script](mtg206-demo/time-pair.py) retain the measurement. This
establishes generation in seconds on the required original cohort. It does not
reproduce the Task's 1.5-second prototype. The previous final-pair measurement
was 51.30 seconds wall / 13.40 seconds CPU for the page; this is historical
context, not a controlled host-performance comparison. The L4 replay audit was
running during these measurements; no synthetic load was induced. Catch-up cost
is visible rather than silently included in the page claim.

Reproduce from the repository root (the original temporary source must remain):

```sh
ETUDE_HOME="$PWD/.runs/mtg206/etude-home" uv run --extra evaluation python scratch/mtg206-demo/time-pair.py
```

## Exclusions and filter experience

The actual generated page script ran under Node with an injected minimal DOM.
This demonstrates script behavior, not browser layout, accessibility geometry,
or visual approval. There is no rendering environment and no screenshot claim.

A controlled mixed cohort went through the existing evaluation fixture and
`write_page`, producing [mixed-cards.html](mtg206-demo/mixed-cards.html).
Selecting draw preserves one run, one replay failure, zero eligible player-games,
and an unavailable win rate. Its table says:

> No eligible games. Card rates are unavailable for this selection.

An absent deck/position combination instead says “No games match these filters.”
A search that hides all cards says “No cards match.” Clearing it restores rows.
The generated L4 page also permits subject and position selection; verify-one
on the draw reports 48 eligible player-games, 12 deals and 52.1% wins.

[Interaction evidence](mtg206-demo/page-interactions.json) and the
[executable probe](mtg206-demo/page-interactions.cjs) retain these assertions.
The entire card-evaluation suite passed: **42 tests in 277.92 seconds**, exit 0.
[Command and result](mtg206-demo/card-tests.txt).

## Real L4 evidence

This demo reused the delivered smoke run `mtg206-cards-l4-verify`, launcher run
`b6fc065fb4e0`, collected at `/private/tmp/mtg206-smoke2/runs`. Its recorded
approximately $0.03 cost and successful L4 launch are prior evidence in the
[current plan](mtg206.md); no new paid evaluation was launched.

The current CLI regenerated [l4-cards.html](mtg206-demo/l4-cards.html) from its
sealed block: 96 games, 192 pooled player-games, 12 deals, 40 cards, one run,
189 embedded filter variants, no exclusions or failed matches. The distinction
between physical games and both players' observations is expected.

The replay audit ran through:

```sh
ETUDE_HOME=/private/tmp/mtg206-smoke2/home uv run --extra evaluation python -m manabot.eval.cards audit --experiment mtg206-cards --source /private/tmp/mtg206-smoke2/runs --replay
```

It exited 0: **96 games, 96 replayed, no problems**. The exact result is
[l4-audit.json](mtg206-demo/l4-audit.json). This freshly proves delivered blob
reads, digest verification, row reconstruction and native replay with the
current code.

## Decision and remaining work

**Advance.** Acting under the explicit headless delegation, the agent accepts
this demo because the stated required pair now generates in seconds, its
statistics are preserved, and the delivered L4 block passes page generation and
full replay audit. The Task describes 1.5 seconds as a prototype measurement,
not a hard acceptance cap. This interpretation is recorded in
[assumptions](questions.md); no human waiver or new human approval is claimed.

No unresolved product question or repairable behavior defect was found. The
missing-data premise is resolved rather than waived. No design change was
agreed or proposed. Full-CLI catch-up remains slower than page generation and
is reported explicitly; this demo does not promise a 1.5-second command.

Recommended next action: advance to the caller's next review/delivery stage,
using these receipts and completing its remaining affected checks. This demo
does not publish, merge, complete the Task, restart a Graph or close a Session.
All commands started by this demo have exited; no service was started.
