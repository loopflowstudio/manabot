# GUI

> **Status: restarted 2026-07-09** with a new charter. The original wave
> (board cards, replay viewer, human-vs-passive play, polish — shipped in
> e234933) was archived as product work. It returns as an **instrument**:
> human-vs-bot play, for learning.

## Vision

A browser table where a human plays full games against any agent the project
produces — a policy checkpoint, flat-MC search at chosen N, or (later) the
expert-iteration ladder rungs. "For learning" in three senses, in priority
order:

1. **The human learns what the bot is.** Aggregate metrics say *that* a policy
   wins; watching says *how*; playing against it says how it *feels* — where
   it's sharp, where it's exploitable, what it never does. Every wince at the
   table becomes a hypothesis; every hypothesis becomes a store metric.
   (Discipline inherited from the replay decision: play generates hypotheses;
   only aggregates confirm them.)
2. **Human games are a benchmark no self-play loop can fake.** Win rate vs.
   the project's owner, a real Magic player, on the interactive-deck mirror is
   the first human-anchored strength measurement — and the demo that anchors
   the release bar (wave/search/01-experiment-loop.md, Release bar).
3. **Human games are data.** Traced like every other game, replayable in the
   existing viewer; the off-model action stream a human naturally produces is
   the surprise-ledger stress test the beliefs wave will eventually need
   (wave/search/02-beliefs-design.md, dormant).

### Not here

- Visual polish beyond unambiguous-and-correct (the archived product charter
  stays archived)
- Public deployment / hosting (release rides with the first citable result)
- Drafting UI, deckbuilding UI, human-vs-human networking

## Inherited foundation (from the original wave — already works)

Human-vs-**passive/random** play through the browser exists: SvelteKit +
FastAPI/WebSocket, one connection = one session, villain auto-played
server-side, event-level traces, replay API, Scryfall images. Key decisions
that stand: raw `managym.Env` (card names and zones, not training tensors);
card `name` exposed via PyO3; `_mini_fastapi.py` fallback; traces record every
engine step. Known gaps that stand: mana-pool display blocked on bindings;
Scryfall rate limits mitigated by tiny pool.

**The single missing piece was always "Not here: trained model opponents."**
That line is the new wave.

## Goals

1. **Pluggable opponents** (in flight — dispatched 2026-07-09): policy
   checkpoint (.pt), flat-MC search-at-N (31ms/decision at N=64 — real-time),
   random. INTERACTIVE_DECK mirror default. Hidden-info integrity asserted in
   tests (bot hand never in human-facing payloads). Step doc from the build
   agent lands as `wave/search/02-play-interface.md`; relocate here on merge.
2. **First sessions**: Jack vs. search-64, then vs. the strongest C5 policy.
   Output is not a win/loss — it is the **competency checklist**: recognizable
   Magic skills observed or missing (block math, removal targeting, wipe
   timing, plays-around-nothing), each converted to a measurable aggregate.
3. **Best-of-N protocol**: fixed match format (best of 10, alternating seats)
   so "the bot beat Jack" is a number with provenance, not an anecdote. Match
   results stored with opponent config + checkpoint hash.
4. **Presentation batching** (design decided 2026-07-09): decouple deciding
   from narrating. The bot thinks against a wall-clock budget (deadline-based
   search, no named tiers — **default: 1s per decision batch**, i.e. per
   villain sequence between two hero decision points, divided across that
   sequence's decisions); the UI presents each villain sequence (everything
   between two hero decision points — already the server's natural unit) as
   semantic beats of ~400ms: one beat per game-meaningful event + its visible
   consequence ("attacks with X, Y, Z" is one beat, "Pyroclasm → sweep" is
   one beat). Click-to-skip. One thinking indicator per composition, not per
   micro-decision. Implementation rides with the C5 deadline-searcher work.
5. Trigger-linked, later: when wave/search/02-beliefs-design.md activates, the table becomes the
   bluff observatory — the first place a human is shown a calibrated
   represent-the-Counterspell line and has to decide whether to play into it.

## Metrics

- Time from launch to playable game vs a **trained/search opponent**: under a
  minute
- Human-vs-bot match results by opponent config, alternating seats, in the
  store
- Hypotheses filed per session (a session producing zero new metrics was
  entertainment — fine, but say so)
- Competency checklist coverage: each item measured in the store or explicitly
  open
