# MTG-236 decisions and remaining acceptance

PR #411 is merged; MTG-236 remains open. [Measurement evidence and remaining
acceptance](mtg-236-postmerge-verification.md) own the benchmark receipts and
review proposal. No qualifying latency receipt exists yet.

## Accepted implementation — 2026-11-02

- `Revisions` has `characteristics`, `mana_pools` and `triggers`. The brief's
  `offers` counter has no consumer: `current_action_space` is recomputed per
  decision. A counter belongs with its first derivation.
- `UnspentManaAtLeast` reads mana pools. `StaticIndex::reads_mana_pools`
  makes views follow that revision only while such a source is present.
- Profiled hot writes use typed accessors; other writes conservatively
  advance every counter through `DerefMut`. Debug reads recompute cached
  views and indexes and assert equality.
- `TrackedState::move_card` advances characteristics for battlefield or
  graveyard moves, triggers for battlefield moves. Graveyard trigger sources
  remain scanned separately; hand/library/stack/exile moves bump neither.
- Items 1–4 shipped in PR #411. Subtype interning was deferred in review:
  after those changes, characteristics string building was under 1% of
  playout time. No speculative interning or mana-search rewrite is selected.
- Jack Heart's "lg2m ship it" authorized delivery, not certification of the
  unmeasured latency target.

T5's retained `base_power` / `base_toughness` fields describe layered
battlefield P/T; restoring printed P/T requires the separate protocol decision
already recorded in Engine memory. This performance work does not reopen it.

## Retained context

The shipped design, review, shipping approval and gate evidence are at
`1e0343b25:scratch/mtg-236-derived-state-revisions.md`. Inherited T5 and
decklist-disclosure notes are at `78129b059:scratch/questions.md`; they do
not select new work. Before this compression, uncommitted notes were saved
under `.cache/mtg-236/pre-compress/`; this file's original Git blob is
`f6721fb172961a042db02631e282ff77857b39f9`.
