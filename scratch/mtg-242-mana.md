# MTG-242 — mana spending and resources (history and live decisions)

The previous implementation, its plan and every measurement receipt are at
`e5b71f269e5ba4a7b4f75ba641b7ca59b95a470b` (`scratch/mtg-242-mana.md`,
`research/mana-risk-2026-11/`). They describe that source only; the port
plan in [mtg-242-port.md](mtg-242-port.md) supersedes its integration strategy.

## Live decisions carried forward

- Jack Heart's 2026-11-04 Task directive: one reviewable PR, stop at review;
  no merge or training experiment.
- Jack Heart's 2026-11-04 acceleration direction after a +6.7% step / +8.0%
  search cost versus main: "make it stupidly fast". The 10% Engine Budget is a
  guardrail, not evidence of that goal; no percentage was promised. Old
  microbenchmarks prove neither a current speedup nor current correctness.
- Fully explicit selection of every land and pool allocation is separate work.

## Evidence from the old line still relevant

- Real-card witnesses chosen after Scryfall checks on 2026-11-04: Scarecrow
  Guide (`woe`, `3dc08461-bec6-4a18-b782-da4c92abb789`) for a non-tapping
  once-turn filter; Supportive Parents (`spm`,
  `d3fe8a5b-4166-46cc-b910-71cd1a19ae1b`) for repeatable shared taps. Both
  reported Standard legal.
- Old v8/v9 schema numbering is historical, not a layout reservation on main.
- Old measurements found a fixed-payment fast path and shared search state
  recovered near-base cost on that source; main's payment model differs, so
  any optimization must be re-justified by current profiles.
