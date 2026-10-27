# MTG-206 assumptions and dependencies — 2026-10-27

The Task directive (Jack Heart, 2026-10-27) is the accepted design; the plan
and remaining work are in `scratch/mtg206.md`. Inherited MTG-188 notes remain
at `37e5ccd8d65527f63486f1d9770af20f612fa8c8:scratch/questions.md`.

- **Blocks are unreleased.** The directive says blocks shipped "from etude-sdk
  0.2.9 onward", but etude #63 (`2323782`) merged after the v0.2.9 tag, so no
  published release has them. manabot pins etude-sdk to that commit through
  `[tool.uv.sources]`, keeping `==0.2.9` in the requirement. Before landing,
  release etude (one release carries #63 and #68) and pin it; a PyPI
  install of manabot otherwise gets 0.2.9 without blocks.
- **EXP-30 merged, unreleased (2026-10-27).** etude #68 (`e670a08`) adds
  `estimate(keys, numerators, denominators) -> list[Estimate | None]` over
  blocks × columns, keyed by deal. Until a release carries it, intervals use the
  per-ratio `estimate` behind `card_query._intervals`, the one function to change.
- **`excluded` now counts player-games**, like `games`: the rollups hold per-seat
  rows, so a game excluded for both seats counts twice when no subject is chosen
  (it was distinct games). With a subject, the counts are unchanged.
- **`source_commit` added to both rollups' `by`** to keep `cards data --commit`;
  not in the directive's column list.
- **Historical per-file runs** (MTG-186 cards, the 0.2.5 fixtures, the MTG-183
  bucket copy) are readable only after `etude convert-eval`. Conversion of retained
  data is a separate explicit step; the MTG-186 docstring names the command.
- **elemental_long_rent.py / GPU `demo` command** live on the MTG-183 branch
  (`jack/mtg-183-...`), not on main, so the demo branch's launcher was not
  brought over. The small real L4 eval needs a launcher on this branch.
- PR #389 overlaps only at import lines of `cards.py`, `mtg186_cards.py` and
  `elemental_long_eval.py`.
