# Release visual references v7

The w9 Temur mana rules change the seeded GW Allies trajectory. At command 5,
Badgermole Cub's mana bonus makes Allies at Last available before combat. The
curated policy casts it, changing later prompts and board states. The GW game
now takes 95 Commands instead of 58, still ending in a hero win on turn 35.
UR seed 51 retains its 24-Command sequence, opponent win, and terminal turn 14.
All ten prompt families remain covered across the two games; PAY_OR_NOT is now
covered only by UR. No policy, assertion, or screenshot tolerance was relaxed.

The recorded failure was head `470ee016df1cd3e797e6dfcc13c29e80461ab82a`,
[Visual Reference Gate run 39635681615](https://github.com/loopflowstudio/manabot/actions/runs/39635681615).
Its stale GW sequence failed before screenshot comparison. The old v6 PNGs and
[matrix](../v6/matrix.json) remain frozen. Retained navigation positions also
have a separate `academy/fixtures/choice-navigation-w9.json`; the original
fixture remains unchanged.

Capture environment: an Ubuntu 24.04.5 x86-64 container, Node 22.23.4,
Playwright 1.61.1, Chromium 149.0.7827.55, Python 3.12.3, and the release-built
native extension from the recorded head. Viewport, bundled fonts, locale,
timezone, reduced motion, scale, and strict pixel tolerance match the active
matrix. This is local reproduction of the pinned Linux profile; the published
head's GitHub-hosted checks remain separate delivery evidence.

All 18 captures were visually reviewed. Seven are byte-identical to v6:
opening board, priority, scry, all three reconnect indicators, and the UR
terminal. The other captures retain the same controls and curated assets while
reflecting current prompts, board states, or rasterization. Both terminal
results, every prompt family, keyboard interaction, accessibility audits,
network guards, and asset checks passed during capture (14 browser tests).

Verification on that Linux profile passed with snapshot updates disabled:

```bash
uv run --no-sync pytest tests/academy/test_release_prompt_matrix.py tests/academy/test_choice_navigation.py -q
npm --prefix frontend run test:e2e:release
```

Results: six Python tests and all 14 browser tests passed. The generation run
also passed all 14 browser tests before the strict comparison. Ruff passed for
both focused Python test modules. GitHub CI must independently verify the
published repair head; local checks do not establish merge or Task completion.
