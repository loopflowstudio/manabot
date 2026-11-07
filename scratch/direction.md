
## Calibration and rehearsal readings (2026-11-07)

Jack Heart: small PRs and experiments under USD3 need no ask; only big runs do.
Spent USD1.51 of the reserve: calibration USD0.67 (incl. USD0.04 failed first
attempt), rehearsal USD0.84. Merged fixes: #440, #441, #443.

- w64 167 s/update (S 1,164); w64 one producer 193; w160l8 400 (S 486). Size gate
  passes. w64 pods reached 4 updates, so medians are updates 2–4.
- Rehearsal `tla-frontier-c5fba4-rehearsal-v1-4610a56bce6f`: 32/32 matches, median
  43 s, slowest 121 s, 1 of 128 games invalid, so its gate reads failed. That game
  overflowed cards (`agent_cards plus sideboard`, 128 per player), w160l8 last vs
  update 0, 639 decisions, 55 dropped actions.
- Launched through a private ledger at `~/.manabot/c5fba4-etude-ledger`: other
  Tasks' `out_of_time` passes block the shared one with `launched_pass_failed`.
- Awaiting Jack Heart: rehearsal gate treatment, fact-embedding check length.

## Rehearsal and check decisions (Jack Heart, 2026-11-07)

- **Capacity overflow stays a labelled abort, and capture the game state** so we
  can see what is happening. The rehearsal gate fails only on time kills.
- **Fact-embedding check: 75 updates, budget USD8** (arms plus pass).
