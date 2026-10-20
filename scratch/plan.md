# MTG-155 plan: open any trained checkpoint in Academy

Status: implemented, uncommitted. Jack Heart (2026-10-20 steer): autonomous —
build, then ship without waiting at demo.

## Exists

- `manabot/model/catalog.py`: weights-only inspection (`inspect_checkpoint`),
  world-binding mismatch sentences (`checkpoint_incompatibility`, e.g.
  `trained in w5, engine is w6`), discovery over `.runs/`, explicit paths, run
  directories and Tournament files pinned by SHA-256 (`discover_checkpoints`).
- `academy/opponent.py`: `opponent_catalog` (configured candidate first, then
  catalog; incompatible entries listed with reason) and `catalog_opponent`
  (digest → verified `Opponent`, refusal names the mismatch).
- `academy/server.py`: `GET /api/opponents` replaces `GET /api/opponent`;
  `new_game` resolves `opponent_sha256` through `_select_opponent`, checks the
  chosen decks against the binding in either seat, records
  `GameConfig.opponent_checkpoint` on the trace. `ACADEMY_PLAY_CANDIDATE`
  still works and remains the default when configured.
- Frontend: Opponent menu has a "Trained checkpoints" group
  (`checkpoint:<sha256>` values), disabled entries show the reason, details
  line shows producer, decks, model shape, parameters.
- Docs: `academy/README.md`, `docs/local-trained-challenger.md`.
- Assumptions in `scratch/questions.md` (sampled play, no seat control,
  Tournament entrant resolution, candidate option value change).

## Remaining

- Commit and publish. The env-gated `frontend/e2e/trained-challenger.spec.ts`
  needs a configured candidate; it is left to gate/CI.

## Check

`uv run pytest tests/model/test_catalog.py tests/academy/test_checkpoint_catalog.py tests/test_package_boundaries.py tests/academy/test_recorded_attempts.py tests/academy/test_server.py -q` → 28 passed; vitest game.svelte.test.ts 15 passed; svelte-check 0 errors (2026-10-20).
