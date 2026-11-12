# MTG-311 plan

Pin `etude-sdk==0.2.16` (steer: "Etude v0.2.16 is released (tag on c368227, on PyPI) and includes a3a10dd; pin etude-sdk==0.2.16."). Public minimum raised to `>=0.2.16` everywhere (log_table needs it).

Built:
- `RunContext.log_table`; training logs `self_play_outcomes` with every clock after `log_update`.
- `update_metrics._updates` merges metric values per update within an attempt (the table row is a separate row at the same clocks), refuses `self_play_outcomes/table` naming `bf4e280f7`, and projects `RunData.read_table` rows.
- Deleted `manabot/training/outcome_tables.py`; tests rewritten in place (same filename, CI list unchanged).
- `scripts/diagnose_outcome_index.py` ported to `manifest.Index` (0.2.16 removed `write_manifest`); dated line in `experiments/self-play-collection-273.md`.

Checks: focused outcome/self-play/rl_runtime 14 passed; isolated Etude suite 31 passed.
Gate (focused): 115 passed, 1 failed (pre-existing packing digest, see questions.md); isolated Etude suite 31 passed; ruff clean.

Compress: deleted `test_damaged_selected_outcome_table_fails` and the direct `read_table` assertion (Etude's own behavior, tested there); commented why `_updates` pops the `self_play_outcomes` digest metric; shortened the retired-format paragraph in `docs/training.md`. Kept `scripts/diagnose_outcome_index.py` (the 273 report cites it). `pytest tests/training/test_outcome_tables.py`: 5 passed; ruff clean.

Proposed Wave memory edit (not applied; `.lf/wave/` is read-only here): the "Self-play metric indexing (MTG-273)" section's second paragraph describes manabot-written digest tables under `artifacts/self-play-outcomes/`. After this change Etude 0.2.16 `log_table` owns storage, selection and verification; manabot refuses the older references, naming `bf4e280f7`.

Realign (synced with origin/main at `70a03a94b`): main commits after `bf4e280f7` still read the retired format, so docs/training.md and `update_metrics.py` now call it an example revision, not the last one. Remaining: publish/land; the packing-digest failure in questions.md stays a separate bug Task candidate; the Wave memory edit above is still only proposed.
