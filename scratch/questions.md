# Questions and assumptions (MTG-435)

Choices made with nobody in the conversation; each is reversible in the PR and
listed in `scratch/concept-review.html`.

- **Single ordered save-admission sequence + non-optional
  `RestoredContinuation.host`.** `continuation.admit_save_format(folder)`
  refuses `training.safetensors` and `learner.pt` before `resume_model` /
  `resume_readings`, which still precede `counts_sha256` and tensor loading.
  `RestoredContinuation.host` becomes required (`HostContinuation`) and
  `resume_training` returns `tuple[Counts, RestoredContinuation | None]` so the
  dead `host=None` legacy half-state cannot be represented.
- **`record_actions` moves to `manabot/sim/replay.py`; `manabot/sim/recording.py`
  is deleted.** After removing `TrainingGames`, `TrainingGame`, `_read_receipt`,
  and `record_games`, `record_actions` (without dead `option_counts`) is the
  only function left and only builds Command traces for `write_trace` (used by
  `scripts/release/academy_replay_smoke.py`). Moving it to `manabot/sim/replay.py`
  eliminates the duplicate `recording.py` filename across `managym` and
  `manabot`. `.github/workflows/manabot-integration.yml:68` drops the deleted
  `tests/sim/test_recording.py`.
- **Legacy source admission deleted with `manabot/env/replay.py`.** Its
  `source_sha256` comparison cannot succeed for any record a previous revision
  wrote, so callers move to `managym.replay.require_replay_runtime` and
  `failures.runtime_identities` goes. This edits one error string in
  `managym/replay.py` and one sentence in `docs/engine-python.md`.
- **`TrainingExposureReceipt`, `NativeTrainingGame`, and `NativeTrainingTrace`
  go** because `_read_receipt` was their only non-test reader and `managym`
  reads `training-exposure-v1` directly.
- **Legacy belief zips get `ValueError("unknown sampler checkpoint format")`**
  from wrapping `decode_safetensors` + JSON decoding errors, with no
  `PK\x03\x04` check left in code; `test_legacy_sampler_zip_is_refused` verifies
  both Torch and MLX reject a real Torch zip with that message.
- **No version bump to 0.3.0 in this PR.** The removals land on main; the
  release Task sets the version.
- **Release notes are not written here.** The 0.3.0 notes need a "removed"
  list; `scratch/plan.md` "What goes" is the source for it.
- The model-settings test also built a legacy save to exercise the initial-policy
  fallback. Both model changes now use portable saves. The cross-backend
  refusal test covers the historical save with its model and readings omitted.
- `StepRecorder` is an annotation-only import in the Torch learner. A runtime
  import under its original name would keep the retired import working.
- Integration CI runs the new replay test in the retired export test's slot.
- The local-update sampler fixture publishes an untrained model with the
  safetensors codec. It has no training dataset or fit metrics.
- Gate needs a disposition for the 20 consumer failures that also reproduce
  with the base sources. Their search and belief behavior is outside MTG-435.
