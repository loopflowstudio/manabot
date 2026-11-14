# MTG-276 — fixed-seat observations

## Dependency and working contract — 2026-11-11

The outcome is one opt-in native batch call writing a requested native seat's
viewer-safe observation per slot, including nonchoosing seats, into caller-owned
buffers. Scalar `Env::observation_for_player` already provides that projection;
the vector API and both wrappers still lack it. Only the guide and plan exist
here; implementation and runtime verification remain.

Jack Heart (Task brief, 2026-11-11): "Yeah i think we should do that one, it can
be its own task, but start it".

Implementation awaits MTG-272 landing, then a sync through lf. Realign's
`lf task status MTG-272 --json` read on 2026-11-11 found PR #458 open and
`compress · loop 3 pass 2` running. Its Steers change the same native vector,
bindings, stubs and wrappers: HandSnapshot naming, generation-based capture
synchronization, a CapturedVector read allowlist, explicit-seed Torch resets,
Gym auto-reset removal and passive-opponent removal. This checkout has neither
HandSnapshot nor current_game_generations. MTG-276 has no Wave.

The intended contract is `docs/engine.md#fixed-seat-observations`; no separate
accepted design exists. Reversible API choices are
`write_observations_for_players(players, buffers, action_overflow="error")`,
per-slot records with `slot`, `player`, `game_id`, `seed`, `revision`, and the
same caller-owned NumPy-buffer API on both wrappers. These choices avoid an
implicit Torch-buffer lifecycle; they are not attributed user decisions.
MTG-272's remote-tracking implementation at `64779909e75041ffa9f35b4bc532a9240b688c55`
uses `env.capture_generation`,
`env.seed()` and `env.selected_revision()` for HandSnapshot identity. That
snapshot predates its later Steers; final identity and wrapper interfaces need
rechecking after sync. No new product decision is needed for the current plan.

Native `Observation::for_player` owns visibility and exposes legal actions only
to the published chooser (`managym/src/agent/observation.rs`). That projection
deliberately omits presentation events, even for the chooser; parity fixtures
must compare to the scalar fixed-seat projection, not ordinary step output.
The batch must preserve it, not remap requested seats or construct observations
from privileged hands. Jack Heart's MTG-272 direction: "the output of the engine
is just waht was in hand, choosing to use it as a belief SL target is the business
of hte client". The guide's pairing example therefore describes snapshots, not
engine-owned supervision. Frozen layouts, worlds and evidence stay unchanged.

## Implementation plan

1. After the dependency lands and syncs, read its final identity, validation and
   capture interfaces. Place the example beside its hand-snapshot example and update Perspective in
   `docs/python-api-audit.md`; neither exists in this checkout yet.
2. In `managym/src/agent/vector_env.rs`, validate all seats and initialized
   slots before writing each `Env::observation_for_player` through one native
   callback call. Reuse HandSnapshot's identity owner without collecting hands.
3. Reuse `input_buffer_config`, companion validation,
   `SendObservationFieldSlices::encoded_row_mut` and `encode_into` in
   `managym/src/python/vector_env_bindings.rs`. `ObservationBuffers` currently
   requires reward/done arrays, while its field slices already separate the
   observation write. Factor that owner enough to accept observation-only
   buffers, retaining one shape-validation and encoding path. Validate every
   supplied array before writing; capacities come from those arrays. Preserve
   registered chooser buffers, configuration and last_info; propagate admission
   and capacity errors, with error-by-default action overflow. Failed writes
   require discarding all output rows. Ordinary reset/step does no extra work.
4. Expose typed records and the method through PyO3, the installed stub,
   CapturedVector and Torch VectorEnv, without another lifecycle or capture.
   Integrate with MTG-272's explicit read surface instead of relying on this
   checkout's unrestricted `CapturedVector.__getattr__` forwarding. A read must
   neither change capture prefixes nor register its buffers as execution input.
5. Use bounded native debug and installed public-API parity fixtures: all input
   versions, mixed/nonchoosing seats, closed decklists, player control, inactive
   slots, terminal replacement, root reassignment and same-seed resets (including
   Torch's explicit-seed rebuild and replacement at the maximum seed). Check
   identity pairing,
   unchanged chooser/capture state, no RNG advancement, malformed requests and
   buffers, supplied capacities and overflow policy.
6. Run the focused debug test, rebuild the extension from the root and replace
   it atomically, then run installed-API parity tests and the doc example.
   Gate owns affected suites and full acceptance: Rust formatting, all-feature
   Clippy and debug suite per `docs/development.md#validation-by-change-area`,
   plus `tests/env/test_observation.py`, `tests/training/test_action_cutoff.py`
   and `tests/verify/test_native_buffer_capacities.py`. Add the affected wrapper
   and capture regressions from MTG-272 after sync. No timing benchmark is needed
   to establish opt-in behavior; verify the unchanged ordinary call paths.

### Delete — do not maintain

No public predecessor exists. If splitting the buffer owner, remove replaced
combined observation plumbing in the same change; retain one encoder and
shape-validation path. Preserve reset/step reward/done writes, chooser views,
Commands, captures, legality, frozen layouts and evidence. No migration or
compatibility shim is needed.

## Checks and history

Realign confirmed the missing batch API and refreshed dependency status and
remaining integration checks; executable code and the guide are unchanged.
The example still requires implementation and MTG-272 before runtime
verification. Write-docs presentation with
`lf open --host desktop --file docs/engine.md` returned
"No Desktop Files view attached; nothing opened."

Compress's `uv run --no-sync python` with `ast.parse` on the fixed-seat example
passed; this is syntax evidence only. Desktop presentation remains for demo;
the installed-API example can run headless once implemented. Realign changes
only this plan, so no runtime tests or extension rebuild are needed here.

Unrelated b015d5/C1/C2 notes, including exact quotes, budgets, run IDs and
contrary findings, remain at
`757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md`.

## Sync resolution — 2026-11-14

Retain this branch's fixed-seat plan despite main deleting the shared scratch
file. Combine main's engine guide additions with the fixed-seat section, placing
it after Hand snapshots. Use current `player_policies` and pair each hand
snapshot by its list index: `HandSnapshot` has no `slot` field. These are
reversible integration choices; the fixed-seat API still needs implementation.
This sync brings in HandSnapshot, generation-based capture synchronization and
the wrapper read interfaces; the earlier missing-interface statements describe
the pre-sync checkout. Recheck their final contracts before implementation.
