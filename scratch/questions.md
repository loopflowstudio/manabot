# MTG-276 — fixed-seat observations

## Dependency and working contract — 2026-11-11

Implementation awaits MTG-272 landing, then a sync through lf. The compress
step rechecked `lf task status MTG-272 --json`: PR #458 remains open, with
`implement · loop 3 pass 2` current and the Graph between steps after implement.
Its Steers change the same native vector, bindings, stubs and wrappers:
HandSnapshot naming, generation-based capture synchronization, a CapturedVector
read allowlist, reset semantics and passive-opponent removal. This checkout
has neither HandSnapshot nor current_game_generations. Do not treat the
documentation or this compression as completed implementation.

The intended contract is `docs/engine.md#fixed-seat-observations`; no separate
accepted design exists. Reversible API choices are
`write_observations_for_players(players, buffers, action_overflow="error")`,
per-slot records with `slot`, `player`, `game_id`, `seed`, `revision`, and the
same caller-owned NumPy-buffer API on both wrappers. These choices avoid an
implicit Torch-buffer lifecycle; they are not attributed user decisions.
MTG-272's remote-tracking docs/stub used those HandSnapshot identity fields
when read during write-docs; confirm its final interface after syncing.

## Implementation plan

1. Read MTG-272's final identity, validation and capture interfaces. Place the
   example beside its hand-snapshot example and update Perspective in
   `docs/python-api-audit.md`; neither exists in this checkout yet.
2. In `managym/src/agent/vector_env.rs`, validate all seats and initialized
   slots before writing each `Env::observation_for_player` through one native
   callback call. Reuse HandSnapshot's identity owner without collecting hands.
3. Reuse `input_buffer_config`, companion validation and `encode_into` in
   `managym/src/python/vector_env_bindings.rs`. Separate observation buffers
   from reward/done plumbing as needed. Preserve registered chooser buffers,
   configuration and last_info; propagate encoding/admission/capacity errors.
   Ordinary reset/step must do no extra observation collection.
4. Expose typed records and the method through PyO3, the installed stub,
   CapturedVector and Torch VectorEnv, without another lifecycle or capture.
5. Use bounded native debug and installed public-API parity fixtures: all input
   versions, mixed/nonchoosing seats, closed decklists, player control, inactive
   slots, terminal replacement and same-seed resets. Check identity pairing,
   unchanged chooser/capture state, no RNG advancement, malformed requests and
   buffers, supplied capacities and overflow policy.
6. Run the focused debug test, rebuild the extension from the root and replace
   it atomically, then run installed-API parity tests and the doc example.
   Gate owns affected suites and full acceptance.

### Delete — do not maintain

No public predecessor exists. If splitting the buffer owner, remove replaced
combined observation plumbing in the same change; retain one encoder and
shape-validation path. Preserve reset/step reward/done writes, chooser views,
Commands, captures, legality, frozen layouts and evidence. No migration or
compatibility shim is needed.

## Checks and history

Compression consolidates the duplicated handoff and removes stale write-docs
status; executable code is unchanged. The example still requires implementation
and MTG-272 before runtime verification. Write-docs presentation with
`lf open --host desktop --file docs/engine.md` returned
"No Desktop Files view attached; nothing opened."

Compress checks: `git diff --check` and `uv run --no-sync python` with
`ast.parse` on the fixed-seat example passed; no runtime tests or build needed
for these prose/comment edits. `lf context` remains within both scratch limits.

Unrelated b015d5/C1/C2 notes, including exact quotes, budgets, run IDs and
contrary findings, remain at
`757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md`.
