# MTG-188 assumptions and dependencies — 2026-10-26

Jack Heart accepted `scratch/mtg188-context.md` with “seems fine, pursue” in the
primary Task conversation. Implementation and publication are authorized; new
paid execution and inherited spending allowances are not. The accepted design,
measurement boundaries, evidence corrections and gate commands live in that file.

- The current checkout integrates MTG-184/185 at
  `558e0186b585228dd2130cf14d9cb40de217e66f`. Historical L4 receipts retain source
  `e747f0445e31ac08d674f855ad0128a4a722b100` and their older renderer. Runtime
  integration is complete; historical rendering does not establish new results.
- Use existing evaluation Perf and raw receipts with a typed manifest. The
  execution protocol freezes source/native/input/runtime facts; the manifest
  binds observations, not authority to execute. Rebuilding runs no model.
- Empirical acceptance remains incomplete. The selected small trained reference
  is MTG-181 seed 18101 raw u616. `ArtifactStore.download` failed because
  `R2_ACCOUNT_ID` is missing; local bytes and current setup admission remain
  unverified. No authentication was changed. Random-weight fixtures cannot
  replace a trained baseline.
- New GPU/LLM measurements require explicit hardware, inclusive time/cost,
  cleanup and attempt limits. CPU observations independently need the pinned
  checkpoint. Artifact publication and actual CPU/GPU plus LLM single/many-game
  observations remain open; the Task cannot be marked complete.
- Missing directories are unavailable; mismatched pins fail validation. Only
  opposing completed/failed declarations and observations are contradictions.
  Completion, replay coverage and availability remain independent. Shared
  allocation pins preserve identity and once-only cost accounting, including
  missing/restored external files; unpinned external measurements are not used.
- Policy readiness precedes model admission; LLM readiness includes loading.
  Worker, supervisor and service latency populations remain separate. RSS and
  CUDA allocator peaks do not establish whole-device occupancy.
- Reversible execution choices: explicit `play_cell` legs preserve request
  identity; sequential LLM games share one session and charge readiness once.
  Whole-attempt wall includes preparation, collection and cleanup, excludes the
  final ledger write, and never sums overlapping worker intervals. Outer failure
  can coexist with completed games; a completed attempt cannot contain failed
  game evidence.
- Collection uses the original deadline/RSS ceiling in a supervised child.
  Interrupted collection retains received pins and raw progress, marks inventory
  incomplete, and projects no game metrics from unpinned files. Incomplete child
  cleanup prevents subsequent writers. No parent hashing/comparison loop returns.
- Instrumentation comparison selects the latest completed same-compute,
  opposite-instrumentation observation (including matching diagnostics), then
  falls back to the first completed cohort observation. Selection and receipt
  reads remain inside supervised collection; scalar-first/rotated order is covered.
- The completed implement session `session_bedb0277823c41e79442e82ebdc8c15b`
  failed to settle because SQLite was locked. Its effects are preserved; stopped
  Graphs do not authorize replay. Jack Heart authorized a direct-session workaround for launch failures.
  Compression, sync and review are complete; the mixed CLI/headless proof passed,
  and affected suites passed (161 tests, one CUDA-only skip). Empirical dependencies do not block independent software work.

Inherited MTG-186 notes remain at
`a0dead2b5195853e0931bf244d84529c08944acb:scratch/questions.md`; MTG-184 notes
remain at `ce43a5cf76d52b99c4c13c6debb21aab79ac9f82:scratch/questions.md`.
The pre-compression uncommitted notes are preserved under
`.runs/inference-compression-review/before/scratch/`.

## Sync integration — 2026-10-27

Preserve the MTG-188 assumptions above and the offline inference import boundary.
The target introduces published Etude SDK 0.2.8, run-owned card evidence, and
SDK-owned historical participant/seat conversion; its updated guidance applies.
The target Task 9da33d notes remain at
`5051c493fc4d47a3a63c6a5043d47639af6c5018:scratch/questions.md`.
