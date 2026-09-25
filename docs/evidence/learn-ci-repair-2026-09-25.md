# Learn CI repair — 2026-09-25

Failed head: `83ca02e8a76b73900c16ac42ba347eafbce79b31`,
[PR #188](https://github.com/loopflowstudio/etude/pull/188),
[CI run 36177994772](https://github.com/loopflowstudio/etude/actions/runs/36177994772).
`lf rebase` left that head unchanged before investigation.

## Causes and repairs

- Clean-machine Play imported `managym._managym` through pack validation before
  building the native extension. The launcher now validates the compiled pack
  after native preparation, inside the cleanup boundary. A regression test
  checks ordering and cleanup on pack validation failure.
- Rust's checked semantic v1 receipts bind the pre-Learn source, content and
  state hash. The new [v2 corpus](../../conformance/semantic-kernel-v2/) keeps
  the same four seeds, 560 commands and terminal outcomes. Differences are
  limited to source/content digests, state hashes, ten Learn prompt/action-space
  changes and eight `SelectCard` → `LearnDiscard` actions. Historical v1
  receipts remain unchanged. This scheduling harness still uses custom main
  decks with empty sideboards and the general content pack; it does not certify
  full authored Learn setup or live/replay parity.
- The release matrix pinned presentation pack v1 and `DISCARD_THEN_DRAW`.
  It now binds pack v2, navigates Learn's discard selector by keyboard without
  submitting a command while browsing, and records visual corpus v4. Original
  v3 images and their matrix are preserved together. The UR scenario retains
  its 24-command loss on turn 14. The GW scenario now loses in 80 commands on
  turn 36 because the random UR opponent has new Learn choices. Direct server
  replay and a separate browser run agree on every prompt and terminal result.
- Inspecting the candidate Linux references exposed zero-height card images
  inside Learn's preview wrappers. The images now fill their existing preview
  dimensions, matching the ordinary Card component. Browser tests assert the
  actual card treatment is visible in both Learn selectors. The empty-preview
  candidate was rejected before accepting the new references.

Repository gate guidance now requires cold launcher ordering and versioned
conformance/browser evidence when authored rules or setup changes.

## Local validation

- Full debug Rust run: 358 passed, with only the two old-corpus failures.
  After selecting v2, all five conformance tests passed; `check` reproduced
  four receipts / 560 commands, and seeded fuzzing passed 32 cases / 5,783
  commands. No engine source changed during this repair.
- Python integration: 149 passed. Launcher/verifier contracts: 39 passed.
  Semantic compiler/coverage/opcode checks: 41 passed; coverage gap generation
  check passed. The 128-timestep CI training smoke passed.
- TypeScript: zero errors or warnings. Changed-file Ruff and whitespace checks
  passed. All three built-release browser tests passed on macOS with screenshot
  comparison explicitly disabled, including the complete prompt matrix,
  accessibility checks, reconnect and both Learn demonstrations.
- Clean-machine proof passed on macOS arm64 with checkout artifacts removed
  and restored afterward: readiness in 30,022.8 ms, playable in 31,266 ms, no
  public requests after installation, same session/state after offline reload,
  and an accepted post-reload action. This is not Ubuntu CI timing evidence.

- Pinned Linux visual validation passed: all three browser tests passed during
  generation and again with strict comparison of all 17 v4 PNGs. The matrix
  contract passed too. The environment was Ubuntu 24.04 x86-64, Node 22.16.0,
  Playwright 1.61.1 and Chromium 149.0.7827.55 in a local Docker container;
  [v4 provenance](../../frontend/e2e/visual-references/v4/README.md) records the
  image digest. Visual inspection confirmed the corrected Learn previews and
  the opening, developed and terminal board states.

No claim is made here about the broader historical Python failures outside
the CI selection, complete w3 checkpoint safety, or finished Learn replay
certification. The new remote CI run remains authoritative for GitHub checks
and Ubuntu clean-machine timing.
