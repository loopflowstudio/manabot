# MTG-214 Wave rename assumptions — 2026-10-29

- `.lf/wave/rules` → `.lf/wave/engine`: lf's `engine` Wave owns the Rules
  chapter Projects and the same Linear initiative. Added the engine Wave id
  (`74d9efbc…`) to its GOAL frontmatter, matching training's explicit id.
- Academy and eval have empty lf goals and no checked-in files; none created.
  Only training gets `config.yaml` (`workflow: iterate`), as directed.
- The top-level `wave/` tree (old `pm:` charters, numbered plans) is historical
  and cited by frozen experiments and code comments; left unrenamed, as are
  `.lf/chapters/` records and wave body prose.
- Live doc pointers moved to `.lf/wave/{engine,training}`. ARCHITECTURE now
  assigns evaluation to Eval (previously Intelligence); reversible.

# bb4160 decisions and assumptions — 2026-10-29

The accepted outcome, current evidence and ordered remaining work live in
`scratch/bb4160-evaluate-qwen-versus.md`; treatment declarations and attempt
accounting live in `experiments/protocols/qwen-mtg183-bb4160.md`.

## Accepted direction

Jack Heart accepted historical w13 for the exact MTG-183 checkpoint on
2026-10-28, preserving current-world admission and the cumulative $10 ceiling.
That evening he stopped the long sampled pilot and prioritized budget-forced
thinking, with a small separately labeled no-thinking floor (~10 games, well
under $1). Short pilots precede frozen cohorts; report reasoning by decision kind.
Concurrency 32 and MTP were accepted subject to pinned-vLLM support and hashed
argv. The mixed-decode MTP crash leaves the replacement cohort without MTP;
no vLLM upgrade is accepted. Native `skip_trivial` stays ON for all Qwen games.

Stop admissions on failure, settle admitted games, then reconcile final receipts,
all 100 scheduled IDs, inclusive charges and allocation deletion before another
commitment. The plan's 07:27 UTC snapshot is interim, not settlement or replay
qualification. `budget-64k` remains unlaunched. These constraints are unchanged.

## Reversible choices and limits

- **Settlement access gap (2026-10-29 07:27 UTC):** retained logs show 42 complete,
  3 failed and 4 pending games with 49 admissions; no final outputs exist yet.
  `RUNPOD_API_KEY` is missing locally and in the configured `loopflow/prd` and
  documented `manabot/prd` environments. Direct billing/allocation reads are
  unavailable; the last provider evidence remains 07:14. The new snapshot and
  log are artifact-published. Keep the existing commitment/reserve and defer
  both new treatments; no auth changes or infeasibility conclusion are justified.
- **No-thinking pilot (2026-10-29):** $0.80, one game/concurrency one, no MTP,
  five-minute decision deadline, original non-thinking sampling, 512 output
  tokens and 16k context. Preserve 15-minute acquisition and 45-minute
  setup/delivery reserves, leaving 20 minutes of gameplay. Reject caps without
  a complete decision window. This replaces the arbitrary 1.5-hour minimum,
  changes no running cohort and establishes neither qualification nor final
  infeasibility. Launch remains after final cohort reconciliation.
- **Service identity (2026-10-28):** derive the served name from the behavior
  hash, excluding that derived argument from base argv to avoid circularity.
  Retain complete argv separately. Acknowledgement binds the supervised
  declaration, not independently measured GPU weight bytes.
- **Process integration (2026-10-29):** bounded supervisor threads and one
  `ChildProcess` per game; resident and one-shot inference share that child owner.
  Keep the outer watchdog process group for descendant cleanup. Historical
  failed terminal selections remain evidence even when close errors prevented
  their summary counter increment. Frozen running source remains unchanged.
- **Capacity estimates:** the supplied architecture estimate was 32 KB KV/token
  over eight full-attention layers, ~50 MB linear-attention state/sequence and
  ~18 GB BF16 weights. Projected 9 GB state at ~7k tokens/sequence and
  900–1200 aggregate tokens/s were planning estimates. Measured single-stream
  ~34 tokens/s and ~5k reasoning tokens/decision support a bandwidth explanation
  without excluding runtime/cache defects. The early MTP pilot's 650–710 tokens/s
  does not establish late-game context fit or 100-game affordability.
- **Memory gap:** `.lf/wave/training/` and child memories are absent. Use the
  supplied chapter; do not substitute a Wave or create accepted memory.

## Historical references

Full original directive and checkpoint evidence:
`02fccd93e1983319d355a1866c101d7d0e01565b:scratch/bb4160-evaluate-qwen-versus.md`
and the same commit's `scratch/questions.md`. Detailed subsequent choices:
`f01fd6a4554b2880b48c77b1b657ed040851313c:scratch/questions.md`.
Obsolete MTG-207 notes: `c5d406e3d2a701525c15c8500a30f7da7896cc40:scratch/questions.md`.
Main's MTG-193 assumptions: `6d069d6e5f0755b392c796aaf6d754913e8996c2:scratch/questions.md`;
these supply historical context, not new direction or rental authority.
Main's f31c9c skip-forced-decisions assumptions: `2fc0af676671e328e7da5378cf8c2ff5ac2a913c:scratch/questions.md`.

- **Stop decision (Jack Heart, 2026-10-29 ~14:35 UTC):** after `cdb0c09f90c6`
  (Qwen 3-42 vs MTG-183; Haiku 5.5 was 51-49 on the same w13 opponent), cancel
  `budget-64k` `81769c1a29c7` (cancelled at ~$3.43 launcher estimate; pod
  absent). No further paid cohorts. Reconcile final provider charges and write up
  3-42 (with its selection bias and context-overflow failures) as the outcome.
- **Encoding caveat:** every Qwen run used the frozen `manabot-text-2` input. The
  MTG-212 encoding experiments (labelled/systematic/player/board/library variants,
  stable card block first for prefix caching) live unmerged on
  `jack/jev-encoding-w13` (`6d5b0466e`) with no recorded findings yet. Report this
  as untested for Qwen, not as a ruled-out cause.
