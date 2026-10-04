# Allies versus Lessons arena instrument — 2026-09-29

ETU-83, requested by Jack Heart. This implements the evaluation slice of
[the accepted training plan](../plans/strongest-manabot-training.md). It makes
no strength, training improvement, promotion or w3 certification claim.

## Result

At evaluator checkpoint `36e7abc`, the existing arena completed a four-game
block between pinned Random and the shipped demo Search-64 on the authored
Allies/Lessons setup, including Lesson sideboards. Both players received both
decks and both starting seats. All **546 Commands replayed exactly**: zero
frame, offer, Command, actor, state, outcome or trace mismatches, zero private
hand exposures and zero failed games. The source-stability check passed.

[Frozen final manifest](../../experiments/data/etu83-allies-lessons-instrument-20260929/final/manifest.json):
`53adbac03bfd07d06263b7b4ccc37fa8646b7c31977fa39a3fed231feee20a30`.
The directory contains the pre-play protocol, player registrations, all four
match rows, compressed Command tape, replay receipt, statistics and chained
resource ledger. After copying it from the local run, every artifact hash,
the manifest digest and ledger chain were verified, and policy-free replay
reproduced the retained receipt again.

| Leg | Starting-seat deck | Random seat | Commands | Worker wall seconds |
| --- | --- | --- | ---: | ---: |
| 0 | UR Lessons | 0 | 112 | 8.82 |
| 1 | UR Lessons | 1 | 62 | 5.55 |
| 2 | GW Allies | 0 | 212 | 22.13 |
| 3 | GW Allies | 1 | 160 | 9.07 |

Deal seed: `83001`. Limits fixed before play: 120 seconds and 10,000 Commands
per game, one CPU worker and one Torch thread. Worker time totaled 45.57 seconds,
including process startup; this excludes preflight and final replay. Focused
tests ran concurrently, so these are incurred local costs, not a controlled
latency benchmark. No training, cloud compute or provisioning ran. New paid
compute spend: $0; electricity was not measured.

## Baseline identity

The supplied checkout had no configured trained bundle. The baseline is the
shipped default from `etude.server._parse_game_config`, calling the actual
`SearchVillain` implementation through the arena factory. Historical seed-79
trained bytes from the [prior evidence](trained-challenger-2026-09-25.md) are
not represented as compatible with this runtime.

- Player: `demo-search-64-v1`; registration SHA-256
  `8c2af43e434dc0a60af79b81386c363318aaa105f9a07a4bcae75f8d0e68dc31`.
- Policy/adapter source SHA-256:
  `805daaf7db4afacf92af8402d5eb6cbaa958328ddc14823d3240d7e06ac1ff2f`.
- Native extension SHA-256:
  `0f41ceca339a1d54c0ca7e6eba07b5ec6a32793d038762b723fc67e987f733f2`.
- Complete authored matchup SHA-256:
  `5b46db09a613880980ac8fbd5d0942fe24d5465ff0789984cc75709adf7be343`.
- Evaluator source bundle SHA-256:
  `6bdae3b441eecc2805e8db8bdf79cc51aad59b6e7874f586df2f7724b5aa48d3`.

Inference: 64 simulations per legal action, four rollouts per world, 2,000-step
playout cap, deterministic first maximum, CPU, one Torch thread. The protocol
pins the remaining rules/content/observation/action identities. These identities
together define the baseline; the player registration alone is not the native
runtime. The extension was built locally from this checkout, not borrowed from
another worktree.

## Failure and compatibility checks

The focused command passed **17 tests in 22.40 seconds**:

```bash
uv run pytest tests/arena/test_match.py tests/arena/test_models.py tests/arena/test_rating.py -q
```

Coverage includes the four deck/seat assignments, sideboard-sensitive replay,
historical frozen registration digests, checkpoint byte rejection, larger
checkpoint observation bounds, terminal draw scoring, process death, timeout,
policy exceptions and retained Command-cap prefixes. Injected failure checks
are tests, not additional live strength observations. Player-attributable
crashes forfeit; total-game timeouts and unattributed failures invalidate the
cohort's statistics without dropping attempts. Failed-prefix replay never
establishes successful completion. Changed-file Ruff lint/format and whitespace
checks passed. No Rust source changed; the broad gate remains with the caller.

The ordinary checkpoint loader remains authoritative. The untrained checkpoint
test uses 128 action rows, 96 card rows and 64 permanent rows and proves their
retention through arena replay; it is not a trained challenger result.

## Attempt accounting and remaining boundary

Two explicit demo-baseline development executions ran. The first, retained in
[development/](../../experiments/data/etu83-allies-lessons-instrument-20260929/development/manifest.json),
completed four games / 588 Commands with no replay mismatch (67.20 worker wall
seconds). Evaluator edits overlapped that run; its source digest is not a claim
that every worker used those exact bytes. It remains development evidence.
The final execution above holds evaluator sources fixed and is the acceptance
proof. No attempts were silently discarded or replaced.

The `w3` arena identity distinguishes the current corrected Learn setup from
frozen w2 mirror evidence; it does not complete ETU-75's outstanding policy-input,
checkpoint-binding or world-certification work. The comparison Task still needs
that gate, independent trained seeds, held-out paired blocks, declared matched
compute and a protocol fixed before scoring. A one-block bootstrap is plumbing
evidence, not useful statistical uncertainty. The existing historical contracts,
checkpoints and receipts were not regenerated.

The [operator workflow](../local-trained-challenger.md#evaluate-the-selected-matchup)
owns repeatable commands and candidate registration requirements. No external
publication, Task closure or strength claim was performed in this implementation
step.

## Gate review — 2026-09-29

The gate reviewed the implementation against base `48910ab4` at HEAD
`f2403ee`; no additional code changes were needed. The affected match, model,
rating and runner suites produced **21 passes and two failures in 22.08 s**.
Shared profile, guidance, competency and Teacher-1 evidence checks added
**12 passes in 6.02 s**. Changed-file Ruff lint/format and branch whitespace
checks passed.

The two runner failures are existing frozen-evidence incompatibilities:
`test_int8_preflight_adds_current_identity_without_mutating_int6` rejects
`uniform-prior-puct-32-v1` source drift, and
`test_fixture_checkpoint_loads_replays_profiles_and_runs_competencies` rejects
the historical observation ABI. Both also fail with the base revision's runner
loaded in memory against current dependencies, with the same rejection messages.
This is a runner comparison, not a full base-checkout test run. Neither frozen
identity nor admission guard was changed to make these tests pass.

The earlier compression regression run remains locally at
`.runs/etu83-compress-1` (not included in the portable evidence directory): four
complete games, 403 Commands, zero failures or replay mismatches, manifest
`8282c33978708fc68cae71342d40daa066febd217ec508500192bb433271d977`.
Its evaluator source bundle matches the gate's current source bytes. Its
source-derived arena key changes policy seeds, so this is regression proof,
not a replacement for the frozen acceptance block. This is the third explicit
demo-baseline execution, following the two implementation attempts above.

Gate verification authenticated all three runs' manifest/artifact hashes and
ledger chains, then independently replayed all **1,537 Commands** with receipts
identical to the originals. No new evaluation cohort or training ran during
gate. Existing exact-source execution proof was reused; unit checks ran fresh.
Rust, frontend and the full Python matrix were left to CI because this branch
changes only the Python arena and its evidence. The two runner failures remain
an explicit limit on a fully green local suite; the w3 certification and scored
comparison prerequisites above remain open.
