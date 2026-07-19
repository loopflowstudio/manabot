# INT-17: first belief-calibration curves

**World:** w2  
**Evidence class:** one-seed selected-trace calibration only  
**Status:** fail-closed systems result; calibration remains `evidence_wait`  
**Contract:**
[`contracts/int-17-belief-calibration-v1.json`](contracts/int-17-belief-calibration-v1.json)

## Question

Across the retained seed-0 UR-Lessons-versus-GW-Allies decision sequence, does
the frozen INT-7 action-likelihood policy assign more mass to the actual hidden
opponent hand than the contemporaneous compatible-deal prior does?

## Prediction

For each fixed viewer, over opponent public-commitment points:

- mean `prior_log_loss_nats - posterior_log_loss_nats` is greater than zero;
- posterior true-hand mass exceeds prior true-hand mass on more than 55% of
  points.

This checkpoint was trained on the interactive mirror rather than the selected
curated matchup. Refutation is therefore a first-class result and will be
retained without choosing another checkpoint.

## Frozen inputs

- Authority trace: seed 0, 132 Commands, SHA-256 `57f5e2d1...`.
- RUL-11 provider receipt: 62 provider commitments, zero exercised gaps,
  SHA-256 `45227941...`.
- Likelihood checkpoint: INT-7 `visit_policy_only` seed 197, SHA-256
  `1673a237...`.
- Tracker: exact compatible-deal support, `epsilon=0.05`, counterfactual seed
  907, batch size 256, CPU.
- Cohort: both viewers, one initial point plus every transition, 266 total
  points.

Truth is read only after each tracker update by the authority audit. It is not
an input to the Observation, likelihood model, tracker, or command tape.

## Cost cap

The checked preflight contains 903,063 counterfactual world rows: 694,187 for
viewer 0 and 208,876 for viewer 1. Generation is limited to one worker,
1,000,000 rows, six wall/core hours, 2 GiB peak RSS, and 64 MiB of retained
bytes. Crossing any cap yields no calibration result.

## Frozen attempt command

```bash
uv run --extra dev python experiments/runners/run_belief_calibration.py \
  --contract experiments/contracts/int-17-belief-calibration-v1.json \
  --out-dir .runs/int-17-belief-calibration-v1
```

This command identifies the retained v1 attempt. It is not the post-RUL-13
rerun command: v1 byte-locks the generation-time engine source, extension, and
INT-17 source identities, so a real provider repair must make v1 fail closed.

## Interpretation boundary

The points are serially correlated observations from one fixed trace. The
result can describe this checkpoint's selected-trace belief updates; it cannot
establish general calibration, a method effect, or gameplay strength.

R4 will join this result to the exact-range arena by world, matchup,
checkpoint, epsilon, and player identity. Better calibration without an arena
gain points to planner use or matched-cost effects; worse calibration alongside
an arena loss points first to the likelihood model. Crossed results remain
ambiguous and do not support a causal strength claim.

## Result

The exact frozen command ran on 2026-07-18 without changing the checkpoint,
cohort, likelihood, or thresholds. It was stopped cleanly after **1:37:56
wall / 84:38 CPU** because completed runtime receipts plus the static provider
path proved that the six-hour cap could not be met. It emitted no curves and
left no partial scientific result.

The retained failure receipt is
[`data/int-17-belief-calibration-v1/sha256/78bde491e16957b743a59cebe6f87fd519dc982793d5d6f7dbb649a98d57e027/manifest.json`](data/int-17-belief-calibration-v1/sha256/78bde491e16957b743a59cebe6f87fd519dc982793d5d6f7dbb649a98d57e027/manifest.json).
The process completed three public-commitment updates:

| ordinal | commitment | support | incremental wall seconds |
|---:|---|---:|---:|
| 0 | play `Island` | 41,806 | 2,234.918 |
| 1 | pass priority | 17,711 | 470.515 |
| 2 | cast `Tiger-Seal` | 17,711 | 253.200 |

It was evaluating ordinal 3, play `Forest`, over 121,485 worlds when stopped.
Stdout contains exactly those three receipts (455 bytes, SHA-256
`b7c0c468...`); stderr is empty (SHA-256 `e3b0c442...`). The largest sampled
RSS was 2,106,195,968 bytes, still below the frozen 2 GiB cap. The runner's
wall and RSS checks occur only after an entire decision ordinal, so they cannot
bound one pathological likelihood call.

## Systems finding

The production materialization path accidentally makes one likelihood update
quadratic in the compatible support size:

1. `FrozenPolicyLikelihood.evaluate` visits every row in the belief support
   and calls `root_space.materialize(row)`.
2. The Python `PossibleWorldSpace` forwards every row to
   `Env.materialize_possible_world`.
3. The Rust provider reconstructs `PossibleWorldSpace::for_viewer` to validate
   identity before each `materialize_index` call.
4. `for_viewer` / `from_parts` recursively enumerates the complete support.

At the frozen maximum support of 121,485, one update therefore performs
14,758,605,225 support-row enumeration operations before policy inference.
Across all 62 frozen updates, the exact support schedule has
`sum(S^2) = 51,506,080,901`. Coefficients from the three completed updates
project **41,575 seconds (11.55 hours)** even under the most optimistic
observed rate, **65,863 seconds (18.30 hours)** at the median, and 77,258
seconds at the slowest. Every projection exceeds the 21,600-second cap, so
continuing would have converted a proven systems failure into an unbounded
run.

## Prerequisite and disposition

Rules/managym must expose an identity-bound materializer that constructs and
validates one canonical `PossibleWorldSpace` once, then materializes many
canonical indices from that retained space without re-enumerating it per row.
The handle must remain bound to viewer, source revision, viewer-state hash,
space identity, seed, and materialization mode, return isolated branches with
the existing `materialize_index` semantics, and fail closed on drift. The
Intelligence consumer must also check wall/RSS caps between inference batches.

After that provider lands, preregister
`experiments/contracts/int-17-belief-calibration-v2.json` before generation.
The new execution contract must copy v1's trace, likelihood checkpoint,
algorithm, cohort, preflight, metrics, prediction, caps, arena interpretation,
and exclusions exactly, while binding the repaired provider receipt and the
new runner, engine-source, and extension identities. Its provider identity
stream must remain `8d9f46fa...` with zero exercised gaps. A focused contract
validator must reject any scientific-section drift before generation. The
prerequisite binds both the unchanged v1 file bytes (`221ef497...`) retained by
the failure receipt and its canonical JSON payload (`5b764e76...`), so neither
whitespace-level replacement nor semantic drift can masquerade as continuity.

Only that new, committed execution contract can rerun the unchanged scientific
experiment after RUL-13; the frozen v1 contract and failure receipt remain
historical evidence and are never rewritten. Until then R3 remains open: this
is honest retained systems evidence, not a negative belief-calibration result.

After publication rebased the branch onto current main, the INT-17-focused
suite still passed 27/27 and the exact preflight remained unchanged. The
broader RUL-11 receipt verifier reported source-provenance drift only: upstream
`tests/sim/test_conditional_search.py` changed, so that file hash and its
aggregate relevant-source hash differ from the checked receipt. The generated
and checked receipts have no functional or identity-stream difference. This
closure does not rewrite the frozen RUL-11 receipt to bless unrelated source
drift.

## Arena interpretation

INT-18's frozen arena rated flat-MC-64 at 1513, flat-MC-16 at 1369, dPUCT-32
at 1333, flat-MC-4 at 1321, scripted at 1203, and random at 1000 over 720
games. The exact-range player remained `evidence_wait`, so no belief-versus-
uniform strength comparison exists. This failed calibration run neither
explains nor changes those ratings and does not retroactively admit the
unselected INT-7 checkpoint. Once both results exist under joined identities,
better calibration without arena gain would point to planner use or matched-
cost effects; worse calibration with arena loss would point first to the
likelihood model; crossed outcomes would remain ambiguous.
