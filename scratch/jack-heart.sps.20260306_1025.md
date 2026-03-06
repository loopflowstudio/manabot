# SPS Closeout Design

## Goal

Close the SPS initiative cleanly.

The point is no longer to chase one more env micro-optimization. The current
state is good enough that training/inference is now the dominant bottleneck, not
 env stepping. This pass should:

- hard-wire the fastest known env implementation into active runtime/training code
- remove code that no longer serves runtime or reusable benchmarking
- keep only minimal benchmark infrastructure that is useful after this wave
- reproduce the pre-SPS quick baseline for historical comparison
- record final measurements and lessons learned in a durable report
- delete the SPS wave docs in the same pass

## Core conclusion

Env throughput is no longer the active bottleneck. Training/inference is now the
main gate.

That is the headline the final report should defend with benchmark evidence.

## What stays true from prior work

- Rayon / manual threading were investigated and did not help at current
  per-env step costs.
- The active fast path is the Rust vector env with Rust-side stepping and
  buffer writes (`step_into()` / `reset_into()` backing the Python wrapper).
- The old AsyncVectorEnv-based path is no longer the intended runtime path.

## Scope decisions

### 1. Active code path

Hard-wire the fastest implementation we know of anywhere the product/runtime
still presents multiple vector-env choices that are no longer real choices.

Active code should clearly reflect:

- Rust vector env is the current path
- legacy async/subprocess vectorization is not part of the shipping design

### 2. Old code deletion

Delete any code that is no longer used once the fastest implementation is
hard-wired.

Bias aggressive:

- remove stale comments and dead branches
- remove legacy env code if it is no longer needed after baseline reproduction
- do not preserve old abstractions just because they were useful during the wave

Exception:

- if a tiny amount of benchmark scaffolding is still needed during the transition
  to reproduce/report the baseline, keep only the minimal amount needed and then
  remove it

### 3. Benchmark infrastructure

Keep infrastructure, not wave-specific modes.

What matters long-term:

- easy reruns of current env-only throughput
- easy reruns of current training-time throughput
- human-readable output

What does **not** need to be preserved:

- sprint-specific benchmark modes
- every temporary ablation path used during SPS development
- extra configuration surface unless it is clearly useful beyond this closeout

Minimal is fine. If future work needs richer perf tooling, adapt later.

### 4. Benchmark outputs

Human-readable reports are enough for now.

No requirement to emit JSON/CSV unless it becomes obviously useful during
implementation.

### 5. Historical baseline

Use the last pre-SPS-wave commit as the historical anchor:

- baseline commit: `4c55c4e501e729c9552859cdcd29d9e57d3e4aa9`

Reproduce baseline numbers from that commit **only when quick to produce**.

Allowed:

- quick env-only measurements
- quick benchmark-script runs needed for apples-to-apples comparison

Do not block closeout on:

- long training reruns on old code
- reconvergence experiments on old code

### 6. Verification scope

Do not block this closeout on learning quality / convergence.

Training verification is still sketchy independent of perf, so the closeout
should focus on:

- code correctness for the active path
- benchmark reproducibility
- enough verification to trust the env/perf conclusions

Do not claim full training validation unless it is actually run.

### 7. Durable writeup

Do not keep the final learnings in `wave/`.

Write the durable artifact to:

- `reports/sps-closeout.md`

That report should contain:

- the baseline commit hash
- measurement scope and commands used
- current results
- historical baseline results
- what produced the gains
- what did not help (rayon/manual threading)
- why SPS work stops here
- what future signal would justify revisiting env perf work

### 8. Wave docs

Delete `wave/sps/` in this same pass after the durable report exists.

Useful content should be migrated into `reports/sps-closeout.md` first. The wave
docs are not meant to remain as the permanent home of this story.

## Recommended execution order

1. Identify and run the quick baseline measurements from pre-SPS commit
   `4c55c4e`
2. Run the corresponding current measurements
3. Write `reports/sps-closeout.md`
4. Hard-wire the active Rust path and remove dead/legacy code
5. Trim benchmark scripts down to the minimal useful reusable surface
6. Delete `wave/sps/`
7. Run final checks (`ruff`, `clippy`, targeted tests/verification)

This order matters because baseline comparison should be captured before the old
machinery disappears.

## Implementation guidance

When choosing between preserving flexibility and simplifying the codebase, bias
toward simplification.

Questions to ask during implementation:

- Is this code used by the hard-wired fast path?
- Is this code required to reproduce or explain the final benchmark story?
- Will anyone reasonably rerun this benchmark path after SPS closeout?

If the answer is no, delete it.

## Done when

- active runtime/training code presents one clear fast env path
- legacy env code that no longer earns its keep is removed
- benchmark tooling remains minimal but usable
- pre-SPS quick baseline is reproduced and recorded
- current measurements are recorded alongside it
- `reports/sps-closeout.md` exists and carries the durable conclusions
- `wave/sps/` is deleted
- the final story is clear: env throughput is no longer the bottleneck
