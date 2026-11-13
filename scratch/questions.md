# MTG-333: assumptions taken (reversible) and open questions

Upstream card/token source facts and decisions remain at
`9fe9eb1e7:scratch/questions.md`; that work is delivered on main.

## MTG-333 assumptions


Kickoff, 2026-11-12. Nobody was asked; each is the simpler safe option.

- **Policy file suffix.** New policy files written by `save_checkpoint` are
  named `.safetensors`; old `.pt` files keep their names and load;
  `load_checkpoint` chooses by bytes, and suffix-based finders accept both.
  The alternative (container bytes under the existing `.pt` names) renames
  nothing in stage outputs, experiment name tables or Etude artifacts.
  Optimizer, update-snapshot and incident files stay torch `.pt`.
- **Torch encoders move** from `manabot/model/` to private modules under
  `manabot/backends/torch/`; no aliases (MTG-325 steer 8332334d).
- **Finished profiling experiments** (`cuda_precision`, `l4_step_profile`,
  `collection_tail_graphs`, `eval_forward_layout`, `tla_frontier_facts`,
  parts of `mlx_pilot`) patch internals this Task deletes. Port mechanical
  call sites; let the rest fail at call time naming the commit that ran them.
- **One PR**, as the brief says, at an estimated 6-8k changed code lines.
  MTG-325's #501 was split at Jack Heart's direction when it passed review
  size (most of that was tracked scratch); slices 4 (container), 5 (backend
  rule) and 6 (aliases) are separable if a person asks for a split.
- **Bitwise on CPU** is kept as the bar although summation-order differences
  are acceptable in general: `Prepared` carries raw known-hand counts and
  signed program integers so the torch arithmetic is unchanged; the MLX
  forward changes by rounding only and is held to its parity tolerances.
- **Failure evidence** from a rejected step becomes the prepared batch's
  arrays by field name (as MLX), with the rollout row indexes already
  recorded beside them, instead of the raw observation dictionary.
- **L4 caps**: equivalence cohort USD 3, speed cohort USD 10 (MTG-325's
  figures), all L4 spend for the Task under USD 25.

Compress, 2026-11-13: simplify the implemented host/MLX slice; the remaining
Torch cutover is implementation work, not a reason to refactor its deletion
targets. Original uncommitted notes were copied to
`/tmp/mtg333-compress-before/` before curation.

- **Container parameter order.** Safetensors sorts tensor names, but learning-state
  imports check Adam's positional order. Record `parameter_names` in metadata
  from `named_parameters`; legacy imports retain their state-dictionary order.
  This preserves the existing refusal rather than inferring order from sorted keys.

- **Inference lifetime.** `Prepared` carries an optional ephemeral guard callback
  set by `PolicyInference`; it preserves closed/changed-policy and autograd
  refusals without a second batch wrapper. Bucket needs remain host tuples on
  the prepared batch for the collector, rather than copied device arrays.

- **Packed tie order.** The shared host builder accepts a length-order function;
  Torch uses its CPU sort for the existing packed-GRU tie order. This keeps
  integer work in the one host preparation without changing accumulation order.
  The 3,248-tensor oracle is exact after catalog and fact packing moved.

- **Child entry points.** A game worker and serving worker are composition roots,
  like a CLI: they select the backend inside the child, then inject it into
  library code. Game selection remains lazy so code players and replay do not
  import Torch. The boundary guard lists these roots explicitly.
- **Executor follow-up boundary.** The executor's coupled collection, minimal
  loop, numerical-health and experiment-launch helpers remain with its deferred
  backend selection. sim/eval forward, sampling and checkpoint consumers move
  here. Selection reporting uses the existing shared NumPy helpers.

- **Card (2026-11-13).** Resolved by comments 73b0776e and 688982cb, quoted in
  the plan's "What people said": any rentable CUDA card, with both arms on one
  card in one rental. The earlier L4/A40/$0.75-template notes are at
  `405ebb85f:scratch/questions.md`.

- **Sync, 2026-11-13.** Retain main’s MTG-313 QA report and source facts.
  Combine its local supervised witness (SGD 0.01, no greedy-convergence
  loop) with this branch’s `device_input` route over `Prepared`.

2026-11-13 #513 integration: reuse `Prepared.is_agent` for MLX's signed
ownership instead of adding a second ownership array. It already follows
bucket-trimmed coordinates and preserves explicit belief ownership. Port
shared capture hooks to NumPy inputs and retain raw `ActionRows.observation`
for cross-backend replay; Prepared evidence is intentionally not raw input.

2026-11-13 19:33–19:38 UTC: A40 optimization admission f6f8c567e281
was refused by the capacity quote. A6000 bcc8fca2fcb9 and A40 retry
49a888481def returned HTTP 400; both bodies said “There are no longer any
instances available with the requested specifications. Please refresh and try
again.” Provider inventory confirmed no pods; their windows were terminated.
Use the authorized Vast 5090 fallback: run 267f945d54a0, one card/rental for
both optimization arms. Absolute times are not compared across cards with
the initial A40 result.

2026-11-13 post-#514: the final host-metadata check uses one process per arm
with three timed inner repeats to fit the existing same-card rental window.
The preceding main/pre-optimization/fact-optimized check uses three processes
per arm. Keep the final inner ranges; do not call them process spread.

2026-11-13 PPO optimization: choose bounded one-minibatch host prefetch over
cross-epoch caching: trimming, selected catalog programs and packed GRU order
are batch-dependent. Keep transfers/model/optimizer on the caller; propagate
caller grad mode to the worker and join it at epoch exit, including failures.
Use current Etude 0.3.2 with a new schema-specific launcher database; register
the existing rental template through CLI, leaving the older database untouched.
