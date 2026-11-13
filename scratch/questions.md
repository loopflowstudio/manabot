# MTG-333: assumptions taken (reversible) and open questions

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

- **L4 hourly quote (2026-11-13).** Shared `l4` refuses the current $0.69/hour
  quote under its $0.60 ceiling. Use a Task-specific one-machine template at
  $0.75/hour, preserving the $3 equivalence/$10 speed cohort caps and $25 total.
  Leave the shared rental configuration unchanged.

- **Card when L4 is out of stock (kickoff, 2026-11-13 12:20 UTC).** The brief
  names L4; RunPod and Vast had none and RunPod's list omitted it. If L4 is
  not rentable at launch, equivalence, speed and optimize run on one RunPod
  A40 (`mtg333-a40`, $0.59/hour quoted, same ceiling and cohort caps). Base
  and head run on the same machine, so the comparison holds; MTG-325's L4
  figures become a reference. Rerunning on L4 later is one capped launch.

- **Upstream engine change before the CUDA launch (kickoff, 2026-11-13 12:27
  UTC).** `105e7e2aa` changes seeded games and adds pooled-mana program rows.
  Merge it first and take the base arm from `105e7e2aa`'s `manabot/` rather
  than `ef28f3a58`, so both arms share one engine and one observation
  encoder; the oracle gains a synthetic pooled-mana case. The rented runs are
  not repeated for later engine-only merges.

- **Retry, 2026-11-13.** L4 became rentable at 16:00 UTC under the existing
  Task ceiling, so retain the original L4 comparison rather than the A40
  fallback. No change to cohort or total spend caps.
