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
