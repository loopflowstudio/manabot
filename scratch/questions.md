# MTG-333: assumptions taken (reversible) and open questions

## Upstream card assumptions and source facts

2026-11-12: Use `token.plain_spirit` for Release to Memory's unrestricted
Spirit (concept review replaced `token.colorless_spirit`: TLA's Spirit is
colorless too), keeping TLA's `token.spirit` unchanged. Use `token.zeppelin` for Lita's
unprinted Vehicle. Both are reversible internal naming choices.

Scryfall API verified the first-printing card ids from the pinned inventory:
- https://api.scryfall.com/cards/b45c4bbf-2c61-4d49-96c8-34a0fb0d6d02
- https://api.scryfall.com/cards/f3a5a044-d474-43f4-95a9-e67808908e6c
The earliest 1/1 colorless Spirit token from the ordered printings search is
P04 (2004-01-01), id 46b60d95-b9bc-40f8-b986-bfa8e3eb74f3, empty Oracle text.
Lita's printed rules say Zeppelin is a 5/5 colorless artifact Vehicle with
flying and crew 3; it becomes a creature only when animated.

2026-11-12: The concept-review plan called existing flicker text "Return it
to the battlefield…". Inspection of main's `move_description` shows "Put it
onto the battlefield under its owner's control" for `Moved`. Preserve that
actual wording in the regression assertion, consistent with the instruction
not to reword existing cards; do not change the move verb.

2026-11-12: Use the descriptive `w17-token-cards` fixture suffix for the new
Academy catalog binding. Preserve the existing `w17` files unchanged; replay
only existing Commands and require their offer ids/labels to match.

2026-11-13: Main now includes MTG-297 (Heartbeat of Spring). Use the complete
61/61 bonus sheet and 1,000-deck admission as the combined acceptance outcome;
do not keep a known obsolete "only Heartbeat missing" expectation. Seed new
`w17-token-cards` fixtures from main's current w17 Commands before refreshing
the combined catalog, because main's mana spending redesign changed Commands.
The managed `lf sync` plan selected merge_target for this branch; preserve
history through that supported integration rather than raw git rebase.


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
