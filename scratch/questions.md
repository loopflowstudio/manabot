# Scope assumption

2026-10-15 explicit scope refinement from Jack Heart: first deliver the
training-only profile/timeline; expand later to parallel evaluation work at the
regime level. Do not require evaluation instrumentation for the initial report.

2026-10-15 latest steering: interpret the requested high-level flamegraph as both
an aggregate phase hierarchy and an experiment timeline, since evaluations can
overlap training. First artifact uses one actual GPU run without pretending its
three-update/no-evaluation recipe represents a mature campaign. A benchmark map
must report target-specific predictive validity; none is assumed from resemblance.
Etude client/Studio are inspected read-only; no cross-repository implementation or
new paid cohort is selected by this research artifact.

2026-10-15: Jack Heart selected inference serving rather than managym simulation. Run018 shows the model forward is a minority of ordinary large-stream collection, with substantial host input/materialization and native stepping/encoding. Interpret the next optimization as improving manabot's data flow feeding one GPU; do not silently turn this into a rules-engine optimization. A parallel production pipeline still needs a concrete design and end-to-end correctness/performance validation. Existing benchmark workers do not satisfy that contract.

2026-10-15: Artifact/source inspection maps019Adam differences to semantic_cards.token_embedding.weight, semantic_cards.encoder.weight_ih_l0/weight_hh_l0 and attention.mha.in_proj_weight (parameter IDs0/1/2/22). The4096difference is attention projection only. A GRU-only cause is not established; attention/backward reduction and other CUDA numerical-order mechanisms remain live alternatives. Experiment._setup_random sets cuDNN deterministic but does not enable torch.use_deterministic_algorithms. Diagnostic flags change several controls together, so their successful512proofs do not identify a single cause. A fixed-input repeated-gradient/backend comparison is the separating experiment if this uncertainty remains decision-bearing after020. No gate relaxed.
The actual recipe uses ataraxos_move, routed through training.objectives.update_iteration -> update_move_iteration -> OptimizerHealth.step; stock Trainer._optimize_step is not the measured loss path. A prospective diagnostic can intercept one real OptimizerHealth.step and repeat backward(retain_graph=True) on the identical saved loss graph with unchanged parameters, clearing gradients each time, before letting the original step finish. This separates backward variability from rollout/storage/Adam. Backend/configuration arms must be declared before measurement; no experiment is authorized by this note alone beyond the active goal/budget. No probe implemented while020's source is frozen.

2026-10-15:020also differs at IDs9/11, mapped from its saved recipe to
card_embedding.projection.0.weight/[64,39] and projection.2.weight/[64,64].
Agent._add_focus uses differentiable torch.gather on expanded post-attention
objects. This is another candidate backward boundary, not a proven cause or
proof that repeated valid indexes occur in the measured batch.
[PyTorch2.10 deterministic algorithms](https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html)
lists CUDA gather gradients among affected operations. Its
[reproducibility guidance](https://docs.pytorch.org/docs/2.10/notes/randomness.html)
distinguishes cuDNN-only controls from broader deterministic mode; ordinary
Experiment._setup_random sets only the former. The
[SDPA documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
also describes automatic backend selection and numerical differences between
implementations. No actual backend is identified without a training trace.
Fixed-graph repetitions must keep inputs/weights/RNG fixed within each arm;
backend/configuration changes may require separately constructed forward graphs.
No causal claim or acceptance tolerance changes follow from these API contracts.
