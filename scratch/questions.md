# Scope assumption

2026-10-16 MTG-135: Linear In Progress, LF started, PR292 attached/pushed.
Desktop cache includes it in Fantasia/Rules and satisfies working-set filters.
Direct open verified selected/open as “Profile GPU training bottlenecks”. Collapsed
Rules is possible, not observed; no assignee filter exists. Title/description
Linear sync remains uncertain. Receipt: .runs/gpu-bottleneck/desktop-inspection-current.json.

Historicalbucketaccess is resolved; authorizedreadpolicy andverifiedreceipt:
.runs/training-report-access-policy.json. Historicalinventoryremains partial.

2026-10-15 run selection: “biggest” means recorded active training duration,
not artifact bytes or planned budget. Read permissions are now resolved, but
historical inventory remains partial. See training-time-profile.md.

2026-10-15 profiler: private reservoir RNG fixes the observer's global-RNG effect.
Basic collection instrumentation is implemented. CPU parity passes; GPU003's
exact-state gate fails. Full evidence and unresolved causes are in the training
profile note and experiments/training-time.md.

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


2026-10-16 delivery: Jack Heart approved the revision-pinned PR292 walkthrough and instructed "approved, land it". `lf land` published no new code; approved head c9556a6504e4dd28dd6cd8fe1eea21c463b3e811 merged at 2026-10-16T16:59:43Z as edf6bb0d7da255427cfb0b1f8ab5f6a499ffd8f4. `lf pr reconcile` recorded the merge. Unpublished study005 edits and all local review/scratch files were isolated during landing, restored, and hash-verified; backup is .runs/pr292-landing-preserved. Approval covers the implemented defaults discussed in the review; it does not change recorded numerical results or claim the GPU discrepancies resolved. Fresh `lf task status MTG-135 --json` confirms completed=true and execution.status=done. Checkout is retained for this live Session and the restored unpublished work.

2026-10-16: Jack Heart authorized all three proposed experiment tracks. Include
small/middle/large, repeat corrected-world size evidence, distinguish experience
from elapsed-time/cost learning curves. Preserve the possibility of slow-starting
large models with higher eventual ceilings: no early size elimination and no
ceiling claim from a bounded run. Current MTG-135 status freshly shows reopened
at17:09:16Z by Jack Heart, superseding earlier completion. Work proceeds here;
no duplicate Task or new Flow/Session. Protocol: experiments/protocols/size-frontier-006.md.

Study006 implementation checkpoint bb235731aca245fe80fbe7fff0eea7f66e3d1449.
Only new experiment runner/evaluator/protocol committed; study005 local edits
remain intact. CPU smoke checks passed: training, retained update milestone,
initial/early inference sweeps and clean collection repeats, six identical
fixed-batch replay model/Adam witnesses, and two arena phases (24 games total,
zero invalid games). These are harness checks, not GPU evidence. Complete226MB
source archive uploaded and hash-readback verified. Rental quote$1.19/hour;
new$18.50 reservation plus$2.94prior conservative charges and$2reserve fits$25.
Frozen009 training cells plus independent small reference; full2048updates each,
no early elimination, and no asymptotic ceiling claim. CUBLAS workspace is fixed
at:4096:8 for both replay modes; ordinary means deterministic algorithms disabled,
not every default CUDA backend setting. This probe does not clear the old outlier.
Operational launch/watch/recovery receipts live in .runs/size-frontier.

Three-GPU admission refused its third rental before any jobs were admitted.
The first two rentals were rolled back: provider absence independently verified;
third was never created. Estimated cost$0.006253, conservative charge$0.41 retained.
Retry changes only capacity to two separate GPUs, preserving all jobs/science and
one attempt per admitted job. Budget exposure with this refusal is$23.85. Original
admission failure and all rental IDs retained in admission-refusal.json. No new
scientific run replaces a failed training run. Queue supervisor waits for the
accepted admission handle; it never retries or creates another cohort itself.

Accepted two-GPU cohort size-frontier-006 at2026-10-16T17:39:26Z.
Profile965c8138d27a; numerical replaybd16b4a8e512; independent reference36fbff1af122.
Nine scientific run IDs, exact jobs and rental handles: .runs/size-frontier/launch.json.
Live observer PID4739 under caffeinate/Doppler process group3553, with queue
progression, early diagnostic metadata retention, provider cleanup reconciliation,
complete hash-verified recovery and bounded two-worker CPU evaluation afterward.
Fresh Task read still incomplete; older merged PR292 is not completion of this
new allocation. Task automation is disabled; CI watcher active. The experiment
supervisor is an actual independent process, not a promise of another chat turn.

## 2026-10-19 — Jack Heart: final report and baseline follow-up

Jack asked for a published final report (PR #314 plus the "Batching and Model Size: Final Report" doc)
and a follow-up PR landing training improvements. Decision: switch the canonical `baseline()` recipe
from 4 to 64 streams with the tested reference-iteration schedule (64×2 unchanged), in its own PR.
