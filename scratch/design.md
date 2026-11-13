# PPO CUDA update throughput — MTG-338

## Decision: iterate

Try a CUDA training embedding lookup in
`SemanticCardEncoder.embed_definitions`. This is the first runtime candidate;
the sole row in `scratch/measure.html` is a successful baseline measurement,
not a rejected optimization. The candidate is implemented; numerical admission
and paired throughput measurement remain before accepting it.

## Hypothesis

Repeated fact-source and continuation IDs send large gradient tensors into a
small definition table through generic advanced-index backward. Replace that
one shared lookup with `torch.nn.functional.embedding` on CUDA when gradients
are enabled and the table requires gradients. Its grouped reduction should
make those repeated-ID gradients cheaper. Predict **at least 20% less median
update time**, from 60.521 s to **48.417 s or less**, equivalently **at least
0.33046 optimizer steps/s** for the same 16 optimizer steps, a 25% rate gain.

The clean baseline is Etude `4a289dc30a82` at `2202ad632cb5`, with five samples
at 0.26366–0.26474 steps/s. The separate profile `e9e5a6bd4649` attributes
1.126 s to fact sources (`facts.py:193`) and 0.542 s to continuation definitions
(`backends/torch/policy.py:722–724`), both through
`semantic_cards.py:410–414`. Together these are 1.668 s, 56.3% of the 2.964 s
recorded GPU kernel duration. The table is 463 × 64; incoming gradients are
6668 × 256 × 64 and 6668 × 64 × 2 × 64. Halving these reductions gives a
credible path to the prediction, but the perturbed first-step profile is not
an end-to-end timing model. Measure the full update to decide.

PyTorch 2.15's large-index embedding backward sorts IDs before reducing their
gradients, so this uses an existing specialized implementation. See the pinned
[CUDA dispatch](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/Embedding.cu)
and [grouped reduction](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu).
Sorting and temporary storage cost time and memory; the reduction can still
use atomics. Neither determinism nor a speedup is established by that source.

## Implementation boundary

- Compute admitted catalog rows with the existing `self._rows(ids)` once.
  In the CUDA autograd case return `F.embedding(rows, definitions)`; otherwise
  keep `definitions[rows]`. Import the functional API at module scope.
- Use `padding_idx=None`, `max_norm=None`, `scale_grad_by_freq=False` and
  `sparse=False` (the defaults). Keep row-zero gradients and repeated-ID sums;
  do not normalize by occurrence count. These options follow the
  [functional embedding contract](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.embedding.html).
- Pass the existing differentiable table directly. It includes encoded card
  programs and public-name embeddings; do not detach it, wrap it in a new
  parameter or cache it across optimizer steps. Preserve shape, dtype, device,
  admitted row mapping and all upstream gradient paths.
- CPU, MPS and inference keep the original indexing operation. Preserve empty
  lookup behavior and connected zero gradients. No new configuration,
  checkpoint fields, world changes or dependencies are needed.
- This first candidate covers every existing `embed_definitions` caller,
  including action/shared-effect sources, but does not rewrite the separate
  visible-card, permanent, known-hand or decklist gathers. Expanding to those
  joins is a later candidate if this mechanism succeeds.

## Delete — do not maintain

Replace generic advanced indexing only for CUDA autograd in
`SemanticCardEncoder.embed_definitions`. No files, schemas or fixtures retire.
Preserve admitted row mapping, row-zero and repeated-ID gradient sums, upstream
program gradients, and the CPU/MPS/inference indexing path. Other definition
joins remain out of this candidate's scope.

## Remaining verification and measurement

Focused check: `uv run --locked --extra dev pytest tests/model/test_semantic_cards.py -k definition_lookup -q`
— 8 passed, 8 CUDA cases skipped, 2 deselected (51.70 s); Ruff passed.

1. Focused lookup tests are implemented in `tests/model/test_semantic_cards.py`:
   repeated IDs, zero IDs, empty indices, multidimensional/noncontiguous indices,
   scalar IDs and a non-leaf definition table with two producers. Dyadic nonzero
   upstream weights allow exact float64-reference gradient checks without masking
   dropped gradients behind a tolerance. Run the CUDA cases on the L4 with
   `MANABOT_REQUIRE_CUDA=1 uv run --locked --extra dev pytest tests/model/test_semantic_cards.py -k definition_lookup -q`;
   local skips do not establish CUDA correctness.
2. Recover MTG-325's `scratch/equivalence/{run,updates,compare}.py` from
   `c958ae2f1^`, adapting import paths to the current backend and using `uv run`.
   Compare current baseline and candidate on CPU bitwise for policy outputs,
   gradients and trained weights. Ensure the comparison fails on differences;
   the historical comparison script only prints them. Run relevant model tests
   and `tests/training/test_active_training.py` per `docs/development.md`.
   Keep game/training checks on the remote allocation and local fixtures small.
3. On CUDA, require exact lookup outputs at identical input weights. Compare
   repeated-ID gradient sums with a small float64 reference, including row 0;
   record absolute and relative errors. For identical real batches and initial
   weights, retain per-parameter gradient and post-update weight differences,
   loss and policy differences alongside same-arm repeat drift. Explicitly
   bound and explain the changed summation order in the report/PR. The observed
   baseline weight drift of 2.486e-5 is not an acceptance tolerance. Unexplained
   gradient loss, changed support/counts or nonfinite values rejects the change.
4. Adapt `experiments/ppo_cuda.py` and the existing speed harness for alternating
   A/B arms on one L4, one locked build and the same captured batch and initial
   state. Use five timed samples per arm after equal warmups, fresh Adam and
   batch copies per update, identical selection/shuffle seeds, production
   threads/cuDNN settings, TF32, bucketed shapes and live tokens. Keep 53,341
   learner rows, 26,671 retained rows, 106,684 exposures and 16 steps/update.
   Count actual steps and retain zero rejected/skipped steps. Profile separately;
   report medians, ranges, section times and peak allocated CUDA memory.
5. Use the existing launcher cap **`--max-usd 4`** for the comparison, with no
   automatic retries. Publish evidence through the artifact store and terminate
   the rental. Update the existing report and measurement table. A paired
   median gain that does not clear the measured spread is inconclusive; a
   candidate above 48.417 s misses this prediction even if it helps somewhat.
   Keep only a gain above spread with the numerical evidence accounted for.

## Alternatives deferred

Stop/reuse/defer options cannot remove these live gradients without changing
learning: definitions depend on current weights and optimizer steps invalidate
cached training embeddings. Fact compaction may eliminate padding but introduces
shape/packing changes; live-token and bucketed execution already exist, so
measure occupancy before redesigning them. Smaller minibatches change the
optimizer schedule. Attention contributes only 0.044 s in the profile and Adam
only 0.029 s/update in clean timing. Synchronization and clipping are not
separately established as the limiter. A custom grouped-reduction kernel adds
maintenance before the built-in embedding path has been tried.

These are priority decisions, not measured rejections. The best remaining idea
after this candidate is the same primitive at the other measured definition
joins if it wins; if sorting costs erase the gain, inspect live/zero-ID occupancy
and consider compacting unused source lookups before writing a custom kernel.
