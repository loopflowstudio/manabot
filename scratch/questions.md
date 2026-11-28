# MTG-507 assumptions (headless kickoff, 2026-11-27)

Each is reversible.

- **Profile by replaying captured batches**, not by instrumenting the live
  arena; replay runs first on the L4 and sizes the arena repeats.
- The trained checkpoint has older saved readings. Seeded grid controls
  inherit them so cross-width comparisons can hold preparation constant.
  The fresh-reading `w384l8` arena checkpoint has its own capture.
- **Cross-width replay re-tags `Prepared.binding` in the harness** after
  asserting equal observation config, readings and decision memory
  (concept review, 2026-11-27). Alternative: prepare per width from raw
  observations.
- **Keep `bf16-linear-once` in the experiment.** The L4 trace showed a
  change in attention GEMMs despite identical logits on the captured batches.
  The cached-autocast diagnostic had identical cast counts to bf16 and was removed.
- **`agent_loader.py` gains `load_torch_policy` and a shared per-call
  context helper now** (review Q4a), Torch types under `TYPE_CHECKING`.
- **One L4 rental under `--max-usd 5`**, no CPU baseline arms.
  `mtg324-l4-step` was rentable at USD 0.69/hour during the local pass.
- **A fix that changes bf16 logits is not taken**: it would be a new compute
  class beside `etude-policy-bf16-auto-v1`. The report documents the
  threshold instead.
- **float32 stays the default** whatever the profile shows.
- Local MPS timings on 2026-11-27 were taken at load 30-40 and are not
  evidence; only the op counts are used.
- The MTG-497 sync keeps one shared `prepare_batch` for both arena batches and
  captured single rows. The capture hook receives the selected coordinates
  after per-seat memory ablation and prepares the singles synchronously (Q2a
  for review).
- The first L4 trace showed that even-head float32 uses native MHA. Add a
  seeded 160-wide, four-head grid control so the aligned narrow comparison
  can retain that float32 path. The planned five-head control keeps its
  separate comparison through SDPA.
- Grid cells use 12 consecutive calls per batch size after skipping 2,048
  single rows, with two timed repeats and three sampled calls for profiling.
  Each precision gets the same calls within its cell.

Ideas outside this Task:

- Evaluation loads the padded forward (`bucketed_shapes` and `live_tokens`
  off). Training measured bucketed shapes 2.4-2.6x faster on CUDA; should the
  eval loader use them?

The MTG-507 sync uses MTG-497's shared trophy pairing and 120-second game
budget for future captures. The recorded L4 runs keep their original receipts.
