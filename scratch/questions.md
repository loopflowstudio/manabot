# b015d5 — remaining decisions and evidence

Historical step-1/2/3 design, implementation, reviews and execution notes are at
`c2c813a475ef012bd0391e510dd7f0cb41859e83:scratch/questions.md`. Original reports,
run projections, bills and cleanup receipts remain published in
`experiments/data/training-throughput-b015d5-v1/`. Those sources retain exact
run IDs and dated direction; no historical result is being rewritten.

## Current authority and boundaries

Jack Heart replied "keep going" to the recommendation "I recommend repairing
BF16, lengthening the packed cells, and completing the missing measurements."
Comment 079b3f46 authorizes the bounded continuation. Comment 8cb4c5af records
C1 review against that authority, not a new approval. The owning protocol is
`experiments/protocols/training-throughput-b015d5-v1.md`, C1; the implementation
handoff and deletion list are in `scratch/step-3-continuation.md`.

Keep effective hashed v6 / 512 facts and the Elemental learning recipe. c5fba4
owns its shared-effect catalog repair; do not infer TLA catalog speed from the
hashed ladder. Exact quotes and external diagnostic findings remain in
`scratch/vocabulary-performance.md`.

The first pass had 11 valid solo cells, seven insufficient-overlap packed cells,
six BF16 startup failures and no a384l8 measurements. Original Float/Half error:
`index_copy_(): self and source expected to have the same dtype, but got (self)
Float and (source) Half`. CPU autocast had not covered CUDA's recurrent behavior.
The repair is delivered here for unstarted ed2300, with no second checkout.

## Execution still open

- Source/world/input freeze after software integration. Original source was
  cb722dc8c / w17; C1 never relabels those results. Current runtime admission
  remains a freeze check. Real CUDA regression and a two-update w160l8 BF16
  worker/reload on one rental must pass before opening other rentals.
- Reconcile bills before allocating new liability. Step 3 $1.741 and total
  $3.781 remain estimates, not invoices. Original caps stay $10 step 3 / $50
  total; C1 adds at most min($10-prior step3, $50-prior total, $8), 15 GPU hours,
  1,600 updates inclusive of failures/setup/diagnostics. No paid work in this step.
- Same isolated client/home as the protocol. Original provider DELETE/GET 429s
  required explicit provider deletion confirmation; all three original pods
  were absent at 13:19 UTC on 2026-11-08. No controller was left running.
- Original TF32 clip-fraction shift was 3.45x the float32 repeat difference;
  finite metrics alone did not establish equivalence. Packing crossover remains
  unknown. Step 2's accepted recipe is four seeds/one producer on A4000;
  L4 recommendation remains an inference until these measurements.

## Reversible implementation choices, 2026-11-08

- Keep the definition table in embedding-weight dtype, including empty rows;
  cast recurrent outputs at the join. This is a float32 no-op with gradients.
- C1 uses the existing launch owner and distinct `c1-...-u8/u12` keys. An
  `-overlap` key is a conditional whole-cell retry, not automatic permission to
  rerun or select the fastest cell. Source freeze and cost admission are external
  execution checks; discovery never rents. Preserve legacy four-update jobs.
- The maintained C1 reader lives under `experiments/`; historical artifact
  readers stay immutable. Explicit per-cell dispositions and rental bills are
  required; steady-state cost never stands in for the whole rental bill.
- Diagnostics take one retained 256-row observation fixture. They instrument
  fixed host recording and learner-style device/host selection/transfer only,
  one child with a 120-second deadline each. They do not reproduce a complete
  update or prove the cause of the original recording stall. No speculative
  performance repair or policy/collector/learning behavior change is included.

## Older unresolved findings

- The schema-7 retained replay fixture in
  `test_resource_limit_receipt_replays_its_retained_encoding_boundary` differs
  from runtime schema 8 (`binding_mismatch`); no fixture rewrite is authorized.
- Original step-1 recording stall and step-2 M2 seed-26214 hang remain unexplained.
  Step-1 RSS near 61.05 GB was consistent with the later measured 62 GB pod
  allowance, not proof of memory-pressure cause. Cgroup memory admission was
  repaired in step 2. Etude 429 cleanup, placement lease failures and absent
  `etude-result.json` blobs remain historical launcher findings; callback results
  were recovered from attempt records, so no missing callback is inferred.
- Proposed infrastructure memory, not written to read-only `.lf/wave/`: compare
  host CPU class/native time before attributing GPU speed; earlier same-label
  host variation was 1.6–1.8x. Packed ~50k-row device recursions were launch-bound.
- The earlier $100/336h planner sketch incorrectly said dollars bind at
  $.256/hour; 336h costs about $86, so hours bind. The retained report corrects it.
- Frozen Elemental recipe omitted input version but resolved to v6; investigate
  legacy loading separately if needed. C1 pins resolved v6 explicitly by admission.

## Main sync — 2026-11-08

Assumption (reversible): keep this Task's b015d5 continuation notes as the
active scratch context. Main's separate 5da5f4 study decisions remain at
`e0832f67fc66a1f2033bafefdb4598e394fd5ca1:scratch/questions.md`; its study
landing authority and budget do not replace this experiment's boundaries.
