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
The repair was delivered here for ed2300 in PR #451, with no second checkout.
Integrated software review is recorded in comment e98993af.

## Execution status and remaining work

- Source/world/input freeze is recorded in the owning protocol's C1 receipt
  (2026-11-09), at integrated PR #451 commit 58eb3c729. All 87 declarations
  passed isolated-client discovery and current-world admission; 20 targeted
  checks passed. The catalog digest changed while the roster and recipe stayed
  unchanged. Original source cb722dc8c results remain separate, with fresh C1
  controls required. On 2026-11-09 the CUDA regression passed and the w160l8
  BF16 worker completed two updates, but admission rejected `cuda:0` against
  literal `cuda` before checkpoint reload. Repair and a new freeze/admission
  remain required before opening other rentals.
- Billing was refreshed before C1: prior conservative bounds $1.917 step 3 /
  $4.012 overall. C1 adds at most $0.133502 pending billing, for inclusive
  bounds $2.050474 / $4.145347. Original $10/$50 caps and the C1 15 GPU-hour /
  1,600-update ceilings remain; C1 used about 0.206 GPU-hours and two updates.
  No rental liability remains beyond this completed window.
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
- Infrastructure memory now retains the host-confound, CUDA-admission and
  overlap lessons. The ~50k-row launch-cost diagnosis remains specific to its
  measured model/input; it is not a conclusion for all sizes.
- The earlier $100/336h planner sketch incorrectly said dollars bind at
  $.256/hour; 336h costs about $86, so hours bind for one machine. The protocol
  sketch now agrees with the retained report; dollars limit the machine count.
- Frozen Elemental recipe omitted input version but resolved to v6; investigate
  legacy loading separately if needed. C1 pins resolved v6 explicitly by admission.

## Main sync — 2026-11-08

Assumption (reversible): keep this Task's b015d5 continuation notes as the
active scratch context. Main's separate 5da5f4 study decisions remain at
`e0832f67fc66a1f2033bafefdb4598e394fd5ca1:scratch/questions.md`; its study
landing authority and budget do not replace this experiment's boundaries.

## C1 execution, 2026-11-09

- Existing C1 work was absent in the isolated Etude database; provider inventory
  was empty. The original seven rentals all had deletion records. Refreshed
  posted bills total $3.804; reserving every unposted second through deletion at
  each pod's maximum observed all-in rate, plus one minute for rounding, bounds
  prior step 3 at $1.917 and the experiment at $4.012. Exact rows are retained
  under `step-3-c1/preflight/` in the artifact evidence directory.
- Reversible execution choice: use a separate setup-only Etude job with the
  frozen training job's source/setup/config and a read-only identity command.
  This validates the remote Linux build before queuing the unchanged scientific
  job; it initializes one native match and plays no games. Its command and run
  receipt are retained separately from the 87 declared cells.
- A40 rental `b015d5-a40-0e2b61730e5f` / pod `iaimonawexhw5a` began at
  22:34:22 UTC, initially bounded to 03:34:22 UTC on November 10. Quote/provider rate
  $0.59/hour, 9 vCPUs, 50 GB; reserve $3.05 including cleanup within the $3.30
  line. Identity job `442a11544648` passed all frozen hashes on real CUDA 12.8 /
  Torch 2.10.0. Admission `3a1380d94b69` passed both CUDA test cases and
  completed 2 updates / 420 optimizer steps / 514 games / 107,129 learner
  decisions, with no nonfinite/rejected steps or discarded ledger samples.
  It then failed at the validator's exact device-name comparison, before reload.
  The unchanged receipt reproduces the error on CPU; the prior synthetic test
  used `actual_device="cuda"`, missing the real `cuda:0` form.
- Etude termination was initially refused by the controller's lease. After
  stopping the controller, termination succeeded at 22:46:44 UTC; provider
  absence and no active attempts were verified. No execution process remains.
  All 87 cell dispositions and export/game accounting are retained under
  `step-3-c1/`; private game ledgers and checkpoints stay in the Etude store.
- Next software work belongs to manabot's C1 admission validator and its
  existing test: parse CUDA device types, preserve CPU/unhealthy/corrupt-byte
  rejection, then freeze prospectively and rerun full admission. No bypass or
  further paid execution is part of this stopped pass; no new approval or
  expanded budget is implied. The parent experiment remains open.
