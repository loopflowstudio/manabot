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
handoff is retained at `c2c813a47:scratch/step-3-continuation.md`. The protocol's
dated C1 device-validator repair status distinguishes implemented software,
offline reload evidence and remaining delivery/freeze/CUDA admission.

Keep effective hashed v6 / 512 facts and the Elemental learning recipe. c5fba4
owns its shared-effect catalog repair; do not infer TLA catalog speed from the
hashed ladder. Exact quotes and external diagnostic findings remain in
`7be759c04:scratch/vocabulary-performance.md`.

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
  literal `cuda` before checkpoint reload. The device-validator repair is now
  implemented and verified offline; delivery and a new freeze/admission remain
  required before opening other rentals.
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
- Comment d85854eb routes the device-validator repair through the existing
  software Graph. New freeze and full worker/reload admission follow delivery;
  no paid execution belongs to this implementation step. The parent experiment
  remains open.

## C1 device-validator repair — 2026-11-09

Implemented: parse `stage.actual_device` with `torch.device` and admit its CUDA
type. The real-checkpoint cases accept `cuda:0`, `cuda` and
`cuda:1`, rejects CPU/MPS/meta and missing/malformed device names, and retains
failed-run, numerical-failure, missing/zero optimizer-step and corrupt-byte
rejection. Each parametrized case starts with a fresh receipt and checkpoint,
so rejection cases need no reset of earlier mutations. No training, launch,
checkpoint or scientific-recipe owner changed.

Offline audit of unchanged run `3a1380d94b69` passed on 2026-11-09. Its receipt
has one completed BF16/256-stream stage, seed 26400, two updates, 420 optimizer
steps and no numerical failure. Retrieved the original 30,398,287-byte raw
checkpoint read-only from Etude, SHA-256
`c4a4cae0600cf4e169d4cb0e11405b24cde426822967b475597e6c030516b5a1`.
The actual `check` entry point passed artifact mapping/hash/size, ordinary
world/input/architecture reload, effective hashed v6 and finite-weight checks.
The receipt retains v6/512 configuration. Temporary local copies were removed;
no historical bytes, run status or artifact manifest changed. The initial fetch
lacked R2 environment variables; the protocol's existing manabot/prd Doppler
context supplied them without auth changes.

The implementation check passed 16 admission cases plus focused Ruff and
`git diff --check`; the protocol records the post-sync check. Gate owns affected
suites; execution owns the future CUDA proof.

Local receipt replay does not retroactively admit the failed execution or
replace the future CUDA worker/reload proof. Gate/delivery, prospective source
freeze/amendment and full admission on one rental remain later Graph steps.
No new paid execution occurred; C1 consumption and inclusive cost bounds above
remain unchanged. The parent experiment remains open.

## Main sync resolution — 2026-11-09

Assumption (reversible): retain b015d5 continuation, billing and device-validator
notes as this Task’s active context. Main’s unrelated MTG-268 study notes remain
verbatim at `e73ed9208915e623173b2f5409d1ad3c301169b3:scratch/questions.md`; their
measurement conclusions and landing direction do not replace this experiment’s
boundaries. Only scratch context conflicted; the source merge required no manual
resolution. The sync completed at `9ed7c133e`; no new experiment freeze is implied.

## Publication verification — 2026-11-09

`lf sync` completed without conflicts at `77cf3641f`. The publication diff
contains the indexed-CUDA validator repair, its checkpoint tests, and retained
C1 execution/protocol evidence. The prior PR #451 is merged; this publication
is the subsequent repair, not a claim that full CUDA admission passed.

`uv run --locked --extra dev --extra artifacts pytest tests/training/test_training_throughput_sizes.py tests/infra/test_experiment_data_untracked.py -q`
passed all 32 tests in 103.01 seconds, including both bounded CPU worker smoke
cases. Focused Ruff and `git diff --check origin/main` passed. Manifest fetch
verified all 538 existing throughput evidence files without downloading or
changing them. No paid execution occurred; the prospective freeze and complete
one-rental CUDA worker/reload proof remain after delivery.

## Gate — 2026-11-09

Comparison: checkout base `cb75f2166` through PR #455 HEAD `910766b93`, plus
the working tree. Only the admission checker and its existing test module
change executable behavior. Protocol, artifact manifest, Infrastructure memory
and scratch changes retain C1 evidence and delivery boundaries.

Validation plan: run `tests/training/test_training_throughput_sizes.py` for
device parsing, checkpoint reload/rejection and bounded launch smoke; run
`tests/infra/test_experiment_data_untracked.py` for evidence storage. Then run
focused Ruff lint/format checks and `git diff --check`. The repository selector
`.github/ci/select-tests.sh` selects integration only and has no local result
cache; all three Linux integration groups already passed on this PR HEAD.
No training/model/native/frontend owner changed, so their broader suites and
performance measurements are outside this repair. Full CUDA worker/reload
admission remains an execution proof after delivery and prospective freeze.

Gate results (no executable changes needed):

- `uv run --locked --extra dev --extra artifacts pytest tests/training/test_training_throughput_sizes.py tests/infra/test_experiment_data_untracked.py -q` — 32 passed in 132.96 seconds; no timeout or skipped test.
- `uv run --locked --extra dev ruff check scripts/check_throughput_admission.py tests/training/test_training_throughput_sizes.py` — passed.
- `uv run --locked --extra dev ruff format --check scripts/check_throughput_admission.py tests/training/test_training_throughput_sizes.py` — both files already formatted.
- `git diff --check cb75f2166` — passed.
- `gh pr view 455 --json state,baseRefName,headRefOid,statusCheckRollup` — open against main at `910766b93`; all selected CI jobs and aggregate passed, unrelated jobs skipped.

The existing PR description matches the repair and its limits. No software
delivery blocker remains. No paid execution occurred, no background process was
left running, and the parent experiment remains open. Full CUDA admission is
deferred to the authorized execution Graph after landing and prospective freeze;
local tests and green Linux CI do not claim that proof.

## Repaired-source freeze — 2026-11-10

Recorded in the protocol's "C1 repaired-source freeze receipt" at merged PR #455
source `b5e8f404adf55a2315a2810906cbb81c49465781`. Identities were recomputed with a
temporary read-only script (not retained). It reproduced every 2026-11-09
world, content, binding, setup, observation, input-schema, learning-schema and
regime digest exactly. This validated the method and showed that no identity moved.
The stale local extension (built 2026-11-08, before Rust changes in `cb75f2166`)
was rebuilt from this source before computing them.

Discovery ran `uv run --locked --extra dev --with <Etude a7718f5>` under the
isolated home, without Doppler, because discovery reads no secrets. The client
SDK reports 0.2.14, the same as at the first freeze.
No quote, bill or rental was touched. Execution review comes next.

## C1 repaired-source execution, 2026-11-10

- Preflight (18:24 UTC): no runs, cohorts or rentals of the repaired source existed in
  the isolated home. Posted bills for the seven earlier pods are unchanged
  ($3.803594); C1's first A40 posted $0.120043 (bound $0.133502 kept). Quotes:
  A40 and L4 at $0.59/h available, A4000 unavailable.
- Reversible choice: the new cohort's evidence goes in
  `experiments/data/training-throughput-b015d5-v1/c1-repaired/step-3-c1/` (the
  reader requires a directory named `step-3-c1`), so the old failed attempt's
  `step-3-c1/` files and their manifest paths stay as they are.
- Reversible choice: each non-admission rental runs a setup-only identity job that
  also runs the fixed-input CUDA regression (`identity_job.py cuda_test=1`), which
  is the per-GPU admission before BF16 legs without another training smoke.
- Other pods on the RunPod account (`etude-8a95fe6d04e9`, `etude-ecca7abeb4df`,
  `etude-61d222a90653`) and the 0.2.15 controller under the default Etude home do
  not belong to this home; they were left alone.
- Rentals (all inside the C1 line windows): A40 `b015d5-a40-ca270fabef30` 18:33–≤23:18 UTC
  ($0.59/h, rent max $3.00); L4 `b015d5-l4x4-63b721717acc` 18:56–≤22:47 ($0.59/h,
  max $2.30); A4000 `b015d5-a4000x4-75457219a81c` 19:07–≤01:07 ($0.25/h, max $1.55).
  Several A40 creates were refused by stock before the A40 rent succeeded (no machine
  records). Admission `537b2f764bba` (A40, seed 26400) passed the whole chain:
  CUDA regression 2 passed, 2 BF16 updates on cuda:0, verified checkpoint reload.
  Identity jobs `1d4183bcd812` / `60470ec08b59` / `c82420726c04` matched every frozen hash
  (Linux extension 1b29bf48…); L4 and A4000 also passed the CUDA regression.
- Driver: `c1-repaired/step-3-c1/campaign.py` (state in `campaign-<gpu>.json`, logs in
  `execution/`), launched via `run_campaign.sh`. Controller for the isolated home runs
  under caffeinate (log `~/.etude-b015d5-step2/logs/c1r-controller.log`).
- A4000: `etude rent b015d5-a4000x4` exited 0 and printed "rented until 01:07" for
  `75457219a81c` although the provider refused the create (machine log: "Refused: no
  NVIDIA RTX A4000 available in any data center"). Queuing with `--rent` then made Etude
  rent a real pod `b9dee825076c` (identity/CUDA job `c82420726c04` passed; deleted
  19:17:04) and later a second refused record `79384dd06468`, losing cell run
  `9c669820d400` (no game played). The driver correctly stopped on "run placed outside
  the named rental". A4000 is unmeasured in this pass (30-minute window expired).
  Etude defect for its owner: refused creates must not report a rental (exit 0),
  and phantom records survive `etude terminate`.
- Operations: the isolated controller's loop hung on a storage read at 19:45 (runs
  waited ~15 min) and was restarted; `etude queue` spends ~4–6 min per seed in a full
  `git fetch` of the repository for source preparation, so K=4 queues idle the rental
  ~20–25 min and K seeds start staggered. A 600 s queue timeout killed the L4 driver
  once after the runs were created; the driver resumed from the existing cohort.
- The maintained C1 reader (`load_run`) rejects real `_updates()` projections, which
  carry constant string labels (gradient, value_kind, …); its fixtures were all-float.
  Runs retain `updates-raw.json` (full) and `updates.json` (numeric fields only); no
  measured value is dropped. Reader defect for its owner.
- L4 ladder ended 21:14 by rule (remaining window < next watchdog + cleanup; a384l8
  K=2 exceeds 85% of L4 memory; w64 not needed). `etude terminate` was refused
  ("busy") until both of this home's controllers were stopped and the lease expired;
  Etude deletion 21:24:22 UTC, provider pod `i0p0guzl2a84m0` confirmed absent.
- Execution complete 22:24 UTC. Evidence published (281 manifest entries), report and
  protocol receipt written. All rentals deleted; no controller, driver, selector or
  watchdog left running. Phantom A4000 records `75457219a81c` / `79384dd06468`
  remain in the isolated database with no provider pod. Evidence review is next;
  the parent experiment stays open.
