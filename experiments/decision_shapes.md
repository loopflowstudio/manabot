# Combat and mana decision shapes on the Small TLA model

**Interrupted by stalls; one baseline continues.** On 2026-11-15 UTC, direct
reads of the nodes' local TrainingRun files confirmed five jobs had stopped
updating despite still being labelled running. They were cancelled after
copying their available checkpoints and diagnostic artifacts. Baseline
seed 26502 continues, last confirmed at 159 updates. No six-seed learning
comparison is available.

| Arm | Seed 26501 | Seed 26502 |
| --- | --- | --- |
| Baseline | cancelled after stall at 111 | running, last confirmed 159 |
| Combat pick-then-done | cancelled after stall at 17 | cancelled after stall at 54 |
| Manual mana taps | cancelled after stall at 17 | cancelled after stall at 40 |

The control pods were provider-confirmed deleted at 05:22:09/05:22:11 UTC.
The stalled baseline process was stopped separately; its healthy sibling
was preserved. All five output directories are retained locally under
`.runs/stalled-20261115/`. The supervisor only waited for terminal outcomes
and did not detect these nonterminal stalls. Cause is not established; no
card or deck has been excluded on suspicion alone.

Diagnostic `bca62f5daf0f` (`mtg265-hang-diagnostic-20261115`) is now running
on the freed slot of the remaining L4: same source b3b47e82d31b,combatseed 26501
and 500-update damping horizon, with calibration-only tags and no scored
milestones. The immutable job command exports PYTHONFAULTHANDLER=1, disables
core files, and uses a 40-minute SIGABRT timeout with a 15-second kill grace
to capture where execution stops. A timeout is an operational bound, not a
learning result or proof of a specific card bug. Its USD2/2-hour job allowance
is contained in the existing capped lease; no new GPU was acquired.

The active `.runs/decision-shapes-watch.py` now includes both study cohorts
and the diagnostic cohort before final cleanup. The investigation note is
`scratch/decision-shapes-stalls.md`.

## Restart placement history

**Training restart, source `b3b47e82d31b`.** Baselines are in
`mtg265-study-20261115`; combat and mana are in `mtg265-study-20261115-l4-48`.
Repaired combat calibration `1fa799261cc9` completed two updates on seed
26501, which had failed before its first update on the earlier source.
Two baseline jobs are active on the original acquired L4. The other four
jobs were re-admitted to a less restrictive L4 placement after three 80 GB
create requests returned HTTP400. Their original unstarted requests were
cancelled at USD0. Read fresh status for placement and update counts; admission
is not a completed training result. Population and
arms are unchanged: 900 training/100 held-out decks, 256 streams, common v10,
baseline/combat-pick/mana-taps at seeds 26501 and 26502. No deck exclusion
has been needed. Prior cohorts remain failed operational attempts.

Replacement hardware is capped at USD48 across exactly three manually rented
L4s (USD16 and 23 hours each). The launcher retained about USD1.1071 of earlier
rental costs when planning this retry, including estimates. With USD5 reserved
for evaluation, this stays within USD55. Jobs retain the 500-update damping
horizon with a 21-hour maximum job window and USD15.50 job allowance. This
shorter allowance fits the first lease after calibration and source setup;
it does not change the update target. The supply permits no automatic rental.
Calibration succeeded before full submission. The single current supervisor
`.runs/decision-shapes-watch.py` waits for both study cohorts to terminate, then
cleans up the leases listed in its machine file. The earlier restart supervisor
is stopped. Machine timers independently cap their windows.

Read current progress with `uv run etude status mtg265-study-20261115`.

| Arm | Seed 26501 | Seed 26502 |
| --- | --- | --- |
| Baseline | `38b377941f9d` | `4218e16c4678` |
| Combat pick-then-done | `957298cf5b02` | `0bbba4c6eb90` |
| Manual mana taps | `06b0117497fc` | `a1e699cc6f54` |

Three study leases are provider-confirmed: `mtg265-l4-held-9e942f7adbee`,
`mtg265-l4-48-held-a8e5905a3b0f`, `mtg265-l4-48-held-57d34622f6b2`.
The two newer L4s request 8 vCPUs and 48 GB instead of 12 vCPUs and 80 GB; GPU model,
price, code, model, streams, decks and updates remain unchanged. The manual
supplies have max 0 and cannot automatically acquire additional machines.

## Earlier failed full cohort


**Training stopped: `mtg265-study-20261114d`, source `ee0281c0d402`.** All six
jobs failed vocabulary admission: baseline at 2/12 updates, combat at 0/15,
and mana taps at 7/6. No completed study learning result is available. The
supervisor terminated all three L4 rentals after the diagnostic retry ended.

| Arm | Seed 26501 | Seed 26502 |
| --- | --- | --- |
| Baseline | `baae8dd0fb5d` | `3cae72a9ae8d` |
| Combat pick-then-done | `8b1208dffc09` | `86569bef127e` |
| Manual mana taps | `ef6848121d91` | `5b578c31fbea` |

The three earlier rentals each had a USD16 cap and 23-hour timer: USD48 maximum
hardware, with USD5 reserved for evaluation and preparation below USD1,
within the USD55 authorization. The supply `mtg265-l4-held` permits manual
rental only (`max=0`), so this cohort cannot acquire additional machines.
The rental caps, rather than the launcher's duplicate per-job reservation,
are the spending authority for these already allocated machines. Jobs have
a 22-hour allowance and USD15.50 per-job ceiling. The previous supervisor waited
for this cohort and then terminated exactly these three manual rentals:
`mtg265-l4-held-08fc662039b4`, `mtg265-l4-held-60c5dd4c4bee`, and
`mtg265-l4-held-996fb972dd40`. The timers remain independent of that supervisor.

Inspect the retained failed cohort with `uv run etude status mtg265-study-20261114d`.
The stopped attempts below remain history; do not relaunch them.

Exploratory study. Compare the existing combat decisions and smart automatic
mana payer with combat pick-then-done and optional per-source mana taps. The
frontier Small model trains for 500 updates on 900 TLA pool decks; 100 decks
are held out, with 50 fixed paired matchups against the heuristic. Two seeds
per arm (26501, 26502) make six training runs. Evaluation includes updates
50, 100, 200, 300, 400 and the final checkpoint. No learning result is available.

## Authorization and existing execution

In conversation `9840d3d8-9b50-41b1-b74f-00c65efd0216`, Jack Heart said
"no i meant 55 dollars" (2026-11-14 12:14:59 UTC), followed by
"maybe 500 updates then" (12:15:20 UTC). The retained continuation summary
also quotes "anyway this is approved". The USD55 total covers calibration,
training and evaluation; it is not an additional USD55 for each arm.

Preparation is committed at `05b2b64622`. The whole-set combat arm was
omitted because the current policy action representation cannot describe an
arbitrary set of combatants. These are ordinary sequential policy decisions;
this study does not restore the removed compound neural decoder.

On 2026-11-14 at 22:03 UTC, launcher records contained no decision-shape
runs, no local launch process was active, and RunPod reported no pods.
Earlier L4/Ada/4090/A4000 attempts had not produced a running calibration.
A fresh A5000 offer was available at USD0.27/hour with two runs per machine;
the A4000 offer had no capacity. A new eight-update, two-run baseline
calibration was submitted as `mtg265-cal-a5000-20261114`, capped at USD1.
Submission was initially preparing the exact git source, not training.

Reserve up to USD49 for training and USD5 for evaluation, with USD1 for
calibration. The final launcher quote and measured calibration determine the
training hours and placement. Do not duplicate a live cohort or spend beyond
the remaining total. Preserve all failed and partial attempts.

## Calibration repair, 2026-11-14

Runs `6574e52d4932` and `21b2a10d2022` failed before training. The numerical
suite passed 40 checks, then its tiny continuation fixture failed in CUDA
attention backward with `Invalid max_seqlen_k:202` on RTX A5000, Torch
2.15.0+cu132. Both receipts and logs remain in Etude. The rental was terminated;
the launcher's retained estimate is about USD0.02 for the machine.

Live-token attention supplied the padded slot count as the maximum sequence
length, even when the packed batch contained fewer tokens. Tighten that
upper bound to the smaller of the slot count and packed token total. This
does not change attention membership or require a device-to-host read. The
sparse-batch forward/backward regression also runs in the CUDA admission
gate; that gate is not disabled. This is consistent with the efficient
backward constraint documented in
[PyTorch issue 129523](https://github.com/pytorch/pytorch/issues/129523).
The focused local live-token tests passed (5 passed, 1 CUDA skip); GPU
verification and successful calibration remain required before full training.

The launcher validates a job's own dollar allowance before sharing its cost.
Use USD16 per job and a USD48 ceiling for the complete six-run cohort, with
two jobs per A5000 and up to three machines. A 48-hour maximum allows slower
packed jobs to reach 500 updates; jobs stop at that target. The machine price
ceiling is USD0.30/hour. Calibration and evaluation stay within the USD55
total; these ceilings are not measured bills or evidence of completion.

The repaired Ada attempt (`728d1b881b40`, `b095dc8808e1`, source
`e3e15828cc48`) passed CUDA numerical admission, including sparse attention
backward and optimizer checks. Training then refused tracking because the
launcher bound the experiment to `decision_shapes`, while the job declared
`decision_shapes_calibration`. Keep that shared experiment name; mark short
calibration regimes with a `-calibration` suffix instead, which also keeps
their checkpoint tags out of scored evaluation. No study updates ran in
either attempt. The Ada rental remains held under its USD0.80 ceiling while
preparation continues.

The tracking-fixed calibration (`bd63b6fe1c88`, `099e3658b035`, source
`eb88310ef319`) reached collection but failed at update zero: native admission
reported `policy input v10 required` for mana that cannot cast nonartifact
spells. The 900-deck regime had inherited v9 from the historical frontier.
Fresh arms now use v10 consistently, retaining width 64, two attention layers,
four heads and 256 streams. No historical checkpoint or evidence is rebound.
The earlier frontier curve is only a planning reference, not a same-input
control for this study.

The six v9 study runs in `mtg265-study-20261114` were stopped before training:
four cancelled, two capacity-refused, all USD0 in the launcher. Preserve those
records. Verify the corrected input with each arm before replacing that cohort.

## Successful smoke and final placement

At source `ee0281c0d402`, all three current-input calibration runs completed
two updates: baseline `362f5ba6146d`, combat `9729cdd9186d`, and mana taps
`a767d947ac05`. These used 256 streams; only the numerical gate's tiny
continuation fixture uses two streams. The baseline TrainingRun records
144.08 seconds for two updates, including collection and learning, excluding
machine setup. This short smoke supports placement planning, not a sustained
throughput or strength claim. The calibration rental was terminated.

The initial v10 cohort `mtg265-study-20261114b` exposed conservative launcher
budget reservation per job despite shared GPUs. Its 48-hour jobs made later
rentals too short, and the worker correctly refused with `remaining allocation
cannot cover the frozen regime`. Stop that cohort; all its attempts ended at
update zero or before training. Its acquired machines were terminated.

The next attempt was `mtg265-study-20261114c`, source `ee0281c0d402`, with the
same six seeds/arms and 500-update target, but a 28-hour maximum per job and
USD9 job ceiling. The whole cohort retains its USD48 ceiling. Placement is
two jobs per Ada GPU, up to three machines, under USD0.28/hour. The launcher
quoted USD23.73 for that full maximum first-wave rental window, including
startup; this is a quote, not a bill. The scientific question, damping horizon,
population and evaluation schedule are unchanged. Earlier failed/cancelled
attempts remain in Etude and are not reused as completed study runs.

All six `...14c` jobs were capacity-refused before execution, at USD0.
A5000s were unavailable. A40 and RTX2000-Ada create requests returned HTTP400;
termination was requested and provider inventory confirmed no pods acquired.
The replacement L4 placement above was acquired successfully. Provider
inventory confirms 16 vCPUs and 86–94 GB RAM per machine at USD0.69/hour.
The question, six seeds/arms, 500-update target and evaluation schedule remain
unchanged; only hardware and admission allowances changed.

## Runtime vocabulary failure (2026-11-14)

Full collection exposed a failure the two-update smoke missed: combat seed
26501 (`8b1208dffc09`) stopped before update one, and baseline seed 26501
(`baae8dd0fb5d`) stopped after update two, both with `policy received a
program token outside the vocabulary`. The other four study runs continue.
These attempts are operational evidence, not completed training results.

Source `972424a47c0b` adds the rejected token kind/hash to the error receipt;
it changes no valid observation, model, vocabulary or training schedule. All
six admission tests pass, and the bounded catalog coverage sweep passes. A
diagnostic combat retry (`13bb90d47e30`, cohort
`mtg265-token-diagnostic-20261114`) retains seed 26501 and the 500-update
damping horizon, with a 1.2-hour maximum job window, on an existing free L4
slot. It is diagnostic only and must not become a scored study candidate.
No additional rental is authorized; the same three USD16 leases bound total
hardware spending. The supervisor now waits for both cohorts before cleanup.

## Identified runtime tokens (2026-11-15 UTC)

All six original run logs ended with the same unknown-program-token admission
error. Diagnostic `13bb90d47e30` reproduced it at update zero and named
`(1, 0x31ddff0e)`, `(2, 0x3decd349)`, `(3, 0x193062ff)`: respectively
`/Effect::HaveAbilitiesOf`, field `definition`, and `Effect::HaveAbilitiesOf`.
Koh, the Face Stealer generates this runtime effect when choosing a
card exiled with its source; printed catalog programs do not supply its
vocabulary. Registration now includes the effect and its field, and the
runtime-token regression serializes an actual `HaveAbilitiesOf` value.
Both debug record-vocabulary tests pass. After rebuilding the cp313 native
extension, all ten admission and bounded record-coverage tests pass. No
repaired training has yet run; do not reuse old policy checkpoints with this
expanded vocabulary. All three study rentals terminated successfully. The
causal investigation is in `scratch/decision-shapes-vocabulary-failure.md`;
its prevention proposals are not implemented by this patch.

Repaired calibration evidence: `1fa799261cc9` (combat-pick seed26501)
completed two updates. Its TrainingRun records 168.26 seconds collecting
and 41.31 seconds learning (212.38 seconds for the stage including other
work). This is a short single-job placement check, not sustained two-job
throughput or a learning result. It remains excluded from scored tags by
its `-calibration` regime ID.

Placement repair, 2026-11-15 UTC: 80 GB requests c3ff7d1f1284 and83ceff99814a
returned HTTP 400; complete provider inventory found no corresponding pods,
and termination was requested. A later 80 GB request ff6caf3f30c5 also returned
HTTP400 and inventory confirmed no pod; its cleanup remains in the supervisor
list after a manual termination call encountered a busy supply. Those are
failed create receipts, not additional acquired GPUs. The 48 GB requests acquired
provider pods opsg4qsv72e1o8 and iz9foqpir9vv9v. The exact reason for the
provider's 400 replies is not established. No scientific or deck-selection
change was required for this capacity workaround.

### 2026-11-15: restart with durable training journals

The surviving baseline seed26502, Etude run4218e16c4678 at sourceb3b47e82d31b,
completed500updates, confirmed by its node's run.json and refreshed launcher
status. Preserve this completed candidate; do not queue its duplicate.
The other five study attempts were cancelled after stalls and remain excluded
from a completed learning comparison. Probe bca62f5daf0f ended at its40-minute
cap after39updates; the backward stack at that instant does not establish a hang.

Retries use the same engine, policy layout, populations, seeds and500-update
schedule with only the diagnostic logging implementation from PR540 backported.
Runtime training_game_logs=True persists rolling producer inputs before native
calls and marks collection/inference/learning/export. The runtime option does
not alter the regime digest or checkpoint binding. Logging can reduce throughput;
these retries measure learning by updates, not a throughput comparison. No decks
were excluded. The existing USD55 total includes retries and evaluation.

### 2026-11-15: exact stalled payment retained; filtered diagnostic replacement

The journaled mana seed26501 run6f59481c7a58 stopped at update2. Producer0's
pending native step was replayed separately: stream24/game seed37037, action196,
turn25/precombat main, actor0 choosing Add {G}, with an exact pre-action frame
match. The single-game replay did not return within60seconds; stream25 from the
same batch advanced. The outer payment context is Yue, the Moon Spirit's
waterbend activation, with {3} remaining after two waterbend taps. This does not
establish Yue as the producing source or a production-default root cause.
MTG-350 now retains the complete minimal source-bound reproducer; #524 and #538
are overlapping payment work, not approved merges. Full failed-attempt outputs
and original journals were copied before cancelling the stuck worker.

A replacement diagnostic training uses the optimized shared eval/training
journal writer from PR540, backported onto this study's unchanged native engine.
Its initial-copy CPU microprobe improvement is29.37% on a fixed retained snapshot;
this is not a GPU, disk or end-to-end training speed measurement. The native
pending-call durability and replay binding remain required.

The replacement excludes every training entry containing Yue in its main deck
or sideboard:879of900 entries remain. The same exclusion helper produces the
same population digest for every arm. The four healthy attempts continue with
their original frozen900-entry population; they are not relabelled. The filtered
replacement is tagged calibration and cannot supply a clean matched learning
comparison with those attempts. Its500-update damping horizon is retained with
a coherent6-hour allocation/regime window on an existing lease; no new rental.
A common-population study comparison remains unfinished. This operational
workaround does not alter native card admission or the held-out evidence.

## 2026-11-16 continuation after the Yue repair

Continue the original seed26502 mana-taps run9b87af18dcdb from its paused
update66 toward the same absolute target500, on all900training decks. Preserve
its raw policy, EMA and Adam exports and500-update damping schedule; collector
and RNG streams start a new segment as the existing portable continuation
contract records. The completed500-update baseline remains retained; no
baseline restart. The169-update filtered calibration remains separate.

Backport only MTG350's announced tap/life reservation repair fromcfee31352e
(already landed throughPR551) onto the frozen study engine, preserving the
experimental source-tap path and checkpoint ABI. Do not import current-main
world/layout changes or PR524 source preferences. Directed debug tests for
announced life and nested Yue payment pass. Native/checkpoint admission is
required before launch.

Disable training_game_logs for this continuation. Retain ordinary update/phase
timings and existing error capture; a new hard native hang may have less
pre-call evidence, so bounded diagnostic retries can re-enable the opt-in.
One exclusive8CPU/48GB L4,12-hour job within a13-hour capped lease, maximum
USD9 including startup; no automatic replacement rentals. Launcher study
rental total at planning USD32.24696207357906484849877806, within the existing
USD55 total authorization. This is a crash-fix/placement continuation of the
same exploratory question, not a new protocol or a clean completed study.

2026-11-16 capacity adjustment: L4 admission was refused before any run
queued; available A5000/A4000 offers also lacked capacity. Select one available
RTX4090/16CPU/60GB offer at USD0.89/hour instead. Shorten the job to8hours
(26,400seconds training watchdog plus reserves), within a10-hour/USD9lease;
retain the same absolute500-update target and500-step damping horizon, seed,
population and raw/EMA/Adam continuation. No additional scientific arm or
replacement rental loop. The CUDA numerics gate remains required.

2026-11-16 capacity adjustment: the4090offer vanished before admission, so
its unsubmitted request was retained as refused. A compatible A40offer is
available at USD0.59/hour (8CPU,48GBhost RAM,48GBGPU RAM). Use that hardware
with the already validated8-hour job and a10-hour/USD9capped lease. Model,
learning state, population, source fix and update target are unchanged.
