# Corrected-world baseline operations handoff

Owner: Intelligence Task `fc765e` (`task_fc765e5f72b242e9ad7dca598658ba0b`), primary operating Session `c1021f5bac5f44b1becbd201268ba63f`. The Session started and explicitly accepted sole operating responsibility on 2026-10-15. Linear synchronization was pending at creation. MTG-122 no longer operates these baseline runs.

Jack Heart requested this dedicated Intelligence Task and transfer of monitoring to its operating Session on 2026-10-15. Own the existing Mini baseline and complementary GPU baseline through retained results, cost settlement and cleanup. This is an operational handoff, not permission to duplicate or restart runs. MTG-122 retains environment complexity/catalog research, deferred until the baseline is understood. Baseline promotion remains MTG-118's responsibility.

## Accepted work and acceptance

Use the public etude-sdk and existing ordinary TrainingRegime/checkpoint/arena path on corrected w6 after PR288/289. Preserve the recipe, exact sources, independent seeds 12261/12262/12263, frozen deal families, failures and private artifacts. Mini targets 10,000 updates/seed; complementary GPU targets 25,000/seed. No local training or evaluations. Report learning relative to initialization and frozen controls, all four deck-role cells against greedy/random, matched 10k points and the GPU 25k extension, independent-seed uncertainty, sample exposures and actual inclusive costs. Historical w4 results remain separately labeled, never a causal speedup comparison.

Deliver retained TrainingRuns/checkpoints, replayed evaluations, editable Etude report/notebook and concise HTML, verified publication, all attempts and an honest completed/incomplete conclusion. No automatic promotion, complexity experiment, or unbounded retries. Exact protocol and declarations: `experiments/corrected-world-baseline.md`, `experiments/corrected-world-baseline-plan.json`, `experiments/runners/corrected_baseline.py`, `baseline_gpu.py`, `baseline_monitoring.py`, `baseline_checkpoint_watch.py`. Read these before changing execution.

GPU authorization: $25 inclusive; one $1/1h calibration, three serial jobs at at most $7.50/9h each, $1.50 reserved for storage/settlement. One rental at a time. Full 25k targets must pass measured admission before science; no silent shortening. Save every 500 updates. Diagnostic milestones 1k,2k,3k,4k,5k,7.5k,10k,12.5k,15k,17.5k,20k,22.5k,25k. During-training diagnostics use rental CPU; post-training evaluations use Mini behind current work. Post-training GPU evaluation staging/runner remains to be completed, not already queued. Mini's latest retained operational plan uses an 80-hour inclusive envelope with 44h successor; preserve the saved reservations and actual history rather than granting another allocation. Jack Heart approved 72h then requested the baseline proceed with flexibility; the existing protocol owns the resulting accounting. Ask if a material extension becomes necessary.

## Adopt these executions, do not recreate them

All absolute control/evidence paths below are intentionally retained in the original checkout:
`/Users/jack/src/fantasia.measure-learning-progress-across-simple` (call it P). Its ignored `.runs` contains live SDK databases and private evidence and must not move or be deleted. A new Task checkout does not copy these files. Use `ETUDE_HOME=P/.runs/...` explicitly; never initialize a second database to represent the same rentals/runs. Local controllers are allowed; scientific work is remote only.

### Mini

- Host `jack@100.96.227.95` / mini-heart.local.
- SDK home `P/.runs/mtg122/launcher`.
- Current run `335bd645dafe`, source `09dc662d992583b8bd9c2cb62430a2a132f82354`, session `f89409f36a91`, attempt `a9878311d7d9`.
- Remote root `/Users/jack/.etude/sessions/f89409f36a91/attempts/a9878311d7d9`; outputs `outputs/l3-12261/training/run.json`. First seed targets 10k, preserving old saves/evaluations at 5k/10k.
- Existing sidecar `/Users/jack/.etude/mtg122-supplemental/baseline_checkpoint_watch.py`, `watch.pid`, `watch.log`; state `5000-eval/watch.json`. It finished 5k diagnostics and waits for first seed completion, then writes the original SDK stop file only at the seed boundary. Do not stop learning early or duplicate this watcher.
- Queued successor `c1ec4899c7dd`, source `135c34a010f3cb21010bee0a16dc44df7916adcc`, 44h cap. Completes first seed post-evals then seeds 12262/12263 at 10k with saves500, diagnostics1k..9k. Existing local bounded controller PID50923 observed alive at handoff; `P/.runs/mtg122/cadence-controller.{py,json,pid,log}`. It calls SDK wait for the queued run. Check liveness before recovery; never add a competing controller.
- Receipts `P/.runs/mtg122/w6-admitted-launch.json`, `w6-cadence-launch.json`.
- 5k result: 104 greedy games,79 wins;104 random,92 wins; all208 replayed. Greedy cells (26 games each): Lessons→Allies12, Allies→Lessons26, Lessons mirror24, Allies mirror17 wins. This is one checkpoint diagnostic, not method evidence. Local retained `P/.runs/mtg122/5000-diagnostic`.

### GPU

- Calibration launch returned successfully: **run `35a4fac5ab20`**, source `4492c7865f5079d5e3e710507e601f1c28af03f9`, last fresh state `setting_up`. No scientific GPU run yet. No second launch is needed.
- SDK home `P/.runs/mtg122/gpu/launcher`, pool `mtg122-gpu`, RunPod L4, one rental max. Live quoted rate $0.59/h; pool ceiling $0.69/h. Calibration seed12260,100 updates plus8 games per anchor. Admission uses 1.5x timing margins and must fit 8.5h before the9h scientific job cap.
- Receipt `P/.runs/mtg122/gpu/calibration-launch.json`; exact launch script `launch.py`; source/bootstrap `scripts/etude_native_job.sh` and GPU runner in source commit above. Initial local launcher TypeError happened before any job/rental; retained separately. Successful launch finished; no pending parent tool call remains.
- GPU SDK has no separately installed long-lived wait/controller yet. The new operator owns continued reconciliation, calibration artifact retrieval/admission, unique scientific launches after passing admission, and bounded durable controller if needed. Inspect SDK-supported operations and current rental before acting.
- Doppler `etude/prd` supplies `RUNPOD_API_KEY` and `MANABOT_REMOTE_ISSUER_CREDENTIALS`; never print secrets. Existing `launch.py` securely maps issuer JSON into AWS env. SDK needs env credentials; ordinary boto3 SSO availability alone does not provide them.
- Jack Heart approved reusable storage permission on existing dedicated issuer: inline policy `manabot-etude-controller-storage`, only `etudefantasia/manabot/etude-launch/*` plus constrained bucket listing and scoped federation issuance. Receipt `P/.runs/mtg122/gpu/storage-admission.json`, installer `admit-storage.py`. Listing, write/read and scoped federation passed. Existing worker role/shared policies were not changed. Do not revoke this reusable grant as if it were a temporary study grant. Workers get run-scoped temporary credentials, never issuer key.
- SDK source transport uses compression wrapper `P/.runs/mtg122/transport/bin/ssh`; prepend directory to PATH for control operations as needed.

## Operating responsibility

The new primary operating Session is the sole baseline operator after handoff; MTG-122 stops launching or recovering these studies. Adopt existing SDK controllers without duplicating them. Reconcile live status, owner liveness, effect history, rental and cost receipts before further action. Preserve original source/evidence paths throughout delivery; do not mutate frozen remote sources. GPU calibration software passed 8 focused tests, not live CUDA admission yet. The calibration's completed SDK status is not admission: read `calibration.json.admitted` and full receipts.

First actions: acknowledge ownership; inspect both SDK homes, existing Mini controller/watcher and GPU calibration; continue the admitted launch through calibration and scientific jobs within the accepted budget. Finish post-training evaluation staging behind existing Mini work and retained reports. A stopped operator conversation is not an automatic monitoring service. Record installed bounded controllers and their deadlines; Task automation was disabled and CI watcher stopped at transfer. No promise of automatic future conversation turns.
