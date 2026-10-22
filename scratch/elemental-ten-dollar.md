# MTG-174 accepted brief and historical execution record

The dated handoffs below retain evidence at their observation time, including
superseded launch plans and failing reproducers. Current payment delivery and SDK
limits are reconciled in `elemental-mana-repair.md` (2026-10-22); the actual failed
run is recorded in `experiments/elemental-uniform-baseline.md`.

Jack Heart requested this on 2026-10-21: “a simple $10 train makes sense and see how it compares to the baseline $10.” This is the first paid training run on the accepted ten-deck Elemental world from MTG-157/MTG-164, owned by Intelligence in loopflowstudio/manabot.

Outcome: one uncomplicated $10-capped Elemental self-play run, with a reproducible checkpoint, honest cost/throughput and outcome evaluation, compared with the existing Allies-versus-Lessons $10 baseline from MTG-159. This is a first baseline, not recipe optimization, strength certification or a $50/$100 ladder extension.

Authorization and ordering
- Jack Heart authorizes at most $10 of new paid training/compute for this run. Count startup, calibration, failed/retried segments and any paid evaluation against that same $10 ceiling; reserve enough to stop/export cleanly. No additional paid calibration/evaluation budget, other paid runs, $50/$100 continuation or automatic top-up is authorized. Local CPU evaluation may run boundedly without a rental. Record actual spend if a safe cutoff finishes below $10. Stop on the first of $10 total authorized spend or the existing one-day $10-tier wall limit; do not increase either.
- Wait for verified MTG-164 delivery and runnable ten-deck admission before launch. Inspect live work and avoid contention/duplicate runs. Pin source, world, rules, complete roster/Lesson pools, observation/action schemas and runtime. Preserve that runnable source and checkpoint binding even if the optional-payment task 4ba5dd or Academy redesign later changes main.
- Reuse the existing MTG-159 $10 baseline checkpoint/receipts; do not retrain or spend on a replacement without new authorization. Verify its exact cumulative spend point, model, recipe, world, seed and evaluation availability. A missing baseline may limit the comparison; do not fabricate it.

Simple recipe and comparison
Use the same baseline recipe/settings and calibrated hardware where compatible: inspect the actual MTG-159 evidence first (its brief describes a 64x2 model, 256 streams, 017 learning settings and horizon-independent schedule by absolute update). Pin the chosen recipe before running. Do not silently fold in two-seat learning (MTG-165), new policy-drift settings (770687), a new representation or other confounded changes. If a compatibility change is necessary for ten decks, document it and its comparison limit rather than relabel recipes identical.
Train across all ten accepted setups with a declared balanced schedule covering every ordered deck pairing and mirrors. Retain deck/seat identities, learner and environment decisions, update/exposure counts, terminal/truncated games, spend and wall time, checkpoints and recovery receipts. Do not add a matchup optimization curriculum in this baseline.
Evaluate with bounded reproducible CPU cohorts using fixed seeds, both seats and appropriate mirrors; report deck/pair coverage, completion/legality, outcome uncertainty and unsupported anchor cases. Reuse valid evaluation machinery and show gaps where existing heuristic/frozen opponents cannot cover Elemental. Do not claim a direct Elo or strength comparison between incompatible worlds or load old-world checkpoints into Elemental. Compare within-world performance versus declared anchors, learning relative to initialization, throughput and learning per dollar, making differences in task difficulty and available opponents explicit.

Acceptance
One budget-compliant run and admitted raw checkpoint, exact source/world/recipe and expenditure receipts, load/replay and bounded legality/completion proof, a concise result/comparison against the existing $10 baseline with failures and limits retained, and a recommended next question. No success claim requires the new world to beat the old one. Actual model training is the outcome, not only a recipe or smoke test. Use existing remote-job lifecycle, spending/shutdown controls and training monitoring. Paid work must stop within the cap; retry never duplicates consumed data or silently resets spend accounting. All Python commands use uv.

Future question: after this result, Jack Heart is interested in a sequel to the earlier “does simple->easy help” work. Read and identify that prior study before designing the sequel. This baseline task does not authorize paid curriculum experiments or change its baseline recipe to test that question. Preserve a clear handoff to a separately scoped follow-up.

Execution direction, 2026-10-21
Jack Heart reaffirmed the order: first the uniform $10 ten-deck baseline, then adaptive sampling BEFORE any hard-coded smaller-world switch. Do not delay the baseline for curriculum research, optional-payment refactoring, Academy redesign or the static-transfer sequel. Preparatory research may start now; paid training waits for verified MTG-164 delivery. Execute the already authorized single run autonomously within the hard $10 total cap, with no top-up and no paid curriculum arm. Review the latest task direction and avoid any existing live job. A blocked preflight is not permission to change the protocol or spend anyway. Keep actual cost, failed attempts and prior paid consumption cumulative across every retry. The scientific outcome is the single actual run, admitted checkpoint and retained comparison report; PR publication is not a prerequisite, and do not turn this into an engine refactor. Retain bounded reports and immutable receipts in the Task checkout/artifact store. Complete only if actual accepted outcome and budget compliance are verified; otherwise retain the concrete blocker. Do not start a second paid run to improve a result.

Live reporting direction — Jack Heart, 2026-10-21

Jack requested live reports featuring regular evaluations launched using the Etude eval system, and explicitly authorized necessary Manabot or Etude PRs to enable that. This adds an actual delivery requirement, not an optional final-report suggestion. Use Etude's actual evaluation/tournament jobs to launch regular checkpoint evaluations and publish its live report; merely uploading old MonitorResult summaries through the legacy tracker API is insufficient. Use a fixed prospective step cadence and frozen, bounded deal/pair/seat cohorts so the evaluations form a reproducible learning curve. Keep monitoring deals distinct from final scientific evaluation. Retain every checkpoint/eval identity, completion/failed/pending state, coverage, uncertainty, timing and real cumulative cost. The live view must make freshness and missing results explicit.

Use bounded local CPU evaluation launched/recorded by Etude when practical. Any rented evaluation resources still share the existing total inclusive$10 cap with training; this instruction authorizes necessary code/PRs, not a larger budget. Preserve the one-run baseline recipe and no additional paid campaign/top-up. Inspect existing work before changing either repository and prefer the smallest end-to-end bridge using the real Etude eval/launch/report APIs. Persist the evaluator/reporting owner so it actually continues between conversations; no promise of future checks in place of a running owner. Supply the actual live report location and observed first successful evaluation, then keep regular evaluations through the endpoint and final report.

Historical API inspection at Etude `1ee96ff`: typed tournaments and evaluation
jobs existed in source; legacy summary upload could not launch games. Automatic
training/evaluation cadence was unimplemented. The accepted bounded owner uses
exact source/runtime isolation, without migrating unrelated services or accounts.

Baseline comparison correction: experiments/scale-ladder.md shows the retained w6 policy scored0.11 initially and0.69 at update9250 AGAINST THE HEURISTIC, over200games per checkpoint. The0.11 is not a trained-policy head-to-head score versus initialization. Its small tier used$9.88 pod time and excluded separately charged calibration/evaluation. The new cap includes all paid costs. Preserve these cost/world/anchor differences in the live/final comparison and do not compare raw cross-world Elo.

The released 0.1.2 wheel lacked those source APIs; a version bump alone could
not enable evaluation. The full superseded API inspection is preserved in
`.runs/mtg174-realign-scratch-before/elemental-ten-dollar.md`. The separate
released-0.2 report incompatibility is recorded in `elemental-mana-repair.md`.
# Live report clarification — Jack Heart, 2026-10-21

Jack clarified in this primary conversation: “the reports i want to see are that training reports that know how to build graphs for different evals based on the eval runs where the x axis is still checkpoints/updates”.

Accepted reporting outcome: extend/use Etude's training report, with separate evaluation graphs sourced from actual linked Etude eval runs and plotted at each evaluated training checkpoint's absolute update coordinate. Preserve training run/checkpoint digest, evaluation identity/cohort/opponent, uncertainty, pending/failure/freshness, and a stable report location refreshed by the persistent owner. Standalone eval tables are drill-down evidence; a bespoke Manabot HTML/SVG learning curve is not the requested product. Necessary Etude/Manabot PRs remain authorized. Freeze this report/evaluation contract before paid launch; do not claim the prior standalone watcher or custom curve fulfills it.

Before the new scientific freeze, reconcile evaluation capacity with the newly verified 43,140-second training window. The superseded one-hour protocol currently says exports/evals every25 updates and 60 games per monitoring checkpoint. Clean engineering job174000000006 took approximately19–27 seconds per game, with two Etude workers. Those measured costs imply roughly10–15 minutes per monitoring checkpoint before overhead. Check that the chosen export/evaluation cadence can stay current and complete within the durable owner's actual lifetime; do not carry a one-hour plan into the longer run unchanged or silently accumulate an unbounded evaluation backlog. Re-freeze a sustainable prospective cadence and exact matching checkpoint exports before any scientific training, preserving the old unexecuted plan. The actual final endpoint and disjoint final scoring remain required; do not select checkpoints post hoc.

Operator review to resolve before paid admission: interrupted evaluation launch reconciliation must distinguish absent stable run IDs from accepted jobs without dropping requests or duplication; terminal monitoring should retain frozen cadence plus final endpoint rather than evaluating every retained checkpoint. Reconcile the earlier supported existing manabot-issuer 43,140-second admission (elemental-baseline-preflight.md:224-229) with the proposed default temporary-role 3,540-second run. Prefer the already supported longer path if fresh admission succeeds, preserving all actual guards and the inclusive $10 cap; no account changes or bypass.

## Verified reporting handoff — 2026-10-22 02:10 UTC

The old admission-native-legality-v1 failure and failed overlay startup in elemental-baseline-preflight.md are retained history, superseded for current engineering admission by the mana repair and reporting work. The mana fix f970a3e9 passed489 debug native tests,42 Python tests and the exact offered-action replay/full-size collector; source/runtime hashes are in scratch/elemental-mana-repair.md and immutable mana-repair-v1 receipts.

Reporting Flow6ebece9e-e6ab-404d-a488-dd7916055619's elemental-training-reports step completed successfully. Actual Etude evaluation174000000006 finished4/4 valid, replay-verified games, engineering score0.5, and is linked at update0 in the native training report `.runs/mtg174-eval-v2/reports/mtg174-engineering-training.html`. Earlier174000000003 remains visible as3/4 valid (one30-second wall timeout); it was not relabeled clean. The bounded clean retry used120-second allowance and the same checkpoint/pair/deal. These are engineering proofs, not scientific baseline scores or training.

Persistent launchd owner `gg.manabot.mtg174.training-report` was observed live and survived a real process restart without resubmitting the same single accepted eval attempt. The owner uses explicit training-run/checkpoint/agent identity, registers queued requests before submission, reconciles interrupted submission, preserves final scoring, and no longer runs a bespoke SVG curve. Its current job is engineering reporting; attach an appropriately bounded scientific owner to the accepted actual job before renting compute.

Published Etude PR28 https://github.com/loopflowstudio/etude/pull/28 (source5c18aca9; GitHub CI allgreen observed), Manabot PR345 https://github.com/loopflowstudio/manabot/pull/345. Manabot report/owner code through7779be63; immutable receipts committed incfd3b956. Read `experiments/data/mtg174/training-report-v1/receipts/readiness.json`, `owner-restart-proof.json`, `ui-proof.json`, source/runtime archives and test logs (fetch through existing artifact tooling if absent).73 new immutable receipts retain all prior manifest entries. The engineering runtime is separately identified (Etude452bd8a/wheel digest in readiness); do not equate it with published5c18aca9. Freeze and verify actual scientific source/runtime bytes after any necessary final update.

Validation:10 focused Etude report tests,5 owner tests,3 SDK-compatibility tests, typing/lint passed. Actual report JavaScript executed in Node confirms separate metrics at exact updates101/507/911/1203 with pending/failed gaps and stable refresh; no browser-rendered screenshot claim. Full Etude suite410 passed,6 skipped,3 deselected,1 unchanged controller timing failure under local load; all GitHub CI checks on Etude PR28 passed. No need to replay unchanged native gates.

Fresh existing manabot-issuer/worker-role admission explicitly passed43,140seconds (projected ceiling7.43 dollars, not actual spend). Recheck live jobs/prior charges/credentials/rates at allocation, use all real guards and inclusive10 dollars, do not choose3540-second temporary-role fallback merely for convenience. No scientific or paid resources started in these engineering steps. Remaining: sustainable prospective cadence/export/final-eval lifetime (see capacity handoff above), re-freeze exact scientific recipe/protocol/source/runtime before science, fresh cumulative-cost/live-job admission, one actual run, verified checkpoints/safe stop/cost receipts, linked periodic and final Etude evaluation, and honest retained MTG159 comparison/publication. No PR merge is an artificial prerequisite to running pinned verified source.

## Actual failed scientific run and user load-test stop — 2026-10-22

The run was actually executed from6006a360, stopped after4 updates on `PyAgentError: illegal mana activation`, and exported only initialization. Both task rentals were independently verified absent. Inclusive conservative lifetime cost0.051831584 dollars, covering guardian0.015915816 plus training/startup/CUDA/storage0.035915768; these are lifetime estimates, not provider invoice values.65,536 learner decisions and all100 ordered pairs were recorded, but no updated checkpoint or scientific eval survived. No replacement was launched. Published failure report: https://github.com/loopflowstudio/manabot/blob/e4a0a2833427fc40b23e92935cb6f6b8d1c7e116/experiments/elemental-uniform-baseline.md . Immutable source/log/control/cost/absence/report/initialization receipts are under experiments/data/mtg174/paid-run-v1/. The native report explicitly shows missing evaluation; scientific owner was disabled/removed. Flow6ebece9e-e6ab-404d-a488-dd7916055619 is terminal/completed via blocked route, Task started/incomplete, no live Flow/Process remains. Do not claim the objective's trained endpoint or scientific learning curve achieved. Local diagnosis/repair is still useful authorized work; further paid execution requires a concrete new decision and must preserve consumed cost/failure history.

Jack explicitly ordered immediate stop of `etude.remove-flaky` stress loops and two hour-plus pytest runs, including intentional load on main. Fresh OS command, lsof/cwd, LF and cmux inventories found no live pytest/stress processes and no process with that checkout open; it has no Task binding. The owning Eval runs design Session (cmux workspace1000080024/surface1000080162) acknowledged messagee2702a9d-5e77-44cd-b36d-d4e17b000808: its flaky-test work already finished, nothing remains running, and it will not restart intentional load/broad suites to chase flakiness. Preserve files/logs. Do not restart stress or broad load testing on main. The two2-hour Etude Python processes were unrelated report watcher/HTTP server from adaptive-curriculum, verified by argv and left alone. This stop applies to that stress workload, not a pause of MTG174's retained failure analysis.

Unverified native diagnosis lead from read-only source: priority_activate_ability_actions in flow/action.rs admits an activation using can_pay_with_mana_abilities (including chosen-output sources). activate_mana_ability then validates against mana_ability_actions, which gates upfront cost using available_mana_without(...).can_pay, apparently a different payment notion. A Prism funded by a chosen-output Treasure (or another converter) may therefore be offered at priority yet rejected on execution. This is a hypothesis, not an exact reconstruction of the failed GPU action; the retained log has no scalar prefix. Next useful step is one minimal bounded native regression proving/refuting this mismatch and inspecting current related optional-payment work. Do not run stress loops, repeat unchanged broad gates, hide a genuinely legal action to mask the error, overwrite frozen failure bytes, or launch another paid run merely to obtain a reproduction.

The minimal hypothesis is now independently reproduced in this checkout, after the first repair: `rules::temur_cards::priority_prism_activation_can_use_a_treasure_for_its_upfront_cost` sets up Prophetic Prism + one real Treasure token with zero pooled/fixed mana, checks that priority offers Prism, then executes that offered index. It fails with the exact `AgentError("illegal mana activation")` at rules/helpers.rs:250. One debug native test ran in0.02seconds (cached compilation1.18seconds), no stress loop or broad suite/main checkout load. The new failing regression is an uncommitted test-only change in managym/tests/rules/temur_cards.rs; runtime/frozen paid failure artifacts are unchanged. Run only that target with `cargo test --manifest-path managym/Cargo.toml --test rules_tests rules::temur_cards::priority_prism_activation_can_use_a_treasure_for_its_upfront_cost -- --exact --nocapture` as needed. This proves a second local offer/execution mismatch, not the exact missing paid-run action prefix. The original Forest/Professor payment regression remains a separate fixed case. Broader optional-payment Task4ba5dd is stopped at landing (no auto-merge request) with no live Flow, checkoutmanabot.optional-payments; read its existing design/code before choosing a shared payment owner, do not relaunch its work or bypass its merge decision.

Jack asked whether PR348 helps. Inspected https://github.com/loopflowstudio/manabot/pull/348 at head5c270f6c77586004fa566249e0a02dff217fea96: it delivers the earlier spell-color/upfront fixed-source planner repair to current main, plus optional-payment compatibility, but priority_activate_ability_actions and mana_ability_actions retain the differing affordability predicates. Verified against an isolated git archive of that exact PR head, appending only the17-line Treasure/Prism reproducer. Single debug rules test failed in0.02seconds with the same `AgentError("illegal mana activation")` (build6.29seconds);297 other rules tests filtered, no stress loop/broad suite, no edits to the active PR checkout, no paid allocation. Stage metadata is `.runs/pr348-check.json`. PR348 helps deliver the original fix but does not resolve this second local reproducer or prove readiness for another paid attempt. Do not claim this reproducer reconstructs the missing paid GPU trace.


## Native training report handoff — 2026-10-22

The interrupted Flow16960ef7-aaf8-484a-9afa-e8fb4e708344 was deliberately stopped for Jack Heart's changed reporting contract. It was not a scientific run, a new native failure, or a Task hold. This reporting step used no paid resources and did not launch scientific training.

Published code: Etude https://github.com/loopflowstudio/etude/pull/28 and supplemental Manabot https://github.com/loopflowstudio/manabot/pull/345. Etude's own training report now draws distinct linked evaluation metrics against exact checkpoint clocks, with prospective requests, statuses, observed/scheduled counts and whole-match uncertainty. The top-level `etude.reports` reader owns the join; training and evaluation run implementations retain their import boundary. Mutable checkpoint tags do not detach points. The Manabot owner registers queued requests before submission, reconciles an absent stable ID versus an accepted job after interruption, keeps cadence multiples plus the terminal endpoint, and reads results by exact candidate identity rather than RoundRobin seat 0. Existing rules, sampler, training SDK pin, and their receipts are preserved.

Actual stable report: `/Users/jack/src/manabot.elemental-ten-dollar/.runs/mtg174-eval-v2/reports/mtg174-engineering-training.html`. Persistent launchd owner: `gg.manabot.mtg174.training-report`; current readiness: `.runs/mtg174-training-report-owner.json`. It refreshes native Etude reports every 30 seconds, has a fixed 24-hour expiry, and cannot submit, rent or start training. An actual restart changed its PID while retaining the same single evaluation attempt. Scientific owner code is `experiments.runners.elemental_live`; attach its frozen config to the accepted scientific job before any rental, rather than pointing it at a nonexistent job intent now.

Recovered engineering jobs 118e59e21094 and 174000000002 failed; 174000000003 finished as a process but its fourth game hit the 30-second game allowance (3/4 valid, only three score rows). All effects/logs remain retained. A separate source-bound local retry, 174000000006 / attempt eab1199efe7b, used the same checkpoint, deck pair and deal with a bounded 120-second game allowance and 600-second subprocess allowance. It completed 4/4 valid replay-checked games, score 0.5, and appears in the native training report at absolute update 0. Exact training run: 8cc9fa4c7e8244009323d55957a85691; checkpoint SHA256 5e1bb95b493470560186b17a3a55d23119f50474e2ecda4cebdf1e0539ef8ef6. These are engineering proofs, not scientific baseline results or strength evidence. Earlier partial data is still visible, with no invented fourth score or uncertainty interval from one match.

The actual game proof used Manabot source 60ad3eb79273ba0321efb059b5ede3e0a47abfac and Etude source 452bd8a (full identity in the retained receipt), built into wheel SHA256 3b46aca22eb1b62fbb3b21e6de24d2313375f8ceb1f74de030932fb24b2228c0. Separate `.runs/mtg174-report-runtime` and `.runs/mtg174-report-worker` environments retain exact dependency freezes; earlier evaluation environments were not overwritten. Public PyPI 0.1.2 remains insufficient. Etude's published source is 5c18aca9e189cdcf2077269e6066585d1431dcba; the original source history is retained at `refs/mtg174/training-report-original-source`. No main integration changed the original Manabot scientific checkout. The supplemental PR uses a separate branch on current main ancestry because the original source branch has rewritten upstream ancestry; its five adapter/test files are copied byte-for-byte from the preserved checkout, plus only the new receipt manifest entries.

Validation: ten focused Etude report/CLI/counting/import-boundary tests, five owner tests, three training-SDK compatibility tests, mypy and Ruff pass. The actual report JavaScript was executed headlessly with exact updates 101/507/911/1203, separate won/hits metrics, out-of-order completion, failed/pending gaps and stable refresh; no browser-render claim is made. A prospective owner request was also registered twice and remained one visible pending request with no launcher DB or submission. Final full Etude suite: 410 passed, six skipped, three deselected; the unchanged controller-loop test `tests/launch/test_controller.py::test_loop_survives_an_empty_queue_and_places_later_work` fails a shared ten-second startup/submission/two-job deadline under load, leaving one pending run. That verification limit is retained in the PR and full log; no unrelated controller changes or other live work intervention was performed. Native source/runtime hashes still match the repaired receipts, so native gates were not repeated.

Immutable evidence: 73 files under `experiments/data/mtg174/training-report-v1/`, published through `manabot.data`; `experiments/data.manifest.json` preserves every previous entry. Includes raw eval tables and game/replay traces, all four engineering jobs' source/status/log receipts, source-built wheel, exact checkpoint, native report, fixture/JavaScript proofs, credential admission and actual owner-restart proof. The untracked-data regression passes.

Fresh credential-only admission at 2026-10-22T01:48:17Z succeeded through existing `manabot-issuer` credentials and the existing worker role for 43,140 seconds, using normal guards and scoped STS issuance. No job was persisted, rental created or account changed. The projected rate ceiling is $7.4296666667 inside the same inclusive $10 cap. Earlier diagnostic attempts made an unnecessary assumption about the IAM user's display name, then attempted an unneeded IAM role read; their failed diagnostic receipts are preserved. Actual admission relies on the existing profile, existing role and worker-credential guards, and did not require that IAM read. The temporary-role 3,540-second default is not a reason to shorten the scientific protocol silently. Repeat all admission fresh at actual allocation.

Remaining work only: fresh cumulative-budget/live-job/credential admission; re-freeze scientific cadence/cohorts, exact source/runtime and inclusive-cost reserves (the old `.runs/mtg174-plan-final.json` short-window plan is not final admission); attach the bounded scientific owner before rental; execute the one original inclusive-$10 run and retain its endpoint and honest baseline comparison. Keep monitoring/final cohorts disjoint and the baseline's $9.88-versus-inclusive-cost and old-world anchor limitations explicit. No Task completion, merge, top-up, second scientific arm or unrelated main sync after scientific freeze is selected by this handoff.

## Scientific execution admission — 2026-10-22 02:15 UTC

Re-frozen runnable source is `6006a360addbe2a3dcf093750070b45a3c659223`,
publicly retained at `mtg174-frozen-uniform-20261022`; no unrelated main sync.
The original branch push was rejected for rewritten upstream ancestry; a separate
retention ref preserves these exact bytes without force-pushing another history.
Superseded unexecuted short protocol/spec are retained as named companions.
Native, lock, roster/cohort and recipe identities remain checked; current sampler
suite passes four tests, and six isolated new-SDK owner tests pass. The released
training SDK correctly skips the new-SDK owner suite; its tests run in the exact
isolated report runtime, with no repository conftest that imports legacy tracking.

Fresh provider/S3 inventory at02:11 UTC found only the unrelated MTG-165 rental,
no MTG-174 job/cohort prefix, and L4 at$0.59/hour. Existing issuer credential
admission again succeeded for43,140 seconds. New guarded job intent is
`mtg174-elemental-uniform-001`; deadline is2026-10-22T14:13:17Z.
No receipt assumes a provider invoice or resets earlier effects. Main projected
ceiling is$7.429667 plus$0.05 guardian observation reserve, inside the same$10.

Frozen cadence:500 absolute updates plus initialization/actual endpoint,
60-game fixed monitoring cohort; disjoint final comparisons retain200games each.
The owner cancels live training on more than two unfinished monitoring points.
Measured19–27-second games imply10–15minutes/monitor point before overhead;
predicted training makes500updates take2.05–8.19hours. Monitoring jobs are capped
at1hour; final comparisons at2hours each; no paid evaluation or retries.
Scientific evaluator/report owner `gg.manabot.mtg174.scientific-evaluator`
was attached and observed the persisted job intent before rental submission.
It has a24-hour absolute expiry and an immutable configuration hash.
Evidence and current ledger: `.runs/mtg174-execution-v2/`.
The engineering report owner remains separate. Actual scientific evaluation,
training, spend, termination and publication are still to be verified.

## Actual single-run failure — 2026-10-22 02:24 UTC

Current accepted follow-up direction from Jack Heart: review mana planning as a
whole, invent bounded adversarial payment scenarios, and use **one payment model
shared by offers and execution**. Review and scenario matrix live in the existing
`scratch/elemental-mana-repair.md`. Five earlier Prism regressions still pass;
the tiny priority Prism + Treasure counterexample still fails. No paid retry or
stress/load testing on main is authorized by this repair direction. Jack also
requested launching a separate whole-Comprehensive-Rules audit with report plus
approximately 10k lines of draft code/test/documentation changes; the report and
rule index are outside that line budget. That audit must coordinate rather than
rewrite the active payment repair in parallel.

The authorized paid job actually ran. CUDA numerical checks passed, then
TrainingRun58c2a9c8aea0464eb45281fbc1756075 failed with
`PyAgentError: illegal mana activation` in
`SeatRoutedCollector._collect → VectorEnv.step_into_buffers` following four
successful updates. Frozen source6006a360 and Linux native digest
b6c487f16c0054fa347f71a25577e58301b9bfdee4594d9a562d1eda2a787f79
remain intact. No rules/recipe changes, alternate seed or paid retry followed.
The failure log does not identify a scalar root/action-prefix reproduction;
that remains engine-owner diagnosis rather than a claimed known cause.

Final generation3 is complete and hash-verified, but contains only initialization
SHA25657b5ee3d64156e41bd88934767e1b4a60371a80cc787dbd72195c88921858b50.
It reloads through ordinary world admission. There is no updated raw/EMA endpoint.
Recorded counters:65,536 learner and131,593 environment decisions,773 roots,
517 terminal games,0 truncations; all10 decks/100 ordered pairs/10 mirrors,
197/200 pair/learner-seat cells. These are last-successful-update counters;
failed collection actions are not asserted complete. No strength/gain/CI exists.

Guardian1dn86izhiw1wmg and traininguo0ykm5i22ijdt are independently confirmed
absent. Immutable deletion estimates are$0.01591581600440873 and
$0.035915767761733795: **$0.051831583766142525 inclusive total**. The later broad
provider inventory could not parse an unrelated pod shape; exact owned-resource
GETs, not that failed listing, establish absence. No other rental was modified.

The scientific owner registered initialization request25314209a04f but was still
publishing source before launcher acceptance when training failed. It was stopped;
that exact stable ID was absent after shutdown, with no replacement submitted.
The installed scientific LaunchAgent plist was removed after preserving it;
its ledger retains failure and confirmed cleanup, so restart cannot revive work.
The actual native training report is
`.runs/mtg174-execution-v2/owner/reports/mtg174-elemental.html`; it retains a
failed training projection and missing pending request, with zero scientific
eval games and no first successful scientific checkpoint. Earlier engineering
proof remains separate. Do not launch final comparisons of initialization as
though it were a trained endpoint. Full eval/endpoint requirements remain unmet.

Actual outcome report: `experiments/elemental-uniform-baseline.md`.
Immutable new receipts: `experiments/data/mtg174/paid-run-v1/` (publication
being verified). MTG-174 and the requested objective remain incomplete because
the native legality failure prevented a trained artifact and real learning curve.
No automatic budget continuation or replacement run is authorized here.

Publication verified at02:27 UTC: all68 initial artifact manifest hashes matched,
all previous manifest entries were preserved, and direct S3 GETs independently
matched summary, native training report and initialization checkpoint digests.
The verification receipt/script and untracked-data test log were subsequently
published as three additional entries. `tests/infra/test_experiment_data_untracked.py`
passes; raw files remain outside git. Report and manifest are being committed
for the actual failed outcome; the frozen scientific source ref remains6006a360.

Final publication: report/README index/71-entry receipt addition committed as
`e4a0a283` and pushed to `mtg174-actual-report-20261022`. The independent frozen
source ref remains`6006a360`; no PR, merge or Task completion was performed.
