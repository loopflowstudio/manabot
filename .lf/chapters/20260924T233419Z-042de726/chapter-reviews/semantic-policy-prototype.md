# Project chapter report: Semantic Policy Prototype — interval unknown (baseline)

Run id: `run_eb9e551a6da54150aa7ed427bf2d1fde`

Reviewed September 24, 2026, at HEAD `7ad9a41f99cf367240ab380b197226a122398820`.

## Project synthesis

**Intended user improvement**: Give manabot developers a learned policy that consumes real game state and semantic programs, executes authoritative Commands, and supports meaningful training experiments. The definition names this developer capability; it does not explicitly promise a better player experience or stronger human opponent.

**Current user experience**: The repository contains the runtime adapter, learned policy, bounded experiment runner, and a dated three-seed result. Developers have a concrete implementation to build on. This review cannot certify current reproducibility or replay of the recorded failures: the inspected checkout lacks the native extension and the experiment’s original checkpoint, dataset, and replay payloads.

**Chapter conclusion**: The Project delivered substantive progress from static diagnostics to learned engine execution. Its retained measurements also expose the prototype’s limits. They do not establish that semantic structure improves playing strength, and the complete current KR conjunctions remain unverified.

**Decisive facts**:

- July 17 commit `36d0784` delivered the semantic runtime join; July 18 commit `fd53966` delivered the learner and INT-11 experiment.
- The July 18 result records nine models across three seeds, 144 accepted Commands, zero illegal Commands, and zero replay mismatches.
- Its 36 arena games used only 36 learned Commands in total. Current source starts each game at a constructed combat position with six Otter-Penguins against an opponent at one life.
- Semantic and structure-shuffled models both achieved perfect identity-holdout agreement. Composition agreement was inconsistent, and every arm’s paired arena score was 0.5. These are explicitly retained as null or ambiguous structural evidence.

**Measurement fit**: Legality, runtime binding, reproducible training, and controlled holdouts are credible prototype measurements. The terminal combat fixture establishes a much narrower capability than sustained gameplay. The old KRs request measurements, not a positive semantic advantage; negative scientific results therefore do not themselves falsify those KRs.

**Next-chapter implication**: Likely **rewrite**—preserve the prototype and negative results within Intelligence’s consolidated training Project. Replace further standalone prototype certification with evidence from the selected Allies versus Lessons training-to-play loop. This is a recommendation, not an applied disposition.

## Objective evidence packet

**Coverage: incomplete.** All five supplied current KRs are included. `lf pm show --wave intelligence --project semantic-policy-prototype --json` returned the same definition and five claims, with no drift.

Bounded history inspection recovered July design documents, implementation history, and experiment results, but no authoritative chapter-start ledger or complete historical KR revisions. Design proposals were not treated as additional KRs without evidence that they became Project commitments.

**Expected KR rows: 5 · Actual KR rows: 5**

**Verdict totals: 0 holds · 0 does not hold · 5 unknown**

Evidence inspected:

- Exact ledger observed at `2026-09-24T21:59:35.585349Z`.
- [INT-11 report](../../../../experiments/int-11-semantic-runtime-policy.md), dated July 18.
- [Retained result summary](../../../../experiments/data/int-11-semantic-runtime-policy-v1.json), including per-seed measurements and artifact hashes.
- [Registered workload](../../../../experiments/workloads/int-11-semantic-runtime-policy-v1.json), current runner, adapter, policy, and focused test definitions.
- July 16–18 repository history and the supplied Task/runtime snapshot.

Read-only consistency checks confirmed nine result rows across seeds 1101/1102/1103, matching aggregate training times, and a workload SHA-256 matching the retained result. These checks establish internal consistency, not successful engine execution.

No experiments, builds, or test suites were run.

## Definition

> Build the smallest runnable manabot that consumes real viewer-safe game facts, typed ability programs, and structured legal offers and emits atomic Commands in managym. Train and evaluate it on engine-generated positions and a bounded playable world; use ablations and katas only to diagnose observed prototype failures.

**Verdict**: unknown.

**Evidence**: Current implementation and July measurements substantiate the intended architecture and a bounded historical execution loop. Current reproduction was not established, and the original executable evidence bundle was unavailable in the inspected checkout. “Smallest” also has no explicit comparison criterion. Earlier static diagnostics inform the design but are not proof of this complete definition.

## KR ledger

### 1. A runnable semantic policy consumes real ExperienceFrame facts, typed program structure, and InteractionOffer values for priority, targeting, and combat, then emits structured Commands with zero illegal decodes on a fixed engine-generated evaluation that includes more than 32 legal choices.

**Status and active interval**: current · exact wording first supported by supplied September 24, 2026 ledger → review now. Establishment date unknown.

**Verdict**: unknown.

**Evidence**: The July 18 result records 108 evaluation applications across nine models, plus 36 arena Commands, with zero illegal Commands. Its fixed dataset includes 35 target candidates and 64 represented attacker subsets.

The current [adapter](../../../../manabot/semantic/policy.py:74) joins the viewer frame, semantic catalog, and Rust offer authority; its `step` method executes through `env.step_structured`. The [evaluation runner](../../../../experiments/runners/run_semantic_runtime_policy.py:720) reconstructs each position and applies the learned Command, failing on an illegal application.

This is substantial historical support. The original evaluation payloads and checkpoints were not available for verification, and current native execution was not exercised. No valid illegal-decode counterexample was observed, so the verdict is unknown rather than failure.

### 2. The prototype plays a bounded authoritative micro-matchup or selected matchup end to end and is compared across at least three seeds with an identity or fixed-action baseline on legality, competencies, seat-balanced strength, policy loss, p50/p95 latency, and environment throughput.

**Status and active interval**: current · exact wording first supported September 24, 2026 → review now. Establishment date unknown.

**Verdict**: unknown.

**Evidence**: The July result includes semantic, identity-only, and structure-shuffled models at three seeds each. It reports policy loss, prompt competencies, inference latency, batch throughput, replay throughput, and paired-seat arena scores.

Recorded measurements include:

- 36/36 terminal games; all paired scores 0.5.
- Single-decision p50: approximately 0.336–0.384 ms.
- Single-decision p95: approximately 0.507–1.929 ms.
- Replay execution: 488.5 authoritative Commands/s.

The [arena implementation](../../../../experiments/runners/run_semantic_runtime_policy.py:964) uses a constructed lethal combat position. The receipt totals exactly one learned Command per game. This limits the strength interpretation, but does not by itself falsify a KR explicitly permitting a bounded micro-matchup.

The inspected summary lacks the underlying per-game traces and complete benchmark payload. Current execution remains unverified; the complete comparison cannot be certified from the summary alone.

### 3. An engine-generated holdout withholds card identities and at least one composition of known semantic operations; semantic, identity-only, and structure-shuffled ablations measure zero-shot or limited-retraining transfer and sample efficiency without admitting unknown primitives.

**Status and active interval**: current · exact wording first supported September 24, 2026 → review now. Establishment date unknown.

**Verdict**: unknown.

**Evidence**: The July workload reserves Fire Nation Cadets for identity transfer and South Pole Voyager for composition transfer. The [generator](../../../../experiments/runners/run_semantic_runtime_policy.py:466) rejects held-out definitions appearing among training objects and checks that the declared composition primitives, `gain_life` and `branch`, occur in training.

Retained mean exact agreement:

| Arm | Identity holdout | Composition holdout |
|---|---:|---:|
| Semantic | 1.000 | 0.333 |
| Identity-only | 0.000 | 0.667 |
| Structure-shuffled | 1.000 | 0.000 |

Per-seed results also record examples required to reach 90% **training** agreement, including null values when the threshold was not reached. This is a bounded sample-efficiency measure, not a transfer learning curve.

The result honestly measures negative or ambiguous outcomes. However, the original generated dataset was unavailable for an independent identity/composition split audit. The visible checks establish primitive presence and definition exclusion; they do not alone verify the full composition-isolation claim over the historical generated data.

### 4. Static program roles bind to actual visible objects, targets, costs, and choice roles in runtime offers, and failures can be replayed as exact engine positions and Commands rather than only classification labels.

**Status and active interval**: current · exact wording first supported September 24, 2026 → review now. Establishment date unknown.

**Verdict**: unknown.

**Evidence**: The July 17 adapter implements visible object/program joins, offer-source bindings, candidate-subject bindings, and revision/prompt-bound Commands. Focused test definitions cover 35 target candidates, attacker subsets, and invalid boundary inputs.

The July 18 runner emits recipes, source/post-state digests, frame and feature identities, Commands, and oracle-agreement results. Its [replay verifier](../../../../experiments/runners/run_semantic_runtime_policy.py:858) reconstructs positions, verifies the source and features, applies the recorded Command, and checks the resulting state digest.

The retained result reports zero mismatches across 108 evaluation rows and 36 arena Commands, including evaluations with incorrect oracle agreement. However, their original replay payloads were unavailable for this review. The inspected evidence also does not separately demonstrate the complete cost-role binding clause. Successful object and target joins cannot close that conjunction.

### 5. One documented command regenerates data, trains the bounded prototype, runs the arena and holdout evaluation, and emits versioned artifacts pinned to content, engine, observation, offer, model, and seed identities within an explicit compute budget.

**Status and active interval**: current · exact wording first supported September 24, 2026 → review now. Establishment date unknown.

**Verdict**: unknown.

**Evidence**: The [documented command](../../../../experiments/int-11-semantic-runtime-policy.md:15) invokes the complete runner through `uv`. Source inspection confirms generation, training, checkpoint reload and hash checks, holdout evaluation, arena execution, replay, and manifest emission.

The July result records content/semantic-pack identities, learning schema, protocol and offer versions, engine-extension hash, model seeds, and nine checkpoint hashes. The workload hash still matches.

Its explicit budget is 600 training wall seconds **per checkpoint**, not a 600-second bound on the entire workflow. Retained aggregate training cost was 75.56 wall seconds and 18.22 process CPU seconds, with peak RSS 399,917,056 bytes.

Current one-command reproduction was not performed. The inspected checkout contains neither the native extension nor the original `.runs/int-11-semantic-runtime-policy-v1` bundle. This is a verification gap, not proof that the documented command fails after normal environment setup.

## Shipped behavior

Repository history establishes delivery of:

- July 16: structural diagnostics and a retained `KILL_REDESIGN structural_capacity` decision.
- July 17: the authoritative semantic runtime adapter.
- July 18: the learned Transformer policy, three experimental arms, experiment runner, and result summary.

The earlier July 16 design describes a missing learned runtime join; the July 18 result supplies bounded execution evidence for that gap. Without a chapter-start ledger, this is a delivery sequence rather than a precise chapter outcome comparison.

## Task and PR receipts

| Task | Supplied PM state | Contribution and runtime finding |
|---|---|---|
| [ETU-25](https://linear.app/loopflow/issue/ETU-25/disambiguate-structural-encoder-optimization-capacity-and-cpu-cost) | Completed | Structural discriminator retained a kill/redesign result. Runtime reports a missing historical worktree and `no_action`, with reason “Task is being abandoned.” |
| [ETU-26](https://linear.app/loopflow/issue/ETU-26/prove-structural-semantic-representations-on-diagnostic-katas) | Completed | Static diagnostic result is retained. Runtime reports a missing worktree and a merged-PR successor recommendation. |
| [ETU-27](https://linear.app/loopflow/issue/ETU-27/build-a-semantic-policy-prototype-in-the-real-engine) | Completed | Adapter shipped in [PR #128](https://github.com/loopflowstudio/etude/pull/128), commit `36d0784`. Runtime reports a missing worktree and successor recommendation. |
| [ETU-28](https://linear.app/loopflow/issue/ETU-28/train-and-play-the-first-learned-semantic-runtime-policy) | Completed | Learner and INT-11 evidence shipped in `fd53966`. Runtime reports a missing worktree and successor recommendation. |

These lifecycle discrepancies do not authorize restarting completed Tasks.

## Priority and ownership findings

The supplied ledger contains no incomplete PM Task in this Project. Static diagnostics are historical contributions; their negative results do not justify restarting a separate kata program.

The accepted future direction emphasizes trained challengers in Allies versus Lessons. The prototype should contribute to that work where useful, with its narrow fixture and null structural result preserved. No Task was identified as actively executing an outcome belonging elsewhere.

## Surprises

- All three arms received identical arena scores because the terminal fixture offered little opportunity to distinguish sustained playing ability.
- Structure-shuffled models matched semantic models on identity transfer.
- The registered 90% semantic competency prediction was refuted. That experimental failure is distinct from the old KRs’ requirement to measure performance.
- Current Wave memory still describes an end-to-end semantic policy as the “next” experiment despite the retained July implementation and result.

## Open work

No incomplete provider Task appears in the supplied Project ledger. Runtime residue remains for all four completed Tasks. This review changed none of it.

## Evidence gaps

- Complete historical KR wording, establishment dates, replacements, and chapter interval.
- Original dataset, checkpoint, manifest, and per-Command replay payloads matching the published hashes.
- Current execution through the documented native-engine path.
- Independent composition-isolation verification and explicit cost-role binding evidence.
- A workload capable of distinguishing playing strength beyond one-command terminal combat.

Ordinal-to-verdict JSON:

```json
{"1":"unknown","2":"unknown","3":"unknown","4":"unknown","5":"unknown"}
```

Execution note: no PM, code, charter, or archival edits were made. A standard-library JSON inspection through `uv run --no-sync` unexpectedly created a local `.venv`; no dependencies were installed or experiments launched.

