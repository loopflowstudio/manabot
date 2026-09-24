**Intelligence Wave verdict: agree. Proposal only; final acceptance remains pending.**

The accepted direction addresses the main evidence gap: Intelligence has substantial training and evaluation machinery, but no established repeatable improvement or human-challenger result. Retain one Project with exactly three KRs. Intelligence owns training, measured strength, and candidate provenance; Game owns usable human play; Rules owns faithful shared semantics.

The existing charter’s broader advice, architecture, Team Sealed, and belief-research ambitions do not become chapter outcomes automatically. Current read-only state confirms three Projects, 19 KRs, ETU-21 abandoned, ETU-31 deferred with preserved work, ETU-34 done, and no unavailable Intelligence Projects.

One clarification applies to the proposal below: evaluate the pinned demo-opponent implementation on the **same corrected world** as the challenger. Historical results from different Learn/Lesson semantics remain separate evidence.

**Child provenance:** `run_6c9247e502994395a787cb713fc8252a` completed successfully through the requested unbound `project/start-chapter` invocation, with no bound subjects. Parent Wave Run: `run_955d00ac772e4a0998227ee546f16bc5`. No nested delegation, machinery repair, experiments, implementation, planning mutations, commits, or pushes were performed. No chapter archive was written.

The complete child contribution follows.

---

**Proposal only — final acceptance pending.** Review date: **2026-10-08**. No files, experiments, PM records, Tasks, or remote state were changed.

**Project:** Trained Challengers for Allies versus Lessons  
**Retained identity:** `73884527-4c81-4682-aa52-25fb586f08f9` — currently Search Teacher and Student Arena  
**Wave:** Intelligence

**Proposed definition:** Make training a manabot produce reproducible, increasingly strong opponents that the User can challenge in complete Allies versus Lessons games in Etude Fantasia. Demonstrate improvement at declared cost and repeatable wins against the User through faithful managym rules, viewer-safe decisions, and the ordinary play experience.

The User should be able to train a candidate, recognize that candidate in Etude, play a complete game against it, and use the result to choose the next training change. Intelligence owns whether the trained opponent improves and can beat the User. Game owns whether playing is legible and comfortable; Rules owns whether both participants play the intended game.

The reviewed baseline contains substantial machinery but no established human-challenger result. INT-18’s verified 720-game result supports historical arena capability. INT-11’s 36 games contained only 36 learned Commands. Neither establishes sustained learned play against this human. Current checkpoint loading exists, but historical model compatibility and full-game execution require fresh evidence. See the [arena review](../chapter-reviews/search-teacher-and-student-arena.md), [semantic-policy review](../chapter-reviews/semantic-policy-prototype.md), and [current training boundary](../../../../manabot/README.md:60).

**Exact proposed KR ledger**

1. **Repeatable training produces playable challengers.**  
   On the frozen Allies versus Lessons world, including the agreed Learn/Lesson pool and behavior, a documented training-to-play procedure succeeds across repeated complete executions during this chapter. Its versioned checkpoints load through Etude’s ordinary opponent path and complete games with both deck assignments. Retained inputs, configurations, seeds, source/content/interface identities, checkpoint bytes, replay evidence, and measured training and inference costs make each execution reproducible within a declared reproducibility contract and compute budget. Every attempted execution and failure remains accounted for.

   **Why credible:** This connects training directly to the opponent the User can face. A saved checkpoint, successful import, or isolated decision is insufficient. Repeated independent executions distinguish a usable procedure from one fortunate run; reproducibility need not imply identical floating-point training bytes unless the contract promises that.

   **Existing evidence:** The [documented local training and play path](../../../../manabot/README.md:7), [checkpoint loader](../../../../manabot/sim/flat_mc.py:185), and [INT-4 review](../chapter-reviews/search-teacher-and-student-arena.md) provide the starting mechanisms and historical limits. They do not close this new KR.

2. **A trained candidate demonstrates full-game improvement at declared cost.**  
   A candidate trained on the frozen corrected world improves over the pinned current demo opponent under a comparison protocol frozen before scored evaluation. The evaluation includes appropriate frozen controls, independent training/data seeds, held-out paired deals, both deck and seat assignments, declared training budgets and matched inference-compute comparisons. Complete-game results establish the predeclared improvement criterion with uncertainty, while meeting legality, information-safety, and competency gates. All candidate attempts, exclusions, failures, latency, throughput, and compute costs remain visible and determine an explicit continue, revise, or stop decision.

   **Why credible:** Full games test sustained decisions; paired deals and balanced assignments reduce matchup confounding; independent training seeds distinguish learning improvement from initialization luck. Separate training and inference accounting prevents additional search expenditure from masquerading as a learning gain. Negative or inconclusive results remain valuable but do **not** satisfy the improvement claim.

   **Existing evidence:** The [arena review](../chapter-reviews/search-teacher-and-student-arena.md) preserves INT-7’s weaker players despite better calibration, INT-8’s rejected learned guidance, and INT-18’s non-promotable challenger. The [belief review](../chapter-reviews/belief-aware-play.md) shows why a conditional-policy change or improved belief metric cannot substitute for stronger play.

3. **An identified trained candidate repeatedly beats the User in complete games.**  
   After faithful Learn/Lesson behavior and Game’s full-game usability prerequisites are demonstrated, a frozen trained candidate meets a prospectively agreed human-challenger success criterion across the complete scheduled evaluation cohort. The protocol fixes candidate identity and inference budget, deck/seat assignments, deal handling, session window, assistance policy, stopping rule, uncertainty treatment, and treatment of interrupted or invalid games before scoring. Retained match records identify the candidate, world, Commands, outcomes, and every attempted game. The result establishes repeatable wins against this User within that declared scope.

   **Why credible:** This directly tests the desired opponent experience. One victory, a cherry-picked session, automated wins, or losses caused by awkward controls cannot establish the outcome. Candidate changes require a new evaluation cohort; discovery games remain separate from scored evidence.

   **Existing evidence:** The [accepted human direction](../chapter-direction.md) establishes the desired result and weakest-link usability requirement. The [current opponent label](../../../../etude/server.py:2571) exposes only “Checkpoint,” demonstrating an identity handoff Game must improve. No qualifying human-match baseline is established.

The numeric improvement criterion, cohort sizes, uncertainty requirements, human success criterion, and new experiment budgets remain protocol decisions for parent reconciliation. They must be justified and frozen before scored measurement. This proposal invents no empirical thresholds and grants no compute authorization. The October 8 review is an assessment date, not permission to relax criteria or stop after a favorable result.

**Old Project and objective dispositions**

| Existing Project | Proposed disposition and treatment of its current KRs |
|---|---|
| **Search Teacher and Student Arena** — `73884527-4c81-4682-aa52-25fb586f08f9` | **Rewrite and retain** as the sole Intelligence Project. Rewrite KR1–3 into the training/full-game proof above without requiring a particular teacher or student architecture. Preserve KR4’s historical arena result as achieved evidence. Rewrite KR5–6 into the new improvement objective, subject to ETU-31’s deferral. Defer KR7’s ordinary Study delivery. |
| **Belief-Aware Play** — `8ec95ad5-d084-4fbb-8281-d44ed471b46b` | **Retire as an independent Project**, preserving all evidence. KR1–2 retain their bounded historical results and current serving failures. Defer KR3 live advice, KR4 calibration, KR5 exact-range comparison, and KR6’s broader learned-belief evaluation unless a selected challenger hypothesis warrants them. Retire KR6’s stale blanket implementation deferral as planning language without declaring its scientific outcome achieved. Retire KR7’s architecture convergence as a standing chapter objective; undertake it only for an observed consumer need. |
| **Semantic Policy Prototype** — `3f97924e-1bfe-41e9-8db2-cc27ac2117bb` | **Retire as an independent Project.** Rewrite KR1, KR2, KR4, and KR5’s relevant legality, runtime-binding, full-game and reproducibility requirements into the new challenger evidence. Defer KR3’s transfer study. Preserve null results and prototype limits; neither semantic architecture nor further kata work inherits priority. |

This accounts for all **19 current Intelligence KRs** without importing them wholesale as hidden acceptance gates.

**Every open Intelligence Task**

| Task | Proposed disposition |
|---|---|
| **ETU-21** | **Retire/reconcile** the provider record with the User’s accepted abandonment. Never restore or resume this attempt. Preserve its recovery evidence and the unresolved advice objective separately. |
| **ETU-31** | **Carry deferred.** Preserve its dirty worktree, scientific deferral, and **16-wall-hour / 64-core-hour hard cap**. Before any later restart, reconcile its controls, interfaces, corrected world and scientific question with this chapter; propose a rewrite if necessary. No run is authorized, and no replacement Task may bypass the deferral. |
| **ETU-34** | **Complete/reconcile** the provider record with its merged implementation and terminal Work state. Describe the delivered agent/belief scope accurately. Do not reopen implementation or claim that its bounded demos establish strength. |

Completed historical Tasks remain completed; missing-worktree residue does not create new assignments.

**Dependencies and opening work — proposed only**

Rules must establish the intended Lesson pool and faithful interaction, then exercise the same authority through agent, human, replay and required fork paths. Semantic changes require a new pinned world identity and fresh comparison cohort. Historical controls cannot be silently adapted.

Game must remove observed transition and action-selection friction, expose checkpoint identity, and retain complete human-match records. Any known awkward action blocks the full-game usability claim. Intelligence may investigate training feasibility while this work proceeds; scored human evaluation waits for these prerequisites.

For **ETU-55**, preserve the shipped provider contribution separately from deferred calibration. Rules should reconcile its remaining Task scope with the ship record; provider delivery does not complete the scientific objective. Game’s **ETU-14** advice continuation remains deferred from opening work. **ETU-74** retains repository planning/application responsibility.

Proposed opening sequence:

- Establish the exact demo-opponent baseline and the smallest current training-to-demo compatibility path; define the corrected-world handoff with Rules and Game.
- Prepare a bounded experiment protocol and budget proposal, reviewing ETU-31 before proposing overlapping work. Select architecture from runnable evidence, not inherited commitments.
- After prerequisite delivery and explicit authorization, execute the training comparison, then the separately agreed human cohort.

**Historical coverage:** The frozen brief lists no unavailable Intelligence Project. Earlier KR revisions and chapter boundaries remain incomplete, and some original semantic-prototype payloads are unavailable. Preserve those gaps; do not infer historical completion or invent missing obligations. Game/Study’s unavailable historical Projects remain parent-owned reconciliation.

**Boundary verdict: agree.** One Intelligence Project focused on reproducible training, demonstrated improvement, and a human challenger matches the accepted direction and the actual evidence gap. Advice, beliefs, search and architecture remain possible means. None becomes an additional chapter outcome merely because its machinery exists.

