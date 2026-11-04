# Decisions and open questions — 5da5f4

- Provider direction, Jack Heart, 2026-11-04: "we can push an etude sdk release
  and then use vast if necessary". A necessary SDK release and Vast use are
  authorized for the GPU plateau continuation within the original $25 total.
  Reuse existing Etude provider/release work; preserve runtime identity and
  rental cleanup. This does not approve merging the representation-study PR.

- Latest direction, Jack Heart, 2026-11-04: the 128-update CPU comparison is too
  short for decision making; actual GPU training must at least reach the part
  of the learning curve starting to plateau. Continue under the existing $25
  total cap with independent paired seeds, initialization/periodic evaluation
  and a prospective sustained-plateau rule. See the current direction at the
  top of `representation-learning-plan.md`. This supersedes the earlier
  restriction on further training below; historical evidence stays frozen.
  No model/default promotion or automatic merge is approved.

- Jack Heart's 2026-11-04 ship approval applied to the bounded shared-effect
  repair. PR419 merged as `fb5ad2c1372f9c967b550f0b8ae0f84883f75563` and passed
  final CI/Engine Budget and Linux/macOS package checks. Post-merge verification
  rebuilt the extension, passed 23 focused Python checks and four debug Rust
  checks. Receipts: `.runs/representation/verify-main/`; complete verification:
  `9dc32125f:scratch/shared-effects-main-verification.md`. No PyPI release claimed.
- Lossless grouping preserves payload, controller, exact source, membership,
  multiplicity and order. Explicit v7 uses 128 payloads / 512 detail tokens /
  2,048 memberships, with active effects removed from local facts. Fresh defaults
  stay v6; frozen v1–v6 and pilot evidence remain unchanged. These capacities
  are bounded allocations, not universal capacity guarantees. Historical captures
  omit native provenance: the 270-row audit proves old-projection roundtrip only;
  new native fixtures supply full-authority parity and viewer-safety evidence.
- Semantic coalescing remains deferred unless identical future behavior is proved
  across expiry, source changes, affected sets/later entrants, intervening grants
  and additive effects. MTG-240 owns the pre-existing keyword-ordering defect:
  Indefinite LoseKeywords(Flying), then Indefinite AddKeywords(Flying), then
  EndOfTurn LoseKeywords still lacks flying after cleanup. The layer code unions
  grants then subtracts losses regardless of timestamp. The shipped directed
  expiry check uses correctly ordered LoseAllAbilities; no records were deleted.
- Jack Heart subsequently authorized actual replicated learning within $25 total,
  separate from MTG-213. That study is complete; see the current plan and
  `experiments/state-representation-v7.md`. The inconclusive result promotes no
  winner. Results demo/review remains the Graph boundary; no new training,
  automatic model promotion or persistent-memory design is authorized here.
- Costs/teardown are in the plan. Provider credentials are Doppler `etude/prd`;
  artifacts are `manabot/prd`. Use nested injection, never copy or print keys.
  Task-local Etude allocations remain independent of MTG-213.
- Historical direction and evidence remain at `9dc32125f:scratch/`; earlier
  complete design/decisions at `8e6cc62df:scratch/`; inherited PR416 notes at
  `3b122de9078c985f846a0f3c5a3a42ebca3b1182:scratch/questions.md`.
- The cleanup compared against shipped PR419's tree, not the pre-squash merge
  base. The subsequent main sync is reconciled below and in the current plan.

## Main sync — 2026-11-04

Reconcile main `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f` with the completed
v7 study cleanup. Keep main's single catalog, schedule/remembered-subject/delayed
frequency relation kinds, configurable card buffers, and v5 pilot entry point.
Keep this branch's v7 study runner, completed receipts and removal of the
obsolete v5 representation prototype. The study remains bound to its retained
scientific source; main's subsequent rules/catalog changes do not authorize
rerunning or rebinding historical evidence. Main's T6 decisions remain at
`5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f:scratch/questions.md`.

Focused sync proof: rebuilt and atomically installed the native extension;
19 Python checks passed for typed T6 relations, retained scalar/vector shared
effects, source identity, experiment discovery and the completed study contract.
The debug Rust exact-source/zone-change shared-effect test passed. Both manifest
parents' entries compare equal in the merged manifest (11,940 entries). No push
or additional training was performed; the sync caller owns publication.

Reconciliation subsequently found default `max_actions` had grown from 64 to
512, changing the study recipe hashes despite the focused sync passes. Explicit
original observation axes restore the retained identities; a regression check
now guards both hashes. All eight retained endpoints again pass the existing
comparison admission and reproduce the published contrast without rewriting data.

## Concept-review scope — 2026-11-04

Treat the completed-cohort results boundary as the current scope, following
Jack Heart's latest continuation direction. Broader memory/relation proposals
remain deferred, not prerequisites or approved follow-up work. No unresolved
product choice blocks this step. The existing working plan records the review,
counterexamples and future questions; results acceptance remains with demo/review.
