# Decisions and open questions — 5da5f4

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
- Scope assumption for this pass: compare against shipped PR419's tree, not the
  pre-squash merge base. Later unrelated main changes are outside this cleanup.
