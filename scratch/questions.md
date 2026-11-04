# Decisions and open questions — 5da5f4

- Latest outcome: the v7 comparison completed on source92becff609. Eight
  replicates trained32,768games/4,056,651decisions; all1,024held-out games replayed.
  GRU minus pool=-0.98percentage points,95%cross-seed interval[-10.13,+8.18];
  elapsed training ratio1.06. The declared advantage criterion failed; no winner
  or default is promoted. See experiments/state-representation-v7.md and its
  published receipts. Graph boundary remains results demo/review.

- Jack Heart's 2026-11-04 02:49 UTC direction prioritizes pre-encoding complete
  effect consolidation after MTG-213 v3 reached270facts. Same-input v5 neural
  factoring is insufficient. The plan now implements an explicit v7 cut; merged
  PR416's v6 and every frozen protocol/input/receipt remain unchanged.
- Lossless grouping preserves full native payload, controller, exact source,
  membership, multiplicity and order. Semantic deletion/coalescing across
  distinct activations is deferred unless every relevant continuation is proved;
  audit intervening grants, expiry, source changes, entrants and additive effects.
- Historical captures omit native provenance. The transferred270-row audit
  proves only old-projection roundtrip; do not synthesize the absent source or
  controller. New native fixtures supply the full-authority proof.
- Engineering choice: explicit v7 with bounded128payloads/512detailtokens/2048
  memberships; local fact table excludes active effects. Fresh defaults stayv6.
  These capacities are not a guarantee over all legal games. Tensor/model/text
  and native scalar/vector/Python paths must move together.
- Historical repair boundary (superseded by the completed v7 study): no scientific
  seed had run. The prior direction was to freeze a common-input comparison after
  the transport passes; previous canaries are excluded. Persistent game memory
  remains separate. Jack Heart’s 03:01 UTC ship approval authorizes delivery of the bounded repair
  after gate; fresh defaults and scientific experiments remain unchanged.
- Task budget is$25total, separate from MTG-213's$10. Only paid rental is deleted;
  prior estimate$0.07083924, now provider-billed$0.07258259388618171.
  This continuation acquired no rental; the$6ancillary reserve remains. Three later availability/admission
  failures created no rental. CPU canary passed all3arms at22–27seconds/update;
  no training process remains. Preserve all evidence under.runs/representation.
- RunPod credentials are Doppler etude/prd, artifacts manabot/prd. Inject through
  nested Doppler commands; never copy/print keys. Task-local Etude state contains
  only this Task's allocations. No further rental is required for investigation.
- Older complete decisions and chronology are retained at
  8e6cc62df:scratch/questions.md and the design at the same commit. PR416's
  inherited notes remain at3b122de9078c985f846a0f3c5a3a42ebca3b1182:scratch/questions.md.

- New native counterexample: an Indefinite LoseKeywords(Flying), later
  Indefinite AddKeywords(Flying), and later EndOfTurn LoseKeywords still lack
  flying after cleanup. Existing statics.rs:1407 subtracts all lost_keywords
  after collecting grants at1478, ignoring grant/removal timestamp order.
  This predates the projection change. No semantic coalescing is shipped;
  records/timestamps remain intact. The new directed expiry/grant behavior
  check uses the existing correctly ordered LoseAllAbilities mechanism.
  A separate rules repair needs its own world/evidence assessment.
