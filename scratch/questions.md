# MTG-213 — decisions and assumptions (2026-11-03)

Jack Heart's comment 1540e0ad-743f-412b-ad5c-5cb55b03a0db supersedes the deferred
Task brief: start preliminary fixed-corpus TLA self-play and evaluation now,
rejecting whole unsupported decks. Comment 6434b4dd-5329-4904-9952-6493c9968b52
authorizes GPUs under a hard $10 cumulative cap and shipping after checks,
superseding local-only/no-merge assumptions. Comment d2108241 requires a launch
milestone only after observed GPU learner updates, with job/GPU/progress/rate
and deadline evidence. Neither quotes nor the CPU proof satisfy it.

- Population: the CC BY 4.0 public TLA PremierDraft dump, because the trophy API
  restricts reuse to 17Lands.com. First 1,000 events with seven observed wins in
  source order; seventh-win main/sideboards and event/build/results retained.
  Missing losses stay unknown. Native whole-list admission yields 134 eligible,
  866 rejected, without resampling. This is historical trophy data, not today's
  feed or a uniform sample of drafts. Live fetching and deck holdout are deferred.
- Recipe: preserve MTG-181/183's 64x2/four-head damped policy; original world/full
  vocabulary/v5 and trophy population are declared deviations. No historical
  checkpoint or budget is inherited. The $1 first allocation, 100-update target,
  2,700-second watchdog and eight-pair/four-leg evaluation are engineering choices
  frozen in `experiments/protocols/tla-trophy-213-v1.{json,md}`.
- Credentials: Jack Heart corrected the prior blocker. Nested Doppler
  `manabot/prd` then `etude/prd` supplies artifacts then provider access without
  copying secrets. Configured GPU alternatives are authorized within $10;
  record hardware/allocation deviations before scoring. Earlier credential and
  zero-spend capacity reports are historical, not current launch instructions.
- Runtime: prepared b8835f63b matched the corpus. V2 explicitly selects
  post-PR-414 source plus the v6 repair, with the current card digest pinned in
  its protocol. All 134 entries admit unchanged. The digest difference is
  isolated to PR-414 keyword serialization and 13 rules-vocabulary locations;
  version/admission alone does not establish runtime parity. Historical v1 correction
  `2adbd0758e40518947daf91569e1938b060d4346` changed only the timing test from
  b8835f63b, retaining native/learner/corpus/recipe/evaluator bytes.
  Ordinary repair checkpoint: c29bf3a3. Keep original failed requests unchanged.
- Allocation: prepare source once through existing `source.freeze`, retain the
  typed Run, then quote/submit via `client.launch_cohort`; quotes reserve nothing.
  Etude 0.2.12 cannot append runs to a cohort, so the corrected attempt used
  `mtg213-pilot-v1-correction1` with a $1 ceiling. Operational cohort names do not
  create scientific samples. T6 learner optimization and MTG-239 launcher work
  retain separate ownership.
- Analysis: retain milestones even after an early endpoint and both comparison
  slots for an unknown endpoint. Failed callbacks count as failed matches;
  unknown games stay missing and attempted games are a lower bound. The companion
  analysis CLI owns the joint criterion; Etude curves are descriptive. Local CPU
  validation and initialization strength establish neither GPU learning nor pace.
- Capacity: stream 181 / seed 31482 / action 87 reproduces `facts 140 > 128`
  with matching frame/error in both encoders. Seven legal Fire Nation Drill
  activations over ten permanents produce two keyword-removal facts each.
  Encoding errors are deliberately fatal. No padding, truncation, quarantine,
  deck exclusion or unchanged paid retry resolves the shared representation
  decision. The 2026-11-03 implementation audit established 140 distinct tuples
  with seven distinct timestamps, zero exact duplicates, matching capture and
  70 rendered text records. Lossless compaction cannot preserve unchanged v5
  row meanings here. Operator comment 5d6cf81e selected prospective v6/max_facts=512;
  0c17ad58 was an operator-issued run reason, not separate human confirmation.
  New operator comment 78f4c226 selects **256 for the pilot**, following Jack
  Heart's question about 256. The 23:20 UTC authoritative launcher/local/R2
  reads found no v2 execution/scoring; the prepared v2 bindings were amended
  in place. Receipt: `.runs/mtg213/v2-pre-amendment-state.json`. The numeric capacity is the operator's engineering choice.
  Preserve timestamp rank / 128, all columns and v5.
  Fresh v6 configuration defaults to 512; legacy configuration omits the new
  field to preserve architecture hashes. Native buffers infer capacity from
  their fact axis; 128-row v6 and v5 have identical encoding semantics. Existing v5 evidence remains unchanged.

Current results, reservations and remaining work: `scratch/mtg-213.md`;
retained receipts: `experiments/tla-trophy-pilot.md`. Detailed decision chronology
is at `72d67d4cf69cb055df4314849e55abf7f7448556:scratch/questions.md`.
Inherited unrelated notes remain at
`22e4aab716f2b582d704b88dc5d22cb1547ce623:scratch/questions.md`; main's MTG-238
assumptions and engine-speed notes at
`44388ad4211b4292cc161a3087eb1230b2005737:scratch/questions.md`.

Comment be5b1394 relays Jack Heart's explicit confirmation of the immediate
larger-capacity hotfix; broader representation/history/memory design belongs
to a separate Task. The primary conversation owns
`docs/observation-architecture.html` and README diagrams; preserve those
contributions for delivery. They do not change model behavior.
