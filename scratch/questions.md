# MTG-213 — current decisions and assumptions (2026-11-05 UTC)

Execution and delivery status: `scratch/tla-pilot-postmerge-verify.md`.
The report `experiments/tla-trophy-pilot.md` owns results, costs and evidence links.
V10 has a trained endpoint and all 64 endpoint games; the 100-update target and
full scientific schedule remain incomplete. No new rental is selected. Preserve
all executed protocols and artifacts; report delivery remains with the caller.

## Accepted direction and ownership

Jack Heart's comment 1540e0ad selects preliminary fixed-corpus TLA self-play
and evaluation now, superseding the deferred Task brief. Comment 6434b4dd
authorizes GPUs under a hard USD10 cumulative cap and shipping enabling fixes
after checks. Comment d2108241 requires reporting actual GPU learner updates
with job/GPU/progress/rate/deadline evidence. V5 reached that milestone but
stopped before exporting a trained checkpoint; quotes and CPU proofs do not
satisfy it. On 2026-11-04 at about 22:00 UTC, Jack Heart accepted the newly
available Vast RTX5090 offer at about USD0.963/hour and roughly USD1.19 for
setup plus the planned run ("that seems fine"). This supersedes the USD1
per-allocation stop for this attempt; the operator uses USD1.25 as the rounded
whole-rental ceiling and USD0.97/hour as its offer ceiling. The cumulative USD10
cap remains unchanged, including failures, setup and teardown. This placement-only
amendment governed the executed v9 Run, preserving its source, recipe and schedule. The subsequently accepted resource-limit repair
was delivered in PR #426 and executed as v10; unchanged v9 retries are superseded.

Comment 42a12145 approved shared-effect fixed tensors, encoding each effect
once and routing it to affected object embeddings before attention. Task 5da5f4
delivered PR #419 and owns redundant-activation investigation and its separate
budget. T6/T7 keep their rules ownership. The primary conversation released
Etude 0.2.14; no competing representation repair or release is selected.

Comments f61beeb7 and 3b0f8826 require native legality/payability first, then an
explicit deterministic first-64 legal-action cutoff with replay-linked diagnostics.
Comment ffd29028 records Jack Heart's concept-review answer **1b**: apply the
same prefix to heuristic, learned candidate and initialization anchor. V9 and
merged PR #423 implement it, superseding v8's full-support heuristic interpretation
and v6/v7's 512-action plans. Experiments Wave memory records this matched-support
decision (the Training Wave was renamed in `9f42e4e58`). Prepared v6/v7/v8 Runs must not launch.

The engine and Academy retain all legal choices. Retained row i executes original
native Command i; heuristic restriction precedes scoring/fallback. Padding is
masked, with no pagination. Later legal actions have zero probability on overflow;
ordering neutrality and strategic redundancy are not established. Diagnostics
count consumed decisions across all participating policies, not learner-only or
actor-specific rates. Samples cannot reconstruct exact actor totals. Normal close
flushes totals; abrupt termination can lose counts since the last flush. A cutoff
is not a failed game; a rules error remains a rules error.

## Scientific contract and evidence limits

- Population: CC BY 4.0 public TLA PremierDraft dump; the trophy API restricts
  reuse to 17Lands.com. First 1,000 events with seven observed wins in source
  order, retaining seventh-win main/sideboards and event/build/results. Missing
  losses remain unknown. Whole-list admission yielded 134 eligible and 866
  rejected without resampling or substitutions. Each seat draws independently,
  including mirrors. Live fetching and held-out-deck generalization are deferred.
  A later 306-list admission audit does not expand the frozen 134.
- Recipe: MTG-181/183's 64x2/four-head damped policy, with explicit new-world,
  full-vocabulary and trophy-population deviations; no checkpoint or budget
  inheritance. The 100-update target, 2,700-second watchdog and eight-pair/four-leg
  evaluation are frozen engineering choices. V9 retains input-v7 shared effects,
  128 cards per seat and the original recipe, seeds and schedule.
- Compatibility: executed protocols/configurations/evidence are immutable.
  Source/world/shape changes need prospective identities and fresh initialization
  and evaluation; a common w17 label does not establish compatibility. The report
  retains the 140/270-fact, 61-card and illegal-X failures and repair decisions;
  these repairs do not guarantee universal reachable-state capacity. Frozen input
  v5 stays at 128 facts; fresh v6 defaults to 512, with rank / 128 timestamp scaling.
  Synthetic world fixtures never authorize real execution.
- Analysis: retain milestones after early endpoints and both comparison slots
  for unknown endpoints. Failed callbacks count as failed matches; unknown games
  remain missing and attempted games are a lower bound. The companion analysis
  CLI owns the joint criterion; Etude curves are descriptive. New games cannot
  fill historical missing slots or reconstruct lost optimizer state from counters.
  GPU execution alone supplies no learning-gain or Elemental-relative pace claim.

## Operational choices

Nested Doppler `manabot/prd` then `etude/prd` supplies artifacts and provider
credentials without copying secrets. Use the published isolated Etude 0.2.14
control client; frozen workers retain 0.2.12. Freeze source once through
`source.freeze`, retain the typed Run, and quote/submit through
`client.launch_cohort`. Etude 0.2.12 could not append cohort runs, so the historical
`mtg213-pilot-v1-correction1` used a separate USD1 ceiling without creating a new
scientific sample. Neither that workaround nor another Task replenishes USD10.

Configured GPU alternatives are authorized within the existing caps; record
hardware/allocation deviations before scoring and retain CUDA numerical admission.
Quotes reserve nothing. Reconcile fresh inventory, bills and whole-rental quotes
before allocation; per-run cost can omit setup/idle cleanup. Never change or charge
another Task's rental. VAST_API_KEY remains in `etude/prd`; the SSH public key was
registered. Prior Vast placement used >=6 vCPUs, >=48GB RAM, 80GB disk,
<=USD0.81/hour and 840 seconds setup plus the unchanged one-hour job.

V10 retained permanent64 and game-local capacity recovery at .25 with the
existing 100/100 window/warmup. Comment ca1660e8 authorizes dropping only the
failed game's uncommitted samples from both seats. Strictly above .25 stops;
other streams and completed updates survive. Evaluation keeps failed slots.
Comments 52d6354e/173f02a6/c98cb420 leave 256/512 capacities, grouping,
variable-length inputs and GPU transfer changes exploratory. Do not broaden the
recipe to address rare growth without a separate selection. Retain overflow
frequency, affected decks and positions for that decision.

## Historical context

Completed review drafts, proof requests and CI repair narration no longer direct
work. Their full pre-compression text is retained at
`6da60684cb628ec9b431dbbb9d7dc91348743622:scratch/questions.md`.
The complete local notes before this reduction, including both placement
assumptions, are preserved as git blob `0b77c45bf972280f57b4a9bb882e945afc27120d`.
Older decisions/source pins remain at
`af4227616ef2cf46bb1f0b05e690feebb28ac2e8:scratch/questions.md`.
Inherited T6/T7 notes remain at
`5cd6ed3f6e54e81f9f39c5c2bb56f322dc1576c1:scratch/questions.md` and
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`;
they do not replace MTG-213's direction. Cleared demo/verification plans remain
at `d2ff60e6a^:scratch/tla-pilot-demo.md` and
`d2ff60e6a^:scratch/tla-pilot-verify.md`.

The complete decisions before this reduction, including A4000/A6000 placement
assumptions and v9's identity-preserving Job.max_usd amendment, remain in git blob
`79baeb35620eddc7eea26f3978254d2e4b5d2548`.
