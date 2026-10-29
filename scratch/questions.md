# bb4160 unresolved decisions — 2026-10-28

The current accepted outcome, implementation and remaining work are in
`scratch/bb4160-evaluate-qwen-versus.md`.

- **World choice ACCEPTED (Jack Heart, 2026-10-28):** run the exact MTG-183
  checkpoint in a separately pinned **historical w13** experiment (w13
  source/native build, w13 decks, policy-input v1). Label all results as w13
  evidence. Do not weaken w14 admission or relabel the checkpoint. Paid
  evaluation within the $10 budget is unblocked by this decision.
- **Treatment choice (2026-10-28, reversible):** Qwen's model-card thinking
  sampling (T=1.0, top-p 0.95, top-k 20, presence 1.5) under the accepted 16k/
  65k/1800 s limits, after both greedy treatments failed by repetition/limits.
  Declared in `experiments/protocols/qwen-mtg183-bb4160.md`; not pooled with
  MTG-188 greedy results.
- **Memory gap:** `.lf/wave/training/` and its child memories are absent. The
  supplied chapter is the available objective; no other Wave is substituted.
- **Service identity implementation choice, 2026-10-28:** derive the sole vLLM
  served name from the existing behavior hash. Its base argv excludes that
  derived argument to avoid a circular hash; the complete command is retained
  separately. Model-list acknowledgement and completion model checks bind the
  supervised service declaration, not independently measured GPU weight bytes.
  This world-independent work does not settle the matchup decision.

Original checkpoint evidence and the full directive are preserved in
`02fccd93e1983319d355a1866c101d7d0e01565b:scratch/bb4160-evaluate-qwen-versus.md`
and `02fccd93e1983319d355a1866c101d7d0e01565b:scratch/questions.md`.
Obsolete MTG-207 notes remain at
`c5d406e3d2a701525c15c8500a30f7da7896cc40:scratch/questions.md`.

- **Treatment change ACCEPTED (Jack Heart, 2026-10-28 ~19:00 PT):** the 16k
  sampled-thinking pilot `dddaf3001dfa` ran ~160 s per Qwen decision (~35/160
  decisions after 93 min, ~$1) and would exhaust its $2 cap without a game. Stop
  it, reconcile its charge, and spend the remaining budget on two separately
  declared, never-pooled treatments: (1) budget-forced thinking (~3k thinking
  tokens, then insert `</think>` and let Qwen answer; the forced close is
  recorded per decision, not counted as fallback) at ~8 concurrent games;
  (2) a no-thinking arm as a cheap labeled baseline. Short pilot each before a
  frozen cohort. Report reasoning length by decision kind to show where tokens go.
  Amended (Jack Heart, same evening): budget-forced thinking is the main arm and
  gets nearly all of the budget. The no-thinking arm is only a small labeled
  floor (~10 games, well under $1), run after the reasoning cohort is under way;
  it is a comparison point, not a candidate for the headline result.
Accepted serving direction from Jack Heart's session (2026-10-28; inference
analysis of pinned Qwen3.5-9B config c2022362), with subsequent evidence limits:

- Architecture: hybrid, only 8/32 layers full attention (4 KV heads x 256) => KV = 32 KB/token; 24 linear-attention layers carry ~50 MB fp32 state/seq. Weights ~18 GB bf16. Single-stream decode is weight-bandwidth bound (~40 tok/s ceiling on A6000; 34 measured) — no KV-cache bug. The pilot's slowness is ~5k tokens/decision x concurrency 1.
- Run the budget-forced cohort at concurrency 32 (existing `le=32` cap; `--max-num-seqs 32`). Memory: 32 x (~7k tokens x 32 KB + 50 MB) ~ 9 GB, fits easily. Decode stays weight-bound to ~batch 64, so aggregate should reach ~900-1200 tok/s.
- MTP was accepted subject to pinned-vLLM support and a declared hashed argv.
  The `mtp` method passed the pilot but crashed the scored cohort's mixed
  speculative/plain decode batch. The replacement cohort disables it; no vLLM
  upgrade is accepted. Pilot success did not qualify full-game serving.
- The accepted short concurrency-32 pilot measured 650–710 aggregate tokens/s
  with MTP; its early-game prompts did not establish late-game context fit.
  The no-MTP cohort remains incomplete. Earlier 100-game cost/time projections
  are estimates, not an affordable remaining commitment or a completed outcome.
- Native `skip_trivial` ON for every Qwen game (standing rule). The campaign already set it via `experiments/qwen_mtg183.py`; the pilot did not pay for forced decisions. `GameConfig.skip_trivial` now defaults to True (Jack Heart: "we should basically never set skip_trivial=False").

Open execution constraints, reconciled 2026-10-29 06:20 UTC: the no-MTP cohort
is still settling (49 admitted, 24 completed, 2 failed, 23 pending in logs;
final outputs unavailable). Preserve the frozen settle-admitted-games rule.
Direct provider bills for the three ended attempts total $1.47606228455 and all
three pods are absent. Retained launcher estimates understate two attempts.
Current $7 commitment plus $0.50 reserve leaves $1.02393771545, below the no-thinking
constructor's $1.05 minimum; no floor commitment is affordable before settlement.
The current pod, final billing and ancillary costs remain unsettled. `budget-64k`
is unlaunched. Full details and published receipt paths are in the protocol.

Reversible implementation choice (2026-10-29): use bounded supervisor threads
with one `ChildProcess` per admitted game, preserving the outer campaign process
group and watchdog for descendant cleanup. Resident and one-shot inference use
the same child owner. Frozen running source is unchanged. Historical close errors
can leave exactly one failed terminal selection outside the summary counter;
the offline reader recognizes that case without rewriting evidence or inferring
successful play. No new paid attempt was launched.

## Sync resolution — 2026-10-28

Preserved this branch's bb4160 decisions. Main's MTG-193 questions and
assumptions remain at
`6d069d6e5f0755b392c796aaf6d754913e8996c2:scratch/questions.md`;
they are historical context for shared evaluation, not new bb4160 direction
or an additional rental budget.
