# bb4160 unresolved decisions — 2026-10-28

The current accepted outcome, implementation and remaining work are in
`scratch/bb4160-evaluate-qwen-versus.md`.

- **World choice ACCEPTED (Jack Heart, 2026-10-28):** run the exact MTG-183
  checkpoint in a separately pinned **historical w13** experiment (w13
  source/native build, w13 decks, policy-input v1). Label all results as w13
  evidence. Do not weaken w14 admission or relabel the checkpoint. Paid
  evaluation within the $10 budget is unblocked by this decision.
- **Launch prerequisite resolved externally:** `~/.etude/launcher.db` is now
  schema 11, matching SDK 0.2.10 (backup `launcher.db.schema10-20261028`); no
  change made here.
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
Serving plan from Jack Heart's session (inference analysis of pinned Qwen3.5-9B config c2022362):

- Architecture: hybrid, only 8/32 layers full attention (4 KV heads x 256) => KV = 32 KB/token; 24 linear-attention layers carry ~50 MB fp32 state/seq. Weights ~18 GB bf16. Single-stream decode is weight-bandwidth bound (~40 tok/s ceiling on A6000; 34 measured) — no KV-cache bug. The pilot's slowness is ~5k tokens/decision x concurrency 1.
- Run the budget-forced cohort at concurrency 32 (existing `le=32` cap; `--max-num-seqs 32`). Memory: 32 x (~7k tokens x 32 KB + 50 MB) ~ 9 GB, fits easily. Decode stays weight-bound to ~batch 64, so aggregate should reach ~900-1200 tok/s.
- Enable the model's native MTP speculative decoding in vLLM 0.17.1 (config has mtp_num_hidden_layers=1). Verify the exact `--speculative-config` method name accepted by the pinned vLLM (e.g. `qwen3_5_mtp`/`mtp`, num_speculative_tokens 1-2) offline before renting; it preserves the sampling distribution and joins the hashed base argv, so it is part of the declared treatment. If unsupported in 0.17.1, record that and proceed without it — do not bump vLLM inside this budget without evidence.
- Pilot: a short concurrency-32 pilot (a few minutes of decisions is enough) measuring tok/s aggregate, accept rate, per-decision reasoning tokens and forced-close rate; then freeze the cohort. Budget estimate: ~400k tokens/game => ~$7 for 100 games without MTP, ~$4-5 with. Wall ~13-16 h on one pod; size the window/cap accordingly.
- Native `skip_trivial` ON for every Qwen game (standing rule). The campaign already set it via `experiments/qwen_mtg183.py`; the pilot did not pay for forced decisions. `GameConfig.skip_trivial` now defaults to True (Jack Heart: "we should basically never set skip_trivial=False").

## Sync resolution — 2026-10-28

Preserved this branch's bb4160 decisions. Main's MTG-193 questions and
assumptions remain at
`6d069d6e5f0755b392c796aaf6d754913e8996c2:scratch/questions.md`;
they are historical context for shared evaluation, not new bb4160 direction
or an additional rental budget.
