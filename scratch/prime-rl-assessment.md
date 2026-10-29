# MTG-209 — Prime RL findings retained for bb4160

Pinned research by the MTG-209 assessment, 2026-10-28; not a fresh upstream
audit. Jack Heart subsequently accepted the existing vLLM evaluation and relevant
behavior-version implementation in bb4160 under a fresh $10 ceiling. The current
plan is `scratch/bb4160-evaluate-qwen-versus.md`; the accepted matchup is historical
w13, with w14 admission preserved. MTG-191/192 remain canceled with closed budgets.
MTG-210's relevant scope is included here; MTG-211's controlled verifiers adapter
and the verifiers-versus-custom fine-tuning trial remain future work, without
training-rental authorization.

## Sources and implementation boundaries

Local source inspected: `c5d406e3d2a701525c15c8500a30f7da7896cc40`.
Prime RL: `6a32e72dc0d3e739cf1d676d8125fb7575125683`; verifiers gitlink:
`60290d436fcb4c2ad6dd7d0986bcf00ff2366031`; renderer pin:
`b6ab8c7d67229be6fb99d57a8395f978d94b0566`. This Prime uses `verifiers.v1`
(Taskset, Env, Agent, Episode, message-graph Trace) and vLLM 0.32.0, while the
campaign pins 0.17.1. Do not mix older MultiTurnEnv tutorials or migrate runtimes
implicitly. Exact renderer/template parity remains unverified.

The full source audit and primary-source links are preserved at
`02fccd93e1983319d355a1866c101d7d0e01565b:scratch/prime-rl-assessment.md`.
The detailed historical comparison and later disposition are preserved at
`5016ad1337c47f0854e0c0376b920c5d1bc5e1df:scratch/prime-rl-assessment.md`.
The assessment ran no benchmarks, tests, inference, rentals or training and
fetched upstream source read-only. Its historical metrics came from retained
reports; the raw MTG-185 corpus was absent. New claims require verified bytes.

## Adopted and deferred

- Keep manabot's synchronous structured-policy trainer. Borrow immutable behavior
  identity, acknowledgement, bounded handoffs and terminal accounting where the
  existing session needs them. Native producers and shared serving already exist;
  `sim/serving.py` and `PolicyKey` own game/actor/revision/ordinal binding and
  per-row sampling seeds. Installing Prime or adding mutable aliases is unnecessary.
- Prime records token policy versions, masks stale tokens (bound eight at the
  inspected revision) and acknowledges weight publication. Its default IPO and
  group-relative credit differ from actor/critic GAE and PPO/damped-policy rules.
  It also supports self-play RAE with per-agent baselines: lack of adversarial
  support is not a reason to reject it. Token ratios cannot repair stale GAE,
  changing opponents or per-seat credit. Training overlap needs a separate design.
- Prime's adaptive cancellation must not drop scored games by outcome. Its saved
  trainer/orchestrator state and repushed weights do not prove recovery of live
  native games, in-flight rollouts or the rental ledger.
- A future verifiers adapter should own one native game through existing capture,
  selection, Command and replay APIs, keeping native state outside token batches.
  Freeze all identities, seeds, seats, decks and limits; preserve viewer safety,
  complete choices, stale rejection, EOS and per-seat credit. The inspected Kuhn
  example's invalid-answer forfeits and accumulated history differ from our
  one-repair/marked-fallback contract. Fallback has no model logprob; token masks
  and restart semantics need design. First prove both seats and failure paths in
  a tiny controlled game with pinned Prime/verifiers/renderers and private monitors
  disabled. This is not an implemented adapter.

## Contrary evidence and qualification limits

Historical reports do not establish a matched backend winner or healthy reasoning:
MTG-185 `4c2a54fdfaac` completed one non-thinking L4 game with 160 initial-valid
selections, no repair/fallback and 308 replayed Commands (280.55 s game, 1.96 s
reported median). MTG-188 Transformers `b6a4716bb321` timed out on its second
reasoning selection. vLLM `12f050b52057` reported about 129 aggregate output
tokens/s at four requests versus 34 at one, but completed no games. Greedy short
output `b1d7ed75492f` completed/replayed 18 games with 519 fallbacks among 793
selections; 8 timed out and 74 were unattempted. Longer output `7b8c16de1b60`
completed none before cancellation; eight archives verified but the server-log
fetch failed integrity validation. These treatments and their budgets remain
separate from bb4160's sampled and budget-forced evidence.

The historical structured-policy A6000 comparison reported medians 27.21 s at
four concurrent games versus 68.75 s serially, with mean batches only 1.13 rows.
It supports bounded evaluation, not asynchronous learning or GPU saturation.

L4 sampled occupancy reached 22,494/23,034 MiB; A4500's 20 GB is not admitted by
that receipt, and A6000 measurements do not certify L4 fit. No matched L4/A4500
trial or fresh price comparison exists. Session readiness includes imports/IPC,
not isolated weight loading. Preserve historical median definitions. Prefix
reuse saves prefill, not long decode; proxy-token prefix counts are not Qwen cache
hit rates. Quantization, context changes, offload or grammars require declared
treatments preserving the full legal action space and native authority.

Retained Commands replay without proving identical regenerated text across
backends/batches. Hosted slugs are weaker identity than pinned checkpoints; a
behavior hash acknowledges the supervised declaration, not GPU weight bytes.
Inclusive costs must retain setup, warmup, failures, repairs, transfer and cleanup;
unknown usage/billing remains unknown. Neither replay nor token throughput alone
qualifies useful Qwen play or a backend promotion.
