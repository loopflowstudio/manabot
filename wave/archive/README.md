# Archived waves

Collapsed 2026-07-09 (WIP = 1 for a solo project). Live waves: **search**
(active — the experiment loop) and **rules** (dormant, pull-driven). Everything
here is preserved as reference; the design thinking in these docs is absorbed
as follows.

| Wave | Disposition |
| --- | --- |
| `first-light` | Closed. Superseded by search; findings 1-2 caveated (derived on a no-interaction deck), findings 3-4 (stochastic eval, causal-chain metrics) carried forward. |
| `self-play` | Absorbed into search C5+ (expert iteration / population play). `01-opponent-pool.md` remains the reference design for the checkpoint ladder; Elo/evaluation thinking feeds the ladder-strength metric. |
| `scale` | `01-throughput.md` became search goal 1 (batched inference, pulled at C5, not pushed). `02-game-theory.md` became the pre-registered pivots: Exit 1 (belief-based) and Exit 2 (model-free game-theoretic). |
| `architecture` | Dissolved into the loop: architecture changes are experiments (question, prediction, cost cap, report) — not a standing front. |
| `game` (formerly `gui`) | **Redesigned 2026-07-15** — active under `wave/game/`: creator-authored human-vs-agent play with a versioned experience protocol, semantic presentation, recovery, curated assets, and portable delivery. The 2026-07-09 instrument-only charter is preserved as history. |
| `publishing` | Replaced by the release bar in `wave/intelligence/01-experiment-loop.md`: promotion rides with the first citable result, not with completeness. |
