# Manabot: results, then methods

The active paper is an internal research note for readers who already know RL.
The October 26, 2026 revision leads with repeated ten-deck learning, the controlled
seat comparison, specialist/generalist results, and the stronger heuristic. It
also separates the single-game LLM integration proof from playing-strength
claims. Both the active and first archived draft subtitles say:

> Written by AI. Not yet intended for distribution.

## Reading and building

```bash
open paper/manabot.pdf
brew install tectonic   # one-time; fetches TeX packages on demand
make -C paper          # -> paper/manabot.pdf
```

`make -C paper watch` rebuilds on save.

## Structure and evidence boundary

1. **What we have demonstrated** (2–4 pages): measured results and failures,
   preserving worlds, recipes, seeds, opponents, cost bases and uncertainty.
2. **Methods as recipes**: current model inputs, collection, learning, search,
   text-agent selection and evaluation, with code owners and cited influences.

Methods describe source `558e0186b` (October 26). The measured Elemental policies
use w13, additive heads and their saved input contracts; they do not certify the
current FiLM/policy-input-v2 recipe. Heuristic v1 and the strengthened heuristic
are separate anchors. Wilson game intervals do not measure seed uncertainty or
resample shared deals. MTG-183's prospective protocol is not a completed
long-training/saturation result in this snapshot.

## Evidence map

The three artifact-store reports were restored from the local shared cache and
verified against the committed manifest's SHA-256 entries. Fetch them with:

```bash
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg181/report.md experiments/data/seat-comparison-v1/report.md experiments/data/mtg186/report.md
```

These bytes are intentionally outside git. The paper links their public
protocols and the [digest manifest](../experiments/data.manifest.json); each
protocol points to its retained report. Broader evidence can be fetched by
passing the corresponding directory instead of `report.md`.

| Topic | Retained report and public entry point |
| --- | --- |
| Ten-deck learning, three seeds, old/new heuristic endpoints | `experiments/data/mtg181/report.md`; [MTG-181 protocol](../experiments/protocols/elemental-181.md) |
| Four-seed one-seat/two-seat comparison | `experiments/data/seat-comparison-v1/report.md`; [seat protocol](../experiments/protocols/seat-comparison-v1.md) |
| UR/GW specialist, direct matches and card associations | `experiments/data/mtg186/report.md`; [MTG-186 protocol](../experiments/protocols/mtg186.md) |
| Failed first Elemental training attempt | [MTG-174 result](../experiments/elemental-uniform-baseline.md) |
| Heuristic revision and expensive search gain | [Heuristic strength](../experiments/heuristic-strength.md) |
| Qwen single-game L4 proof, timing and memory limits | [LLM harness](../docs/llm-harness.md#cached-full-game-gpu-proof) |
| Longer Elemental run: protocol, no final result claimed | [MTG-183 protocol](../experiments/protocols/elemental-183.md) |
| Earlier batching comparison | [Longer learning 017](../experiments/longer-learning-017.md) |
| Unresolved floor/pooling and critic-coverage screens | [Pooling follow-up](../experiments/pooling-filter-followup.md), [technique screen](../experiments/ataraxos-technique-screen.md) |
| Historical reset-seeding correction | [World-growth investigation](../experiments/exp-06-newworld-training.md), [corrections F1–F3](understanding.md) |

## Implementation map

- [ActorCritic](../manabot/model/actor_critic.py) and
  [continuation input](../manabot/model/continuation.py): model and versioned context.
- [Collection](../manabot/training/rollout/collection.py),
  [returns](../manabot/training/learning/returns.py) and
  [damped policy learning](../manabot/training/learning/damped_policy.py):
  seat trajectories, targets and update ownership.
- [Ordinary baseline](../manabot/recipes/baseline.py) and
  [Elemental recipe](../experiments/regimes/elemental-two-seat-181-v1.json):
  current defaults versus the frozen experiment.
- [Ataraxos fidelity](../docs/ataraxos.md),
  [local search](../docs/local-policy-search.md) and
  [LLM selection](../docs/llm-harness.md): influences, adaptations and limits.

## Files and history

- `main.tex` / `manabot.pdf`: active results-first paper and built output.
- `refs.bib`: cited influences and older entries retained for the archives.
- `drafts/02-platform-and-challenges.tex` / `.pdf`: July platform draft;
  its extreme random-play seat claims were later retracted.
- `drafts/01-refuted-ledger.tex`: historical negative-results draft.
- `understanding.md`: historical findings and corrections, not a live queue.

Frozen reports and archives retain their original claims and corrections.
