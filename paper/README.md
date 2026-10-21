# Manabot: results, then methods

The active paper is a dated internal research note for readers who already know
RL: what Manabot has demonstrated, where it failed, and how the current recipes
work. Both the active and archived draft subtitles say:

> Written by AI. Not yet intended for distribution.

## Reading and building

```bash
open paper/manabot.pdf
brew install tectonic   # one-time; fetches TeX packages on demand
make -C paper          # -> paper/manabot.pdf
```

`make -C paper watch` rebuilds on save.

## Structure

1. **What we have demonstrated** (2–4 pages): measured results and failures,
   each linked to its retained report with its original world, recipe, budget,
   opponents and uncertainty. Earlier results do not certify w9 or FiLM.
2. **Methods as recipes**: the current model, collection, learning, search and
   evaluation paths, with code owners, cited influences and local adaptations.

The paper describes source at `a04dcdc2`. It is selective rather than an
exhaustive experiment ledger or a claim of strong human play. The
[research program](../manabot/RESEARCH.md), [architecture](../docs/ARCHITECTURE.md)
and individual reports hold broader context. Frozen experiment evidence remains
unchanged.

## Evidence and implementation map

| Topic | Retained source |
| --- | --- |
| Batching, model size, cost and qualification rule | [Longer learning 017](../experiments/longer-learning-017.md) |
| Pooling and advantage floor | [Paired follow-up](../experiments/pooling-filter-followup.md) |
| Critic coverage/filter scope | [Technique screen](../experiments/ataraxos-technique-screen.md) |
| Failed short full-game improvement criterion | [Current-baseline debugging results](../experiments/current-baseline.md#retained-debugging-results) |
| Reset seeding and retracted seat claims | [World-growth investigation](../experiments/exp-06-newworld-training.md), [understanding F1–F3](understanding.md) |
| Value-search gate and narrow training deals | [Value gate](../experiments/exp-10-value-gate.md) |
| Failed expert-iteration crank / control behavior | [Expert iteration](../experiments/exp-07-expert-iteration.md), [competency suite](../experiments/exp-09-control-competency.md) |
| Current model and ordinary baseline | [Agent](../manabot/model/agent.py), [baseline recipe](../manabot/training/baseline.py) |
| Ataraxos fidelity and departures | [Fidelity guide](../docs/ataraxos.md), [objective](../manabot/training/ataraxos.py) |
| Search and distillation contracts | [Local policy search](../docs/local-policy-search.md) |

## Files and history

- `main.tex` — active results-first paper; report links pin its source base.
- `manabot.pdf` — built active paper.
- `refs.bib` — cited influences and older entries retained for the archives.
- `drafts/02-platform-and-challenges.tex` / `.pdf` — preserved July platform
  and challenge-set draft, also available at `a04dcdc2:paper/main.tex`.
  Its extreme random-play seat claims were later retracted.
- `drafts/01-refuted-ledger.tex` — historical negative-results draft with the
  distribution subtitle above.
- `understanding.md` — historical findings and corrections, including the seeding
  bug. It is not a second current paper or the live work queue.

Archived claims are historical evidence, not current baselines.
