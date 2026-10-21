# The Manabot working report

`main.tex` is the current draft (October 2026): a working report on training
*Magic: The Gathering* agents by self-play. It says what was built, what was
tried, what worked and what did not. Part I is a short account of what we have
demonstrated, owning the failures; Part II shares the recipe with honest
citation of the work behind each component; Appendix A maps game-RL
difficulties to the literature.

The July 2026 draft, written when the project was called Etude Fantasia and
leaned on search and distillation, is kept unchanged as
`drafts/02-platform-and-challenges.tex` (with its PDF). Its search-era tables,
challenge set and calibration ledger remain the detailed record of that period.

> **Charter (2026-07-10, still in force):** this document serves *internal
> understanding first*. Its job is that the owner can state, for every result:
> what was measured, why the mechanism explanation is believed, and what would
> change our mind. Publication is a later fork of a subset.

## Reading it

```bash
open paper/manabot.pdf
```

## Building it

```bash
brew install tectonic   # one-time; fetches TeX packages on demand, no TeXLive
cd paper && make        # -> paper/manabot.pdf
make watch              # rebuild on save
```

## Files

- `main.tex` — the paper. Figures are inline pgfplots; no external image files.
- `refs.bib` — bibliography.
- `drafts/` — superseded drafts, kept for the record.
  - `02-platform-and-challenges.tex` / `.pdf` — the July 2026 platform and
    challenge-set draft (search era).
  - `01-refuted-ledger.tex` — the first framing: a negative-results paper
    organized around the ledger of refuted claims. Superseded when the goal
    became "set up the problem space for new researchers," but its §7 survives
    as the calibration-findings section of the current draft.

## Structure of the current draft

The title page says "Written by AI. Not yet intended for distribution."

- **Part I — What we have demonstrated** (short): status, what works, what
  failed and what it taught us, what is not yet demonstrated, and how the
  approach evolved.
- **Part II — Methods: the recipe and its influences.** Assumes the reader
  knows why game RL is hard. Opens with a component → what we do → influences
  → code table, then one section per component (rules engine, network,
  collection, update, beliefs and local search, evaluation), with departures
  from each source stated.
- **Appendix A** — the problem-first map: difficulty → how it showed up →
  techniques (citations) → Manabot now.

## Where the numbers come from

Every figure traces to a report in `experiments/`, which is the source of
truth; `manabot/RESEARCH.md` summarizes current findings with links. When a
report lands, update the paper.

| Paper section | Report |
|---|---|
| Engine path and throughput | `sps-closeout.md`, `exp-07-expert-iteration.md` |
| Where training time goes | `training-time.md` |
| Self-play cohort | `current-baseline.md` |
| Batching and model size | `longer-learning-017.md` |
| Opponent effect | `mirror-curriculum.md` |
| Technique screen | `ataraxos-technique-screen.md` |
| Learnability | `learnability-search.md` |
| Ataraxos recipe fidelity | `docs/ataraxos.md` |
| Search era | `exp-02`, `exp-03`, `exp-04`, `exp-07`, `exp-08b`, `exp-09`, `exp-11` |
| Fusion test design (unrun) | `wave/intelligence/04-2x2-diagnosis.md` |

## Known gaps in the current draft

- **Citations to verify.** Bibliography entries added in October 2026 carry a
  `note = {verify}` where details were written from memory. The Ataraxos entry
  has a placeholder title and author list.
- The "energy reserve" wording should be checked against the Ataraxos paper.
- The MMD comparison with the paper's α and β assumes comparable value scales;
  this is unchecked.
- The held-out comparison for the self-play cohort is pending.
- No human evaluation; scripted anchors are saturating.
- The strategy-fusion 2×2 test is designed but has not been run.
