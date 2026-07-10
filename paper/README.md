# The manabot paper

A standing snapshot of the paper we would write today: manabot as a **research
platform** for *Magic: The Gathering*, with calibrated baselines and a stated set
of open challenge areas for others to build on. Sections are expected to be thin;
the gaps are the roadmap.

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
  - `01-refuted-ledger.tex` — the first framing: a negative-results paper
    organized around the ledger of refuted claims. Superseded when the goal
    became "set up the problem space for new researchers," but its §7 survives
    as the calibration-findings section of the current draft.

## Structure of the current draft

1. Why *Magic* is hard, precisely, and how manabot differs from prior MTG
   search, representation, and benchmark work.
2. The platform — managym (engine, determinization, throughput, Milestone 1's
   two cube decks and the scenario-injection surface) and manabot (encoder,
   PPO, search, batched inference, distillation).
3. The measurement protocol — seat balance, Wilson intervals, the pass gate,
   estimated and confirmed **ladder strength**, the cost axis, pre-registration.
4. Reference baselines — the frontier table and figure, plus the second
   generation: `student_r0` (search-256 distillation) and the refuted
   expert-iteration crank.
5. **The control-competency suite** — five scenarios with known-correct lines;
   the behavioral benchmark and the acceptance test for the belief challenge.
6. **The five challenge areas** — the core. Each has a statement, current best,
   entry criterion, and a concrete next experiment.
7. Scope and limitations — card/rules coverage, opponents, comparison
   confounds, and the absence of deck construction.
8. Calibration findings — the fourteen refuted claims, reframed as what the
   domain will do to an entrant who skips the protocol.

## The five challenges

| # | Challenge | Current best | Entry criterion |
|---|---|---|---|
| 1 | Beat the search ladder | `student_r0`, ladder N≈7, ~$2.90 (current world) | ladder N ≥ 16, <10 ms/dec, <$5 |
| 2 | Model the belief state | none — uniform determinization | held-out NLL *and* matched-sim win rate *and* competency-suite report |
| 3 | Long-horizon credit assignment | terminal-only PPO, 75.5% vs random | ladder N ≥ 16 from terminal reward alone |
| 4 | Compositional generalization | none | zero-shot to held-out `cardsets/tla.rs` (26 cards) |
| 5 | Scale the rules substrate | 55 cards, 162 rules tests, 1.8e5 SPS | add a CR slice (replacement/prevention), keep suite + conformance green, hold 1e5 SPS |

## Where the numbers come from

Every figure traces to a report in `reports/`, which is the source of truth. When
a report lands, update the paper.

| Paper section | Report |
|---|---|
| Throughput (engine) | `sps-closeout.md` |
| Throughput (batched inference, MPS) | `exp-07-expert-iteration.md` |
| Cost axis | `exp-00-cost-basis.md` |
| Decision profile / horizon | `exp-00-decision-profile.md` |
| Seat contamination | `exp-00c-seat-balanced-baselines.md` |
| PPO from scratch | `exp-01-c1-training.md` |
| Search ladder | `exp-02-flat-mc.md` |
| Distillation | `exp-03-distillation.md` |
| Reward shaping | `exp-04-potential-shaping.md` |
| student_r0 / expert iteration | `exp-07-expert-iteration.md` |
| Two-deck matchup | `exp-08-two-deck-matchup.md` |
| Competency suite / micro-format | `exp-09-control-competency.md` |
| Conformance audit | `card-conformance-audit.md` |
| Milestone 1 | `wave/rules/01-two-deck-slice.md` |

Win-rate confidence intervals are Wilson 95%. Where a report published only a
lower bound, the interval in the paper was recomputed from raw
successes/trials and checked to reproduce that bound.

## Known gaps in the current draft

- No self-play, Elo, or human evaluation — everything is anchored on a uniform
  random opponent and on the `search-N` ladder. This is the largest hole: the
  ladder's pilot now provably fails the competency suite.
- Ladder strength is **unmeasured** for the terminal-only, potential-Φ, and
  `bc-fifth` policies. Those matchups are cheap and would sharpen Table 2.
- ~~The 4×Ancestral-Recall deck-quality probe (23.7% → 78.25%) cited in the
  calibration section has no standalone report yet~~ — resolved:
  `reports/exp-08b-ancestral-dose.md` (single-seed-batch caveat noted there).
- There is no controlled comparison with the contemporaneous MTG-Causal-RL
  benchmark; the paper currently positions the two systems from their reported
  interfaces and scopes.
- Deck construction — arguably the real game — is not discussed.
- The exploitability probe (an adversarially trained exploiter against each
  frozen policy) remains deferred; every ladder number is provisional
  against it.
