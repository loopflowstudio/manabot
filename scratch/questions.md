# MTG-245 — growing-pool protocol

Jack Heart's 2026-11-04 Task directive selects methodology and executable
protocol plumbing. Task 04 owns actual training/results; Task 08 owns the later
campaign. No training launch, rental or spending is authorized here.

The inherited MTG-213 notes are unrelated historical evidence, preserved at
`8e8d702f807d61675fc1bca46417eee69316daa0:scratch/questions.md`.
In particular its first-64 exception does not authorize truncation here.

## Implementation assumptions (2026-11-04, headless)

- After the initial implementation began, Jack Heart selected exploration A
  on 2026-11-04: “i agree A is the place to start”. Start with the reference
  league: authored functional decks, frozen mixtures and stable evaluation
  anchors. Decision and acceptance slices: `scratch/explore-design.md`; review
  page: `scratch/explore.html`. Exact mixture percentages and numeric pilot
  remain prospective engineering choices; no games, rental or spending approved.
- Use existing UniformDecks independent draws. Integer population tickets
  express a fixed mixture without a second sampler or native changes. Training
  decks, including opponent decks, exclude evaluation-only lists and cards.
- Freeze a runnable current-snapshot example, separate from Validation A's
  future integrated snapshot. Adventure/prepare and planeswalkers are future
  manifest revisions, not prerequisites or claims of present support.
- Strict capacity errors, native trivial skipping, fresh initialization and
  full world vocabulary. Changes to world/input require prospective rebinding;
  no reinterpretation of frozen checkpoints.

## Delete — do not maintain

Unrelated inherited MTG-213 ambient notes have been removed (reference above).
Duplicate Python budget defaults and independent evaluation-schedule formulas
are removed: the JSON supplies bounds, and `Budget.evaluation_rounds` owns the
contrasts used for scheduling, callback admission and planned game counts.
No remaining runtime deletion targets: this is an additive protocol. Reuse Deck,
UniformDecks, TrainingRegime, TrainingJob and EvaluationRequest as owners;
do not add a parallel training engine, card representation or admission gate.

## Implemented outcome and review boundaries

The executable contract is `experiments/protocols/growing-pool-245-v1.json`;
methodology, alternatives, two-batch example and Task 04 handoff are in the
adjacent `.md`. `experiments/growing_pool_protocol.py` compiles the authored
50/25/25 reference league into existing UniformDecks/TrainingRegime/TrainingJob.
`experiments/growing_pool.py` supplies canonical Etude discovery and five
separate paired panels, with missing export/game slots retained. Checkpoint
admission includes the exact training regime and raw/initial role, not just
world/architecture. The fixed heuristic source identity excludes manifest edits.

Nine current-world proxy lists, eight tickets, two seeds, 512 planned training
games and 800 planned evaluation games; proposed $5 inclusive ceiling. These
are engineering parameters for review. The 40-card Elemental proxy is not a
Standard metagame or Validation A's eventual integrated snapshot. No games,
training, rentals or spending occurred. Source commit publication remains with
the Graph's delivery step; job declaration rejects an old commit without this
protocol. No engine/model changes or generator were introduced.

Exposure: existing sealed card blobs retain actual opening/drawn copies;
`growing_pool_exposure.py` adds committed choices by participant/kind/action.
Per-card opportunities, effect resolutions and training per-card exposure remain
unavailable, explicitly distinguished from evaluation counts. Tasks 01/03 own
the broader capture proposal; no mechanic-coverage or learning claim follows
from these software checks. Future Adventure/prepare and walker scenarios and
complete lists must be authored at their actual native snapshot, not guessed.

Check (2026-11-04): `uv run --no-sync pytest tests/training/test_growing_pool_protocol.py tests/eval/test_experiment_files.py::test_etude_finds_each_current_experiment_and_nothing_else -q` → 26 passed; read-only protocol check admitted all nine lists; Ruff and diff whitespace passed. Gate owns the remaining regime/discovery suites named in
`docs/development.md`; demo owns review of proposed lists, shares and bounds.
Task 04 owns authorized real training/evaluation and the integrated-source
freeze. The shared native sampler's existing recovery/determinism tests remain
its coverage owner; the new tests prove exact ticket serialization and ordering.
