"""Claude Haiku against MTG-183's final checkpoint over the Elemental decks.

Not runnable yet. Etude refuses a seated checkpoint that no training run of
the experiment stored (`checkpoint_not_stored`), and an `EvalJob` has nowhere
to name an API key or an API dollar limit apart from its rental's; Etude
EXP-70 ("Evaluate players served by an inference server, and players with no
checkpoint") and the Manabot Task "Run the Claude and Qwen evaluation
experiments on Etude" carry the rest. The seated checkpoint is also bound to
world w13 and does not load in the current world.

An evaluation with no training: `eval_jobs` seats a hosted language model and
one frozen checkpoint in every ordered pairing of the ten Elemental decks, one
game each, the model's seat alternating. `play` is one game through
`manabot.verify.llm_match`; the model answers through OpenRouter under
`configs/llm/openrouter-haiku-high.json`.

```bash
uv run --extra openrouter etude eval experiments/llm_claude.py
uv run etude report experiments/llm_claude.py --serve
```

API spending is admitted by a `BudgetLedger` beside the games' output: every
request reserves its bound before it is sent, and a game that cannot reserve
fails unscored. The ledger's ceiling is USD 10, about 27 games at the USD 0.37
a game measured in `experiments/reports/haiku-mtg183.md`; the remaining games
fail admission. `OPENROUTER_API_KEY` comes from the environment and is never
written.
"""

from decimal import Decimal
import os
from pathlib import Path

from etude.checkpoints import CheckpointRef, checkpoint_id
from etude.eval.match import environment
from etude.eval.records import (
    Benchmark,
    CheckpointPlayer,
    Code,
    Inference,
    Match,
    Seating,
)
from etude.eval.run import Result
from etude.jobs import EvalJob, ExperimentState
from etude.records import ArtifactRef
from etude.report import Matches, Matrix, Ranking, Report, Score

from experiments.elemental_grid import DECKS, PACK
from manabot.llm.budget import BudgetConfig, BudgetLedger
from manabot.llm.openrouter import OpenRouterConfig, OpenRouterSession
from manabot.verify.llm_match import (
    Game,
    checkpoint_opponent,
    play_game,
    schedule,
    seated,
)
import managym

ROOT = Path(__file__).resolve().parents[1]
TOURNAMENT = "llm-claude-haiku-v-mtg183"
INFERENCE = ROOT / "configs/llm/openrouter-haiku-high.json"
GAME_SECONDS = 3600.0

# MTG-183 run f3748bcabac8, final raw update 5950.
MTG183_FINAL = (
    "r2://manabot/jobs/f3748bcabac8/checkpoints/learner_transitions-331180341.pt"
)
MTG183_SHA256 = "4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4"
MTG183_BYTES = 1071166

BUDGET = BudgetConfig(
    ceiling_usd=Decimal(10),
    # Measured requests averaged USD 0.0011; the bound leaves room for the
    # longest late-game prompt and a full reasoning answer.
    request_bound_usd=Decimal("0.05"),
    overhead_usd=Decimal(0),
    charge_multiplier=Decimal(1),
    bound_evidence="experiments/reports/haiku-mtg183.md",
)


def players() -> tuple[Code, CheckpointPlayer]:
    """The hosted model, by its OpenRouter config, and the frozen checkpoint."""
    inference = OpenRouterConfig.model_validate_json(INFERENCE.read_text())
    llm = Code(
        name=inference.model_id,
        source=[ROOT / "manabot/llm", ROOT / "manabot/verify/llm_game.py"],
        config=inference,
    )
    checkpoint = CheckpointPlayer(
        name="mtg183-u5950",
        checkpoint=CheckpointRef(
            key=checkpoint_id(address=MTG183_FINAL),
            artifact=ArtifactRef(
                uri=MTG183_FINAL, sha256=MTG183_SHA256, size_bytes=MTG183_BYTES
            ),
        ),
        inference=Inference(source=[ROOT / "manabot/verify/llm_game.py"]),
    )
    return llm, checkpoint


def output() -> Path:
    """Where the ledger and checkpoint cache live: a launched attempt's output,
    else `.runs/llm_claude`."""
    return Path(os.environ.get("ETUDE_OUTPUT_DIR", ROOT / ".runs/llm_claude"))


def play(match: Match[Game]) -> Result:
    """One game: Claude through OpenRouter against the checkpoint on the CPU."""
    _, llm, checkpoint = seated(match)
    opponent = checkpoint_opponent(checkpoint, output() / "checkpoints")
    ledger = BudgetLedger(output() / "budget.sqlite", BUDGET)
    ledger.check_admission()
    inference = OpenRouterConfig.model_validate(llm.config)
    with OpenRouterSession(inference, budget=ledger, budget_owner=match.key) as session:
        return play_game(match, session, opponent)


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """Every ordered deck pairing once: 100 games, Claude on the play in half."""
    games = schedule(
        PACK,
        DECKS,
        deals=1,
        deal_base=213183000,
        opponent_seed_base=213595000,
        max_seconds=GAME_SECONDS,
    )
    tournament = Benchmark(
        name=TOURNAMENT,
        environment=environment(
            "elemental-llm",
            files=[ROOT / "experiments/elemental_grid.py"],
            config={"world": managym.WORLD_VERSION, "pack": PACK},
        ),
        seatings=(
            Seating(name="claude-v-mtg183", players=players(), configs=games, games=1),
        ),
    )
    return [EvalJob(play=play, tournament=tournament, max_hours=12)]


def report() -> Report:
    """Claude's score by the deck it piloted and the deck it faced."""
    played = Matches(tournament=TOURNAMENT)
    return Report(
        score=Score(table="games", value="score", reference=0.5, unit="games"),
        evals=(
            Matrix(
                title="Claude deck × checkpoint deck",
                rows="opponent_deck",
                columns="deck",
                matches=played,
            ),
            Ranking(title="By Claude deck", by="deck", matches=played),
            Ranking(title="By checkpoint deck", by="opponent_deck", matches=played),
        ),
    )
