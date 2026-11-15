"""Qwen3.5-9B on one vLLM server against MTG-183's final checkpoint.

Not runnable yet. An `EvalJob` cannot hold a resource for its lifetime, so
nothing here starts the vLLM server its games share, nor give a game its own
time limit; Etude EXP-70 ("Evaluate players served by an inference server, and
players with no checkpoint") and the Manabot Task "Run the Claude and Qwen
evaluation experiments on Etude" carry the rest. The seated checkpoint, an
outside address, is also bound to world w13 and does not load in the current
world.

An evaluation with no training: `eval_jobs` seats the served model and one
frozen checkpoint in every ordered pairing of the ten Elemental decks, one game
each, the model's seat alternating. `play` is one game through
`manabot.verify.llm_match`. Its worker is a client: `VLLMRuntime` sends exact
token prompts to the one server on the machine's GPU
(`manabot.llm.vllm_server.VLLMServer`), which must already be serving
`INFERENCE` and have written its identity under `output()`. `WORKERS` games
run at once and share that server; each checkpoint plays on the CPU.

```bash
uv run etude launch experiments/llm_qwen.py --rent l4
uv run etude report experiments/llm_qwen.py --source r2://manabot/jobs --serve
```

Setup installs the game environment, the pinned vLLM environment
(`configs/llm/vllm-0.17.1-linux.lock`) and the model snapshot.
"""

from functools import partial
import os
from pathlib import Path

from etude.checkpoints import CheckpointRef
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
from etude.spec import Variable

from experiments.elemental_grid import DECKS, PACK
from manabot.backends import resolve
from manabot.llm.schema import InferenceConfig
from manabot.llm.session import InferenceSession
from manabot.llm.vllm_runtime import VLLMRuntime
from manabot.verify.llm_match import (
    Game,
    checkpoint_opponent,
    play_game,
    schedule,
    seated,
)
import managym

ROOT = Path(__file__).resolve().parents[1]
TOURNAMENT = "llm-qwen-v-mtg183"
# Sampled reasoning, closed at 3,072 tokens before the answer.
INFERENCE = ROOT / "configs/llm/qwen3.5-9b-reasoning-budget.json"
# Games at once, which is also the server's request concurrency.
WORKERS = 8
# A long game waits on its neighbours' generations; ten hours bounds the pass.
GAME_SECONDS = 10 * 3600.0

# MTG-183 run f3748bcabac8, final raw update 5950.
MTG183_FINAL = (
    "r2://manabot/jobs/f3748bcabac8/checkpoints/learner_transitions-331180341.pt"
)
MTG183_SHA256 = "4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4"
MTG183_BYTES = 1071166


def players() -> tuple[Code, CheckpointPlayer]:
    """The served model, by its inference config, and the frozen checkpoint."""
    inference = InferenceConfig.model_validate_json(INFERENCE.read_text())
    llm = Code(
        name=inference.model_id,
        source=[ROOT / "manabot/llm", ROOT / "manabot/verify/llm_game.py"],
        config=inference,
    )
    checkpoint = CheckpointPlayer(
        name="mtg183-u5950",
        checkpoint=CheckpointRef(
            address=MTG183_FINAL,
            artifact=ArtifactRef(
                uri=MTG183_FINAL, sha256=MTG183_SHA256, size_bytes=MTG183_BYTES
            ),
        ),
        inference=Inference(source=[ROOT / "manabot/verify/llm_game.py"]),
    )
    return llm, checkpoint


def output() -> Path:
    """Where the server's identity and the checkpoint cache live: a launched
    attempt's output, else `.runs/llm_qwen`."""
    return Path(os.environ.get("ETUDE_OUTPUT_DIR", ROOT / ".runs/llm_qwen"))


def play(match: Match[Game]) -> Result:
    """One game: a client of the shared server against the checkpoint on the CPU."""
    _, llm, checkpoint = seated(match)
    opponent = checkpoint_opponent(checkpoint, output() / "checkpoints")
    inference = InferenceConfig.model_validate(llm.config)
    served = partial(VLLMRuntime, identity_path=output() / "vllm-identity.json")
    with InferenceSession(inference, load_seconds=60, factory=served) as session:
        return play_game(match, session, opponent, backend=resolve("torch"))


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """Every ordered deck pairing once: 100 games, Qwen on the play in half."""
    games = schedule(
        PACK,
        DECKS,
        deals=1,
        deal_base=214183000,
        opponent_seed_base=214595000,
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
            Seating(name="qwen-v-mtg183", players=players(), configs=games, games=1),
        ),
    )
    return [
        EvalJob(
            play=play,
            tournament=tournament,
            workers=WORKERS,
            setup="sh scripts/qwen-mtg183-vllm-setup.sh",
            environment=(
                # Setup installs no workspace package and places a prebuilt
                # native engine, which a syncing `uv run` would rebuild.
                Variable(name="UV_NO_SYNC", value="1"),
                Variable(name="HF_HUB_OFFLINE", value="1"),
                Variable(name="OMP_NUM_THREADS", value="1"),
            ),
            max_hours=11,
        )
    ]


def report() -> Report:
    """Qwen's score by the deck it piloted and the deck it faced."""
    played = Matches(tournament=TOURNAMENT)
    return Report(
        score=Score(table="games", value="score", reference=0.5, unit="games"),
        evals=(
            Matrix(
                title="Qwen deck × checkpoint deck",
                rows="opponent_deck",
                columns="deck",
                matches=played,
            ),
            Ranking(title="By Qwen deck", by="deck", matches=played),
            Ranking(title="By checkpoint deck", by="opponent_deck", matches=played),
        ),
    )
