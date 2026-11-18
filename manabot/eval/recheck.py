"""Remeasure a strength reference, resampling the same units as its writer."""

from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Literal

import numpy as np
from pydantic import Field, TypeAdapter, model_validator

from manabot.eval.evaluation import evaluate_local
from manabot.eval.local import prepare_local_request
from manabot.eval.results import RateInterval, bootstrap_mean_interval
from manabot.infra.artifacts import StoredArtifact
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel
from manabot.infra.source import source_commit
from manabot.sim.matchup import DeckMatchup


class RecordedStrength(StrictModel):
    score: float = Field(ge=0, le=1)
    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)
    games: int = Field(gt=0)
    revision: str
    report: str

    @model_validator(mode="after")
    def valid_interval(self) -> "RecordedStrength":
        if not self.low <= self.score <= self.high:
            raise ValueError("recorded score outside its interval")
        return self


class StrengthReference(StrictModel):
    name: str
    checkpoint: StoredArtifact
    opponent: Literal["heuristic"]
    max_actions: int = Field(gt=0)
    matchups: Path
    deal_offsets: tuple[int, ...] = Field(min_length=1)
    recorded: RecordedStrength
    opponent_changes: tuple[str, ...] = ()


class ReferencePair(StrictModel):
    matchup: DeckMatchup
    deal_seed: int = Field(ge=0)


def measure_pairs(
    reference: StrengthReference,
    pairs: list[ReferencePair],
    *,
    out: Path,
    deadline: float,
) -> RateInterval:
    """Require every crossed leg and replay before scoring any of the cohort."""
    if not pairs:
        raise ValueError("empty deck-pair cohort")
    scores = []
    for index, pair in enumerate(pairs):
        if time.time() >= deadline:
            raise TimeoutError("allocation exhausted before the next pair")
        request = prepare_local_request(
            reference.checkpoint,
            reference.opponent,
            tuple(pair.deal_seed + offset for offset in reference.deal_offsets),
            pair.matchup,
            deadline=deadline,
            heuristic_actions=reference.max_actions,
            show_binding=index == 0,
        )
        directory = out / f"pair-{index:02d}"
        result = evaluate_local(request, deadline=deadline, out_dir=directory)
        if result.status != "completed" or result.score is None:
            raise ValueError(
                f"pair {index} incomplete ({len(result.rows)}/{request.expected_games} rows): {result.error or directory}"
            )
        scores.append(result.score.mean)
    # Deals and crossed legs from a deck pair stay together, as in the frontier report.
    indices = np.random.default_rng(21592001).integers(
        len(scores), size=(2000, len(scores))
    )
    return bootstrap_mean_interval(scores, indices)


def recheck(path: Path, *, out: Path | None, seconds: float) -> int:
    deadline = time.time() + seconds
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    reference = StrengthReference.model_validate_json(path.read_bytes())
    pairs = TypeAdapter(list[ReferencePair]).validate_json(
        (path.parent / reference.matchups).read_bytes()
    )
    r = reference.recorded
    print(f"{reference.name} against {reference.opponent}, {len(pairs)} deck pairs")
    print(
        f"  recorded  {r.score:.2f} ({r.low:.2f}–{r.high:.2f}), {r.games} games, at {r.revision}",
        flush=True,
    )
    if reference.opponent_changes:
        print(
            f"  opponent  heuristic changed since the record ({', '.join(reference.opponent_changes)})"
        )
    with TemporaryDirectory(prefix="manabot-recheck-") as temp:
        directory = out or Path(temp)
        if out is not None:
            directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "reference.json", reference.model_dump(mode="json"))
        score = measure_pairs(reference, pairs, out=directory, deadline=deadline)
        games = len(pairs) * len(reference.deal_offsets) * 4
        revision = source_commit()
        atomic_json(
            directory / "measured.json",
            dict(score=score.model_dump(), games=games, revision=revision),
        )
        print(
            f"  measured  {score.mean:.2f} ({score.lower:.2f}–{score.upper:.2f}), {games} games, at {revision[:10]}"
        )
        inside = r.low <= score.mean <= r.high
        print(f"  {'inside' if inside else 'outside'} the recorded interval")
        return 0 if inside else 1
