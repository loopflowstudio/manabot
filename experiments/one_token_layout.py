"""Exploratory MTG-344 comparison: v10 object rows against input v11's one token per game thing.

Both arms are the ten-deck Elemental recipe (`elemental_variants.elemental_recipe`,
two-seat self-play over uniform deck pairs) and differ only in
`observation.policy_input_version`: 10, the deprecated rows in which a
battlefield object or a stack spell is also a card row, or 11, in which each
game thing is one row (cards outside the battlefield and stack, permanents with
their characteristics, a stack segment). Both carry the same turn state, so the
arms see the same information laid out differently. Two shared seeds per arm
start from matched initialization and meet the same deals until their policies
diverge. The report is experiments/one_token_layout.md.

    uv run etude launch experiments/one_token_layout.py --rent l4 --parallel 4 \\
        --cohort mtg344 --max-usd 25 --revision COMMIT
    uv run etude eval experiments/one_token_layout.py --rent cpu32 --source r2://manabot/jobs
    uv run etude report experiments/one_token_layout.py --source r2://manabot/jobs --open
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
import time
from typing import Literal
import warnings

import numpy as np
import torch

from etude.eval.records import Inference, Seating
from etude.experiment import ExperimentState, TrainingRun
from etude.launch.records import Job, Run
from etude.report import Curve, Matches, Ranking, Report, Score
import managym

from experiments.elemental_variants import elemental_recipe
from experiments.learning_signal import (
    DECKS,
    ELEMENTAL_PAIRS,
    FINALIZATION_SECONDS,
    SETUP_SECONDS,
    Conditions,
    benchmark_jobs,
    scheduled_matchups,
    training_runs,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationEncoder, ObservationSpace
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.infra.runtime import regime_tag
from manabot.model.admission import admit_observation
from manabot.training.panels import ML, PERF
from manabot.training.regime import TrainingRegime

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
Arm = Literal[10, 11]
# The deprecated rows first: the reference arm and the report's baseline.
ARMS: tuple[Arm, ...] = (10, 11)
SEEDS = (34401, 34402)
UPDATES = 200
# MTG-181 measured 28-36 s per two-seat update of this recipe at 256 streams
# on an L4; 200 updates at 40 s plus setup and the final export.
SECONDS_PER_UPDATE = 40.0
HOURS = (UPDATES * SECONDS_PER_UPDATE + SETUP_SECONDS + FINALIZATION_SECONDS) / 3600
USD_PER_RUN = 2.25
EVALUATION = "one-token-layout-v1"
# Scheduled checkpoints against the heuristic and four head-to-heads, each
# over 50 pairings x 4 legs.
EVALUATION_HOURS = 1.0
EVALUATION_USD = 2.0


def regime_id(version: Arm) -> str:
    return f"one-token-layout-v{version}-v1"


def arm(version: Arm) -> TrainingRegime:
    """The Elemental recipe at `UPDATES` reading policy input `version`."""
    regime = elemental_recipe(regime_id(version), UPDATES)
    regime.observation = regime.observation.model_copy(
        update={"policy_input_version": version}
    )
    return TrainingRegime.model_validate(regime.authored_dump())


def require_single_axis(baseline: TrainingRegime, candidate: TrainingRegime) -> None:
    """Reject any resolved difference other than the label and the input version."""
    left = baseline.model_dump(mode="json")
    right = candidate.model_dump(mode="json")
    for dump in (left, right):
        dump.pop("id")
        dump["observation"].pop("policy_input_version")
    if left != right:
        raise ValueError("layout arm changes more than observation.policy_input_version")


def runs(
    revision: str,
    hours: float = HOURS,
    usd_per_run: float = USD_PER_RUN,
    seeds: str = ",".join(str(seed) for seed in SEEDS),
    arms: str = ",".join(str(version) for version in ARMS),
) -> list[Run]:
    """Both arms under the same seeds; `arms` and one seed re-queue a lost run."""
    chosen = tuple(int(seed) for seed in seeds.split(","))
    if len(set(chosen)) != len(chosen):
        raise ValueError("seeds repeat")
    wanted = {int(version) for version in arms.split(",")}
    if not wanted <= set(ARMS):
        raise ValueError(f"unknown arms {sorted(wanted - set(ARMS))}")
    reference, candidate = (arm(version) for version in ARMS)
    require_single_axis(reference, candidate)
    return training_runs(
        tuple(
            regime
            for version, regime in zip(ARMS, (reference, candidate), strict=True)
            if version in wanted
        ),
        updates=UPDATES,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=chosen,
        hours=hours,
        usd_per_run=usd_per_run,
    )


# --------------------------------------------------------------------------- evaluation


def arm_runs(state: ExperimentState, version: Arm) -> list[TrainingRun]:
    """The arm's runs: their checkpoints' regime tag names them."""
    tag = regime_tag(regime_id(version))
    return [run for run in state.runs if run.checkpoints.tagged(tag)]


def eval(state: ExperimentState) -> list[Job]:
    return benchmark_jobs(
        evaluation_matchups(state),
        EVALUATION,
        ELEMENTAL_PAIRS,
        hours=EVALUATION_HOURS,
        usd=EVALUATION_USD,
    )


def evaluation_matchups(state: ExperimentState) -> tuple[Seating, ...]:
    """Scheduled checkpoints against the heuristic, then v11 against v10.

    A head-to-head pairs two ended runs at the last update both checkpointed,
    v10 first; the four legs give each both decks on the play and the draw.
    """
    found = [
        matchup
        for run in state.runs
        for matchup in scheduled_matchups(run, ELEMENTAL_PAIRS)
    ]
    source = [
        *sorted((ROOT / "manabot").rglob("*.py")),
        ROOT / "experiments/learning_signal.py",
    ]
    inference = Inference(source=source)
    conditions = tuple(
        Conditions(design=ELEMENTAL_PAIRS.name, **pairing.model_dump()).model_dump()
        for pairing in ELEMENTAL_PAIRS.pairings
    )
    reference, candidate = (arm_runs(state, version) for version in ARMS)
    for left, right in product(reference, candidate):
        if not (left.ended and right.ended):
            continue
        shared = {int(c.at["updates"]) for c in left.checkpoints} & {
            int(c.at["updates"]) for c in right.checkpoints
        }
        if not shared:
            continue
        update = max(shared)
        first = left.checkpoints.at("updates", update).last
        second = right.checkpoints.at("updates", update).last
        assert first is not None and second is not None
        found.append(
            Seating(
                name=f"v10 {left.name} v v11 {right.name} u{update}",
                players=(first.model(inference), second.model(inference)),
                conditions=conditions,
                games=ELEMENTAL_PAIRS.legs,
                tags=("head-to-head",),
            )
        )
    return tuple(found)


def report() -> Report:
    """Score against the heuristic by update, and v11 against v10 at matched updates."""
    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="updates",
        baseline=regime_id(ARMS[0]),
        evals=(
            Curve(title="Score against the heuristic", matches=Matches(tag="heuristic")),
            Ranking(
                title="Head-to-head at matched updates, by seat",
                by="on_play",
                matches=Matches(tag="head-to-head"),
            ),
        ),
        perf=PERF,
        ml=ML,
    )


# --------------------------------------------------------------------------- measurement


@dataclass(frozen=True)
class LayoutCost:
    """One arm's rows and costs over the same seeded random Elemental games."""

    version: int
    observations: int
    # Per seat-side row counts: maximum and 99th percentile.
    card_rows_max: int
    card_rows_p99: float
    permanent_rows_max: int
    stack_rows_max: int
    stack_rows_p99: float
    # Native encode into preallocated buffers, microseconds per observation.
    encode_us_median: float
    # Policy forward at the arm's recipe, CPU, batch 256, milliseconds.
    forward_ms_median: float


def _games(seed: int, count: int) -> list[managym.Observation]:
    """Acting-viewer observations of seeded random games over the ten decks."""
    rng = np.random.default_rng(seed)
    env = managym.Env(seed=seed, skip_trivial=True)
    observations: list[managym.Observation] = []
    for game in range(count):
        left, right = rng.choice(len(DECKS), 2, replace=False)
        match = Match(
            MatchConfig.authored(
                ELEMENTAL_PAIRS.pack, DECKS[int(left)], DECKS[int(right)]
            )
        )
        obs, _ = env.reset(match.to_rust())
        for _ in range(2000):
            observations.append(obs)
            if obs.game_over:
                break
            obs, *_ = env.step(int(rng.integers(len(obs.action_space.actions))))
    return observations


def measure(output: Path, games: int = 40) -> list[LayoutCost]:
    """Rows, native encode time and policy forward time for both arms.

    Card capacity is raised to 128 so the v10 arm's peak is measured, not refused.
    """
    observations = _games(34400, games)
    env = managym.Env(seed=0, skip_trivial=True)
    torch.set_num_threads(4)
    results: list[LayoutCost] = []
    for version in ARMS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = arm(version).observation.model_copy(
                update={"max_cards_per_player": 128}
            )
            config = ObservationConfig.model_validate(config.model_dump())
        encoder = ObservationEncoder(config)
        buffers = encoder.allocate()
        cards: list[int] = []
        permanents: list[int] = []
        stack: list[int] = []
        encode_us: list[float] = []
        rows: list[dict[str, np.ndarray]] = []
        for obs in observations:
            start = time.perf_counter()
            env.encode_observation_into(obs, buffers)
            encode_us.append((time.perf_counter() - start) * 1e6)
            cards += [
                int(buffers[f"{side}_cards_valid"].sum())
                for side in ("agent", "opponent")
            ]
            permanents += [
                int(buffers[f"{side}_permanents_valid"].sum())
                for side in ("agent", "opponent")
            ]
            # Rows holding stack things: v11's stack segment; v10's spell
            # card rows (its abilities have none).
            stack.append(
                int(buffers["stack_valid"].sum())
                if "stack_valid" in buffers
                else sum(item.card is not None for item in obs.stack_objects)
            )
            if len(rows) < 256 and obs.action_space.actions:
                rows.append({key: value.copy() for key, value in buffers.items()})
        space = ObservationSpace(config)
        torch.manual_seed(0)
        model = ActorCritic(space, arm(version).agent).eval()
        batch = {
            key: torch.from_numpy(np.stack([row[key] for row in rows]))
            for key in rows[0]
        }
        forward_ms: list[float] = []
        with torch.no_grad():
            for _ in range(12):
                admitted = admit_observation(model, batch)
                start = time.perf_counter()
                model(admitted)
                forward_ms.append((time.perf_counter() - start) * 1e3)
        results.append(
            LayoutCost(
                version=version,
                observations=len(observations),
                card_rows_max=max(cards),
                card_rows_p99=float(np.percentile(cards, 99)),
                permanent_rows_max=max(permanents),
                stack_rows_max=max(stack),
                stack_rows_p99=float(np.percentile(stack, 99)),
                encode_us_median=float(np.median(encode_us)),
                forward_ms_median=float(np.median(forward_ms[2:])),
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(row) for row in results], indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("measure",))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/data/one_token_layout/costs.json",
    )
    parser.add_argument("--games", type=int, default=40)
    args = parser.parse_args()
    for row in measure(args.out, args.games):
        print(row)


if __name__ == "__main__":
    main()
