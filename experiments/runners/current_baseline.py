"""ETU-118 declaration and bounded positive-control execution.

Experiment resolves recipes; execute_regime/VerifyStore own actual learning.
The shared asynchronous monitor is Allies/Lessons-only, so injected-root scoring
is explicit here. Every selected action is re-executed from its original root.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import signal
import time

import torch

from experiments.runners.current_baseline_evidence import Attempt, Result, Score
from experiments.runners.run_value_models import smoke_baseline
from manabot.arena.models import file_sha256
from manabot.env import Match
from manabot.env.target_practice import resolve_target, target_root
from manabot.sim.flat_mc import AgentMatchupPlayer, load_checkpoint_agent
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.experiments import Baseline, Experiment
from manabot.training.models import (
    TrainingCoordinates,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import (
    MonitorProtocol,
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.verify.store import VerifyStore

SEEDS = (11801, 11802, 11803)
DEALS = tuple(range(1_911_180_000, 1_911_180_064))


def declaration(*, frozen: bool = False, full_game: bool = False) -> Experiment:
    base = smoke_baseline()
    base.id = "lethal-target-frozen" if frozen else "lethal-target-learning"
    base.agent.value_aggregation = "masked_mean"
    base.wall_seconds = 780
    stage = deepcopy(base.stages[0])
    assert isinstance(stage, TrainSelfPlay)
    stage.root = "lethal-target-v1"
    stage.trainable = "none" if frozen else "policy_value"
    stage.updates = 100
    stage.transitions = 16
    stage.execution.wall_seconds = 760
    if full_game:
        base.id = "full-game-frozen" if frozen else "full-game-learning"
        base.wall_seconds = 600
        stage.root = None
        stage.behavior = "random"
        stage.updates = 128
        stage.transitions = 64
        stage.execution.wall_seconds = 580
    base.stages = [stage]
    base = TrainingRegime.model_validate(base.model_dump())
    return Experiment(name=base.id, baseline=Baseline.capture(base.id, base))


def evaluate(checkpoint: Path, regime: TrainingRegime) -> list[Score]:
    agent, space = load_checkpoint_agent(str(checkpoint))
    player = AgentMatchupPlayer(agent)
    rows: list[Score] = []
    for deal in DEALS:
        for seat in (0, 1):
            env = target_root(Match(regime.match), space, deal, seat)
            before = env._engine.state_digest()
            obs = space.encode(env.last_raw_obs)
            with torch.no_grad():
                logits, _ = agent(
                    {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
                )
                probabilities = logits.softmax(-1)[0]
            if not torch.isfinite(probabilities).all() or not torch.isclose(
                probabilities.sum(), torch.tensor(1.0)
            ):
                raise ValueError("invalid legal distribution")
            if (probabilities[2:] != 0).any():
                raise ValueError("policy assigned probability outside legal support")
            torch.manual_seed(deal + seat)
            action = player.act(env, obs)
            winner = resolve_target(env, action)
            after = env._engine.state_digest()
            replay = target_root(Match(regime.match), space, deal, seat)
            if (
                replay._engine.state_digest() != before
                or resolve_target(replay, action) != winner
                or replay._engine.state_digest() != after
            ):
                raise ValueError("target-root exact replay diverged")
            rows.append(
                Score(
                    deal=deal,
                    seat=seat,
                    action=action,
                    winner=winner,
                    win=winner == seat,
                    probabilities=probabilities[:2].tolist(),
                    root=before,
                    terminal=after,
                    replay=True,
                )
            )
    return rows


def _expire(signum: int, frame: object) -> None:
    raise TimeoutError("ETU-118 declared attempt allowance exhausted")


def score(run: TrainingRun, kind: str, out: Path, full_game: bool) -> list[Score]:
    artifact = run.stages[0].artifacts[kind]
    if not full_game:
        return evaluate(Path(artifact["path"]), run.regime)
    final = stage_checkpoint(run, run.stages[0].id)
    assert final is not None
    coordinates = (
        final.coordinates
        if kind == "raw"
        else TrainingCoordinates(
            stage_id="initialization", updates=0, training_seconds=0
        )
    )
    measured = evaluate_checkpoint(
        run,
        artifact,
        coordinates,
        out,
        protocol=MonitorProtocol(
            deal_seeds=tuple(range(1_911_181_000, 1_911_181_012)), game_seconds=30
        ),
    )
    if measured.status != "completed":
        raise RuntimeError(f"full-game evaluation incomplete: {measured.error}")
    return [
        Score(
            deal=row.deal_seed,
            leg=row.leg,
            win=row.score_a,
            replay=row.replay_passed,
        )
        for row in measured.rows
    ]


def run(out: Path, prior: Path | None = None, *, full_game: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    prior_seconds = 0.0
    if prior is not None:
        previous = Result.model_validate_json(prior.read_text())
        if previous.status != ("completed" if full_game else "failed"):
            raise ValueError("explicit recovery requires retained failure")
        prior_seconds = previous.seconds + previous.prior_seconds
        if full_game:
            for attempt in previous.attempts:
                if not attempt.frozen:
                    initial = sum(row.win for row in attempt.initial) / len(
                        attempt.initial
                    )
                    trained = sum(row.win for row in attempt.rows) / len(attempt.rows)
                    if trained < 0.85 or trained - initial < 0.25:
                        raise ValueError(
                            "positive control failed; full-game debugging forbidden"
                        )
    elif full_game:
        raise ValueError(
            "full-game debugging requires completed positive-control evidence"
        )
    result = Result(
        status="running",
        attempts=[],
        prior_seconds=prior_seconds,
        prior_sha256=file_sha256(prior) if prior is not None else None,
        full_game=full_game,
    )

    def persist() -> None:
        result.seconds = time.monotonic() - started
        atomic_json(
            out / "result.json", result.model_dump(mode="json", exclude_unset=True)
        )

    signal.signal(signal.SIGALRM, _expire)
    try:
        with VerifyStore(out / "verify.sqlite") as store:
            for seed in (11821, 11822, 11823) if full_game else SEEDS:
                initial_state: dict[str, torch.Tensor] | None = None
                initial_rows: list[Score] | None = None
                for frozen in (False, True):
                    allowance = (
                        min(2700, 3600 - prior_seconds)
                        if full_game
                        else 3600 - prior_seconds
                    )
                    remaining = allowance - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("ETU-118 aggregate allowance exhausted")
                    signal.setitimer(signal.ITIMER_REAL, min(900, remaining))
                    tick = time.monotonic()
                    regime = (
                        declaration(frozen=frozen, full_game=full_game)
                        .resolve()
                        .cases[0]
                        .regime
                    )
                    attempt = Attempt(seed=seed, frozen=frozen, status="running")
                    result.attempts.append(attempt)
                    persist()
                    try:
                        training = execute_regime(
                            regime,
                            seed,
                            out / f"{seed}-{'frozen' if frozen else 'learning'}",
                            store,
                        )
                        attempt.run_id = training.id
                        if training.status != "completed":
                            raise RuntimeError(training.error)
                        artifacts = training.stages[0].artifacts
                        initial = Path(artifacts["initial_raw"]["path"])
                        final = Path(artifacts["raw"]["path"])
                        init_agent, _ = load_checkpoint_agent(str(initial))
                        final_agent, _ = load_checkpoint_agent(str(final))
                        if initial_state is None:
                            initial_state = deepcopy(init_agent.state_dict())
                            initial_rows = score(
                                training,
                                "initial_raw",
                                out / f"{seed}-initial-eval",
                                full_game,
                            )
                            attempt.initial = initial_rows
                        if any(
                            not torch.equal(value, initial_state[key])
                            for key, value in init_agent.state_dict().items()
                        ):
                            raise ValueError("paired initialization changed")
                        unchanged = all(
                            torch.equal(value, initial_state[key])
                            for key, value in final_agent.state_dict().items()
                        )
                        if unchanged != frozen:
                            raise ValueError(
                                "unexpected frozen/learned weight identity"
                            )
                        scored = score(
                            training,
                            "raw",
                            out / f"{seed}-{'frozen' if frozen else 'trained'}-eval",
                            full_game,
                        )
                        if frozen and scored != initial_rows:
                            raise ValueError(
                                "frozen control differs from initialization"
                            )
                        attempt.status = "completed"
                        attempt.checkpoint_sha256 = file_sha256(final)
                        attempt.initial_sha256 = file_sha256(initial)
                        attempt.rows = scored
                    except BaseException as error:
                        attempt.status = "failed"
                        attempt.error = f"{type(error).__name__}: {error}"
                        raise
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        attempt.seconds = time.monotonic() - tick
                        persist()
        result.status = "completed"
    except BaseException as error:
        result.status = "failed"
        result.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        persist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--prior-result", type=Path)
    parser.add_argument("--full-game", action="store_true")
    args = parser.parse_args()
    if args.plan:
        atomic_json(
            args.plan,
            {
                "seeds": [11821, 11822, 11823] if args.full_game else list(SEEDS),
                "held_out_deals": list(range(1_911_181_000, 1_911_181_012))
                if args.full_game
                else list(DEALS),
                "attempt_seconds": 900,
                "aggregate_seconds": 2700 if args.full_game else 3600,
                "minimum_win_rate_each_seed": None if args.full_game else 0.85,
                "minimum_gain_each_seed": 0.0 if args.full_game else 0.25,
                "minimum_mean_gain": 0.10 if args.full_game else 0.25,
                "recipes": [
                    declaration(frozen=f, full_game=args.full_game).resolve().receipt()
                    for f in (False, True)
                ],
            },
        )
    elif args.run:
        run(args.run, args.prior_result, full_game=args.full_game)
    else:
        parser.error("select --plan or --run")


if __name__ == "__main__":
    main()
