"""Analysis: what a policy would have done at one seat's decisions of a played game.

An analysis re-executes a finished or stopped Play game (`record_of`) with a
`PolicyRunner` in one seat. At each of that seat's deliberate decisions the
runner reads the seat's own observation, its probabilities and value are kept
(`AnalysedDecision`), its sampled action is discarded, and the walk applies the
action that was played, so a decision-memory policy sees the history a bot in
that seat would have.

Each offer also gets the policy's estimate after it (`AnalysedDecision.after`).
A policy's value belongs to the seat that chooses, so the estimate is read at
the next decision after the offer: from the seat itself when it chooses next,
or from the opponent's view negated when the opponent does. The opponent's view
is read in a world re-dealt from what the seat knew (`Env.determinize` with the
seat's perspective), never the real hidden cards; with closed decklists no
world can be re-dealt and those estimates are absent. An offer that ends the
game reads its result.

Scores are on one 0–1 scale from the analysed seat: `signed_outcome` values
map to expected score ((v + 1) / 2, a draw half a win) and `win_logit` values
to a win chance. An undeclared semantic has no scale, so its offers have no
estimate. Entry points: `analyse_game`, a generator of decisions;
`run_analysis`, which saves them (`AttemptStore.analyses`) as they come; and
`analysis_choices`, the policies that may analyse a game.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
import logging
import math

from pydantic import BaseModel, ConfigDict

import managym
from managym.academy.attempts import AttemptStore, RetainedAttempt
from managym.academy.opponent import Opponent
from managym.academy.recording import record_of, recorded_build, step_matches
from managym.academy.trace import GameConfig
from managym.policy import PolicyRegistry, PolicyRunner
from managym.recording import DecisionAnnotation
from managym.replay import walk

# What each scored value semantic is called on the 0–100 chart.
SCORED_SEMANTICS = {"signed_outcome": "Expected score", "win_logit": "Win chance"}


class AnalysedDecision(BaseModel):
    """The policy's reading of one decision: `annotation.probabilities` in the
    decision's offer order and `annotation.value` from the analysed seat;
    `played` the offer that was played; `after[i]` the score after offer `i`
    (None where it could not be read)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event: int
    played: int
    annotation: DecisionAnnotation
    semantic: str | None
    after: list[float | None]


class AnalysisStop(Exception):
    """The analysis cannot go on; decisions before it keep their evidence."""


def score(value: float, semantic: str | None) -> float | None:
    """A value as the analysed seat's 0–1 score, or None without a scale."""
    if semantic == "signed_outcome":
        return min(1.0, max(0.0, (value + 1) / 2))
    if semantic == "win_logit":
        return 1 / (1 + math.exp(-max(-60.0, min(60.0, value))))
    return None


def analysed_events(attempt: RetainedAttempt, seat: int) -> list[int]:
    """The events an analysis of `seat` reads: its deliberate decisions, not automation."""
    human = attempt.trace["config"].get("human_seat", 0)
    role = "hero" if seat == human else "villain"
    return [
        i
        for i, event in enumerate(attempt.trace["events"])
        if event["actor"] == role and not event.get("auto")
    ]


def analyse_game(
    attempt: RetainedAttempt, seat: int, runner: PolicyRunner, *, seed: int
) -> Iterator[AnalysedDecision]:
    """Each analysed decision of `seat`, in order, as the walk reaches it.

    Raises AnalysisStop where the game stops re-executing as recorded (another
    seat acts, the action differs) or the policy fails. The walk checks each
    step's acting seat and action description, not its whole board: Review's
    views check boards (`ReplayedViews`); here a forward pass is the cost.
    """
    events = attempt.trace["events"]
    human = attempt.trace["config"].get("human_seat", 0)
    wanted = set(analysed_events(attempt, seat))
    try:
        walker = walk(record_of(attempt))
    except (KeyError, TypeError, ValueError, managym.AgentError) as error:
        raise AnalysisStop(f"This game cannot be re-executed here: {error}") from None
    steps = iter(walker)
    for index, event in enumerate(events):
        observation = walker.observation
        env = walker.env
        if not step_matches(observation, event, human):
            raise AnalysisStop(f"Step {index + 1} re-executed differently.")
        if index in wanted:
            try:
                runner(env, observation)
            except (ValueError, managym.AgentError) as error:
                raise AnalysisStop(f"The policy failed at step {index + 1}: {error}")
            decision = runner.last_decision
            assert decision is not None
            if not decision.probabilities:
                raise AnalysisStop("This policy gives no probabilities to read.")
            yield AnalysedDecision(
                event=index,
                played=event["action"],
                annotation=DecisionAnnotation(
                    probabilities=decision.probabilities, value=decision.value
                ),
                semantic=decision.value_semantic,
                after=offer_estimates(
                    env,
                    len(observation.action_space.actions),
                    seat,
                    runner,
                    decision.value_semantic,
                    seed=(seed * 1_000_003 + index) % 2**63,
                ),
            )
        if next(steps, None) is None:
            stop = walker.stop
            raise AnalysisStop(
                f"Step {index + 1} re-executed differently: {stop.reason if stop else 'stopped'}"
            )


def offer_estimates(
    env: managym.Env,
    count: int,
    seat: int,
    runner: PolicyRunner,
    semantic: str | None,
    *,
    seed: int,
) -> list[float | None]:
    """The policy's score from `seat` after each offer of `env`'s decision.

    Reads the next chooser's value through the session's history-free
    `evaluate` (the search contract); a policy without it has no estimates.
    `env` is not changed.
    """
    evaluate = getattr(runner.session, "evaluate", None)
    if (
        score(0.0, semantic) is None
        or not callable(evaluate)
        or runner.base_descriptor.recurrent
    ):
        return [None] * count
    base = env.clone_env()
    try:
        base.determinize(seed=seed, perspective=seat)
        dealt = True
    except managym.AgentError:
        # Closed decklists: nothing to re-deal from, so the real world is used
        # only where it reveals nothing (the seat chooses next, or the game ends).
        dealt = False
    estimates: list[float | None] = []
    for offer in range(count):
        world = base.clone_env()
        observation = world.step(offer)[0]
        if world.is_game_over():
            winner = world.winner_index()
            estimates.append(0.5 if winner is None else float(winner == seat))
            continue
        chooser = world.current_agent_index()
        if chooser is None or (chooser != seat and not dealt):
            estimates.append(None)
            continue
        view = replace(
            runner.view(observation),
            seat=chooser,
            capture_json=world.capture_decision(chooser, 0).to_json(),
        )
        result = evaluate(view)
        value = result.value
        reading = (
            score(value.value, value.semantic)
            if value.seat == chooser and math.isfinite(value.value)
            else None
        )
        estimates.append(
            None if reading is None else reading if chooser == seat else 1 - reading
        )
    return estimates


# ———————————————— Which policies may analyse a game


@dataclass(frozen=True)
class AnalysisChoices:
    """The policies that may analyse a game's seat, and the one opened by default.

    `unavailable` says why none may (another build, a watched game); `default`
    is the opponent's own policy when it is one of `policies`.
    """

    policies: list[dict[str, str]]
    default: str | None
    unavailable: str | None


def analysis_choices(
    attempt: RetainedAttempt, registry: PolicyRegistry
) -> AnalysisChoices:
    config = GameConfig(**attempt.trace["config"])
    if config.watch_players is not None:
        return AnalysisChoices([], None, "A watched game has no seat of yours.")
    try:
        walk(record_of(attempt))
    except (KeyError, TypeError, ValueError, managym.AgentError):
        return AnalysisChoices(
            [],
            None,
            "Analysis re-executes the game, which needs the build that recorded it "
            f"({recorded_build(attempt.world_json)}).",
        )
    setup = config.to_rust()
    policies = []
    for descriptor in registry.descriptors():
        if descriptor.base_policy is not None or not descriptor.artifacts:
            # Search plays out from the walk's real hidden cards; a scripted
            # player has no probabilities to read.
            continue
        try:
            registry.admit(descriptor.id, setup)
        except (ImportError, OSError, ValueError):
            continue
        policies.append({"sha256": descriptor.id, "name": Opponent(descriptor).name})
    opponent = (config.opponent or {}) if config.villain_type == "policy" else {}
    key = opponent.get("base_sha256", opponent.get("sha256"))
    default = key if any(p["sha256"] == key for p in policies) else None
    return AnalysisChoices(policies, default, None)


def run_analysis(
    store: AttemptStore,
    analysis: str,
    attempt: RetainedAttempt,
    seat: int,
    runner: PolicyRunner,
) -> None:
    """Walk the game, saving each decision as it is read, then close `runner`."""
    reason: str | None = None
    try:
        seed = int(attempt.trace["config"]["seed"])
        for decision in analyse_game(attempt, seat, runner, seed=seed):
            store.add_analysed_decision(
                analysis, decision.event, decision.model_dump_json()
            )
    except AnalysisStop as stop:
        reason = str(stop)
    except Exception as error:  # noqa: BLE001 - any failure stops this analysis only
        logging.getLogger(__name__).exception("Analysis %s failed", analysis)
        reason = f"The analysis failed: {type(error).__name__}"
    finally:
        runner.close()
        store.end_analysis(analysis, reason)
