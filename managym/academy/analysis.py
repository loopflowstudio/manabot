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
`run_analysis`, which saves them (`AttemptStore.analyses`) as they come;
`analysis_choices`, the policies that may analyse a game, admitted once per
setup by `PolicyAdmissions`; `agent_line`,
the policy's own line from one analysed decision; and `LiveAnalysis`, the same
reading of a game while it is played, when the person shows it.
"""

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
import json
import logging
import math
import queue
import threading
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

import managym
from managym.academy.attempts import AttemptStore, RetainedAttempt
from managym.academy.opponent import Opponent, policy_registry
from managym.academy.recording import record_of, recorded_build, step_matches
from managym.academy.trace import GameConfig, academy_actions
from managym.bots import BotPolicy, choose_bot
from managym.experience_projection import decision_surface_view
from managym.policy import PolicyDescriptor, PolicyRegistry, PolicyRunner
from managym.recording import DecisionAnnotation
from managym.replay import RecordWalker, walk

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


def analysed_events(trace: dict[str, Any], seat: int) -> list[int]:
    """The events an analysis of `seat` reads: its deliberate decisions, not automation."""
    human = trace["config"].get("human_seat", 0)
    role = "hero" if seat == human else "villain"
    return [
        i
        for i, event in enumerate(trace["events"])
        if event["actor"] == role and not event.get("auto")
    ]


def open_walk(attempt: RetainedAttempt) -> RecordWalker:
    """A walk of the played game; AnalysisStop if this build cannot re-execute it."""
    try:
        return walk(record_of(attempt))
    except (KeyError, TypeError, ValueError, managym.AgentError) as error:
        raise AnalysisStop(f"This game cannot be re-executed here: {error}") from None


def recorded_steps(
    walker: RecordWalker, trace: dict[str, Any]
) -> Iterator[tuple[int, managym.Env, managym.Observation]]:
    """Walk `trace`'s events, giving each as `(event, env, observation)` before
    its action is applied. Once exhausted, `walker.env` is at the position after
    the last event.

    Raises AnalysisStop where the game stops re-executing as recorded (another
    seat acts, the action differs). The walk checks each step's acting seat and
    action description, not its whole board: Review's views check boards
    (`ReplayedViews`); here a forward pass is the cost.
    """
    human = trace["config"].get("human_seat", 0)
    steps = iter(walker)
    for index, event in enumerate(trace["events"]):
        if not step_matches(walker.observation, event, human):
            raise AnalysisStop(f"Step {index + 1} re-executed differently.")
        yield index, walker.env, walker.observation
        if next(steps, None) is None:
            stop = walker.stop
            raise AnalysisStop(
                f"Step {index + 1} re-executed differently: {stop.reason if stop else 'stopped'}"
            )


def seat_decisions(
    walker: RecordWalker, trace: dict[str, Any], seat: int
) -> Iterator[tuple[int, managym.Env, managym.Observation]]:
    """`recorded_steps` at `seat`'s analysed decisions only."""
    wanted = set(analysed_events(trace, seat))
    return (step for step in recorded_steps(walker, trace) if step[0] in wanted)


def ask(
    runner: PolicyRunner,
    env: managym.Env,
    observation: managym.Observation,
    event: int,
) -> None:
    """Ask `runner` at `event`; AnalysisStop if it fails."""
    try:
        runner(env, observation)
    except (ValueError, managym.AgentError) as error:
        raise AnalysisStop(f"The policy failed at step {event + 1}: {error}")


def read_decision(
    env: managym.Env,
    observation: managym.Observation,
    event: int,
    seat: int,
    runner: PolicyRunner,
    *,
    seed: int,
) -> tuple[DecisionAnnotation, str | None, list[float | None]]:
    """The policy's reading of `seat`'s decision at `env`: its annotation, value
    semantic and estimate after each offer. Raises AnalysisStop if it fails."""
    ask(runner, env, observation, event)
    decision = runner.last_decision
    assert decision is not None
    if not decision.probabilities:
        raise AnalysisStop("This policy gives no probabilities to read.")
    after = offer_estimates(
        env,
        len(observation.action_space.actions),
        seat,
        runner,
        decision.value_semantic,
        seed=(seed * 1_000_003 + event) % 2**63,
    )
    annotation = DecisionAnnotation(
        probabilities=decision.probabilities, value=decision.value
    )
    return annotation, decision.value_semantic, after


def analyse_game(
    attempt: RetainedAttempt, seat: int, runner: PolicyRunner, *, seed: int
) -> Iterator[AnalysedDecision]:
    """Each analysed decision of `seat`, in order, as the walk reaches it.

    Raises AnalysisStop where the game stops re-executing or the policy fails.
    """
    events = attempt.trace["events"]
    for index, env, observation in seat_decisions(
        open_walk(attempt), attempt.trace, seat
    ):
        annotation, semantic, after = read_decision(
            env, observation, index, seat, runner, seed=seed
        )
        yield AnalysedDecision(
            event=index,
            played=events[index]["action"],
            annotation=annotation,
            semantic=semantic,
            after=after,
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


type AdmissionKey = tuple[tuple[str, ...], tuple[str, ...]]


class PolicyAdmissions:
    """Which of a registry's policies may analyse a game's setups, admitted once.

    Admitting every registered policy takes seconds with many checkpoints, and
    reads only the registered policies and each seat's deck, sideboard and
    world, never the game. So Play's Show analysis and Review's two analysis
    endpoints share one answer per (registered policy ids, setups).

    Guarded invariant: each key is admitted at most once; a second asker for a
    key being admitted waits for the first. Registering a policy changes the
    key. `PolicyRegistry.load` still admits, so a cached answer never stands
    in for the check at load.
    """

    def __init__(self, registry: PolicyRegistry) -> None:
        self.registry = registry
        self._lock = threading.Lock()
        self._keys: dict[AdmissionKey, threading.Lock] = {}
        self._admitted: dict[AdmissionKey, tuple[str, ...]] = {}

    def admitted(self, setup: managym.GameSetup) -> list[PolicyDescriptor]:
        """The registered policies, in registration order, that admit `setup`
        and read probabilities: neither search nor a scripted player."""
        descriptors = self.registry.descriptors()
        key = (
            tuple(descriptor.id for descriptor in descriptors),
            tuple(
                json.dumps(
                    [dict(player.decklist), dict(player.sideboard), player.world],
                    sort_keys=True,
                )
                for player in setup.players
            ),
        )
        with self._lock:
            admitting = self._keys.setdefault(key, threading.Lock())
        with admitting:
            ids = self._admitted.get(key)
            if ids is None:
                ids = tuple(
                    descriptor.id
                    for descriptor in descriptors
                    if self._admits(descriptor, setup)
                )
                self._admitted[key] = ids
        return [descriptor for descriptor in descriptors if descriptor.id in ids]

    def _admits(self, descriptor: PolicyDescriptor, setup: managym.GameSetup) -> bool:
        if descriptor.base_policy is not None or not descriptor.artifacts:
            # Search plays out from the walk's real hidden cards; a scripted
            # player has no probabilities to read.
            return False
        try:
            self.registry.admit(descriptor.id, setup)
        except (ImportError, OSError, ValueError):
            return False
        return True


_admissions: PolicyAdmissions | None = None


def policy_admissions() -> PolicyAdmissions:
    """The server registry's admissions, renewed when the registry is replaced."""
    global _admissions
    registry = policy_registry()
    if _admissions is None or _admissions.registry is not registry:
        _admissions = PolicyAdmissions(registry)
    return _admissions


def analysis_choices(
    attempt: RetainedAttempt, admissions: PolicyAdmissions
) -> AnalysisChoices:
    config = GameConfig(**attempt.trace["config"])
    if config.watch_players is not None:
        return AnalysisChoices([], None, "A watched game has no seat of yours.")
    try:
        open_walk(attempt)
    except AnalysisStop:
        return AnalysisChoices(
            [],
            None,
            "Analysis re-executes the game, which needs the build that recorded it "
            f"({recorded_build(attempt.world_json)}).",
        )
    return analysing_policies(config, admissions)


def analysing_policies(
    config: GameConfig, admissions: PolicyAdmissions
) -> AnalysisChoices:
    """The registered policies that can read `config`'s game: admitted for its
    decks and giving probabilities, without search. The default is the
    opponent's own policy when it is one of them."""
    policies = [
        {"sha256": descriptor.id, "name": Opponent(descriptor).name}
        for descriptor in admissions.admitted(config.to_rust())
    ]
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


# ———————————————— Analysis while you play


class LiveAnalysis:
    """A policy reading the person's decisions of a game in play, on its own walk.

    Guarded invariant: the walk's Env is the game's position whenever a
    decision is read, because steps and decisions arrive in the game's order on
    one queue and a worker thread alone applies them. Nothing reads the game's
    own Env, so a slow or failed pass never holds the game.

    The worker first loads the policy and walks the game so far (`seat_decisions`),
    asking the policy at each of the seat's earlier decisions so a
    decision-memory policy has the history a bot in the seat would have; then
    it applies each `advance` and reads each `decide`, giving `publish` one
    message per read. `publish` returns a future that is true once the message
    reached the person; only then is the decision `shown`. `close` stops the
    worker and the policy; nothing is published after it.
    """

    def __init__(
        self,
        attempt: RetainedAttempt,
        seat: int,
        policy: dict[str, str],
        load: Callable[[], PolicyRunner],
        publish: Callable[[dict[str, Any]], Future[bool]],
    ) -> None:
        self.policy = policy
        self._seat = seat
        self._seed = int(attempt.trace["config"]["seed"])
        self._publish = publish
        self._queue: queue.SimpleQueue[tuple[str, int, int] | None] = (
            queue.SimpleQueue()
        )
        self._lock = threading.Lock()
        self._shown: set[int] = set()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            args=(attempt, load),
            name="live-analysis",
            daemon=True,
        )
        self._thread.start()

    def advance(self, action: int) -> None:
        """The game applied `action` at its next step."""
        self._queue.put(("step", action, 0))

    def decide(self, event: int, revision: int) -> None:
        """The person must choose at `event`, published at protocol `revision`."""
        self._queue.put(("decide", event, revision))

    def shown(self, event: int) -> bool:
        """Whether the reading of `event` reached the person."""
        with self._lock:
            return event in self._shown

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._queue.put(None)

    def _send(self, message: dict[str, Any], event: int | None = None) -> None:
        with self._lock:
            if self._closed:
                return
        future = self._publish({"type": "analysis", "policy": self.policy, **message})
        if event is not None:

            def delivered(done: Future[bool]) -> None:
                if not done.cancelled() and done.exception() is None and done.result():
                    with self._lock:
                        if not self._closed:
                            self._shown.add(event)

            future.add_done_callback(delivered)

    def _run(self, attempt: RetainedAttempt, load: Callable[[], PolicyRunner]) -> None:
        runner: PolicyRunner | None = None
        try:
            runner = load()
            walker = open_walk(attempt)
            for index, env, observation in seat_decisions(
                walker, attempt.trace, self._seat
            ):
                ask(runner, env, observation, index)
            env = walker.env
            position = len(attempt.trace["events"])
            read: tuple[int, dict[str, Any]] | None = None
            while (item := self._queue.get()) is not None:
                kind, value, revision = item
                if kind == "step":
                    env.step(value)
                    position += 1
                    continue
                if value != position:
                    continue
                if read is None or read[0] != value:
                    # One read per decision: a republished prompt (new stops)
                    # is the same decision, and a recurrent policy is asked once.
                    observation = env.observation_for_player(self._seat)
                    annotation, semantic, after = read_decision(
                        env, observation, value, self._seat, runner, seed=self._seed
                    )
                    read = (
                        value,
                        {
                            "annotation": annotation.model_dump(mode="json"),
                            "semantic": semantic,
                            "after": after,
                        },
                    )
                self._send({"event": value, "revision": revision, **read[1]}, value)
        except AnalysisStop as stop:
            self._send({"stopped": str(stop)})
        except Exception as error:  # noqa: BLE001 - a failure stops this analysis only
            logging.getLogger(__name__).exception("Live analysis failed")
            self._send({"stopped": f"The analysis failed: {type(error).__name__}"})
        finally:
            if runner is not None:
                runner.close()


# ———————————————— Its line: the policy's first choice, played on

# A line runs to the end of the turn after the one its decision is in.
LINE_TURNS = 1
# A guard against a line that never reaches a turn's end.
MAX_LINE_STEPS = 400


class LineStep(BaseModel):
    """One move of a line: who made it, its name, and the analysed seat's
    board after it (`decision_surface_view`, the seat's own view)."""

    model_config = ConfigDict(extra="forbid")
    actor: Literal["agent", "opponent"]
    description: str
    projection: dict[str, Any]


class AgentLine(BaseModel):
    """What might have followed one decision had the policy chosen: its first
    choice at `event` (`offer`), then the policy in the seat and the game's own
    opponent in theirs. `end` says why the line stops; `winner` is set when the
    line ends the game. A sample, not a forecast: the opponent's choices and
    every later card come from one play-out of the real game's world."""

    model_config = ConfigDict(extra="forbid")
    event: int
    offer: int
    steps: list[LineStep]
    end: Literal["game_over", "turn", "limit"]
    winner: int | None


def agent_line(
    attempt: RetainedAttempt,
    decision: AnalysedDecision,
    seat: int,
    runner: PolicyRunner,
    opponent: BotPolicy,
) -> AgentLine:
    """The policy's line from `decision`, one of `seat`'s analysed decisions.

    Walks the record to the decision, plays the policy's first choice on a
    copy and continues: the policy's first choice at each of the seat's
    decisions, the opponent's own choice at theirs, to the end of the next
    turn. A recurrent policy (either seat's) is shown the history it would
    have had: it is asked at each of its seat's decisions on the way and its
    answer discarded. Priority passes fold into the move before them.
    The opponent's moves are named by their actions only once the game is
    complete, when both hands may show; before that they are "Opponent's move".
    Raises AnalysisStop where the game stops re-executing as recorded.
    """
    first = first_choice(decision.annotation.probabilities)
    if first is None:
        raise AnalysisStop("This decision has no probabilities to follow.")
    human = attempt.trace["config"].get("human_seat", 0)
    complete = attempt.trace["end_reason"] == "game_over"
    replayed = [
        policy
        for policy in (runner, opponent)
        if isinstance(policy, PolicyRunner) and policy.base_descriptor.recurrent
    ]
    for index, walked, observation in recorded_steps(open_walk(attempt), attempt.trace):
        if index == decision.event:
            break
        for policy in replayed:
            if policy.seat == int(observation.agent.player_index):
                policy(walked, observation)
    else:
        raise AnalysisStop(f"Step {decision.event + 1} was not recorded.")
    if int(observation.agent.player_index) != seat:
        raise AnalysisStop(f"Step {decision.event + 1} re-executed differently.")
    env = walked.clone_env()
    start = int(observation.turn.turn_number)
    moves: list[LineStep] = []
    action = first
    end: Literal["game_over", "turn", "limit"] = "limit"
    for _ in range(MAX_LINE_STEPS):
        actor = int(observation.agent.player_index)
        row = academy_actions(observation, human)[action]
        env.step(action)
        passed = row["type"] == "PRIORITY_PASS_PRIORITY"
        if moves and passed:
            # A pass belongs to the move before it, which shows the board after it.
            moves[-1] = moves[-1].model_copy(
                update={"projection": decision_surface_view(env, seat)}
            )
        else:
            moves.append(
                LineStep(
                    actor="agent" if actor == seat else "opponent",
                    description=row["description"]
                    if actor == seat or complete
                    else "Opponent's move",
                    projection=decision_surface_view(env, seat),
                )
            )
        if env.is_game_over():
            end = "game_over"
            break
        chooser = env.current_agent_index()
        assert chooser is not None
        observation = env.observation_for_player(chooser)
        if int(observation.turn.turn_number) > start + LINE_TURNS:
            end = "turn"
            break
        if chooser == seat:
            runner(env, observation)
            chosen = runner.last_decision
            assert chosen is not None
            action = first_choice(chosen.probabilities)
            action = chosen.action if action is None else action
        else:
            action = choose_bot(opponent, env, observation).action
    winner = env.winner_index() if end == "game_over" else None
    return AgentLine(
        event=decision.event, offer=first, steps=moves, end=end, winner=winner
    )


def first_choice(probabilities: Sequence[float] | None) -> int | None:
    """The offer with the most probability (the first on a tie), or None without any."""
    if not probabilities:
        return None
    return max(range(len(probabilities)), key=probabilities.__getitem__)
