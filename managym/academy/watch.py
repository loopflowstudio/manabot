"""Bot observation over GameSession's existing replay and Study authorities.

Controllers advance exactly one Command; inspection exposes only one acting
view. The mixed canonical record and retained roots remain server-private.
"""

from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
import logging
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from managym.recording import GameRecord

from fastapi import HTTPException
from pydantic import BaseModel, Field

from managym.bots import BotPolicy
from managym.disclosure import DecklistDisclosure
from managym.policy import PolicyRunner
from managym.search import SearchConfig, SearchError

from . import server
from .attempts import AttemptStore, RecordedPlayer, bot_player
from .deck_import import DeckImportInput
from .extensions import bot_extension
from .opponent import policy_registry
from .replay_index import DecisionAddressV2
from .villain import NATIVE_WATCH_PLAYERS, build_villain_policy, player_fields
from .work import Work


class WatchCreated(TypedDict):
    id: str
    finished: bool


class WatchStep(TypedDict):
    revision: int
    finished: bool


class WatchRequest(BaseModel):
    players: list[str] = Field(
        default_factory=lambda: ["random", "random"], min_length=2, max_length=2
    )
    hero_deck: str = "ur_lessons"
    villain_deck: str = "gw_allies"
    seed: int | None = Field(default=0, ge=0, lt=2**64)
    deck_import: DeckImportInput | None = None
    villain_sims: int = Field(default=16, ge=1, le=4096)
    mirror: list[SearchConfig | None] | None = Field(
        default=None, min_length=2, max_length=2
    )
    decklists: DecklistDisclosure = "open"


def resolve_player(
    kind: str, request: WatchRequest, seat: int
) -> tuple[BotPolicy, RecordedPlayer]:
    """Resolve a seat once through Play's admission and policy construction."""
    if kind not in (*NATIVE_WATCH_PLAYERS, "passive") and not kind.startswith(
        ("policy:", "search-")
    ):
        kind = f"policy:{policy_registry().descriptor(kind).id}"
    assert request.seed is not None
    data = {**request.model_dump(), "seed": (request.seed + seat) % 2**64}
    mirror = request.mirror[seat] if request.mirror else None
    data["mirror"] = asdict(mirror) if mirror is not None else None
    data.update(player_fields(kind))
    config = server._parse_game_config(data)
    policy = build_villain_policy(config, seat=seat)
    return policy, replace(
        bot_player(asdict(config)),
        seat=seat,
        deck=config.hero_deck_name if seat == 0 else config.villain_deck_name,
    )


def create_watch(
    request: WatchRequest,
    recorded_players: list[RecordedPlayer] | None = None,
    *,
    work: Work | None = None,
) -> WatchCreated:
    requested_seed = request.seed is not None
    if request.seed is None:
        request = request.model_copy(update={"seed": server.secrets.randbits(32)})
    try:
        expanded = server.resolve_pool_decks(request.model_dump(), seed=request.seed)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if expanded.get("deck_import") is not None and request.deck_import is None:
        request = request.model_copy(
            update={
                "deck_import": DeckImportInput.model_validate(expanded["deck_import"])
            }
        )
    with ExitStack() as candidates:
        seats: list[tuple[BotPolicy, RecordedPlayer]] = []
        try:
            for seat, kind in enumerate(request.players):
                if work is not None:
                    work.stage("load", loading_seat=seat)
                policy, player = resolve_player(kind, request, seat)
                seats.append((policy, player))
                if isinstance(policy, PolicyRunner):
                    candidates.callback(policy.close)
        except (ValueError, OSError, ImportError, SearchError) as error:
            detail = (
                {"code": error.reason, "message": str(error)}
                if isinstance(error, SearchError)
                else str(error)
            )
            raise HTTPException(400, detail) from error
        if work is not None:
            work.stage("advance")
        game = server.GameSession(watch_mode=True)
        game.record_players = (
            recorded_players
            if recorded_players is not None
            else [player for _, player in seats]
        )
        try:
            game.new_game(
                {
                    **request.model_dump(exclude={"mirror"}),
                    "villain_type": "random",
                    "auto_pass": False,
                }
            )
            game.watch_policies = (seats[0][0], seats[1][0])
            game._observe_policies()
            assert game.trace is not None
            game.trace.config.opponent = {
                "name": " vs ".join(player.name for _, player in seats),
                "kind": "watch",
            }
            game.trace.config.seed_was_requested = requested_seed
            game.trace.config.watch_players = request.players
            game._persist_attempt(config_changed=True)
        except BaseException:
            game.close("setup_failed")
            raise
        candidates.pop_all()
    game.work.finish()
    server.LIVE_GAMES[game.match_id] = server.LiveGame(game)
    return {"id": game.match_id, "finished": False}


@dataclass
class WatchSetupAttempt:
    """One start, named by its handle: status reads never load or recreate
    its policies."""

    request: WatchRequest
    work: Work
    result: WatchCreated | None = None
    error: str | None = None
    finished_at: float | None = None
    lock: Lock = field(default_factory=Lock)

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "work": self.work.payload(),
                "created": self.result,
                "error": self.error,
            }


_SETUP_LOCK = Lock()
_SETUP_ATTEMPTS: dict[str, WatchSetupAttempt] = {}


def setup_attempt(handle: str) -> WatchSetupAttempt:
    with _SETUP_LOCK:
        attempt = _SETUP_ATTEMPTS.get(handle)
    if attempt is None:
        raise HTTPException(404, "Watch setup unavailable. Start again explicitly.")
    return attempt


def start_setup(handle: str, request: WatchRequest) -> dict[str, Any]:
    """Start the Watch game `handle` names, once: the same request sent again
    reads its progress."""
    if (
        len(handle) < 32
        or len(handle) > 128
        or not handle.isascii()
        or not all(c.isalnum() or c in "-_" for c in handle)
    ):
        raise HTTPException(400, "invalid_setup_handle")
    with _SETUP_LOCK:
        previous = _SETUP_ATTEMPTS.get(handle)
        if previous is not None:
            if previous.request != request:
                raise HTTPException(
                    409, "Watch setup already binds another configuration."
                )
            return previous.payload()
        now = monotonic()
        finished = sorted(
            (
                (key, ended)
                for key, item in _SETUP_ATTEMPTS.items()
                if (ended := item.finished_at) is not None
            ),
            key=lambda item: item[1],
        )
        for index, (key, ended) in enumerate(finished):
            if now - ended > 900 or index < len(finished) - 127:
                del _SETUP_ATTEMPTS[key]
        if sum(item.finished_at is None for item in _SETUP_ATTEMPTS.values()) >= 16:
            raise HTTPException(429, "Watch setup capacity reached. Try again later.")
        attempt = WatchSetupAttempt(request, Work(handle))
        _SETUP_ATTEMPTS[handle] = attempt
    try:
        created = create_watch(request, work=attempt.work)
        with attempt.lock:
            attempt.result = created
    except Exception:
        # Providers may include local paths in exceptions. Keep those out of
        # this progress surface; log the traceback on the server and let the
        # native setup cleanup own candidate close.
        logging.getLogger(__name__).exception("Watch setup failed")
        status = attempt.work.read()
        operation = (
            f"Seat {status.loading_seat + 1} bot loading"
            if status is not None
            and status.kind == "load"
            and status.loading_seat is not None
            else "Watch setup"
        )
        with attempt.lock:
            attempt.error = f"{operation} failed. Your selections are unchanged. Start again explicitly."
        attempt.work.finish(failed=True)
    else:
        attempt.work.finish()
    finally:
        with attempt.lock:
            attempt.finished_at = monotonic()
    return attempt.payload()


def live_watch(trace_id: str) -> server.LiveGame:
    """The watched game in memory; one dropped from memory is over."""
    live = server._live_game_for_trace(trace_id)
    if not live.game.watch_mode:
        raise HTTPException(409, "not_a_bot_watch")
    return live


def step_watch(trace_id: str, expected_revision: int, retry: bool = False) -> WatchStep:
    game = live_watch(trace_id).game
    assert game.env is not None and game.obs is not None
    if expected_revision != game.revision:
        raise HTTPException(409, "stale_watch_revision")
    if game.obs.game_over:
        return {"finished": True, "revision": game.revision}
    if game.mirror_retry is not None and not retry:
        raise HTTPException(
            409, {**game.mirror_retry, "code": "explicit_retry_required"}
        )
    seat = int(game.obs.agent.player_index)
    context = (
        game._publish_current_prompt()
        if seat == 0
        else game._build_decision_context(game.obs, viewer=seat)
    )
    assert context is not None
    assert game.watch_policies is not None
    game.mirror_retry = None
    proposal = game._choose_bot(game.watch_policies[seat])
    if proposal is None:
        assert game.mirror_retry is not None
        if game.mirror_retry["code"].startswith(("search_", "mirror_")):
            raise SearchError(game.mirror_retry["code"], game.mirror_retry["message"])
        raise HTTPException(422, dict(game.mirror_retry))
    offer_id = next(
        key
        for key, index in context.action_by_offer.items()
        if index == proposal.action
    )
    command = game._authority_command(context, offer_id, "policy")
    game._apply_bound_command(context, command, source="policy")
    game._persist_attempt(end_reason="game_over" if game.obs.game_over else None)
    return {"finished": bool(game.obs.game_over), "revision": game.revision}


def inspect_watch(trace_id: str, ordinal: int, reveal: bool = False) -> dict[str, Any]:
    recording, metadata = server._recording_read(trace_id)
    payload = recording.payload
    finished = payload.get("end_reason") == "game_over" and bool(
        payload["final_observation"].get("game_over")
    )
    if not recording.watch and not recording.completed_mirror:
        raise HTTPException(409, "not_a_bot_watch")
    closed = finished or payload.get("end_reason") in {
        "record_interrupted",
        "record_diverged",
    }
    live = None if closed else live_watch(trace_id)
    identities = metadata.get("players", [])
    mirrors = server._recorded_mirrors(identities)
    watched = payload["config"].get("watch_players")
    replay = recording.replay
    if ordinal == -1:
        ordinal = max(0, len(replay.decisions) - 1) if replay else 0
    if replay is None or ordinal < 0 or ordinal >= len(replay.decisions):
        if live is not None and (replay is None or not replay.decisions):
            pending = live.game.mirror_retry
            raise HTTPException(
                409,
                {
                    "code": pending["code"] if pending else "watch_pending",
                    "message": pending["message"]
                    if pending
                    else "No decision recorded yet. Continue the pending bot decision explicitly.",
                    "revision": live.game.revision,
                    "mirror": mirrors,
                },
            )
        raise HTTPException(404, "decision_not_found")
    key = DecisionAddressV2.from_decision(replay, replay.decisions[ordinal]).serialize()
    position = recording.read(
        key, reveal=reveal, history_limit=server.MAX_PRESENTATION_EVENTS
    )
    opponent = payload["config"].get("opponent") or {}
    return {
        "id": trace_id,
        "decision": position["decision"],
        "observation": position["observation"],
        "outputs": position["outputs"],
        "presentation": position["history"],
        "count": len(replay.decisions),
        "finished": finished,
        "revealed": reveal,
        "revision": replay.decisions[ordinal].revision,
        "winner": payload.get("winner"),
        "players": watched or [identity["name"] for identity in identities],
        "mirror": mirrors,
        "mirror_retry": live.game.mirror_retry if live is not None else None,
        "source_kind": "play"
        if not watched
        else opponent.get(
            "source_kind", "evaluation" if opponent.get("kind") == "arena" else None
        ),
        "identities": metadata.get("players", []),
    }


def import_game(game_record: dict[str, Any]) -> dict[str, Any]:
    """Verify a training or evaluation game, then project its Commands without policies.

    Source digests and receipt stay in their original namespace; the imported
    Academy match gets its own identity and retained Study roots.
    """
    from managym.recording import records_from_source

    from .trace import GameConfig

    records = records_from_source(game_record)
    if records is not None:
        if len(records) != 1:
            raise ValueError("select exactly one game record")
        return import_record(records[0])
    extension = bot_extension()
    if extension is None:
        raise HTTPException(400, "arena_import_unavailable")
    arena = extension.verify_arena(game_record)
    # Verify the supplied bytes even when the source identity was imported before.
    existing = AttemptStore(
        server.trace_store.TRACES_DIR / "play.sqlite"
    ).arena_recording(game_record["game_trace_sha256"])
    if existing is not None:
        return {
            "id": existing,
            "finished": True,
            "source_sha256": game_record["game_trace_sha256"],
        }
    request = WatchRequest(
        players=["random", "random"],
        hero_deck="interactive",
        villain_deck="interactive",
        seed=game_record["deal_seed"],
    )
    created = create_watch(
        request,
        recorded_players=[
            RecordedPlayer(
                seat,
                name,
                "bot",
                name,
                game_record.get("seat_decks", ["interactive", "interactive"])[seat],
                inference={
                    "source_match": game_record["match_id"],
                    "source_sha256": game_record["game_trace_sha256"],
                },
            )
            for seat, name in enumerate(game_record["seat_players"])
        ],
    )
    record = live_watch(created["id"])
    game = record.game
    try:
        configs = arena.configs
        assert game.trace is not None
        game.env = arena.env
        game.obs = arena.observation
        game.published_prompt = None
        game.trace.config = GameConfig(
            hero_deck=dict(configs[0].decklist),
            villain_deck=dict(configs[1].decklist),
            hero_sideboard=dict(configs[0].sideboard),
            villain_sideboard=dict(configs[1].sideboard),
            villain_type="random",
            seed=game_record["deal_seed"],
            watch_players=list(game_record["seat_players"]),
            opponent={
                "kind": "arena",
                "source_kind": (game_record.get("source") or {}).get(
                    "kind", "evaluation"
                ),
                "name": str(game_record["match_id"]),
                "sha256": game_record["game_trace_sha256"],
            },
        )
        for expected, outputs in zip(
            game_record["decisions"], arena.decisions, strict=True
        ):
            seat = int(game.obs.agent.player_index)
            if game.env.state_digest() != expected["pre_state_digest"]:
                raise ValueError("Arena import root drifted")
            context = (
                game._publish_current_prompt()
                if seat == 0
                else game._build_decision_context(game.obs, viewer=seat)
            )
            assert context is not None
            index = int(expected["command"]["offer_id"])
            offer = next(
                key
                for key, action in context.action_by_offer.items()
                if action == index
            )
            if outputs.probabilities and len(outputs.probabilities) != len(
                context.offers
            ):
                raise ValueError("Arena search evidence differs from the acting offers")
            game._bot_decision = outputs
            game._apply_bound_command(
                context,
                game._authority_command(context, offer, "policy"),
                source="policy",
            )
            if game.env.state_digest() != expected["post_state_digest"]:
                raise ValueError("Arena import transition drifted")
        game._persist_attempt(end_reason="game_over")
        return {
            **created,
            "finished": True,
            "count": len(game.canonical_decisions),
            "source_sha256": game_record["game_trace_sha256"],
        }
    except Exception:
        server.end_game(game.match_id, "import_failed")
        raise


def import_record(record: "GameRecord") -> dict[str, Any]:
    """Project the verified prefix through GameSession's Command authority."""
    from managym.bots import BotDecision
    from managym.decision import Command
    from managym.replay import RecordedDecision, walk

    from .trace import BRANCHING_INTERVENTIONS, GameConfig, RecordingImport

    if not record.skip_trivial:
        raise ValueError("Academy requires forced-decision skipping")
    if record.resolved_player_policies != ("agent", "agent"):
        raise ValueError(
            "Academy requires both seats' surfaced decisions; native-seat records are not yet viewable"
        )
    replay = walk(record)
    source = record.sha256
    store = AttemptStore(server.trace_store.TRACES_DIR / "play.sqlite")
    existing = store.arena_recording(source)
    if existing is not None:
        return {"id": existing, "finished": True, "source_sha256": source}
    setups = replay.game.players
    if len({p.disclosure for p in setups}) != 1:
        raise ValueError("mixed deck disclosure is not supported by Watch")
    created = create_watch(
        WatchRequest(seed=replay.game.seed),
        recorded_players=[
            RecordedPlayer(
                seat, p.name, "bot", p.name, p.name, inference={"source_sha256": source}
            )
            for seat, p in enumerate(setups)
        ],
    )
    session = live_watch(created["id"]).game
    try:
        assert session.trace is not None
        session.env = replay.env
        session.trace.config = GameConfig(
            hero_deck=setups[0].decklist,
            villain_deck=setups[1].decklist,
            hero_sideboard=setups[0].sideboard,
            villain_sideboard=setups[1].sideboard,
            seed=replay.game.seed,
            villain_type="random",
            decklists=setups[0].disclosure,
            watch_players=[p.name for p in setups],
            opponent={
                "kind": "record",
                "name": "Recorded game",
                **(
                    {"source_kind": record.context["source_kind"]}
                    if record.context.get("source_kind") in ("training", "evaluation")
                    else {}
                ),
            },
        )

        def apply(decision: RecordedDecision) -> None:
            # The verified observation includes any interventions at this boundary.
            session.obs = decision.observation
            session.published_prompt = None
            seat = int(session.obs.agent.player_index)
            context = (
                session._publish_current_prompt()
                if seat == 0
                else session._build_decision_context(session.obs, viewer=seat)
            )
            assert context is not None
            annotation = decision.annotation
            probabilities = annotation.probabilities if annotation else None
            if probabilities is not None and len(probabilities) != len(context.offers):
                raise ValueError("recorded probabilities differ from legal offers")
            session._bot_decision = BotDecision(
                decision.chosen,
                probabilities=probabilities or [],
                search_scores=annotation.search_scores if annotation else [],
                belief=annotation.belief if annotation else None,
                value=annotation.value if annotation else None,
                update=annotation.update if annotation else None,
                unavailable="not retained" if annotation is None else None,
            )
            offer = next(
                key
                for key, action in context.action_by_offer.items()
                if action == decision.chosen
            )
            bound = session._authority_command(context, offer, "policy")
            native = decision.commitment
            if isinstance(native, Command):
                bound["command_id"] = native.command_id
                bound["answers"] = list(native.answers)
            step = len(session.trace.events)
            session._apply_bound_command(
                context,
                bound,
                source="policy",
                native_command=native.to_json()
                if isinstance(native, Command)
                else None,
            )
            branched = [
                item.op
                for item in decision.interventions
                if item.op in BRANCHING_INTERVENTIONS
            ]
            if branched:
                # Replaced, not edited: the trace re-encodes replaced events only.
                events = session.trace.events
                events[step] = replace(events[step], branched=branched)

        for _ in replay.run(apply):
            pass
        session.obs = replay.observation
        session.published_prompt = None
        assert replay.stop is not None
        stop = replay.stop
        reason = "game_over" if stop.status == "complete" else f"record_{stop.status}"
        session.trace.recording_import = RecordingImport(stop, replay.build_differs)
        session._persist_attempt(end_reason=reason)
        return {
            **created,
            "finished": True,
            "count": len(session.canonical_decisions),
            "stop": stop.status,
        }
    finally:
        server.end_game(session.match_id, "record_import_closed")
