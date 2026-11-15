"""Bot observation over GameSession's existing replay and Study authorities.

Controllers advance exactly one Command; inspection exposes only one acting
view. The mixed canonical record and retained roots remain server-private.
"""

from contextlib import ExitStack
from dataclasses import asdict, replace
import logging
from typing import Any, Literal, TypedDict

from fastapi import HTTPException
from pydantic import BaseModel, Field

from managym.bots import BotPolicy
from managym.policy import PolicyRunner
from managym.search import SearchConfig, SearchError

from . import server
from .attempts import AttemptStore, RecordedPlayer, bot_player
from .extensions import bot_extension
from .launch import refuse_training_receipt
from .opponent import policy_registry
from .replay_index import DecisionAddressV2, load_canonical_replay
from .trace import load_trace
from .villain import NATIVE_WATCH_PLAYERS, build_villain_policy, player_fields


class WatchCreated(TypedDict):
    id: str
    token: str
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
    seed: int = Field(default=0, ge=0, lt=2**64)
    villain_sims: int = Field(default=16, ge=1, le=4096)
    mirror: list[SearchConfig | None] | None = Field(
        default=None, min_length=2, max_length=2
    )
    decklists: Literal["open", "closed"] = "open"


def resolve_player(
    kind: str, request: WatchRequest, seat: int
) -> tuple[BotPolicy, RecordedPlayer]:
    """Resolve a seat once through Play's admission and policy construction."""
    if kind not in (*NATIVE_WATCH_PLAYERS, "passive") and not kind.startswith(
        ("policy:", "search-")
    ):
        kind = f"policy:{policy_registry().descriptor(kind).id}"
    data = {**request.model_dump(), "seed": (request.seed + seat) % 2**64}
    mirror = request.mirror[seat] if request.mirror else None
    data["mirror"] = asdict(mirror) if mirror is not None else None
    data.update(player_fields(kind))
    config = server._parse_game_config(data)
    policy = build_villain_policy(config, seat=seat)
    return policy, replace(
        bot_player(asdict(config)),
        seat=seat,
        deck=request.hero_deck if seat == 0 else request.villain_deck,
    )


def create_watch(
    request: WatchRequest, recorded_players: list[RecordedPlayer] | None = None
) -> WatchCreated:
    with ExitStack() as candidates:
        seats: list[tuple[BotPolicy, RecordedPlayer]] = []
        try:
            for seat, kind in enumerate(request.players):
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
        token = server.secrets.token_urlsafe(32)
        game = server.GameSession(watch_mode=True)
        game.attempt_owner = server.participant_id(token)
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
            record = server.SessionRecord(game.match_id, token, game)
            assert game.trace is not None
            game.trace.config.opponent = {
                "name": " vs ".join(player.name for _, player in seats),
                "kind": "watch",
            }
            game.trace.config.watch_players = request.players
            game._persist_attempt()
        except BaseException:
            game.close("setup_failed")
            raise
        candidates.pop_all()
    server.SESSION_REGISTRY[game.match_id] = record
    return {"id": game.match_id, "token": token, "finished": False}


def record_for_watch(trace_id: str, token: str | None) -> server.SessionRecord:
    record = server._live_record_for_trace(trace_id)
    server._rest_study_owner(record, token)
    if not record.game.watch_mode:
        raise HTTPException(409, "not_a_bot_watch")
    return record


def step_watch(trace_id: str, token: str | None, expected_revision: int) -> WatchStep:
    record = record_for_watch(trace_id, token)
    game = record.game
    assert game.env is not None and game.obs is not None
    if expected_revision != game.revision:
        raise HTTPException(409, "stale_watch_revision")
    if game.obs.game_over:
        return {"finished": True, "revision": game.revision}
    seat = int(game.obs.agent.player_index)
    context = (
        game._publish_current_prompt()
        if seat == 0
        else game._build_decision_context(game.obs, viewer=seat)
    )
    assert context is not None
    assert game.watch_policies is not None
    try:
        decision = game._choose_bot(game.watch_policies[seat])
    except SearchError:
        raise
    except ValueError as exc:
        # Nothing was committed; the pending decision stays for an explicit retry.
        raise HTTPException(
            422, f"Seat {seat + 1} policy could not choose: {exc}"
        ) from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Watch policy failed")
        raise HTTPException(
            500,
            f"Seat {seat + 1} policy failed ({type(exc).__name__}); see the server log.",
        ) from exc
    offer_id = next(
        key
        for key, index in context.action_by_offer.items()
        if index == decision.action
    )
    command = game._authority_command(context, offer_id, "policy")
    game._apply_bound_command(context, command, source="policy")
    game._persist_attempt(end_reason="game_over" if game.obs.game_over else None)
    return {"finished": bool(game.obs.game_over), "revision": game.revision}


def inspect_watch(
    trace_id: str, ordinal: int, token: str | None, reveal: bool = False
) -> dict[str, Any]:
    """Read a recorded decision; -1 restores the latest without advancing play."""
    payload = load_trace(trace_id)
    watched = payload["config"].get("watch_players")
    finished = (
        payload["end_reason"] == "game_over"
        and payload["final_observation"]["game_over"]
    )
    if not watched and not (finished and payload["config"].get("mirror") is not None):
        raise HTTPException(409, "not_a_bot_watch")
    record = None if finished else record_for_watch(trace_id, token)
    identities = (
        AttemptStore(server.trace_store.TRACES_DIR / "play.sqlite")
        .metadata(trace_id)
        .get("players", [])
    )
    mirror_by_seat = {
        identity["seat"]: (identity.get("inference") or {})
        .get("runtime", {})
        .get("config", {})
        .get("mirror")
        for identity in identities
    }
    mirrors = [mirror_by_seat.get(seat) for seat in (0, 1)]
    replay = load_canonical_replay(payload)
    if ordinal == -1:
        ordinal = max(0, len(replay.decisions) - 1)
    if ordinal < 0:
        raise HTTPException(404, "decision_not_found")
    if ordinal >= len(replay.decisions):
        if record is not None and not replay.decisions:
            pending = record.game.mirror_retry
            raise HTTPException(
                409,
                {
                    "code": pending["code"] if pending else "watch_pending",
                    "message": pending["message"]
                    if pending
                    else "No decision recorded yet. Step the pending bot decision explicitly.",
                    "revision": record.game.revision,
                    "mirror": mirrors,
                },
            )
        raise HTTPException(404, "decision_not_found")
    row = replay.decisions[ordinal]
    event = payload["events"][ordinal]
    address = DecisionAddressV2.from_decision(replay, row).serialize()
    observation = row.frame.projection.model_dump(mode="json")
    if reveal:
        observation = event["analysis_observation"]
    # Only the acting viewer's committed prefix belongs to this decision.
    # Keep the response bounded; unavailable older history is explicit in the UI.
    track = next(
        track for track in replay.presentation_tracks if track.viewer == row.viewer
    )
    presentation = track.events[
        max(
            0, row.presentation_cursor - server.MAX_PRESENTATION_EVENTS
        ) : row.presentation_cursor
    ]
    opponent = payload["config"].get("opponent") or {}
    return {
        "id": trace_id,
        "decision": {**row.model_dump(mode="json"), "address": address},
        "observation": observation,
        "outputs": event.get("bot_decision"),
        "presentation": [event.model_dump(mode="json") for event in presentation],
        "count": len(replay.decisions),
        "finished": finished,
        "revealed": reveal,
        "revision": row.revision,
        "winner": payload.get("winner"),
        "players": watched or [identity["name"] for identity in identities],
        "mirror": mirrors,
        "mirror_retry": record.game.mirror_retry if record is not None else None,
        "source_kind": "play"
        if not watched
        else opponent.get(
            "source_kind", "evaluation" if opponent.get("kind") == "arena" else None
        ),
        "identities": identities,
    }


def import_arena(game_record: dict[str, Any]) -> dict[str, Any]:
    """Verify a training or evaluation game, then project its Commands without policies.

    Source digests and receipt stay in their original namespace; the imported
    Academy match gets its own identity and retained Study roots.
    """
    from .trace import GameConfig

    extension = bot_extension()
    if extension is None:
        raise HTTPException(400, "arena_import_unavailable")
    refuse_training_receipt(game_record)
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
    record = record_for_watch(created["id"], created["token"])
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
        server._drop_session(game.match_id, "import_failed")
        raise
