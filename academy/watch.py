"""Bot observation over GameSession's existing replay and Study authorities.

Controllers advance exactly one Command; inspection exposes only one acting
view. The mixed canonical record and retained roots remain server-private.
"""

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, Literal, TypedDict, cast

from fastapi import HTTPException
from pydantic import BaseModel, Field

import managym

from . import server
from .attempts import AttemptStore, RecordedPlayer, bot_player
from .bot_decision import BeliefSummary, BotDecision
from .replay_index import DecisionAddressV2, load_canonical_replay
from .trace import load_trace
from .villain import VillainPolicy, build_villain_policy, choose_bot


class WatchCreated(TypedDict):
    id: str
    token: str
    finished: bool


class WatchStep(TypedDict):
    revision: int
    finished: bool


class WatchRequest(BaseModel):
    players: list[Literal["random", "passive", "greedy", "heuristic", "search"]] = (
        Field(default_factory=lambda: ["random", "random"], min_length=2, max_length=2)
    )
    hero_deck: str = "ur_lessons"
    villain_deck: str = "gw_allies"
    seed: int = Field(default=0, ge=0, lt=2**64)
    villain_sims: int = Field(default=16, ge=1, le=4096)


def _policy(kind: str, request: WatchRequest, seat: int) -> VillainPolicy:
    if kind in {"greedy", "heuristic"}:
        from manabot.env import Env
        from manabot.sim.heuristic import HeuristicPlayer
        from manabot.sim.players import ScriptedGreedyPlayer

        player = HeuristicPlayer() if kind == "heuristic" else ScriptedGreedyPlayer()

        def anchor(env: managym.Env, obs: managym.Observation) -> int:
            # Arena anchors read only last_raw_obs; no encoded features or engine
            # capability are supplied to this observation-only boundary.
            return player.act(cast(Env, SimpleNamespace(last_raw_obs=obs)), {})

        return anchor
    config = server._parse_game_config({**request.model_dump(), "villain_type": kind})
    config.seed = request.seed + seat
    return build_villain_policy(config)


def create_watch(
    request: WatchRequest, recorded_players: list[RecordedPlayer] | None = None
) -> WatchCreated:
    policies = (
        _policy(request.players[0], request, 0),
        _policy(request.players[1], request, 1),
    )
    token = server.secrets.token_urlsafe(32)
    game = server.GameSession(watch_mode=True)
    # Session closure releases controllers, including checkpoint tensors.
    # Policies never cross the trace or HTTP boundary.
    game.watch_policies = policies
    game.attempt_owner = server.participant_id(token)
    game.record_players = []
    for seat, kind in enumerate(request.players):
        config = asdict(
            server._parse_game_config(
                {
                    **request.model_dump(),
                    "villain_type": kind
                    if kind not in {"greedy", "heuristic"}
                    else "passive",
                }
            )
        )
        config["villain_type"] = kind
        game.record_players.append(
            replace(
                bot_player(config),
                seat=seat,
                deck=request.hero_deck if seat == 0 else request.villain_deck,
            )
        )
    if recorded_players is not None:
        game.record_players = recorded_players
    game.new_game(
        {**request.model_dump(), "villain_type": "random", "auto_pass": False}
    )
    record = server.SessionRecord(game.match_id, token, game)
    server.SESSION_REGISTRY[game.match_id] = record
    assert game.trace is not None
    # Persist bot seat identities, avoiding a fictional human in shared history.
    game.trace.config.opponent = {"name": " vs ".join(request.players), "kind": "watch"}
    game.trace.config.watch_players = request.players
    game._persist_attempt()
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
    game._bot_decision = choose_bot(game.watch_policies[seat], game.env, game.obs)
    offer_id = next(
        key
        for key, index in context.action_by_offer.items()
        if index == game._bot_decision.action
    )
    command = game._authority_command(context, offer_id, "policy")
    game._apply_bound_command(context, command, source="policy")
    game._persist_attempt(end_reason="game_over" if game.obs.game_over else None)
    return {"finished": bool(game.obs.game_over), "revision": game.revision}


def inspect_watch(
    trace_id: str, ordinal: int, token: str | None, reveal: bool = False
) -> dict[str, Any]:
    payload = load_trace(trace_id)
    if not payload["config"].get("watch_players"):
        raise HTTPException(409, "not_a_bot_watch")
    finished = (
        payload["end_reason"] == "game_over"
        and payload["final_observation"]["game_over"]
    )
    if not finished:
        record_for_watch(trace_id, token)
    replay = load_canonical_replay(payload)
    if ordinal >= len(replay.decisions):
        raise HTTPException(404, "decision_not_found")
    row = replay.decisions[ordinal]
    event = payload["events"][ordinal]
    address = DecisionAddressV2.from_decision(replay, row).serialize()
    observation = row.frame.projection.model_dump(mode="json")
    if reveal:
        observation = event["analysis_observation"]
    return {
        "id": trace_id,
        "decision": {**row.model_dump(mode="json"), "address": address},
        "observation": observation,
        "outputs": event.get("bot_decision"),
        "count": len(replay.decisions),
        "finished": finished,
        "revealed": reveal,
        "revision": row.revision,
        "winner": payload.get("winner"),
        "players": payload["config"]["watch_players"],
        "identities": AttemptStore(server.trace_store.TRACES_DIR / "play.sqlite")
        .metadata(trace_id)
        .get("players", []),
    }


def import_arena(game_record: dict[str, Any]) -> dict[str, Any]:
    """Verify an arena/Tournament game, then project its Commands without policies.

    Source digests and receipt stay in their original namespace; the imported
    Academy match gets its own identity and retained Study roots.
    """
    from manabot.sim.replay import replay_environment, replay_games

    from .trace import GameConfig

    receipt = replay_games([game_record])
    if not receipt.passed or game_record.get("failure") is not None:
        raise ValueError("Arena replay verification failed; import rejected")
    arena, _ = replay_environment(game_record)
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
        configs = arena.match.to_rust()
        assert game.trace is not None
        game.env = arena._engine
        game.obs = arena.last_raw_obs
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
                "name": str(game_record["match_id"]),
                "sha256": game_record["game_trace_sha256"],
            },
        )
        for expected in game_record["decisions"]:
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
            game._bot_decision = BotDecision(
                index,
                unavailable="Policy, value and search outputs were not retained in this arena recording.",
            )
            if expected.get("local_update_receipt") is not None:
                from manabot.sim.search.local_update import LocalUpdateReceipt

                receipt = LocalUpdateReceipt.from_json(expected["local_update_receipt"])
                if receipt.viewer != seat or len(receipt.target) != len(context.offers):
                    raise ValueError(
                        "Arena search evidence differs from the acting offers"
                    )
                game._bot_decision = BotDecision(
                    action=index,
                    probabilities=list(receipt.target),
                    search_scores=list(receipt.values),
                    belief=BeliefSummary(
                        receipt.belief_model,
                        receipt.belief_digest,
                        receipt.condition_mass,
                        len(
                            {
                                (rollout.world_index, rollout.world_seed)
                                for rollout in receipt.rollouts
                            }
                        ),
                    ),
                    unavailable="Search values are signed frozen-policy rollout estimates; no separate root critic was retained.",
                )
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
