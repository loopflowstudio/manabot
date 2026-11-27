"""Vectorized async match runner parity, batching, backfill and failure isolation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
import json
from pathlib import Path
from typing import Any, Literal

from etude.checkpoints import CheckpointRef
from etude.eval.agents import Forward, ForwardError
from etude.eval.records import (
    Benchmark,
    Code,
    Environment,
    Match,
    Seating,
    checkpoint_player,
)
from etude.eval.result import Result
from etude.eval.tournament import evaluating
from etude.records import ArtifactRef
import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace
from manabot.eval import agents as agent_module, vector as vector_runner
from manabot.eval.agent_loader import load
from manabot.eval.agents import build, local_forward
from manabot.eval.card_query import CardSelection, audit, card_data, card_db
from manabot.eval.pairing import PairingSpec, play_pairing
from manabot.eval.vector import LocalMatches, play_pairings
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.policy.prepare import Prepared
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import AuthoredMatchup, DeckMatchup
from manabot.sim.replay import read_trace, replay_failure, replay_game
import managym
from managym.recording import GameRecord, RecordedVectorEnv


@pytest.fixture
def cpu_eval(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin loader to 1-thread CPU for row-pure deterministic float32 parity."""
    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "cpu")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _fixture_checkpoint(
    directory: Path,
    seed: int,
    space: ObservationSpace,
    *,
    memory: Literal["none", "gru"] = "none",
    setup: managym.GameSetup | None = None,
) -> CheckpointRef:
    torch.manual_seed(seed)
    model = ActorCritic(
        space,
        ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=8,
            num_attention_heads=2,
            attention_on=True,
            attention_layers=1,
            policy_film=True,
            value_aggregation="masked_mean",
            decision_memory=memory,
        ),
    )
    path = directory / f"ckpt-{seed}-{memory}.safetensors"
    save_checkpoint(model, space, path, player_configs=setup)
    return CheckpointRef(
        experiment="fixture",
        job="vector",
        seed=seed,
        name=f"fixture-{seed}",
        at={"updates": 0},
        artifact=ArtifactRef(
            uri=path.as_uri(),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        ),
    )


def observed_forward(
    sync_forward: Callable[[list[object]], list[object]],
    batch_sizes: list[int],
) -> Forward:
    """Record the row count declared for each pre-collated input to the loader."""

    async def forward(inputs: list[object], *, rows: int | None = None) -> list[object]:
        assert len(inputs) == 1 and isinstance(inputs[0], Prepared)
        batch = inputs[0]
        assert rows == batch.batch
        batch_sizes.append(batch.batch)
        await asyncio.sleep(0)
        return sync_forward(inputs)

    return forward


def play_local(
    source: LocalMatches,
    spec_for: Callable[[Match], PairingSpec],
    *,
    slots: int,
    output_dir: Path,
) -> list[tuple[Match, Result | Exception]]:
    async def run() -> list[tuple[Match, Result | Exception]]:
        return [
            result
            async for result in play_pairings(
                source, spec_for, slots=slots, output_dir=output_dir
            )
        ]

    return asyncio.run(run())


def _canonical_decisions(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic per-decision fields independent of wall-clock latency."""
    return [
        {k: v for k, v in decision.items() if k != "latency_seconds"}
        for decision in game["decisions"]
    ]


def _canonical_game_summary(game: dict[str, Any]) -> dict[str, Any]:
    record = GameRecord.model_validate(game["record"])
    assert game["game_trace_sha256"] == record.sha256
    inputs = record.games[0]
    return {
        "match_id": game["match_id"],
        "deal_seed": game["deal_seed"],
        "leg": game["leg"],
        "seat_players": game["seat_players"],
        "seat_decks": game.get("seat_decks"),
        "match_hypers": game["match_hypers"],
        "observation_hypers": game["observation_hypers"],
        "registrations": game["registrations"],
        "execution_limits": game["execution_limits"],
        "runtime_identities": game["runtime_identities"],
        "winner": game["winner"],
        "terminated": game["terminated"],
        "truncated": game["truncated"],
        "termination_reason": game["termination_reason"],
        "failure": game["failure"],
        "seed": inputs.seed,
        "players": [p.model_dump(mode="json") for p in inputs.players],
        "actions": inputs.actions,
        "committed": inputs.committed,
        "digests": inputs.digests,
        "end": inputs.end.model_dump(mode="json") if inputs.end else None,
        "decisions": _canonical_decisions(game),
    }


def _strip_record_digests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            k: v
            for k, v in row.items()
            if k not in {"evidence_sha256", "evidence_bytes", "trace_sha256"}
        }
        for row in rows
    ]


def _allies_lessons_matchup() -> DeckMatchup:
    roles = ("ur_lessons", "gw_allies")
    first, second = (
        managym.authored_deck_setup("ur-lessons-vs-gw-allies", role) for role in roles
    )
    return DeckMatchup(
        population_sha256="a" * 64,
        entries=roles,
        lists=(
            managym.Deck(first.decklist, first.sideboard),
            managym.Deck(second.decklist, second.sideboard),
        ),
        decklists="closed",
    )


def test_slot_load_fresh_explicit_seed_matches_scalar_reset() -> None:
    matchup = _allies_lessons_matchup()
    setup_a = EnvMatch(matchup.match(0)).to_rust()
    setup_b = EnvMatch(matchup.match(1)).to_rust()
    space = ObservationSpace(ObservationConfig())

    vector = RecordedVectorEnv(
        num_envs=2,
        seed=0,
        skip_trivial=True,
        player_policies=("agent", "agent"),
        record_fraction=1.0,
        record_in_memory=True,
    )
    try:
        state = {
            "rewards": np.zeros(2, dtype=np.float64),
            "terminated": np.zeros(2, dtype=np.uint8),
            "truncated": np.zeros(2, dtype=np.uint8),
        }
        vector.set_observation_layout(state, "error", layout=space.layout)
        vector.reset_each_into_buffers([setup_a, setup_a], seeds=[11, 22])

        explicit_seed = 777_123
        loaded = vector.load_slots_into_buffers(
            [managym.SlotLoad.fresh(1, setup_b, seed=explicit_seed)]
        )
        assert len(loaded) == 1 and loaded[0].seed == explicit_seed
        assert vector.current_game_seeds() == [11, explicit_seed]
        assert vector.games[1].seed == explicit_seed

        scalar = managym.Env(seed=explicit_seed)
        scalar.reset(setup_b)
        assert vector.state_digests([False, True])[1] == scalar.state_digest()
    finally:
        vector.close()


def test_vector_runner_matches_play_pairing_against_heuristic_and_backfills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_eval: None,
) -> None:
    matchup = _allies_lessons_matchup()
    space = ObservationSpace(
        ObservationConfig(mana_payment="auto", mana_floats="usable")
    )
    ref = _fixture_checkpoint(tmp_path, 19, space)
    candidate = checkpoint_player(ref)
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config={"max_actions": 64}
    )

    deals = (123, 124)
    benchmark = Benchmark(
        name="vector-parity",
        environment=Environment(name="fixture", files="a" * 64),
        seatings=tuple(
            Seating(
                name=f"deal-{deal}",
                players=(candidate, heuristic),
                config={"deal": deal},
                games=1,
            )
            for deal in deals
        ),
    )

    serial_root = tmp_path / "serial"
    vector_root = tmp_path / "vector"
    track_vector = tmp_path / "track-vector"
    sync_forward = load(ref)

    with evaluating(
        benchmark, experiment="vector-parity", directory=track_vector
    ) as active:
        matches = list(active.matches)
        assert len(matches) == 2

        # 1. Serial baseline via play_pairing.
        monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(serial_root))
        serial_results: dict[str, Result] = {}
        for match in matches:
            deal = int(match.config["deal"])
            serial_results[match.key] = play_pairing(
                match,
                [sync_forward, build(heuristic, {})],
                matchup=matchup,
                deal=deal,
                matchup_mode="cross",
                game_seconds=60,
                arena="fixture-vector",
            )

        # 2. Vector runner with 6 slots over 8 total legs (2 matches x 4 legs)
        #    so slots 0..5 start full and the first two finished legs backfill
        #    legs 6 & 7 immediately without waiting for stragglers.
        batch_sizes: list[int] = []
        observed_active_counts: list[int] = []
        forwards_per_step: list[int] = []
        clock = 0.0
        elapsed_per_round = 0.1
        previous_forwards = 0
        original_step = RecordedVectorEnv.step_isolated_into_buffers

        def spy_step(
            self: RecordedVectorEnv,
            actions: list[int],
            active: list[bool] | None = None,
            recover_capacity: bool = False,
            *,
            annotations: list[dict[str, object] | None] | None = None,
        ) -> None:
            nonlocal clock, previous_forwards
            if active is not None:
                observed_active_counts.append(sum(active))
            forwards_per_step.append(len(batch_sizes) - previous_forwards)
            previous_forwards = len(batch_sizes)
            clock += elapsed_per_round
            original_step(
                self,
                actions,
                active,
                recover_capacity,
                annotations=annotations,
            )

        monkeypatch.setattr(RecordedVectorEnv, "step_isolated_into_buffers", spy_step)
        finalize = vector_runner.finalize_match
        replay_pause = 0.0
        steps_at_settlement: list[int] = []

        def delayed_finalize(active_match: vector_runner.ActiveMatch) -> Result:
            nonlocal replay_pause
            replay_pause += 90.0
            steps_at_settlement.append(len(observed_active_counts))
            return finalize(active_match)

        # The first completed deal's replay takes longer than the game budget.
        # Other live legs should continue once the coordinator resumes stepping.
        monkeypatch.setattr(vector_runner, "monotonic", lambda: clock + replay_pause)
        monkeypatch.setattr(vector_runner, "finalize_match", delayed_finalize)
        reads = 0
        read_checkpoint = agent_module.read_checkpoint_inputs

        def count_checkpoint(
            path: Path, digest: str | None, size: int | None
        ) -> agent_module.CheckpointInputs:
            nonlocal reads
            reads += 1
            return read_checkpoint(path, digest, size)

        monkeypatch.setattr(agent_module, "read_checkpoint_inputs", count_checkpoint)
        source = LocalMatches(
            [
                (
                    match,
                    [
                        observed_forward(sync_forward, batch_sizes),
                        build(heuristic, {}),
                    ],
                )
                for match in matches
            ]
        )

        yielded = play_local(
            source,
            lambda match: PairingSpec(
                matchup=matchup,
                deal=int(match.config["deal"]),
                matchup_mode="cross",
                game_seconds=60,
                arena="fixture-vector",
            ),
            slots=6,
            output_dir=vector_root,
        )
        assert len(yielded) == 2
        assert steps_at_settlement[0] < len(observed_active_counts)

        vector_results: dict[str, Result] = {}
        for match, outcome in yielded:
            assert isinstance(outcome, Result), outcome
            vector_results[match.key] = outcome
            active.store(match, outcome)

    assert reads == 1
    assert max(forwards_per_step) == 1
    # Pre-collated batches carried more than one row.
    assert max(batch_sizes) > 1
    # With eight legs, all six slots start full and only drain after the queue empties.
    assert observed_active_counts[0] == 6
    assert observed_active_counts == sorted(observed_active_counts, reverse=True)

    charged_seconds = 0.0
    # Exact trace, Command, digest, score, behaviour and card parity per deal.
    for match in matches:
        serial_res = serial_results[match.key]
        vector_res = vector_results[match.key]
        assert vector_res.tables["plays"] == serial_res.tables["plays"]
        assert vector_res.tables["behaviour"] == serial_res.tables["behaviour"]
        assert vector_res.tables["cards"] == serial_res.tables["cards"]
        assert vector_res.tables["card_counts"] == serial_res.tables["card_counts"]
        assert _strip_record_digests(
            vector_res.tables["card_plays"]
        ) == _strip_record_digests(serial_res.tables["card_plays"])
        assert len(vector_res.blobs) == 4

        serial_games = read_trace(
            serial_root / "arena" / match.key / "games" / "candidate__opponent.jsonl"
        )
        vector_games = read_trace(
            vector_root / "arena" / match.key / "games" / "candidate__opponent.jsonl"
        )
        assert len(vector_games) == len(serial_games) == 4
        for s_game, v_game in zip(serial_games, vector_games, strict=True):
            charged_seconds += v_game["execution_seconds"]
            assert _canonical_game_summary(v_game) == _canonical_game_summary(s_game)
            if v_game["seat_players"][0] == "candidate":
                assert v_game["match_hypers"]["mana_payment"] == ["auto", "manual"]
                assert v_game["match_hypers"]["mana_floats"] == ["usable", "off"]
            else:
                assert v_game["match_hypers"]["mana_payment"] == ["manual", "auto"]
                assert v_game["match_hypers"]["mana_floats"] == ["off", "usable"]
            receipt, _ = replay_game(v_game)
            assert receipt.passed

        vector_eval = json.loads(
            (vector_root / "arena" / match.key / "evaluation.json").read_bytes()
        )
        assert vector_eval["status"] == "completed"
        assert vector_eval["error"] is None

    assert charged_seconds == pytest.approx(
        len(observed_active_counts) * elapsed_per_round
    )

    with card_db("vector-parity", track_vector) as db:
        data = card_data(db, CardSelection(experiment="vector-parity"))
        assert data.games == 16 and data.cards
        checked = audit(db, track_vector)
        assert checked.games == 8
        assert checked.problems == ()


def test_vector_runner_matches_play_pairing_with_gru_memory_and_blanking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_eval: None,
) -> None:
    matchup = AuthoredMatchup(
        pack_key="mtg122-complexity-v1",
        role_a="l1_swift",
        role_b="l1_heavy",
        decklists="closed",
        world=managym.WORLD_SETUP,
    )
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=512,
            max_decision_reference_details=16384,
            max_decision_references=256,
            max_decision_members=8192,
        )
    )
    ref = _fixture_checkpoint(tmp_path, 21, space, memory="gru")
    candidate = checkpoint_player(ref)
    opponent = candidate.model_copy(update={"config": {"blank_decision_memory": True}})
    match = Match(players=(candidate, opponent), games=4)

    sync_forward = load(ref)
    serial_root = tmp_path / "serial-mem"
    vector_root = tmp_path / "vector-mem"

    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(serial_root))
    serial_res = play_pairing(
        match,
        [sync_forward, sync_forward],
        matchup=matchup,
        deal=1,
        matchup_mode="cross",
        game_seconds=60,
        arena="fixture-memory",
    )

    batch_sizes: list[int] = []
    async_forward = observed_forward(sync_forward, batch_sizes)
    source = LocalMatches(
        [(match, [async_forward, build(opponent, {ref.key: async_forward})])]
    )

    [(got_match, got_result)] = play_local(
        source,
        lambda _: PairingSpec(
            matchup=matchup,
            deal=1,
            matchup_mode="cross",
            game_seconds=60,
            arena="fixture-memory",
        ),
        slots=4,
        output_dir=vector_root,
    )
    assert got_match == match
    assert isinstance(got_result, Result), got_result
    assert batch_sizes[0] == 4
    assert got_result.tables["plays"] == serial_res.tables["plays"]

    serial_games = read_trace(
        serial_root / "arena" / match.key / "games" / "candidate__opponent.jsonl"
    )
    vector_games = read_trace(
        vector_root / "arena" / match.key / "games" / "candidate__opponent.jsonl"
    )
    for s_game, v_game in zip(serial_games, vector_games, strict=True):
        assert _canonical_game_summary(v_game) == _canonical_game_summary(s_game)
        receipt, _ = replay_game(v_game)
        assert receipt.passed


def test_vector_runner_isolates_native_crash_and_keeps_other_legs_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_eval: None,
) -> None:
    matchup = _allies_lessons_matchup()
    space = ObservationSpace(ObservationConfig())
    ref = _fixture_checkpoint(tmp_path, 31, space)
    candidate = checkpoint_player(ref)
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config={"max_actions": 64}
    )
    matches = [
        Match(players=(candidate, heuristic), games=4, config={"deal": deal})
        for deal in (201, 202)
    ]

    # Trigger a real native AgentError on slot 1 (leg 1 of deal 201) on the second
    # vector step by sending an out-of-bounds action index only for that slot.
    original_step = RecordedVectorEnv.step_isolated_into_buffers
    step_count = 0

    def inject_native_error(
        self: RecordedVectorEnv,
        actions: list[int],
        active: list[bool] | None = None,
        recover_capacity: bool = False,
        *,
        annotations: list[dict[str, object] | None] | None = None,
    ) -> None:
        nonlocal step_count
        step_count += 1
        patched = list(actions)
        if step_count == 2:
            patched[1] = 999_999
        original_step(
            self,
            patched,
            active,
            recover_capacity,
            annotations=annotations,
        )

    monkeypatch.setattr(
        RecordedVectorEnv, "step_isolated_into_buffers", inject_native_error
    )

    forward = local_forward(load(ref))
    source = LocalMatches(
        [(match, [forward, build(heuristic, {})]) for match in matches]
    )
    vector_root = tmp_path / "crash-isolation"

    settled = play_local(
        source,
        lambda match: PairingSpec(
            matchup=matchup,
            deal=int(match.config["deal"]),
            matchup_mode="cross",
            game_seconds=60,
            arena="fixture-crash",
        ),
        slots=4,
        output_dir=vector_root,
    )
    results = {match.key: outcome for match, outcome in settled}
    assert len(results) == 2

    crashed_res = results[matches[0].key]
    healthy_res = results[matches[1].key]
    assert isinstance(crashed_res, Result), crashed_res
    assert isinstance(healthy_res, Result), healthy_res

    # Match 0: leg 1 failed with null score, while legs 0, 2, 3 stayed valid.
    leg1_rows = [r for r in crashed_res.tables["plays"] if r["game"] == 1]
    other_rows = [r for r in crashed_res.tables["plays"] if r["game"] != 1]
    assert len(leg1_rows) == 2
    assert all(not r["valid"] and r["score"] is None for r in leg1_rows)
    assert all("AgentError" in str(r["failure"]) for r in leg1_rows)
    assert len(other_rows) == 6 and all(r["valid"] for r in other_rows)

    # Match 1 (backfilled into the poisoned slot 1 too!): all 4 legs valid.
    assert len(healthy_res.tables["plays"]) == 8
    assert all(r["valid"] for r in healthy_res.tables["plays"])

    # Failure evidence on disk replays and matches the injected native error.
    crashed_games = read_trace(
        vector_root / "arena" / matches[0].key / "games" / "candidate__opponent.jsonl"
    )
    failed_leg = next(g for g in crashed_games if g["leg"] == 1)
    assert failed_leg["failure"] is not None
    assert (
        vector_root
        / "arena"
        / matches[0].key
        / "failures"
        / "candidate__opponent.201.1.jsonl"
    ).is_file()
    reproduced = replay_failure(failed_leg)
    assert reproduced["status"] == "failed"
    assert reproduced["at_retained_failure"] is True
    assert reproduced["matches_binding"] is True
    assert reproduced["matches_error"] is True


def test_vector_runner_isolates_admission_and_checkpoint_failures(
    tmp_path: Path,
    cpu_eval: None,
) -> None:
    matchup = _allies_lessons_matchup()
    space = ObservationSpace(ObservationConfig())
    ref = _fixture_checkpoint(tmp_path, 41, space)
    candidate = checkpoint_player(ref)
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config={"max_actions": 64}
    )

    # 1. Invalid match yields Exception without breaking sibling valid match.
    bad_match = Match(players=(candidate, heuristic), games=99, config={"deal": 300})
    good_match = Match(players=(candidate, heuristic), games=4, config={"deal": 301})
    forward = local_forward(load(ref))

    def spec_for(match: Match) -> PairingSpec:
        return PairingSpec(
            matchup=matchup,
            deal=int(match.config["deal"]),
            matchup_mode="cross",
            game_seconds=60,
            arena="fixture-mixed",
        )

    opponent = build(heuristic, {})
    mixed = {
        match.key: outcome
        for match, outcome in play_local(
            LocalMatches(
                [(match, [forward, opponent]) for match in (bad_match, good_match)]
            ),
            spec_for,
            slots=4,
            output_dir=tmp_path / "mixed",
        )
    }
    assert isinstance(mixed[bad_match.key], ValueError)
    assert isinstance(mixed[good_match.key], Result)

    failed_ref = _fixture_checkpoint(tmp_path, 42, space)
    failed_match = Match(
        players=(checkpoint_player(failed_ref), heuristic),
        games=4,
        config={"deal": 302},
    )
    error = ForwardError(failed_ref.key, "checkpoint unavailable")

    async def broken_checkpoint(
        inputs: list[object], *, rows: int | None = None
    ) -> list[object]:
        assert rows is not None and rows >= 1
        raise error

    settled = play_local(
        LocalMatches(
            [
                (failed_match, [broken_checkpoint, opponent]),
                (good_match, [forward, opponent]),
            ]
        ),
        spec_for,
        slots=5,
        output_dir=tmp_path / "checkpoint-failure",
    )
    outcomes = {match.key: outcome for match, outcome in settled}
    assert len(settled) == 2
    assert outcomes[failed_match.key] is error
    assert isinstance(outcomes[good_match.key], Result)

    async def malformed_logits(
        inputs: list[object], *, rows: int | None = None
    ) -> list[object]:
        assert rows is not None
        return [np.full((rows, space.config.max_actions), np.nan)]

    outcomes = {
        match.key: outcome
        for match, outcome in play_local(
            LocalMatches(
                [
                    (failed_match, [malformed_logits, opponent]),
                    (good_match, [forward, opponent]),
                ]
            ),
            spec_for,
            slots=5,
            output_dir=tmp_path / "malformed-logits",
        )
    }
    malformed = outcomes[failed_match.key]
    assert isinstance(malformed, ForwardError)
    assert malformed.checkpoint == failed_ref.key
    assert str(malformed) == "forward returned malformed legal logits"
    assert isinstance(outcomes[good_match.key], Result)

    # An unexpected forward exception still escapes the generator.
    async def exploding_forward(
        inputs: list[object], *, rows: int | None = None
    ) -> list[object]:
        raise RuntimeError("inference server unavailable")

    with pytest.raises(RuntimeError, match="inference server unavailable"):
        play_local(
            LocalMatches([(good_match, [exploding_forward, opponent])]),
            spec_for,
            slots=4,
            output_dir=tmp_path / "explode",
        )


def test_vector_runner_bounds_deferred_layouts_before_play(
    tmp_path: Path,
    cpu_eval: None,
) -> None:
    checkpoint = _fixture_checkpoint(tmp_path, 51, ObservationSpace())
    different_layout = _fixture_checkpoint(
        tmp_path, 52, ObservationSpace(ObservationConfig(max_actions=32))
    )
    heuristic = Code(name="heuristic")
    matches = [
        Match(
            players=(checkpoint_player(ref), heuristic),
            games=4,
            config={"deal": deal},
        )
        for deal, ref in enumerate([checkpoint, *([different_layout] * 6)])
    ]

    async def forward(inputs: list[object], *, rows: int | None = None) -> list[object]:
        [batch] = inputs
        assert isinstance(batch, Prepared) and rows == batch.batch
        return [np.zeros(batch.legal.shape, dtype=np.float32)]

    class CountingMatches(LocalMatches):
        granted = 0

        async def take(self, limit: int) -> list[tuple[Match, Sequence[object]]]:
            granted = await super().take(limit)
            self.granted += len(granted)
            return granted

    source = CountingMatches(
        [(match, [forward, build(heuristic, {})]) for match in matches]
    )
    matchup = _allies_lessons_matchup()

    def spec_for(match: Match) -> PairingSpec:
        return PairingSpec(
            matchup=matchup,
            deal=int(match.config["deal"]),
            matchup_mode="cross",
            game_seconds=60,
            max_commands=1,
            arena="fixture-layouts",
        )

    async def run() -> list[tuple[str, int]]:
        settled = []
        async for match, outcome in play_pairings(
            source, spec_for, slots=5, output_dir=tmp_path
        ):
            assert isinstance(outcome, Result), outcome
            settled.append((match.key, source.granted))
        return settled

    settled = asyncio.run(run())
    assert settled[0] == (matches[0].key, 6)
    assert {key for key, _ in settled} == {match.key for match in matches}
