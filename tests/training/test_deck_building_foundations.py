"""Small end-to-end contracts for pools, Limited builds and public game results."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.tla_corpus import Corpus, GameEvidence, Source, Trophy
from experiments.tla_pools import derive_pools, main as pools_main
from manabot import managym
from manabot.game_results import read_game_results
from manabot.infra.config import ModelConfig
from manabot.infra.identity import file_sha256
from manabot.managym.format import LimitedBuild
from manabot.pools import PoolCorpus
from manabot.training.decks import UniformDecks
from manabot.training.execution import execute_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
from managym.format import Limited


def test_limited_builds_select_from_a_pool_and_reject_bad_input() -> None:
    pool = managym.CardPool({"Llanowar Elves": 6})
    limited = managym.format.Limited()
    build = limited.build(pool)
    for _ in range(6):
        build.select("Llanowar Elves")
    with pytest.raises(ValueError, match="not available"):
        build.select("Llanowar Elves")
    assert build.deck == {"Llanowar Elves": 6}  # A rejected selection changes nothing.
    for _ in range(34):
        build.select("Forest")
    assert build.can_stop and build.available == dict.fromkeys(
        limited.basic_cards(), 20
    )
    assert limited.issues(pool, build.deck) == []
    assert limited.issues(pool, {"Forest": 61}) == []  # Construction is not the cap.
    assert limited.issues(pool, {"Llanowar Elves": 7, "Forest": 40})
    capped = limited.build(pool, max_cards=40)
    for _ in range(40):
        capped.select("Forest")
    assert capped.available == {} and capped.can_stop
    assert set(limited.basic_cards()) >= {"Forest", "Island"}
    with pytest.raises(ValueError):
        limited.build(pool, max_cards=39)
    for cards in ({"Ally": 1}, {"Not a Card": 1}, {"Forest": 0}):
        with pytest.raises(ValueError):
            managym.CardPool(cards)
    for count in (True, 1.5, -1, 0, "2"):
        with pytest.raises(ValueError, match="Forest: count"):
            managym.CardPool({"Forest": count})


def _card_axis() -> list[str]:
    return [d["name"] for d in json.loads(managym.card_definitions_json())]


def test_array_rules_agree_with_the_name_form_at_every_step() -> None:
    names = _card_axis()
    limited = managym.format.Limited()
    pool = managym.CardPool({"Llanowar Elves": 2, "Island": 1})
    assert limited.min_cards == 40 and pool.counts.shape == (len(names),)
    assert managym.CardPool.from_counts(pool.counts).cards == pool.cards
    assert {names[i] for i in np.flatnonzero(limited.basics)} == set(
        limited.basic_cards()
    )
    build = limited.build(pool, max_cards=42)
    picks = ["Llanowar Elves", "Island", "Llanowar Elves"] + ["Forest"] * 39
    pools, decks = [], []
    for pick in [*picks, None]:
        deck = np.zeros(len(names), dtype=np.int64)
        for name, count in build.deck.items():
            deck[names.index(name)] = count
        pools.append(pool.counts)
        decks.append(deck)
        if pick is not None:
            build.select(pick)
    available, can_stop = limited.available(np.stack(pools), np.stack(decks), 42)
    legal = limited.legal(np.stack(pools), np.stack(decks))
    assert available.dtype == np.int32 and can_stop.dtype == np.bool_
    assert list(legal) == list(can_stop) == [n >= 40 for n in range(len(picks) + 1)]
    replay = limited.build(pool, max_cards=42)
    for row, pick in zip(available, [*picks, None]):
        assert {names[i]: int(row[i]) for i in np.flatnonzero(row)} == replay.available
        if pick is not None:
            replay.select(pick)
    assert not available[-1].any()


def test_array_rules_reject_unreachable_rows_and_report_illegal_decks() -> None:
    names = _card_axis()
    limited = managym.format.Limited()
    pools = np.zeros((2, len(names)), dtype=np.int32)
    decks = np.zeros_like(pools)
    decks[1, names.index("Llanowar Elves")] = 1
    assert list(limited.legal(pools, decks)) == [False, False]
    with pytest.raises(ValueError, match="row 1: Llanowar Elves: not in the pool"):
        limited.available(pools, decks)
    for bad in (decks[0], decks.astype(np.float32), decks[:, :-1], -decks):
        with pytest.raises(ValueError):
            limited.available(pools, bad)
    with pytest.raises(ValueError, match="rows"):
        limited.legal(pools, decks[:1])
    token = np.zeros(len(names), dtype=np.int32)
    token[names.index("Ally")] = 1
    with pytest.raises(ValueError, match="Ally: not supported"):
        managym.CardPool.from_counts(token)
    assert not limited.legal(pools[:1], token[None])[0]


def _pool_source(event_ids: tuple[str, ...] = ("a", "b", "c")) -> Corpus:
    return Corpus(
        target=len(event_ids),
        source=Source(
            retrieved_at="2026-11-11",
            etag=None,
            last_modified=None,
            sha256="a" * 64,
            bytes=1,
        ),
        world={"version": "historical"},
        scanned_rows=21,
        exhausted=False,
        trophies=tuple(
            Trophy(
                event_id=name,
                draft_time="2026-11-11",
                games=tuple(
                    GameEvidence(
                        row=i,
                        match_number=i,
                        game_time="2026-11-11",
                        build_index=1,
                        won=True,
                    )
                    for i in range(1, 8)
                ),
                decklist=managym.Decklist(
                    f"Main\n6 Llanowar Elves\n{34 + (name == 'c')} Forest\nSideboard\n4 Llanowar Elves\n2 Island\n"
                ),
                deck=None,  # Earlier admission must not gate current-world rebuilding.
            )
            for name in event_ids
        ),
    )


def test_format_imports_as_a_module() -> None:
    assert Limited is managym.format.Limited
    assert LimitedBuild is managym.format.LimitedBuild
    assert Limited.__module__ == "managym.format"


def test_pool_derivation_keeps_drafts_human_builds_and_frozen_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pool_source()
    path = tmp_path / "source.json"
    payload = source.model_dump(mode="json")
    payload["trophies"][0]["decklist"]["issues"] = [
        {
            "line": 2,
            "section": "main",
            "card_name": "Llanowar Elves",
            "code": "unknown",
            "message": "unknown in the historical world",
        }
    ]
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    corpus = derive_pools(path, held_out=1)
    assert derive_pools(path, held_out=1) == corpus
    assert path.read_bytes() == before
    assert [pool.id for pool in corpus.pools] == ["a", "b", "c"]
    assert sum(pool.split == "held_out" for pool in corpus.pools) == 1
    assert all(pool.cards == {"Llanowar Elves": 10} for pool in corpus.pools)
    assert [sum(pool.human_build.values()) for pool in corpus.pools] == [40, 40, 41]
    assert corpus.pools[0].human_build == corpus.pools[1].human_build
    assert corpus.provenance.human_build_indexes == {"a": 1, "b": 1, "c": 1}
    output = tmp_path / "pools-v2" / "corpus.json"
    monkeypatch.setattr(
        sys, "argv", ["tla_pools", str(path), str(output), "--held-out", "1"]
    )
    pools_main()
    with pytest.raises(SystemExit, match="choose a new versioned path"):
        pools_main()
    assert PoolCorpus.read(output).digest == corpus.digest
    for damage in (
        lambda c: c["pools"][0].update(
            split="train" if c["pools"][0]["split"] == "held_out" else "held_out"
        ),
        lambda c: c["pools"][0]["human_build"].update({"Llanowar Elves": 11}),
        lambda c: c["pools"][0]["cards"].update({"Forest": 1}),
    ):
        damaged = corpus.model_dump(mode="json")
        damage(damaged)
        with pytest.raises(ValueError):
            PoolCorpus.model_validate(damaged)


def test_unknown_cards_stay_in_their_pools_without_moving_the_split(
    tmp_path: Path,
) -> None:
    source = _pool_source(("a", "b", "c", "d"))
    path = tmp_path / "source.json"
    path.write_text(source.model_dump_json())
    complete = derive_pools(path, held_out=2)
    unknown = {"a", "c"}
    payload = source.model_dump(mode="json")
    for trophy in payload["trophies"]:
        if trophy["event_id"] in unknown:
            trophy["decklist"] = managym.Decklist(
                "Main\n40 Forest\n1 Unknown main card\nSideboard\n1 Unknown sideboard card\n"
            ).to_dict()
    path.write_text(json.dumps(payload))
    corpus = derive_pools(path, held_out=2)
    assert [p.split for p in corpus.pools] == [p.split for p in complete.pools]
    assert corpus.pools[0].cards == {
        "Unknown main card": 1,
        "Unknown sideboard card": 1,
    }
    assert corpus.pools[0].human_build == {"Forest": 40, "Unknown main card": 1}
    playable = [
        p.id
        for p in corpus.pools
        if not managym.unsupported_cards([*p.cards, *p.human_build])
    ]
    assert playable == ["b", "d"]


def test_pool_and_result_types_import_no_engine_or_training() -> None:
    probe = (
        "import sys; import manabot.pools, manabot.game_results; "
        "loaded = [m for m in sys.modules if m.startswith(('manabot.managym', "
        "'manabot.training', 'manabot.model', 'torch'))]; assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_pool_derivation_does_not_exclude_malformed_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _pool_source().model_dump(mode="json")
    payload["trophies"][0]["decklist"] = managym.Decklist("Main\n0 Forest\n").to_dict()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(managym.DeckImportError) as rejected:
        derive_pools(path, held_out=1)
    output = tmp_path / "pools.json"
    monkeypatch.setattr(
        sys, "argv", ["tla_pools", str(path), str(output), "--held-out", "1"]
    )
    with pytest.raises(SystemExit) as cli:
        pools_main()
    assert str(cli.value) == f"No corpus written: {rejected.value}"
    assert not output.exists()


def _tiny_regime() -> TrainingRegime:
    return TrainingRegime(
        id="foundation-proof",
        world=managym.WORLD_VERSION,
        agent=ModelConfig(
            hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP
        ),
        deck_source=UniformDecks(
            decks=(managym.Deck({"Forest": 10}), managym.Deck({"Forest": 10})),
            pools=("pool-a", "pool-a"),
            builds=(None, "build-a-1"),
            sampling="uniform-independent-v1",
        ),
        schedule_clock="iteration_fraction",
        stages=[
            TrainSelfPlay(
                id="play",
                operation="train_self_play",
                trainable="none",
                streams=2,
                transitions=1,
                updates=1,
                games_per_stream=1,
            )
        ],
    )


def test_self_play_without_recovery_emits_public_game_rows(tmp_path: Path) -> None:
    regime = _tiny_regime()
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 17402, tmp_path / "run", store)
    assert run.status == "completed"
    assert run.game_error_ledgers == []
    artifact = run.stages[0].artifacts["game_results"]
    path = Path(artifact["path"])
    assert path.name == f"{file_sha256(path)}.jsonl"
    assert "/artifacts/game-results/" in str(path)
    results = read_game_results(path)
    assert len(results) == run.stages[0].games == 2
    assert {(row.stream, row.game) for row in results} == {(0, 1), (1, 1)}
    for row in results:
        assert row.winner in (0, 1, None)
        for seat in row.seats:
            assert seat.pool == "pool-a" and seat.build in (None, "build-a-1")
    assert not any(key in path.read_text() for key in ("actions", "seed", "matchup"))


def test_completed_rows_survive_optimizer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("controlled optimizer failure")

    regime = _tiny_regime()
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.trainable = "policy_value"
    monkeypatch.setattr("manabot.training.execution.update_iteration", fail)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(RuntimeError, match="controlled optimizer failure"):
            execute_regime(regime, 17402, tmp_path / "run", store)
        saved = json.loads((tmp_path / "run/run.json").read_text())
        run = store.training_run(saved["id"])
    assert run.status == "failed"
    results = read_game_results(Path(run.stages[0].artifacts["game_results"]["path"]))
    assert len(results) == 2


def test_pooled_deck_sources_require_unique_aligned_builds() -> None:
    decks = (managym.Deck({"Forest": 40}), managym.Deck({"Forest": 40}))
    source = UniformDecks(
        decks=decks,
        pools=("p", None),
        builds=(None, None),
        sampling="uniform-independent-v1",
    )
    assert source.seats_by_player() == {"p": ("p", None)}
    for pools, builds in (
        (("p", "p"), (None, None)),
        ((None, None), ("b", None)),
        (("p",), (None,)),
        (("p", "q"), None),
    ):
        with pytest.raises(ValueError):
            UniformDecks(
                decks=decks,
                pools=pools,
                builds=builds,
                sampling="uniform-independent-v1",
            )
