"""Offline hand labels never enter the production summary path."""

from collections import Counter
from dataclasses import fields, replace
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.decision_memory_probe import HandProbe, _predict, fit
from manabot.env.memory_store import MemoryStore, PrefixIdentity, PrefixObservation
from manabot.policy.prepare import Prepared
import manabot.verify.decision_memory_probe as probe_module
from manabot.verify.decision_memory_probe import (
    CAPACITIES,
    Corpus,
    ProbeRow,
    analytic_presence,
    choose_k,
    collect_games,
    game_losses,
    hand_presence,
    new_corpus,
    probe_space,
    split_game,
)
import managym
from managym.decision_memory import DecisionContext, DecisionRecord, PublicChoice
from tests.model.test_decision_memory import encode_with_records


def _planted() -> Corpus:
    space = probe_space()
    env = managym.Env(seed=111)
    deck = managym.authored_deck_setup(
        "elemental-world-v2", managym.authored_deck_keys("elemental-world-v2")[0]
    )
    raw, _ = env.reset([deck, deck])
    store = MemoryStore(CAPACITIES)
    observations = []
    for signal in (0, 1):
        record = DecisionRecord(
            actor=2,
            ordinal=0,
            first_position=0,
            last_position=0,
            count=1,
            kind="DeclareBlocker" if signal else "Priority",
            context=DecisionContext(
                turn=3,
                phase=1,
                step=3,
                life=(20, 2 if signal else 38),
                hand=(6, 3),
                board_counts=((0,) * 8, (0,) * 8),
                mana_sources=[],
            ),
            response=[],
            choice=PublicChoice(
                kind="DeclineChoice" if signal else "PassPriority",
                mode=None,
                ability=None,
                named=None,
                name_row=None,
                declared=None,
                references=[],
            ),
        )
        encoded = encode_with_records(space, raw, [record])
        dense = {key: value[None] for key, value in encoded.items()}
        prefix = store.capture(dense, 0, PrefixIdentity("fixture", 0, str(signal)))
        observations.append(PrefixObservation(dense, [prefix], CAPACITIES))
    corpus = Corpus([], [], ("target",))
    # Both signals occur in every game, with separate whole-game splits.
    for game in range(60):
        for signal in (0, 1):
            corpus.rows.append(
                ProbeRow(
                    game,
                    0,
                    signal,
                    "open",
                    observations[signal],
                    np.zeros(3, np.float32),
                    np.array([0.5], np.float32),
                )
            )
            corpus.labels.append(np.array([signal], np.float32))
    return corpus


def test_game_splits_and_analytic_baseline() -> None:
    corpus = _planted()
    groups = [
        {corpus.rows[int(i)].game for i in corpus.indexes(split)}
        for split in ("fit", "validation", "test")
    ]
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert split_game(31) == split_game(31)
    counts = Counter((split_game(seed), seed % 2) for seed in range(48120, 48160))
    for mode in (0, 1):
        assert [counts[split, mode] for split in ("fit", "validation", "test")] == [
            14,
            3,
            3,
        ]
    np.testing.assert_allclose(
        analytic_presence(2, np.array([3.0, 1.0])), [1 - 1e-6, 0.5]
    )
    assert choose_k(np.array([-0.02, -0.03, -0.01]), {1: 1.0, 4: 2.0})[0] == 4
    assert choose_k(np.array([-0.1, 0.1]), {1: 1.0, 4: 2.0})[0] == 1
    assert choose_k(np.array([-0.1, 0.1]), {1: 2.0, 4: 1.0})[0] == 4
    assert choose_k(np.array([-0.1, 0.1]), {1: 1.0, 4: 1.0})[0] == 1


def test_hand_targets_cover_main_and_sideboard_without_silent_omission() -> None:
    corpus = new_corpus()
    for key in managym.authored_deck_keys("elemental-world-v2"):
        player = managym.authored_deck_setup("elemental-world-v2", key)
        for cards in (player.decklist, player.sideboard):
            assert cards.keys() <= set(corpus.names)
            presence = hand_presence(cards, corpus.names)
            assert presence.sum() == len(cards)
    with pytest.raises(ValueError, match="vocabulary omits hand"):
        hand_presence({"not a roster card": 1}, corpus.names)


def test_labels_stay_outside_recorded_and_prepared_inputs(tmp_path: Path) -> None:
    corpus = new_corpus()
    # Seed 48103 reaches trick planning where one shared selector changes a
    # choice; replay must match independently owned seat plans throughout.
    collect_games(corpus, 1, tmp_path, start=3)
    assert corpus.rows and corpus.labels
    assert len(corpus.rows) <= 32
    journal = json.loads((tmp_path / "games.jsonl").read_text())
    replay = managym.Env(seed=journal["seed"])
    live, _ = replay.reset(
        managym.GameSetup(
            [
                managym.authored_deck_setup("elemental-world-v2", key)
                for key in journal["pair"]
            ],
            decklists=journal["decklists"],
        )
    )
    selectors = (managym.Heuristic(), managym.Heuristic())
    selected = {row.decision: i for i, row in enumerate(corpus.rows)}
    for decision, action in enumerate(journal["actions"]):
        viewer = replay.current_agent_index()
        assert action == selectors[viewer].select_action(live)
        if decision in selected:
            i = selected[decision]
            assert corpus.rows[i].viewer == viewer
            hand = replay.hand_snapshot(1 - viewer).cards
            np.testing.assert_array_equal(
                corpus.labels[i], [hand.get(name, 0) > 0 for name in corpus.names]
            )
        live, _, _, _, _ = replay.step(action)
    assert not {"label", "labels", "presence", "target"} & {
        f.name for f in fields(Prepared)
    }
    model = HandProbe(corpus, "gru", 1)
    assert model.policy is not None
    for row in corpus.rows:
        assert set(row.observation) <= set(probe_space().shapes)
        assert not {"label", "labels", "presence", "target"} & row.observation.keys()
    with np.load(tmp_path / "row-00000.npz") as recorded:
        assert set(recorded.files) == set(probe_space().shapes)
        for key, value in corpus.rows[0].observation.materialize().items():
            np.testing.assert_array_equal(value, recorded[key])
    dense = corpus.rows[0].observation.materialize()
    batch = on_device(model.policy.prepare(dense), "cpu")
    before = model.policy.encode_decision_memory(batch).detach().clone()
    corpus.labels[0][:] = 1 - corpus.labels[0]
    torch.testing.assert_close(
        before, model.policy.encode_decision_memory(batch), atol=0, rtol=0
    )
    with pytest.raises(TypeError):
        model.policy.encode_decision_memory(dense)  # type: ignore[arg-type]


def test_native_capacity_failure_discards_the_whole_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe_module, "CAPACITIES", (1, 8192, 256, 16384))
    corpus = new_corpus()
    collect_games(corpus, 1, tmp_path)
    assert corpus.overflows == 1
    assert not corpus.rows and not corpus.labels
    assert not list(tmp_path.glob("row-*.npz"))


@pytest.mark.parametrize("encoder", ["gru", "transformer"])
def test_production_summary_learns_planted_signal_but_not_shuffled_labels(
    tmp_path: Path, encoder: Literal["gru", "transformer"]
) -> None:
    torch.set_num_threads(1)
    corpus = _planted()
    candidate = fit(corpus, encoder, 1, 123, tmp_path, epochs=12)
    valid = corpus.indexes("validation")
    loss = game_losses(np.stack(corpus.labels)[valid], candidate.validation).mean()
    assert loss < 0.25
    shuffled = replace(
        corpus,
        labels=list(
            np.stack(corpus.labels)[
                np.random.default_rng(3).permutation(len(corpus.labels))
            ]
        ),
    )
    baseline = fit(shuffled, "none", 1, 123, tmp_path, epochs=12)
    candidate = fit(shuffled, encoder, 1, 123, tmp_path, epochs=12)
    truth = np.stack(shuffled.labels)[valid]
    assert (
        game_losses(truth, candidate.validation).mean()
        >= game_losses(truth, baseline.validation).mean() - 0.04
    )
    # The trained production API is the sole encoder exercised by the probe.
    model = HandProbe(shuffled, candidate.encoder, 1)
    model.load_state_dict(torch.load(candidate.weights, weights_only=True))
    np.testing.assert_allclose(
        _predict(model, shuffled, valid), candidate.validation, atol=0, rtol=0
    )
