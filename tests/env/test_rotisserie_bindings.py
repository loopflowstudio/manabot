"""Public native drafting: printing membership, exact choices and immutable views."""

from collections import Counter

import numpy as np
import pytest

import managym
from managym.format import Limited, Rotisserie, set_printings


def test_printing_selection_retains_first_printing_identity_and_complete_supply() -> (
    None
):
    rows = set_printings("tla")
    assert len(rows) == 281
    assert Counter(row["rarity"] for row in rows) == {
        "common": 91,
        "uncommon": 110,
        "rare": 60,
        "mythic": 20,
    }
    printed = {row["key"]: row for row in managym.set_cards(["tla"], printings=True)}
    original = {row["key"] for row in managym.set_cards(["tla"])}
    assert len(original) == 278
    assert set(printed) - original == {
        "ptk.corrupt_court_official",
        "eld.epic_downfall",
        "ths.lightning_strike",
    }
    for row in rows:
        assert row["set"] == "tla"
        assert row["definition_scryfall_id"] == printed[row["key"]]["scryfall_id"]
        assert row["card_def_id"] == printed[row["key"]]["card_def_id"]
    with pytest.raises(ValueError, match="no complete printing inventory"):
        managym.set_cards(["tla", "tle"], printings=True)
    # The established prefix API keeps its empty-selection semantics.
    assert managym.set_cards(["missing"]) == []
    assert not set(printed).intersection(
        row["key"] for row in managym.set_cards(["tle"])
    )


def test_rotisserie_public_snapshots_conserve_supply_and_reject_stale_choices() -> None:
    draft = Rotisserie("tla", "bindings")
    initial = draft.observe()
    assert sum(initial["initial"]) == 664
    assert len(initial["printing_digest"]) == 64
    assert not np.asarray(initial["initial"])[Limited().basics].any()
    with pytest.raises(ValueError, match="not complete"):
        draft.pool(0)
    for revision in range(360):
        state = draft.observe()
        seat = revision % 8 if revision // 8 % 2 == 0 else 7 - revision % 8
        assert state["revision"] == revision and state["chooser"] == seat
        assert state["legal"] == [
            i for i, count in enumerate(state["remaining"]) if count
        ]
        card = state["legal"][0]
        for episode, rev, chooser, selected in (
            ("other", revision, seat, card),
            ("bindings", revision + 1, seat, card),
            ("bindings", revision, (seat + 1) % 8, card),
            ("bindings", revision, seat, len(state["initial"])),
        ):
            with pytest.raises(ValueError):
                draft.pick(episode, rev, chooser, selected)
            assert draft.observe() == state
        draft.pick("bindings", revision, seat, card)
        after = draft.observe()
        np.testing.assert_array_equal(
            np.asarray(after["remaining"]) + np.sum(after["pools"], axis=0),
            initial["initial"],
        )
        assert after["pools"][seat][card] == state["pools"][seat][card] + 1
        # Containers returned to callers are detached from the native authority.
        state["pools"][seat][card] = 9999
        state["remaining"][card] = 9999
        assert draft.observe() == after
    final = draft.observe()
    assert final["chooser"] is None and final["legal"] == []
    assert sum(final["remaining"]) == 304
    assert [sum(draft.pool(seat).cards.values()) for seat in range(8)] == [45] * 8
    with pytest.raises(ValueError):
        draft.pick("bindings", 360, 0, card)
