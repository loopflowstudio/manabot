"""Contract checks for the release-stack selected-matchup prompt matrix."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from etude.curated_pack import CURATED_PACK
from manabot.env.observation import ActionSpaceEnum

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "frontend" / "e2e" / "release-prompt-matrix.json"


def test_release_prompt_matrix_classifies_and_covers_selected_matchup():
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    assert matrix["schema_version"] == 2

    pack = matrix["asset_pack"]
    assert pack == {
        "id": CURATED_PACK.pack_id,
        "version": CURATED_PACK.version,
        "manifest_sha256": CURATED_PACK.manifest_sha256,
    }

    action_spaces = matrix["action_spaces"]
    reachable = action_spaces["reachable"]
    terminal = action_spaces["terminal"]
    excluded_records = action_spaces["excluded"]
    excluded = [record["family"] for record in excluded_records]

    assert len(reachable) == len(set(reachable)) == 9
    assert terminal == ["GAME_OVER"]
    assert excluded == ["MODAL"]
    classifications = [*reachable, *terminal, *excluded]
    assert len(classifications) == len(set(classifications))
    assert set(classifications) == set(ActionSpaceEnum.__members__)

    selected_cards = set(CURATED_PACK.hero_deck) | set(CURATED_PACK.villain_deck)
    assert excluded_records[0]["proof_card"] not in selected_cards

    families = matrix["families"]
    assert set(families) == set(reachable)
    for family, record in families.items():
        assert record["sources"], family
        for source in record["sources"]:
            assert source["kind"] in {"core", "card"}
            assert source["name"]
            if source["kind"] == "card":
                assert source["name"] in selected_cards

    policies = matrix["policies"]
    assert set(policies) == {"curated-v1"}
    cast_orders = policies["curated-v1"]["priority_cast_order"]
    deck_cards = {
        CURATED_PACK.hero_deck_id: set(CURATED_PACK.hero_deck),
        CURATED_PACK.villain_deck_id: set(CURATED_PACK.villain_deck),
    }
    assert set(cast_orders) == set(deck_cards)
    for deck_id, order in cast_orders.items():
        assert len(order) == len(set(order))
        assert set(order) <= deck_cards[deck_id]

    scenarios = matrix["scenarios"]
    scenario_ids = [scenario["id"] for scenario in scenarios]
    assert len(scenario_ids) == len(set(scenario_ids)) == 2
    assert {
        (scenario["hero_deck"], scenario["villain_deck"]) for scenario in scenarios
    } == {
        (CURATED_PACK.hero_deck_id, CURATED_PACK.villain_deck_id),
        (CURATED_PACK.villain_deck_id, CURATED_PACK.hero_deck_id),
    }

    covered: set[str] = set()
    scenario_families: dict[str, set[str]] = {}
    for scenario in scenarios:
        assert scenario["villain_type"] == "random"
        assert scenario["policy"] in policies
        assert isinstance(scenario["seed"], int)
        assert 0 < scenario["max_commands"] <= 300

        expected = scenario["expected"]
        counts = expected["prompt_counts"]
        assert counts
        assert set(counts) <= set(reachable)
        assert all(isinstance(count, int) and count > 0 for count in counts.values())
        assert expected["commands"] == sum(counts.values())
        assert expected["commands"] < scenario["max_commands"]
        assert expected["winner"] in {0, 1}
        assert isinstance(expected["turn"], int) and expected["turn"] > 0
        sequence = expected["prompt_sequence"]
        assert len(sequence) == expected["commands"]
        assert Counter(sequence) == Counter(counts)
        assert set(sequence) <= set(reachable)

        scenario_families[scenario["id"]] = set(counts)
        covered.update(counts)

    assert covered == set(reachable)
    for family, record in families.items():
        expected_ids = {
            scenario_id
            for scenario_id, observed in scenario_families.items()
            if family in observed
        }
        assert set(record["scenario_ids"]) == expected_ids

    visual = matrix["visual_references"]
    assert visual["version"] == 2
    assert visual["directory"] == "visual-references/v2"
    assert visual["profile"] == {
        "name": "ubuntu-24.04-chromium",
        "operating_system": "Ubuntu 24.04 x86-64",
        "node": "22",
        "playwright": "1.61.1",
        "chromium": "149.0.7827.55",
        "viewport": {"width": 1600, "height": 1200},
        "device_scale_factor": 1,
        "color_scheme": "dark",
        "locale": "en-US",
        "timezone": "UTC",
        "reduced_motion": "reduce",
        "pixel_threshold": 0.2,
        "font": {
            "body_family": "Lato",
            "display_family": "Cormorant Garamond",
            "mono_family": "JetBrains Mono",
            "source_revision": "loopflow@1e68ba107",
        },
    }

    scenarios_by_id = {scenario["id"]: scenario for scenario in scenarios}
    prompt_references = visual["prompts"]
    assert {reference["family"] for reference in prompt_references} == set(reachable)
    assert len(prompt_references) == len(reachable)

    ids: list[str] = []
    for reference in [*prompt_references, *visual["boards"]]:
        assert reference["id"]
        ids.append(reference["id"])
        scenario = scenarios_by_id[reference["scenario_id"]]
        family = reference["family"]
        assert family in scenario["expected"]["prompt_counts"]
        assert 0 < reference["occurrence"] <= scenario["expected"]["prompt_counts"][family]

    assert {reference["id"] for reference in visual["boards"]} == {
        "board-opening",
        "board-combat",
        "board-developed",
    }
    assert visual["reconnect"] == {
        "scenario_id": "ur-lessons-seed-51",
        "command": 0,
        "states": ["disconnected", "reconnecting", "connected"],
    }
    reconnect_scenario = scenarios_by_id[visual["reconnect"]["scenario_id"]]
    assert visual["reconnect"]["command"] < reconnect_scenario["expected"]["commands"]
    ids.extend(f"reconnect-{state}" for state in visual["reconnect"]["states"])

    terminals = visual["terminals"]
    assert {reference["scenario_id"] for reference in terminals} == set(scenarios_by_id)
    assert len(terminals) == len(scenarios_by_id)
    ids.extend(reference["id"] for reference in terminals)
    assert len(ids) == len(set(ids)) == 17

    reference_dir = MATRIX_PATH.parent / visual["directory"]
    assert reference_dir.is_dir()
    assert {path.name for path in reference_dir.glob("*.png")} == {
        f"{reference_id}.png" for reference_id in ids
    }
