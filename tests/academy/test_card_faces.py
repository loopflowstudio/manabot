"""Printed faces answer only the names asked for, from the engine's definitions."""

from fastapi.testclient import TestClient

from managym.academy import server


def test_faces_print_cost_type_line_and_keywords_for_requested_names() -> None:
    with TestClient(server.app) as client:
        assert client.get("/api/card-faces").json() == {}
        names = [
            "Hama, the Bloodbender",
            "Island",
            "Ally",
            "Earth King's Lieutenant",
            "Not a card",
        ]
        faces = client.get("/api/card-faces", params={"names": names}).json()
    assert faces == {
        "Hama, the Bloodbender": {
            "cost": "{2}{U/B}{U/B}{U/B}",
            "colors": ["U", "B"],
            "type_line": "Legendary Creature — Human Warlock",
            "keywords": [],
            "power": 3,
            "toughness": 3,
        },
        "Island": {
            "cost": "",
            "colors": [],
            "type_line": "Basic Land — Island",
            "keywords": [],
            "power": None,
            "toughness": None,
        },
        "Ally": {
            "cost": "",
            "colors": ["W"],
            "type_line": "Token Creature — Ally",
            "keywords": [],
            "power": 1,
            "toughness": 1,
        },
        "Earth King's Lieutenant": {
            "cost": "{G}{W}",
            "colors": ["G", "W"],
            "type_line": "Creature — Human Soldier Ally",
            "keywords": ["trample"],
            "power": 1,
            "toughness": 1,
        },
    }
