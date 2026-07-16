"""Regenerate the canonical STU-1 consumer fixtures and semantic receipt."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from etude.study_index import canonical_json_bytes, main, pretty_json_bytes
from etude.study_protocol import RecordedDecisionInput, StudyIdentity

ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "protocol"
FIXTURES = PROTOCOL / "fixtures"
SOURCE_PATH = FIXTURES / "recorded-match-decisions-curated.json"
IDENTITY_PATH = FIXTURES / "study-index-identity-curated.json"
INDEX_PATH = FIXTURES / "study-decision-index-curated.json"
RECEIPT_PATH = ROOT / "experiments" / "data" / "w2-220-study-decision-index-v1.json"


def _offer(
    offer_id: int,
    verb: str,
    label: str,
    action_type: str,
    focus: list[int],
) -> dict[str, Any]:
    return {
        "id": offer_id,
        "actor": 0,
        "verb": verb,
        "source": None,
        "label": label,
        "help": None,
        "choices": [],
        "confirm_label": label,
        "action_type": action_type,
        "focus": focus,
    }


def _event(
    seq: int,
    revision: int,
    command_id: str,
    importance: str,
    kind: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seq": seq,
        "from_revision": revision,
        "to_revision": revision + 1,
        "caused_by": command_id,
        "group": seq,
        "importance": importance,
        "suggested_ms": 0,
        "sound": None,
        "kind": kind,
    }


def _frame(
    base: dict[str, Any],
    ordinal: int,
    action_space: str,
    step: str,
    offers: list[dict[str, Any]],
) -> dict[str, Any]:
    frame = deepcopy(base)
    revision = 10 + ordinal
    frame["revision"] = revision
    frame["frame_hash"] = "pending"
    frame["action_space"] = action_space
    frame["prompt"] = {
        "id": 100 + ordinal,
        "actor": 0,
        "kind": action_space.lower(),
        "title": f"Recorded decision {ordinal}",
        "instruction": "Choose an action",
    }
    frame["projection"]["turn"]["turn_number"] = 2
    frame["projection"]["turn"]["step"] = step
    frame["offers"] = offers
    frame["frame_hash"] = hashlib.sha256(canonical_json_bytes(frame)).hexdigest()
    return frame


def _record(
    frame: dict[str, Any],
    ordinal: int,
    selected: int,
    presentation: list[dict[str, Any]],
    *,
    automatic: bool = False,
) -> dict[str, Any]:
    offer = frame["offers"][selected]
    command_id = f"study-index-command-{ordinal}"
    for event in presentation:
        event["caused_by"] = command_id
    return {
        "ordinal": ordinal,
        "event_cursor": 1000 + ordinal * 10,
        "automatic": automatic,
        "frame": frame,
        "offer": deepcopy(offer),
        "played": {
            "command_id": command_id,
            "match_id": frame["match_id"],
            "expected_revision": frame["revision"],
            "prompt_id": frame["prompt"]["id"],
            "offer_id": offer["id"],
            "answers": [],
        },
        "presentation": presentation,
    }


def _source_fixture() -> dict[str, Any]:
    bundle = json.loads((FIXTURES / "bolt-target.json").read_text(encoding="utf-8"))
    base = bundle["recovery"]["frame"]
    base.pop("auto_passed", None)
    base.pop("deck_names", None)
    base.pop("asset_pack", None)
    base.pop("log", None)

    frames = [
        _frame(
            base,
            0,
            "PRIORITY",
            "PRECOMBAT_MAIN_STEP",
            [
                _offer(0, "pass_priority", "Pass priority", "PASS", []),
                _offer(1, "cast", "Cast public spell", "CAST", [31]),
            ],
        ),
        _frame(
            base,
            1,
            "PRIORITY",
            "PRECOMBAT_MAIN_STEP",
            [
                _offer(10, "pass_priority", "Pass priority", "PASS", []),
                _offer(11, "cast", "Respond with spell", "CAST", [41]),
            ],
        ),
        _frame(
            base,
            2,
            "CHOOSE_TARGET",
            "PRECOMBAT_MAIN_STEP",
            [
                _offer(20, "choose", "Target hero", "CHOOSE_TARGET", [0]),
                _offer(21, "choose", "Target villain", "CHOOSE_TARGET", [1]),
            ],
        ),
        _frame(
            base,
            3,
            "DECLARE_ATTACKER",
            "DECLARE_ATTACKERS_STEP",
            [
                _offer(30, "declare_attackers", "Keep scout back", "NO_ATTACK", [51]),
                _offer(31, "declare_attackers", "Attack with scout", "ATTACK", [51]),
            ],
        ),
        _frame(
            base,
            4,
            "DECLARE_ATTACKER",
            "DECLARE_ATTACKERS_STEP",
            [
                _offer(40, "declare_attackers", "Keep adept back", "NO_ATTACK", [52]),
                _offer(41, "declare_attackers", "Attack with adept", "ATTACK", [52]),
            ],
        ),
        _frame(
            base,
            5,
            "DECLARE_BLOCKER",
            "DECLARE_BLOCKERS_STEP",
            [
                _offer(50, "declare_blockers", "Take the hit", "NO_BLOCK", [61]),
                _offer(51, "declare_blockers", "Block the attacker", "BLOCK", [61]),
            ],
        ),
        _frame(
            base,
            6,
            "PRIORITY",
            "POSTCOMBAT_MAIN_STEP",
            [
                _offer(60, "pass_priority", "Pass priority", "PASS", []),
                _offer(61, "cast", "Cast optional spell", "CAST", [71]),
            ],
        ),
        _frame(
            base,
            7,
            "CHOOSE_TARGET",
            "END_STEP",
            [_offer(70, "choose", "Only legal target", "CHOOSE_TARGET", [1])],
        ),
    ]

    frames[1]["projection"]["agent"]["stack"] = [
        {
            "id": 41,
            "registry_key": 41,
            "name": "Public stack spell",
            "zone": "STACK",
            "owner_id": 0,
            "power": 0,
            "toughness": 0,
            "mana_value": 2,
            "types": {
                "is_creature": False,
                "is_land": False,
                "is_spell": True,
                "is_artifact": False,
                "is_enchantment": False,
                "is_planeswalker": False,
                "is_battle": False,
            },
        }
    ]
    frames[1]["frame_hash"] = "pending"
    frames[1]["frame_hash"] = hashlib.sha256(
        canonical_json_bytes(frames[1])
    ).hexdigest()

    revisions = [frame["revision"] for frame in frames]
    records = [
        _record(
            frames[0],
            0,
            1,
            [
                _event(
                    100,
                    revisions[0],
                    "pending",
                    "critical",
                    {
                        "kind": "damage",
                        "source": None,
                        "target": {"kind": "player", "id": 1},
                        "amount": 3,
                    },
                )
            ],
        ),
        _record(
            frames[1],
            1,
            0,
            [
                _event(
                    110,
                    revisions[1],
                    "pending",
                    "normal",
                    {"kind": "resolved", "stack": 4100},
                )
            ],
        ),
        _record(
            frames[2],
            2,
            1,
            [
                _event(
                    120,
                    revisions[2],
                    "pending",
                    "emphasized",
                    {
                        "kind": "targeted",
                        "source": {
                            "kind": "object",
                            "id": {"entity": 31, "incarnation": 0},
                        },
                        "target": {"kind": "player", "id": 1},
                    },
                )
            ],
        ),
        _record(
            frames[3],
            3,
            1,
            [
                _event(
                    130,
                    revisions[3],
                    "pending",
                    "critical",
                    {
                        "kind": "attack_group",
                        "attackers": [{"entity": 51, "incarnation": 0}],
                        "defender": {"kind": "player", "id": 1},
                    },
                )
            ],
        ),
        _record(
            frames[4],
            4,
            1,
            [
                _event(
                    140,
                    revisions[4],
                    "pending",
                    "normal",
                    {
                        "kind": "attack_group",
                        "attackers": [{"entity": 52, "incarnation": 0}],
                        "defender": {"kind": "player", "id": 1},
                    },
                )
            ],
        ),
        _record(
            frames[5],
            5,
            1,
            [
                _event(
                    150,
                    revisions[5],
                    "pending",
                    "emphasized",
                    {
                        "kind": "blocked",
                        "attacker": {"entity": 62, "incarnation": 0},
                        "blockers": [{"entity": 61, "incarnation": 0}],
                    },
                )
            ],
        ),
        _record(frames[6], 6, 0, [], automatic=True),
        _record(frames[7], 7, 0, []),
    ]
    return {
        "version": 1,
        "source_replay_id": "study-index/curated-recorded-decisions-v1",
        "decision_count": len(records),
        "decisions": records,
    }


def _identity(source: dict[str, Any]) -> dict[str, Any]:
    template = json.loads(
        (FIXTURES / "study-curated-decision.json").read_text(encoding="utf-8")
    )["identity"]
    template["artifact_id"] = "study-decision-index-curated-v1"
    template["source_replay_id"] = source["source_replay_id"]
    template["source_replay_sha256"] = hashlib.sha256(
        canonical_json_bytes(source)
    ).hexdigest()
    return template


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(value))


def run() -> None:
    source = _source_fixture()
    identity = _identity(source)
    RecordedDecisionInput.model_validate(source)
    StudyIdentity.model_validate(identity)
    _write(SOURCE_PATH, source)
    _write(IDENTITY_PATH, identity)
    main(
        [
            str(SOURCE_PATH),
            "--identity",
            str(IDENTITY_PATH),
            "--expected-index",
            str(INDEX_PATH),
            "--semantic-receipt",
            str(RECEIPT_PATH),
            "--repeats",
            "1000",
            "--write",
        ]
    )


if __name__ == "__main__":
    run()
