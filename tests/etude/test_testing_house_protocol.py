"""Cross-language conformance for the testing-house-v1 control envelope."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from etude import server
from etude.testing_house_protocol import (
    REQUEST_ADAPTER,
    REQUEST_TYPES,
    TestingHouseV1ConformanceBundle as ControlBundle,
    testing_house_schema as control_schema,
)

PROTOCOL = Path(__file__).parents[2] / "protocol"
FIXTURE = json.loads(
    (PROTOCOL / "fixtures" / "testing-house-control-v1.json").read_text(
        encoding="utf-8"
    )
)
SCHEMA = json.loads(
    (PROTOCOL / "testing-house-v1.schema.json").read_text(encoding="utf-8")
)


def test_control_fixture_round_trips_through_checked_schema_and_python():
    Draft202012Validator.check_schema(SCHEMA)
    Draft202012Validator(SCHEMA).validate(FIXTURE)
    bundle = ControlBundle.model_validate(FIXTURE)
    assert bundle.model_dump(mode="json") == FIXTURE
    assert control_schema() == SCHEMA


def test_request_union_and_authorization_dispatch_are_the_same_closed_set():
    fixture_types = tuple(request["type"] for request in FIXTURE["requests"])
    assert fixture_types == REQUEST_TYPES
    assert set(server.DISPATCH_CAPABILITIES) | {"join_table", "resume"} == set(
        REQUEST_TYPES
    )
    for request in FIXTURE["requests"]:
        assert REQUEST_ADAPTER.validate_python(request).type == request["type"]


def test_control_contract_rejects_unknown_operations_and_sidecars():
    unknown = {"type": "chat", "message": "not part of Etude"}
    with pytest.raises(ValueError):
        REQUEST_ADAPTER.validate_python(unknown)

    sidecar = json.loads(json.dumps(FIXTURE))
    sidecar["requests"][0]["room_default"] = "shared"
    with pytest.raises(ValueError):
        ControlBundle.model_validate(sidecar)
