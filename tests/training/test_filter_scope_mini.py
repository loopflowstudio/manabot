"""Frozen closeout admission never drops seeds, deals or original endpoints."""

import numpy as np
from pydantic import ValidationError
import pytest

from experiments.runners.filter_scope_report import interval
from experiments.runners.training_protocol import EvaluationProtocol


def protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        study="filter-scope-mini",
        purpose="scientific",
        regime_digests=("a" * 64, "b" * 64),
        training_seeds=(10501, 10502, 10503),
        paired_deals=(),
        anchor_deals=tuple(range(1910105100, 1910105125)),
        endpoint_anchor_deals=tuple(range(9305000, 9305025)),
        random_diagnostic_deals=tuple(range(9205000, 9205025)),
        checkpoint_count=2,
        cost_cutoffs_seconds=(3600,),
        anchors=("scripted-greedy", "random"),
        process_seconds=7200,
        uncertainty="paired-seed-descriptive",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_seeds", (10501, 10502)),
        ("endpoint_anchor_deals", tuple(range(9305000, 9305024))),
        ("random_diagnostic_deals", tuple(range(9305000, 9305025))),
        ("anchors", ("scripted-greedy",)),
        ("process_seconds", 86400),
    ],
)
def test_closeout_preserves_cohort(field: str, value: object) -> None:
    original = protocol()
    assert (
        EvaluationProtocol.model_validate_json(original.model_dump_json()) == original
    )
    with pytest.raises(ValidationError):
        EvaluationProtocol.model_validate({**original.model_dump(), field: value})


def test_paired_uncertainty_subtracts_before_resampling() -> None:
    # Arbitrary seed/deal variation cancels for exactly paired equal treatments.
    scores = np.arange(75, dtype=float).reshape(3, 25) / 100
    result = interval(scores - scores)
    assert (result["mean"], result["lower"], result["upper"]) == (0, 0, 0)
    positive = interval(np.full((3, 25), 0.05))
    assert positive["mean"] == pytest.approx(0.05)
    assert positive["lower"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="three paired seeds"):
        interval(scores[:2])
