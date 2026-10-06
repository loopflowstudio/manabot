"""Whole-game dependence and fit-seed admission, without training or gameplay."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.runners.sampler_uncertainty_example import (
    synthetic_report,
    write_example,
)
from manabot.belief.sampling_cohort import (
    CohortSpec,
    Estimate,
    main,
    report_sampler_cohort,
)
from manabot.belief.sampling_report import SamplerQualityReport


def _spec(root: Path) -> CohortSpec:
    return CohortSpec.model_validate_json(write_example(root).read_bytes())


def _estimate(
    spec: CohortSpec, root: Path, comparison: str, metric: str = "joint_nll"
) -> Estimate:
    return next(
        e
        for e in report_sampler_cohort(spec, root=root).estimates
        if e.comparison == comparison and e.metric == metric
    )


def _replace_report(
    spec: CohortSpec, root: Path, index: int, report: SamplerQualityReport
) -> CohortSpec:
    attempt = spec.attempts[index]
    assert attempt.report is not None
    data = json.dumps(asdict(report), allow_nan=False).encode()
    (root / attempt.report).write_bytes(data)
    attempts = list(spec.attempts)
    attempts[index] = attempt.model_copy(
        update={"report_sha256": hashlib.sha256(data).hexdigest()}
    )
    return spec.model_copy(update={"attempts": tuple(attempts)})


def test_equal_games_separate_fit_uncertainty_and_paired_cancellation(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    result = _estimate(spec, tmp_path, "baseline learned")
    # Equal games: (1+5+9)/3 minus mean fit effect .2, not 7.2-.2.
    assert result.mean == pytest.approx(4.8)
    assert result.per_fit == pytest.approx((4.8, 4.9, 4.7))
    assert result.fit_interval is not None and result.game_interval is not None
    assert result.game_interval.high - result.game_interval.low > 1
    paired = _estimate(spec, tmp_path, "dropout minus baseline")
    assert paired.mean == pytest.approx(-0.1)
    assert paired.fit_interval is not None and paired.game_interval is not None
    assert paired.fit_interval.low == pytest.approx(-0.1)
    assert paired.game_interval.high == pytest.approx(-0.1)
    assert report_sampler_cohort(spec, root=tmp_path) == report_sampler_cohort(
        spec, root=tmp_path
    )


def test_correlated_views_never_become_bootstrap_units(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    before = _estimate(spec, tmp_path, "baseline learned")
    for index, attempt in enumerate(spec.attempts):
        report = synthetic_report(attempt.training_seed, attempt.treatment)
        panel = report.measurements[0]
        games = tuple(
            replace(
                g,
                examples=g.examples * 100,
                physical_prior=replace(
                    g.physical_prior, sampled_hands=g.physical_prior.sampled_hands * 100
                ),
                learned=replace(g.learned, sampled_hands=g.learned.sampled_hands * 100)
                if g.learned
                else None,
            )
            for g in panel.games
        )
        report = replace(
            report,
            measurements=(replace(panel, examples=panel.examples * 100, games=games),),
        )
        spec = _replace_report(spec, tmp_path, index, report)
    assert _estimate(spec, tmp_path, "baseline learned") == before


def test_single_fit_and_single_game_have_distinct_limits(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec = spec.model_copy(update={"attempts": spec.attempts[:1], "contrasts": ()})
    result = _estimate(spec, tmp_path, "baseline minus physical_prior")
    assert result.mean == pytest.approx(-0.2)
    assert result.fit_interval is None
    assert result.game_interval is not None
    assert "independent-fit uncertainty unavailable" in result.unavailable
    report = synthetic_report(11, "baseline")
    panel = report.measurements[0]
    game = panel.games[0]
    report = replace(
        report,
        measurements=(
            replace(
                panel, games=(game,), game_ids=(game.game_id,), examples=game.examples
            ),
        ),
    )
    spec = _replace_report(spec, tmp_path, 0, report)
    assert _estimate(spec, tmp_path, "baseline learned").game_interval is None


@pytest.mark.parametrize("status", ["missing", "failed"])
def test_all_attempts_retained_and_incomplete_cohort_not_estimated(
    tmp_path: Path, status: str
) -> None:
    spec = _spec(tmp_path)
    attempts = list(spec.attempts)
    attempts[1] = attempts[1].model_copy(
        update={
            "status": status,
            "reason": "retained fixture failure",
            "report": None,
            "report_sha256": None,
        }
    )
    spec = CohortSpec.model_validate(
        spec.model_copy(update={"attempts": tuple(attempts)}).model_dump()
    )
    report = report_sampler_cohort(spec, root=tmp_path)
    assert not report.complete and len(report.manifest.attempts) == 6
    assert all(
        e.mean is None and e.fit_interval is None and e.game_interval is None
        for e in report.estimates
    )
    assert _estimate(spec, tmp_path, "baseline learned").fit_seeds == (11, 13)


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_identity", "wrong"),
        ("schema_identity", "wrong"),
        ("evaluation_dataset_identity", "wrong"),
        ("training_dataset_identity", "wrong"),
        ("training_policy_identity", "wrong"),
        ("evaluation_policy_identity", "wrong"),
        ("evaluator_identity", "wrong"),
        ("seed", 99),
        ("training_seed", None),
        ("training_seed", 99),
        ("torch_version", "wrong"),
    ],
)
def test_identity_mismatch_never_silently_drops_a_fit(
    tmp_path: Path, field: str, value: str | int | None
) -> None:
    spec = _spec(tmp_path)
    report = replace(synthetic_report(11, "baseline"), **{field: value})
    spec = _replace_report(spec, tmp_path, 0, report)
    with pytest.raises(ValueError, match="mismatch"):
        report_sampler_cohort(spec, root=tmp_path)


def test_unequal_game_membership_and_reused_fits_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = synthetic_report(11, "baseline")
    panel = report.measurements[0]
    report = replace(
        report,
        measurements=(
            replace(
                panel, game_ids=panel.game_ids[:2], games=panel.games[:2], examples=10
            ),
        ),
    )
    changed = _replace_report(spec, tmp_path, 0, report)
    with pytest.raises(ValueError, match="membership"):
        report_sampler_cohort(changed, root=tmp_path)
    reused = replace(
        synthetic_report(12, "baseline"), checkpoint_identity="SYNTHETIC-baseline-11"
    )
    changed = _replace_report(spec, tmp_path, 0, synthetic_report(11, "baseline"))
    changed = _replace_report(changed, tmp_path, 1, reused)
    with pytest.raises(ValueError, match="not independent fits"):
        report_sampler_cohort(changed, root=tmp_path)
    raw = spec.model_dump()
    raw["attempts"][1]["training_seed"] = 11
    with pytest.raises(ValueError, match="repeated training seeds"):
        CohortSpec.model_validate(raw)


def test_unsupported_conjunction_and_missing_report_bytes(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = synthetic_report(11, "baseline")
    panel = report.measurements[0]
    games = tuple(
        replace(
            g, learned=replace(g.learned, conjunction_brier=None) if g.learned else None
        )
        for g in panel.games
    )
    spec = _replace_report(
        spec, tmp_path, 0, replace(report, measurements=(replace(panel, games=games),))
    )
    result = _estimate(spec, tmp_path, "baseline learned", "conjunction_brier")
    assert result.mean is None and result.unavailable == (
        "metric unsupported in declared cohort",
    )
    (tmp_path / "baseline-11.json").unlink()
    with pytest.raises(FileNotFoundError):
        report_sampler_cohort(spec, root=tmp_path)


def test_cli_statistical_regeneration_is_byte_identical(tmp_path: Path) -> None:
    write_example(tmp_path)
    for name in ("first", "second"):
        main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--out",
                str(tmp_path / f"{name}.json"),
            ]
        )
    assert (tmp_path / "first.json").read_bytes() == (
        tmp_path / "second.json"
    ).read_bytes()
    assert (tmp_path / "first.md").read_bytes() == (tmp_path / "second.md").read_bytes()
    assert "SYNTHETIC REVIEW FIXTURE" in (tmp_path / "first.md").read_text()
    with pytest.raises(FileExistsError):
        main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--out",
                str(tmp_path / "first.json"),
            ]
        )
