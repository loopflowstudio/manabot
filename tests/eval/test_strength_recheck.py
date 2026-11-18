"""The CLI compares complete, paired evidence to the unrounded reference."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from experiments.tla_frontier import PROTOCOL, corpus, deck_matchup
from manabot.cli import app
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.eval.recheck import ReferencePair, StrengthReference
from manabot.infra.artifacts import StoredArtifact
from manabot.sim.matchup import DeckMatchup
from tests.eval.test_local_evaluation import _anchor_request, _replayed_rows

REFERENCES = Path(__file__).resolve().parents[2] / "experiments/references"


@pytest.mark.parametrize(
    "scores,missing,code", [((1, 0), False, 0), ((0, 0), False, 1), ((1, 0), True, 1)]
)
def test_cli_requires_every_pair_before_comparing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scores: tuple[int, int],
    missing: bool,
    code: int,
) -> None:
    source = json.loads((REFERENCES / "tla-frontier-small-u1952.json").read_bytes())
    source["recorded"].update(score=0.5, low=0.5, high=0.6)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(source))
    pairs = json.loads((REFERENCES / source["matchups"]).read_bytes())[:2]
    (tmp_path / source["matchups"]).write_text(json.dumps(pairs))
    observed = []

    def prepare(
        checkpoint: StoredArtifact,
        opponent: str,
        seeds: tuple[int, ...],
        matchup: DeckMatchup,
        *,
        deadline: float,
        heuristic_actions: int,
        show_binding: bool,
    ) -> EvaluationRequest:
        index = len(observed)
        observed.append(seeds)
        pair = ReferencePair.model_validate(pairs[index])
        assert matchup == pair.matchup
        assert seeds == (pair.deal_seed, pair.deal_seed + 16)
        assert opponent == "heuristic" and heuristic_actions == 64
        request = _anchor_request(seeds)
        key = request.key.model_copy(update={"content_suite": matchup.content_suite})
        return request.model_copy(update={"key": key, "authored_matchup": matchup})

    def evaluate(
        request: EvaluationRequest, *, deadline: float, out_dir: Path
    ) -> EvaluationResult:
        rows = _replayed_rows(request)
        for row in rows:
            row.score_a = float(scores[len(observed) - 1])
        if missing and len(observed) == 2:
            rows.pop()
        result = EvaluationResult(request=request, rows=rows)
        result.validate_cohort()
        return result

    monkeypatch.setattr("manabot.eval.recheck.prepare_local_request", prepare)
    monkeypatch.setattr("manabot.eval.recheck.evaluate_local", evaluate)
    out = tmp_path / "proof"
    result = CliRunner().invoke(app, ["recheck", str(path), "--out", str(out)])
    assert result.exit_code == code, result.output
    assert len(observed) == 2
    assert "recorded  0.50" in result.output
    assert "heuristic changed" in result.output
    if missing:
        assert "incomplete" in result.output
        assert not (out / "measured.json").exists()
    else:
        assert "measured  " in result.output
        assert (
            json.loads((out / "measured.json").read_bytes())["score"]["mean"]
            == sum(scores) / 2
        )


def test_references_keep_the_original_frontier_population() -> None:
    population = corpus()
    for update in (25, 800, 1952):
        ref = StrengthReference.model_validate_json(
            (REFERENCES / f"tla-frontier-small-u{update}.json").read_bytes()
        )
        pairs = [
            ReferencePair.model_validate(row)
            for row in json.loads((REFERENCES / ref.matchups).read_bytes())
        ]
        assert ref.recorded.games == 4 * len(pairs) == 64
        for pair, original in zip(pairs, PROTOCOL.pairs, strict=True):
            assert pair.matchup == deck_matchup(population, PROTOCOL, original)
            assert pair.deal_seed == original.deal
