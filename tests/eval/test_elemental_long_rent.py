"""MTG-183's rented evaluation: which units a batch takes and how it is frozen."""

from decimal import Decimal
from pathlib import Path

from experiments.elemental_long_eval import units
from experiments.elemental_long_rent import (
    SCALAR_RUN,
    Batch,
    _match_identity,
    job,
    plan,
)
from manabot.infra.source import SourceBundle
from tests.eval.test_elemental_long_eval import _state


def test_plan_takes_pending_units_newest_first(tmp_path: Path) -> None:
    design = units(_state(tmp_path, (0, 50, 100, 150, 200), 200))
    newest = [u for u in design if u.newer == 200]
    done = {_match_identity(m) for m in newest[0].matches}
    chosen = plan(design, done, {newest[1].key}, games=2_500)
    assert [u.newer for u in chosen] == [200] * len(chosen)
    assert newest[0].key not in {u.key for u in chosen}
    assert newest[1].key not in {u.key for u in chosen}
    assert sum(m.games for u in chosen for m in u.matches) <= 2_500


def test_job_seats_each_checkpoint_once_as_a_digest_checked_input(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, (0, 50, 100), 100)
    for checkpoint in state.checkpoints.values():
        checkpoint.artifact = checkpoint.artifact.model_copy(
            update={"uri": f"r2://manabot/jobs/w1/{checkpoint.updates}"}
        )
    chosen = [u for u in units(state) if u.newer == 100]
    batch = Batch(
        run=SCALAR_RUN,
        state=state,
        units=tuple(u.key for u in chosen),
        workers=2,
        compute=None,
        source_bundle=SourceBundle(commit="0" * 40, files={"a": "0" * 64}),
    )
    frozen = job(batch, revision="0" * 40, max_hours=1.0, max_usd=Decimal("1.10"))
    seated = {
        seated.artifact.sha256
        for u in chosen
        for m in u.matches
        for seated in m.agents
        if seated.artifact is not None
    }
    assert sorted(i.sha256 for i in frozen.inputs) == sorted(seated)
    assert all(i.name == f"{i.sha256}.pt" for i in frozen.inputs)
    assert Batch.model_validate_json(frozen.config or "") == batch
