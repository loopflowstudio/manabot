"""Export the exact admitted protocol017 recipes and their schedule coordinates.

These are candidate configurations, not a selected recommendation. After scoring,
the final handoff can copy the selected recipe without reconstructing defaults.
"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess

from pydantic import BaseModel

from experiments.runners.batching_grid import CAPACITIES, recipe
from manabot.training.models import AtaraxosMoveLearning, TrainSelfPlay

ROOT = Path('.runs/grid-007/learning-017')
OUTPUT = ROOT/'resolved-recipes'


class CellPlan(BaseModel):
    cell: str
    seconds: int


class GroupPlan(BaseModel):
    seed: int
    cells: list[CellPlan]


class Allocation(BaseModel):
    source: str
    transitions: int
    checkpoint_transitions: list[int]
    groups: list[GroupPlan]


class RecipeRecord(BaseModel):
    cell: str
    size: str
    hidden_dim: int
    attention_layers: int
    streams: int
    transitions_per_stream: int
    total_transitions: int
    total_iterations: int
    raw_checkpoint_iterations: list[int]
    schedule_reference_iterations: int
    initial_learning_rate: float
    final_learning_rate: float
    constant_learning_rate_over_tested_horizon: bool
    initial_tau: float
    final_tau: float
    ema_per_iteration: float | None
    training_seed_values: list[int]
    learner_seconds_limit: int
    regime_path: str
    regime_sha256: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = Allocation.model_validate_json((ROOT/'allocation.json').read_text())
    subprocess.run(['git','diff','--exit-code',plan.source,'--',
        'experiments/runners/batching_grid.py',
        'experiments/regimes/early-learning-rate-mini-source.json',
        'manabot/training/models.py'], check=True, stdout=subprocess.DEVNULL)
    assert plan.transitions == 524288
    assert [g.seed for g in plan.groups] == [13591,13592]
    assert plan.groups[0].cells == list(reversed(plan.groups[1].cells))
    assert not OUTPUT.exists(), 'Inspect an existing export rather than replacing it'
    OUTPUT.mkdir()
    records: list[RecipeRecord] = []
    for cell in plan.groups[0].cells:
        size, streams_text = cell.cell.split(':')
        streams = int(streams_text)
        capacity = next(c for c in CAPACITIES if c.name == size)
        regime = recipe(capacity, streams, plan.transitions, seconds=cell.seconds)
        stage = regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        learning = stage.learning
        assert isinstance(learning, AtaraxosMoveLearning)
        assert stage.updates is not None
        rates = [learning.rates(i) for i in range(1,stage.updates+1)]
        reference = plan.transitions//256
        if learning.iteration_schedule is not None:
            assert learning.iteration_schedule.reference_steps == reference
            assert learning.iteration_schedule.at(1) == 1
            assert learning.iteration_schedule.at(stage.updates) == reference
        path = OUTPUT/f'{size}-s{streams}.json'
        path.write_text(regime.model_dump_json(indent=2)+'\n')
        records.append(RecipeRecord(cell=f'{size}-s{streams}', size=size,
            hidden_dim=capacity.width, attention_layers=capacity.depth, streams=streams,
            transitions_per_stream=stage.transitions, total_transitions=plan.transitions,
            total_iterations=stage.updates,
            raw_checkpoint_iterations=[0,*[n//(streams*stage.transitions)
                                          for n in plan.checkpoint_transitions]],
            schedule_reference_iterations=reference, initial_learning_rate=rates[0][0],
            final_learning_rate=rates[-1][0],
            constant_learning_rate_over_tested_horizon=len({rate for rate,_ in rates})==1,
            initial_tau=rates[0][1], final_tau=rates[-1][1], ema_per_iteration=learning.ema,
            training_seed_values=[g.seed for g in plan.groups], learner_seconds_limit=cell.seconds,
            regime_path=path.name, regime_sha256=digest(path)))
    assert len(records) == 11
    result = {'observed_at':datetime.now(UTC).isoformat(), 'source':plan.source,
        'allocation_sha256':digest(ROOT/'allocation.json'), 'exporter_sha256':digest(Path(__file__)),
        'status':'exact candidate recipes exported; selection pending full experiment',
        'checkpoint_transitions':[0,*plan.checkpoint_transitions],
        'schedule_formula':'For K iterations and R=2048 reference iterations, r(i)=1+(i-1)*(R-1)/(K-1); four streams use r(i)=i. LR=clamp(0.5/r(i)^1.1,5e-6,1e-4); tau=0.05/r(i)^0.3.',
        'records':[r.model_dump(mode='json') for r in records]}
    (OUTPUT/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'recipes':len(records),
        'rates':{str(r.streams):{'iterations':r.total_iterations,
            'checkpoint_iterations':r.raw_checkpoint_iterations,
            'constant_learning_rate':r.constant_learning_rate_over_tested_horizon,
            'learning_rate':r.final_learning_rate,'tau_start':r.initial_tau,'tau_end':r.final_tau}
            for r in records}}))


if __name__ == '__main__':
    main()
