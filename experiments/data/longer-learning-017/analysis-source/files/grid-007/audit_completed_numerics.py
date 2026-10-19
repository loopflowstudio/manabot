"""Offline coarse numerical audit; does not establish deterministic correctness."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.models import TrainingRun

ROOT = Path('.runs/grid-007/feasibility')

@dataclass(frozen=True)
class Movement:
    maximum_absolute_change: float
    parameter: str
    global_adam_upper_bound: float
    elements_above_bound: int
    maximum_saved_adam_steps: int
    all_parameters_finite: bool
    all_moments_finite: bool
    negative_second_moments: int

def coordinate_bound(steps: int, rate: float, beta1: float, beta2: float) -> float:
    """Exact-arithmetic Cauchy bound, using max LR and max saved step count.

    Same formula as retained study003 outlier audit. Zero initial moments and
    zero weight decay are required. This intentionally loose whole-trajectory
    bound catches gross movement; passing does not prove each update correct.
    """
    ratio = beta1 * beta1 / beta2
    return rate * sum((1-beta1)/(1-beta1**n) * math.sqrt(
        (1-beta2**n)/(1-beta2) * (1-ratio**n)/(1-ratio)) for n in range(1,steps+1))

def audit(directory: Path) -> dict[str, object]:
    proof=json.loads((directory/'verification.json').read_text())
    assert proof['all_manifest_files_verified'] and proof['checkpoint_loaded']
    path=directory/'study/training/run'
    run=TrainingRun.model_validate_json((path/'run.json').read_text())
    assert run.status=='completed' and run.parent_run_id is None
    initial,_=load_checkpoint_agent(str(path/'self-play-initial-raw.pt'))
    final,_=load_checkpoint_agent(str(path/'self-play-raw.pt'))
    optimizer=torch.load(path/'self-play-optimizer.pt',map_location='cpu',weights_only=True)
    assert isinstance(optimizer,dict) and set(optimizer)=={'state','param_groups'}
    assert len(optimizer['param_groups'])==1
    group=optimizer['param_groups'][0]
    assert group['weight_decay']==0 and not group['amsgrad'] and not group['maximize']
    beta1,beta2=group['betas']; assert (beta1,beta2)==(0.9,0.999)
    state=optimizer['state']; assert isinstance(state,dict) and state
    steps=max(int(v['step'].item()) for v in state.values())
    diagnostics=run.stages[0].diagnostics
    rates=[float(d['learning_rate']) for d in diagnostics if 'learning_rate' in d]
    assert rates and all(math.isfinite(x) and x>0 for x in rates)
    bound=coordinate_bound(steps,max(rates),beta1,beta2)
    largest=0.0; name=''; exceeding=0; finite=True
    before=dict(initial.named_parameters());after=dict(final.named_parameters())
    assert before.keys()==after.keys()
    for key,value in after.items():
        old=before[key]; assert value.shape==old.shape
        finite=finite and bool(torch.isfinite(value).all()) and bool(torch.isfinite(old).all())
        delta=(value.detach().double()-old.detach().double()).abs()
        observed=float(delta.max())
        if observed>largest:largest,name=observed,key
        exceeding+=int((delta>bound).sum())
    moments_finite=all(bool(torch.isfinite(v[k]).all()) for v in state.values() for k in ('exp_avg','exp_avg_sq'))
    negative=sum(int((v['exp_avg_sq']<0).sum()) for v in state.values())
    counts={k:sum(int(d.get('numerical',{}).get(k,0)) for d in diagnostics)
            for k in ('nonfinite_count','rejected_steps','skipped_steps','legal_underflow_count','optimizer_steps')}
    ranges={}
    for key in ('loss','gradient_norm','entropy','reference_kl','collection_kl'):
        values=[float(d[key]) for d in diagnostics if key in d]
        ranges[key]={'count':len(values),'finite':all(math.isfinite(v) for v in values),
                     'first':values[0] if values else None,'last':values[-1] if values else None,
                     'minimum':min(values) if values else None,'maximum':max(values) if values else None}
    result={'run_id':run.id,'seed':run.seed,'source':proof['source'],
            'movement':asdict(Movement(largest,name,bound,exceeding,steps,finite,moments_finite,negative)),
            'diagnostic_counts':counts,'diagnostic_ranges':ranges,
            'interpretation':'Coarse retained-state audit; passing does not clear historical anomaly or establish determinism. Diagnostic loss/gradient samples need not cover every optimizer step.'}
    (directory/'numerical-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

assert abs(coordinate_bound(86,.0001,.9,.999)-.0130134)<1e-7
receipts={}
for cell in json.loads((ROOT/'seven-complete-results.json').read_text())['cells']:
    receipts[cell['name']]=audit(ROOT/'results'/cell['name'])
output={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cells':receipts}
(ROOT/'seven-cell-numerical-audit.json').write_text(json.dumps(output,indent=2)+'\n')
for name,result in receipts.items():print(name,json.dumps(result['movement']),result['diagnostic_counts'])
