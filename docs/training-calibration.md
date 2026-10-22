# Complete-loop training calibration

## Freeze prospective step targets

Use a separately authorized, representative completed pilot for each capacity.
`StepCalibration` reads an existing single-stage fresh self-play TrainingRun; it
runs no training and grants no allocation. It binds exact export bytes and uses
completed iterations divided by measured collection plus learning seconds.
Targets are floored to positive whole iterations. Initialization, evaluation,
export, setup, upload and cleanup remain separate machine/cost reserves.

```python
import json
from pathlib import Path
from manabot.training.experiment_execution import StepCalibration

pilots = {
    name: StepCalibration.from_export(Path(f".runs/{name}-pilot/run.json"))
    for name in ("small", "large")
}
requested_hours = {"small": 4, "large": 8}
targets = {name: pilots[name].updates_for(hours) for name, hours in requested_hours.items()}
# One shared grid; this example freezes roughly half an hour of the slower arm.
interval = min(pilot.updates_for(0.5) for pilot in pilots.values())
with Path(".runs/step-target-freeze.json").open("x") as output:
    json.dump({
        "calibration": {name: pilot.model_dump(mode="json") for name, pilot in pilots.items()},
        "requested_active_hours": requested_hours,
        "target_updates": targets,
        "checkpoint_updates": interval,
    }, output, indent=2)
```

Apply each target to `TrainSelfPlay.updates` and the **same** interval to every
`PlannedRun.checkpoint_updates` (or the local `ExperimentSchedule`). Commit the
frozen declaration/protocol before scientific launch. Record independent seeds,
held-out deals, hardware/source identities and the judgment that the pilot is
representative. A short or contended smoke proves the machinery only; count
extrapolation does not prove sustained throughput or strength.

Execution never consults pilot rates or elapsed hours to trigger scientific
checkpoints. Unequal targets retain extra endpoints, while comparisons use only
shared measured update milestones. Equal steps do not mean equal transitions,
optimizer calls, exposures, active time or dollars. Portable continuation's
`updates` target is absolute, not an additional segment count; its fresh stream
boundary remains explicit. JobSpec still owns hours, credentials, watchdogs,
export/upload/cleanup reserves and spending limits. Calibrate once per frozen
configuration rather than retuning a running cohort when throughput changes.

New declarations use Experiment/JobSpec and step cadence. This helper neither
launches a cohort nor migrates saved records. Non-self-play and multi-stage
recipes keep their own epoch/game units and explicit legacy cadence until their
clock contract is defined.
