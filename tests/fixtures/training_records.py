"""Real Etude records for bounded persistence fixtures, without launching a job.

Product recipes enter through the assigned `track`; storage tests construct the
public declaration and writer resource so their clocks and lineage are explicit.
"""

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from unittest.mock import patch

from etude.records import Clock, Continuation
from etude.run import declare, destinations
from etude.storage import Storage
from etude.train.tracking import TrainingRun
from pydantic import BaseModel


def recording(
    *,
    experiment: str,
    clocks: Sequence[str | Clock],
    directory: Path,
    run: str | None = None,
    kind: str | None = None,
    job: str | None = None,
    seed: int | None = None,
    config: BaseModel | Mapping[str, object] | str | None = None,
    continues: Continuation | None = None,
    storages: Sequence[Storage] = (),
    strict: bool = False,
    resume: bool = False,
) -> TrainingRun:
    assigned = {
        name: str(value)
        for name, value in (("ETUDE_JOB", job), ("ETUDE_SEED", seed))
        if value is not None
    }
    with patch.dict(os.environ, assigned):
        declared = declare(
            experiment=experiment,
            run=run,
            clocks=tuple(
                Clock(name=c, kind="step", meaning=c) if isinstance(c, str) else c
                for c in clocks
            ),
            kind=kind,
            config=config.model_dump(mode="json")
            if isinstance(config, BaseModel)
            else config,
            continues=continues,
            directory=directory,
            resume=resume,
        )
    return TrainingRun(
        experiment=experiment,
        name=declared.name,
        root=declared.root,
        clocks=declared.record.clocks,
        destinations=destinations(declared.base, storages),
        strict=strict,
        closed_clocks=True,
        inherited=None,
        seed=declared.record.seed or 0,
    )
