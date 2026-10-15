"""Load the immutable pre-storage collector for paired experimental controls.

Only the experiment and its differential tests import this module. Production
has one collector implementation. The frozen source's hash and original commit
are recorded beside it; no behavior is repaired or extended in that fixture.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import cast
from unittest.mock import patch

from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training import execution


@lru_cache(maxsize=1)
def frozen_collector() -> type[SeatRoutedCollector]:
    path = Path(__file__).parents[1] / "data/worker-scaling/collector-a07b3cd3.py"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("frozen collector source differs from its receipt")
    name = "experiments.frozen_collector_a07b3cd3"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen collector module has no loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = (
        module  # Dataclasses and private recovery pickles name their module.
    )
    spec.loader.exec_module(module)
    # The historical class has the same public contract but a separate Python
    # identity. Keep that dynamic, immutable-source boundary inside the control.
    return cast(type[SeatRoutedCollector], module.SeatRoutedCollector)


@contextmanager
def collector_implementation(reference: bool) -> Iterator[type[SeatRoutedCollector]]:
    kind = frozen_collector() if reference else SeatRoutedCollector
    with patch.object(execution, "SeatRoutedCollector", kind):
        yield kind
