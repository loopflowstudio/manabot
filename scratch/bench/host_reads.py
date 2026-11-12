"""Count host reads per forward (MTG-299 acceptance).

uv run --extra dev python scratch/bench/host_reads.py
Instruments Tensor.item/__bool__/cpu/tolist/nonzero and torch.equal/nonzero
during one forward on inputs as collection and learning feed them.
"""

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
import traceback

import numpy as np
import torch

from manabot.model.inference import PolicyInference

from capture import admit, build

COUNTS: Counter[str] = Counter()
SITES: Counter[str] = Counter()


def _wrap(name: str, original: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        if not getattr(wrapped, "busy", False):
            wrapped.busy = True
            try:
                COUNTS[name] += 1
                frame = next(
                    f
                    for f in reversed(traceback.extract_stack()[:-1])
                    if "manabot/model" in f.filename
                )
                SITES[f"{name} {frame.filename.split('/')[-1]}:{frame.lineno}"] += 1
            except StopIteration:
                pass
            finally:
                wrapped.busy = False
        return original(*args, **kwargs)

    return wrapped


@contextmanager
def counting() -> Iterator[None]:
    patched = [
        (torch.Tensor, "item"),
        (torch.Tensor, "__bool__"),
        (torch.Tensor, "cpu"),
        (torch.Tensor, "tolist"),
        (torch.Tensor, "nonzero"),
        (torch, "equal"),
        (torch, "nonzero"),
    ]
    saved = [(owner, name, getattr(owner, name)) for owner, name in patched]
    for owner, name, original in saved:
        label = f"{'Tensor.' if owner is torch.Tensor else 'torch.'}{name}"
        setattr(owner, name, _wrap(label, original))
    try:
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


def report(label: str) -> None:
    print(f"== {label}: total {sum(COUNTS.values())}  {dict(COUNTS)}")
    for site, count in sorted(SITES.items()):
        print(f"   {count:4d} {site}")
    COUNTS.clear()
    SITES.clear()


def main() -> None:
    for variant in ("semantic", "history"):
        agent, collector = build(variant)
        batch = collector.collect(agent, 16)
        rows = batch.observations(np.arange(batch.actions.size))
        obs = {
            key: torch.from_numpy(value).float()
            if key == "action_focus"
            else torch.from_numpy(value)
            for key, value in rows.observation.items()
        }
        with PolicyInference(agent) as scope, torch.no_grad():
            prepared = scope.prepare(obs)
            agent.forward_distribution(prepared)
            with counting():
                agent.forward_distribution(prepared)
        report(f"{variant} collection forward (prepared)")
        trained = deepcopy(agent).train()
        admitted = admit(trained, obs)
        with counting():
            trained.forward_distribution(admitted)
        report(f"{variant} learner forward")


if __name__ == "__main__":
    main()
