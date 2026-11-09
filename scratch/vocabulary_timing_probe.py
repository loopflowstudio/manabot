"""Bounded same-L4 diagnostic over identical frozen-frontier observations.

Measures cached collection inference and differentiable forward/backward, excluding
collection, optimizer updates and transfer. This isolates model cost, not update
throughput or learning quality. ABBA ordering bounds drift; profiles are separate.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from experiments.tla_frontier import SMALL, frozen
from experiments.tla_frontier_facts import sample_positions
from manabot.env.observation import ObservationSpace
from manabot.model.actor_critic import ActorCritic
from manabot.model.inference import PolicyInference


@dataclass
class Reading:
    vocabulary: str
    mode: str
    batch: int
    repetition: int
    seconds: float


def evaluate(model: ActorCritic, batch: dict[str, torch.Tensor], learning: bool) -> None:
    if learning:
        model.zero_grad(set_to_none=True)
    with nullcontext() if learning else torch.no_grad():
        logits, values = model.forward_distribution(batch)
        # A bounded diagnostic loss traverses both heads, without any optimizer step.
        valid = batch['actions_valid'].bool()
        loss = logits[valid].square().mean() + values.square().mean()
        if learning:
            loss.backward()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic requires CUDA; no local CPU fallback')
    torch.set_num_threads(1)
    out = Path(os.environ['ETUDE_OUTPUT_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    regimes, protocol = frozen()
    regime = regimes[SMALL]
    positions = sample_positions(regime, protocol)
    np.savez_compressed(out / 'positions.npz', **positions)
    digest = hashlib.sha256()
    for key, value in sorted(positions.items()):
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    models: dict[str, ActorCritic] = {}
    for vocabulary in ('catalog', 'hashed'):
        torch.manual_seed(21501)
        models[vocabulary] = ActorCritic(
            ObservationSpace(regime.observation),
            regime.agent.model_copy(update={'record_vocabulary': vocabulary}),
        ).cuda()
    readings: list[Reading] = []
    for size in (64, 256):
        if len(positions['facts']) < size:
            raise RuntimeError('Insufficient distinct positions for batch')
        # Spread selection across the entire retained corpus, not only its first decks.
        indexes = np.linspace(0, len(positions['facts']) - 1, size, dtype=int)
        cpu = {key: torch.from_numpy(value[indexes]) for key, value in positions.items()}
        device = {key: value.cuda() for key, value in cpu.items()}
        for learning in (False, True):
            mode = 'forward-backward' if learning else 'cached-inference'
            for repetition, vocabulary in enumerate(('catalog', 'hashed', 'hashed', 'catalog')):
                model = models[vocabulary]
                model.train(learning)
                with PolicyInference(model) as scope:
                    batch = device if learning else scope.prepare(cpu)
                    # Two ordinary warmup calls; measurements have synchronized endpoints.
                    for _ in range(2):
                        evaluate(model, batch, learning)
                    for _ in range(5):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        evaluate(model, batch, learning)
                        torch.cuda.synchronize()
                        readings.append(Reading(vocabulary, mode, size, repetition, time.perf_counter() - start))
            # Instrumentation has no bearing on the preceding wall-time samples.
            for vocabulary, model in models.items():
                model.train(learning)
                with PolicyInference(model) as scope:
                    batch = device if learning else scope.prepare(cpu)
                    evaluate(model, batch, learning)
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as trace:
                        evaluate(model, batch, learning)
                        torch.cuda.synchronize()
                    label = f'{vocabulary}-{mode}-b{size}'
                    trace.export_chrome_trace(str(out / f'{label}.trace.json'))
                    (out / f'{label}.txt').write_text(trace.key_averages().table(sort_by='self_cpu_time_total', row_limit=50))
    payload = {
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(), 'positions': len(positions['facts']),
        'positions_sha256': digest.hexdigest(), 'readings': [asdict(row) for row in readings],
        'limitations': 'Fresh parameters; fixed random-play corpus; no optimizer step, transfer or engine timing. Not whole-update throughput.',
    }
    (out / 'timing.json').write_text(json.dumps(payload, indent=2))
    for size in (64, 256):
        for mode in ('cached-inference', 'forward-backward'):
            medians = {v: median(r.seconds for r in readings if (r.vocabulary, r.batch, r.mode) == (v, size, mode)) for v in models}
            print(size, mode, medians, 'catalog/hashed', medians['catalog'] / medians['hashed'], flush=True)


if __name__ == '__main__':
    main()
