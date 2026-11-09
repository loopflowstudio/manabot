"""Bounded same-L4 diagnostic over identical frozen-frontier observations.

Measures cached collection inference and differentiable forward/backward, excluding
collection, optimizer updates and transfer. This isolates model cost, not update
throughput or learning quality. ABBA ordering bounds drift; profiles are separate.
"""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import sys
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
    spec = importlib.util.spec_from_file_location('candidate_shared', '/tmp/c5fba4_shared_effects.py')
    assert spec is not None and spec.loader is not None
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    models: dict[str, ActorCritic] = {}
    for vocabulary in ('catalog', 'batched'):
        torch.manual_seed(21501)
        models[vocabulary] = ActorCritic(
            ObservationSpace(regime.observation),
            regime.agent.model_copy(update={'record_vocabulary': 'catalog'}),
        ).cuda()
        if vocabulary == 'batched':
            original = models[vocabulary].shared_effects
            replacement = candidate.SharedEffectEncoder(regime.agent.hidden_dim, version=regime.observation.policy_input_version, vocabulary='catalog').cuda()
            replacement.load_state_dict(original.state_dict())
            models[vocabulary].shared_effects = replacement
    check_batch = {key: torch.from_numpy(value[:64]).cuda() for key, value in positions.items()}
    checks = []
    for model in models.values():
        model.train()
        torch.manual_seed(21501)
        logits, values = model.forward_distribution(check_batch)
        (logits[check_batch['actions_valid'].bool()].square().mean() + values.square().mean()).backward()
        checks.append((logits.detach(), values.detach(), {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}))
    for old, new in zip(checks[0][:2], checks[1][:2], strict=True):
        torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-6)
    assert checks[0][2].keys() == checks[1][2].keys()
    for name in checks[0][2]:
        torch.testing.assert_close(checks[0][2][name], checks[1][2][name], rtol=1e-3, atol=1e-5, msg=name)
    (out / 'equivalence.txt').write_text('Full-model CUDA outputs and all parameter gradients agree at rtol=1e-5/1e-3, atol=1e-6/1e-5.\n')
    del checks
    torch.manual_seed(21501)
    models['hashed'] = ActorCritic(
        ObservationSpace(regime.observation),
        regime.agent.model_copy(update={'record_vocabulary': 'hashed'}),
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
            for repetition, vocabulary in enumerate(('catalog', 'batched', 'hashed', 'hashed', 'batched', 'catalog')):
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
            print(size, mode, medians, 'original/batched', medians['catalog'] / medians['batched'], 'batched/hashed', medians['batched'] / medians['hashed'], flush=True)


if __name__ == '__main__':
    main()
