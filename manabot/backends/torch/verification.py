"""Torch frozen-policy invariance diagnostics over admitted inputs.

`inspect_policy` measures the verifier's host choice corpus using inference
preparation. The verifier owns the cases and result records; this module owns
the tensor arithmetic and frozen-checkpoint forwards.
"""

from __future__ import annotations

from pathlib import Path

import torch

from manabot.backends.torch.arrays import device_of, on_device, to_numpy
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.infra.identity import file_sha256
from manabot.model.world import validate_agent_setup
from manabot.verify.action_cases import reorder_objects, reorder_offers
from manabot.verify.choice_corpus import PACK, Case
from manabot.verify.learning_regression import ChoiceScore, PolicySnapshot
import managym


@torch.no_grad()
def inspect_policy(path: Path, cases: list[Case], updates: int) -> PolicySnapshot:
    """Use the checkpoint loader and exact setup; never refit weights."""
    if not cases:
        raise ValueError("empty frozen choice corpus")
    if len(cases) > 32:
        chunks = [
            inspect_policy(path, cases[i : i + 32], updates)
            for i in range(0, len(cases), 32)
        ]
        return PolicySnapshot(
            updates=updates,
            checkpoint_sha256=chunks[0].checkpoint_sha256,
            choices=[choice for chunk in chunks for choice in chunk.choices],
            maximum_invariance_error=max(
                chunk.maximum_invariance_error for chunk in chunks
            ),
        )
    agent, _ = load_checkpoint(str(path))
    configs = [
        managym.authored_deck_setup(PACK, key) for key in ("ur_lessons", "gw_allies")
    ]
    validate_agent_setup(agent, configs)
    batch = on_device(
        agent.prepare(
            {
                key: to_numpy(torch.cat([case.obs[key] for case in cases]))
                for key in cases[0].obs
            },
            inference=True,
        ),
        device_of(agent),
    )
    logits, values = agent(batch)
    probabilities = logits.softmax(-1)
    if not torch.isfinite(probabilities).all() or not torch.isfinite(values).all():
        raise ValueError("nonfinite frozen checkpoint output")
    restored, _ = load_checkpoint(str(path))
    loaded_logits, loaded_values = restored(batch)
    errors = [
        float((loaded_logits.softmax(-1) - probabilities).abs().max()),
        float((loaded_values - values).abs().max()),
    ]
    scores = []
    for row, case in enumerate(cases):
        obs = case.obs
        single, value = agent(
            on_device(
                agent.prepare(
                    {key: to_numpy(value) for key, value in (obs).items()},
                    inference=True,
                ),
                device_of(agent),
            )
        )
        p = single.softmax(-1)
        count = int(obs["actions_valid"].sum())
        order = torch.arange(count - 1, -1, -1)
        offers, offer_value = agent(
            on_device(
                agent.prepare(
                    {
                        key: to_numpy(value)
                        for key, value in (reorder_offers(obs, order)).items()
                    },
                    inference=True,
                ),
                device_of(agent),
            )
        )
        objects, object_value = agent(
            on_device(
                agent.prepare(
                    {
                        key: to_numpy(value)
                        for key, value in (reorder_objects(obs)).items()
                    },
                    inference=True,
                ),
                device_of(agent),
            )
        )
        errors.extend(
            [
                float((p[0] - probabilities[row]).abs().max()),
                float((value[0] - values[row]).abs().max()),
                float((offers.softmax(-1)[0, :count] - p[0, order]).abs().max()),
                float((objects.softmax(-1) - p).abs().max()),
                float((offer_value - value).abs().max()),
                float((object_value - value).abs().max()),
                float(p[obs["actions_valid"] == 0].sum()),
            ]
        )
        # Log-sum-exp avoids underflow and gives identical public copies shared
        # probability mass. Its sign has no prescribed strategic interpretation.
        left, right = case.groups
        log_odds = float(single[0, left].logsumexp(0) - single[0, right].logsumexp(0))
        scores.append(
            ChoiceScore(
                case=case.name,
                family=case.family,
                log_odds=log_odds,
                value=float(value[0]),
            )
        )
    maximum = max(errors)
    if maximum > 1e-4:
        raise ValueError(f"frozen-policy invariance error {maximum} exceeds 1e-4")
    return PolicySnapshot(
        updates=updates,
        checkpoint_sha256=file_sha256(path),
        choices=scores,
        maximum_invariance_error=maximum,
    )
