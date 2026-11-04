"""Bounded same-host cost observations against published 9db184b9.

Three measured passes after one warmup, CPU one thread, no optimizer or games
beyond the existing directed scenario. Native code is unchanged in the costed
paths; its current timings are attribution, not a claimed before/after speedup.
Run with `uv run --extra dev python scratch/naming-risk-probe.py`.
"""

from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
from time import perf_counter
from types import ModuleType
from typing import TypeVar

import numpy as np
import torch

from manabot.env.observation import PERMANENT_DEFINITION_INDEX
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.training.rollout.actions import PackedActions
import managym

T = TypeVar("T")
BASE = "9db184b9aa3df9286b07abdafa2d41db0d93e651"


def timed(operation: Callable[[], T]) -> tuple[T, dict[str, object]]:
    operation()
    times: list[float] = []
    for _ in range(3):
        start = perf_counter()
        result = operation()
        times.append(1000 * (perf_counter() - start))
    return result, {"median_ms": statistics.median(times), "samples_ms": times}


def main() -> None:
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path("tests/model").resolve()))
    namespace = runpy.run_path("tests/model/test_catalog_naming.py")
    root = namespace["naming_trace"].__wrapped__().root
    space, obs = namespace["_mixed_naming_batch"](root)
    # Load the published Python implementation without changing checkout/native
    # state. Its absolute imports resolve unchanged shared model dependencies.
    baseline = ModuleType("manabot.model.naming_published_reference")
    sys.modules[baseline.__name__] = baseline
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:manabot/model/actor_critic.py"], text=True
    )
    exec(compile(source, f"{BASE}:actor_critic.py", "exec"), baseline.__dict__)
    config = ModelConfig(hidden_dim=16, world=managym.WORLD_SETUP, policy_film=True)
    models: dict[str, ActorCritic] = {}
    setup: dict[str, object] = {}
    for label, cls in (("published", baseline.ActorCritic), ("retained", ActorCritic)):

        def construct() -> ActorCritic:
            torch.manual_seed(243)
            return cls(space, config).eval()

        model, setup_cost = timed(construct)
        setup[label] = {"model_setup": setup_cost}
        models[label] = model
    costs: dict[str, object] = {}
    for label, model in models.items():
        target = obs["actions_valid"].argmax(-1)

        def forward() -> tuple[torch.Tensor, torch.Tensor]:
            return model.forward_distribution(obs)

        _, forward_cost = timed(forward)

        def backward() -> None:
            model.zero_grad(set_to_none=True)
            logits, value = forward()
            (
                torch.nn.functional.cross_entropy(logits, target)
                + value.square().mean()
            ).backward()

        _, combined_cost = timed(backward)
        backward_samples: list[float] = []
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            logits, value = forward()
            loss = (
                torch.nn.functional.cross_entropy(logits, target)
                + value.square().mean()
            )
            start = perf_counter()
            loss.backward()
            backward_samples.append(1000 * (perf_counter() - start))
        backward_cost = {
            "median_ms": statistics.median(backward_samples),
            "samples_ms": backward_samples,
        }
        state = model._gather_object_embeddings(obs)
        objects = model._attend_objects(state.objects, state.is_agent, state.validity)[
            :, : state.validity.shape[1]
        ]

        def score_actions() -> torch.Tensor:
            if label == "published":
                used = obs["actions_valid"].bool().any(0).nonzero()
                count = int(used[-1]) + 1
                features = model._gather_informed_actions(
                    obs, objects, state, action_count=count
                )
                features = model._condition_actions(features, objects, state.validity)
            else:
                batches, slots = obs["actions_valid"].bool().nonzero(as_tuple=True)
                features = model._gather_informed_actions(
                    obs, objects, state, batches, slots
                )
                valid = state.validity.unsqueeze(-1)
                pooled = (objects * valid).sum(1) / valid.sum(1).clamp_min(1)
                features = features * (1 + model.policy_film(pooled)[batches])
            return model.policy_head(features)

        _, scoring_cost = timed(score_actions)
        programs: list[dict[str, object]] = []
        encoder = model.semantic_cards
        assert encoder is not None
        for name, selection in (("ordinary", slice(2, 3)), ("naming", slice(0, 1))):
            subset = {key: value[selection] for key, value in obs.items()}
            permanent_ids = torch.stack(
                [
                    subset[key][..., PERMANENT_DEFINITION_INDEX]
                    for key in ("agent_permanents", "opponent_permanents")
                ],
                dim=1,
            )
            ids = model._referenced_definitions(subset, permanent_ids)
            _, elapsed = timed(lambda: encoder.definitions(ids))
            lengths = encoder.lengths[ids - 1]
            programs.append(
                {
                    "case": name,
                    "executable_programs": len(ids),
                    "expanded_tokens": int(lengths.sum()),
                    **elapsed,
                }
            )
        costs[label] = {
            "forward": forward_cost,
            "forward_and_backward": combined_cost,
            "backward": backward_cost,
            "action_networks": scoring_cost,
            "program_encoding": programs,
        }
    with torch.no_grad():
        logits, _ = models["published"](obs)
        logs = logits.masked_fill(obs["actions_valid"] == 0, -torch.inf).log_softmax(-1)
    arrays = {key: value.numpy() for key, value in obs.items()}
    probabilities = logs.exp().numpy()
    packed, packing = timed(
        lambda: PackedActions.pack(arrays, probabilities, logs.numpy())
    )
    _, unpacking = timed(lambda: packed.unpack(np.array([2, 0])))
    fields = ("actions", "actions_valid", "action_focus", "name_candidates")
    dense_action_bytes = (
        sum(arrays[key].nbytes for key in fields)
        + probabilities.nbytes
        + logs.numpy().nbytes
    )
    objects_bytes = sum(
        value.nbytes for key, value in arrays.items() if key not in fields
    )
    raw, projection = timed(
        lambda: root.observation_for_player(root.current_agent_index())
    )
    buffers = space.encoder.allocate()
    _, encoding = timed(lambda: root.encode_observation_into(raw, buffers))
    _, fork = timed(root.clone_env)
    _, decision = timed(root.semantic_decision_frame_json)
    report = {
        "baseline": BASE,
        "device": "cpu",
        "threads": 1,
        "width": 16,
        "measured_passes": 3,
        "optimizer_steps": 0,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "legal_rows": obs["actions_valid"].sum(-1).int().tolist(),
        "setup": setup,
        "costs": costs,
        "catalog": asdict(models["retained"].semantic_cards.catalog_shape),
        "storage": {
            "dense_action_bytes": dense_action_bytes,
            "packed_action_bytes": packed.nbytes,
            "object_bytes": objects_bytes,
            "dense_total_bytes": dense_action_bytes + objects_bytes,
            "packed_total_bytes": packed.nbytes + objects_bytes,
            "dense_live_observation_bytes": sum(
                value.nbytes for value in arrays.values()
            ),
            "packing": packing,
            "two_row_materialization": unpacking,
        },
        "native_current_only": {
            "projection": projection,
            "encoding_existing_buffers": encoding,
            "naming_fork": fork,
            "semantic_decision_projection": decision,
        },
        "limitations": [
            "Three local CPU observations are not a throughput or strength claim.",
            "928 validated names do not constitute 928 executable programs; this measures the installed catalog.",
            "Native buffers and returned logits remain dense. Each minibatch reconstructs the configured action axis.",
            "The native engine cost paths are unchanged by this reduction; no native speedup is claimed.",
            "Only CPU is configured for this probe; GPU learning equivalence and peak device memory are not measured.",
            "Equal-rule competency uses a two-example linear-head fit. A 16-step SGD probe did not converge.",
        ],
    }
    Path("scratch/naming-risk-results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
