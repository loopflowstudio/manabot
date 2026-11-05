"""Bounded profiling and balanced CPU timing of real naming observations.

Run with uv run --extra dev python scratch/naming-speed.py profile|measure.
The reference is loaded from Git without changing the checkout or checkpoint.
No optimizer steps, synthetic load, or detached training cache are used.
"""

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import torch

from manabot.env.observation import PERMANENT_DEFINITION_INDEX
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.policy_distribution import clipped_surrogate, policy_logs, reverse_kl
from manabot.training.learning.damped_policy import damped_policy_loss
from manabot.training.rollout.actions import PackedActions
import managym

BASE = "b31998cc38bb3b86abedce397e2531c4e0d91379"
OUT = Path(".lf/logs/naming-speed")


def measured(operation: Callable[[], object]) -> dict[str, object]:
    before = resource.getrusage(resource.RUSAGE_SELF)
    load = os.getloadavg()
    start = perf_counter()
    operation()
    elapsed = perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "wall_s": elapsed,
        "cpu_s": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "voluntary_switches": after.ru_nvcsw - before.ru_nvcsw,
        "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
        "load_before": load,
        "load_after": os.getloadavg(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("profile", "measure"))
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--label", default="final")
    parser.add_argument(
        "--side", choices=("candidate", "reference"), default="candidate"
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{args.label}-model-{args.width}.json"
    previous_profiles = list(OUT.glob(f"{args.label}-profile-*-{args.width}.*"))
    if (args.mode == "measure" and destination.exists()) or (
        args.mode == "profile" and previous_profiles
    ):
        raise FileExistsError("retain prior evidence; choose a new --label")
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path("tests/model").resolve()))
    namespace = runpy.run_path("tests/model/test_catalog_naming.py")
    root = namespace["naming_trace"].__wrapped__().root
    space, mixed = namespace["_mixed_naming_batch"](root)
    baseline = ModuleType("naming_speed_reference")
    sys.modules[baseline.__name__] = baseline
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:manabot/model/semantic_cards.py"], text=True
    )
    exec(compile(source, f"{BASE}:semantic_cards.py", "exec"), baseline.__dict__)
    models: dict[str, ActorCritic] = {}
    setup: dict[str, object] = {}
    for side in ("reference", "candidate"):
        torch.manual_seed(243)
        start = perf_counter()
        model = ActorCritic(
            space,
            ModelConfig(
                hidden_dim=args.width, world=managym.WORLD_SETUP, policy_film=True
            ),
        ).eval()
        if side == "reference":
            assert model.semantic_cards is not None
            model.semantic_cards.definitions = (
                baseline.SemanticCardEncoder.definitions.__get__(model.semantic_cards)
            )
        models[side] = model
        setup[side] = perf_counter() - start
    samples: list[dict[str, object]] = []
    for case, selection in (
        ("mixed", slice(None)),
        ("ordinary", slice(2, 3)),
        ("naming", slice(0, 1)),
    ):
        obs = {key: value[selection] for key, value in mixed.items()}
        valid = obs["actions_valid"].bool()
        actions = valid.long().argmax(-1)
        advantages = torch.linspace(-0.2, 0.7, len(actions))
        reference = valid.float() / valid.sum(-1, keepdim=True)
        with torch.no_grad():
            behavior = policy_logs(
                models["reference"].forward_distribution(obs)[0], valid
            )
        arrays = {key: value.numpy() for key, value in obs.items()}
        packed = PackedActions.pack(arrays, behavior.exp().numpy(), behavior.numpy())
        indexes = np.arange(len(actions))
        for side in ("candidate", "reference", "reference", "candidate") * 2:
            model = models[side]

            def forward() -> tuple[torch.Tensor, torch.Tensor]:
                return model.forward_distribution(obs)

            def backward() -> None:
                model.zero_grad(set_to_none=True)
                logits, value = forward()
                (
                    torch.nn.functional.cross_entropy(
                        logits, obs["actions_valid"].argmax(-1)
                    )
                    + value.square().mean()
                ).backward()

            def learner(rule: str) -> None:
                model.zero_grad(set_to_none=True)
                restored = packed.unpack(indexes)
                minibatch = dict(obs)
                minibatch.update(
                    {
                        key: torch.from_numpy(value)
                        for key, value in restored.observation.items()
                    }
                )
                logits, value = model.forward_distribution(minibatch)
                logs = torch.from_numpy(restored.log_probabilities)
                if rule == "damped":
                    loss = damped_policy_loss(
                        logits,
                        actions,
                        advantages,
                        logs,
                        reference,
                        valid,
                        clip=0.2,
                        collection_kl=0.1,
                        tau=0.05,
                    ).mean()
                else:
                    current = policy_logs(logits, valid)
                    ratio = (current - logs).gather(-1, actions[:, None]).squeeze(-1)
                    loss = (
                        clipped_surrogate(ratio, advantages, 0.2)
                        + 0.1 * reverse_kl(current, logs, valid)
                        + 0.05
                        * reverse_kl(
                            current, reference.masked_fill(~valid, 1).log(), valid
                        )
                    ).mean()
                (loss + value.square().mean()).backward()

            permanent_ids = torch.stack(
                [
                    obs[key][..., PERMANENT_DEFINITION_INDEX]
                    for key in ("agent_permanents", "opponent_permanents")
                ],
                dim=1,
            )
            ids = model._referenced_definitions(obs, permanent_ids)
            assert model.semantic_cards is not None
            encoder = model.semantic_cards
            with torch.no_grad():
                state = model._gather_object_embeddings(obs)
                objects = model._attend_objects(
                    state.objects, state.is_agent, state.validity
                )[:, : state.validity.shape[1]]

            def score_actions() -> torch.Tensor:
                batches, slots = valid.nonzero(as_tuple=True)
                features = model._gather_informed_actions(
                    obs, objects, state, batches, slots
                )
                mask = state.validity.unsqueeze(-1)
                pooled = (objects * mask).sum(1) / mask.sum(1).clamp_min(1)
                assert model.policy_film is not None
                return model.policy_head(
                    features * (1 + model.policy_film(pooled)[batches])
                )

            if args.mode == "profile":
                if side != args.side:
                    continue
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    profile_memory=True,
                    record_shapes=True,
                ) as profile:
                    backward()
                (OUT / f"{args.label}-profile-{case}-{args.width}.txt").write_text(
                    profile.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=30
                    )
                )
                (OUT / f"{args.label}-profile-{case}-{args.width}.json").write_text(
                    json.dumps(
                        [
                            {
                                "name": event.key,
                                "calls": event.count,
                                "self_cpu_us": event.self_cpu_time_total,
                                "self_allocated_bytes": event.self_cpu_memory_usage,
                                "allocated_bytes": event.cpu_memory_usage,
                            }
                            for event in profile.key_averages()
                        ],
                        indent=2,
                    )
                    + "\n"
                )
                # One sample per workload; timing is a separate invocation.
                break
            backward()
            sample = {
                "case": case,
                "side": side,
                "programs": len(ids),
                "tokens": int(encoder.lengths[ids - 1].sum()),
                "forward": measured(forward),
                "forward_backward": measured(backward),
                "program_encoding": measured(lambda: encoder.definitions(ids)),
                "action_scoring": measured(score_actions),
                "materialization": measured(lambda: packed.unpack(indexes)),
                "ppo_minibatch": measured(lambda: learner("ppo")),
                "damped_minibatch": measured(lambda: learner("damped")),
            }
            model.zero_grad(set_to_none=True)
            logits, value = forward()
            loss = (
                torch.nn.functional.cross_entropy(logits, actions)
                + value.square().mean()
            )
            sample["backward_only"] = measured(loss.backward)
            samples.append(sample)
            print(f"{case} {side}: {samples[-1]}", flush=True)
    if args.mode == "measure":
        raw = root.observation_for_player(root.current_agent_index())
        buffers = space.encoder.allocate()
        native = {
            "projection": measured(
                lambda: root.observation_for_player(root.current_agent_index())
            ),
            "encoding": measured(lambda: root.encode_observation_into(raw, buffers)),
            "fork": measured(root.clone_env),
            "decision_projection": measured(root.semantic_decision_frame_json),
        }
        destination.write_text(
            json.dumps(
                {
                    "baseline": BASE,
                    "width": args.width,
                    "threads": 1,
                    "setup_s": setup,
                    "samples": samples,
                    "native_unchanged": native,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
