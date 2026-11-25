"""Matched native decision fixtures and provenance for observation benchmarks.

Fixture restoration and hashing happen outside timing. Each sample executes one
ordinary vector step, including Python failure capture, tensor sync and rewards.
Allocation profiling is a separate pass: Rust requested bytes, Torch CPU tensor
allocation events, and Python traced peak/live bytes have distinct scopes.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import tracemalloc
from typing import Any

import numpy as np
import torch

from manabot.env.observation import encode_observation
from manabot.env.vector_env import VectorEnv
import manabot.managym._managym as native
from manabot.semantic.decision_contract import SemanticDecisionContract
import managym


@dataclass
class DecisionCase:
    name: str
    root: managym.Env
    action: int


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        hasher.update(key.encode())
        hasher.update(str((value.shape, value.dtype.str)).encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def identity() -> dict[str, object]:
    """Binary/runtime identity; the paired runner also retains every source hash."""
    binary = Path(native.__file__).resolve()
    return {
        "extension": str(binary),
        "extension_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "package": str(Path(managym.__file__).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
        "load": os.getloadavg(),
    }


def _root(large: bool) -> managym.Env:
    env = managym.Env(seed=23)
    deck = {"Swamp": 20, "Deadly Precision": 40, "Gray Ogre": 4}
    env.reset([managym.PlayerConfig(name, deck) for name in ("a", "b")])
    # Both seats keep the seeded deal before the scenario installs its board.
    env.step(0)
    obs = env.step(0)[0]
    assert obs.turn.phase == managym.PhaseEnum.PRECOMBAT_MAIN
    actor = env.current_agent_index()
    assert actor is not None
    env.scenario_clear_hand(actor)
    for _ in range(32 if large else 1):
        env.scenario_force_card_in_hand(actor, "Deadly Precision")
    env.scenario_force_battlefield(actor, "Swamp", ready=True)
    env.scenario_force_battlefield(actor, "Gray Ogre", ready=True)
    env.scenario_force_battlefield(1 - actor, "Gray Ogre", ready=True)
    env.scenario_refresh()
    return env


def decision_cases() -> list[DecisionCase]:
    root = _root(False)
    cast = next(
        i
        for i, action in enumerate(
            root.observation_for_player(root.current_agent_index()).action_space.actions
        )
        if action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
    )
    target = root.clone_env()
    target.step(cast)
    payment = target.clone_env()
    payment.step(0)
    large = _root(True)
    # Retain an upstream Command whose *returned* tensor has 33 legal actions.
    # Casting from the large root covers a large input/offer set but returns a
    # small target decision, so it alone cannot measure large-output encoding.
    cursor = large.clone_env()
    for _ in range(20):
        actions = cursor.observation_for_player(
            cursor.current_agent_index()
        ).action_space.actions
        index = next(
            (
                i
                for i, action in enumerate(actions)
                if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            ),
            len(actions) - 1,
        )
        previous = cursor.clone_env()
        returned = cursor.step(index)[0]
        if len(returned.action_space.actions) >= 32:
            large_output = DecisionCase("large_legal_output", previous, index)
            break
    else:
        raise AssertionError("large postcombat-main output not reached")
    return [
        DecisionCase("priority", root, cast),
        DecisionCase("target", target, 0),
        DecisionCase("additional_cost", payment, 0),
        DecisionCase("large_legal", large, cast),
        large_output,
    ]


def matched_breakdown(env: VectorEnv, steps: int) -> list[dict[str, Any]]:
    """Return per-family exact-match proofs, timings and separate allocation data."""
    rows: list[dict[str, Any]] = []
    vector = env._rust_env
    for case in decision_cases():
        root = case.root
        actor = root.current_agent_index()
        assert actor is not None
        observation = root.observation_for_player(actor)
        contract = SemanticDecisionContract.from_env(root)
        command = contract.command(int(contract.frame.offers[case.action]["id"]))
        # Prove the positional training action is the retained semantic Command.
        semantic = root.clone_env()
        contract.apply_command(semantic, command)
        positional = root.clone_env()
        post_observation = positional.step(case.action)[0]
        assert semantic.state_digest() == positional.state_digest()
        expected = encode_observation(env.observation_space, post_observation)
        encoded = encode_observation(env.observation_space, observation)
        actions = torch.full((env.num_envs,), case.action, dtype=torch.int64)

        snapshot = root.snapshot()
        loads = [
            managym.SlotLoad.snapshot(slot, snapshot) for slot in range(env.num_envs)
        ]

        vector.load_slots_into_buffers(loads)
        env.step(actions)
        for key, value in expected.items():
            for row in env._buffers[key]:
                np.testing.assert_array_equal(row, value, err_msg=case.name + ":" + key)
        frames = vector.semantic_decision_frames_json()
        assert all(
            json.loads(frame) == json.loads(positional.semantic_decision_frame_json())
            for frame in frames
        )
        for _ in range(20):
            vector.load_slots_into_buffers(loads)
            env.step(actions)
        samples = []
        for _ in range(steps):
            vector.load_slots_into_buffers(loads)
            start = time.perf_counter_ns()
            env.step(actions)
            samples.append(time.perf_counter_ns() - start)

        vector.load_slots_into_buffers(loads)
        vector.native._benchmark_allocations_start()
        try:
            env.step(actions)
        finally:
            allocations, allocated_bytes = vector.native._benchmark_allocations_stop()
        vector.load_slots_into_buffers(loads)
        tracemalloc.start()
        env.step(actions)
        python_live, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        vector.load_slots_into_buffers(loads)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
        ) as profile:
            env.step(actions)
        events = profile.events()
        torch_allocations = [
            e.self_cpu_memory_usage for e in events if e.self_cpu_memory_usage > 0
        ]
        output_bytes = sum(a.nbytes for a in env._buffers.values())
        sync_bytes = sum(
            dst.numel() * dst.element_size()
            for key, dst in env._obs_tensors.items()
            if dst.data_ptr() != env._obs_cpu_views[key].data_ptr()
        ) + sum(
            t.numel() * t.element_size()
            for t in (env._reward_tensor, env._terminated_tensor, env._truncated_tensor)
        )
        rows.append(
            {
                "case": case.name,
                "seed": 23,
                "actions": len(observation.action_space.actions),
                "output_actions": len(post_observation.action_space.actions),
                "input": json.loads(observation.toJSON()),
                "input_digest": digest(json.loads(observation.toJSON())),
                "continuation_digest": digest(
                    json.loads(observation.continuation_json())
                )
                if hasattr(observation, "continuation_json")
                else None,
                "encoded_digest": array_digest(encoded),
                "legacy_encoded_digest": array_digest(
                    {k: v for k, v in encoded.items() if k != "relations"}
                ),
                "command": json.loads(command.to_json()),
                "command_digest": digest(json.loads(command.to_json())),
                "output_digest": array_digest(expected),
                "legacy_output_digest": array_digest(
                    {
                        key: value
                        for key, value in expected.items()
                        if key != "relations"
                    }
                ),
                "post_state_digest": positional.state_digest(),
                "warmup_steps": 20,
                "samples_ns_per_batch": samples,
                "median_ns_per_env": float(np.median(samples)) / env.num_envs,
                "steps_per_second": env.num_envs * steps * 1e9 / sum(samples),
                "allocations": {
                    "rust_requests_per_batch": allocations,
                    "rust_requested_bytes_per_batch": allocated_bytes,
                    "torch_cpu_positive_allocation_events_per_batch": len(
                        torch_allocations
                    ),
                    "torch_cpu_positive_allocation_bytes_per_batch": sum(
                        torch_allocations
                    ),
                    "python_traced_live_bytes_per_batch": python_live,
                    "python_traced_peak_bytes_per_batch": python_peak,
                },
                "bytes": {
                    "native_dense_output_per_batch": output_bytes,
                    "tensor_sync_destination_per_batch": sync_bytes,
                    "relations_per_batch": env._buffers.get(
                        "relations", np.empty(0)
                    ).nbytes,
                    "scope": "Logical output/copy bytes, not physical memory-bus traffic or repeated stores. Python peaks are not allocation traffic.",
                },
            }
        )
    return rows
