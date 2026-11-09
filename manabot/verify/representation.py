"""Constructed Command roots and bounded representation proofs.

Shared roots serve policy tests and cost measurements; the v5 proof below is
not a learning or speed ranking.

One forward/backward per arm per native fixture, CPU and one thread. Retains
both viewers' v5 tensors and hashes so a result cannot silently substitute a
new input. The source-dependent effect gap and frozen capacities are unchanged;
this command does not certify the proposed replacement native transport.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time

from managym.decision_capture import DecisionInput
import numpy as np
import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig, StateEncoder
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.infra.source import source_commit, source_dirty
from manabot.model.actor_critic import ActorCritic
from manabot.model.architecture import architecture_receipt
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import native_fixtures, priority_root
import managym

ARMS: tuple[StateEncoder, ...] = ("rows", "typed_pool", "typed_gru")


def source_effect_root() -> managym.Env:
    """Reach Ty Lee's source-dependent effect through a cast and target Command."""
    env = priority_root(
        ("Ty Lee, Chi Blocker",), ("Island",) * 3, opponent=("Gray Ogre",)
    )
    for step in range(12):
        obs = env.observation_for_player(env.current_agent_index())
        if any(p.effects for p in (*obs.agent_permanents, *obs.opponent_permanents)):
            return env
        decision = capture_text_decision(env)
        source = DecisionInput.model_validate_json(decision.capture_json)
        assert source.observation.decision is not None
        action = next(
            action
            for offer, action in zip(
                source.observation.decision.offers, decision.actions, strict=True
            )
            if (step == 0 and offer["verb"] == "cast")
            or (
                step > 0
                and (
                    offer["verb"] in ("pay_cost", "pass_priority")
                    or (
                        offer["verb"] == "choose_target"
                        and "Gray Ogre" in action.description
                    )
                )
            )
        )
        submit_response(env, decision, action.reference)
    raise AssertionError("Ty Lee did not resolve within the declared prefix")


def shared_effect_root(activations: int) -> managym.Env:
    """Resolve Drill activations through Commands in a bounded constructed root.

    Six activations fit v5; seven reproduce the 140-fact capacity failure.
    This is a new directed fixture, not a replay of the pilot's retained seed.
    """
    if not 1 <= activations <= 7:
        raise ValueError(
            "the shared-effect fixture admits one through seven activations"
        )
    env = priority_root(
        (),
        ("The Fire Nation Drill",) + ("Mountain",) * 7,
        opponent=("Gray Ogre",) * 10,
    )
    for activation in range(activations):
        decision = capture_text_decision(env)
        source = DecisionInput.model_validate_json(decision.capture_json)
        assert source.observation.decision is not None
        action = next(
            action
            for offer, action in zip(
                source.observation.decision.offers, decision.actions, strict=True
            )
            if offer["verb"] == "activate_ability"
            and "The Fire Nation Drill" in action.description
        )
        submit_response(env, decision, action.reference)
        for _ in range(8):
            obs = env.observation_for_player(env.current_agent_index())
            records = sum(
                len(p.effects)
                for p in (*obs.agent_permanents, *obs.opponent_permanents)
            )
            if records == 10 * (activation + 1):
                break
            decision = capture_text_decision(env)
            source = DecisionInput.model_validate_json(decision.capture_json)
            assert source.observation.decision is not None
            action = next(
                action
                for offer, action in zip(
                    source.observation.decision.offers, decision.actions, strict=True
                )
                if offer["verb"] in ("pay_cost", "pass_priority")
            )
            submit_response(env, decision, action.reference)
        else:
            raise AssertionError("Drill did not resolve within the declared prefix")
    return env


def kellan_upgrade_root(hand: tuple[str, ...] = ()) -> managym.Env:
    """Resolve Kellan's first upgrade through Commands in a bounded fixture."""
    env = priority_root(hand, ("Kellan, Planar Trailblazer",) + ("Mountain",) * 5)
    for step in range(16):
        obs = env.observation_for_player(env.current_agent_index())
        if any(r.effect.program for r in obs.shared_effects or []):
            return env
        decision = capture_text_decision(env)
        actions = obs.action_space.actions
        index = next(
            i
            for i, action in enumerate(actions)
            if (step == 0 and action.ability_index == 0)
            or (
                step > 0
                and (
                    "pass" in decision.actions[i].description.lower()
                    or "pay" in decision.actions[i].description.lower()
                )
            )
        )
        submit_response(env, decision, decision.actions[index].reference)
    raise AssertionError("Kellan upgrade did not resolve")


def kellan_trigger_root(hand: tuple[str, ...] = ()) -> managym.Env:
    """Reach the upgraded Kellan's captured combat trigger on the stack."""
    env = kellan_upgrade_root(hand)
    for _ in range(24):
        obs = env.observation_for_player(env.current_agent_index())
        if any(frame.program for frame in obs.continuation):
            return env
        decision = capture_text_decision(env)
        action = next(
            a
            for a in decision.actions
            if a.description.startswith(("Pass priority", "Attack with"))
        )
        submit_response(env, decision, action.reference)
    raise AssertionError("Kellan's combat trigger did not reach the stack")


@dataclass
class Measurement:
    fixture: str
    arm: StateEncoder
    parameters: int
    architecture: str
    forward_seconds: float
    backward_seconds: float
    valid_facts: int
    shared_fact_payloads: int
    relation_rows: int
    frame_headers: int
    input_bytes: int
    input_sha256: str
    error: str | None = None


def prove(out: Path) -> int:
    """Retain all attempts in a fresh directory; any failed fixture fails the proof."""
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    space = ObservationSpace(ObservationConfig(policy_input_version=5))
    measurements: list[Measurement] = []
    root = Path(__file__).resolve().parents[2]
    # Bind current Python and Rust sources, including untracked prototype files.
    # Committed source is the retrievable archive; dirty runs stay exploratory.
    sources = [
        *root.joinpath("manabot").rglob("*.py"),
        root / "experiments/state_representation.py",
    ]
    for directory in ("managym/src", "managym/state/src", "managym/cards/src"):
        sources.extend(root.joinpath(directory).rglob("*.rs"))
    sources.extend(
        root / name for name in ("pyproject.toml", "uv.lock", "managym/Cargo.lock")
    )
    atomic_json(
        out / "identity.json",
        {
            "schema": "state-representation-proof-1",
            "seed": 73200,
            "world": managym.WORLD_VERSION,
            "source_commit": source_commit(),
            "source_dirty": source_dirty(),
            "sources": {
                str(path.relative_to(root)): file_sha256(path)
                for path in sorted(sources)
            },
            "native_sha256": file_sha256(managym._managym.__file__),
            "observation": space.encoder.config.model_dump(mode="json"),
            "scope": "same-v5-input engineering proof; no learning or capacity claim",
        },
    )
    fixtures = native_fixtures()
    del fixtures["priority"]
    del fixtures["optional_payment"]
    fixtures["shared_effects"] = shared_effect_root(6)
    fixtures["source_effect"] = source_effect_root()
    for name, env in fixtures.items():
        actor = env.current_agent_index()
        assert actor is not None
        encoded: dict[str, np.ndarray] | None = None
        for viewer in (actor, 1 - actor):
            obs = env.observation_for_player(viewer)
            native = env.encode_observation(obs, 5)
            reference = space.encode(obs)
            for key in native:
                np.testing.assert_array_equal(
                    native[key], reference[key], err_msg=f"{name}: {key}"
                )
            np.savez(out / f"{name}-viewer-{viewer}.npz", **native)
            if viewer == actor:
                encoded = native
        assert encoded is not None
        facts = encoded["facts"][encoded["facts"][:, 11] == 1]
        rows = encoded["relations"][encoded["relations"][:, 31] == 1]
        inputs = {key: torch.from_numpy(value)[None] for key, value in encoded.items()}
        for arm in ARMS:
            torch.manual_seed(73200)
            agent = ActorCritic(
                space,
                ModelConfig(
                    hidden_dim=64,
                    attention_layers=2,
                    state_encoder=arm,
                    policy_film=True,
                ),
            )
            architecture = architecture_receipt(agent)
            measurement = Measurement(
                fixture=name,
                arm=arm,
                parameters=architecture.parameters.total,
                architecture=architecture.identity,
                forward_seconds=0,
                backward_seconds=0,
                valid_facts=len(facts),
                shared_fact_payloads=len(np.unique(facts[:, 1:11], axis=0)),
                relation_rows=len(rows),
                frame_headers=int((rows[:, 0] == 1).sum()),
                input_bytes=sum(value.nbytes for value in encoded.values()),
                input_sha256=file_sha256(out / f"{name}-viewer-{actor}.npz"),
            )
            try:
                start = time.perf_counter()
                logits, value = agent(inputs)
                measurement.forward_seconds = time.perf_counter() - start
                legal = logits[inputs["actions_valid"].bool()]
                if not torch.isfinite(legal).all() or not torch.isfinite(value).all():
                    raise ValueError("nonfinite model output")
                start = time.perf_counter()
                (legal.square().mean() + value.square().mean()).backward()
                measurement.backward_seconds = time.perf_counter() - start
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in agent.parameters()
                ):
                    raise ValueError("nonfinite model gradient")
            except Exception as exc:
                measurement.error = f"{type(exc).__name__}: {exc}"
            measurements.append(measurement)
            atomic_json(
                out / "measurements.json", [asdict(item) for item in measurements]
            )
    failures = sum(item.error is not None for item in measurements)
    print(f"{len(measurements)} fixture/arm attempts; {failures} failures; {out}")
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return prove(parser.parse_args().out)


if __name__ == "__main__":
    raise SystemExit(main())
