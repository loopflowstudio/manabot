"""Headless whole-game memory proof, with retained Commands and CPU receipts.

Constructed positions establish known public/private choices. Subsequent steps
use ordinary native Commands, with trivial decisions skipped. Four untrained
policies prove wiring and checkpoint identity; one ordinary update proves the
training path. These checks establish no playing strength or CUDA admission.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.semantic.text_observation import capture_text_decision, render_decision
import managym
from managym.opponent_memory import read_records


class DemoBackend(Protocol):
    def prove_policies(
        self,
        space: ObservationSpace,
        obs: dict[str, np.ndarray],
        out: Path,
        player_configs: list[managym.PlayerConfig],
    ) -> list[dict[str, object]]: ...

    def ordinary_update(self, out: Path) -> dict[str, object]: ...


@dataclass
class Proof:
    env: managym.Env
    replay: managym.Env
    out: Path
    space: ObservationSpace
    commands: int = 0

    def choose(self, verb: str, contains: str = "") -> None:
        decision = capture_text_decision(self.env)
        contract = SemanticDecisionContract.from_env(self.env)
        for offer, action in zip(contract.frame.offers, decision.actions, strict=True):
            if offer["verb"] == verb and contains in action.description:
                command = contract.command(offer["id"])
                contract.apply_command(self.env, command)
                SemanticDecisionContract.from_env(self.replay).apply_command(
                    self.replay, command
                )
                self.out.mkdir(parents=True, exist_ok=True)
                with (self.out / "commands.jsonl").open(
                    "w" if self.commands == 0 else "a"
                ) as stream:
                    stream.write(command.to_json() + "\n")
                self.commands += 1
                assert self.env.state_digest() == self.replay.state_digest()
                return
        raise AssertionError(f"missing {verb} {contains}: {decision.text[-1600:]}")

    def checkpoint(self, name: str) -> dict[str, int]:
        directory = self.out / name
        directory.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        before = self.env.state_digest()
        for viewer in range(2):
            obs = self.env.observation_for_player(viewer)
            source = obs.opponent_memory_json()
            assert (
                source
                == self.replay.observation_for_player(viewer).opponent_memory_json()
            )
            expected = self.space.encode(obs)
            native = self.space.encoder.allocate()
            self.env.encode_observation_into(obs, native)
            restored = self.space.encode(managym.Observation.from_json(obs.toJSON()))
            for key in (
                "opponent_records",
                "opponent_details",
                "actions",
                "action_focus",
            ):
                np.testing.assert_array_equal(native[key], expected[key])
                np.testing.assert_array_equal(restored[key], expected[key])
            counts[str(viewer)] = len(read_records(source))
            (directory / f"viewer-{viewer}.json").write_text(source + "\n")
        text = capture_text_decision(self.env)
        assert render_decision(text.capture_json).text == text.text
        (directory / "capture.json").write_text(text.capture_json + "\n")
        (directory / "prompt.txt").write_text(text.text)
        assert self.env.state_digest() == before
        return counts


def _configs(disclosure: Literal["open", "closed"]) -> list[managym.PlayerConfig]:
    deck = {
        name: 20
        for name in (
            "Island",
            "Mountain",
            "Gray Ogre",
            "Accumulate Wisdom",
            "Lightning Bolt",
        )
    }
    configs = [managym.PlayerConfig(name, deck) for name in ("a", "b")]
    for config in configs:
        config.disclosure = disclosure
    return configs


def _root(disclosure: Literal["open", "closed"]) -> managym.Env:
    env = managym.Env(seed=23)
    env.reset(_configs(disclosure))
    for _ in range(20):
        contract = SemanticDecisionContract.from_env(env)
        if int(env.observation_for_player(contract.frame.actor).turn.phase) == 1:
            break
        contract.apply(env, contract.frame.find_verb("pass_priority")["id"])
    else:
        raise AssertionError("main phase not reached")
    for seat in range(2):
        env.scenario_clear_hand(seat)
        for name in ("Accumulate Wisdom", "Lightning Bolt"):
            env.scenario_force_card_in_hand(seat, name)
        for name in ("Island", "Island", "Mountain", "Gray Ogre"):
            env.scenario_force_battlefield(seat, name, ready=True)
    env.scenario_refresh()
    return env


def _advance(proof: Proof, verb: str, contains: str = "") -> None:
    for _ in range(40):
        contract = SemanticDecisionContract.from_env(proof.env)
        if any(offer["verb"] == verb for offer in contract.frame.offers):
            proof.choose(verb, contains)
            return
        proof.choose("pass_priority")
    raise AssertionError(f"did not reach {verb}")


def _native_proof(out: Path, disclosure: Literal["open", "closed"]) -> Proof:
    proof = Proof(
        _root(disclosure),
        _root(disclosure),
        out,
        ObservationSpace(
            ObservationConfig(max_opponent_records=128, max_opponent_details=131072)
        ),
    )
    # A branch's public play must leave the parent's log and legal choices alone.
    parent = [
        proof.env.observation_for_player(v).opponent_memory_json() for v in range(2)
    ]
    branch = proof.env.clone_env()
    contract = SemanticDecisionContract.from_env(branch)
    branch_replay = proof.env.clone_env()
    command = contract.command(contract.frame.find_verb("cast")["id"])
    contract.apply_command(branch, command)
    SemanticDecisionContract.from_env(branch_replay).apply_command(
        branch_replay, command
    )
    Proof(branch, branch_replay, out / "branch", proof.space).checkpoint("cast")
    assert parent == [
        proof.env.observation_for_player(v).opponent_memory_json() for v in range(2)
    ]
    assert any(
        parent[v] != branch.observation_for_player(v).opponent_memory_json()
        for v in range(2)
    )
    proof.choose("cast", "Accumulate Wisdom")
    _advance(proof, "select_card", "Mountain")
    # Private selection is unavailable to the opponent even with open decklists.
    observer = 1 - proof.env.current_agent_index()
    retained = read_records(
        proof.env.observation_for_player(observer).opponent_memory_json()
    )
    assert any(record.choice.kind is None for record in retained)
    proof.checkpoint("private-selection")
    for _ in range(10):
        contract = SemanticDecisionContract.from_env(proof.env)
        if not any(o["verb"] == "select_card" for o in contract.frame.offers):
            break
        proof.choose("select_card")
    else:
        raise AssertionError("private selection did not finish")
    # The selected Mountain (seed 23) becomes public via the normal land action.
    proof.choose("play_land", "Mountain")
    after = read_records(
        proof.env.observation_for_player(observer).opponent_memory_json()
    )
    assert after[: len(retained)] == retained
    assert any(
        record.choice.kind == "PassPriority" and record.context.mana_sources
        for record in after
    )
    proof.checkpoint("public-play")
    _advance(proof, "declare_attacker", "Attack with")
    _advance(proof, "declare_blocker", "Do not block")
    counts = proof.checkpoint("declined-block")
    all_records = [
        record
        for v in range(2)
        for record in read_records(
            proof.env.observation_for_player(v).opponent_memory_json()
        )
    ]
    assert any(record.choice.declared is False for record in all_records)
    (out / "native.json").write_text(
        json.dumps(
            {
                "records_by_viewer": counts,
                "commands_replayed": proof.commands,
                "native_python_and_json_parity": True,
                "parent_immutable": True,
                "private_history_unchanged_after_public_play": True,
                "legality_and_focus_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
    return proof


def run_demo(out: Path, backend: DemoBackend) -> None:
    out.mkdir(parents=True, exist_ok=True)
    policies = {}
    for disclosure in ("open", "closed"):
        proof = _native_proof(out / disclosure, disclosure)
        actor = proof.env.current_agent_index()
        assert actor is not None
        obs = {
            key: value[None]
            for key, value in proof.space.encode(
                proof.env.observation_for_player(actor)
            ).items()
        }
        policies[disclosure] = backend.prove_policies(
            proof.space, obs, proof.out, _configs(disclosure)
        )
    summary = {
        "policies": policies,
        "ordinary_cpu_update": backend.ordinary_update(out),
        "scope": "software proof; no playing-strength or CUDA result",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    from manabot.backends.torch import verify_opponent_memory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".runs/opponent-memory-demo"))
    run_demo(parser.parse_args().out, verify_opponent_memory)


if __name__ == "__main__":
    main()
