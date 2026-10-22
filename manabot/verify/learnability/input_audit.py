"""Audit retained native choices for exact policy-input collisions.

No fitting occurs. Compare every legal action using its encoded action row and
role/type-aware focused object rows, including categorical definition joins.
Different public descriptors with identical fingerprints are candidates for
information loss. Field-removal controls measure the instrument's sensitivity;
they are not historical-runtime results. This does not prove injectivity of the
learned encoder or detect public relations missing from the descriptor itself.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import cast

from pydantic import BaseModel
import torch

from manabot.verify.action_cases import encode
from manabot.verify.choice_corpus import (
    ActionInput,
    Mutation as Mutation,
    action_inputs as _fingerprints,
)
import managym


class CaseReceipt(BaseModel):
    name: str
    family: int
    signatures: list[str]
    seed: int
    swapped: bool
    trace: list[int]
    digest: str


def _replay(case: CaseReceipt, expected: dict[str, torch.Tensor]) -> dict[str, object]:
    env = managym.Env(seed=case.seed, skip_trivial=False)
    configs = [
        managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
        for key in ("ur_lessons", "gw_allies")
    ]
    if case.swapped:
        configs.reverse()
    raw, _ = env.reset(configs)
    for index in case.trace:
        raw, _, done, truncated, _ = env.step(index)
        assert not done and not truncated, case.name
    assert env.state_digest() == case.digest, case.name
    actual = encode(env, raw)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    return {
        "case": case.name,
        "family": case.family,
        "steps": len(case.trace),
        "digest": case.digest,
        "passed": True,
    }


def audit(root: Path, *, replay: bool = False) -> dict[str, object]:
    torch.set_num_threads(1)
    if json.loads((root / "result.json").read_text())["world"] != managym.WORLD_VERSION:
        raise ValueError("replay/audit world does not match retained evidence")
    replayed: dict[int, dict[str, object]] = {}
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(root.glob("round-*/cases.json")):
        raw_cases = json.loads(path.read_text())
        if not isinstance(raw_cases, list):
            raise ValueError(f"expected case list: {path}")
        cases = [CaseReceipt.model_validate(value) for value in raw_cases]
        # This experiment's weights-only artifact has a fixed nested tensor map.
        loaded = torch.load(path.parent / "observations.pt", weights_only=True)
        if not isinstance(loaded, dict) or not all(
            isinstance(value, dict)
            and all(isinstance(t, torch.Tensor) for t in value.values())
            for value in loaded.values()
        ):
            raise ValueError("invalid observation artifact")
        observations = cast(dict[str, dict[str, torch.Tensor]], loaded)
        mutations: tuple[Mutation, ...] = (
            "clean",
            "without_declaration",
            "without_definition",
        )
        for case in cases:
            if replay and case.family not in replayed:
                replayed[case.family] = _replay(case, observations[case.name])
            for mutation in mutations:
                fingerprints = _fingerprints(
                    observations[case.name], len(case.signatures), mutation
                )
                groups: dict[ActionInput, list[int]] = defaultdict(list)
                for index, fingerprint in enumerate(fingerprints):
                    groups[fingerprint].append(index)
                collisions = [
                    group
                    for group in groups.values()
                    if len({case.signatures[i] for i in group}) > 1
                ]
                counts[f"{mutation}_states"] += 1
                counts[f"{mutation}_colliding_states"] += bool(collisions)
                counts[f"{mutation}_collision_classes"] += len(collisions)
                for group in collisions:
                    if len(examples[mutation]) < 12:
                        examples[mutation].append(
                            {"case": case.name, "family": case.family, "offers": group}
                        )
    if not counts:
        raise ValueError("no retained cases found")
    return {
        "method": "Exact action and role/type-aware focused object rows, including categorical joins; all retained states, including unscored final batches.",
        "replay": list(replayed.values()),
        "counts": dict(counts),
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay one retained native state per family",
    )
    args = parser.parse_args()
    result = audit(args.root, replay=args.replay)
    (args.root / "input-collisions.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
