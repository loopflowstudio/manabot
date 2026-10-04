"""Registered Learn cohort over the existing live, headless and replay owners."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import managym

from .authored_match_parity import (
    ROOT,
    _canonical_bytes,
    _checkpoint,
    _equal,
    _run_engine,
    _sha256,
    _source_manifest,
)
from .authored_match_receipt import _ledger, play_fixed_authored_match
from .replay_index import CanonicalReplayV1, project_replay
from .server import GameSession
from .trace import GameConfig


def source_identity():
    files = {row["path"]: row for row in _source_manifest()["files"]}
    for folder in (
        "etude",
        "manabot/model",
        "manabot/env",
        "manabot/semantic",
        "manabot/infra",
        "manabot/sim",
        "managym",
    ):
        for path in (ROOT / folder).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            files[relative] = {"path": relative, "sha256": _sha256(path.read_bytes())}
    for relative in (
        "scripts/verify-learn-lesson",
        "content/semantic/v1/learning_schema.json",
        "tests/model/test_world.py",
        "tests/model/test_semantic_cards.py",
        "tests/semantic/test_learn_contract.py",
        "tests/etude/test_learn_setup.py",
        "tests/etude/test_learn_evidence.py",
        "pyproject.toml",
        "uv.lock",
    ):
        files[relative] = {
            "path": relative,
            "sha256": _sha256((ROOT / relative).read_bytes()),
        }
    rows = [files[key] for key in sorted(files)]
    return {"files": rows, "sha256": _sha256(_canonical_bytes(rows))}


def registration():
    engine = managym.Env(seed=0)
    configs = [
        managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
        for key in ("ur_lessons", "gw_allies")
    ]
    engine.reset(configs)
    return {
        "world": managym.WORLD_VERSION,
        "engine_extension_sha256": _sha256(
            Path(managym._managym.__file__).read_bytes()
        ),
        "source": source_identity(),
        "manifest": engine.content_pack_manifest(),
        "setups": [{"deck": p.decklist, "sideboard": p.sideboard} for p in configs],
        "policy": "DeterministicServerOfferPolicy; independent per-seat Random(deal_seed)",
        "max_commands": 10000,
        "attempts": [
            {"seed": seed, "reverse": reverse}
            for seed in range(8)
            for reverse in (False, True)
        ],
        "claim": "automated same-Command parity; no human completion, strength or runtime-budget claim",
    }


def verify_attempt(result):
    if result["status"] != "passed":
        raise ValueError(f"attempt failed: {result.get('error')}")
    config = GameConfig(**result["config"])
    keys = ["ur_lessons", "gw_allies"]
    if result["reverse"]:
        keys.reverse()
    expected = [
        managym.authored_deck_setup("ur-lessons-vs-gw-allies", key) for key in keys
    ]
    _equal("registration", 0, "seed", result["seed"], config.seed)
    _equal(
        "registration", 0, "setups",
        [(p.decklist, p.sideboard) for p in expected],
        [(p.decklist, p.sideboard) for p in config.to_rust()],
    )
    tape = result["tape"]
    if not 0 < len(tape) <= 10000:
        raise ValueError("attempt violates registered Command cap")
    replay = CanonicalReplayV1.model_validate(result["replay"])
    _equal(
        "persisted",
        0,
        "commands",
        [row["command"] for row in tape],
        [row.command.model_dump(mode="json") for row in replay.decisions],
    )
    for viewer in (0, 1):
        projection = project_replay(replay, viewer)
        for row in projection.decisions:
            if row.frame.projection.opponent.hand:
                raise ValueError("persisted replay exposed the opponent hand")
    replay_tape = [
        {**row, "command": saved.command.model_dump(mode="json")}
        for row, saved in zip(tape, replay.decisions, strict=True)
    ]
    for surface, commands in (("headless", tape), ("replay", replay_tape)):
        actual, _ = _run_engine(surface, commands, config=config)
        _equal(
            surface, 0, "all_checkpoints", result["checkpoints"], actual["checkpoints"]
        )


def attempt(seed, reverse):
    result = {"seed": seed, "reverse": reverse, "status": "failed", "tape": []}
    with TemporaryDirectory() as temporary:
        session = GameSession(
            Path(temporary),
            capture_authority_evidence=True,
            id_factory=lambda kind: f"learn-{seed}-{int(reverse)}-{kind}",
        )
        try:
            session, _ = play_fixed_authored_match(
                seed=seed, reverse=reverse, max_commands=10000, session=session
            )
            tape, _ = _ledger(session, {})
            result["tape"] = tape
            result["config"] = asdict(session.trace.config)
            result["replay"] = session.canonical_replay().model_dump(mode="json")
            result["checkpoints"] = [
                _checkpoint(session._study_roots[0], "live", 0, [])
            ]
            for ordinal, transition in enumerate(session.authority_transitions):
                root = (
                    session.env
                    if ordinal + 1 == len(tape)
                    else session._study_roots[ordinal + 1]
                )
                result["checkpoints"].append(
                    _checkpoint(root, "live", ordinal + 1, transition.semantic_events)
                )
            result["status"] = "passed"
            verify_attempt(result)
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            if not result["tape"]:
                result["partial_commands"] = [
                    row.command.model_dump(mode="json")
                    for row in session.canonical_decisions
                ]
        finally:
            session.close(end_reason="evidence_complete")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("register", "run", "verify"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    path = directory / "registration.json"
    if args.operation == "register":
        directory.mkdir(parents=True, exist_ok=True)
        with path.open("x") as output:
            json.dump(registration(), output, indent=2, sort_keys=True)
        return
    registered = json.loads(path.read_text())
    if registered != registration():
        raise ValueError(
            "registered source/world/setup differs; retain this cohort and register a new one"
        )
    results = []
    for row in registered["attempts"]:
        receipt = (
            directory / f"seed-{row['seed']}-reverse-{int(row['reverse'])}.json.gz"
        )
        if args.operation == "run":
            if receipt.exists():
                raise ValueError(f"refusing to overwrite registered attempt: {receipt}")
            result = attempt(**row)
            receipt.write_bytes(gzip.compress(_canonical_bytes(result), mtime=0))
        else:
            result = json.loads(gzip.decompress(receipt.read_bytes()))
            _equal("registration", 0, "attempt", row, {key: result[key] for key in row})
            verify_attempt(result)
        results.append(
            {
                **row,
                "status": result["status"],
                "error": result.get("error"),
                "commands": len(result["tape"]),
                "sha256": _sha256(receipt.read_bytes()),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    summary = {
        "registration_sha256": _sha256(path.read_bytes()),
        "attempts": results,
        "passed": sum(row["status"] == "passed" for row in results),
        "total": 16,
    }
    if args.operation == "run":
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    if summary["passed"] != 16:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
