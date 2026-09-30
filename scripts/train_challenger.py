"""Bounded local Allies/Lessons search distillation using the existing trainer.

Run with uv run scripts/train_challenger.py --out .runs/challenger-N.
The parent records the recipe and world identity before launching a
resource-bounded worker. Teacher games use the authored setup, including
sideboards, so Learn offers its complete choice. A run completes only after
admission finds no omitted legal choice and the exported checkpoint plays
both deck assignments through the play server's configured-opponent path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from zipfile import ZipFile

import psutil

from managym import WORLD_VERSION

PACK_KEY = "ur-lessons-vs-gw-allies"
DECKS = ("ur_lessons", "gw_allies")
CAP_HIT_RATE_LIMIT = 0.01
DEMO_MOVE_LIMIT = 10_000


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def world_identity(observation: dict) -> dict:
    """Identify the rules, content, Lesson pool and tensor ABI a run binds to."""
    from manabot.env import ObservationSpace
    from manabot.env.observation import ActionEnum, ActionSpaceEnum
    from manabot.infra.hypers import ObservationSpaceHypers
    import managym

    setup = {}
    for deck in DECKS:
        authored = managym.authored_deck_setup(PACK_KEY, deck)
        setup[deck] = {
            "decklist": dict(authored.decklist),
            "sideboard": dict(authored.sideboard),
        }
    env = managym.Env(seed=0, skip_trivial=True)
    env.reset(
        [
            managym.PlayerConfig(deck, entry["decklist"], entry["sideboard"])
            for deck, entry in setup.items()
        ]
    )
    manifest = env.content_pack_manifest()
    encoder = ObservationSpace(ObservationSpaceHypers(**observation)).encoder
    abi = {
        "hypers": encoder.hypers.model_dump(),
        "shapes": {key: list(shape) for key, shape in encoder.shapes.items()},
        "action_types": {member.name: int(member) for member in ActionEnum},
        "decision_kinds": {member.name: int(member) for member in ActionSpaceEnum},
    }
    lesson_pool = {deck: entry["sideboard"] for deck, entry in setup.items()}
    native = Path(managym._managym.__file__)
    return {
        "version": WORLD_VERSION,
        "dims": {
            "player": encoder.player_dim,
            "card": encoder.card_dim,
            "permanent": encoder.permanent_dim,
            "action_types": encoder.num_actions,
        },
        "pack_key": PACK_KEY,
        "content_manifest": manifest,
        "content_manifest_sha256": canonical_digest(manifest),
        "rules_runtime": {"extension": native.name, "sha256": digest(native)},
        "setup": setup,
        "setup_sha256": canonical_digest(setup),
        "lesson_pool": lesson_pool,
        "lesson_pool_sha256": canonical_digest(lesson_pool),
        "observation_action_abi": abi,
        "observation_action_abi_sha256": canonical_digest(abi),
    }


def admit(dataset: dict, summaries: list[dict], recipe: dict) -> dict:
    """Report whether every engine-legal choice reached the training rows."""
    import numpy as np

    from manabot.env.observation import ActionEnum, ActionSpaceEnum

    world = recipe["world"]
    legal = dataset["num_legal"].astype(np.int64)
    encoded = dataset["num_valid"].astype(np.int64)
    kinds = dataset["decision_kind"]

    def rows(mask) -> dict:
        return {
            "decisions": int(mask.sum()),
            "legal_choices": int(legal[mask].sum()),
            "omitted_legal_choices": int((legal[mask] - encoded[mask]).sum()),
            "decisions_differing_from_engine": int((legal != encoded)[mask].sum()),
            "max_legal_choices": int(legal[mask].max()) if mask.any() else 0,
        }

    everything = np.ones(len(legal), dtype=bool)
    learn = kinds == int(ActionSpaceEnum.LEARN)
    take_lesson = dataset["actions"][:, :, int(ActionEnum.LEARN_TAKE_LESSON)] > 0
    simulations = sum(row["search"]["simulations"] for row in summaries)
    cap_hits = sum(row["search"]["cap_hits"] for row in summaries)
    report = {
        **rows(everything),
        "max_actions": recipe["observation"]["max_actions"],
        "learn": {
            **rows(learn),
            "decisions_offering_a_lesson": int(take_lesson.any(axis=1).sum()),
            "lesson_offers": int(take_lesson.sum()),
        },
        "by_decision_kind": {
            member.name: rows(kinds == int(member))
            for member in ActionSpaceEnum
            if (kinds == int(member)).any()
        },
        "search": {
            "simulations": simulations,
            "cap_hits": cap_hits,
            "cap_hit_rate": cap_hits / max(1, simulations),
        },
        "inputs": "acting viewer's observation only; no hidden-truth columns",
    }
    other_setup = other_identity = False
    for index, row in enumerate(summaries):
        provenance = row["provenance"]
        for seat, deck in zip(("hero", "villain"), deck_order(index)):
            played = {
                "decklist": provenance["match"][f"{seat}_deck"],
                "sideboard": provenance["match"][f"{seat}_sideboard"],
            }
            other_setup |= played != world["setup"][deck]
        other_identity |= (
            provenance["content_manifest"] != world["content_manifest"]
            or provenance["observation_hypers"]
            != world["observation_action_abi"]["hypers"]
        )
    failures = []
    if report["decisions_differing_from_engine"]:
        failures.append("encoded legal choices differ from the engine's offers")
    if not report["learn"]["decisions_offering_a_lesson"]:
        failures.append("no Learn decision offered a Lesson")
    if report["search"]["cap_hit_rate"] > CAP_HIT_RATE_LIMIT:
        failures.append("search continuation cap exceeded its admitted rate")
    if other_setup:
        failures.append("a teacher game used a different deck or sideboard setup")
    if other_identity:
        failures.append("a shard binds a different content or observation identity")
    report["failures"] = failures
    report["admitted"] = not failures
    return report


def deck_order(index: int) -> tuple[str, str]:
    """Alternate which deck takes the first turn across independent games."""
    return DECKS if index % 2 == 0 else DECKS[::-1]


def teacher_game(recipe: dict, index: int, shard: Path) -> dict:
    """Play one authored teacher game into its own shard; return its summary."""
    from manabot.infra.hypers import MatchHypers, ObservationSpaceHypers
    from manabot.sim.distill import generate_selfplay_shard

    return generate_selfplay_shard(
        num_games=1,
        teacher_spec=recipe["teacher"],
        seed=recipe["seed"],
        game_offset=index,
        out_path=shard,
        match_hypers=MatchHypers.authored(PACK_KEY, *deck_order(index)),
        observation_hypers=ObservationSpaceHypers(**recipe["observation"]),
    )


def train(out: Path, recipe: dict) -> None:
    import torch

    from manabot.infra.hypers import AgentHypers, ObservationSpaceHypers
    from manabot.sim.distill import load_shards, save_bc_checkpoint
    from manabot.sim.flat_mc import load_checkpoint_agent
    from manabot.sim.search_supervised import train_search_supervised

    torch.set_num_threads(1)
    world = recipe["world"]
    if world_identity(recipe["observation"]) != world:
        raise RuntimeError("Worker runtime differs from the recorded world")
    observation = ObservationSpaceHypers(**recipe["observation"])
    shards = []
    summaries = []
    phases = {}
    phase_start = time.monotonic()
    # Each completed game survives interruption. Both seats contribute labels.
    for index in range(recipe["games"]):
        shard = out / f"shard_{index:05d}.npz"
        summary = teacher_game(recipe, index, shard)
        summaries.append(summary)
        write_json(out / "games.json", summaries)
        if summary["winners"] == [-1]:
            raise RuntimeError("Teacher game did not reach an authoritative winner")
        shards.append(shard)
        print(f"Teacher games: {index + 1}/{recipe['games']}", flush=True)
    phases["teacher_wall_seconds"] = time.monotonic() - phase_start
    phases["teacher_decisions"] = sum(row["decisions"] for row in summaries)
    phases["teacher_decisions_per_second"] = (
        phases["teacher_decisions"] / phases["teacher_wall_seconds"]
    )
    write_json(out / "phases.json", phases)
    phase_start = time.monotonic()
    dataset = load_shards(shards)
    admission = admit(dataset, summaries, recipe)
    write_json(out / "admission.json", admission)
    if not admission["admitted"]:
        raise RuntimeError(f"Teacher data rejected: {admission['failures']}")
    agent, space, initial, history = train_search_supervised(
        dataset,
        policy_target_kind=recipe["policy_target"],
        value_target_kind=recipe["value_target"],
        policy_weight=recipe["policy_weight"],
        value_weight=recipe["value_weight"],
        agent_hypers=AgentHypers(**recipe["agent"]),
        observation_hypers=observation,
        epochs=recipe["epochs"],
        seed=recipe["seed"],
        batch_size=recipe["batch_size"],
        lr=recipe["learning_rate"],
        val_fraction=recipe["validation_fraction"],
        log=True,
    )
    write_json(
        out / "training.json",
        {
            "initial_validation": asdict(initial),
            "epochs": [asdict(row) for row in history],
        },
    )
    phases["training_wall_seconds"] = time.monotonic() - phase_start
    write_json(out / "phases.json", phases)
    phase_start = time.monotonic()
    checkpoint = out / "candidate.pt"
    save_bc_checkpoint(agent, space, checkpoint, extra={"recipe": recipe})
    loaded, _ = load_checkpoint_agent(str(checkpoint))
    sample = {
        key: torch.as_tensor(value[:2])
        for key, value in dataset.items()
        if key in space.shapes
    }
    agent.eval()
    with torch.no_grad():
        if not torch.equal(agent(sample)[0], loaded(sample)[0]):
            raise RuntimeError("Saved checkpoint does not reproduce trained logits")
    write_json(
        out / "candidate.json",
        {
            "name": f"Local challenger · seed {recipe['seed']}",
            "producer": "manabot",
            "bot_id": "etude:local-challenger",
            "checkpoint": checkpoint.name,
            "sha256": digest(checkpoint),
            "inference": {"deterministic": False},
            "content_manifest": world["content_manifest"],
            "decks": {deck: world["setup"][deck]["decklist"] for deck in DECKS},
            # recipe.json holds the bodies these digests stand for.
            "world": {
                key: value
                for key, value in world.items()
                if key not in ("content_manifest", "setup", "observation_action_abi")
            },
            "recipe_sha256": digest(out / "recipe.json"),
            "data_sha256": {path.name: digest(path) for path in shards},
            "admission_sha256": digest(out / "admission.json"),
            "observation": observation.model_dump(),
            "selection": "final epoch of one fixed run; strength unmeasured",
        },
    )
    phases["export_reload_wall_seconds"] = time.monotonic() - phase_start
    write_json(out / "phases.json", phases)


def demo_check(out: Path, recipe: dict) -> None:
    """Play the exported candidate through the play server in both assignments."""
    import random

    from etude.opponent import configured_opponent
    from etude.server import GameSession

    opponent = configured_opponent()
    if opponent is None:
        raise RuntimeError("The play server found no configured opponent")
    games = []
    for index, (hero_deck, villain_deck) in enumerate((DECKS, DECKS[::-1])):
        seed = recipe["seed"] + index
        moves = random.Random(seed)
        session = GameSession()
        message = session.new_game(
            {
                "villain_type": "checkpoint",
                "opponent_sha256": opponent.sha256,
                "hero_deck": hero_deck,
                "villain_deck": villain_deck,
                "seed": seed,
            }
        )
        hero_moves = 0
        while message["type"] != "game_over":
            if hero_moves >= DEMO_MOVE_LIMIT:
                raise RuntimeError("Demo game exceeded its move limit")
            choice = moves.choice(message["actions"])
            message = session.hero_action(choice["index"])
            hero_moves += 1
        games.append(
            {
                "attempt": session.trace_id,
                "seed": seed,
                "random_hero_deck": hero_deck,
                "candidate_deck": villain_deck,
                "hero_moves": hero_moves,
                "winner_seat": message["winner"],
            }
        )
        session.close("demo_check")
        write_json(out / "demo_check.json", {"complete": False, "games": games})
    write_json(
        out / "demo_check.json",
        {
            "complete": True,
            "opponent": opponent.identity(),
            "path": "etude.opponent.configured_opponent -> etude.server.GameSession",
            "games": games,
        },
    )


def supervise(
    arguments: list[str],
    log,
    seconds: float,
    rss_bytes: int,
    environment: dict | None = None,
) -> tuple[int | None, str | None, int]:
    """Run one bounded child of this script; return (exit code, reason, peak RSS)."""
    start = time.monotonic()
    peak = 0
    reason = None
    # sys.executable is the uv-managed interpreter running this script.
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        stdout=log,
        stderr=log,
        start_new_session=True,
        env=environment,
    )
    try:
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                peak = max(
                    peak,
                    sum(
                        p.memory_info().rss
                        for p in [parent, *parent.children(recursive=True)]
                    ),
                )
            except psutil.Error:
                pass
            if time.monotonic() - start > seconds:
                reason = "wall_limit"
                break
            if peak > rss_bytes:
                reason = "rss_limit"
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return process.returncode, reason, peak


def main() -> int:
    operator_start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--rss-mib", type=int, default=4096)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--demo-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker:
        train(out, json.loads((out / "recipe.json").read_text()))
        return 0
    if args.demo_check:
        demo_check(out, json.loads((out / "recipe.json").read_text()))
        return 0
    if (
        args.games < 2
        or args.epochs < 1
        or not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.rss_mib <= 0
    ):
        parser.error("need >=2 games, >=1 epoch, and finite positive resource limits")
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [
        *root.glob("manabot/**/*.py"),
        *root.glob("etude/**/*.py"),
        *root.glob("managym/*.py"),
        *root.glob("managym/src/**/*.rs"),
        *root.glob("managym/_managym*.so"),
        root / "uv.lock",
        root / "pyproject.toml",
        root / "managym/Cargo.lock",
        Path(__file__),
    ]
    source_bytes = {
        str(path.relative_to(root)): path.read_bytes() for path in sorted(sources)
    }
    observation = {
        "max_actions": 128,
        "max_cards_per_player": 96,
        "max_permanents_per_player": 64,
    }
    recipe = {
        "method": "determinized-PUCT-64 visit distillation",
        "games": args.games,
        "epochs": args.epochs,
        "seed": args.seed,
        "teacher": {
            "kind": "determinized_puct",
            "sims": 64,
            "worlds": 4,
            "max_steps": 2000,
        },
        "policy_target": "visit_distribution",
        "value_target": "terminal_outcome",
        "policy_weight": 1.0,
        "value_weight": 0.0,
        "agent": {"hidden_dim": 64, "num_attention_heads": 4, "attention_on": True},
        "batch_size": 128,
        "learning_rate": 0.001,
        "validation_fraction": 0.1,
        "wall_seconds_limit": args.seconds,
        "rss_mib_limit": args.rss_mib,
        "device": "cpu",
        "threads": 1,
        "selection": "final epoch",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
        },
        "runtime": {
            "python": sys.version,
            "torch": version("torch"),
            "numpy": version("numpy"),
        },
        "reproducibility": "Fixed inputs and procedure; no byte-identical checkpoint promise",
        "observation": observation,
        "world": world_identity(observation),
        "sources": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in source_bytes.items()
        },
    }
    with ZipFile(out / "sources.zip", "w") as archive:
        for name, data in source_bytes.items():
            archive.writestr(name, data)
    del source_bytes
    recipe["source_archive_sha256"] = digest(out / "sources.zip")
    write_json(out / "recipe.json", recipe)
    start = time.monotonic()
    rss_bytes = args.rss_mib * 1024**2
    with (out / "run.log").open("w") as log:
        exit_code, reason, peak = supervise(
            ["--worker", "--out", str(out)], log, args.seconds, rss_bytes
        )

    demo_seconds = None

    def receipt(status: str) -> dict:
        return {
            "status": status,
            "ending_reason": reason,
            "exit_code": exit_code,
            "wall_seconds": time.monotonic() - start,
            "operator_wall_seconds": time.monotonic() - operator_start,
            "demo_check_wall_seconds": demo_seconds,
            "rss_scope": "sampled worker process tree; excludes supervising process",
            "peak_rss_bytes": peak,
            "actual_new_cloud_spend_usd": 0,
            "estimated_incremental_electricity_usd": None,
        }

    if exit_code == 0 and reason is None:
        # The play server admits only a complete run, so the load check reads
        # a provisional receipt; any failure below replaces it.
        write_json(out / "receipt.json", receipt("complete"))
        demo_start = time.monotonic()
        try:
            with (out / "demo_check.log").open("w") as log:
                exit_code, reason, demo_peak = supervise(
                    ["--demo-check", "--out", str(out)],
                    log,
                    max(1.0, args.seconds - (time.monotonic() - start)),
                    rss_bytes,
                    {
                        **os.environ,
                        "ETUDE_PLAY_CANDIDATE": str(out / "candidate.json"),
                        "ETUDE_TRACES_DIR": str(out / "demo-check"),
                        "ETUDE_PLAY_RECORD_ORIGIN": "automated_validation",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )
        except BaseException:
            reason = "demo_check"
            write_json(out / "receipt.json", receipt("failed"))
            raise
        demo_seconds = time.monotonic() - demo_start
        peak = max(peak, demo_peak)
        if exit_code != 0 and reason is None:
            reason = "demo_check"
    final = receipt("complete" if exit_code == 0 and reason is None else "failed")
    write_json(out / "receipt.json", final)
    print(json.dumps(final, indent=2))
    print(f"Retained run: {out}")
    return 0 if final["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
