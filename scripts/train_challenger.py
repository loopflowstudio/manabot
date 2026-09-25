"""Bounded local Allies/Lessons search distillation using the existing trainer.

Run with uv run scripts/train_challenger.py --out .runs/challenger-N.
The parent records the recipe before launching a resource-bounded worker.
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


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(out: Path, recipe: dict) -> None:
    import torch

    from manabot.infra.hypers import MatchHypers, ObservationSpaceHypers
    from manabot.sim.distill import (
        generate_selfplay_shard,
        load_shards,
        save_bc_checkpoint,
        train_bc,
    )
    from manabot.sim.flat_mc import load_checkpoint_agent
    from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK

    torch.set_num_threads(1)
    observation = ObservationSpaceHypers(**recipe["observation"])
    shards = []
    summaries = []
    phases = {}
    phase_start = time.monotonic()
    # Each completed game survives interruption. Both seats contribute labels;
    # alternate the decks' starting seat across independent games.
    for index in range(recipe["games"]):
        decks = [UR_LESSONS_DECK, GW_ALLIES_DECK]
        if index % 2:
            decks.reverse()
        shard = out / f"shard_{index:05d}.npz"
        summary = generate_selfplay_shard(
            num_games=1,
            teacher_spec=recipe["teacher"],
            seed=recipe["seed"],
            game_offset=index,
            out_path=shard,
            match_hypers=MatchHypers(hero_deck=decks[0], villain_deck=decks[1]),
            observation_hypers=observation,
        )
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
    agent, space, history = train_bc(
        dataset,
        observation_hypers=observation,
        epochs=recipe["epochs"],
        seed=recipe["seed"],
        batch_size=recipe["batch_size"],
        lr=recipe["learning_rate"],
        val_fraction=recipe["validation_fraction"],
        log=True,
    )
    write_json(out / "training.json", [asdict(row) for row in history])
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
            "content_manifest": summaries[0]["provenance"]["content_manifest"],
            "decks": {"ur_lessons": UR_LESSONS_DECK, "gw_allies": GW_ALLIES_DECK},
            "recipe_sha256": digest(out / "recipe.json"),
            "data_sha256": {path.name: digest(path) for path in shards},
            "observation": observation.model_dump(),
            "selection": "final epoch of one fixed run; strength unmeasured",
        },
    )
    phases["export_reload_wall_seconds"] = time.monotonic() - phase_start
    write_json(out / "phases.json", phases)


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
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker:
        train(out, json.loads((out / "recipe.json").read_text()))
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
    recipe = {
        "method": "flat-search-64 behavior cloning",
        "games": args.games,
        "epochs": args.epochs,
        "seed": args.seed,
        "sims": 64,
        "teacher": {"kind": "search", "sims": 64, "max_steps": 2000},
        "batch_size": 64,
        "learning_rate": 0.001,
        "validation_fraction": 0.2,
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
        "observation": {
            "max_actions": 128,
            "max_cards_per_player": 96,
            "max_permanents_per_player": 64,
        },
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
    peak = 0
    reason = None
    with (out / "run.log").open("w") as log:
        # sys.executable is the uv-managed interpreter running this script.
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--out",
                str(out),
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
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
                if time.monotonic() - start > args.seconds:
                    reason = "wall_limit"
                    break
                if peak > args.rss_mib * 1024**2:
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
    receipt = {
        "status": "complete"
        if process.returncode == 0 and reason is None
        else "failed",
        "ending_reason": reason,
        "exit_code": process.returncode,
        "wall_seconds": time.monotonic() - start,
        "operator_wall_seconds": time.monotonic() - operator_start,
        "rss_scope": "sampled worker process tree; excludes supervising process",
        "peak_rss_bytes": peak,
        "actual_new_cloud_spend_usd": 0,
        "estimated_incremental_electricity_usd": None,
    }
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    print(f"Retained run: {out}")
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
