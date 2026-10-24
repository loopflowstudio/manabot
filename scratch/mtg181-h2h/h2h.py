"""Seed 18101 consecutive-checkpoint head-to-head on the 10x10 deck grid.

Newer checkpoint (x axis deck) plays the older one (y axis deck), N games per
cell, model seat drawn per game from one RNG. Reuses MTG-181's grid legs and
arena play_cell; local CPU only.

    cd <mtg-181 checkout>; uv run --extra artifacts python <this> run OUT
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, os.getcwd())

from experiments.elemental_grid import DECKS, GAME_SECONDS, MAX_COMMANDS, PACK, Group, _legs  # noqa: E402

JOB = "mtg181-elemental-two-seat-18101-r2"
SEED = 18101
PAIRS = ((200, 0), (400, 200), (600, 400), (600, 200))
GAMES_PER_CELL = 10
RNG_SEED = 18_190_101
DEAL_BASE = 1_829_000_000
WORKERS = 5


def checkpoints(cache: Path) -> dict[int, tuple[str, str]]:
    from manabot.verify.learnability.elemental_remote import observe

    found = observe(JOB, cache)["checkpoints"]
    return {c["coordinates"]["updates"]: (c["path"], c["sha256"]) for c in found}


def schedule() -> list[tuple[str, str, int, list[int]]]:
    rng = random.Random(RNG_SEED)
    out: dict[tuple[str, str, int], list[int]] = {}
    for x, new in enumerate(DECKS):
        for y, old in enumerate(DECKS):
            for i in range(GAMES_PER_CELL):
                seat = 0 if rng.random() < 0.5 else 1
                deal = DEAL_BASE + (x * len(DECKS) + y) * 1_000 + i
                out.setdefault((new, old, seat), []).append(deal)
    return [(n, o, s, d) for (n, o, s), d in out.items()]


def worker(spec: dict) -> list[dict]:
    import torch

    from manabot.arena.match import play_cell
    from manabot.arena.matchup import AuthoredMatchup
    from manabot.arena.models import ArenaKey, PlayerRegistration, canonical_sha256
    from manabot.infra.provenance import runtime_fingerprints
    from manabot.model.checkpoint import load_checkpoint
    import managym

    torch.set_num_threads(1)
    group = Group(model_deck=spec["new_deck"], heuristic_deck=spec["old_deck"],
                  model_seat=spec["seat"], deals=tuple(spec["deals"]))
    role_a, role_b, leg = _legs(group)
    matchup = AuthoredMatchup(pack_key=PACK, role_a=role_a, role_b=role_b)
    fp = runtime_fingerprints(match_config=matchup.match())
    common = dict(
        world=managym.WORLD_VERSION, content_suite=matchup.content_suite,
        information_boundary="acting-viewer",
        observation_abi_sha256=fp["observation_abi_sha256"],
        action_abi_sha256=fp["action_abi_sha256"],
        matchup_sha256=canonical_sha256(matchup.match().model_dump()),
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )

    def reg(pid: str, path: str, sha: str, role: str) -> PlayerRegistration:
        model, _ = load_checkpoint(Path(path))
        return PlayerRegistration(
            **common, player_id=pid, display_name=pid, role=role,
            compute_class_id="policy-cpu-one-thread-one-pass", runner_kind="checkpoint",
            player_spec={"kind": "checkpoint", "device": "cpu", "batch_size": 1,
                         "deterministic": False, "allow_deck_repetition": True},
            checkpoint_sha256=sha, checkpoint_bytes=Path(path).stat().st_size,
            parameter_count=sum(p.numel() for p in model.parameters()),
            training_seed=SEED, artifact_id=f"mtg181/{sha}",
        )

    new = reg("new", spec["new_path"], spec["new_sha"], "challenger")
    old = reg("old", spec["old_path"], spec["old_sha"], "anchor")
    key = ArenaKey(
        world=managym.WORLD_VERSION, content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer", arena_version="mtg181-h2h-18101-v1",
        rating_model_version="unrated", rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(old.model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    rows, _, _ = play_cell(
        key=key, player_a=new, player_b=old, deal_seeds=group.deals, out_dir=out,
        checkpoint_paths={"new": spec["new_path"], "old": spec["old_path"]},
        game_seconds=GAME_SECONDS, max_commands=MAX_COMMANDS, authored_matchup=matchup,
        matchup_mode="cross" if leg < 4 else "mirrors", legs=(leg,),
    )
    result = []
    for row in rows:
        seat = row["player_a_seat"]
        decks = row["seat_decks"]
        if decks[seat] != spec["new_deck"] or decks[1 - seat] != spec["old_deck"] or seat != spec["seat"]:
            raise ValueError("leg did not assign the scheduled decks/seat")
        result.append({
            "new_deck": spec["new_deck"], "old_deck": spec["old_deck"], "new_seat": seat,
            "deal": row["deal_seed"], "score": row["score_a"],
            "valid": bool(row["terminated"] and not row["truncated"] and row["failure"] is None and row["replay_passed"]),
            "failure": None if row["failure"] is None else str(row["failure"]),
        })
    return result


def run(out: Path) -> None:
    cps = checkpoints(Path(".runs/mtg181/follow/cache") / JOB)
    jobs = []
    for new_u, old_u in PAIRS:
        if new_u not in cps or old_u not in cps:
            print(f"missing checkpoint for {new_u} vs {old_u}; have {sorted(cps)}", flush=True)
            continue
        for new_deck, old_deck, seat, deals in schedule():
            target = out / f"{new_u}v{old_u}" / f"{new_deck}--{old_deck}--seat{seat}"
            if (target / "games.json").exists() or (target / "failed.json").exists():
                continue
            jobs.append((target, {
                "new_deck": new_deck, "old_deck": old_deck, "seat": seat, "deals": deals,
                "new_path": cps[new_u][0], "new_sha": cps[new_u][1],
                "old_path": cps[old_u][0], "old_sha": cps[old_u][1],
            }))

    def one(item: tuple[Path, dict]) -> None:
        target, spec = item
        target.mkdir(parents=True, exist_ok=True)
        spec = {**spec, "output": str(target / f"attempt-{os.getpid()}-{len(os.listdir(target))}")}
        p = subprocess.run([sys.executable, __file__, "worker"], input=json.dumps(spec), text=True,
                           capture_output=True, timeout=3600,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        if p.returncode == 0:
            (target / "games.json").write_text(p.stdout)
        else:
            (target / "failed.json").write_text(json.dumps({"error": p.stderr[-4000:]}))

    print(f"{len(jobs)} groups pending", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, _ in enumerate(pool.map(one, jobs), 1):
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} groups", flush=True)


def summary(out: Path) -> None:
    import math
    for pair in sorted(out.iterdir(), key=lambda p: int(p.name.split("v")[0])):
        rows = [r for f in pair.glob("*/games.json") for r in json.loads(f.read_text())]
        failed = len(list(pair.glob("*/failed.json")))
        v = [r for r in rows if r["valid"]]
        if not v:
            print(pair.name, "no games", "failed groups", failed); continue
        n, p = len(v), sum(r["score"] for r in v) / len(v)
        z = 1.96
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        print(f"{pair.name}: newer scores {p:.3f} [{c-h:.3f}, {c+h:.3f}] over {n} games; invalid {len(rows)-n}; failed groups {failed}")


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        print(json.dumps(worker(json.loads(sys.stdin.read()))))
    elif sys.argv[1] == "run":
        run(Path(sys.argv[2]))
    else:
        summary(Path(sys.argv[2]))
