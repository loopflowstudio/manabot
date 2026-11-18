"""Record and replay complete games with immutable policy fixtures."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory

import numpy as np
from pydantic import BaseModel
import torch

from manabot.backends.torch.bot import evaluate_observation
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.backends.torch.policy import PolicyInference
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.model.interface import input_columns
import managym

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/pinned_pilot"
PILOTS = {"v9": "pilot.pt", "v11": "pilot.safetensors"}
BEST_PLAYER = "MANABOT_U1952_CHECKPOINT"
RECORD = 'uv run python scripts/pinned_pilot.py record --reason "<why>"'


def pilot_path(name: str) -> Path | None:
    """A committed pilot, or the optional local best player when its variable is set."""
    if name in PILOTS:
        return FIXTURES / name / PILOTS[name]
    return Path(os.environ[BEST_PLAYER]) if BEST_PLAYER in os.environ else None


class Decision(BaseModel):
    offers: str
    action: int
    probabilities: list[float]


class Game(BaseModel):
    seed: int
    decks: tuple[dict[str, int], dict[str, int]]
    decisions: list[Decision]
    winner: int | None
    player_columns: list[str]


class Recording(BaseModel):
    checkpoint_sha256: str
    revision: str
    reason: str
    game: Game


def play(
    path: Path,
    seed: int,
    decks: tuple[dict[str, int], dict[str, int]],
    replay: Game | None = None,
) -> Game:
    """Follow the recorded indices only while their ordered legal offers match."""
    try:
        pilot, space = load_checkpoint(str(path))
    except ValueError as exc:
        raise ValueError(f"the loader refused {path.name}: {exc}") from exc
    width = pilot.column_widths.get("agent_player", space.encoder.player_dim)
    columns = input_columns(space)["agent_player"][:width]
    env = managym.Env(seed=seed)
    obs, _ = env.reset(
        [managym.PlayerConfig(str(i), deck) for i, deck in enumerate(decks)]
    )
    rng = np.random.default_rng(seed)
    decisions: list[Decision] = []
    with PolicyInference(pilot) as scope:
        while True:
            index = len(decisions)
            action_space = json.loads(obs.toJSON())["action_space"]
            for offer in action_space["actions"]:
                offer.pop("description", None)
            offers = canonical_sha256(
                [obs.agent.player_index, obs.action_space.focus, action_space]
            )
            probs, _ = evaluate_observation(scope, space, obs)
            if replay is not None:
                if (
                    index >= len(replay.decisions)
                    or offers != replay.decisions[index].offers
                ):
                    raise AssertionError(
                        f"decision {index}: the legal actions changed. {RECORD}"
                    )
                action = replay.decisions[index].action
            else:
                weights = np.asarray(probs)
                action = int(rng.choice(len(weights), p=weights / weights.sum()))
            decisions.append(
                Decision(offers=offers, action=action, probabilities=list(probs))
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if truncated:
                raise AssertionError(f"game truncated at decision {index}")
            if terminated:
                break
    return Game(
        seed=seed,
        decks=decks,
        decisions=decisions,
        winner=env.winner_index(),
        player_columns=columns,
    )


def verify(path: Path, recording: Recording) -> None:
    if file_sha256(path) != recording.checkpoint_sha256:
        raise AssertionError(f"checkpoint bytes differ for {path}")
    expected = recording.game
    current = play(path, expected.seed, expected.decks, expected)
    if (
        len(current.decisions) != len(expected.decisions)
        or current.winner != expected.winner
    ):
        raise AssertionError(
            f"game ended differently after {len(current.decisions)} decisions. {RECORD}"
        )
    if expected.player_columns != current.player_columns:
        raise AssertionError(
            f"the ordered player columns read by {path.name} changed. {RECORD}"
        )
    for index, (old, new) in enumerate(
        zip(expected.decisions, current.decisions, strict=True)
    ):
        before, after = np.asarray(old.probabilities), np.asarray(new.probabilities)
        rank_before = np.count_nonzero(before > before[old.action])
        rank_after = np.count_nonzero(after > after[old.action])
        if (
            not np.allclose(before, after, atol=1e-5, rtol=0)
            or rank_before != rank_after
        ):
            raise AssertionError(
                f"the game replays; policy changed at decision {index} "
                f"(maximum probability difference {np.max(np.abs(before - after)):.3g}, "
                f"chosen-action rank {rank_before + 1} -> {rank_after + 1}). {RECORD}"
            )


def record(reason: str) -> None:
    if not reason.strip():
        raise ValueError("recording a change requires a nonempty reason")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    updates: dict[Path, Recording] = {}
    for name in (*PILOTS, "u1952"):
        path = pilot_path(name)
        if path is None:
            continue
        target = FIXTURES / name / "game.json"
        previous = Recording.model_validate_json(target.read_bytes())
        digest = file_sha256(path)
        if digest != previous.checkpoint_sha256:
            raise ValueError(f"record cannot replace the pinned weights for {name}")
        seed, decks = previous.game.seed, previous.game.decks
        updates[target] = Recording(
            checkpoint_sha256=digest,
            revision=revision,
            reason=reason,
            game=play(path, seed, decks),
        )
    for target, recording in updates.items():
        atomic_json(target, recording.model_dump(mode="json"))


def names_at(revision: str) -> None:
    """Trace names at the writer's hash boundary, including generated serde paths."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    with TemporaryDirectory() as directory:
        archive = Path(directory) / "writer.tar"
        subprocess.run(
            [
                "git",
                "archive",
                "-o",
                str(archive),
                commit,
                "managym",
                "content/semantic/current",
            ],
            cwd=ROOT,
            check=True,
        )
        with tarfile.open(archive) as source:
            source.extractall(directory, filter="data")
        crate = Path(directory) / "managym"
        serializer = crate / "src/cards/tokens.rs"
        source_text = serializer.read_text()
        # These two functions are the writer ABI for the supported v9-v11 family.
        # Work only in the temporary archive so its checkout and imports stay intact.
        hooks = {
            "pub fn name_value(name: &str) -> i32 {": """
                let value = original_value(name);
                NAMES.with_borrow_mut(|names| {
                    if let Some(old) = names.insert(value, name.to_owned()) {
                        assert_eq!(old, name, "ambiguous token hash {value}");
                    }
                });
                value
            }
            thread_local! {
                static NAMES: std::cell::RefCell<std::collections::BTreeMap<i32, String>> =
                    Default::default();
            }
            pub fn writer_names() -> std::collections::BTreeMap<i32, String> {
                NAMES.with_borrow(std::clone::Clone::clone)
            }
            fn original_value(name: &str) -> i32 {
            """,
            "fn path_value(name: &str, variant: Option<&str>, closing: bool) -> i32 {": """
                let path = variant.map_or_else(|| name.to_owned(), |v| format!("{name}::{v}"));
                name_value(&format!("{}{path}", if closing { "/" } else { "" }));
            """,
        }
        for signature, probe in hooks.items():
            if source_text.count(signature) != 1:
                raise ValueError(
                    f"{revision} has an unsupported token writer: {signature}"
                )
            source_text = source_text.replace(signature, signature + probe)
        serializer.write_text(source_text)
        shutil.copyfile(
            Path(__file__).with_name("pilot_names.rs"),
            crate / "examples/pilot_names.rs",
        )
        subprocess.run(
            [
                "cargo",
                "run",
                "--locked",
                "--quiet",
                "--jobs",
                "2",
                "--example",
                "pilot_names",
            ],
            cwd=crate,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("record").add_argument("--reason", required=True)
    commands.add_parser("names").add_argument("--at", required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    if arguments.command == "names":
        names_at(arguments.at)
    else:
        record(arguments.reason)
