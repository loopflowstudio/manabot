"""The checkpoint catalog lists trained policies and names world mismatches."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from manabot.env import ObservationSpace
<<<<<<< HEAD
from manabot.infra.hypers import AgentSpec
from manabot.model import Agent
=======
from manabot.infra.config import ModelConfig
from manabot.model import ActorCritic
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from manabot.model.catalog import (
    NotAPolicyCheckpoint,
    discover_checkpoints,
    inspect_checkpoint,
)
<<<<<<< HEAD
from manabot.model.world import WORLD
from manabot.sim.distill import save_bc_checkpoint
=======
from manabot.model.checkpoint import save_checkpoint
from manabot.model.world import WORLD
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
import managym

PACK = "ur-lessons-vs-gw-allies"


def _write_checkpoint(path: Path) -> None:
<<<<<<< HEAD
    agent = Agent(
        ObservationSpace(),
        AgentSpec(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK),
    )
    save_bc_checkpoint(
=======
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK),
    )
    save_checkpoint(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        agent,
        agent.observation_space,
        path,
        player_configs=[
            managym.authored_deck_setup(PACK, deck)
            for deck in ("ur_lessons", "gw_allies")
        ],
    )


def _rewrite(path: Path, target: Path, **binding: object) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["world_binding"].update(binding)
    torch.save(checkpoint, target)


def test_current_checkpoint_is_admissible_with_its_identity(tmp_path: Path) -> None:
    path = tmp_path / "run" / "step_0.pt"
    _write_checkpoint(path)

    summary = inspect_checkpoint(path)

    assert summary.compatible
    assert summary.world == WORLD
    assert summary.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary.model == "width 8, 1 attention layer × 2 heads"
    assert summary.parameters > 0
<<<<<<< HEAD
    assert summary.producer == "supervised distillation"
=======
    assert summary.producer == "unknown"
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    assert summary.label == "run/step_0"
    assert len(summary.setups) == 2


def test_mismatches_are_named(tmp_path: Path) -> None:
    current = tmp_path / "current.pt"
    _write_checkpoint(current)
    _rewrite(current, tmp_path / "old-world.pt", world="w5")
    _rewrite(
        current,
        tmp_path / "old-content.pt",
        content_manifest={"content_digest": "0123456789abcdef"},
    )
    unbound = torch.load(current, map_location="cpu", weights_only=False)
    del unbound["world_binding"]
    torch.save(unbound, tmp_path / "unbound.pt")

    assert (
        inspect_checkpoint(tmp_path / "old-world.pt").incompatibility
        == f"trained in w5, engine is {WORLD}"
    )
    content = inspect_checkpoint(tmp_path / "old-content.pt").incompatibility
    assert content is not None and "trained on content 0123456789ab" in content
    assert (
        inspect_checkpoint(tmp_path / "unbound.pt").incompatibility
        == f"trained without a world binding; engine is {WORLD}"
    )


def test_non_policy_files_are_refused_without_rereading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    optimizer = tmp_path / "optimizer.pt"
    torch.save({"state": {}}, optimizer)
    with pytest.raises(NotAPolicyCheckpoint):
        inspect_checkpoint(optimizer)

    def no_reload(*args: object, **kwargs: object) -> None:
        raise AssertionError("an unchanged refusal is read again")

    monkeypatch.setattr(torch, "load", no_reload)
    with pytest.raises(NotAPolicyCheckpoint, match="holds no policy"):
        inspect_checkpoint(optimizer)


def test_discovery_reads_runs_paths_and_tournaments(tmp_path: Path) -> None:
    runs = tmp_path / ".runs"
    _write_checkpoint(runs / "self-play" / "step_1.pt")
    torch.save({"state": {}}, runs / "self-play" / "optimizer.pt")
    models = tmp_path / "models"
    models.mkdir()
    _rewrite(runs / "self-play" / "step_1.pt", models / "entrant.pt", world="w5")
    entrant_sha = hashlib.sha256((models / "entrant.pt").read_bytes()).hexdigest()
    tournament = models / "grid.json"
    tournament.write_text(
        json.dumps(
            {
                "entrants": [
                    {"id": "scripted-greedy", "label": "greedy", "origin": "code"},
                    {
                        "id": "entrant",
                        "label": "width 8 entrant",
                        "origin": "grid run final raw",
                        "file": "entrant.pt",
                        "sha256": entrant_sha,
                    },
                    {
                        "id": "absent",
                        "label": "absent entrant",
                        "origin": "elsewhere",
                        "file": "absent.pt",
                        "sha256": "0" * 64,
                    },
                ]
            }
        )
    )

    found, missing = discover_checkpoints([tournament], runs_dir=runs)

    by_label = {summary.label: summary for summary in found}
    assert set(by_label) == {"width 8 entrant", "self-play/step_1"}
    assert by_label["self-play/step_1"].source == "runs"
    entrant = by_label["width 8 entrant"]
    assert entrant.source == "tournament"
    assert entrant.producer == "Tournament grid: grid run final raw"
    assert entrant.incompatibility == f"trained in w5, engine is {WORLD}"
    assert [entry.label for entry in missing] == ["absent entrant"]
