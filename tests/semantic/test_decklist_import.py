"""Decklist files through managym's Python surface into manabot setups.

Parsing, name resolution and admission diagnostics are tested natively in
`managym/src/semantic/decklist.rs`; these tests cover file reading, retained
records in configurations, the sim CLI and checkpoint execution.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from manabot import cli
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    RuntimeConfig,
    SimulationConfig,
)
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.sim.player import ModelPlayer
from manabot.sim.sim import load_player, simulate_models
import managym
from managym import DeckImportError, Decklist, ImportedMatch

PACK = "tla-pool-v1"
HERO = "# Novel development deck\nMain\n10 Forest\n10 Mountain\n2 Avatar Enthusiasts\nSideboard\n1 Island\n"
VILLAIN = "20 Island\n"


def _import_files(hero: Path, villain: Path) -> ImportedMatch:
    return managym.import_match(Decklist.read(hero), Decklist.read(villain), PACK)


def test_decklist_novel_setup_resolves_same_identity_from_files(tmp_path: Path) -> None:
    hero, villain = tmp_path / "hero.txt", tmp_path / "villain.txt"
    hero.write_bytes(HERO.replace("\n", "\r\n").encode())
    villain.write_text(VILLAIN)
    imported = _import_files(hero, villain)
    assert imported.hero.decklist.source == HERO.replace("\n", "\r\n")
    assert imported.hero.decklist.filename == "hero.txt"
    assert "definitions" not in imported.to_dict()["pool"]
    config = MatchConfig.from_import(imported)
    restored = MatchConfig.model_validate_json(config.model_dump_json())
    assert restored.imported_decks == imported
    assert restored.imported_decks.setup_id == imported.setup_id
    assert imported.hero.deck.main == {
        "Forest": 10,
        "Mountain": 10,
        "Avatar Enthusiasts": 2,
    }
    assert imported.hero.deck.sideboard == {"Island": 1}
    assert all(
        entry.card_def_id is not None for entry in imported.hero.decklist.entries
    )
    assert not any(
        managym.authored_deck_setup("elemental-world-v2", key).decklist
        == imported.hero.deck.main
        for key in managym.authored_deck_keys("elemental-world-v2")
    )
    env = managym.Env(seed=7, skip_trivial=False)
    env.reset(Match(config).to_rust())
    assert (
        env.content_pack_manifest()["content_digest"] == imported.pool["content_digest"]
    )
    assert (
        env.content_pack_manifest()["compiled_semantics"]
        == imported.pool["compiled_semantics"]
    )


def test_decklist_errors_carry_each_failing_seat_with_line_issues() -> None:
    source = "10 Forest\n1 Definitely Not A Card\nSideboard\n1 Accumulate Wisdom\n"
    with pytest.raises(DeckImportError) as caught:
        managym.import_match(Decklist(source), Decklist(VILLAIN), PACK)
    assert list(caught.value.decklists) == ["hero"]
    decklist = caught.value.decklists["hero"]
    assert decklist.source == source
    assert {(issue.line, issue.code) for issue in decklist.issues} == {
        (2, "unknown"),
        (4, "unsupported"),
    }
    assert "hero deck.txt:2" in str(caught.value)


@pytest.mark.parametrize("with_files", [False, True])
def test_sim_cli_preserves_overrides_with_optional_deck_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_files: bool
) -> None:
    configs: list[SimulationConfig] = []

    def capture(config: SimulationConfig, runtime: RuntimeConfig) -> None:
        configs.append(config)

    monkeypatch.setattr(cli, "run_simulation", capture)
    args = ["sim", "--set", "sim.max_steps=4", "--set", "sim.hero=random"]
    if with_files:
        hero, villain = tmp_path / "hero.txt", tmp_path / "villain.txt"
        hero.write_text(HERO)
        villain.write_text(VILLAIN)
        args.extend(
            ["--hero-deck-file", str(hero), "--villain-deck-file", str(villain)]
        )
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert len(configs) == 1
    assert configs[0].max_steps == 4
    assert configs[0].hero == "random"
    imported = configs[0].match.imported_decks
    if with_files:
        assert imported is not None
        assert imported.hero.decklist.source == HERO
        assert imported.setup_id in result.output
    else:
        assert imported is None


def test_retained_import_rejects_tampered_sources_and_setups() -> None:
    imported = managym.import_match(Decklist(HERO), Decklist(VILLAIN), PACK)
    retained = imported.to_dict()
    retained["hero"]["decklist"]["source"] = "10 Island"
    with pytest.raises(ValueError, match="source"):
        ImportedMatch.from_dict(retained).validate_current()
    retained = imported.to_dict()
    retained["pool"]["content_digest"] = "stale"
    with pytest.raises(ValueError, match="identity"):
        MatchConfig.model_validate(
            {
                **MatchConfig.from_import(imported).model_dump(),
                "imported_decks": retained,
            }
        )
    edited = MatchConfig.from_import(imported).model_dump()
    edited["villain_deck"] = {"Island": 19}
    with pytest.raises(ValueError, match="admitted decklist"):
        MatchConfig.model_validate(edited)
    swapped = Match(MatchConfig.from_import(imported)).swapped().config
    assert swapped.imported_decks == imported.swapped()
    assert swapped.hero_deck == imported.villain.deck.main


def _untrained(path: Path, pack: str = PACK, **binding: object) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace()
    agent = ActorCritic(
        space, ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=pack)
    )
    save_checkpoint(agent, space, path, **binding)


class _RecordingModel(ModelPlayer):
    """Fail the proof on a model error the simulator would replace randomly."""

    def __init__(self, path: Path) -> None:
        player = load_player(str(path))
        assert isinstance(player, ModelPlayer)
        super().__init__(player.name, player.agent)
        self.actions = 0
        self.errors: list[Exception] = []

    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        try:
            action = super().get_action(obs)
        except Exception as error:
            self.errors.append(error)
            raise
        self.actions += 1
        return action


def test_decklist_pool_checkpoint_simulates_imported_files(tmp_path: Path) -> None:
    hero, villain = tmp_path / "hero.txt", tmp_path / "villain.txt"
    hero.write_text(HERO)
    villain.write_text(VILLAIN)
    match = MatchConfig.from_import(_import_files(hero, villain))
    pool = tmp_path / "pool.pt"
    _untrained(pool, content_pool=PACK)
    model = _RecordingModel(pool)
    config = SimulationConfig(
        hero=str(pool), num_games=1, num_threads=1, max_steps=40, match=match
    )
    stats = simulate_models(model, load_player("random"), config)
    assert len(stats.games) == 1
    assert model.actions > 0 and not model.errors


@pytest.mark.parametrize(
    ("pack", "decks"),
    [
        ("ur-lessons-vs-gw-allies", ("ur_lessons", "gw_allies")),
        ("elemental-world-v2", ("elemental_bg", "elemental_gr")),
    ],
)
def test_decklist_setup_bound_checkpoint_rejects_imported_files(
    tmp_path: Path, pack: str, decks: tuple[str, str]
) -> None:
    imported = managym.import_match(Decklist(HERO), Decklist(VILLAIN), PACK)
    match = MatchConfig.from_import(imported)
    path = tmp_path / "authored.pt"
    _untrained(
        path,
        pack,
        player_configs=[managym.authored_deck_setup(pack, key) for key in decks],
    )
    agent, _ = load_checkpoint(str(path))
    players = Match(match).to_rust()
    with pytest.raises(ValueError, match="outside checkpoint"):
        validate_agent_setup(agent, players)
    config = SimulationConfig(hero=str(path), num_games=1, match=match)
    with pytest.raises(ValueError, match="outside checkpoint"):
        simulate_models(load_player(str(path)), load_player("random"), config)
