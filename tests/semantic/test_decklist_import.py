"""Decklist files through managym's Python surface into manabot setups.

Parsing, name resolution and admission diagnostics are tested natively in
`managym/src/cards/decklist.rs`; these tests cover file reading, retained
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
from managym import DeckImport, DeckImportError, Decklist

PACK = managym.WORLD_SETUP
HERO = "# Novel development deck\nMain\n10 Forest\n10 Mountain\n2 Avatar Enthusiasts\nSideboard\n1 Island\n"
VILLAIN = "20 Island\n"


def _import_match(hero: Decklist, villain: Decklist) -> tuple[DeckImport, DeckImport]:
    first, second = managym.admit_decklists([("hero", hero), ("villain", villain)])
    return first, second


def _import_files(hero: Path, villain: Path) -> tuple[DeckImport, DeckImport]:
    return _import_match(Decklist.read(hero), Decklist.read(villain))


def _imported_match() -> MatchConfig:
    return MatchConfig.from_import(*_import_match(Decklist(HERO), Decklist(VILLAIN)))


def test_decklist_novel_setup_resolves_same_identity_from_files(tmp_path: Path) -> None:
    hero, villain = tmp_path / "hero.txt", tmp_path / "villain.txt"
    hero.write_bytes(HERO.replace("\n", "\r\n").encode())
    villain.write_text(VILLAIN)
    imported, villain_import = _import_files(hero, villain)
    assert imported.decklist.source == HERO.replace("\n", "\r\n")
    assert imported.decklist.filename == "hero.txt"
    config = MatchConfig.from_import(imported, villain_import)
    dumped = config.model_dump(mode="json")
    assert dumped["hero_source"] == imported.decklist.to_dict()
    assert dumped["hero_deck"] == imported.deck.main
    assert "deck" not in dumped["hero_source"]
    restored = MatchConfig.model_validate_json(config.model_dump_json())
    assert restored == config
    assert restored.hero_source == imported.decklist
    assert imported.deck.main == {
        "Forest": 10,
        "Mountain": 10,
        "Avatar Enthusiasts": 2,
    }
    assert imported.deck.sideboard == {"Island": 1}
    assert all(entry.card_def_id is not None for entry in imported.decklist.entries)
    assert not any(
        managym.authored_deck_setup("elemental-world-v2", key).decklist
        == imported.deck.main
        for key in managym.authored_deck_keys("elemental-world-v2")
    )
    env = managym.Env(seed=7)
    env.reset(Match(config).to_rust())
    world = managym.world_identity()
    assert env.content_pack_manifest()["content_digest"] == world["card_digest"]
    assert env.content_pack_manifest()["world_version"] == world["world_version"]


def test_decklist_errors_carry_each_failing_seat_with_line_issues() -> None:
    source = "10 Forest\n1 Definitely Not A Card\nSideboard\n1 Accumulate Wisdom\n"
    with pytest.raises(DeckImportError) as caught:
        _import_match(Decklist(source), Decklist(VILLAIN))
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
    match = configs[0].match
    if with_files:
        assert match.hero_source is not None
        assert match.hero_source.source == HERO
        hero_deck = managym.Deck(match.hero_deck, match.hero_sideboard)
        assert hero_deck.identity in result.output
    else:
        assert match.hero_source is None and match.villain_source is None


def test_retained_sources_reject_tampering_and_follow_seat_swaps() -> None:
    config = _imported_match()
    tampered = config.model_dump()
    tampered["hero_source"]["source"] = "10 Island"
    with pytest.raises(ValueError, match="retained source"):
        MatchConfig.model_validate(tampered)
    edited = config.model_dump()
    edited["villain_deck"] = {"Island": 19}
    with pytest.raises(ValueError, match="retained source"):
        MatchConfig.model_validate(edited)
    authored = config.model_dump()
    authored["world"] = "ur-lessons-vs-gw-allies"
    with pytest.raises(ValueError, match="plays in the world"):
        MatchConfig.model_validate(authored)
    swapped = Match(config).swapped().config
    assert swapped.hero_source == config.villain_source
    assert swapped.villain_source == config.hero_source
    assert swapped.hero_deck == config.villain_deck


def _untrained(path: Path, pack: str = PACK, **binding: object) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace()
    agent = ActorCritic(
        space, ModelConfig(hidden_dim=8, num_attention_heads=2, world=pack)
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


def test_decklist_world_checkpoint_simulates_imported_files(tmp_path: Path) -> None:
    hero, villain = tmp_path / "hero.txt", tmp_path / "villain.txt"
    hero.write_text(HERO)
    villain.write_text(VILLAIN)
    match = MatchConfig.from_import(*_import_files(hero, villain))
    world = tmp_path / "world.pt"
    _untrained(world, player_configs=None)
    model = _RecordingModel(world)
    config = SimulationConfig(
        hero=str(world), num_games=1, num_threads=1, max_steps=40, match=match
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
    match = _imported_match()
    path = tmp_path / "authored.pt"
    _untrained(
        path,
        pack,
        player_configs=[managym.authored_deck_setup(pack, key) for key in decks],
    )
    agent, _ = load_checkpoint(str(path))
    players = Match(match).to_rust()
    rejected = "outside checkpoint|setup differs"
    with pytest.raises(ValueError, match=rejected):
        validate_agent_setup(agent, players)
    config = SimulationConfig(hero=str(path), num_games=1, match=match)
    with pytest.raises(ValueError, match=rejected):
        simulate_models(load_player(str(path)), load_player("random"), config)
