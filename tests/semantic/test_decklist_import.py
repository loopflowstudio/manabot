"""File imports preserve source while native admission owns playable setups."""

from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from manabot import cli
from manabot.decklist import (
    ContentPool,
    DeckDocument,
    DeckImportError,
    ImportedMatch,
    admit_document,
    import_files,
    import_match,
    parse_decklist,
)
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RuntimeConfig, SimulationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.sim.player import ModelPlayer
from manabot.sim.sim import load_player, simulate_models
import managym

PACK = "tla-pool-v1"
HERO = "# Novel development deck\nMain\n10 Forest\n10 Mountain\n2 Avatar Enthusiasts\nSideboard\n1 Island\n"
VILLAIN = "20 Island\n"


def test_decklist_dialect_keeps_original_lines_and_reports_bad_counts() -> None:
    source = "\ufeffDeck\r\n# comment\r\n2 Forest\r\nSideboard:\r\n0 Island\r\n-1 Mountain\r\n2x Forest\r\n99999999999999999999 Island\r\n"
    document = parse_decklist(source, "input.txt")
    assert document.source == source
    assert document.filename == "input.txt"
    assert [(entry.line, entry.name, entry.count) for entry in document.entries] == [
        (3, "Forest", 2)
    ]
    assert [(issue.line, issue.code) for issue in document.issues] == [
        (5, "count"),
        (6, "count"),
        (7, "syntax"),
        (8, "count"),
    ]


def test_decklist_novel_setup_resolves_same_identity_from_files(tmp_path: Path) -> None:
    hero, villain = tmp_path / "hero.txt", tmp_path / "villain.txt"
    hero.write_bytes(HERO.replace("\n", "\r\n").encode())
    villain.write_text(VILLAIN)
    imported = import_files(hero, villain, PACK)
    assert imported.hero.document.source == HERO.replace("\n", "\r\n")
    assert "definitions" not in imported.model_dump()["pool"]
    restored = ImportedMatch.model_validate_json(imported.model_dump_json())
    restored.validate_current()
    assert restored.setup_id == imported.setup_id
    reformatted = import_match(
        parse_decklist(
            "10 mountain\n5 Forest\n5 forest\n2 Avatar Enthusiasts\nSideboard\n1 Island"
        ),
        parse_decklist(VILLAIN),
        PACK,
    )
    assert imported.setup_id == reformatted.setup_id
    assert imported.hero.deck.main == {
        "Forest": 10,
        "Mountain": 10,
        "Avatar Enthusiasts": 2,
    }
    assert imported.hero.deck.sideboard == {"Island": 1}
    assert all(
        entry.card_def_id is not None for entry in imported.hero.document.entries
    )
    assert not any(
        managym.authored_deck_setup("elemental-world-v2", key).decklist
        == imported.hero.deck.main
        for key in managym.authored_deck_keys("elemental-world-v2")
    )
    env = managym.Env(seed=7, skip_trivial=False)
    env.reset(Match(imported.match_config()).to_rust())
    assert env.content_pack_manifest()["content_digest"] == imported.pool.content_digest
    assert (
        env.content_pack_manifest()["compiled_semantics"]
        == imported.pool.compiled_semantics.model_dump()
    )


def test_decklist_reports_unknown_unsupported_and_retains_source() -> None:
    pool = ContentPool.load(PACK)
    unsupported = next(
        row.registry_name
        for row in pool.definitions
        if row.eligibility == "unsupported"
    )
    source = f"10 Forest\n1 Definitely Not A Card\nSideboard\n1 {unsupported}\n2 {unsupported}\nMain\n1 {unsupported}\n"
    with pytest.raises(DeckImportError) as caught:
        import_match(parse_decklist(source), parse_decklist(VILLAIN), PACK)
    document = caught.value.documents["hero"]
    assert document.source == source
    assert {(issue.line, issue.code) for issue in document.issues} >= {
        (2, "unknown"),
        (4, "unsupported"),
        (5, "unsupported"),
        (7, "unsupported"),
    }


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
        assert imported.hero.document.source == HERO
        assert imported.setup_id in result.output
    else:
        assert imported is None


def test_decklist_ambiguous_names_fail_without_substitution() -> None:
    pool = ContentPool.load(PACK)
    forest = next(row for row in pool.definitions if row.registry_name == "Forest")
    ambiguous = pool.model_copy(
        update={
            "definitions": (
                *pool.definitions,
                forest.model_copy(update={"registry_name": "FOREST"}),
            )
        }
    )
    result = admit_document(parse_decklist("10 Forest"), ambiguous, "development-v1")
    assert isinstance(result, DeckDocument)
    assert any(issue.code == "ambiguous" and issue.line == 1 for issue in result.issues)
    assert result.source == "10 Forest"


@pytest.mark.parametrize(
    "source", ["Sideboard\n1 Island", "1 Forest\nSideboard\n0 Island", "1001 Forest"]
)
def test_decklist_rejects_construction_errors(source: str) -> None:
    with pytest.raises(DeckImportError):
        import_match(parse_decklist(source), parse_decklist(VILLAIN), PACK)


def test_decklist_rejects_unknown_format_and_retained_source_tampering() -> None:
    with pytest.raises(DeckImportError, match="ruleset"):
        import_match(parse_decklist(HERO), parse_decklist(VILLAIN), PACK, "standard")
    imported = import_match(parse_decklist(HERO), parse_decklist(VILLAIN), PACK)
    retained = imported.model_dump()
    retained["hero"]["document"]["source"] = "10 Island"
    with pytest.raises(ValueError, match="source"):
        ImportedMatch.model_validate(retained).validate_current()
    retained = imported.model_dump()
    retained["pool"]["content_digest"] = "stale"
    with pytest.raises(ValueError, match="identity"):
        ImportedMatch.model_validate(retained).validate_current()


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
    match = import_files(hero, villain, PACK).match_config()
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
    imported = import_match(parse_decklist(HERO), parse_decklist(VILLAIN), PACK)
    path = tmp_path / "authored.pt"
    _untrained(
        path,
        pack,
        player_configs=[managym.authored_deck_setup(pack, key) for key in decks],
    )
    agent, _ = load_checkpoint(str(path))
    players = Match(imported.match_config()).to_rust()
    with pytest.raises(ValueError, match="outside checkpoint"):
        validate_agent_setup(agent, players)
    config = SimulationConfig(
        hero=str(path), num_games=1, match=imported.match_config()
    )
    with pytest.raises(ValueError, match="outside checkpoint"):
        simulate_models(load_player(str(path)), load_player("random"), config)


@pytest.mark.parametrize(
    "name", ["Avatar Aang", "Avatar Aang // Aang, Master of Elements"]
)
def test_decklist_double_faced_identity_is_known_but_unsupported(name: str) -> None:
    with pytest.raises(DeckImportError) as caught:
        import_match(
            parse_decklist(f"10 Forest\n1 {name}"), parse_decklist(VILLAIN), PACK
        )
    assert any(
        issue.code == "unsupported" and issue.line == 2
        for issue in caught.value.documents["hero"].issues
    )


def test_decklist_reverse_face_is_not_a_separate_card() -> None:
    with pytest.raises(DeckImportError) as caught:
        import_match(
            parse_decklist("10 Forest\n1 Aang, Master of Elements"),
            parse_decklist(VILLAIN),
            PACK,
        )
    assert any(
        issue.code == "face" and issue.line == 2
        for issue in caught.value.documents["hero"].issues
    )
