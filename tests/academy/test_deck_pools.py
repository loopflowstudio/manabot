"""Direct managym.Deck launches, preserved custom deck names, and DeckPool dealing."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import server
from managym.academy.deck_pools import (
    DeckPool,
    NamedDeck,
    set_deck_pools,
)
from managym.academy.game_setup import continuation_config, setup_draft
from managym.academy.launch import AcademyLaunchError, serve
from managym.academy.watch import WatchRequest, create_watch, inspect_watch, step_watch
from tests.academy.socket_messages import settled_message


def _deck(
    main: dict[str, int], sideboard: dict[str, int] | None = None
) -> managym.Deck:
    return managym.Deck(main, sideboard or {})


def _sample_decks() -> tuple[NamedDeck, NamedDeck, NamedDeck]:
    ur = managym.authored_deck_setup("two_deck", "ur_lessons")
    gw = managym.authored_deck_setup("two_deck", "gw_allies")
    return (
        NamedDeck(
            "TLA Trophy · UR · aaaa1111", _deck(dict(ur.decklist), dict(ur.sideboard))
        ),
        NamedDeck(
            "TLA Trophy · GW · bbbb2222", _deck(dict(gw.decklist), dict(gw.sideboard))
        ),
        NamedDeck(
            "TLA Trophy · R · cccc3333",
            _deck({"Mountain": 20, "Gray Ogre": 20}, {"Lightning Bolt": 2}),
        ),
    )


@pytest.fixture
def site_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("Academy")
    monkeypatch.setenv("ACADEMY_FRONTEND_BUILD", str(site))
    return site


@pytest.fixture(autouse=True)
def _clean_pools() -> Iterator[None]:
    yield
    set_deck_pools(None)


def test_direct_managym_deck_and_named_deck_preserve_names_in_play_and_watch(
    site_dir: Path, tmp_path: Path
) -> None:
    del site_dir
    d1, d2, d3 = _sample_decks()
    with serve(port=0, traces_dir=tmp_path / "traces") as academy:
        # Bare managym.Deck, NamedDeck, and (name, Deck) tuple all work directly.
        launched = academy.new_game(
            deck=d1,
            opponent_deck=("Custom Ally List", d2.deck),
            opponent="random",
            seed=11,
            decklists="closed",
        )
        launch_id = parse_qs(urlparse(launched.url).query)["launch"][0]
        parsed = server._parse_game_config(server.LAUNCHES[launch_id].config)
        assert parsed.hero_deck_name == "TLA Trophy · UR · aaaa1111"
        assert parsed.villain_deck_name == "Custom Ally List"
        assert parsed.decklists == "closed"

        # Live Play preserves hero's custom deck name and hides closed opponent's deck name.
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/play") as ws:
                ws.send_json({"type": "new_game", "config": {"launch": launch_id}})
                msg = settled_message(ws)
                frame = msg["frame"]
                assert frame["deck_names"] == {
                    "hero": "TLA Trophy · UR · aaaa1111",
                    "villain": server.CLOSED_DECK_NAME,
                }
                server.end_game(msg["game"], "test")

        # Bare managym.Deck gets a deterministic Deck <hash> label rather than "imported".
        bare = academy.new_game(
            deck=d3.deck,
            opponent_deck=d2.deck,
            opponent="random",
            seed=12,
        )
        bare_id = parse_qs(urlparse(bare.url).query)["launch"][0]
        bare_parsed = server._parse_game_config(server.LAUNCHES[bare_id].config)
        assert bare_parsed.hero_deck_name == f"Deck {d3.deck.identity[:8]}"
        assert bare_parsed.villain_deck_name == f"Deck {d2.deck.identity[:8]}"

        # academy.watch() accepts direct Deck/NamedDeck, decklists, and seed via launch setup.
        watched = academy.watch(
            players=("random", "random"),
            decks=(d1, d2),
            seed=19,
            decklists="closed",
        )
        params = parse_qs(urlparse(watched.url).query)
        assert params["watch"] == ["1"]
        watch_launch = params["launch"][0]
        with TestClient(server.app) as client:
            setup = client.get(f"/api/launch/{watch_launch}").json()
        assert setup["decklists"] == "closed"
        assert setup["seed"] == 19
        assert setup["deck_import"]["hero"]["name"] == "TLA Trophy · UR · aaaa1111"
        assert setup["deck_import"]["villain"]["name"] == "TLA Trophy · GW · bbbb2222"


def test_deck_pool_samples_fresh_matchups_and_rematch_vs_deal_new_decks(
    site_dir: Path, tmp_path: Path
) -> None:
    del site_dir
    decks = _sample_decks()
    pool = DeckPool(
        id="tla_trophies",
        label="Random 17Lands TLA trophy",
        decks=decks,
        default_decklists="closed",
    )
    with serve(port=0, traces_dir=tmp_path / "traces", deck_pools=[pool]) as academy:
        with TestClient(server.app) as client:
            catalog = client.get("/api/deck-pools").json()
        assert catalog == {
            "pools": [
                {
                    "id": "tla_trophies",
                    "key": "pool:tla_trophies",
                    "label": "Random 17Lands TLA trophy",
                    "count": 3,
                    "default_decklists": "closed",
                }
            ]
        }

        # Seeded pool launch deterministically samples two distinct decks from the pool.
        seeded = academy.new_game(
            deck="pool:tla_trophies",
            opponent_deck="pool:tla_trophies",
            opponent="random",
            seed=42,
            decklists="closed",
        )
        launch_id = parse_qs(urlparse(seeded.url).query)["launch"][0]
        cfg1 = server._parse_game_config(server.LAUNCHES[launch_id].config)
        cfg2 = server._parse_game_config(server.LAUNCHES[launch_id].config)
        assert cfg1.hero_deck_name == cfg2.hero_deck_name
        assert cfg1.villain_deck_name == cfg2.villain_deck_name
        assert cfg1.hero_deck_name != cfg1.villain_deck_name
        assert cfg1.hero_deck_pool == "pool:tla_trophies"
        assert cfg1.villain_deck_pool == "pool:tla_trophies"

        # Rematch keeps the exact same sampled decks; "Deal new decks" re-samples from pool.
        draft = setup_draft(cfg1)
        assert draft["hero_deck_pool"] == "pool:tla_trophies"
        assert draft["deck_import"]["hero"]["name"] == cfg1.hero_deck_name
        rematch_cfg = server._parse_game_config(
            continuation_config(draft, {}, "rematch")
        )
        assert rematch_cfg.hero_deck_name == cfg1.hero_deck_name
        assert rematch_cfg.villain_deck_name == cfg1.villain_deck_name
        assert rematch_cfg.hero_deck_pool == "pool:tla_trophies"

        redeal_overrides = {
            "hero_deck": draft["hero_deck_pool"],
            "villain_deck": draft["villain_deck_pool"],
            "seed": 999,
        }
        redeal_raw = continuation_config(draft, redeal_overrides, "rematch")
        assert "deck_import" not in redeal_raw
        redeal_cfg = server._parse_game_config(redeal_raw)
        assert redeal_cfg.hero_deck_pool == "pool:tla_trophies"
        assert redeal_cfg.hero_deck_name in {d.name for d in decks}
        assert redeal_cfg.hero_deck_name != redeal_cfg.villain_deck_name

        # Watch with pool resolves one consistent pair across both seats and steps cleanly.
        created = create_watch(
            WatchRequest(
                players=["random", "random"],
                hero_deck="pool:tla_trophies",
                villain_deck="pool:tla_trophies",
                seed=77,
                decklists="closed",
            )
        )
        step_watch(created["id"], 0)
        row = inspect_watch(created["id"], 0)
        recorded_decks = [p["deck"] for p in row["identities"]]
        assert len(set(recorded_decks)) == 2
        assert set(recorded_decks) <= {d.name for d in decks}
        server.end_game(created["id"], "test")

        # Unknown pool raises AcademyLaunchError.
        with pytest.raises(AcademyLaunchError) as missing:
            academy.new_game(deck="pool:does_not_exist", opponent="random")
        assert missing.value.code == "deck.unknown"


def test_inline_policy_factory_in_new_game_and_watch(
    site_dir: Path, tmp_path: Path
) -> None:
    from collections.abc import Sequence

    from managym.policy import PolicyDecision, PolicyDescriptor, PolicyView

    del site_dir

    class InlineSession:
        def observe(self, view: PolicyView) -> None:
            del view

        def choose(self, view: PolicyView) -> PolicyDecision:
            return PolicyDecision(view.choices[0], view.binding)

        def close(self) -> None:
            pass

    class InlineFactory:
        def __init__(self, label: str) -> None:
            self.label = label

        def describe(self) -> PolicyDescriptor:
            return PolicyDescriptor(
                self.label, "jace", "jace/0.1", config={"label": self.label}
            )

        def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
            assert len(setup) == 2

        def load(self, *, seat: int, seed: int) -> InlineSession:
            del seat, seed
            return InlineSession()

    d1, d2, _ = _sample_decks()
    bot_a, bot_b = InlineFactory("Jace A"), InlineFactory("Jace B")
    with serve(port=0, traces_dir=tmp_path / "traces") as academy:
        game = academy.new_game(
            deck=d1,
            opponent_deck=d2,
            opponent=bot_a,
            seed=5,
            decklists="closed",
        )
        launch_id = parse_qs(urlparse(game.url).query)["launch"][0]
        cfg = server._parse_game_config(server.LAUNCHES[launch_id].config)
        assert cfg.opponent is not None
        assert cfg.opponent["name"] == "Jace A"
        assert cfg.hero_deck_name == d1.name

        watched = academy.watch(
            players=(bot_a, bot_b),
            decks=(d1, d2),
            seed=6,
            decklists="closed",
        )
        params = parse_qs(urlparse(watched.url).query)
        assert params["seat1"][0].startswith("policy:")
        assert params["seat2"][0].startswith("policy:")
        assert params["seat1"][0] != params["seat2"][0]
