"""
trace.py
Trace dataclasses, viewer projection, and SQLite/legacy JSON readers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import TypeAdapter

import managym
from managym.academy.append_json import encode_list, splice
from managym.bots import BotDecision
from managym.disclosure import DecklistDisclosure
from managym.experience_projection import redact_observation
from managym.replay import ReplayStop
from managym.search import SearchConfig

TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
TRACES_DIR = Path(os.getenv("ACADEMY_TRACES_DIR", "academy/traces"))


@dataclass
class GameConfig:
    hero_deck: dict[str, int]
    villain_deck: dict[str, int]
    villain_type: str
    seed: int | None = None
    seed_was_requested: bool = False
    # Hero/villain are setup roles; native seats are independent.
    human_seat: Literal[0, 1] = 0
    # Named-deck identifiers ("interactive" / "ur_lessons" / "gw_allies" /
    # "custom") — recorded so every trace is attributable to an exact matchup.
    hero_deck_name: str = "custom"
    villain_deck_name: str = "custom"
    # Exact installed presentation pack for this oriented matchup. ``None``
    # keeps legacy/custom games and old traces backwards-compatible.
    asset_pack: dict[str, str] | None = None
    # Opponent parameters (recorded in traces so every game is attributable
    # to an exact opponent configuration).
    villain_sims: int | None = None  # search: simulations per legal action
    villain_checkpoint: str | None = None  # checkpoint: path to .pt file
    villain_deterministic: bool = False  # checkpoint: argmax instead of sampling
    opponent: dict[str, str] | None = None
    # Historical checkpoint provenance, or a managym-policy-1 descriptor with
    # complete provider/runner/config/artifact identity. Read without a loader.
    opponent_checkpoint: dict[str, Any] | None = None
    # Priority-stop configuration at game start (MTGO-style auto-pass).
    # ``stops`` maps "my"/"opponent" to the stop step names that surface;
    # None means the server defaults. set_stops updates the live session,
    # not this record.
    stops: dict[str, list[str]] | None = None
    stop_on_stack: bool = True  # always surface when the stack is non-empty
    auto_pass: bool = True  # master switch; False = surface every window
    hero_sideboard: dict[str, int] = field(default_factory=dict)
    villain_sideboard: dict[str, int] = field(default_factory=dict)
    watch_players: list[str] | None = None
    # One bot composition; Watch records each bot through its descriptor.
    mirror: SearchConfig | None = None
    # "open": each player knows the other's main deck and sideboard;
    # "closed": neither. Traces that predate the option were open.
    decklists: DecklistDisclosure = "open"
    # Private import provenance: each seat's decklist (``managym.Decklist``
    # JSON form), admitted by the world. Viewer exports strip them.
    hero_source: dict[str, Any] | None = None
    villain_source: dict[str, Any] | None = None

    # The human controller owns this mode, independent of the chosen seat.
    mana_payment: managym.ManaPayment = "manual"
    mana_floats: managym.ManaFloats = "off"

    def __post_init__(self) -> None:
        if type(self.human_seat) is not int or self.human_seat not in (0, 1):
            raise ValueError("human_seat must be 0 or 1.")
        self.mana_payment = TypeAdapter(managym.ManaPayment).validate_python(
            self.mana_payment
        )
        self.mana_floats = TypeAdapter(managym.ManaFloats).validate_python(
            self.mana_floats
        )
        self.mirror = TypeAdapter(SearchConfig | None).validate_python(self.mirror)
        if self.decklists == "closed" and self.mirror is not None:
            raise ValueError("Search requires open decklists.")

    def to_record(self) -> dict[str, Any]:
        """Omit default-off mana offers from retained game configuration."""
        config = asdict(self)
        if self.mana_payment == "manual":
            del config["mana_payment"]
        if self.mana_floats == "off":
            del config["mana_floats"]
        return config

    def to_rust(self) -> managym.GameSetup:
        """Order human/opponent setup roles by their recorded native seats."""
        configs = self._role_configs()
        return managym.GameSetup(
            list(self.by_seat(*configs)),
            decklists=self.decklists,
            mana_payment=self.by_seat(self.mana_payment, "manual"),
            mana_floats=self.by_seat(self.mana_floats, "off"),
        )

    def by_seat[T](self, human: T, opponent: T) -> tuple[T, T]:
        """Order a human/opponent pair by native seat."""
        return (human, opponent) if self.human_seat == 0 else (opponent, human)

    def _role_configs(self) -> list[managym.PlayerConfig]:
        """Rebuild the recorded root with its explicit outside-game copies."""
        from .curated_pack import curated_pack_for_matchup

        for sideboard in (self.hero_sideboard, self.villain_sideboard):
            if not isinstance(sideboard, dict) or any(
                not isinstance(name, str)
                or not name
                or type(count) is not int
                or count <= 0
                for name, count in sideboard.items()
            ):
                raise ValueError("Recorded sideboard requires positive integer counts")
        if self.hero_source is not None or self.villain_source is not None:
            # Imported: re-admit each retained source against the installed world.
            if self.hero_source is None or self.villain_source is None:
                raise ValueError("Imported trace requires both sources")
            configs = []
            for name, source, deck, sideboard in (
                ("Hero", self.hero_source, self.hero_deck, self.hero_sideboard),
                (
                    "Villain",
                    self.villain_source,
                    self.villain_deck,
                    self.villain_sideboard,
                ),
            ):
                managym.Decklist.from_dict(source).validate_deck(
                    managym.Deck(deck, sideboard)
                )
                configs.append(
                    managym.PlayerConfig(name, deck, sideboard, managym.WORLD_SETUP)
                )
            return configs
        pack = curated_pack_for_matchup(self.hero_deck_name, self.villain_deck_name)
        configs: list[managym.PlayerConfig] = []
        if pack is not None:
            if self.asset_pack != pack.reference:
                raise ValueError("Recorded asset pack differs from the installed setup")
            for seat, name, deck, sideboard in (
                ("Hero", self.hero_deck_name, self.hero_deck, self.hero_sideboard),
                (
                    "Villain",
                    self.villain_deck_name,
                    self.villain_deck,
                    self.villain_sideboard,
                ),
            ):
                expected = pack.player_config(name, seat)
                if deck != expected.decklist or sideboard != expected.sideboard:
                    raise ValueError(
                        f"Recorded setup for {name!r} differs from compiled deck/sideboard setup"
                    )
                configs.append(expected)
            return configs
        return [
            managym.PlayerConfig("Hero", self.hero_deck, self.hero_sideboard),
            managym.PlayerConfig(
                "Villain",
                self.villain_deck,
                self.villain_sideboard,
            ),
        ]


@dataclass
class TraceEvent:
    actor: str
    observation: dict[str, Any]
    actions: list[dict[str, Any]]
    action: int
    action_description: str
    reward: float
    # Authoritative semantic beats caused by this exact engine step. Live play,
    # replay, and the decision-inspector seam consume these same persisted
    # objects; no consumer reconstructs them from observation differences.
    # True when the server passed priority (stops / F6) or, in traces from
    # before native skipping, advanced a forced action. False for explicit
    # Commands, including declared validation prefixes.
    # Competency metrics must also use record origin; False alone is not human
    # provenance and auto-passes are never deliberate passes.
    auto: bool = False
    # Ordered semantic facts committed by the same authority transition. Old
    # traces omit this field and remain valid because replay treats it as an
    # empty sequence.
    presentation: list[dict[str, Any]] = field(default_factory=list)
    bot_decision: BotDecision | None = None
    # Authority-private analysis view; never included in ordinary trace reads.
    analysis_observation: dict[str, Any] | None = None


@dataclass(frozen=True)
class RecordingImport:
    stop: ReplayStop
    build_differs: bool


@dataclass
class Trace:
    config: GameConfig
    events: list[TraceEvent]
    final_observation: dict[str, Any]
    winner: int | None
    end_reason: str
    timestamp: str
    # Complete mixed-view replay truth. It remains authority-private and is
    # removed from the established trace API before any redaction mode is
    # applied. Legacy traces omit it.
    canonical_replay: dict[str, Any] | None = None
    recording_import: RecordingImport | None = None

    def to_record(self) -> dict[str, Any]:
        """Borrow JSON payloads for immediate, read-only persistence.

        Copying every observation and canonical frame before JSON encoding
        makes each live save pay for the entire history twice. Only dataclass
        boundaries need conversion; the store serializes before gameplay resumes.
        Callers must not mutate this record or retain it across a game step.
        """
        record = {item.name: getattr(self, item.name) for item in fields(self)}
        if self.recording_import is None:
            record.pop("recording_import")
        else:
            record["recording_import"] = asdict(self.recording_import)
        record["config"] = self.config.to_record()
        record["mana_payment"] = list(
            self.config.by_seat(self.config.mana_payment, "manual")
        )
        record["mana_floats"] = list(
            self.config.by_seat(self.config.mana_floats, "off")
        )
        record["events"] = [event_record(event) for event in self.events]
        return record

    def to_json(
        self,
        encoded: list[tuple[TraceEvent, str]],
        canonical_replay: str | None = None,
    ) -> str:
        """`json.dumps(self.to_record())`, encoding only events that changed.

        A live trace is saved after every step and holds a full observation
        per event. `encoded` keeps each event's JSON (`append_json`); an event
        is re-encoded only when appended or replaced (its presentation is
        committed after the step). `canonical_replay`, when given, is that
        field's own encoding, built the same way by the caller.
        """
        record = self.to_record()
        parts = {
            "events": encode_list(
                self.events, encoded, lambda event: json.dumps(event_record(event))
            )
        }
        if canonical_replay is not None:
            parts["canonical_replay"] = canonical_replay
        return splice(record, parts)


def event_record(event: TraceEvent) -> dict[str, Any]:
    """One event as its stored record (borrowed, like `Trace.to_record`)."""
    return {
        **{item.name: getattr(event, item.name) for item in fields(event)},
        "bot_decision": (
            None if event.bot_decision is None else asdict(event.bot_decision)
        ),
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_path(trace_id: str, trace_dir: Path) -> Path:
    if not TRACE_ID_PATTERN.fullmatch(trace_id):
        raise ValueError("Invalid trace id")
    return trace_dir / f"{trace_id}.json"


def load_trace(trace_id: str, trace_dir: Path | None = None) -> dict[str, Any]:
    target_dir = trace_dir or TRACES_DIR
    from .attempts import AttemptStore

    recorded = AttemptStore(target_dir / "play.sqlite").load(trace_id)
    if recorded is not None:
        return recorded
    path = _trace_path(trace_id, target_dir)
    if not path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_id}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def list_trace_summaries(trace_dir: Path | None = None) -> list[dict[str, Any]]:
    target_dir = trace_dir or TRACES_DIR
    if not target_dir.exists():
        return []

    summaries: list[dict[str, Any]] = []
    for path in target_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("config"), dict)
            or not isinstance(payload.get("events"), list)
            or not isinstance(payload.get("final_observation"), dict)
        ):
            continue
        events = payload["events"]
        summaries.append(
            {
                "id": path.stem,
                "timestamp": payload.get("timestamp"),
                "winner": payload.get("winner"),
                "end_reason": payload.get("end_reason"),
                "num_events": len(events),
            }
        )

    summaries.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return summaries


def normalize_observation_to_hero(
    observation: dict[str, Any], human_seat: int = 0
) -> None:
    """Present an observation from the recorded human's native perspective.

    Trace events record the raw engine observation, whose perspective is the
    player to act — villain events are villain-perspective. The replay viewer
    (and any human-facing payload) renders ``agent`` as the hero, so swap the
    sides back when needed. Mutates in place; no-op when player_index is
    missing (e.g. minimal test fixtures).
    """
    if not isinstance(observation, dict):
        return
    agent = observation.get("agent")
    opponent = observation.get("opponent")
    if not isinstance(agent, dict) or not isinstance(opponent, dict):
        return
    if (
        agent.get("player_index") == 1 - human_seat
        and opponent.get("player_index") == human_seat
    ):
        observation["agent"], observation["opponent"] = opponent, agent
        if observation.get("game_over") and "won" in observation:
            observation["won"] = not bool(observation["won"])


def prepare_trace_payload(
    payload: dict[str, Any], reveal_hidden: bool = False
) -> dict[str, Any]:
    """Normalize a stored trace for the viewer: hero-perspective throughout,
    with the villain's hand redacted unless ``reveal_hidden``."""
    prepared = deepcopy(payload)
    # ``reveal_hidden`` is a legacy post-game observation aid. It never grants
    # access to the mixed-view canonical authority artifact.
    prepared.pop("canonical_replay", None)
    for event in prepared.get("events", []):
        event.pop("analysis_observation", None)
    config = prepared.get("config", {})
    config.pop("villain_checkpoint", None)
    checkpoint = config.get("opponent_checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("checkpoint", None)
    config.pop("seed", None)
    sources = [config.pop(key, None) for key in ("hero_source", "villain_source")]
    if any(source is not None for source in sources):
        # Submitted lists are private setup provenance. Native viewer observations
        # still expose exactly the public information in the shared contract.
        for key in ("hero_deck", "villain_deck", "hero_sideboard", "villain_sideboard"):
            config[key] = {}
    reveal_hidden = (
        reveal_hidden
        and payload.get("end_reason") == "game_over"
        and bool(payload.get("final_observation", {}).get("game_over"))
    )
    if config.get("decklists") == "closed" and not reveal_hidden:
        config["villain_deck"] = {}
        config["villain_sideboard"] = {}
        config["villain_deck_name"] = "closed"
        config.pop("asset_pack", None)

    observations = [
        event.get("observation", {}) for event in prepared.get("events", [])
    ]
    observations.append(prepared.get("final_observation", {}))

    if not reveal_hidden:
        for event in prepared.get("events", []):
            if event.get("actor") == "villain":
                event["actions"] = []
                event["action"] = -1
                event["action_description"] = "Opponent decision"
                event.pop("bot_decision", None)

    for observation in observations:
        normalize_observation_to_hero(observation, config.get("human_seat", 0))
        if not reveal_hidden:
            redact_observation(observation)

    return prepared
