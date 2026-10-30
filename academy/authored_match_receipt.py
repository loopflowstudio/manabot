"""Checked terminal authority receipt for the selected authored matchup."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import Any

from managym.experience import InteractionOffer

from manabot.semantic.learning import PROGRAM_KINDS
import managym

from .server import DecisionContext, GameSession

ROOT = Path(__file__).parents[1]
SEMANTIC_PACK_KEY = "ur-lessons-vs-gw-allies"
DEFAULT_RECEIPT_PATH = (
    ROOT / "conformance/authored-match-w17-v1/release-stack-ur-vs-gw-seed-0.json"
)
FIXED_TIME = "2026-07-17T00:00:00+00:00"
MATCH_SEED = 0
POLICY_SEED = 0
MAX_HERO_COMMANDS = 10_000


class UnsupportedAuthorityPrompt(RuntimeError):
    """The fixed trace could not continue through a server-built offer."""


class DeterministicServerOfferPolicy:
    """Choose uniformly from server-built offers without inspecting card text."""

    def __init__(self, seed: int, max_commands: int | None = None):
        self._rng = Random(seed)
        self.max_commands = max_commands

    def choose(
        self,
        offers: list[dict[str, Any]],
        *,
        actor: int,
        revision: int,
        prompt_family: str,
    ) -> dict[str, Any]:
        if self.max_commands is not None and revision >= self.max_commands:
            raise UnsupportedAuthorityPrompt("registered total Command cap reached")
        if not offers:
            raise UnsupportedAuthorityPrompt(
                f"unsupported prompt actor={actor} revision={revision} "
                f"family={prompt_family}: no server offers"
            )
        validated = [
            InteractionOffer.model_validate(offer).model_dump(mode="json")
            for offer in offers
        ]
        offer = validated[self._rng.randrange(len(validated))]
        if offer["choices"]:
            raise UnsupportedAuthorityPrompt(
                f"unsupported prompt actor={actor} revision={revision} "
                f"family={prompt_family}: deterministic structured answers required"
            )
        return offer

    def __call__(self, context: DecisionContext) -> int:
        offer = self.choose(
            context.offers,
            actor=context.viewer,
            revision=context.revision,
            prompt_family=context.action_space,
        )
        return int(offer["id"])


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _program_index() -> tuple[
    dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]]
]:
    """Name every program of the authored pack from the engine's own catalog.

    A program is one span of ``managym.learning_catalog``: each triggered,
    mana, triggered-mana and activated ability, plus the spell. Its key is
    ``<definition key>.<kind>.<ordinal within the definition>``.
    """
    pool = managym.content_pool(SEMANTIC_PACK_KEY)
    catalog = managym.learning_catalog(SEMANTIC_PACK_KEY)
    kind_names = {value: name for name, value in PROGRAM_KINDS.items()}
    definitions = {int(row["card_def_id"]): row for row in pool["definitions"]}
    offsets = catalog["definition_program_offsets"]
    by_definition: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_key: dict[str, dict[str, Any]] = {}
    for definition_index in sorted(definitions):
        definition = definitions[definition_index]
        start, stop = offsets[definition_index], offsets[definition_index + 1]
        for ordinal, program_index in enumerate(
            catalog["definition_program_rows"][start:stop]
        ):
            first = catalog["program_offsets"][program_index]
            kind = kind_names[catalog["token_value"][first]]
            row = {
                "program_index": int(program_index),
                "key": f"{definition['key']}.{kind}.{ordinal}",
                "kind": kind,
                "definition_index": definition_index,
                "definition_key": str(definition["key"]),
                "registry_name": str(definition["registry_name"]),
            }
            by_definition[definition_index].append(row)
            by_key[row["key"]] = row
    return dict(by_definition), by_key


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("pack_key") != SEMANTIC_PACK_KEY:
        raise RuntimeError("authored release-stack match did not select its pack")
    pool = managym.content_pool(SEMANTIC_PACK_KEY)
    if manifest["content_digest"] != pool["content_digest"]:
        raise RuntimeError("authored content digest drifted from the native pool")
    names = {
        int(row["card_def_id"]): row["registry_name"] for row in pool["definitions"]
    }
    actual = {
        int(row["card_def_id"]): row["registry_name"] for row in manifest["definitions"]
    }
    if actual != names:
        raise RuntimeError("authored registry binding drifted from the native pool")


def _player_witness(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_index": int(player["player_index"]),
        "id": int(player["id"]),
        "life": int(player["life"]),
        "zone_counts": dict(sorted(player["zone_counts"].items())),
        "battlefield": sorted(
            (
                {
                    "id": int(permanent["id"]),
                    "name": permanent["name"],
                    "controller_id": int(permanent["controller_id"]),
                    "tapped": bool(permanent["tapped"]),
                    "damage": int(permanent["damage"]),
                    "power": permanent["power"],
                    "toughness": permanent["toughness"],
                    "plus1_counters": int(permanent["plus1_counters"]),
                }
                for permanent in player["battlefield"]
            ),
            key=lambda permanent: permanent["id"],
        ),
    }


def _terminal_witness(
    session: GameSession,
    frame: dict[str, Any],
) -> dict[str, Any]:
    if session.env is None:
        raise RuntimeError("terminal environment is unavailable")
    projection = frame["projection"]
    players = sorted(
        (_player_witness(projection[side]) for side in ("agent", "opponent")),
        key=lambda player: player["player_index"],
    )
    return {
        "terminal": bool(projection["game_over"]),
        "winner": frame["winner"],
        "revision": int(frame["revision"]),
        "frame_hash": str(frame["frame_hash"]),
        "state_digest": session.env.state_digest(),
        "turn": deepcopy(projection["turn"]),
        "players": players,
    }


def play_fixed_authored_match(
    *,
    seed: int = MATCH_SEED,
    reverse: bool = False,
    max_commands: int | None = None,
    session: GameSession | None = None,
) -> tuple[GameSession, dict[str, Any]]:
    """Return the deterministic authored Game session with retained Study roots."""

    hero_policy = DeterministicServerOfferPolicy(seed, max_commands)
    villain_policy = DeterministicServerOfferPolicy(seed, max_commands)
    if session is None:
        temporary = TemporaryDirectory()
        session = GameSession(
            Path(temporary.name),
            id_factory=lambda kind: f"authored-authority-{kind}",
            clock=lambda: FIXED_TIME,
            capture_authority_evidence=True,
        )
        session._authority_receipt_temporary = temporary  # type: ignore[attr-defined]
    session._villain_offer_policy = villain_policy
    message = session.new_game(
        {
            "hero_deck": "gw_allies" if reverse else "ur_lessons",
            "villain_deck": "ur_lessons" if reverse else "gw_allies",
            "villain_type": "random",
            "seed": seed,
            "auto_pass": False,
        }
    )
    frame = message["frame"]
    for ordinal in range(MAX_HERO_COMMANDS):
        if frame["status"] == "game_over":
            break
        prompt = frame.get("prompt")
        if not isinstance(prompt, dict):
            raise UnsupportedAuthorityPrompt(
                f"unsupported prompt actor=0 revision={frame['revision']}: missing prompt"
            )
        offer = hero_policy.choose(
            frame["offers"],
            actor=0,
            revision=int(frame["revision"]),
            prompt_family=str(frame["action_space"]),
        )
        outcome = session.hero_command(
            {
                "command_id": f"authored-authority.hero.{ordinal}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": prompt["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        if outcome["status"] != "accepted":
            raise UnsupportedAuthorityPrompt(
                f"command rejected at revision={frame['revision']} "
                f"family={frame['action_space']}: {outcome}"
            )
        frame = outcome["update"]["frame"]
    else:
        raise RuntimeError("authored authority match exceeded hero command limit")

    if frame["status"] != "game_over" or frame["winner"] not in (0, 1):
        raise RuntimeError("authored authority match did not reach a winner")
    return session, frame


def _ledger(
    session: GameSession,
    programs_by_definition: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, set[int]]]:
    transitions = session.authority_transitions
    decisions = session.canonical_decisions
    if len(transitions) != len(decisions):
        raise RuntimeError(
            "fixed receipt requires one deliberate decision per engine transition"
        )

    program_revisions: dict[str, set[int]] = defaultdict(set)
    ledger: list[dict[str, Any]] = []
    for ordinal, (transition, decision) in enumerate(
        zip(transitions, decisions, strict=True)
    ):
        if (
            transition.automatic
            or transition.command is None
            or transition.offer is None
        ):
            raise RuntimeError(
                f"transition {ordinal} bypassed a deliberate bound command"
            )
        decision_offer = decision.offer.model_dump(mode="json")
        decision_command = decision.command.model_dump(mode="json")
        if transition.offer != decision_offer or transition.command != decision_command:
            raise RuntimeError(
                f"canonical decision {ordinal} differs from authority step"
            )
        if (
            int(decision.ordinal) != ordinal
            or int(decision.revision) != transition.from_revision
            or int(decision.prompt_id) != transition.prompt_id
        ):
            raise RuntimeError(f"canonical decision {ordinal} identity drifted")
        if transition.offer_count != transition.legal_action_count:
            raise RuntimeError(f"server offer surface capped decision {ordinal}")

        encountered_programs = sorted(
            {
                program["key"]
                for definition_id in transition.encountered_definition_ids
                for program in programs_by_definition.get(definition_id, [])
            }
        )
        for key in encountered_programs:
            program_revisions[key].add(transition.from_revision)
        semantic_events = [
            {"ordinal": event_ordinal, **event}
            for event_ordinal, event in enumerate(transition.semantic_events)
        ]
        ledger.append(
            {
                "ordinal": ordinal,
                "actor": transition.actor,
                "source": transition.source,
                "from_revision": transition.from_revision,
                "to_revision": transition.to_revision,
                "prompt_family": transition.action_space,
                "legal_action_count": transition.legal_action_count,
                "offer_count": transition.offer_count,
                "offer_family": {
                    "verb": transition.offer["verb"],
                    "action_type": transition.offer["action_type"],
                },
                "prompt_id": transition.prompt_id,
                "offer": transition.offer,
                "command": transition.command,
                "command_receipt": {
                    "command_id": transition.command["command_id"],
                    "actor": transition.actor,
                    "accepted_at": transition.from_revision,
                    "resulting_revision": transition.to_revision,
                    "resulting_state_digest": transition.state_after,
                },
                "state": {
                    "before": transition.state_before,
                    "after": transition.state_after,
                },
                "semantic_events": semantic_events,
                "presentation_events": transition.presentation_events,
                "encountered_definition_ids": transition.encountered_definition_ids,
                "encountered_programs": encountered_programs,
            }
        )
    return ledger, program_revisions


def generate_authored_match_receipt() -> dict[str, Any]:
    programs_by_definition, programs_by_key = _program_index()
    session, terminal_frame = play_fixed_authored_match()
    if session.env is None:
        raise RuntimeError("authored authority environment is unavailable")
    manifest = session.env.content_pack_manifest()
    _validate_manifest(manifest)
    ledger, program_revisions = _ledger(session, programs_by_definition)

    encountered_programs = []
    for key, revisions in sorted(
        program_revisions.items(),
        key=lambda item: programs_by_key[item[0]]["program_index"],
    ):
        row = deepcopy(programs_by_key[key])
        row["encountered_revisions"] = sorted(revisions)
        encountered_programs.append(row)

    prompt_families = Counter(row["prompt_family"] for row in ledger)
    offer_families = Counter(
        f"{row['offer_family']['verb']}:{row['offer_family']['action_type']}"
        for row in ledger
    )
    semantic_event_families = Counter(
        event["event_type"] for row in ledger for event in row["semantic_events"]
    )
    presentation_event_families = Counter(
        event["kind"]["kind"] for row in ledger for event in row["presentation_events"]
    )
    fallback_counters = dict(session.authority_fallback_counters)
    if any(fallback_counters.values()):
        raise RuntimeError(f"authority fallback exercised: {fallback_counters}")

    return {
        "version": 1,
        "identity": {
            "matchup": "ur-lessons-vs-gw-allies",
            "seed": MATCH_SEED,
            "generated_at": FIXED_TIME,
            "policy": {
                "hero": "uniform-server-offer-v1",
                "villain": "uniform-server-offer-v1",
                "hero_seed": POLICY_SEED,
                "villain_seed": POLICY_SEED,
            },
            "content_pack_manifest": manifest,
            "asset_pack": terminal_frame["asset_pack"],
            "content_hash": terminal_frame["content_hash"],
            "asset_manifest_hash": terminal_frame["asset_manifest_hash"],
        },
        "summary": {
            "deliberate_commands": len(ledger),
            "maximum_offer_count": max(row["offer_count"] for row in ledger),
            "automatic_rules_actions": sum(
                transition.automatic for transition in session.authority_transitions
            ),
            "prompt_families": dict(sorted(prompt_families.items())),
            "offer_families": dict(sorted(offer_families.items())),
            "semantic_event_families": dict(sorted(semantic_event_families.items())),
            "presentation_event_families": dict(
                sorted(presentation_event_families.items())
            ),
            "fallback_counters": fallback_counters,
        },
        "encountered_typed_programs": encountered_programs,
        "decisions": ledger,
        "terminal_witness": _terminal_witness(session, terminal_frame),
    }


def write_or_check_authored_match_receipt(
    path: Path = DEFAULT_RECEIPT_PATH,
    *,
    check: bool,
) -> None:
    generated = _json_bytes(generate_authored_match_receipt())
    if check:
        if not path.exists() or path.read_bytes() != generated:
            raise RuntimeError(
                "authored match receipt is stale; regenerate with "
                "`uv run --extra dev python scripts/generate_authored_match_receipt.py`"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(generated)
