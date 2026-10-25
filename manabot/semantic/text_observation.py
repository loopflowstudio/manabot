"""Deterministic English projection and exact action-reference binding.

`render_decision` consumes only an atomic native capture (or retained JSON).
`capture_text_decision` retains the opaque live origin handle; `submit_response`
uses it for atomic origin/digest/staleness validation. No tokenizer, model,
recovery policy, private state, or writable model memory lives here.
"""

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import re
from typing import Callable

from managym.decision import Command, SemanticContractError, SemanticTransition
from managym.decision_capture import (
    Action,
    CardReference,
    DecisionInput,
    DecisionObject,
    HistoryReference,
    Keywords,
    ManaCost,
    ObjectRenderId,
    PlayerReference,
    UnavailableReference,
)
from pydantic import ValidationError

import managym

REPRESENTATION_VERSION = "manabot-text-1"
_ZONES = ("library", "hand", "battlefield", "graveyard", "stack", "exile", "command")
_PHASES = ("beginning", "first main", "combat", "second main", "ending")
_STEPS = (
    "untap",
    "upkeep",
    "draw",
    "main",
    "beginning of combat",
    "declare attackers",
    "declare blockers",
    "combat damage",
    "end of combat",
    "second main",
    "end",
    "cleanup",
)
_SPACES = (
    "game over",
    "priority",
    "declare attacker",
    "declare blocker",
    "choose target",
    "scry",
    "look and select",
    "pay or decline",
    "mode",
    "learn",
    "waterbend",
    "discard",
    "legend rule",
)
_VERBS = (
    "play_land",
    "cast",
    "pass_priority",
    "declare_attacker",
    "declare_blocker",
    "choose_target",
    "activate_ability",
    "scry_keep",
    "scry_bottom",
    "select_card",
    "decline",
    "pay_cost",
    "choose_mode",
    "waterbend_tap",
    "learn_take_lesson",
    "learn_discard",
)
_STAGES = frozenset(
    (
        "priority",
        "targets",
        "additional_cost",
        "payment",
        "mode",
        "declare_attackers",
        "declare_blockers",
        "mana_payment",
        "mana_color",
        "resolution_sacrifice",
        "private_search",
        "learn",
        "private_inspection",
        "scry",
        "resolution_payment",
        "resolution_mode",
        "discard",
        "select_milled",
        "legend_rule",
    )
)


class TextEncodingError(ValueError):
    """Unsupported or inconsistent representation input; never a protocol retry."""


class RepresentationOverflow(TextEncodingError):
    """The complete prompt exceeds a declared byte or tokenizer budget."""


class ResponseProtocolError(ValueError):
    """Response is not exactly one of the listed action references."""


@dataclass(frozen=True, slots=True)
class TextAction:
    reference: str
    description: str
    command: Command


@dataclass(frozen=True, slots=True)
class TextDecision:
    representation_version: str
    text: str
    actions: tuple[TextAction, ...]
    capture_json: str
    capture_digest: str
    native_capture: managym.DecisionCapture | None = field(
        default=None, repr=False, compare=False
    )

    def check_token_budget(
        self, count_tokens: Callable[[str], int], maximum: int
    ) -> int:
        """Count the whole prompt; the harness must also budget its chat template."""
        count = count_tokens(self.text)
        if maximum < 1 or count < 0 or count > maximum:
            raise RepresentationOverflow(
                f"complete prompt has {count} tokens; budget {maximum}"
            )
        return count


def _enum(values: tuple[str, ...], index: int) -> str:
    if not 0 <= index < len(values):
        raise TextEncodingError(f"unsupported enum value {index}")
    return values[index]


def _keywords(value: Keywords) -> str:
    return ", ".join(
        key.replace("_", " ") for key, enabled in value.model_dump().items() if enabled
    )


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _selectors(mode: int | None, ability: int | None) -> list[str]:
    return [
        f"{label} {value + 1}"
        for label, value in (("mode", mode), ("ability", ability))
        if value is not None
    ]


def _mana(cost: ManaCost | None) -> str:
    if cost is None:
        return "no mana cost"
    if len(cost.cost) not in (6, 7) or any(value < 0 for value in cost.cost):
        raise TextEncodingError("unsupported mana cost")
    generic = (
        cost.cost[6]
        if len(cost.cost) == 7
        else cost.mana_value - sum(cost.cost) - len(cost.hybrid)
    )
    if generic < 0:
        raise TextEncodingError("inconsistent mana value")
    symbols = (f"{{{generic}}}" if generic else "") + "".join(
        f"{{{symbol}}}" * count
        for symbol, count in zip("WUBRGC", cost.cost[:6], strict=True)
    )
    colors = {
        name: symbol
        for name, symbol in zip(
            ("White", "Blue", "Black", "Red", "Green", "Colorless"),
            "WUBRGC",
            strict=True,
        )
    }
    for pair in cost.hybrid:
        if len(pair) != 2:
            raise TextEncodingError("unsupported hybrid cost")
        symbols += (
            "{"
            + "/".join(
                _enum(tuple("WUBRGC"), value)
                if isinstance(value, int)
                else colors[value]
                for value in pair
            )
            + "}"
        )
    return (symbols or "{0}") + (
        " (Lesson spending context)" if cost.spend_on_lesson else ""
    )


class _Renderer:
    def __init__(self, capture: DecisionInput) -> None:
        self.capture = capture
        self.state = capture.observation.viewer_state
        self.names: dict[int, str] = {}
        self.seats: dict[int, str] = {}
        self.objects: dict[tuple[int, int], str] = {}
        self.stack: dict[int, str] = {}
        self.definitions = {
            int(key): f"D{index + 1}"
            for index, key in enumerate(sorted(capture.definitions, key=int))
        }
        for player, label in (
            (self.state.agent, "You"),
            (self.state.opponent, "Opponent"),
        ):
            self.names[player.id] = label
            self.seats[player.player_index] = label
        for index, card in enumerate(
            self.state.agent_cards + self.state.opponent_cards
        ):
            label = f"C{index + 1} ({card.name})"
            self.names[card.id] = label
            link = capture.objects[str(card.id)]
            self.objects[card.id, link.incarnation] = label
            if link.permanent_id is not None:
                self.names[link.permanent_id] = label
                self.objects[link.permanent_id, link.incarnation] = label
        for index, card in enumerate(self.state.agent_sideboard):
            self.names[card.candidate_id] = f"O{index + 1} ({card.name})"
        self.stack = {
            item.stack_object_id: f"S{index + 1}"
            for index, item in enumerate(self.state.stack_objects)
        }

    def name(self, identity: int) -> str:
        try:
            return self.names[identity]
        except KeyError as error:
            raise TextEncodingError("unresolved visible object reference") from error

    def definition(self, identity: int) -> str:
        try:
            return self.definitions[identity]
        except KeyError as error:
            raise TextEncodingError("unbound definition") from error

    def decision_object(self, value: DecisionObject) -> str:
        if value.subject.kind == "player" and isinstance(value.subject.id, int):
            return self.seats[value.subject.id]
        if value.subject.kind != "object" or not isinstance(
            value.subject.id, ObjectRenderId
        ):
            raise TextEncodingError("unsupported continuation subject")
        # An expired incarnation must not inherit the alias of a returned card.
        return self.objects.get(
            (value.subject.id.entity, value.subject.id.incarnation),
            f"{value.label} (expired object; no current link)",
        )

    def history_reference(self, value: HistoryReference) -> str:
        if isinstance(value, UnavailableReference):
            return "unavailable"
        if isinstance(value, PlayerReference):
            return self.seats[value.seat]
        assert isinstance(value, CardReference)
        definition = self.definition(value.definition)
        zone = "unavailable zone" if value.zone is None else _enum(_ZONES, value.zone)
        link = (
            "no current object link (expired or unavailable)"
            if value.context_id is None
            else self.name(value.context_id)
        )
        return f"{definition}, owner {self.seats[value.owner]}, {zone}, {link}"

    def action(self, action: Action) -> str:
        kind = _enum(_VERBS, action.type)
        focus = [self.name(identity) for identity in action.focus]
        first = focus[0] if focus else ""
        if (
            kind
            in (
                "play_land",
                "cast",
                "activate_ability",
                "choose_target",
                "scry_keep",
                "scry_bottom",
                "select_card",
                "waterbend_tap",
                "learn_take_lesson",
                "learn_discard",
            )
            and not first
        ):
            raise TextEncodingError("action missing required visible focus")
        if kind == "declare_attacker":
            if action.declared is None or len(focus) != 1:
                raise TextEncodingError("attack declaration missing")
            label = (
                "Attack with " if action.declared else "Do not attack with "
            ) + first
        elif kind == "declare_blocker":
            if len(focus) not in (1, 2):
                raise TextEncodingError("unsupported blocker focus")
            label = (
                f"Block {focus[1]} with {first}"
                if len(focus) == 2
                else f"Do not block with {first}"
            )
        else:
            verb = {
                "play_land": "Play land",
                "cast": "Cast",
                "pass_priority": "Pass priority",
                "activate_ability": "Activate ability of",
                "choose_target": "Target",
                "scry_keep": "Keep on top",
                "scry_bottom": "Put on bottom",
                "select_card": "Select",
                "decline": "Decline / finish the offered choice",
                "pay_cost": "Pay the offered cost",
                "choose_mode": "Choose mode",
                "waterbend_tap": "Tap for cost",
                "learn_take_lesson": "Take Lesson",
                "learn_discard": "Discard for Learn",
            }[kind]
            label = verb + (" " + ", ".join(focus) if focus else "")
        # Attack/block verbs already express participation. Keep other declared
        # values and every present selector; omit absent optional metadata.
        details = _selectors(action.mode_index, action.ability_index)
        if action.declared is not None and kind not in (
            "declare_attacker",
            "declare_blocker",
        ):
            details.insert(0, f"declaration {_yes(action.declared)}")
        if details:
            label += "; " + "; ".join(details)
        if action.description:
            label += f". Native description: {action.description}"
        return label

    def render(self) -> tuple[str, tuple[TextAction, ...]]:
        capture, state = self.capture, self.state
        identity, frame = capture.observation.identity, capture.observation.decision
        if (
            frame.revision != identity.revision
            or frame.actor != identity.viewer
            or state.agent.player_index != identity.viewer
            or capture.continuation.actor != frame.actor
            or state.game_over
            or state.recent_events
            or capture.observation.events
            or capture.continuation.recent_facts
        ):
            raise TextEncodingError(
                "inconsistent acting decision or unsupported event increment"
            )
        if capture.continuation.stage not in _STAGES:
            raise TextEncodingError("unsupported continuation stage")
        if (
            not 0 <= capture.history.capacity <= 32
            or len(capture.history.rows) > capture.history.capacity
        ):
            raise TextEncodingError("invalid public history bound")
        if len(frame.offers) != len(state.action_space.actions) or not frame.offers:
            raise TextEncodingError("incomplete action/offer binding")
        if len({offer.id for offer in frame.offers}) != len(frame.offers):
            raise TextEncodingError("duplicate offer identity")
        if any(card.zone == 1 for card in state.opponent_cards):
            raise TextEncodingError("opponent hand identity in capture")
        lines = [
            "You are choosing one legal action in the current game.",
            "Return only one listed action reference, such as A0. No other text.",
            "All legal alternatives are listed. Further choices occur only when offered.",
            "Optional details are shown when provided; unavailable facts are marked explicitly.",
            "",
            f"Turn {state.turn.turn_number}. Phase: {_enum(_PHASES, state.turn.phase)}. Step: {_enum(_STEPS, state.turn.step)}.",
            f"Active player: {self.name(state.turn.active_player_id)}. Acting player: {self.name(state.turn.agent_player_id)}.",
            f"Decision: {_enum(_SPACES, state.action_space.type)}. Game over: {_yes(state.game_over)}. Won: {_yes(state.won)}.",
            "",
            "Players:",
        ]
        for player in (state.agent, state.opponent):
            if (
                len(player.zone_counts) != 7
                or len(player.mana) != 6
                or len(player.lesson_mana) != 6
            ):
                raise TextEncodingError("unsupported player quantities")
            counts = "; ".join(
                f"{zone} {count}"
                for zone, count in zip(_ZONES, player.zone_counts, strict=True)
            )
            lines += [
                f"{self.name(player.id)}: life {player.life}; active {_yes(player.is_active)}; viewer {_yes(player.is_agent)}; {counts}.",
                "Mana pool: "
                + ", ".join(
                    f"{symbol} {count}"
                    for symbol, count in zip("WUBRGC", player.mana, strict=True)
                )
                + ".",
                "Lesson-only mana: "
                + ", ".join(
                    f"{symbol} {count}"
                    for symbol, count in zip("WUBRGC", player.lesson_mana, strict=True)
                )
                + f". Until-end-of-combat mana total: {player.combat_mana}.",
                f"Lessons in graveyard: {player.graveyard_lessons}.",
            ]
            for label, quantities in (
                ("Publicly known hand minima", player.known_hand),
                ("Initial sideboard", player.sideboard_counts),
                ("Remaining sideboard", player.remaining_sideboard_counts),
            ):
                lines.append(
                    label
                    + ": "
                    + (
                        ", ".join(
                            f"{self.definition(int(key))} × {count}"
                            for key, count in sorted(
                                quantities.items(), key=lambda item: int(item[0])
                            )
                        )
                        or "empty"
                    )
                    + "."
                )
        lines += [
            "Opponent's other hand identities and both library orders are unavailable.",
            "",
            "Visible cards (separate aliases for separate copies):",
        ]
        for card in state.agent_cards + state.opponent_cards:
            flags = ", ".join(
                key.removeprefix("is_").replace("_", " ")
                for key, enabled in card.card_types.model_dump().items()
                if enabled
            )
            optional = f"; type flags {flags}" if flags else ""
            if keywords := _keywords(card.keywords):
                optional += f"; printed keywords {keywords}"
            lines.append(
                f"{self.name(card.id)} [{self.definition(card.registry_key)}]: owner {self.name(card.owner_id)}; zone {_enum(_ZONES, card.zone)}; card power/toughness {card.power}/{card.toughness}; cost {_mana(card.mana_cost)}; mana value {card.mana_cost.mana_value}{optional}; unlisted type and keyword flags false; token {_yes(card.is_token)}; legendary {_yes(card.is_legendary)}; Ally {_yes(card.is_ally)}; Lesson {_yes(card.is_lesson)}; ward mana value {card.ward_cost}; kicker mana value {card.kicker_cost}. Rules: {card.text_box or '(empty)'}"
            )
        if not state.agent_cards and not state.opponent_cards:
            lines.append("Empty.")
        lines += ["", "Owned outside-game candidates:"]
        for card in state.agent_sideboard:
            lines.append(
                f"{self.name(card.candidate_id)} [{self.definition(card.registry_key)}]: owner {self.name(card.owner_id)}. Rules: {card.text_box or '(empty)'}"
            )
        if not state.agent_sideboard:
            lines.append("Empty.")
        lines += ["", "Battlefield effective state:"]
        for permanent in state.agent_permanents + state.opponent_permanents:
            optional = ""
            if permanent.temp_subtypes:
                optional += f"; temporary subtypes {', '.join(permanent.temp_subtypes)}"
            if keywords := _keywords(permanent.keywords):
                optional += f"; effective keywords {keywords}"
            lines.append(
                f"{self.name(permanent.id)} [{self.definition(permanent.registry_key)}]: controller {self.name(permanent.controller_id)}; power/toughness {permanent.power}/{permanent.toughness}; damage {permanent.damage}; tapped {_yes(permanent.tapped)}; summoning sick {_yes(permanent.is_summoning_sick)}; +1/+1 counters {permanent.plus1_counters}; animated {_yes(permanent.is_animated)}; holds exile link {_yes(permanent.has_exile_link)}; cannot be blocked this turn {_yes(permanent.cant_be_blocked_this_turn)}; exile if dies this turn {_yes(permanent.exile_if_dies_this_turn)}{optional}; unlisted effective keywords false."
            )
        if not state.agent_permanents and not state.opponent_permanents:
            lines.append("Empty.")
        lines += ["", "Stack (native order, top to bottom):"]
        for item in state.stack_objects:
            targets: list[str] = []
            for target in item.targets:
                ids = (
                    target.player_id,
                    target.permanent_id,
                    target.stack_object_id,
                    target.card_id,
                )
                _enum(("player", "permanent", "stack", "card"), target.kind)
                selected = ids[target.kind]
                if selected is None or sum(value is not None for value in ids) != 1:
                    raise TextEncodingError("inconsistent stack target")
                targets.append(
                    self.stack[selected] if target.kind == 2 else self.name(selected)
                )
            source = (
                "unavailable"
                if item.source_permanent_id is None
                else self.name(item.source_permanent_id)
            )
            details = [
                f"controller {self.name(item.controller_id)}",
                f"source permanent {source}",
            ]
            details.extend(_selectors(item.chosen_mode, item.ability_index))
            if targets:
                details.append("targets " + ", ".join(targets))
            lines.append(
                f"{self.stack[item.stack_object_id]}: {_enum(('spell', 'activated ability', 'triggered ability'), item.kind)} [{self.definition(item.source_card_registry_key)}]; {'; '.join(details)}."
            )
        if not state.stack_objects:
            lines.append("Empty.")
        context = capture.continuation
        lines += [
            "",
            "Current continuation:",
            f"Stage: {context.stage.replace('_', ' ')}. {context.instruction}",
        ]
        if context.source is not None:
            lines.append(f"Source: {self.decision_object(context.source)}.")
        if context.chosen_mode is not None:
            lines.append(f"Chosen mode: {context.chosen_mode + 1}.")
        if context.targets:
            lines.append(
                "Selected targets in order: "
                + "; ".join(
                    f"clause {target.clause + 1}: {self.decision_object(target.object)}"
                    for target in context.targets
                )
                + "."
            )
        if context.reserved_sacrifice is not None:
            lines.append(
                f"Reserved sacrifice: {self.decision_object(context.reserved_sacrifice)}."
            )
        if context.mana_cost is not None:
            lines.append(f"Outstanding mana cost: {context.mana_cost}.")
        lines += ["", "Bound public card definitions:"]
        for key in sorted(capture.definitions, key=int):
            definition = capture.definitions[key]
            types = " ".join(definition.supertypes + definition.types.types)
            optional = ""
            if definition.power is not None and definition.toughness is not None:
                optional += (
                    f"; power/toughness {definition.power}/{definition.toughness}"
                )
            elif definition.power is not None:
                optional += f"; power {definition.power}"
            elif definition.toughness is not None:
                optional += f"; toughness {definition.toughness}"
            if definition.subtypes:
                optional += f"; subtypes {', '.join(definition.subtypes)}"
            if keywords := _keywords(definition.keywords):
                optional += f"; keywords {keywords}"
            if definition.kicker is not None:
                optional += f"; kicker {_mana(definition.kicker)}"
            if definition.ward is not None:
                optional += f"; ward {_mana(definition.ward)}"
            lines.append(
                f"{self.definition(int(key))} — {definition.name}. {types}; cost {_mana(definition.mana_cost)}{optional}; unlisted keywords false; token {_yes(definition.is_token)}. Rules: {definition.text_box or '(empty)'}"
            )
        lines += [
            "",
            f"Public policy history v1: chronological suffix, capacity {capture.history.capacity} records; no writable model memory.",
        ]
        for row in capture.history.rows:
            lines.append(
                f"{_enum(('Arrival', 'Damage', 'Life change', 'Cast', 'Resolved', 'Countered'), row.kind)}: amount {row.amount}; source {self.history_reference(row.source)}; target {self.history_reference(row.target)}."
            )
        if not capture.history.rows:
            lines.append("Empty." if capture.history.capacity else "History disabled.")
        lines += ["", "Legal actions:"]
        actions: list[TextAction] = []
        for index, (offer, descriptor) in enumerate(
            zip(frame.offers, state.action_space.actions, strict=True)
        ):
            if (
                offer.actor != frame.actor
                or offer.choices
                or offer.verb != _enum(_VERBS, descriptor.type)
            ):
                raise TextEncodingError("unsupported or inconsistent native offer")
            description = self.action(descriptor)
            reference = f"A{index}"
            command = Command(
                f"text:{capture.digest}:{reference}", frame.revision, offer.id
            )
            actions.append(TextAction(reference, description, command))
            lines.append(f"{reference} — {description}.")
        return "\n".join(lines) + "\n", tuple(actions)


def render_decision(capture_json: str, *, max_bytes: int = 262_144) -> TextDecision:
    """Pure projection of retained capture JSON; unknown fields fail closed."""
    try:
        capture = DecisionInput.model_validate_json(capture_json)
        payload = json.loads(capture_json)
        del payload["digest"]
        digest = sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if digest != capture.digest:
            raise TextEncodingError("capture digest mismatch")
        text, actions = _Renderer(capture).render()
    except (
        ValidationError,
        KeyError,
        TypeError,
        IndexError,
        json.JSONDecodeError,
    ) as error:
        raise TextEncodingError(f"unsupported capture: {error}") from error
    size = len(text.encode())
    if max_bytes < 1 or size > max_bytes:
        raise RepresentationOverflow(
            f"complete prompt has {size} bytes; budget {max_bytes}"
        )
    return TextDecision(
        REPRESENTATION_VERSION, text, actions, capture_json, capture.digest
    )


def capture_text_decision(
    env: managym.Env, *, history_capacity: int = 0, max_bytes: int = 262_144
) -> TextDecision:
    actor = env.current_agent_index()
    if actor is None:
        raise TextEncodingError("no acting decision")
    try:
        capture = env.capture_decision(actor, history_capacity)
    except Exception as error:  # Native capture admission fails before mutation.
        raise TextEncodingError(str(error)) from error
    return replace(
        render_decision(capture.to_json(), max_bytes=max_bytes), native_capture=capture
    )


def resolve_action(decision: TextDecision, response: str) -> Command:
    reference = response.strip()
    if not re.fullmatch(r"A(?:0|[1-9][0-9]*)", reference):
        raise ResponseProtocolError("return exactly one listed action reference")
    for action in decision.actions:
        if action.reference == reference:
            return action.command
    raise ResponseProtocolError("unknown action reference")


def submit_response(
    env: managym.Env, decision: TextDecision, response: str
) -> SemanticTransition:
    """Parse first, then submit once through the originating native authority."""
    command = resolve_action(decision, response)
    if decision.native_capture is None:
        raise SemanticContractError(
            "offline capture has no live origin", code="wrong_origin"
        )
    try:
        result = env.execute_captured_command_json(
            decision.native_capture, command.to_json()
        )
    except Exception as error:  # PyO3 errors are normalized at the native boundary.
        message = str(error)
        raise SemanticContractError(message, code=message.split(":", 1)[0]) from error
    return SemanticTransition.from_json(result)
