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

REPRESENTATION_VERSION = "manabot-text-5"
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
    "order triggers",
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
    "choose_trigger",
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
        "order_triggers",
        "trigger_targets",
        "trigger_mode",
        "revealed_hand",
        "split_revealed",
        "free_cast",
        "linked_exile",
        "distribute_counters",
        "keep_creatures",
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
    index = int(index)
    if not 0 <= index < len(values):
        raise TextEncodingError(f"unsupported enum value {index}")
    return values[index]


def _keywords(value: Keywords | managym.Keywords) -> str:
    return ", ".join(
        key.replace("_", " ") for key in Keywords.model_fields if getattr(value, key)
    )


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _selectors(mode: int | None, ability: int | None) -> list[str]:
    return [
        f"{label} {value + 1}"
        for label, value in (("mode", mode), ("ability", ability))
        if value is not None
    ]


def _turn_history(history: managym.PlayerTurnHistory) -> str:
    """What a player has done this turn, or empty when nothing yet."""
    parts = []
    if history.attacked:
        parts.append("attacked")
    if history.permanents_sacrificed:
        parts.append(f"sacrificed {history.permanents_sacrificed} permanents")
    if history.land_entered:
        parts.append("had a land enter under their control")
    if history.cards_drawn:
        parts.append(f"drew {history.cards_drawn} cards")
    return "; ".join(parts)


def _spending(restricted: managym.RestrictedMana) -> str:
    """ "Lesson or Shrine spells", "Ally spells or abilities of Ally sources"."""
    parts = []
    if restricted.spells:
        parts.append(" or ".join(restricted.spells) + " spells")
    if restricted.abilities:
        parts.append("abilities of " + " or ".join(restricted.abilities) + " sources")
    return " or ".join(parts)


def _mana(cost: ManaCost | managym.ManaCost | None) -> str:
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
        " (restricted spending)" if getattr(cost, "spend_on_lesson", False) else ""
    )


def effect_phrase(effect: managym.ContinuousEffect) -> str:
    """What one continuous-effect record does, in the words Academy shows."""
    modification = effect.modification
    if modification == "ModifyPt":
        return f"{effect.power:+d}/{effect.toughness:+d}"
    if modification == "SetBasePt":
        power = "unchanged" if effect.power is None else effect.power
        toughness = "unchanged" if effect.toughness is None else effect.toughness
        return f"base power/toughness {power}/{toughness}"
    if modification == "AddKeywords":
        return ", ".join(keyword.replace("_", " ") for keyword in effect.keywords)
    if modification == "LoseKeywords":
        return "loses " + ", ".join(keyword.replace("_", " ") for keyword in effect.keywords)
    if modification == "AddSubtypes":
        return "is also " + ", ".join(effect.subtypes)
    phrases = {
        "BecomeArtifactCreature": "is an artifact creature",
        "LoseAllAbilities": "loses all abilities",
        "Firebending": f"firebending {effect.amount}",
        "CantBeBlocked": "can't be blocked",
        "CantBlock": "can't block",
        "DoesntUntap": "doesn't untap during its controller's untap step",
        "ExileIfWouldDie": "exiled instead if it would die",
        "KeepUnspentMana": f"keeps unspent {(effect.color or '').lower()} mana",
        "NoMaximumHandSize": "no maximum hand size",
        "ControlChanged": f"control changed from seat {(effect.amount or 0) + 1}",
        "ModeChosen": f"mode {(effect.amount or 0) + 1} already chosen",
    }
    try:
        return phrases[modification]
    except KeyError as error:
        raise TextEncodingError(f"unknown modification {modification}") from error


class _Renderer:
    def __init__(self, capture: DecisionInput) -> None:
        self.capture = capture
        self.state = managym.Observation.from_json(
            json.dumps(capture.observation.viewer_state)
        )
        self.names: dict[int, str] = {}
        self.seats: dict[int, str] = {}
        self.objects: dict[tuple[int, int], str] = {}
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

    def fact_reference(self, reference: managym.VisibleReference) -> str:
        if reference.player is not None:
            if reference.object is not None or reference.definition is not None:
                raise TextEncodingError("player reference carries object fields")
            return self.seats[reference.player]
        if reference.object is not None:
            if reference.definition is None:
                raise TextEncodingError("current object reference lacks its definition")
            return self.name(reference.object)
        if reference.definition is not None:
            return self.definition(reference.definition) + " (no current object link)"
        return "unavailable or expired object"

    def action(self, action: managym.Action) -> str:
        kind = _enum(_VERBS, int(action.action_type))
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
                "choose_trigger": "Put the triggered ability on the stack next",
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
        if action.continuation_index is not None:
            details.append(f"trigger context R{action.continuation_index + 1}")
        if details:
            label += "; " + "; ".join(details)
        if action.description:
            label += f". Native description: {action.description}"
        return label

    def relation(self, index: int, relation: managym.ContinuationFrame) -> str:
        source = self.fact_reference(relation.source)
        details = [
            f"R{index + 1}: {_enum(('unused', 'stack spell', 'stack activated ability', 'stack triggered ability', 'announcement', 'resolution', 'suspended decision', 'waiting trigger', 'trigger being placed', 'trigger placement round'), relation.kind)}",
            f"controller {self.seats[relation.controller]}",
            f"source {source}",
        ]
        if relation.parent is not None:
            details.append(f"parent R{relation.parent + 1}")
        details += _selectors(relation.mode, relation.ability)
        if relation.kicked:
            details.append("kicker included")
        if relation.active_clause is not None:
            details.append(f"active target clause {relation.active_clause + 1}")
        if relation.decision:
            details.append(
                _enum(
                    (
                        "none",
                        "resolving payment",
                        "mana activation cost",
                        "mana output",
                        "sacrifice",
                        "select milled",
                        "search",
                        "optional discard and draw",
                        "scry",
                        "inspect and select",
                        "mode",
                        "Learn",
                        "discard",
                        "legend rule",
                        "order triggers",
                    ),
                    relation.decision,
                )
            )
        if relation.cost is not None:
            details.append(
                "remaining cost "
                + _mana(
                    ManaCost(
                        cost=list(relation.cost),
                        mana_value=sum(relation.cost) + len(relation.hybrid),
                        hybrid=[list(pair) for pair in relation.hybrid],
                        spend_on_lesson=relation.lesson,
                    )
                )
            )
            details.append(
                f"waterbend {_yes(relation.waterbend)}; may decline {_yes(relation.can_decline)}"
            )
        for ordinal, target in enumerate(relation.targets, 1):
            legal = target.legal_at_resolution_start
            eligibility = "unknown" if legal is None else _yes(legal)
            details.append(
                f"target {ordinal}, clause {target.clause + 1}: {self.fact_reference(target.reference)}; "
                f"Legal when resolution began: {eligibility}"
            )
        if relation.context is not None:
            details.append("trigger context " + self.fact_reference(relation.context))
        if relation.sacrifice is not None:
            details.append(
                "reserved sacrifice " + self.fact_reference(relation.sacrifice)
            )
        if relation.resolutions_this_turn:
            details.append(
                f"ability resolutions this turn {relation.resolutions_this_turn}"
            )
        for output in relation.mana_outputs:
            for restricted, amounts in (
                (False, output.mana),
                (True, output.lesson),
            ):
                if any(amounts):
                    cost = ManaCost(
                        cost=[*amounts, 0],
                        mana_value=sum(amounts),
                        spend_on_lesson=restricted,
                    )
                    details.append(f"output mode {output.mode + 1}: {_mana(cost)}")
        return "; ".join(details) + "."

    def _players_lines(self) -> list[str]:
        state = self.state
        lines: list[str] = []
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
                "Restricted mana: "
                + ", ".join(
                    f"{symbol} {count}"
                    for symbol, count in zip("WUBRGC", player.lesson_mana, strict=True)
                )
                + f". Until-end-of-combat mana total: {player.combat_mana}.",
                f"Lessons in graveyard: {player.graveyard_lessons}.",
            ]
            for restricted in player.restricted_mana:
                lines.append(
                    f"Restricted mana for {_spending(restricted)}: "
                    + ", ".join(
                        f"{symbol} {count}"
                        for symbol, count in zip("WUBRGC", restricted.mana, strict=True)
                        if count
                    )
                    + "."
                )
            if player.hand_only_until is not None:
                lines.append(
                    "Can cast spells only from hand, overriding permissions to cast "
                    f"from elsewhere, until {self.name(player.hand_only_until)} "
                    "begins a turn."
                )
            if player.no_maximum_hand_size:
                lines.append("Has no maximum hand size for the rest of the game.")
            if player.keeps_red_mana:
                lines.append("Keeps unspent red mana until end of turn.")
            if history := _turn_history(player.turn_history):
                lines.append(f"This turn: {history}.")
            for effect in player.effects:
                lines.append(f"Effect: {self._effect(effect)}.")
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
        return lines

    def _cards_lines(self) -> list[str]:
        state = self.state
        lines: list[str] = []
        for card in state.agent_cards + state.opponent_cards:
            flags = ", ".join(
                key.removeprefix("is_").replace("_", " ")
                for key in (
                    "is_castable",
                    "is_permanent",
                    "is_non_land_permanent",
                    "is_non_creature_permanent",
                    "is_spell",
                    "is_creature",
                    "is_land",
                    "is_planeswalker",
                    "is_enchantment",
                    "is_artifact",
                    "is_kindred",
                    "is_battle",
                )
                if getattr(card.card_types, key)
            )
            optional = f"; type flags {flags}" if flags else ""
            if keywords := _keywords(card.keywords):
                optional += f"; printed keywords {keywords}"
            if card.face_down:
                optional += "; foretold, face down (hidden from other players)"
            if card.x is not None:
                optional += f"; announced X {card.x}"
            if card.cast_permission is not None:
                optional += "; " + self._cast_permission(card.cast_permission)
            lines.append(
                f"{self.name(card.id)} [{self.definition(card.registry_key)}]: owner {self.name(card.owner_id)}; zone {_enum(_ZONES, int(card.zone))}; card power/toughness {card.power}/{card.toughness}; cost {_mana(card.mana_cost)}; mana value {card.mana_cost.mana_value}{optional}; unlisted type and keyword flags false; token {_yes(card.is_token)}; legendary {_yes(card.is_legendary)}; Ally {_yes(card.is_ally)}; Lesson {_yes(card.is_lesson)}; ward mana value {card.ward_cost}; kicker mana value {card.kicker_cost}. Rules: {card.text_box or '(empty)'}"
            )
        if not state.agent_cards and not state.opponent_cards:
            lines.append("Empty.")
        return lines

    def _cast_permission(self, permission: managym.CastPermission) -> str:
        terms = [
            "as a land play"
            if permission.cost is None
            else f"cast for {_mana(permission.cost)}"
        ]
        if permission.this_turn:
            terms.append("this turn only")
        if permission.own_turn_only:
            terms.append("during their own turn only")
        if permission.after_turn is not None:
            terms.append(f"only after turn {permission.after_turn}")
        if permission.source_permanent_id is not None:
            terms.append(
                f"while they control {self.name(permission.source_permanent_id)}"
            )
        if permission.flash:
            terms.append("as though it had flash")
        if permission.any_mana:
            terms.append("spending mana of any type")
        if permission.waterbend:
            terms.append("generic may be paid by waterbending")
        return (
            f"may be played from exile by {self.name(permission.player_id)}: "
            + ", ".join(terms)
        )

    def _permanents_lines(self) -> list[str]:
        state = self.state
        lines: list[str] = []
        for permanent in state.agent_permanents + state.opponent_permanents:
            optional = ""
            if permanent.attacking:
                optional += "; attacking"
            if permanent.temp_subtypes:
                optional += f"; temporary subtypes {', '.join(permanent.temp_subtypes)}"
            if permanent.exhausted_abilities:
                optional += f"; exhaust abilities used {permanent.exhausted_abilities}"
            if permanent.once_each_turn_used:
                optional += (
                    "; once-each-turn abilities used this turn "
                    f"{permanent.once_each_turn_used}"
                )
            if permanent.granted_firebending:
                optional += (
                    f"; firebending {permanent.granted_firebending} until end of turn"
                )
            if keywords := _keywords(permanent.keywords):
                optional += f"; effective keywords {keywords}"
            optional += self._characteristics(permanent)
            lines.append(
                f"{self.name(permanent.id)} [{self.definition(permanent.registry_key)}]: controller {self.name(permanent.controller_id)}; power/toughness {permanent.power}/{permanent.toughness}; damage {permanent.damage}; tapped {_yes(permanent.tapped)}; summoning sick {_yes(permanent.is_summoning_sick)}; +1/+1 counters {permanent.plus1_counters}; animated {_yes(permanent.is_animated)}; holds exile link {_yes(permanent.has_exile_link)}; cannot be blocked this turn {_yes(permanent.cant_be_blocked_this_turn)}; exile if dies this turn {_yes(permanent.exile_if_dies_this_turn)}{optional}; unlisted effective keywords false."
            )
        if not state.agent_permanents and not state.opponent_permanents:
            lines.append("Empty.")
        return lines

    def _characteristics(self, permanent: managym.Permanent) -> str:
        """Effect-derived state, rendered only when present so other text is unchanged."""
        parts = [
            f"{kind.replace('_', ' ')} counters {count}"
            for kind, count in sorted(permanent.counters.items())
            if count
        ]
        if permanent.attached_to is not None:
            parts.append(f"attached to {self.name(permanent.attached_to)}")
        if permanent.name is not None:
            parts.append(f"name set to {permanent.name}")
        if permanent.creature_types is not None:
            types = ", ".join(permanent.creature_types) or "none"
            parts.append(f"creature types set to {types}")
        if permanent.is_granted_land:
            parts.append("land through an effect")
        if permanent.blocking is not None:
            parts.append(f"blocking {self.name(permanent.blocking)}")
        if permanent.abilities_lost:
            parts.append("has lost all abilities")
        if permanent.used_abilities:
            used = ", ".join(str(index + 1) for index in permanent.used_abilities)
            parts.append(f"used-up abilities {used}")
        if permanent.dealt_damage_this_turn:
            parts.append("dealt damage this turn")
        parts.extend(f"effect {self._effect(effect)}" for effect in permanent.effects)
        return "".join(f"; {part}" for part in parts)

    def _effect(self, effect: managym.ContinuousEffect) -> str:
        """One continuous-effect record: how long, what, and its source."""
        until = {
            "EndOfTurn": "until end of turn: ",
            "WhileYouControlSource": "while its source's controller controls it: ",
        }.get(effect.duration, "")
        source = (
            f" (from {self.definition(effect.source_registry_key)})"
            if effect.source_registry_key is not None
            else ""
        )
        return f"{until}{effect_phrase(effect)}{source}"

    def _definitions_lines(self) -> list[str]:
        capture = self.capture
        lines: list[str] = []
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
        return lines

    def render(self) -> tuple[str, tuple[TextAction, ...]]:
        capture, state = self.capture, self.state
        identity, frame = capture.observation, capture.observation.decision
        if frame is None:
            raise TextEncodingError("no acting decision")
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
        if len({offer["id"] for offer in frame.offers}) != len(frame.offers):
            raise TextEncodingError("duplicate offer identity")
        if any(card.zone == 1 for card in state.opponent_cards):
            raise TextEncodingError("opponent hand identity in capture")
        lines = [
            "You are choosing one legal action in the current game.",
            "Return only one listed action reference, such as A0. No other text.",
            "All legal alternatives are listed. Further choices occur only when offered.",
            "Optional details are shown when provided; unavailable facts are marked explicitly.",
            "",
            f"Turn {state.turn.turn_number}. Phase: {_enum(_PHASES, int(state.turn.phase))}. Step: {_enum(_STEPS, int(state.turn.step))}.",
            f"Active player: {self.name(state.turn.active_player_id)}. Acting player: {self.name(state.turn.agent_player_id)}.",
            f"Decision: {_enum(_SPACES, int(state.action_space.action_space_type))}. Game over: {_yes(state.game_over)}. Won: {_yes(state.won)}.",
            "",
            "Players:",
        ]
        lines.extend(self._players_lines())
        lines += [
            "Opponent's other hand identities and both library orders are unavailable.",
            "",
            "Visible cards (separate aliases for separate copies):",
        ]
        lines.extend(self._cards_lines())
        lines += ["", "Owned outside-game candidates:"]
        for card in state.agent_sideboard:
            lines.append(
                f"{self.name(card.candidate_id)} [{self.definition(card.registry_key)}]: owner {self.name(card.owner_id)}. Rules: {card.text_box or '(empty)'}"
            )
        if not state.agent_sideboard:
            lines.append("Empty.")
        lines += ["", "Battlefield effective state:"]
        lines.extend(self._permanents_lines())
        lines += ["", "Stack (native order, top to bottom):"]
        for index, item in enumerate(state.stack_objects):
            stack_frame = capture.frames[index]
            if stack_frame.kind != int(item.kind) + 1:
                raise TextEncodingError("stack/continuation order mismatch")
            # Shared native frames own exact target incarnations and clauses.
            targets = [
                f"clause {target.clause + 1}: {self.fact_reference(target.reference)}"
                for target in stack_frame.targets
            ]
            source = self.fact_reference(stack_frame.source)
            details = [
                f"controller {self.name(item.controller_id)}",
                f"source {source}",
            ]
            if item.description:
                details.append(item.description)
            details.extend(_selectors(item.chosen_mode, item.ability_index))
            if item.x:
                details.append(f"X = {item.x}")
            if item.granted_keywords is not None and (
                granted := _keywords(item.granted_keywords)
            ):
                details.append(f"granted keywords {granted}")
            if targets:
                details.append("targets " + ", ".join(targets))
            lines.append(
                f"S{index + 1}: {_enum(('spell', 'activated ability', 'triggered ability'), item.kind)} [{self.definition(item.source_card_registry_key)}]; {'; '.join(details)}."
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
        lines.extend(self._definitions_lines())
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
        if capture.frames:
            lines += ["", "Current stack and continuation relations:"]
            lines.extend(
                self.relation(index, relation)
                for index, relation in enumerate(capture.frames)
            )
        lines += ["", "Legal actions:"]
        actions: list[TextAction] = []
        for index, (offer, descriptor) in enumerate(
            zip(frame.offers, state.action_space.actions, strict=True)
        ):
            if (
                offer["actor"] != frame.actor
                or offer["choices"]
                or offer["source"] is not None
                or offer["help"] is not None
                or offer["verb"] != _enum(_VERBS, int(descriptor.action_type))
            ):
                raise TextEncodingError("unsupported or inconsistent native offer")
            description = self.action(descriptor)
            reference = f"A{index}"
            command = Command(
                f"text:{capture.digest}:{reference}", frame.revision, offer["id"]
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
