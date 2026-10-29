"""Alternative viewer-safe text encodings of one retained decision capture.

Every variant renders the same acting player's capture and lists the same legal
actions in the same order under the same references (A0..An). Variants differ
only in presentation, except ``library``, which adds one named fact.

- ``base``        the frozen `manabot-text-2` text, unchanged.
- ``labels``      base, with each mode and ability choice spelled out. This is
                  the floor: every variant below labels its choices too.
- ``systematic``  a cleaned-up, regular key/value layout.
- ``player``      the board as one Magic player would describe it to another.
- ``board``       a visual two-sided table-top layout.
- ``library``     systematic plus the acting player's own remaining library
                  composition (deck list minus cards seen outside the library).

Completeness rule: the three presentations carry every fact `manabot-text-2`
carries. A fact may be left unstated only when unstated means its default
(untapped, no damage, no counters, empty pool, not a token, owner = controller,
...). Meta statements (history capacity, "unavailable" placeholders) are
dropped. `_Facts` gathers each fact once; the formatters only arrange them.
`coverage_problems` checks the rule mechanically for one capture.

The task instruction and the action list are not repeated in the state text of
the three presentations: the typed Decisions request already carries both, as
its question and its choices.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
import re

from managym.decision_capture import CardReference, DecisionInput, PlayerReference

from manabot.semantic.text_observation import (
    _PHASES,
    _SPACES,
    _STEPS,
    _VERBS,
    _ZONES,
    _Renderer,
    _enum,
    _keywords,
    _mana,
    render_decision,
)


@dataclass(frozen=True)
class Encoded:
    text: str
    descriptions: dict[str, str]
    # The exact card-text section inside `text`; it changes only when a new
    # card becomes visible, so chat transports place it first and cache it.
    card_block: str | None = None


@dataclass(frozen=True)
class DeckContext:
    own: dict[str, int]
    opponent: dict[str, int]


_HISTORY_KINDS = ("Arrival", "Damage", "Life change", "Cast", "Resolved", "Countered")
_RELATION_KINDS = (
    "unused",
    "stack spell",
    "stack activated ability",
    "stack triggered ability",
    "announcement",
    "resolution",
    "suspended decision",
)
_RELATION_DECISIONS = (
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
)
_ZONE_WORDS = {
    "library": "the library",
    "hand": "hand",
    "battlefield": "the battlefield",
    "graveyard": "the graveyard",
    "stack": "the stack",
    "exile": "exile",
    "command": "the command zone",
}
_ACTIVATED = re.compile(r"^(\{[^}]+\}|Sacrifice|Tap|Exile|Discard|Pay)[^:]{0,80}:")
_NUMBERS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _modes(text: str) -> list[str]:
    return [line.lstrip("•").strip() for line in text.splitlines() if line.lstrip().startswith("•")]


def _abilities(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if _ACTIVATED.match(line.strip())]


def _possessive(label: str) -> str:
    return "Your" if label == "You" else f"{label}'s"


def _article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def _count_word(count: int) -> str:
    return _NUMBERS[count] if count < len(_NUMBERS) else str(count)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# -----------------------------------------------------------------------------
# Facts
# -----------------------------------------------------------------------------


@dataclass
class Permanent:
    alias: str
    name: str
    creature: bool
    power_toughness: str | None
    states: list[str]


@dataclass
class Side:
    label: str
    life: int
    counts: dict[str, int]
    extras: list[str]
    hand: list[str]
    permanents: list[Permanent]
    zones: dict[str, list[str]] = field(default_factory=dict)
    learnable: list[str] = field(default_factory=list)
    learned: list[str] = field(default_factory=list)
    known_hand: list[str] = field(default_factory=list)


class _Facts(_Renderer):
    """Every fact of one capture, each stated once, defaults omitted."""

    def __init__(self, capture: DecisionInput) -> None:
        super().__init__(capture)
        cards = self.state.agent_cards + self.state.opponent_cards
        self.cards = {card.id: card for card in cards}
        self.texts: dict[int, str] = {}
        for card in cards:
            self.texts[card.id] = card.text_box
            link = capture.objects[str(card.id)]
            if link.permanent_id is not None:
                self.texts[link.permanent_id] = card.text_box

    # -- choices --------------------------------------------------------------

    def choice_detail(self, source: int | None, mode: int | None, ability: int | None) -> str | None:
        text = self.texts.get(source, "") if source is not None else ""
        if mode is not None:
            modes = _modes(text)
            return f"mode {mode + 1}: {modes[mode]}" if mode < len(modes) else f"mode {mode + 1}"
        if ability is not None:
            abilities = _abilities(text)
            return f"ability {ability + 1}: {abilities[ability]}" if ability < len(abilities) else f"ability {ability + 1}"
        return None

    def labelled_action(self, action) -> str:
        label = self.action(action)
        detail = self.choice_detail(action.focus[0] if action.focus else None, action.mode_index, action.ability_index)
        if detail is None:
            return label
        return re.sub(r"; (mode|ability) \d+", "", label) + f" — {detail}"

    def actions(self) -> list[tuple[str, str]]:
        return [(f"A{i}", self.labelled_action(a)) for i, a in enumerate(self.state.action_space.actions)]

    # -- turn and decision ----------------------------------------------------

    def turn(self) -> tuple[int, bool, str]:
        turn = self.state.turn
        phase, step = _enum(_PHASES, int(turn.phase)), _enum(_STEPS, int(turn.step))
        where = phase if step in ("main", "second main") else f"{phase} phase, {step} step"
        return turn.turn_number, turn.active_player_id == self.state.agent.id, where

    def decision(self) -> list[str]:
        context = self.capture.continuation
        space = _enum(_SPACES, int(self.state.action_space.action_space_type))
        lines = [f"{space} decision", context.instruction.rstrip(".")]
        if context.source is not None:
            lines.append(f"for {self.decision_object(context.source)}")
        if context.chosen_mode is not None:
            source = getattr(context.source.subject, "id", None) if context.source else None
            entity = getattr(source, "entity", None)
            lines.append("chosen " + (self.choice_detail(entity, context.chosen_mode, None) or f"mode {context.chosen_mode + 1}"))
        if context.targets:
            lines.append("targets so far: " + ", ".join(self.decision_object(t.object) for t in context.targets))
        if context.reserved_sacrifice is not None:
            lines.append(f"sacrificing {self.decision_object(context.reserved_sacrifice)}")
        if context.mana_cost is not None:
            lines.append(f"mana still owed {context.mana_cost}")
        return lines

    # -- players --------------------------------------------------------------

    def _named(self, quantities: dict) -> list[str]:
        out = []
        for key, count in sorted(quantities.items(), key=lambda item: int(item[0])):
            name = self.capture.definitions[str(key)].name
            out.append(name if count == 1 else f"{name} ×{count}")
        return out

    def _card_states(self, card) -> list[str]:
        states = []
        if card.is_token:
            states.append("token")
        if card.ward_cost:
            states.append(f"ward {card.ward_cost}")
        if card.kicker_cost:
            states.append(f"kicker mana value {card.kicker_cost}")
        return states

    def _entry(self, card) -> str:
        states = self._card_states(card)
        return self.name(card.id) + (f" ({', '.join(states)})" if states else "")

    def permanent(self, permanent) -> Permanent:
        definition = self.capture.definitions[str(permanent.registry_key)]
        creature = "Creature" in definition.types.types or permanent.is_animated
        states: list[str] = []
        if keywords := _keywords(permanent.keywords):
            states.append(keywords)
        if permanent.tapped:
            states.append("tapped")
        if creature and permanent.is_summoning_sick:
            states.append("summoning sick")
        if permanent.damage:
            states.append(f"{permanent.damage} damage marked")
        if permanent.plus1_counters:
            states.append(f"{permanent.plus1_counters} +1/+1 counter{'s' if permanent.plus1_counters > 1 else ''}")
        if permanent.is_animated:
            states.append("animated")
        if permanent.temp_subtypes:
            states.append("also " + ", ".join(permanent.temp_subtypes) + " this turn")
        if permanent.has_exile_link:
            states.append("holding an exiled card")
        if permanent.cant_be_blocked_this_turn:
            states.append("can't be blocked this turn")
        if permanent.exile_if_dies_this_turn:
            states.append("exiled if it dies this turn")
        card = self.cards.get(self._card_id(permanent.id))
        if card is not None:
            states += self._card_states(card)
            if card.owner_id != permanent.controller_id:
                states.append(f"owned by {self.name(card.owner_id)}")
        alias = self.name(permanent.id)
        pt = f"{permanent.power}/{permanent.toughness}" if creature else None
        return Permanent(alias, definition.name, creature, pt, states)

    def _card_id(self, permanent_id: int) -> int | None:
        for card_id, link in self.capture.objects.items():
            if link.permanent_id == permanent_id:
                return int(card_id)
        return None

    def side(self, viewer: bool) -> Side:
        state = self.state
        player = state.agent if viewer else state.opponent
        cards = state.agent_cards if viewer else state.opponent_cards
        permanents = state.agent_permanents if viewer else state.opponent_permanents
        counts = dict(zip(_ZONES, player.zone_counts, strict=True))
        extras: list[str] = []
        if any(player.mana):
            extras.append("mana pool " + "".join(f"{{{s}}}" * n for s, n in zip("WUBRGC", player.mana, strict=True)))
        if any(player.lesson_mana):
            extras.append("Lesson-only mana " + "".join(f"{{{s}}}" * n for s, n in zip("WUBRGC", player.lesson_mana, strict=True)))
        if player.combat_mana:
            extras.append(f"{player.combat_mana} mana until end of combat")
        if player.graveyard_lessons:
            extras.append(f"{player.graveyard_lessons} Lesson{'s' if player.graveyard_lessons > 1 else ''} in graveyard")
        by_zone: dict[str, list[str]] = {}
        for card in cards:
            zone = _ZONES[int(card.zone)]
            if zone in ("battlefield", "stack"):
                continue  # described by permanents and the stack
            by_zone.setdefault(zone, []).append(self._entry(card))
        hand = by_zone.pop("hand", [])
        side = Side(
            label=self.name(player.id),
            life=player.life,
            counts=counts,
            extras=extras,
            hand=hand,
            permanents=[self.permanent(p) for p in permanents],
            zones=by_zone,
        )
        if viewer:
            side.learnable = [self.name(card.candidate_id) for card in state.agent_sideboard]
        else:
            side.learnable = self._named(player.remaining_sideboard_counts)
        learned = {
            key: count - player.remaining_sideboard_counts.get(key, 0)
            for key, count in player.sideboard_counts.items()
            if count - player.remaining_sideboard_counts.get(key, 0) > 0
        }
        side.learned = self._named(learned)
        side.known_hand = self._named(player.known_hand)
        return side

    # -- stack, relations and history ----------------------------------------

    def _relation_extras(self, relation) -> list[str]:
        details: list[str] = []
        if relation.parent is not None:
            details.append(f"part of R{relation.parent + 1}")
        if relation.kicked:
            details.append("kicked")
        if relation.active_clause is not None:
            details.append(f"choosing target clause {relation.active_clause + 1}")
        if relation.decision:
            details.append("pending " + _enum(_RELATION_DECISIONS, relation.decision))
        if relation.cost is not None:
            from managym.decision_capture import ManaCost

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
            if relation.waterbend:
                details.append("waterbend may pay")
            if relation.can_decline:
                details.append("may decline")
        if relation.context is not None:
            details.append("trigger context " + self.fact_reference(relation.context))
        if relation.sacrifice is not None:
            details.append("reserved sacrifice " + self.fact_reference(relation.sacrifice))
        if relation.resolutions_this_turn:
            details.append(f"resolved {relation.resolutions_this_turn}× this turn")
        for output in relation.mana_outputs:
            for restricted, amounts in ((False, output.mana), (True, output.lesson)):
                if any(amounts):
                    symbols = "".join(f"{{{s}}}" * n for s, n in zip("WUBRGC", amounts, strict=True))
                    details.append(f"output option {output.mode + 1}: {symbols}{' (Lesson-only)' if restricted else ''}")
        return details

    def stack(self) -> list[str]:
        lines = []
        for index, item in enumerate(self.state.stack_objects):
            frame = self.capture.frames[index]
            kind = ("spell", "activated ability", "triggered ability")[int(item.kind)]
            source = self.fact_reference(frame.source)
            text = f"{_possessive(self.name(item.controller_id))} {kind} from {source}"
            source_id = getattr(frame.source, "object", None)
            if (detail := self.choice_detail(source_id, item.chosen_mode, item.ability_index)) is not None:
                text += f" — {detail}"
            if frame.targets:
                text += " — targeting " + ", ".join(
                    f"{self.fact_reference(t.reference)}" + (f" (clause {t.clause + 1})" if len(frame.targets) > 1 else "")
                    for t in frame.targets
                )
            if extras := self._relation_extras(frame):
                text += " — " + "; ".join(extras)
            lines.append(text)
        return lines

    def relations(self) -> list[str]:
        """Continuation frames that are not stack objects (announcements, resolutions)."""
        lines = []
        for index in range(len(self.state.stack_objects), len(self.capture.frames)):
            frame = self.capture.frames[index]
            source = self.fact_reference(frame.source)
            source_id = getattr(frame.source, "object", None)
            text = f"R{index + 1}: {_possessive(self.seats[frame.controller])} {_enum(_RELATION_KINDS, frame.kind)} of {source}"
            if (detail := self.choice_detail(source_id, frame.mode, frame.ability)) is not None:
                text += f" — {detail}"
            for target in frame.targets:
                text += f" — target clause {target.clause + 1}: {self.fact_reference(target.reference)}"
            if extras := self._relation_extras(frame):
                text += " — " + "; ".join(extras)
            lines.append(text)
        return lines

    def _history_name(self, value) -> str:
        if isinstance(value, PlayerReference):
            return self.seats[value.seat]
        if isinstance(value, CardReference):
            if value.context_id is not None and value.context_id in self.names:
                return self.names[value.context_id]
            return self.capture.definitions[str(value.definition)].name
        return "an unknown source"

    def history(self) -> list[str]:
        lines: list[str] = []
        rows = self.capture.history.rows
        for index, row in enumerate(rows):
            kind = _enum(_HISTORY_KINDS, row.kind)
            source, target = self._history_name(row.source), self._history_name(row.target)
            owner = self.seats[row.source.owner] if isinstance(row.source, CardReference) else None
            if kind == "Arrival":
                zone = _ZONES[row.source.zone] if isinstance(row.source, CardReference) and row.source.zone is not None else None
                following = rows[index + 1] if index + 1 < len(rows) else None
                if (
                    zone == "stack"
                    and following is not None
                    and _enum(_HISTORY_KINDS, following.kind) == "Cast"
                    and isinstance(following.source, CardReference)
                    and following.source.definition == row.source.definition
                ):
                    continue  # the Cast row that follows says it
                lines.append(f"{_possessive(owner)} {source} → {_ZONE_WORDS.get(zone, 'an unknown zone')}")
            elif kind == "Cast":
                lines.append(f"{owner} cast {source}")
            elif kind == "Resolved":
                lines.append(f"{source} resolved")
            elif kind == "Countered":
                lines.append(f"{source} was countered")
            elif kind == "Damage":
                lines.append(f"{source} dealt {row.amount} damage to {target}")
            elif kind == "Life change":
                previous = rows[index - 1] if index else None
                if (
                    previous is not None
                    and _enum(_HISTORY_KINDS, previous.kind) == "Damage"
                    and previous.amount == -row.amount
                    and self._history_name(previous.target) == target
                ):
                    continue  # the damage row above already implies it
                lines.append(f"{target} {'gained' if row.amount > 0 else 'lost'} {abs(row.amount)} life")
        return lines

    # -- card text -------------------------------------------------------------

    def definition_lines(self) -> list[str]:
        lines = []
        for key in sorted(self.capture.definitions, key=int):
            definition = self.capture.definitions[key]
            types = " ".join(definition.supertypes + definition.types.types)
            if definition.subtypes:
                types += " — " + " ".join(definition.subtypes)
            parts = [definition.name]
            if definition.mana_cost is not None:
                parts.append(_mana(definition.mana_cost))
            parts.append(types)
            if definition.power is not None or definition.toughness is not None:
                parts.append(f"{definition.power if definition.power is not None else '*'}/{definition.toughness if definition.toughness is not None else '*'}")
            if definition.is_token:
                parts.append("token")
            if definition.kicker is not None:
                parts.append(f"kicker {_mana(definition.kicker)}")
            if definition.ward is not None:
                parts.append(f"ward {_mana(definition.ward)}")
            rules = (definition.text_box or "").replace("\n", " ")
            lines.append(" · ".join(parts) + (f": {rules}" if rules else ""))
        return lines

    def remaining_library(self, deck: dict[str, int]) -> tuple[Counter, int]:
        seen = Counter(
            card.name
            for card in self.cards.values()
            if card.owner_id == self.state.agent.id and not card.is_token and _ZONES[int(card.zone)] != "library"
        )
        remaining = Counter(deck)
        player = self.state.agent
        for key, count in player.sideboard_counts.items():
            learned = count - player.remaining_sideboard_counts.get(key, 0)
            if learned > 0:
                remaining[self.capture.definitions[str(key)].name] += learned
        for name, count in seen.items():
            remaining[name] = max(0, remaining[name] - count)
        return +remaining, dict(zip(_ZONES, player.zone_counts, strict=True))["library"]


def _zone_count_note(side: Side, zone: str) -> str:
    """State a zone's size when its listing does not already show it."""
    listed = len(side.zones.get(zone, [])) if zone != "hand" else len(side.hand)
    return "" if listed == side.counts[zone] else f" ({side.counts[zone]} cards)"


# -----------------------------------------------------------------------------
# Presentations
# -----------------------------------------------------------------------------


def systematic(capture: DecisionInput, decks: DeckContext | None = None) -> Encoded:
    facts = _Facts(capture)
    number, mine, where = facts.turn()
    lines = [f"Turn {number} · {'your' if mine else 'opponent'} turn · {where}", ""]
    for viewer in (True, False):
        side = facts.side(viewer)
        header = [f"{side.life} life", f"library {side.counts['library']}"]
        if not viewer:
            header.insert(1, f"hand {side.counts['hand']}")
        lines.append(f"{side.label.upper()}: " + ", ".join(header + side.extras))
        if viewer:
            lines.append(f"Hand{_zone_count_note(side, 'hand')}: " + ("; ".join(side.hand) or "empty"))
        if side.known_hand:
            who = "Opponent knows you hold" if viewer else "Known in hand"
            lines.append(f"{who}: " + ", ".join(side.known_hand))
        lines.append("Battlefield:" + ("" if side.permanents else " empty"))
        for p in side.permanents:
            extra = ", ".join(([p.power_toughness] if p.power_toughness else []) + p.states)
            lines.append(f"- {p.alias}" + (f" [{extra}]" if extra else ""))
        for zone in ("graveyard", "exile", "command", "library"):
            entries = side.zones.get(zone, [])
            if zone == "library":
                if entries:
                    lines.append("Known cards in library: " + "; ".join(entries))
                continue
            note = _zone_count_note(side, zone)
            if entries or note:
                lines.append(f"{zone.capitalize()}{note}: " + "; ".join(entries))
        if side.learnable:
            lines.append("Can still Learn: " + "; ".join(side.learnable))
        if side.learned:
            lines.append("Already Learned: " + ", ".join(side.learned))
        if viewer and decks is not None:
            remaining, library = facts.remaining_library(decks.own)
            lines.append(
                f"Your library's {library} cards: "
                + ", ".join(f"{n} ×{c}" if c > 1 else n for n, c in sorted(remaining.items()))
            )
        lines.append("")
    if stack := facts.stack():
        lines += ["STACK (top first):", *(f"{i + 1}. {s}" for i, s in enumerate(stack)), ""]
    if relations := facts.relations():
        lines += ["IN PROGRESS:", *relations, ""]
    card_block = "\n".join(["CARD TEXT:", *facts.definition_lines()])
    lines += [card_block, ""]
    if history := facts.history():
        lines += ["RECENT EVENTS (oldest first):", *history, ""]
    lines.append("DECISION: " + "; ".join(facts.decision()))
    return Encoded("\n".join(lines) + "\n", dict(facts.actions()), card_block)


def _describe_permanents(permanents: list[Permanent]) -> str:
    """Group identical permanents: 'two tapped Forests (C22, C24)'."""
    groups: dict[tuple[str, tuple[str, ...], str | None], list[str]] = {}
    for p in permanents:
        groups.setdefault((p.name, tuple(p.states), p.power_toughness), []).append(p.alias.split(" ")[0])
    parts = []
    for (name, states, pt), aliases in groups.items():
        count = len(aliases)
        noun = name if count == 1 else name + ("es" if name.endswith(("s", "sh", "ch")) else "s")
        adjectives = ["tapped" if "tapped" in states else "untapped"]
        if pt:
            adjectives.append(pt)
        phrase = " ".join([*adjectives, noun])
        phrase = _article(phrase) if count == 1 else f"{_count_word(count)} {phrase}"
        rest = [s for s in states if s != "tapped"]
        parts.append(f"{phrase} ({', '.join(aliases)})" + (f" — {', '.join(rest)}" if rest else ""))
    return "; ".join(parts)


def player(capture: DecisionInput, decks: DeckContext | None = None) -> Encoded:
    facts = _Facts(capture)
    number, mine, where = facts.turn()
    me, them = facts.side(True), facts.side(False)
    paragraphs = []
    whose = "It's your turn" if mine else "It's your opponent's turn"
    paragraphs.append(f"{whose} — turn {number}, {where}.")

    you = [f"You're on {me.life} life with {me.counts['library']} cards left in your library"]
    if me.extras:
        you[0] += f" ({'; '.join(me.extras)})"
    you[0] += "."
    if me.hand:
        you.append(f"In hand you have {_join(me.hand)}{_zone_count_note(me, 'hand').replace('(', '— ').replace(')', ' in total')}.")
    else:
        you.append("Your hand is empty.")
    you.append(f"On your side of the battlefield: {_describe_permanents(me.permanents)}." if me.permanents else "You have no permanents.")
    for zone, word in (("graveyard", "Your graveyard has"), ("exile", "You have in exile"), ("command", "In your command zone")):
        if me.zones.get(zone) or _zone_count_note(me, zone):
            you.append(f"{word} {_join(me.zones.get(zone, [])) or 'cards'}{_zone_count_note(me, zone)}.")
    if me.zones.get("library"):
        you.append(f"You know these cards are in your library: {_join(me.zones['library'])}.")
    if me.known_hand:
        you.append(f"Your opponent knows you're holding {_join(me.known_hand)}.")
    if me.learnable:
        you.append(f"Lessons you could still Learn: {_join(me.learnable)}.")
    if me.learned:
        you.append(f"You've already Learned {_join(me.learned)}.")
    paragraphs.append(" ".join(you))

    opp = [f"Your opponent is on {them.life} life with {_count_word(them.counts['hand'])} card{'s' if them.counts['hand'] != 1 else ''} in hand and {them.counts['library']} in library"]
    if them.extras:
        opp[0] += f" ({'; '.join(them.extras)})"
    opp[0] += "."
    if them.known_hand:
        opp.append(f"You know they're holding {_join(them.known_hand)}.")
    opp.append(f"They have {_describe_permanents(them.permanents)}." if them.permanents else "They have no permanents.")
    for zone, word in (("graveyard", "Their graveyard has"), ("exile", "They have in exile"), ("command", "In their command zone")):
        if them.zones.get(zone) or _zone_count_note(them, zone):
            opp.append(f"{word} {_join(them.zones.get(zone, [])) or 'cards'}{_zone_count_note(them, zone)}.")
    if them.zones.get("library"):
        opp.append(f"Known cards in their library: {_join(them.zones['library'])}.")
    if them.learnable:
        opp.append(f"They could still Learn {_join(them.learnable)}.")
    if them.learned:
        opp.append(f"They've already Learned {_join(them.learned)}.")
    paragraphs.append(" ".join(opp))

    if stack := facts.stack():
        paragraphs.append("On the stack, top first: " + "; then ".join(stack) + ".")
    if relations := facts.relations():
        paragraphs.append("Still being worked out: " + "; ".join(relations) + ".")
    if history := facts.history():
        paragraphs.append("What just happened, oldest first: " + "; ".join(history) + ".")
    paragraphs.append("Right now you need to: " + "; ".join(facts.decision()) + ".")
    card_block = "For reference, the cards in this game:\n" + "\n".join(facts.definition_lines())
    paragraphs.append(card_block)
    return Encoded("\n\n".join(paragraphs) + "\n", dict(facts.actions()), card_block)


def board(capture: DecisionInput, decks: DeckContext | None = None) -> Encoded:
    facts = _Facts(capture)
    number, mine, where = facts.turn()
    me, them = facts.side(True), facts.side(False)

    def tile(p: Permanent) -> str:
        bits = [p.alias.split(" ")[0], p.name] + ([p.power_toughness] if p.power_toughness else [])
        flags = [{"tapped": "T", "summoning sick": "S"}.get(s, s) for s in p.states]
        return "[" + " ".join(bits) + (" | " + ", ".join(flags) if flags else "") + "]"

    def rows(side: Side) -> tuple[str, str]:
        creatures = "  ".join(tile(p) for p in side.permanents if p.creature) or "—"
        other = "  ".join(tile(p) for p in side.permanents if not p.creature) or "—"
        return creatures, other

    def header(side: Side, viewer: bool) -> str:
        bits = [f"♥ {side.life}", f"library {side.counts['library']}"]
        if not viewer:
            bits.insert(1, f"hand {side.counts['hand']}")
        return " · ".join(bits + side.extras)

    def zones(side: Side) -> list[str]:
        out = []
        for zone in ("graveyard", "exile", "command"):
            if side.zones.get(zone) or _zone_count_note(side, zone):
                out.append(f"│ {zone.capitalize():<10}│ {'  '.join(side.zones.get(zone, []))}{_zone_count_note(side, zone)}")
        if side.zones.get("library"):
            out.append(f"│ {'In library':<10}│ {'  '.join(side.zones['library'])}")
        if side.learnable:
            out.append(f"│ {'Learnable':<10}│ {'  '.join(side.learnable)}")
        if side.learned:
            out.append(f"│ {'Learned':<10}│ {'  '.join(side.learned)}")
        return out

    their_creatures, their_other = rows(them)
    my_creatures, my_other = rows(me)
    lines = [
        f"Turn {number} · {'your' if mine else 'opponent'} turn · {where}",
        "Legend: T = tapped, S = summoning sick; unmarked permanents are untapped.",
        "",
        f"┌─ OPPONENT  {header(them, False)}",
    ]
    if them.known_hand:
        lines.append(f"│ {'Known hand':<10}│ {'  '.join(them.known_hand)}")
    lines += zones(them)
    lines += [
        f"│ {'Lands etc.':<10}│ {their_other}",
        f"│ {'Creatures':<10}│ {their_creatures}",
        "├" + "─" * 30 + " battlefield " + "─" * 30,
        f"│ {'Creatures':<10}│ {my_creatures}",
        f"│ {'Lands etc.':<10}│ {my_other}",
    ]
    lines += zones(me)
    lines.append(f"│ {'Hand':<10}│ {'  '.join(me.hand) or '—'}{_zone_count_note(me, 'hand')}")
    if me.known_hand:
        lines.append(f"│ {'Opp. knows':<10}│ {'  '.join(me.known_hand)}")
    lines.append(f"└─ YOU  {header(me, True)}")
    if stack := facts.stack():
        lines += ["", "STACK (top first):", *(f"{i + 1}. {s}" for i, s in enumerate(stack))]
    if relations := facts.relations():
        lines += ["", "IN PROGRESS:", *relations]
    if history := facts.history():
        lines += ["", "RECENT EVENTS (oldest first):", *history]
    card_block = "\n".join(["CARD TEXT:", *facts.definition_lines()])
    lines += ["", card_block, "", "DECISION: " + "; ".join(facts.decision())]
    return Encoded("\n".join(lines) + "\n", dict(facts.actions()), card_block)


def base(capture_json: str) -> Encoded:
    decision = render_decision(capture_json)
    return Encoded(decision.text, {a.reference: a.description for a in decision.actions})


def labels(capture_json: str) -> Encoded:
    decision = render_decision(capture_json)
    labelled = dict(_Facts(DecisionInput.model_validate_json(capture_json)).actions())
    text = decision.text
    for action in decision.actions:
        text = text.replace(f"{action.reference} — {action.description}.", f"{action.reference} — {labelled[action.reference]}.")
    return Encoded(text, labelled)


# -----------------------------------------------------------------------------
# Completeness check
# -----------------------------------------------------------------------------


def coverage_problems(capture_json: str, encoded: Encoded) -> list[str]:
    """Facts `manabot-text-2` states that ``encoded`` fails to state."""
    capture = DecisionInput.model_validate_json(capture_json)
    facts = _Facts(capture)
    text = encoded.text
    problems: list[str] = []
    base_text = render_decision(capture_json).text
    for alias in sorted(set(re.findall(r"\b[CO]\d+\b", base_text))):
        if not re.search(rf"\b{alias}\b", text):
            problems.append(f"alias {alias} missing")
    for viewer in (True, False):
        side = facts.side(viewer)
        if str(side.life) not in text:
            problems.append(f"{side.label} life missing")
        if str(side.counts["library"]) not in text:
            problems.append(f"{side.label} library count missing")
        for p in side.permanents:
            alias = p.alias.split(" ")[0]
            for state in p.states + ([p.power_toughness] if p.power_toughness else []):
                token = {"tapped": "T", "summoning sick": "S"}.get(state, state)
                at = text.find(alias)
                window = text[max(0, at - 80):at + 200]
                if state not in window and f" {token}" not in window and token not in window:
                    problems.append(f"{alias} state '{state}' missing")
        for name in side.known_hand + side.learned:
            if name.split(" ×")[0] not in text:
                problems.append(f"{side.label} {name} missing")
    for definition in capture.definitions.values():
        if definition.text_box and definition.text_box.replace("\n", " ")[:40] not in text.replace("\n", " "):
            problems.append(f"rules text of {definition.name} missing")
    for state_line in facts.stack() + facts.relations():
        if state_line[:30] not in text:
            problems.append("stack/relation line missing")
    for event in facts.history():
        if event not in text:
            problems.append(f"event '{event}' missing")
    context = capture.continuation
    if context.instruction.rstrip(".") not in text:
        problems.append("decision instruction missing")
    space = _enum(_SPACES, int(facts.state.action_space.action_space_type))
    if f"{space} decision" not in text:
        problems.append("decision type missing")
    return problems


Variant = Callable[[str, DeckContext | None], Encoded]

VARIANTS: dict[str, Variant] = {
    "base": lambda c, d: base(c),
    "labels": lambda c, d: labels(c),
    "systematic": lambda c, d: systematic(DecisionInput.model_validate_json(c)),
    "player": lambda c, d: player(DecisionInput.model_validate_json(c)),
    "board": lambda c, d: board(DecisionInput.model_validate_json(c)),
    "library": lambda c, d: systematic(DecisionInput.model_validate_json(c), d),
}


# -----------------------------------------------------------------------------
# Typed-choice ablation and chat form
# -----------------------------------------------------------------------------


def with_actions(encoded: Encoded) -> Encoded:
    """The same encoding with its labelled action list restated in the text."""
    actions = "\n".join(f"{ref} — {desc}" for ref, desc in encoded.descriptions.items())
    return Encoded(encoded.text + "\nLEGAL ACTIONS:\n" + actions + "\n", encoded.descriptions, encoded.card_block)


def chat_form(encoded: Encoded) -> tuple[str, int | None]:
    """Card text first (a cacheable prefix), then state, then the action list.

    Chat requests carry no typed choices, so the labelled actions must be in
    the text. Returns the text and the cacheable prefix length in characters.
    """
    full = with_actions(encoded)
    if encoded.card_block is None:
        return full.text, None
    rest = full.text.replace(encoded.card_block, "", 1).lstrip("\n").replace("\n\n\n", "\n\n")
    prefix = encoded.card_block + "\n\n"
    return prefix + rest, len(prefix)


for _name in ("systematic", "player", "board"):
    VARIANTS[f"{_name}+actions"] = (lambda inner: lambda c, d: with_actions(inner(c, d)))(VARIANTS[_name])


_LEGEND = (
    "Unlisted details are defaults: permanents untapped, undamaged, no counters, "
    "not summoning sick; mana pools empty; no other effects."
)


def with_legend(encoded: Encoded) -> Encoded:
    """State the defaults convention and an empty stack explicitly."""
    text = encoded.text
    if "STACK" not in text and "On the stack" not in text:
        marker = "CARD TEXT:" if "CARD TEXT:" in text else "For reference, the cards"
        text = text.replace(marker, "STACK: empty.\n\n" + marker, 1)
    return Encoded(_LEGEND + "\n" + text, encoded.descriptions, encoded.card_block)


for _name in ("systematic", "player", "board"):
    VARIANTS[f"{_name}+legend"] = (lambda inner: lambda c, d: with_legend(inner(c, d)))(VARIANTS[_name])
    VARIANTS[f"{_name}+legend+actions"] = (lambda inner: lambda c, d: with_actions(with_legend(inner(c, d))))(VARIANTS[_name])


_DEFAULT_FIELDS = (
    r"; unlisted type and keyword flags false",
    r"; unlisted effective keywords false",
    r"; unlisted keywords false",
    r"; token no",
    r"; legendary no",
    r"; Ally no",
    r"; Lesson no",
    r"; ward mana value 0",
    r"; kicker mana value 0",
    r"; damage 0",
    r"; tapped no",
    r"; summoning sick no",
    r"; \+1/\+1 counters 0",
    r"; animated no",
    r"; holds exile link no",
    r"; cannot be blocked this turn no",
    r"; exile if dies this turn no",
    r"; card power/toughness 0/0",
    r"; power/toughness 0/0",
    r"; (?:stack|exile|command) 0",
    r" Game over: no\. Won: no\.",
)


def trimmed(capture_json: str) -> Encoded:
    """`labels` in the frozen layout, with always-default fields removed."""
    encoded = labels(capture_json)
    text = encoded.text
    for pattern in _DEFAULT_FIELDS:
        text = re.sub(pattern, "", text)
    text = re.sub(r"(Mana pool|Lesson-only mana): W 0, U 0, B 0, R 0, G 0, C 0\.( Until-end-of-combat mana total: 0\.)?\n", "", text)
    text = re.sub(r"Until-end-of-combat mana total: 0\.", "", text)
    text = re.sub(r"Lessons in graveyard: 0\.\n", "", text)
    text = re.sub(r"(Remaining sideboard|Publicly known hand minima): empty\.\n", "", text)
    return Encoded(text, encoded.descriptions)


VARIANTS["trimmed"] = lambda c, d: trimmed(c)


def without_history(capture_json: str) -> str:
    """The capture as `history_capacity=0` would record it, re-digested."""
    from hashlib import sha256
    import json

    payload = json.loads(capture_json)
    payload["history"] = {**payload["history"], "capacity": 0, "rows": []}
    del payload["digest"]
    payload["digest"] = sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return json.dumps(payload)


for _name in list(VARIANTS):
    if "@" not in _name:
        VARIANTS[f"{_name}@nohist"] = (lambda inner: lambda c, d: inner(without_history(c), d))(VARIANTS[_name])
