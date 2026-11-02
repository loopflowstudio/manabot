"""Rule-of-thumb arena anchor, `heuristic`.

`ScriptedGreedyPlayer` (v1) plays a land, casts the first castable spell,
attacks and blocks with everything, and takes the first target offer. Target
offers arrive in absolute engine order (player 0's objects first), so v1 picks
a target side by seat: from seat 0 it burns itself with Igneous Inspiration.

`HeuristicPlayer` keeps v1 for every prompt it has no rule for and adds:

- Mana: play a land, crack fetch lands, then cast the castable spell with the
  highest mana value that has a worthwhile use, one spell at a time (its own
  spell resolves before the next). The engine only offers affordable casts, so
  repeated highest-first casting spends as much mana as the offers allow.
  Kicker is paid when untapped lands cover spell plus kicker.
- Instant timing: nothing in its own upkeep or draw step. Instants without a
  target that matters now (card draw, selection) and draw or life abilities
  wait for the opponent's end step, so the turn's mana goes to permanents first.
- Targets: effect polarity is derived from the engine's card definitions
  (`managym.card_definitions_json`: damage, destroy, bounce, counter and exile
  are harmful; pumps, untaps and keyword grants are beneficial). Harmful
  effects go to the opponent's side, beneficial ones to its own. Removal is
  held until it kills something (with the damage the available mana and
  graveyard Lessons allow), burn that can hit players until it kills a creature
  or the opponent, counters until the opponent has a spell on the stack.
- Card choices: Learn takes a Lesson (removal while the opponent has creatures);
  discards are spare lands or else the costliest card, optional ones declined;
  a look-and-select keeps a land while short of four.
- Combat: hold back the toughest attackers while the opponent's swing back
  would be lethal; attack with a creature unless the opponent has an untapped
  creature that can block it, kill it and survive, or with everything when
  unblockable damage is lethal. Blocks are planned once per combat, one
  blocker per attacker, largest attackers first: kill and survive, trade up,
  absorb for free; then chump the largest attackers while the rest is lethal.
- Own combat tricks: a pump in hand with mana to cast it lets one dominated
  creature attack anyway. After blocks, pumps are cast only where they change
  a one-on-one fight or make unblocked damage lethal, and self-pump abilities
  go on unblocked attackers. The opponent's tricks are ignored: the Elemental
  lists hold too few for a hidden-hand estimate to change a decision.
- Optional costs (ward, "unless pays"): pay when offered.

Combat assignments come from the observation's attacker/blocker events, whose
object IDs are the current permanent IDs; the player remembers them for the
turn. Mana is approximated by untapped lands, ignoring colors and other
sources. Creature identity uses the visible card definitions; a permanent
without one counts as a creature when its toughness is positive.

This lives outside `anchors.py` because arena registrations pin that file's
bytes as the v1 source identity; v1 stays frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path
import re
from typing import Any, Literal

from managym.enums import ActionEnum, ActionSpaceEnum, EventTypeEnum, StepEnum, ZoneEnum
import numpy as np

from manabot.env.env import Env
from manabot.sim.anchors import ScriptedGreedyPlayer
import managym

Polarity = Literal["harmful", "beneficial"]

# Effects that act on a spell's target, by what they do to that target's
# controller. Names are the engine's `Effect` variants.
HARMFUL_EFFECTS = frozenset(
    {
        "DealDamage",
        "DealDamageValue",
        "TargetCreaturesDealPowerDamageToLastTarget",
        "Destroy",
        "Exile",
        "ExileWithControllerToken",
        "ExileUntilSourceLeaves",
        "ReturnToHand",
        "ReturnToHandDrawIfControlled",
        "CounterSpell",
        "CounterUnlessPays",
        "DiscardTargetPlayer",
    }
)
BENEFICIAL_EFFECTS = frozenset(
    {
        "UntapTarget",
        "GrantKeywordsToTarget",
        "Earthbend",
        "BuffTargetValue",
        "CantBeBlockedTarget",
        "AddSubtypeToTarget",
    }
)
# Effects with no `target` field that still act on the spell's target.
IMPLICIT_TARGET_EFFECTS = frozenset(
    {"CounterUnlessPays", "TargetCreaturesDealPowerDamageToLastTarget"}
)

# ---------------------------------------------------------------------------
# Card knowledge derived from the engine card definitions


@dataclass(frozen=True)
class TargetedSpell:
    """What the player needs to know about one targeted card program."""

    polarity: Polarity | None
    selector: str
    # Every target selector already restricts the side (`*_you_control`,
    # `*_opponent_controls`); offer order then cannot pick the wrong side.
    side_restricted: bool
    # Damage dealt to the target without and with kicker; 0 for no damage.
    damage: int = 0
    kicked_damage: int = 0
    # Damage also grows by one per Lesson in the caster's graveyard.
    lesson_damage: bool = False
    # Power/toughness added to each target until end of turn.
    pump: tuple[int, int] = (0, 0)


Effect = dict[str, Any] | str


# A continuous effect on the target, by its modification, read as the
# targeted effect it amounts to.
_TARGET_MODIFICATIONS = {
    "AddKeywords": "GrantKeywordsToTarget",
    "CantBeBlocked": "CantBeBlockedTarget",
    "AddSubtypes": "AddSubtypeToTarget",
}


def _variant(effect: Effect) -> tuple[str, Any]:
    """An effect's variant name and body (unit variants serialize as strings).

    `ApplyContinuous` reads as what it does to whom: a fixed P/T change on the
    target is `BuffTarget`, any other on the target `BuffTargetValue`, a
    keyword, subtype or unblockability grant to the target its `*Target`
    name, and a P/T change on the source `ModifyUntilEot`. Other continuous
    effects keep their name and carry no polarity.
    """
    if isinstance(effect, str):
        return effect, {}
    ((name, body),) = effect.items()
    if name != "ApplyContinuous":
        return name, body
    affected = body["affected"]
    modification, change = _variant(body["modification"])
    if modification == "ModifyPt":
        power, toughness = change["power"], change["toughness"]
        fixed = "Fixed" in power and "Fixed" in toughness
        if affected == "Source" and fixed:
            return "ModifyUntilEot", {
                "power_delta": power["Fixed"],
                "toughness_delta": toughness["Fixed"],
            }
        if isinstance(affected, dict) and "Target" in affected:
            if fixed:
                return "BuffTarget", {
                    "power": power["Fixed"],
                    "toughness": toughness["Fixed"],
                    "target": affected["Target"],
                }
            return "BuffTargetValue", {"target": affected["Target"]}
    if isinstance(affected, dict) and "Target" in affected:
        return _TARGET_MODIFICATIONS.get(modification, name), {
            "target": affected["Target"]
        }
    return name, body


def _children(name: str, body: Any) -> list[list[Effect]]:
    """Nested effect lists of one effect. A modal spell's side depends on the
    chosen mode, so mode bodies stay opaque to polarity and damage."""
    if name == "Modal" or not isinstance(body, dict):
        return []
    return [
        value
        for value in body.values()
        if isinstance(value, list) and value and isinstance(value[0], (dict, str))
    ]


def _target_effects(effects: list[Effect]) -> list[tuple[str, Any]]:
    """Every effect, at any depth outside modes, that acts on the target."""
    found: list[tuple[str, Any]] = []
    for effect in effects:
        name, body = _variant(effect)
        for nested in _children(name, body):
            found.extend(_target_effects(nested))
        if name in IMPLICIT_TARGET_EFFECTS or (
            isinstance(body, dict) and "target" in body
        ):
            found.append((name, body))
    return found


def _specs(effects: list[Effect]) -> list[Any]:
    """Target specs in program order, including those inside modes."""
    specs: list[Any] = []
    for effect in effects:
        name, body = _variant(effect)
        if not isinstance(body, dict):
            continue
        if name == "TargetClause":
            specs.append(body["requirement"]["spec"])
        elif "target" in body:
            specs.append(body["target"])
        for key, value in body.items():
            if key == "modes":
                for mode in value:
                    specs.extend(_specs(mode))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                specs.extend(_specs(value))
    return specs


def _selector(spec: Any) -> str:
    """A target spec's variant in snake case: `CreatureOrPlayer` is
    `creature_or_player`."""
    name = spec if isinstance(spec, str) else next(iter(spec))
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _damage(effects: list[Effect], kicked: bool) -> int:
    """Largest damage to the target, following kicker branches as given."""
    best = 0
    for effect in effects:
        name, body = _variant(effect)
        if name == "If" and body["condition"] == "Kicked":
            branch = body["then"] if kicked else body.get("otherwise", [])
            best = max(best, _damage(branch, kicked))
            continue
        for nested in _children(name, body):
            best = max(best, _damage(nested, kicked))
        if name == "DealDamage":
            best = max(best, body["amount"])
        elif name == "DealDamageValue" and "GraveyardMatching" in body["amount"]:
            best = max(best, body["amount"]["GraveyardMatching"]["base"])
    return best


def _counts_lessons(amount: dict[str, Any]) -> bool:
    """Whether a damage amount grows per Lesson in the caster's graveyard."""
    predicate = amount.get("GraveyardMatching", {}).get("predicate", {})
    return predicate.get("subtype") == "Lesson" and not any(
        value for key, value in predicate.items() if key != "subtype"
    )


def _effect_polarity(name: str, body: Any) -> Polarity | None:
    if name in HARMFUL_EFFECTS:
        return "harmful"
    if name in BENEFICIAL_EFFECTS:
        return "beneficial"
    if name == "BuffTarget":
        change = body["power"] + body["toughness"]
        return "beneficial" if change > 0 else "harmful" if change < 0 else None
    return None


def derive_targeted_spells(
    definitions: list[dict[str, Any]],
) -> dict[str, TargetedSpell]:
    """Map card name to its targeted spell, from engine definitions.

    A spell whose target effects are unknown or of mixed polarity gets polarity
    None, and the player falls back to v1 for it.
    """
    spells: dict[str, TargetedSpell] = {}
    for definition in definitions:
        effects = definition.get("spell_effects", [])
        specs = [row["spec"] for row in definition.get("targeting", [])]
        specs = specs or _specs(effects)
        if not specs:
            continue
        targeted = _target_effects(effects)
        # Effects with no polarity (such as "exile if it dies") ride along with
        # the effect that decides the side.
        polarities = {_effect_polarity(*effect) for effect in targeted} - {None}
        pumps = [body for name, body in targeted if name == "BuffTarget"]
        spells[definition["name"]] = TargetedSpell(
            polarity=polarities.pop() if len(polarities) == 1 else None,
            selector=_selector(specs[0]),
            side_restricted=all(
                _selector(spec).endswith(("_you_control", "_opponent_controls"))
                for spec in specs
            ),
            damage=_damage(effects, kicked=False),
            kicked_damage=_damage(effects, kicked=True),
            lesson_damage=any(
                name == "DealDamageValue" and _counts_lessons(body["amount"])
                for name, body in targeted
            ),
            pump=(
                sum(body["power"] for body in pumps),
                sum(body["toughness"] for body in pumps),
            ),
        )
    return spells


def heuristic_source_paths() -> list[Path]:
    """Files whose bytes define this player's behavior, for arena identity.

    Its card knowledge comes from the world's engine definitions, which the
    world version and content digest identify.
    """
    sim = Path(__file__).resolve().parent
    return [Path(__file__).resolve(), sim / "anchors.py"]


@cache
def _definitions() -> list[dict[str, Any]]:
    return json.loads(managym.card_definitions_json())


@cache
def instant_names() -> frozenset[str]:
    """Card names whose definition is an instant."""
    return frozenset(
        d["name"] for d in _definitions() if "Instant" in d["types"]["types"]
    )


@dataclass(frozen=True)
class Abilities:
    """Card names by when their activated ability is worth its cost."""

    # As soon as offered: fetch lands.
    prompt: frozenset[str]
    # With spare mana at the opponent's end step: draw, life.
    spare: frozenset[str]
    # On an unblocked attacker after blocks: self-pumps.
    pump: frozenset[str]


@cache
def abilities() -> Abilities:
    prompt: set[str] = set()
    spare: set[str] = set()
    pump: set[str] = set()
    for definition in _definitions():
        name = definition["name"]
        for ability in definition.get("activated_abilities", []):
            if ability.get("zone", "Battlefield") != "Battlefield":
                continue
            for effect in ability["effects"]:
                kind, body = _variant(effect)
                if kind == "SearchBasicLand":
                    prompt.add(name)
                elif kind in {"DrawCards", "GainLife"}:
                    spare.add(name)
                elif kind == "ModifyUntilEot" and body["power_delta"] > 0:
                    pump.add(name)
    return Abilities(frozenset(prompt), frozenset(spare), frozenset(pump))


@cache
def targeted_spells() -> dict[str, TargetedSpell]:
    return derive_targeted_spells(_definitions())


# ---------------------------------------------------------------------------
# Board reading


@dataclass(frozen=True)
class _Body:
    """A creature's combat-relevant numbers, optionally pumped."""

    power: int
    toughness: int
    damage: int
    keywords: managym.Keywords

    @classmethod
    def of(cls, permanent: managym.Permanent, pump: tuple[int, int] = (0, 0)) -> _Body:
        return cls(
            permanent.power + pump[0],
            permanent.toughness + pump[1],
            permanent.damage,
            permanent.keywords,
        )


@dataclass(frozen=True)
class _Offers:
    action_space_type: int
    actions: list[managym.Action]
    focus: list[int]


class _Observation:
    """One decision's native observation, each field read at most once.

    A native field read converts its whole list into new Python objects, so
    repeated reads of the same field used to dominate a decision's time. The
    fields every decision uses are read up front; the rest on first use.
    """

    def __init__(self, native: managym.Observation) -> None:
        space = native.action_space
        self.native = native
        self.action_space = _Offers(
            int(space.action_space_type), space.actions, list(space.focus)
        )
        self.turn: managym.Turn = native.turn
        self.agent: managym.Player = native.agent
        self.opponent: managym.Player = native.opponent
        self.recent_events: list[managym.EventData] = native.recent_events
        self._lists: dict[str, list[Any]] = {}

    def _list(self, name: str) -> list[Any]:
        found = self._lists.get(name)
        if found is None:
            found = self._lists[name] = getattr(self.native, name)
        return found

    @property
    def agent_cards(self) -> list[managym.Card]:
        return self._list("agent_cards")

    @property
    def opponent_cards(self) -> list[managym.Card]:
        return self._list("opponent_cards")

    @property
    def agent_permanents(self) -> list[managym.Permanent]:
        return self._list("agent_permanents")

    @property
    def opponent_permanents(self) -> list[managym.Permanent]:
        return self._list("opponent_permanents")

    @property
    def stack_objects(self) -> list[managym.StackObject]:
        return self._list("stack_objects")

    @property
    def agent_sideboard(self) -> list[managym.SideboardCard]:
        return self._list("agent_sideboard")


class _Board:
    """Viewer-relative facts about one observation."""

    def __init__(self, native: managym.Observation) -> None:
        raw = _Observation(native)
        self.raw = raw
        self.own_id = int(raw.agent.id)
        # Built on first use: most priority passes never need them.
        self._cards: dict[int, managym.Card] | None = None
        self._by_key: dict[int, managym.Card] | None = None
        self._permanents: dict[int, managym.Permanent] | None = None

    @property
    def cards(self) -> dict[int, managym.Card]:
        if self._cards is None:
            self._cards = {
                int(c.id): c for c in self.raw.agent_cards + self.raw.opponent_cards
            }
        return self._cards

    @property
    def by_key(self) -> dict[int, managym.Card]:
        if self._by_key is None:
            self._by_key = {
                int(c.registry_key): c
                for c in self.raw.agent_cards + self.raw.opponent_cards
            }
        return self._by_key

    @property
    def permanents(self) -> dict[int, managym.Permanent]:
        if self._permanents is None:
            self._permanents = {
                int(p.id): p
                for p in self.raw.agent_permanents + self.raw.opponent_permanents
            }
        return self._permanents

    def is_creature(self, permanent: managym.Permanent) -> bool:
        card = self.by_key.get(int(permanent.registry_key))
        if permanent.is_animated or (card is not None and card.card_types.is_creature):
            return True
        return card is None and permanent.toughness > 0

    def mana_value(self, permanent: managym.Permanent) -> int:
        card = self.by_key.get(int(permanent.registry_key))
        return int(card.mana_cost.mana_value) if card is not None else 0

    def creatures(self, own: bool) -> list[managym.Permanent]:
        side = self.raw.agent_permanents if own else self.raw.opponent_permanents
        return [p for p in side if self.is_creature(p)]

    def is_own(self, permanent: managym.Permanent) -> bool:
        return int(permanent.controller_id) == self.own_id

    def untapped_lands(self) -> int:
        count = 0
        for permanent in self.raw.agent_permanents:
            card = self.by_key.get(int(permanent.registry_key))
            if card is not None and card.card_types.is_land and not permanent.tapped:
                count += 1
        return count

    def can_kick(self, card: managym.Card) -> bool:
        cost = int(card.mana_cost.mana_value) + int(card.kicker_cost)
        return card.kicker_cost > 0 and self.untapped_lands() >= cost

    def hand(self) -> list[managym.Card]:
        return [c for c in self.raw.agent_cards if int(c.zone) == ZoneEnum.HAND]

    def own_turn(self) -> bool:
        return int(self.raw.turn.active_player_id) == self.own_id

    def lands_in_play(self) -> int:
        return sum(
            1
            for p in self.raw.agent_permanents
            if (card := self.by_key.get(int(p.registry_key))) is not None
            and card.card_types.is_land
        )

    def graveyard_lessons(self) -> int:
        return int(self.raw.agent.graveyard_lessons)

    def own_spell_on_stack(self) -> bool:
        return any(
            int(o.kind) == int(managym.StackObjectKindEnum.SPELL)
            and int(o.controller_id) == self.own_id
            for o in self.raw.stack_objects
        )

    def opponent_spell_on_stack(self) -> bool:
        return any(
            int(o.kind) == int(managym.StackObjectKindEnum.SPELL)
            and int(o.controller_id) != self.own_id
            for o in self.raw.stack_objects
        )


def _spell_damage(board: _Board, spell: TargetedSpell, kicked: bool) -> int:
    damage = spell.kicked_damage if kicked else spell.damage
    if spell.lesson_damage:
        damage += board.graveyard_lessons()
    return damage


def _dies(damage: int, deathtouch: bool, body: _Body) -> bool:
    return damage > 0 and (deathtouch or damage >= body.toughness - body.damage)


def _can_block(blocker: managym.Permanent, attacker: managym.Permanent) -> bool:
    if attacker.cant_be_blocked_this_turn or blocker.tapped:
        return False
    if attacker.keywords.flying:
        return blocker.keywords.flying or blocker.keywords.reach
    return True


def _first_strike(body: _Body) -> bool:
    return body.keywords.first_strike or body.keywords.double_strike


def _fight(attacker: _Body, blocker: _Body) -> tuple[bool, bool]:
    """(attacker dies, blocker dies) in one-on-one combat."""
    attacker_dies = _dies(blocker.power, blocker.keywords.deathtouch, attacker)
    blocker_dies = _dies(attacker.power, attacker.keywords.deathtouch, blocker)
    if _first_strike(attacker) and not _first_strike(blocker) and blocker_dies:
        attacker_dies = False
    if _first_strike(blocker) and not _first_strike(attacker) and attacker_dies:
        blocker_dies = False
    return attacker_dies, blocker_dies


def _outcome(ours_dies: bool, theirs_dies: bool) -> int:
    """Higher is better for us: their creature dying, ours surviving."""
    return 2 * int(theirs_dies) + int(not ours_dies)


# ---------------------------------------------------------------------------
# Player


class HeuristicPlayer(ScriptedGreedyPlayer):
    """Rule-of-thumb anchor; see the module docstring for its rules.

    Per-game memory: kicker decisions by card ID, and this turn's attack pool,
    combat assignments and reserved trick. A lower turn number means a new
    game, so reusing one instance across games is safe.
    """

    def __init__(self) -> None:
        self._kicked: dict[int, bool] = {}
        self._turn = -1
        self._attack_pool: set[int] | None = None
        self._held: set[int] = set()
        self._blocks: dict[int, set[int]] = {}
        self._block_plan: dict[int, int] | None = None
        self._trick_reserved = False

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        board = _Board(env.last_raw_obs)
        self._observe(board)
        kind = board.raw.action_space.action_space_type
        choice: int | None = None
        if kind == ActionSpaceEnum.PRIORITY:
            choice = self._priority(board)
        elif kind == ActionSpaceEnum.CHOOSE_TARGET:
            choice = self._target(board)
        elif kind == ActionSpaceEnum.DECLARE_ATTACKER:
            choice = self._attack(board)
        elif kind == ActionSpaceEnum.DECLARE_BLOCKER:
            choice = self._block(board)
        elif kind == ActionSpaceEnum.PAY_OR_NOT:
            choice = self._pay(board)
        elif kind == ActionSpaceEnum.LEARN:
            choice = self._learn(board)
        elif kind == ActionSpaceEnum.DISCARD:
            choice = self._discard(board)
        elif kind == ActionSpaceEnum.LOOK_AND_SELECT:
            choice = self._select(board)
        return choice if choice is not None else super().act(env, obs)

    def _observe(self, board: _Board) -> None:
        turn = int(board.raw.turn.turn_number)
        if turn < self._turn:
            self._kicked.clear()
        if turn != self._turn:
            self._turn = turn
            self._attack_pool = None
            self._held = set()
            self._blocks = {}
            self._block_plan = None
            self._trick_reserved = False
        for event in board.raw.recent_events:
            if int(event.event_type) == EventTypeEnum.BLOCKERS_DECLARED:
                attacker, blocker = int(event.source_id), int(event.target_id)
                self._blocks.setdefault(attacker, set()).add(blocker)
            elif int(event.event_type) == EventTypeEnum.COMBAT_ATTACKERS_DECLARED:
                self._blocks.setdefault(int(event.source_id), set())

    # Priority -------------------------------------------------------------

    def _priority(self, board: _Board) -> int | None:
        # One pass over the offers: most priority prompts offer only a pass.
        land = passing = None
        casts: list[tuple[int, int]] = []
        activations: list[tuple[int, int]] = []
        for index, action in enumerate(board.raw.action_space.actions):
            kind = int(action.action_type)
            if kind == ActionEnum.PRIORITY_PASS_PRIORITY:
                passing = index
            elif kind == ActionEnum.PRIORITY_PLAY_LAND:
                land = index if land is None else land
            elif kind == ActionEnum.PRIORITY_CAST_SPELL:
                focus = action.focus
                if len(focus) == 1:
                    casts.append((index, int(focus[0])))
            elif kind == ActionEnum.PRIORITY_ACTIVATE_ABILITY:
                focus = action.focus
                if focus:
                    activations.append((index, int(focus[0])))
        if land is not None:
            return land
        if not casts and not activations:
            return passing
        if board.own_spell_on_stack():
            # Let it resolve first: a second removal spell would otherwise pick
            # the same, already doomed target.
            return passing
        known = abilities()
        fetch = self._ability(board, activations, known.prompt)
        if fetch is not None:
            return fetch
        step = int(board.raw.turn.step)
        if board.own_turn() and step == StepEnum.COMBAT_DECLARE_BLOCKERS:
            unblocked = {
                a
                for a, blockers in self._blocks.items()
                if not blockers and a in board.permanents and board.permanents[a].tapped
            }
            pump = self._ability(
                board, [a for a in activations if a[1] in unblocked], known.pump
            )
            if pump is not None:
                return pump
        ranked = sorted(
            (
                (-int(board.cards[card_id].mana_cost.mana_value), index, card_id)
                for index, card_id in casts
                if card_id in board.cards
            )
        )
        for _, index, card_id in ranked:
            card = board.cards[card_id]
            if self._timely(board, card) and self._worth_casting(board, card):
                return index
        if not board.own_turn() and step == StepEnum.ENDING_END:
            spare = self._ability(board, activations, known.spare)
            if spare is not None:
                return spare
        return passing

    def _ability(
        self, board: _Board, activations: list[tuple[int, int]], names: frozenset[str]
    ) -> int | None:
        """The first offered activation whose source card is one of `names`."""
        for index, source_id in activations:
            source = board.permanents.get(source_id)
            card = board.by_key.get(int(source.registry_key)) if source else None
            if card is not None and card.name in names:
                return index
        return None

    def _timely(self, board: _Board, card: managym.Card) -> bool:
        """Whether this priority window suits an instant; other spells only
        arrive at sorcery timing.

        Nothing is gained by casting in one's own upkeep or draw step. Instants
        without a target that matters now (card draw, selection) wait for the
        opponent's end step, so the turn's mana goes to permanents first and
        what is left still works.
        """
        if card.name not in instant_names():
            return True
        step = int(board.raw.turn.step)
        if board.own_turn() and step in (
            StepEnum.BEGINNING_UPKEEP,
            StepEnum.BEGINNING_DRAW,
        ):
            return False
        spell = targeted_spells().get(card.name)
        if spell is not None and spell.polarity is not None:
            return True
        return not board.own_turn() and step == StepEnum.ENDING_END

    def _worth_casting(self, board: _Board, card: managym.Card) -> bool:
        spell = targeted_spells().get(card.name)
        if spell is None or spell.polarity is None:
            return True
        if spell.polarity == "beneficial":
            return bool(self._pump_targets(board, spell))
        if spell.selector in {"spell", "spell_or_permanent"}:
            return board.opponent_spell_on_stack()
        kicked = spell.kicked_damage > 0 and board.can_kick(card)
        damage = _spell_damage(board, spell, kicked)
        if damage > 0:
            # Burn that can hit players is held for a creature it kills, or
            # the last points of life.
            if (
                spell.selector == "creature_or_player"
                and damage >= board.raw.opponent.life
            ):
                return True
            return any(
                _dies(damage, False, _Body.of(p)) for p in board.creatures(own=False)
            )
        return True

    def _pay(self, board: _Board) -> int | None:
        raw = board.raw
        focus = list(raw.action_space.focus)
        card = board.cards.get(int(focus[0])) if len(focus) == 1 else None
        if (
            card is not None
            and card.kicker_cost > 0
            and int(card.owner_id) == board.own_id
        ):
            # Kicker is chosen before targets and payment; lands are untapped.
            kick = board.can_kick(card)
            self._kicked[int(card.id)] = kick
            return _first(
                raw, ActionEnum.PAY_COST if kick else ActionEnum.DECLINE_CHOICE
            )
        return _first(raw, ActionEnum.PAY_COST)

    # Card choices -----------------------------------------------------------

    def _learn(self, board: _Board) -> int | None:
        """Take a Lesson rather than rummage: removal while the opponent has
        creatures, otherwise card advantage."""
        raw = board.raw
        removal_wanted = bool(board.creatures(own=False))
        # Lesson offers name sideboard candidates, not visible cards.
        lessons = {int(c.candidate_id): c.name for c in raw.agent_sideboard}
        best: tuple[int, int] | None = None
        for index, action in enumerate(raw.action_space.actions):
            if int(action.action_type) != ActionEnum.LEARN_TAKE_LESSON:
                continue
            name = lessons.get(int(action.focus[0])) if action.focus else None
            spell = targeted_spells().get(name) if name is not None else None
            removal = spell is not None and spell.polarity == "harmful"
            rank = 2 if removal == removal_wanted else 1
            if best is None or rank > best[0]:
                best = (rank, index)
        if best is not None:
            return best[1]
        return self._discard(board)

    def _discard(self, board: _Board) -> int | None:
        """Discard a land once five are in play, else the costliest card; an
        optional discard is declined when nothing is spare."""
        raw = board.raw
        options: list[tuple[int, int]] = []
        decline = None
        for index, action in enumerate(raw.action_space.actions):
            if int(action.action_type) == ActionEnum.DECLINE_CHOICE:
                decline = index
                continue
            card = board.cards.get(int(action.focus[0])) if action.focus else None
            if card is None:
                continue
            if card.card_types.is_land:
                rank = 100 if board.lands_in_play() >= 5 else -1
            else:
                rank = int(card.mana_cost.mana_value)
            options.append((rank, index))
        if not options:
            return decline
        rank, index = max(options)
        if decline is not None and rank < 100:
            return decline
        return index

    def _select(self, board: _Board) -> int | None:
        """Keep a land while short of four, else the costliest spell."""
        raw = board.raw
        short = (
            board.lands_in_play() + sum(1 for c in board.hand() if c.card_types.is_land)
            < 4
        )
        options: list[tuple[int, int]] = []
        for index, action in enumerate(raw.action_space.actions):
            card = board.cards.get(int(action.focus[0])) if action.focus else None
            if card is None or int(action.action_type) == ActionEnum.DECLINE_CHOICE:
                continue
            if card.card_types.is_land:
                rank = 100 if short else -1
            else:
                rank = int(card.mana_cost.mana_value)
            options.append((rank, index))
        return max(options)[1] if options else None

    # Combat tricks ----------------------------------------------------------

    def _pump_targets(self, board: _Board, spell: TargetedSpell) -> list[int]:
        """Own creatures, best first, where this pump changes the combat result.

        Only after blocks: a one-on-one fight it improves, or an unblocked
        attacker whose extra power makes unblocked damage lethal.
        """
        if int(board.raw.turn.step) != StepEnum.COMBAT_DECLARE_BLOCKERS:
            return []
        gains: list[tuple[int, int, int]] = []
        unblocked = [
            board.permanents[a]
            for a, blockers in self._blocks.items()
            if not blockers and a in board.permanents
        ]
        unblocked_power = sum(p.power for p in unblocked if board.is_own(p))
        for attacker_id, blockers in self._blocks.items():
            attacker = board.permanents.get(attacker_id)
            if attacker is None:
                continue
            if not blockers:
                lethal = unblocked_power + spell.pump[0] >= board.raw.opponent.life
                if board.is_own(attacker) and lethal:
                    gains.append((3, attacker.power, attacker_id))
                continue
            if len(blockers) != 1:
                continue
            blocker = board.permanents.get(next(iter(blockers)))
            if blocker is None:
                continue
            ours, theirs = (
                (attacker, blocker) if board.is_own(attacker) else (blocker, attacker)
            )
            if not board.is_own(ours):
                continue
            before = self._fight_for(board, ours, theirs, (0, 0))
            after = self._fight_for(board, ours, theirs, spell.pump)
            if after > before:
                gains.append((after - before, ours.power, int(ours.id)))
        return [pid for *_, pid in sorted(gains, reverse=True)]

    def _fight_for(
        self,
        board: _Board,
        ours: managym.Permanent,
        theirs: managym.Permanent,
        pump: tuple[int, int],
    ) -> int:
        body = _Body.of(ours, pump)
        if board.is_own(ours) and int(ours.id) in self._blocks:
            ours_dies, theirs_dies = _fight(body, _Body.of(theirs))
        else:
            theirs_dies, ours_dies = _fight(_Body.of(theirs), body)
        return _outcome(ours_dies, theirs_dies)

    def _trick_in_hand(self, board: _Board) -> TargetedSpell | None:
        lands = board.untapped_lands()
        for card in board.hand():
            spell = targeted_spells().get(card.name)
            if (
                spell is not None
                and spell.polarity == "beneficial"
                and spell.pump != (0, 0)
                and int(card.mana_cost.mana_value) <= lands
            ):
                return spell
        return None

    # Targets --------------------------------------------------------------

    def _target(self, board: _Board) -> int | None:
        raw = board.raw
        source = next(
            (board.cards[i] for i in raw.action_space.focus if i in board.cards), None
        )
        spell = targeted_spells().get(source.name) if source is not None else None
        if spell is None or spell.polarity is None:
            return None
        wanted = "opponent" if spell.polarity == "harmful" else "own"
        kicked = source is not None and self._kicked.get(int(source.id), False)
        damage = _spell_damage(board, spell, kicked)
        pumps = (
            self._pump_targets(board, spell) if spell.polarity == "beneficial" else []
        )
        best: tuple[tuple[int, ...], int] | None = None
        decline = None
        for index, action in enumerate(raw.action_space.actions):
            if int(action.action_type) == ActionEnum.DECLINE_CHOICE:
                decline = index
                continue
            if self._offer_side(board, index) != wanted:
                continue
            key = self._target_value(board, spell, damage, pumps, index)
            if best is None or key > best[0]:
                best = (key, index)
        if best is not None and (best[0][0] > 0 or decline is None):
            return best[1]
        return decline

    def _offer_side(
        self, board: _Board, index: int
    ) -> Literal["own", "opponent"] | None:
        """Viewer-relative side of a player, permanent or stack-spell target.

        Stack spells use card ownership as control: this pool has no
        control-changing spells.
        """
        action = board.raw.action_space.actions[index]
        if int(action.action_type) != ActionEnum.CHOOSE_TARGET:
            return None
        focus = action.focus
        if len(focus) != 1:
            return None
        target = int(focus[0])
        if target == board.own_id:
            return "own"
        if target == int(board.raw.opponent.id):
            return "opponent"
        permanent = board.permanents.get(target)
        if permanent is not None:
            controller = int(permanent.controller_id)
        elif target in board.cards:
            controller = int(board.cards[target].owner_id)
        else:
            return None
        return "own" if controller == board.own_id else "opponent"

    def _target_value(
        self,
        board: _Board,
        spell: TargetedSpell,
        damage: int,
        pumps: list[int],
        index: int,
    ) -> tuple[int, ...]:
        """Higher is better; a first element of 0 is worse than declining."""
        target = int(board.raw.action_space.actions[index].focus[0])
        permanent = board.permanents.get(target)
        if permanent is None:
            # A stack spell is what a held counter or bounce waits for. A
            # player ranks below a creature that removal kills, above one it
            # does not.
            if target in board.cards:
                return (3,)
            # Burn to the face wins outright when it is lethal.
            return (4,) if damage >= board.raw.opponent.life else (1,)
        if spell.polarity == "beneficial":
            if target in pumps:
                return (2, len(pumps) - pumps.index(target))
            return (1 if target in self._blocks else 0, permanent.power)
        dies = damage == 0 or _dies(damage, False, _Body.of(permanent))
        return (2 if dies else 0, permanent.power, board.mana_value(permanent))

    # Combat ---------------------------------------------------------------

    def _attack(self, board: _Board) -> int | None:
        raw = board.raw
        attacker = board.permanents.get(int(raw.action_space.actions[0].focus[0]))
        if attacker is None:
            return None
        if self._attack_pool is None:
            # Declared attackers tap, so fix the pool at the first prompt.
            self._attack_pool = {
                int(p.id)
                for p in board.creatures(own=True)
                if not p.tapped and (not p.is_summoning_sick or p.keywords.haste)
            }
            self._held = self._defenders_to_hold(board)
        blockers = [p for p in board.creatures(own=False) if not p.tapped]
        unblockable = sum(
            p.power
            for p in board.creatures(own=True)
            if int(p.id) in self._attack_pool
            and not any(_can_block(b, p) for b in blockers)
        )
        if unblockable >= raw.opponent.life:
            return _declared(raw, True)
        if int(attacker.id) in self._held:
            return _declared(raw, False)
        dominators = [
            b
            for b in blockers
            if _can_block(b, attacker)
            and _fight(_Body.of(attacker), _Body.of(b)) == (True, False)
        ]
        if attacker.keywords.menace and len(blockers) < 2:
            dominators = []
        if dominators and not self._trick_reserved:
            trick = self._trick_in_hand(board)
            pumped = _Body.of(attacker, trick.pump) if trick else None
            if pumped is not None and all(
                _fight(pumped, _Body.of(b)) != (True, False) for b in dominators
            ):
                self._trick_reserved = True
                dominators = []
        return _declared(raw, not dominators)

    def _defenders_to_hold(self, board: _Board) -> set[int]:
        """Attack-pool creatures to keep home so the swing back is not lethal.

        Every opposing creature untaps and may attack; each defender is assumed
        to stop one of the largest attackers. The toughest creatures stay home
        first, as they block best.
        """
        incoming = sorted((p.power for p in board.creatures(own=False)), reverse=True)
        assert self._attack_pool is not None
        defenders = sum(
            1 for p in board.creatures(own=True) if int(p.id) not in self._attack_pool
        )
        candidates = sorted(
            (p for p in board.creatures(own=True) if int(p.id) in self._attack_pool),
            key=lambda p: (-p.toughness, p.power),
        )
        held: set[int] = set()
        for creature in candidates:
            if sum(incoming[defenders:]) < board.raw.agent.life:
                break
            held.add(int(creature.id))
            defenders += 1
        return held

    def _block(self, board: _Board) -> int | None:
        """Serve this blocker's part of one plan made for the whole combat."""
        raw = board.raw
        actions = raw.action_space.actions
        if self._block_plan is None:
            self._block_plan = self._plan_blocks(board)
        blocker_id = next((int(a.focus[0]) for a in actions if len(a.focus) >= 1), None)
        target = self._block_plan.get(blocker_id) if blocker_id is not None else None
        for index, action in enumerate(actions):
            if len(action.focus) == 2 and int(action.focus[1]) == target:
                return index
        return next((i for i, a in enumerate(actions) if len(a.focus) == 1), None)

    def _plan_blocks(self, board: _Board) -> dict[int, int]:
        """Blocker id to attacker id, at most one blocker per attacker.

        First the blocks that win (kill and survive), then trades up, then free
        absorptions, larger attackers first within each. Then, while the
        unblocked damage is lethal, the cheapest remaining creatures chump the
        largest attackers.
        """
        # The offers name what this blocker can block; tapped recorded attackers
        # add any it cannot (fliers). Recent events can still carry an earlier
        # turn's attack, so an untapped one is not counted.
        offered = {
            int(a.focus[1]) for a in board.raw.action_space.actions if len(a.focus) == 2
        }
        attackers = [
            p
            for p in board.creatures(own=False)
            if int(p.id) in offered or (int(p.id) in self._blocks and p.tapped)
        ]
        attackers.sort(key=lambda p: -p.power)
        blockers = [p for p in board.creatures(own=True) if not p.tapped]
        pairs: list[tuple[int, int, int, int, int]] = []
        for attacker in attackers:
            for blocker in blockers:
                if not _can_block(blocker, attacker):
                    continue
                attacker_dies, blocker_dies = _fight(
                    _Body.of(attacker), _Body.of(blocker)
                )
                if attacker_dies and not blocker_dies:
                    rank = 3
                elif attacker_dies and board.mana_value(attacker) >= board.mana_value(
                    blocker
                ):
                    rank = 2
                elif not blocker_dies:
                    rank = 1
                else:
                    continue
                pairs.append(
                    (
                        rank,
                        attacker.power,
                        -board.mana_value(blocker),
                        int(blocker.id),
                        int(attacker.id),
                    )
                )
        plan: dict[int, int] = {}
        blocked: set[int] = set()
        for *_, blocker_id, attacker_id in sorted(pairs, reverse=True):
            if blocker_id not in plan and attacker_id not in blocked:
                plan[blocker_id] = attacker_id
                blocked.add(attacker_id)
        unblocked = [a for a in attackers if int(a.id) not in blocked]
        spare = sorted(
            (b for b in blockers if int(b.id) not in plan),
            key=lambda b: (board.mana_value(b), b.power),
        )
        incoming = sum(a.power for a in unblocked)
        for attacker in unblocked:
            if incoming < board.raw.agent.life:
                break
            chump = next((b for b in spare if _can_block(b, attacker)), None)
            if chump is None:
                continue
            spare.remove(chump)
            plan[int(chump.id)] = int(attacker.id)
            incoming -= attacker.power
        return plan


def _first(raw: _Observation, action_type: ActionEnum) -> int | None:
    return next(
        (
            index
            for index, action in enumerate(raw.action_space.actions)
            if int(action.action_type) == action_type
        ),
        None,
    )


def _declared(raw: _Observation, value: bool) -> int | None:
    return next(
        (
            index
            for index, action in enumerate(raw.action_space.actions)
            if action.declared is value
        ),
        None,
    )
