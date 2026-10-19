"""Rule-of-thumb arena anchor, `heuristic`.

`ScriptedGreedyPlayer` (v1) plays a land, casts the first castable spell,
attacks and blocks with everything, and takes the first target offer. Target
offers arrive in absolute engine order (player 0's objects first), so v1 picks
a target side by seat: from seat 0 it burns itself with Igneous Inspiration.

`HeuristicPlayer` keeps v1 for every prompt it has no rule for and adds:

- Mana: play a land, then cast the castable spell with the highest mana value
  that has a worthwhile use. The engine only offers affordable casts, so
  repeated highest-first casting spends as much mana as the offers allow.
  Kicker is paid when untapped lands cover spell plus kicker.
- Targets: effect polarity is derived from the semantic card source (damage,
  bounce, counter and exile are harmful; pumps, untaps and keyword grants are
  beneficial). Harmful effects go to the opponent's side, beneficial ones to
  its own. Removal is held until it kills something (with the damage the
  available mana allows), counters until the opponent has a spell on the stack.
- Combat: attack with a creature unless the opponent has an untapped creature
  that can block it, kill it and survive, or attack with everything when
  unblockable damage is lethal. Block when the blocker kills and survives,
  trade when the attacker costs at least as much, and chump only against
  lethal damage.
- Own combat tricks: a pump in hand with mana to cast it lets one dominated
  creature attack anyway. After blocks, pumps are cast only where they change
  a one-on-one fight or make unblocked damage lethal. The opponent's tricks are
  ignored: they depend on a hidden hand.
- Optional costs (ward, "unless pays"): pay when offered.

Combat assignments come from the observation's attacker/blocker events, whose
object IDs are the current permanent IDs; the player remembers them for the
turn. Mana is approximated by untapped lands, ignoring colors and other
sources. Creature identity uses the visible card definitions; a permanent
without one counts as a creature when its toughness is positive.

This lives outside `players.py` because arena registrations pin that file's
bytes as the v1 and demo-search source identity; v1 stays frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from academy.enums import ActionEnum, ActionSpaceEnum, EventTypeEnum, StepEnum, ZoneEnum
from manabot.env import Env
from manabot.study.targeting import inspect_target
import managym

from .players import ScriptedGreedyPlayer

Polarity = Literal["harmful", "beneficial"]

SEMANTIC_SOURCE = (
    Path(__file__).resolve().parents[2] / "content/semantic/v1/two_deck.source.json"
)

# Ops that act on a target role, by what they do to that target's controller.
HARMFUL_OPS = frozenset(
    {
        "deal_damage",
        "deal_power_damage",
        "return_to_hand",
        "counter_unless_pays",
        "exile_until_source_leaves",
    }
)
BENEFICIAL_OPS = frozenset({"untap", "grant_keywords", "earthbend"})

# ---------------------------------------------------------------------------
# Card knowledge derived from the semantic source


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
    # Power/toughness added to each target until end of turn.
    pump: tuple[int, int] = (0, 0)


def _target_ops(ops: list[dict[str, Any]], roles: set[str]) -> list[dict[str, Any]]:
    """Every op, at any nesting depth, whose target is one of `roles`."""
    found: list[dict[str, Any]] = []
    for op in ops:
        for value in op.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if "op" in value[0]:
                    found.extend(_target_ops(value, roles))
        if op.get("target") in roles or op.get("target") == "current_target":
            found.append(op)
    return found


def _damage(ops: list[dict[str, Any]], roles: set[str], kicked: bool) -> int:
    """Largest damage to a target role, following kicker branches as given."""
    best = 0
    for op in ops:
        if op["op"] == "branch":
            if op["condition"].get("kind") == "kicked":
                taken = [op["then"] if kicked else op.get("otherwise", [])]
            else:
                taken = [op["then"], op.get("otherwise", [])]
            best = max([best, *(_damage(b, roles, kicked) for b in taken)])
            continue
        for value in op.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if "op" in value[0]:
                    best = max(best, _damage(value, roles, kicked))
        if op["op"] == "deal_damage" and op.get("target") in roles:
            best = max(best, op["amount"])
    return best


def _op_polarity(op: dict[str, Any]) -> Polarity | None:
    if op["op"] in HARMFUL_OPS:
        return "harmful"
    if op["op"] in BENEFICIAL_OPS:
        return "beneficial"
    if op["op"] == "modify_pt":
        change = op.get("power", 0) + op.get("toughness", 0)
        return "beneficial" if change > 0 else "harmful" if change < 0 else None
    return None


def derive_targeted_spells(source: dict[str, Any]) -> dict[str, TargetedSpell]:
    """Map card name to its targeted spell program, by semantic source.

    A program whose target ops are unknown or of mixed polarity gets polarity
    None, and the player falls back to v1 for it.
    """
    spells: dict[str, TargetedSpell] = {}
    for definition in source["definitions"]:
        for program in definition.get("programs", []):
            targets = program.get("targets", [])
            if program["kind"] != "spell" or not targets:
                continue
            roles = {t["role"] for t in targets}
            ops = _target_ops(program["ops"], roles)
            polarities = {_op_polarity(op) for op in ops}
            pumps = [op for op in ops if op["op"] == "modify_pt"]
            spells[definition["registry_name"]] = TargetedSpell(
                polarity=polarities.pop() if len(polarities) == 1 else None,
                selector=targets[0]["selector"]["kind"],
                side_restricted=all(
                    t["selector"]["kind"].endswith(
                        ("_you_control", "_opponent_controls")
                    )
                    for t in targets
                ),
                damage=_damage(program["ops"], roles, kicked=False),
                kicked_damage=_damage(program["ops"], roles, kicked=True),
                pump=(
                    sum(op.get("power", 0) for op in pumps),
                    sum(op.get("toughness", 0) for op in pumps),
                ),
            )
    return spells


def heuristic_source_paths() -> list[Path]:
    """Files whose bytes define this player's behavior, for arena identity."""
    arena = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        arena / "players.py",
        arena.parent / "study/targeting.py",
        SEMANTIC_SOURCE,
    ]


@cache
def targeted_spells() -> dict[str, TargetedSpell]:
    return derive_targeted_spells(json.loads(SEMANTIC_SOURCE.read_text()))


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


class _Board:
    """Viewer-relative facts about one observation."""

    def __init__(self, raw: managym.Observation) -> None:
        self.raw = raw
        self.cards = {int(c.id): c for c in raw.agent_cards + raw.opponent_cards}
        self.by_key = {
            int(c.registry_key): c for c in raw.agent_cards + raw.opponent_cards
        }
        self.own_id = int(raw.agent.id)
        self.permanents = {
            int(p.id): p for p in raw.agent_permanents + raw.opponent_permanents
        }

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

    def opponent_spell_on_stack(self) -> bool:
        return any(
            int(o.kind) == int(managym.StackObjectKindEnum.SPELL)
            and int(o.controller_id) != self.own_id
            for o in self.raw.stack_objects
        )


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
        self._blocks: dict[int, set[int]] = {}
        self._trick_reserved = False

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        raw = env.last_raw_obs
        board = _Board(raw)
        self._observe(board)
        kind = int(raw.action_space.action_space_type)
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
        return choice if choice is not None else super().act(env, obs)

    def _observe(self, board: _Board) -> None:
        turn = int(board.raw.turn.turn_number)
        if turn < self._turn:
            self._kicked.clear()
        if turn != self._turn:
            self._turn = turn
            self._attack_pool = None
            self._blocks = {}
            self._trick_reserved = False
        for event in board.raw.recent_events:
            if int(event.event_type) == EventTypeEnum.BLOCKERS_DECLARED:
                attacker, blocker = int(event.source_id), int(event.target_id)
                self._blocks.setdefault(attacker, set()).add(blocker)
            elif int(event.event_type) == EventTypeEnum.COMBAT_ATTACKERS_DECLARED:
                self._blocks.setdefault(int(event.source_id), set())

    # Priority -------------------------------------------------------------

    def _priority(self, board: _Board) -> int | None:
        raw = board.raw
        land = _first(raw, ActionEnum.PRIORITY_PLAY_LAND)
        if land is not None:
            return land
        casts = [
            (int(board.cards[int(a.focus[0])].mana_cost.mana_value), index)
            for index, a in enumerate(raw.action_space.actions)
            if int(a.action_type) == ActionEnum.PRIORITY_CAST_SPELL
            and len(a.focus) == 1
            and int(a.focus[0]) in board.cards
        ]
        for _, index in sorted(casts, key=lambda c: (-c[0], c[1])):
            card = board.cards[int(raw.action_space.actions[index].focus[0])]
            if self._worth_casting(board, card):
                return index
        return _first(raw, ActionEnum.PRIORITY_PASS_PRIORITY)

    def _worth_casting(self, board: _Board, card: managym.Card) -> bool:
        spell = targeted_spells().get(card.name)
        if spell is None or spell.polarity is None:
            return True
        if spell.polarity == "beneficial":
            return bool(self._pump_targets(board, spell))
        if spell.selector in {"spell", "spell_or_permanent"}:
            return board.opponent_spell_on_stack()
        if spell.selector == "creature_or_player":
            return True
        if spell.kicked_damage > 0:
            damage = spell.kicked_damage if board.can_kick(card) else spell.damage
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
        prompt = inspect_target(raw, 0)
        source = (
            board.cards.get(int(raw.action_space.focus[0]))
            if raw.action_space.focus
            else None
        )
        spell = targeted_spells().get(prompt.source or "") if prompt else None
        if spell is None or spell.polarity is None or spell.side_restricted:
            return None
        wanted = "opponent" if spell.polarity == "harmful" else "own"
        damage = spell.damage
        if source is not None and self._kicked.get(int(source.id)):
            damage = spell.kicked_damage
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
        """Viewer-relative side; stack spells use card ownership as control."""
        choice = inspect_target(board.raw, index)
        if choice is None:
            return None
        if choice.target_kind != "unresolved":
            return choice.side
        focus = board.raw.action_space.actions[index].focus
        if len(focus) != 1 or int(focus[0]) not in board.cards:
            return None
        # This pool has no control-changing spells, so owner is controller.
        owner = int(board.cards[int(focus[0])].owner_id)
        return "own" if owner == board.own_id else "opponent"

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
            return (3,) if target in board.cards else (1,)
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
        blockers = [p for p in board.creatures(own=False) if not p.tapped]
        unblockable = sum(
            p.power
            for p in board.creatures(own=True)
            if int(p.id) in self._attack_pool
            and not any(_can_block(b, p) for b in blockers)
        )
        if unblockable >= raw.opponent.life:
            return _declared(raw, True)
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

    def _block(self, board: _Board) -> int | None:
        raw = board.raw
        actions = raw.action_space.actions
        attackers = {
            int(a.focus[1]): board.permanents[int(a.focus[1])]
            for a in actions
            if len(a.focus) == 2 and int(a.focus[1]) in board.permanents
        }
        incoming = sum(p.power for p in attackers.values())
        options: list[tuple[int, int, int]] = []
        for index, action in enumerate(actions):
            if len(action.focus) != 2:
                continue
            blocker = board.permanents.get(int(action.focus[0]))
            attacker = attackers.get(int(action.focus[1]))
            if blocker is None or attacker is None:
                continue
            attacker_dies, blocker_dies = _fight(_Body.of(attacker), _Body.of(blocker))
            if attacker_dies and not blocker_dies:
                rank = 3
            elif attacker_dies and board.mana_value(attacker) >= board.mana_value(
                blocker
            ):
                rank = 2
            elif not blocker_dies:
                rank = 1  # Absorbs damage for free.
            elif incoming >= raw.agent.life:
                rank = 0  # Chump only against lethal.
            else:
                continue
            options.append((rank, attacker.power, index))
        if options:
            return max(options)[2]
        return next((i for i, a in enumerate(actions) if len(a.focus) == 1), None)


def _first(raw: managym.Observation, action_type: ActionEnum) -> int | None:
    return next(
        (
            index
            for index, action in enumerate(raw.action_space.actions)
            if int(action.action_type) == action_type
        ),
        None,
    )


def _declared(raw: managym.Observation, value: bool) -> int | None:
    return next(
        (
            index
            for index, action in enumerate(raw.action_space.actions)
            if action.declared is value
        ),
        None,
    )
