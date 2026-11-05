"""Separate training and evaluation exposure from shared native capture.

Evaluation uses replay-backed card_metrics and committed trace choices; its
traces do not measure opportunities or resolutions. Training uses private
collector receipts with complete choices and completed effect frames. Admission
and budget reconciliation precede aggregation; absent evidence is never zero.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
import gzip
from hashlib import sha256
from pathlib import Path
from typing import Literal

from etude.eval.match import Row
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from experiments.growing_pool_protocol import Protocol
from manabot.env.card_capture import CapturedCards, Seat
from manabot.env.failures import BufferSpec, PlayerSetup, TrainingExposureReceipt
from manabot.infra.identity import canonical_sha256
from manabot.sim.card_evidence import GameEvidence


class _Offer(BaseModel):
    action_type: str


class _Decision(BaseModel):
    actor: Literal[0, 1]
    player_id: str
    action_space_kind: str
    chosen_offer: _Offer
    post_state_digest: str


def decision_rows(blobs: list[bytes]) -> list[Row]:
    """Retain separate seat totals, even when a participant made no decisions."""
    rows: list[Row] = []
    for body in blobs:
        game = GameEvidence.model_validate_json(gzip.decompress(body))
        decisions = TypeAdapter(list[_Decision]).validate_python(
            game.trace.data["decisions"]
        )
        counts: Counter[tuple[int, str, str]] = Counter()
        for seat in (0, 1):
            counts[seat, "all", "all"] = 0
        for decision in decisions:
            if game.seat_players[decision.actor] != decision.player_id:
                raise ValueError("decision actor differs from retained seating")
            seat = ("candidate", "opponent").index(decision.player_id)
            counts[seat, "all", "all"] += 1
            counts[
                seat, decision.action_space_kind, decision.chosen_offer.action_type
            ] += 1
        digest = sha256(body).hexdigest()
        eligible = (
            game.outcome in ("win_loss", "draw") and game.replay.get("passed") is True
        )
        for (seat, kind, action), count in sorted(counts.items()):
            rows.append(
                {
                    "seat": seat,
                    "game": game.leg,
                    "game_seat": game.seat_players.index(
                        ("candidate", "opponent")[seat]
                    ),
                    "decision_kind": kind,
                    "action_type": action,
                    "committed_choices": count,
                    "eligible": eligible,
                    "evidence_sha256": digest,
                    "metric_version": "committed-native-choices-v1",
                }
            )
    return rows


class CardExposure(BaseModel):
    """Per-seat counts over captured games; focus denotes involvement, not benefit."""

    seat: Literal[0, 1]
    definition: int
    name: str
    opening_copies: int = 0
    drawn_copies: int = 0
    hand_at_end: int = 0
    opportunity_decisions: int = 0
    chosen_decisions: int = 0
    completed_frames: int = 0


class DecisionExposure(BaseModel):
    seat: Literal[0, 1]
    kind: str
    action_type: str
    opportunity_decisions: int = 0
    chosen_decisions: int = 0


class TrainingExposure(BaseModel):
    expected_games: int
    terminal_games: int = 0
    terminal_games_by_stream: dict[int, int] = Field(default_factory=dict)
    interrupted_games: int = 0
    decisions_by_seat: tuple[int, int] = (0, 0)
    cards: list[CardExposure] = Field(default_factory=list)
    actions: list[DecisionExposure] = Field(default_factory=list)
    receipts: list[str] = Field(default_factory=list)
    complete: bool = False
    issues: list[str] = Field(default_factory=list)
    unattributed_frames: int = 0
    replay_status: Literal["inputs retained; not replay-verified"] = (
        "inputs retained; not replay-verified"
    )


def read_training_exposure(paths: Sequence[Path]) -> list[TrainingExposureReceipt]:
    """Read downloaded job output, rejecting changed bytes and duplicate records."""
    result = []
    seen: set[str] = set()
    for root in paths:
        if not root.is_dir():
            raise ValueError(f"training exposure directory is unavailable: {root}")
        for path in sorted(root.glob("*.jsonl")):
            body = path.read_bytes()
            receipt = TrainingExposureReceipt.model_validate_json(body)
            digest = sha256(body).hexdigest()
            if path.stem != digest or digest != receipt.sha256 or digest in seen:
                raise ValueError("training exposure receipt digest differs or repeats")
            seen.add(digest)
            result.append(receipt)
    return result


@dataclass
class _ExposureCounts:
    """Aggregate admitted terminal games; interrupted prefixes never enter totals."""

    cards: dict[tuple[Seat, int], CardExposure] = field(default_factory=dict)
    actions: dict[tuple[Seat, str, str], DecisionExposure] = field(default_factory=dict)
    decisions: list[int] = field(default_factory=lambda: [0, 0])
    unattributed_frames: int = 0

    def _card(self, seat: Seat, definition: int, name: str) -> CardExposure:
        key = (seat, definition)
        if key not in self.cards:
            self.cards[key] = CardExposure(seat=seat, definition=definition, name=name)
        return self.cards[key]

    def add(self, cards: CapturedCards) -> None:
        facts = cards.capture.exposure
        assert facts is not None
        initial = {card.copy_id: card for card in cards.capture.initial}
        for card in initial.values():
            self._card(card.owner, card.definition, card.name)
        for seat in (0, 1):
            for copy in cards.capture.opening[seat]:
                card = initial[copy]
                self._card(seat, card.definition, card.name).opening_copies += 1
        for move in cards.capture.moves:
            card = move.card
            if move.cause == "draw":
                self._card(
                    move.controller, card.definition, card.name
                ).drawn_copies += 1
        for card in cards.final_cards:
            if card.zone == "Hand":
                self._card(card.owner, card.definition, card.name).hand_at_end += 1
        for choice in facts.choices:
            self.decisions[choice.actor] += 1
            chosen_type = choice.options[choice.selected].action_type
            for action_type in {option.action_type for option in choice.options}:
                key = (choice.actor, choice.kind, action_type)
                if key not in self.actions:
                    self.actions[key] = DecisionExposure(
                        seat=choice.actor, kind=choice.kind, action_type=action_type
                    )
                self.actions[key].opportunity_decisions += 1
                self.actions[key].chosen_decisions += int(action_type == chosen_type)
            available = {
                card.definition: card.name
                for option in choice.options
                for card in option.objects
            }
            selected = {
                card.definition: card.name
                for card in choice.options[choice.selected].objects
            }
            for definition, name in available.items():
                self._card(choice.actor, definition, name).opportunity_decisions += 1
            for definition, name in selected.items():
                self._card(choice.actor, definition, name).chosen_decisions += 1
        names = {
            card.definition: card.name
            for card in (*cards.capture.initial, *cards.final_cards)
        }
        for resolution in facts.resolutions:
            if resolution.definition is None:
                self.unattributed_frames += 1
                continue
            name = names.get(resolution.definition)
            if name is None:
                raise ValueError("resolution source definition missing from capture")
            self._card(
                resolution.controller, resolution.definition, name
            ).completed_frames += 1


def training_exposure(
    receipts: Sequence[TrainingExposureReceipt],
    *,
    seed: int,
    run_id: str | None,
    identities: dict[str, JsonValue],
    protocol: Protocol,
) -> TrainingExposure:
    """Reconcile observed training games against the frozen complete-game budget.

    Declined focus opportunities count once per card/decision, independent of
    the number of legal variants. A completed frame is not an effective outcome.
    Interrupted prefixes are retained but excluded from the terminal population.
    """
    report = TrainingExposure(
        expected_games=protocol.budget.training_games // len(protocol.budget.seeds)
    )
    counts = _ExposureCounts()
    expected_regime = (
        canonical_sha256(protocol.regime().model_dump(mode="json"))
        if any(r.context.get("seed") == seed for r in receipts)
        else None
    )
    seen: set[int] = set()
    population = protocol.manifest.source()
    # The sampled setup includes disclosure and native admission, not only lists.
    players_by_name = {
        player.name: PlayerSetup.retain(player) for player in population.players()
    }
    for receipt in receipts:
        if receipt.context.get("seed") != seed:
            continue
        if run_id is None or receipt.context.get("run") != run_id:
            raise ValueError("training exposure belongs to another run")
        if (
            receipt.context.get("regime_digest") != expected_regime
            or receipt.context.get("identities") != identities
            or receipt.identities.get("world") != protocol.world.world_version
            or receipt.game.cards.capture.content_digest != protocol.world.card_digest
            or receipt.identities.get("engine_extension_sha256")
            != identities.get("engine_extension_sha256")
            or not isinstance(identities.get("engine_extension_sha256"), str)
            or receipt.identities.get("action_overflow", "error") != "error"
            or not receipt.skip_trivial
        ):
            raise ValueError("training exposure world/input/source identity differs")
        buffers = TypeAdapter(dict[str, BufferSpec]).validate_python(
            receipt.identities.get("input_buffers")
        )
        if any(
            name not in buffers
            or tuple(buffers[name].shape) != shape
            or buffers[name].dtype != "float32"
            for name, shape in protocol.observation_shapes.items()
        ):
            raise ValueError(
                "training exposure input buffers differ from the frozen protocol"
            )
        if not 0 <= receipt.stream < protocol.budget.streams:
            raise ValueError("training exposure stream is outside the frozen collector")
        key = receipt.game.seed
        if key in seen:
            raise ValueError("duplicate training game exposure")
        seen.add(key)
        for player in receipt.game.trace.player_configs:
            if player != players_by_name.get(player.name):
                raise ValueError(
                    "training exposure player setup is outside the frozen tickets"
                )
        report.receipts.append(receipt.sha256)
        if not receipt.game.terminal:
            report.interrupted_games += 1
            continue
        report.terminal_games += 1
        report.terminal_games_by_stream[receipt.stream] = (
            report.terminal_games_by_stream.get(receipt.stream, 0) + 1
        )
        counts.add(receipt.game.cards)
    per_stream = protocol.budget.updates * protocol.budget.games_per_stream
    if any(count > per_stream for count in report.terminal_games_by_stream.values()):
        raise ValueError("training capture exceeds a frozen stream budget")
    if report.terminal_games > report.expected_games:
        raise ValueError("training capture exceeds the frozen game budget")
    report.decisions_by_seat = (counts.decisions[0], counts.decisions[1])
    report.cards = [counts.cards[key] for key in sorted(counts.cards)]
    report.actions = [counts.actions[key] for key in sorted(counts.actions)]
    report.unattributed_frames = counts.unattributed_frames
    if report.terminal_games != report.expected_games:
        report.issues.append("Expected terminal training captures are missing.")
    if report.interrupted_games:
        report.issues.append(
            "Interrupted training prefixes are present; terminal coverage is incomplete."
        )
    report.complete = not report.issues
    return report
