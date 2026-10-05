"""Canonical conditional policy rows from a retained compound decoder root.

CompoundCursor keeps the original viewer tensors, native offers and forced
prefix across canonical microsteps. Projection partitions the next Command for
independent binary attackers and optional/single-target roles, verifying every
route through native lowering. No declaration subsets are enumerated. Payment
subsets are deliberately rejected: their ordered lowering can give zero mass to
otherwise legal micro-actions, outside the current local-update target contract.
"""

from collections import deque
from dataclasses import dataclass
import json
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from manabot.model.agent import Agent
from manabot.model.compound import CompoundOutput
from manabot.sim.search_branch import SelectedFullCloneBackend
from manabot.sim.structured_policy import RaggedOfferBatch, flatten_projection
import managym
from managym.decision import Command, DecisionFrame, Observation


@dataclass(frozen=True)
class CompoundPrefixReceipt:
    root_observation_json: str
    projection_json: str
    tokens: tuple[int, ...]
    commands: tuple[Command, ...]


@dataclass(frozen=True)
class CompoundRoot:
    engine: managym.Env
    observation: dict[str, Tensor]
    offers: RaggedOfferBatch
    observation_json: str
    projection_json: str


@dataclass(frozen=True)
class CompoundCursor:
    root: CompoundRoot
    tokens: tuple[int, ...] = ()
    commands: tuple[Command, ...] = ()

    def receipt(self) -> CompoundPrefixReceipt:
        return CompoundPrefixReceipt(
            self.root.observation_json,
            self.root.projection_json,
            self.tokens,
            self.commands,
        )


@dataclass(frozen=True)
class CompoundChoice:
    prefix: tuple[int, ...]
    command: Command
    probability: float


@dataclass(frozen=True)
class CompoundProjection:
    cursor: CompoundCursor
    choices: tuple[CompoundChoice, ...]  # canonical action order
    value: float  # signed acting-seat prefix value

    @property
    def probabilities(self) -> NDArray[np.float64]:
        return np.asarray([choice.probability for choice in self.choices])

    def action_index(self, command: Command) -> int:
        for index, choice in enumerate(self.choices):
            if _routes_match(command, choice.command):
                return index
        raise ValueError(
            "sampled compound suffix does not match current legal Commands"
        )


def _routes_match(left: Command, right: Command) -> bool:
    return (
        left.expected_revision == right.expected_revision
        and left.offer_id == right.offer_id
        and left.answers == right.answers
        and left.object_preconditions == right.object_preconditions
    )


def _decode(
    agent: Agent,
    cursor: CompoundCursor,
    prefix: tuple[int, ...],
    check: Callable[[], None],
    *,
    generator: torch.Generator | None = None,
) -> CompoundOutput:
    check()
    with torch.inference_mode():
        output = agent.compound(
            cursor.root.observation,
            cursor.root.offers,
            prefix=prefix,
            deterministic=generator is None,
            generator=generator,
        )
    check()
    return output


def _lower(
    cursor: CompoundCursor, output: CompoundOutput, check: Callable[[], None]
) -> tuple[Command, ...]:
    check()
    engine = cursor.root.engine
    rows = json.loads(
        engine.compound_commands_json(
            engine.compound_offers(), output.submission.to_json()
        )
    )
    commands = tuple(
        Command(
            command_id=row["command_id"],
            expected_revision=row["expected_revision"],
            offer_id=row["offer_id"],
            answers=tuple(row.get("answers", ())),
            object_preconditions=tuple(row.get("object_preconditions", ())),
        )
        for row in rows
    )
    check()
    if len(commands) < len(cursor.commands) or any(
        not _routes_match(prior, replay)
        for prior, replay in zip(cursor.commands, commands)
    ):
        raise ValueError("compound prefix does not reproduce retained Commands")
    return commands


def compound_cursor(
    agent: Agent, engine: managym.Env, check: Callable[[], None]
) -> CompoundCursor:
    check()
    actor = int(engine.current_agent_index())
    raw = engine.observation_for_player(actor)
    encoded = agent.observation_space.encode(raw)
    if int(encoded["actions_valid"].sum()) != len(raw.action_space.actions):
        raise ValueError("compound search exceeds checkpoint observation capacity")
    projection = engine.compound_offers().projection_json()
    offers = flatten_projection(json.loads(projection))
    if any(offer["verb"] == "pay_waterbend" for offer in offers.offers):
        raise ValueError(
            "compound search does not support ordered payment-subset boundaries"
        )
    root = CompoundRoot(
        SelectedFullCloneBackend()
        .open_session(match_id="compound-root", audit=False)
        .fork_exact(engine, "world"),
        {key: torch.as_tensor(value).unsqueeze(0) for key, value in encoded.items()},
        offers,
        engine.semantic_observation_json(actor),
        projection,
    )
    check()
    return CompoundCursor(root)


def project_compound(
    agent: Agent,
    engine: managym.Env,
    cursor: CompoundCursor | None,
    check: Callable[[], None],
) -> CompoundProjection:
    """Complete canonical action probabilities, verified on the authoritative root."""
    if cursor is None:
        cursor = compound_cursor(agent, engine, check)
    root_view = Observation.from_json(cursor.root.observation_json)
    actor = int(engine.current_agent_index())
    if actor != root_view.viewer:
        raise ValueError("compound prefix interrupted by an actor change")
    frame = DecisionFrame.from_json(engine.semantic_decision_frame_json())
    prefix_session = SelectedFullCloneBackend().open_session(
        match_id="compound-prefix", audit=False
    )
    expected = prefix_session.fork_exact(cursor.root.engine, "world")
    for command in cursor.commands:
        check()
        expected_frame = DecisionFrame.from_json(
            expected.semantic_decision_frame_json()
        )
        if expected_frame.revision != command.expected_revision:
            raise ValueError("compound prefix revision differs during replay")
        indexes = [
            index
            for index, offer in enumerate(expected_frame.offers)
            if offer["id"] == command.offer_id
        ]
        if len(indexes) != 1:
            raise ValueError("compound prefix Command is no longer legal")
        prefix_session.apply_policy_choice(
            expected, site="child", policy_index=indexes[0]
        )
    if (
        Observation.from_json(
            expected.semantic_observation_json(actor)
        ).viewer_state_hash
        != Observation.from_json(
            engine.semantic_observation_json(actor)
        ).viewer_state_hash
    ):
        raise ValueError("compound prefix belongs to another viewer state")
    offers = cursor.root.offers
    prefixes: list[tuple[int, ...]] = []
    tokens = cursor.tokens
    compound_kind = str(offers.offers[0]["verb"])
    if not tokens and compound_kind not in {"declare_attackers", "declare_blockers"}:
        # Complete ordinary/priority offers keep canonical ordering. Cast target
        # selection is a later Command from this same root, not a new forward root.
        prefixes = [(index,) for index in range(len(offers.offers))]
    else:
        tokens = tokens or (0,)  # the sole compound offer is a forced factor
        offer = offers.offers[tokens[0]]
        start, stop = offers.choice_offsets[tokens[0] : tokens[0] + 2]
        rows = offers.choices[start:stop]
        if offer["verb"] == "declare_attackers":
            if len(rows) != 1 or rows[0].minimum != 0:
                raise ValueError("unsupported compound attacker role")
            output = _decode(agent, cursor, tokens, check)
            ordinal = len(tokens)
            if ordinal >= len(output.tokens):
                raise ValueError("compound attacker prefix is already complete")
            prefixes = [
                tokens + (bit,)
                for bit in (0, 1)
                if output.probabilities[ordinal][bit] > 0
            ]
        elif offer["verb"] == "declare_blockers" or offer["verb"] == "cast":
            row_index = (
                len(cursor.commands) if offer["verb"] == "declare_blockers" else 0
            )
            if row_index >= len(rows):
                raise ValueError("compound role prefix is already complete")
            row = rows[row_index]
            if row.maximum > 1 or row.minimum not in (0, 1):
                raise ValueError("unsupported compound target cardinality")
            count = row.candidate_stop - row.candidate_start
            expected_prefix = 1 + sum(
                r.candidate_stop - r.candidate_start for r in rows[:row_index]
            )
            if len(tokens) != expected_prefix:
                raise ValueError("compound tokens do not end at a role boundary")
            prefixes = [
                tokens + tuple(int(i == selected) for i in range(count))
                for selected in range(count)
            ]
            if row.minimum == 0:
                prefixes.append(tokens + (0,) * count)
        else:
            raise ValueError("unsupported compound prefix-to-Command boundary")
    choices: dict[int, CompoundChoice] = {}
    value: float | None = None
    for prefix in prefixes:
        output = _decode(agent, cursor, prefix, check)
        tape = _lower(cursor, output, check)
        ordinal = len(cursor.commands)
        if len(tape) <= ordinal:
            raise ValueError("compound choice lowered to no current Command")
        command = tape[ordinal]
        if command.expected_revision != frame.revision:
            raise ValueError("compound prefix interrupted by a revision change")
        if command.offer_id in choices:
            raise ValueError("compound factors alias a canonical Command")
        # Unselected suffix factors integrate to one. Multiplying only newly
        # fixed conditionals yields an exact canonical conditional probability.
        probability = float(
            output.log_probs[len(cursor.tokens) : len(prefix)].double().sum().exp()
        )
        choices[command.offer_id] = CompoundChoice(prefix, command, probability)
        prefix_value = float(
            output.values[len(cursor.tokens)]
            if len(cursor.tokens) < len(output.tokens)
            else output.end_value
        )
        if value is not None and not np.isclose(
            value, prefix_value, rtol=1e-6, atol=1e-7
        ):
            raise ValueError("compound value depends on a not-yet-chosen suffix")
        value = prefix_value
    if set(choices) != {int(offer["id"]) for offer in frame.offers}:
        raise ValueError("compound projection does not cover canonical legal offers")
    aligned = tuple(choices[int(offer["id"])] for offer in frame.offers)
    probabilities = np.asarray([choice.probability for choice in aligned])
    if (
        value is None
        or not np.isfinite(value)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0)
        or not np.isclose(probabilities.sum(), 1, atol=1e-6)
    ):
        raise ValueError(
            "compound canonical distribution is not normalized positive support"
        )
    return CompoundProjection(cursor, aligned, value)


def advance_compound(
    agent: Agent,
    projection: CompoundProjection,
    action: int,
    check: Callable[[], None],
) -> CompoundCursor | None:
    """Retain the original root until its entire native Command boundary ends."""
    choice = projection.choices[action]
    cursor = CompoundCursor(
        projection.cursor.root,
        choice.prefix,
        projection.cursor.commands + (choice.command,),
    )
    output = _decode(agent, cursor, cursor.tokens, check)
    tape = _lower(cursor, output, check)
    if len(tape) == len(cursor.commands):
        if any(
            int(torch.count_nonzero(p)) != 1
            for p in output.probabilities[len(cursor.tokens) :]
        ):
            raise ValueError(
                "unconsumed stochastic factors outlive the compound boundary"
            )
        return None
    return cursor


def sample_suffix(
    agent: Agent,
    cursor: CompoundCursor,
    generator: torch.Generator,
    check: Callable[[], None],
) -> tuple[Command, ...]:
    """Sample once, then drain the native suffix without new policy decisions."""
    output = _decode(agent, cursor, cursor.tokens, check, generator=generator)
    return _lower(cursor, output, check)[len(cursor.commands) :]


class CompoundRollout:
    """Drain one declaration before admitting the next actor's policy root.

    Only one cursor can be live: changing actors inside a retained declaration
    is an interruption, checked by project_compound against its original viewer.
    """

    def __init__(
        self,
        agent: Agent,
        root: CompoundProjection,
        seed: int,
        check: Callable[[], None],
    ) -> None:
        self.agent = agent
        self.cursor: CompoundCursor | None = root.cursor
        self.pending: deque[Command] | None = None
        self.generator = torch.Generator().manual_seed(seed)
        self.check = check
        self.factors = 0

    def project(self, engine: managym.Env) -> CompoundProjection:
        return project_compound(self.agent, engine, self.cursor, self.check)

    def choose(self, projection: CompoundProjection) -> int:
        if self.pending is None:
            self.pending = deque(
                sample_suffix(self.agent, projection.cursor, self.generator, self.check)
            )
        if not self.pending:
            raise ValueError("compound rollout has an empty pending suffix")
        command = self.pending.popleft()
        return projection.action_index(command)

    def advance(self, projection: CompoundProjection, action: int) -> None:
        next_cursor = advance_compound(self.agent, projection, action, self.check)
        if next_cursor is None:
            output = _decode(
                self.agent,
                projection.cursor,
                projection.choices[action].prefix,
                self.check,
            )
            self.factors += len(output.tokens) - len(projection.cursor.tokens)
            if self.pending:
                raise ValueError("compound suffix crosses its native boundary")
            self.pending = None
        else:
            self.factors += len(next_cursor.tokens) - len(projection.cursor.tokens)
            # A forced root action has no pre-sampled tape. Draw its conditional
            # suffix exactly once; later canonical decisions only drain it.
            if self.pending is None:
                self.pending = deque(
                    sample_suffix(self.agent, next_cursor, self.generator, self.check)
                )
        self.cursor = next_cursor
