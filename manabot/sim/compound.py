"""Execute learned compound decisions through managym's canonical Commands.

Native lowering owns legality and the decomposition. The policy sees only the
root observation and public offer projection. A queued suffix contains no model
inputs and is discarded on interruption; revisions prevent stale execution.
"""

from collections import deque
from dataclasses import dataclass
import json
from time import perf_counter

import torch
from torch import Tensor

from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.model.compound import CompoundOutput
from manabot.sim.structured_policy import RaggedOfferBatch, flatten_projection
import managym
from managym.decision import Command, DecisionFrame


@dataclass(frozen=True)
class CompoundDecision:
    observation: dict[str, Tensor]
    offers: RaggedOfferBatch
    output: CompoundOutput
    commands: tuple[Command, ...]
    seconds: float


def sample_compound(
    agent: Agent,
    engine: managym.Env,
    raw: managym.Observation,
    *,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
) -> CompoundDecision:
    """Read a root and lower its sampled submission without mutating the match."""
    start = perf_counter()
    offers = engine.compound_offers()
    batch = flatten_projection(json.loads(offers.projection_json()))
    space = agent.observation_space
    if len(raw.action_space.actions) > space.encoder.max_actions:
        space = ObservationSpace(
            space.encoder.hypers.model_copy(
                update={
                    "max_actions": len(raw.action_space.actions),
                }
            )
        )
    observation = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in space.encode(raw).items()
    }
    output = agent.compound(
        observation,
        batch,
        generator=generator,
        deterministic=deterministic,
    )
    # JSON is emitted by the native Command serializer; parse at this boundary.
    payload = json.loads(
        engine.compound_commands_json(offers, output.submission.to_json())
    )
    commands = tuple(Command(**row) for row in payload)
    if not commands:
        raise ValueError("compound submission lowered to no Commands")
    return CompoundDecision(
        observation, batch, output, commands, perf_counter() - start
    )


class CompoundPolicy:
    """Ordinary action-index consumer backed by one learned compound policy.

    GameSession and arena keep their existing canonical Command recording. This
    adapter supplies each already-decided microchoice, never resamples a suffix.
    """

    def __init__(self, agent: Agent, *, deterministic: bool = False) -> None:
        self.agent = agent
        self.deterministic = deterministic
        self.pending: deque[Command] = deque()
        self.engine: managym.Env | None = None
        self.decisions = 0
        self.microchoices = 0
        self.seconds = 0.0

    def reset(self) -> None:
        self.pending.clear()
        self.engine = None

    def act(self, engine: managym.Env, observation: managym.Observation) -> int:
        if self.engine is not engine:
            self.reset()
            self.engine = engine
        frame = DecisionFrame.from_json(engine.semantic_decision_frame_json())
        if not self.pending:
            with torch.no_grad():
                decision = sample_compound(
                    self.agent,
                    engine,
                    observation,
                    deterministic=self.deterministic,
                )
            self.pending.extend(decision.commands)
            self.decisions += 1
            self.seconds += decision.seconds
        command = self.pending.popleft()
        if command.expected_revision != frame.revision:
            self.reset()
            raise ValueError("compound sequence interrupted by another decision")
        self.microchoices += 1
        return next(
            index
            for index, offer in enumerate(frame.offers)
            if offer["id"] == command.offer_id
        )
