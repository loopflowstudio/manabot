"""Bind semantic programs to real structured engine decisions.

The adapter joins the viewer-safe protocol frame, the immutable semantic pack,
and one exact Rust-owned structured offer set. Model code may score the
resulting rows, but only Rust decodes their addresses and mutates the game.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal, Mapping

from etude.experience_protocol import Command, ExperienceFrame
from manabot.sim.structured_policy import (
    DecodedSubmission,
    RaggedOfferBatch,
    StructuredPolicyError,
    flatten_projection,
)

from .learning import (
    DEFAULT_IR_PATH,
    DEFAULT_SCHEMA_PATH,
    BoundSemanticPack,
    SemanticProjectionError,
)


class SemanticDecisionError(SemanticProjectionError):
    """A real engine decision cannot bind to the admitted semantic surface."""


@dataclass(frozen=True)
class RuntimeSubject:
    """A typed wire address; addresses are joins, never model features."""

    kind: Literal["object", "stack", "player"]
    entity: int
    incarnation: int = 0


@dataclass(frozen=True)
class RuntimeObjectRow:
    """One visible runtime object joined to its immutable typed programs."""

    subject: RuntimeSubject
    object_kind: Literal["card", "permanent", "stack_ability"]
    definition_row: int
    program_rows: tuple[int, ...]
    viewer_role: int
    viewer_slot: int
    opaque_identity_id: int
    controller: int
    zone: str


@dataclass(frozen=True)
class SubjectBinding:
    """A source or candidate subject resolved to a runtime row or player."""

    subject: RuntimeSubject
    object_row: int | None
    player_index: int | None


@dataclass(frozen=True)
class SemanticDecision:
    """One revision-bound semantic decision and its private Rust authority."""

    frame: ExperienceFrame
    offer_set: Any
    batch: RaggedOfferBatch
    objects: tuple[RuntimeObjectRow, ...]
    offer_sources: tuple[SubjectBinding | None, ...]
    candidate_subjects: tuple[SubjectBinding, ...]
    semantic_pack_hash: str

    def command(
        self,
        submission: DecodedSubmission,
        *,
        command_id: str | None = None,
    ) -> Command:
        """Wrap one decoded ID selection in the current protocol revision."""

        self._validate_submission(submission.offer_id, submission.answers)
        prompt = self.frame.prompt
        if prompt is None:
            raise SemanticDecisionError("semantic decision frame has no prompt")
        return Command(
            command_id=command_id
            or f"{self.frame.match_id}:{self.frame.revision}:{submission.offer_id}",
            match_id=self.frame.match_id,
            expected_revision=self.frame.revision,
            prompt_id=prompt.id,
            offer_id=submission.offer_id,
            answers=list(submission.answers),
        )

    def submission_json(self, command: Command) -> str:
        """Lower only engine-minted offer, role, and candidate IDs."""

        prompt = self.frame.prompt
        if prompt is None:
            raise SemanticDecisionError("semantic decision frame has no prompt")
        if command.match_id != self.frame.match_id:
            raise SemanticDecisionError("command match does not match bound frame")
        if command.expected_revision != self.frame.revision:
            raise SemanticDecisionError("command revision does not match bound frame")
        if command.prompt_id != prompt.id:
            raise SemanticDecisionError("command prompt does not match bound frame")
        answers = tuple(answer.model_dump(mode="json") for answer in command.answers)
        self._validate_submission(command.offer_id, answers)
        return json.dumps(
            {"offer_id": command.offer_id, "answers": list(answers)},
            sort_keys=True,
            separators=(",", ":"),
        )

    def step(self, env: Any, command: Command) -> tuple[Any, ...]:
        """Apply the selected command atomically through the bound offer set."""

        return env.step_structured(self.offer_set, self.submission_json(command))

    def _validate_submission(
        self, offer_id: int, answers: tuple[Mapping[str, Any], ...]
    ) -> None:
        offer_index = next(
            (
                index
                for index, offer in enumerate(self.batch.offers)
                if int(offer["id"]) == offer_id
            ),
            None,
        )
        if offer_index is None:
            raise SemanticDecisionError(f"offer {offer_id} is absent from bound frame")

        rows = self.batch.choices[
            self.batch.choice_offsets[offer_index] : self.batch.choice_offsets[
                offer_index + 1
            ]
        ]
        answer_by_role: dict[int, Mapping[str, Any]] = {}
        for answer in answers:
            if answer.get("kind") != "candidates":
                raise SemanticDecisionError("only candidate answers are admitted")
            role = _integer(answer.get("role"), "answer.role")
            if role in answer_by_role:
                raise SemanticDecisionError(f"duplicate answer role {role}")
            answer_by_role[role] = answer
        if set(answer_by_role) != {row.role for row in rows}:
            raise SemanticDecisionError("command answers do not match offer roles")

        for row in rows:
            raw_candidates = answer_by_role[row.role].get("candidates")
            if not isinstance(raw_candidates, list):
                raise SemanticDecisionError("answer.candidates must be a list")
            selected = [
                _integer(value, "answer.candidates[]") for value in raw_candidates
            ]
            if len(selected) != len(set(selected)):
                raise SemanticDecisionError("answer contains duplicate candidates")
            if not row.minimum <= len(selected) <= row.maximum:
                raise SemanticDecisionError(
                    f"role {row.role} requires {row.minimum}..={row.maximum} candidates"
                )
            admitted = {
                _integer(self.batch.candidates[index].get("id"), "candidate.id")
                for index in range(row.candidate_start, row.candidate_stop)
            }
            unknown = set(selected) - admitted
            if unknown:
                raise SemanticDecisionError(
                    f"role {row.role} contains unknown candidates {sorted(unknown)}"
                )


class SemanticDecisionAdapter:
    """Build fail-closed semantic decisions from live managym observations."""

    _ACTION_TYPES = {
        "cast": "PRIORITY_CAST_SPELL",
        "play_land": "PRIORITY_PLAY_LAND",
        "pass_priority": "PRIORITY_PASS_PRIORITY",
        "declare_attackers": "DECLARE_ATTACKER",
    }
    _FRAME_CORE_FIELDS = (
        "protocol",
        "match_id",
        "revision",
        "content_hash",
        "asset_manifest_hash",
        "status",
        "prompt",
        "projection",
        "offers",
    )

    def __init__(self, pack: BoundSemanticPack) -> None:
        self.pack = pack

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        schema_path: str = str(DEFAULT_SCHEMA_PATH),
        ir_path: str = str(DEFAULT_IR_PATH),
    ) -> "SemanticDecisionAdapter":
        return cls(
            BoundSemanticPack.from_env(
                env,
                schema_path=schema_path,
                ir_path=ir_path,
            )
        )

    def bind(
        self,
        env: Any,
        observation: Any,
        *,
        match_id: str,
        revision: int,
        content_hash: str,
        asset_manifest_hash: str,
    ) -> SemanticDecision:
        """Join one viewer frame, semantic catalog, and exact offer authority."""

        manifest = env.content_pack_manifest()
        if manifest.get("content_digest") != self.pack.content_pack_hash:
            raise SemanticDecisionError(
                "environment content pack changed after binding"
            )

        # Validate every visible definition, including stack abilities, before
        # asking model code to score any rows.
        self.pack.project_observation(observation, identity_mode="semantic_only")
        objects, object_index, players = self._runtime_objects(observation)

        offer_set = env.structured_offers()
        try:
            projection = json.loads(offer_set.projection_json())
        except (json.JSONDecodeError, TypeError) as error:
            raise SemanticDecisionError(
                "structured offer projection is invalid JSON"
            ) from error
        if not isinstance(projection, Mapping):
            raise SemanticDecisionError("structured offer projection must be an object")
        actor = _integer(projection.get("actor"), "projection.actor")
        if actor != int(observation.agent.player_index):
            raise SemanticDecisionError("structured offer actor is not the viewer")

        wire_projection = self._wire_projection(projection)
        try:
            batch = flatten_projection(wire_projection)
        except StructuredPolicyError as error:
            raise SemanticDecisionError(str(error)) from error

        offer_sources = tuple(
            self._bind_subject(
                offer.get("source"), object_index, players, optional=True
            )
            for offer in batch.offers
        )
        candidate_subjects = tuple(
            self._bind_candidate(candidate, object_index, players)
            for candidate in batch.candidates
        )
        frame = self._frame(
            observation,
            wire_projection,
            match_id=match_id,
            revision=revision,
            content_hash=content_hash,
            asset_manifest_hash=asset_manifest_hash,
        )
        return SemanticDecision(
            frame=frame,
            offer_set=offer_set,
            batch=batch,
            objects=objects,
            offer_sources=offer_sources,
            candidate_subjects=candidate_subjects,
            semantic_pack_hash=self.pack.semantic_pack_hash,
        )

    def _runtime_objects(
        self, observation: Any
    ) -> tuple[
        tuple[RuntimeObjectRow, ...],
        dict[RuntimeSubject, int],
        set[int],
    ]:
        players_by_id = {
            int(observation.agent.id): int(observation.agent.player_index),
            int(observation.opponent.id): int(observation.opponent.player_index),
        }
        rows: list[RuntimeObjectRow] = []
        row_by_subject: dict[RuntimeSubject, int] = {}

        def append(
            *,
            subject: RuntimeSubject,
            object_kind: Literal["card", "permanent", "stack_ability"],
            card_def_id: int,
            viewer_role: int,
            viewer_slot: int,
            controller_id: int,
            zone: str,
        ) -> None:
            if subject in row_by_subject:
                raise SemanticDecisionError(f"duplicate visible subject {subject}")
            try:
                controller = players_by_id[controller_id]
            except KeyError as error:
                raise SemanticDecisionError(
                    f"runtime object controller {controller_id} is not visible"
                ) from error
            definition_row = self.pack.definition_row(card_def_id)
            row_by_subject[subject] = len(rows)
            rows.append(
                RuntimeObjectRow(
                    subject=subject,
                    object_kind=object_kind,
                    definition_row=definition_row,
                    program_rows=self.pack.program_rows(definition_row),
                    viewer_role=viewer_role,
                    viewer_slot=viewer_slot,
                    opaque_identity_id=card_def_id,
                    controller=controller,
                    zone=zone,
                )
            )

        for role_name, cards, permanents in (
            (
                "agent_card",
                observation.agent_cards,
                observation.agent_permanents,
            ),
            (
                "opponent_card",
                observation.opponent_cards,
                observation.opponent_permanents,
            ),
        ):
            role = self.pack.schema.object_roles[role_name]
            battlefield_cards = [
                card for card in cards if _zone_name(card.zone) == "battlefield"
            ]
            if len(battlefield_cards) != len(permanents):
                raise SemanticDecisionError(
                    f"{role_name} battlefield cards and permanents do not align"
                )
            for slot, card in enumerate(cards):
                append(
                    subject=RuntimeSubject("object", int(card.id)),
                    object_kind="card",
                    card_def_id=int(card.registry_key),
                    viewer_role=role,
                    viewer_slot=slot,
                    controller_id=int(card.owner_id),
                    zone=_zone_name(card.zone),
                )
            for slot, (card, permanent) in enumerate(
                zip(battlefield_cards, permanents, strict=True)
            ):
                append(
                    subject=RuntimeSubject("object", int(permanent.id)),
                    object_kind="permanent",
                    card_def_id=int(card.registry_key),
                    viewer_role=role,
                    viewer_slot=slot,
                    controller_id=int(permanent.controller_id),
                    zone="battlefield",
                )

        stack_role = self.pack.schema.object_roles["stack_ability"]
        for slot, stack_object in enumerate(observation.stack_objects):
            if int(stack_object.kind) == 0:
                continue
            append(
                subject=RuntimeSubject("stack", int(stack_object.stack_object_id)),
                object_kind="stack_ability",
                card_def_id=int(stack_object.source_card_registry_key),
                viewer_role=stack_role,
                viewer_slot=slot,
                controller_id=int(stack_object.controller_id),
                zone="stack",
            )
        return tuple(rows), row_by_subject, set(players_by_id.values())

    def _wire_projection(self, projection: Mapping[str, Any]) -> dict[str, Any]:
        wire = json.loads(json.dumps(projection))
        for offer in wire.get("offers", []):
            verb = offer.get("verb")
            try:
                offer["action_type"] = self._ACTION_TYPES[verb]
            except KeyError as error:
                raise SemanticDecisionError(
                    f"unadmitted offer verb {verb!r}"
                ) from error
            offer["focus"] = self._focus(offer)
        return wire

    @staticmethod
    def _focus(offer: Mapping[str, Any]) -> list[int]:
        focus: list[int] = []
        source = offer.get("source")
        if isinstance(source, Mapping) and source.get("kind") == "object":
            object_id = source.get("id")
            if isinstance(object_id, Mapping):
                focus.append(_integer(object_id.get("entity"), "source.id.entity"))
        if offer.get("verb") == "declare_attackers":
            for choice in offer.get("choices", []):
                initial = choice.get("candidates", {}).get("initial", [])
                for candidate in initial:
                    subject = candidate.get("value", {}).get("subject", {})
                    if subject.get("kind") == "object":
                        entity = _integer(
                            subject.get("id", {}).get("entity"),
                            "candidate.subject.id.entity",
                        )
                        if entity not in focus:
                            focus.append(entity)
        return focus

    @staticmethod
    def _bind_candidate(
        candidate: Mapping[str, Any],
        object_index: Mapping[RuntimeSubject, int],
        players: set[int],
    ) -> SubjectBinding:
        value = candidate.get("value")
        if not isinstance(value, Mapping) or value.get("kind") != "subject":
            raise SemanticDecisionError("only subject candidates are admitted")
        return SemanticDecisionAdapter._bind_subject(
            value.get("subject"), object_index, players, optional=False
        )

    @staticmethod
    def _bind_subject(
        raw: Any,
        object_index: Mapping[RuntimeSubject, int],
        players: set[int],
        *,
        optional: bool,
    ) -> SubjectBinding | None:
        if raw is None and optional:
            return None
        if not isinstance(raw, Mapping):
            raise SemanticDecisionError("subject must be an object")
        kind = raw.get("kind")
        if kind == "object":
            object_id = raw.get("id")
            if not isinstance(object_id, Mapping):
                raise SemanticDecisionError("object subject id must be an object")
            subject = RuntimeSubject(
                "object",
                _integer(object_id.get("entity"), "subject.id.entity"),
                _integer(object_id.get("incarnation"), "subject.id.incarnation"),
            )
            try:
                row = object_index[subject]
            except KeyError as error:
                raise SemanticDecisionError(
                    f"object subject {subject} is absent from the viewer frame"
                ) from error
            return SubjectBinding(subject, object_row=row, player_index=None)
        if kind == "stack":
            subject = RuntimeSubject("stack", _integer(raw.get("id"), "subject.id"))
            try:
                row = object_index[subject]
            except KeyError as error:
                raise SemanticDecisionError(
                    f"stack subject {subject.entity} is absent from the viewer frame"
                ) from error
            return SubjectBinding(subject, object_row=row, player_index=None)
        if kind == "player":
            player = _integer(raw.get("id"), "subject.id")
            if player not in players:
                raise SemanticDecisionError(
                    f"player subject {player} is absent from the viewer frame"
                )
            return SubjectBinding(
                RuntimeSubject("player", player),
                object_row=None,
                player_index=player,
            )
        raise SemanticDecisionError(f"unadmitted subject kind {kind!r}")

    @classmethod
    def _frame(
        cls,
        observation: Any,
        projection: Mapping[str, Any],
        *,
        match_id: str,
        revision: int,
        content_hash: str,
        asset_manifest_hash: str,
    ) -> ExperienceFrame:
        # This existing evidence helper owns the exact protocol-v1 viewer
        # projection. Keep its native-engine dependency lazy so compiler-only
        # imports of manabot.semantic work before managym is built.
        from manabot.sim.teacher1_evidence import (  # noqa: PLC0415
            build_viewer_frame,
            canonical_sha256,
        )

        frame = build_viewer_frame(
            observation,
            match_id=match_id,
            revision=revision,
            content_hash=content_hash,
            asset_manifest_hash=asset_manifest_hash,
        )
        frame["offers"] = list(projection["offers"])
        frame["prompt"]["actor"] = int(projection["actor"])
        frame["prompt"]["kind"] = str(projection["kind"])
        core = {field: frame[field] for field in cls._FRAME_CORE_FIELDS}
        frame["frame_hash"] = canonical_sha256(core)
        return ExperienceFrame.model_validate(frame)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticDecisionError(f"{field} must be a non-negative integer")
    return value


def _zone_name(value: Any) -> str:
    names = (
        "library",
        "hand",
        "battlefield",
        "graveyard",
        "stack",
        "exile",
        "command",
    )
    try:
        return names[int(value)]
    except (IndexError, TypeError, ValueError) as error:
        raise SemanticDecisionError(f"unknown runtime zone {value!r}") from error


__all__ = [
    "RuntimeObjectRow",
    "RuntimeSubject",
    "SemanticDecision",
    "SemanticDecisionAdapter",
    "SemanticDecisionError",
    "SubjectBinding",
]
