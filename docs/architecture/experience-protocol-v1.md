# Experience protocol: concrete v1 code sketch

Status: the positional-action adapter is integrated and its recovery/command
slice is certified across Rust, Python, and TypeScript. The richer semantic
projection, non-empty presentation vocabulary, and structured offer examples
below remain the target design rather than a claim about current runtime code.

The executable contract lives in `managym/src/experience.rs`. It generates
`protocol/experience-v1.schema.json`; Rust, the live Python producer, and the
TypeScript consumer all validate the same fixture in
`protocol/fixtures/bolt-target.json`. See `protocol/README.md` for the exact
certification boundary and regeneration command.

This makes the proposed `ExperienceFrame`, `InteractionOffer`/`Command`,
`PresentationEvent`, and `RecoveryEnvelope` concrete. The Rust definitions are
the intended authority/wire contract. The Python and TypeScript fragments show
how the current FastAPI/Svelte path can adopt the contract incrementally before
the rules engine itself emits structured offers.

The non-negotiable invariants are:

1. a frame and its offers describe exactly one authoritative revision;
2. an offer ID is meaningful only inside its match, revision, and prompt;
3. commands are atomic, revision-bound, prompt-bound, and idempotent;
4. presentation can be skipped without changing authoritative truth;
5. recovery always has a complete viewer-safe frame;
6. the client never receives or reconstructs hidden authoritative state.

## Rust authority and wire types

This is deliberately a narrow generated boundary, not a serialization of
`GameState` or the learning observation.

```rust
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

macro_rules! numeric_id {
    ($name:ident, $inner:ty) => {
        #[derive(
            Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq,
            PartialOrd, Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub $inner);
    };
}

macro_rules! string_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd,
            Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

numeric_id!(ProtocolVersion, u16);
numeric_id!(Revision, u64);
numeric_id!(PromptId, u64);
numeric_id!(OfferId, u32);
numeric_id!(CandidateId, u32);
numeric_id!(CandidateSourceId, u32);
numeric_id!(RoleId, u16);
numeric_id!(PlayerId, u8);
numeric_id!(CardDefId, u32);
numeric_id!(StackRenderId, u64);
numeric_id!(PresentationSeq, u64);
numeric_id!(PresentationGroupId, u64);
numeric_id!(ReplayCursor, u64);

string_id!(MatchId);
string_id!(CommandId);       // Client-generated UUID/ULID.
string_id!(ContentHash);
string_id!(AssetManifestHash);
string_id!(FrameHash);
string_id!(CheckpointId);
string_id!(PaymentPlanId);

/// Exact visible incarnation. A blink keeps the entity and increments the
/// incarnation, so old animation targets and command candidates become stale.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Serialize)]
pub struct ObjectRenderId {
    pub entity: u32,
    pub incarnation: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubjectRef {
    Object { id: ObjectRenderId },
    Stack { id: StackRenderId },
    Player { id: PlayerId },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZoneKind {
    Library,
    Hand,
    Battlefield,
    Graveyard,
    Stack,
    Exile,
    Command,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Serialize)]
pub struct ZoneRef {
    pub kind: ZoneKind,
    pub owner: Option<PlayerId>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CounterView {
    pub kind: String,
    pub count: i32,
}

/// A visible public object. Do not create entries for unknown hand/library
/// cards. A publicly visible face-down object may exist with definition/name
/// omitted.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ObjectView {
    pub id: ObjectRenderId,
    pub definition: Option<CardDefId>,
    pub name: Option<String>,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub zone: ZoneRef,
    pub tapped: bool,
    pub attacking: bool,
    pub blocking: bool,
    pub summoning_sick: bool,
    pub power: Option<i32>,
    pub toughness: Option<i32>,
    pub damage: Option<i32>,
    pub counters: Vec<CounterView>,
    pub badges: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ZoneView {
    pub zone: ZoneRef,
    /// Rules-significant order only. Presentation ordering is client state.
    pub visible_objects: Vec<ObjectRenderId>,
    pub hidden_count: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlayerView {
    pub id: PlayerId,
    pub display_name: String,
    pub life: i32,
    pub mana: BTreeMap<String, u16>,
    pub active: bool,
    pub priority: bool,
    pub alive: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TurnView {
    pub number: u32,
    pub active_player: PlayerId,
    pub phase: String,
    pub step: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ViewerProjection {
    pub viewer: PlayerId,
    pub game_over: bool,
    pub winner: Option<PlayerId>,
    pub turn: TurnView,
    pub players: Vec<PlayerView>,
    pub zones: Vec<ZoneView>,
    pub objects: Vec<ObjectView>,
    pub stack: Vec<StackRenderId>,
    pub notices: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityStatus {
    Ready,
    Thinking,
    Resolving,
    Reconnecting,
    GameOver,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PromptView {
    pub id: PromptId,
    pub actor: PlayerId,
    pub kind: String,
    pub title: String,
    pub instruction: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OfferVerb {
    Cast,
    PlayLand,
    Activate,
    PassPriority,
    DeclareAttackers,
    DeclareBlockers,
    Choose,
    Pay,
    Special,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateValue {
    Subject { subject: SubjectRef },
    Mode { key: String },
    PaymentPlan { id: PaymentPlanId },
    Boolean { value: bool },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Candidate {
    pub id: CandidateId,
    pub value: CandidateValue,
    pub label: String,
    pub help: Option<String>,
    pub preview: Option<String>,
}

/// Candidate sets may depend on earlier answers. If `initial` is absent, the
/// client makes a pure ChoiceDraftQuery after filling `depends_on`. That query
/// never mutates MatchState and never exposes a partial rules action.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CandidateSource {
    pub id: CandidateSourceId,
    pub depends_on: Vec<RoleId>,
    pub initial: Option<Vec<Candidate>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChoiceStep {
    Select {
        role: RoleId,
        label: String,
        candidates: CandidateSource,
        min: u16,
        max: u16,
        ordered: bool,
        distinct: bool,
    },
    Number {
        role: RoleId,
        label: String,
        min: i32,
        max: i32,
    },
    Assign {
        role: RoleId,
        label: String,
        sources: CandidateSource,
        destinations: CandidateSource,
        min_per_source: u16,
        max_per_source: u16,
    },
    Order {
        role: RoleId,
        label: String,
        candidates: CandidateSource,
    },
    Payment {
        role: RoleId,
        label: String,
        plans: CandidateSource,
        allow_auto: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InteractionOffer {
    pub id: OfferId,
    pub actor: PlayerId,
    pub verb: OfferVerb,
    pub source: Option<SubjectRef>,
    pub label: String,
    pub help: Option<String>,
    pub choices: Vec<ChoiceStep>,
    pub confirm_label: String,
}

/// Complete, atomic, viewer-safe state at one revision. Offers and prompt are
/// never delivered separately from their projection.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExperienceFrame {
    pub protocol: ProtocolVersion,
    pub match_id: MatchId,
    pub revision: Revision,
    /// Hash only this viewer-safe projection, prompt, and offers. Never expose
    /// a canonical hidden-state hash as a comparison oracle.
    pub frame_hash: FrameHash,
    pub content_hash: ContentHash,
    pub asset_manifest_hash: AssetManifestHash,
    pub status: AuthorityStatus,
    pub prompt: Option<PromptView>,
    pub projection: ViewerProjection,
    pub offers: Vec<InteractionOffer>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChoiceAnswer {
    Candidates {
        role: RoleId,
        candidates: Vec<CandidateId>,
    },
    Number {
        role: RoleId,
        value: i32,
    },
    Assignments {
        role: RoleId,
        pairs: Vec<(CandidateId, CandidateId)>,
    },
    Order {
        role: RoleId,
        candidates: Vec<CandidateId>,
    },
    Payment {
        role: RoleId,
        plan: PaymentPlanId,
    },
}

/// A pure query used only when later candidates depend on earlier answers.
/// It does not advance revision, emit events, or expose the draft to opponents.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChoiceDraftQuery {
    pub match_id: MatchId,
    pub expected_revision: Revision,
    pub prompt_id: PromptId,
    pub offer_id: OfferId,
    pub answers: Vec<ChoiceAnswer>,
    pub candidate_source: CandidateSourceId,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChoiceDraftResult {
    pub revision: Revision,
    pub prompt_id: PromptId,
    pub offer_id: OfferId,
    pub candidate_source: CandidateSourceId,
    pub candidates: Vec<Candidate>,
}

/// The only game-mutating client request.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Command {
    pub command_id: CommandId,
    pub match_id: MatchId,
    pub expected_revision: Revision,
    pub prompt_id: PromptId,
    pub offer_id: OfferId,
    pub answers: Vec<ChoiceAnswer>,
}
```

### Presentation and recovery types

```rust
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationImportance {
    Ambient,
    Normal,
    Emphasized,
    Critical,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PresentationKind {
    MoveZone {
        old: ObjectRenderId,
        /// The new rules incarnation, or None if the viewer must not be able
        /// to follow the object into its destination.
        new: Option<ObjectRenderId>,
        from: ZoneRef,
        to: ZoneRef,
    },
    Cast {
        object: ObjectRenderId,
        controller: PlayerId,
        stack: StackRenderId,
    },
    Damage {
        source: Option<SubjectRef>,
        target: SubjectRef,
        amount: i32,
    },
    LifeChanged {
        player: PlayerId,
        delta: i32,
        new_total: i32,
    },
    Tapped {
        object: ObjectRenderId,
        tapped: bool,
    },
    CounterChanged {
        object: ObjectRenderId,
        counter: String,
        delta: i32,
        new_total: i32,
    },
    AttackGroup {
        attackers: Vec<ObjectRenderId>,
        defender: SubjectRef,
    },
    Blocked {
        attacker: ObjectRenderId,
        blockers: Vec<ObjectRenderId>,
    },
    Reveal {
        objects: Vec<ObjectRenderId>,
        audience: Vec<PlayerId>,
    },
    Destroyed {
        objects: Vec<ObjectRenderId>,
    },
    Message {
        text: String,
    },
}

/// Viewer-safe meaning for animation/audio. This is not the rules DomainEvent
/// and the client does not infer it by diffing frames.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PresentationEvent {
    pub seq: PresentationSeq,
    pub from_revision: Revision,
    pub to_revision: Revision,
    pub caused_by: Option<CommandId>,
    pub group: PresentationGroupId,
    pub importance: PresentationImportance,
    pub suggested_ms: u32,
    pub sound: Option<String>,
    pub kind: PresentationKind,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CommandReceipt {
    pub command_id: CommandId,
    pub actor: PlayerId,
    pub accepted_at: Revision,
    pub resulting_revision: Revision,
    pub resulting_frame_hash: FrameHash,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FrameUpdate {
    /// The last frame on which this update is based. A mismatch requests full
    /// recovery instead of attempting a speculative patch/merge.
    pub base_revision: Revision,
    pub frame: ExperienceFrame,
    pub presentation: Vec<PresentationEvent>,
    pub receipt: Option<CommandReceipt>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryReason {
    InitialConnect,
    ExplicitResync,
    RevisionGap,
    Reconnect,
    DuplicateCommand,
    StaleCommand,
    AuthorityRestart,
}

/// Full-state recovery is permanent. Deltas may optimize steady state later,
/// but every client can always converge from this envelope.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RecoveryEnvelope {
    pub protocol: ProtocolVersion,
    pub engine_version: String,
    pub content_hash: ContentHash,
    pub asset_manifest_hash: AssetManifestHash,
    pub reason: RecoveryReason,
    pub frame: ExperienceFrame,
    /// Already filtered for this viewer; hidden destinations and private
    /// reveals are absent or redacted.
    pub presentation_tail: Vec<PresentationEvent>,
    /// Recent receipts for commands submitted by this authenticated actor,
    /// sufficient for retry/idempotency. Do not broadcast other players'
    /// private command identifiers.
    pub accepted_commands: Vec<CommandReceipt>,
    pub replay_cursor: ReplayCursor,
    pub checkpoint: Option<CheckpointId>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectCode {
    WrongMatch,
    StaleRevision,
    StalePrompt,
    UnknownOffer,
    InvalidSelection,
    NotActor,
    AuthorityBusy,
    ProtocolMismatch,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CommandRejection {
    pub command_id: CommandId,
    pub code: RejectCode,
    pub message: String,
    pub current_revision: Revision,
    pub current_prompt: Option<PromptId>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CommandOutcome {
    Accepted { update: FrameUpdate },
    Duplicate {
        receipt: CommandReceipt,
        recovery: RecoveryEnvelope,
    },
    Rejected {
        rejection: CommandRejection,
        recovery: Option<RecoveryEnvelope>,
    },
}
```

## Authority acceptance path

The server keeps the wire offer and an internal lowering together. The client
never sends an engine action index or an object ID that was not offered.

```rust
struct PublishedPrompt {
    revision: Revision,
    prompt_id: PromptId,
    wire_offers: Vec<InteractionOffer>,
    internal_offers: BTreeMap<OfferId, InternalOffer>,
}

impl MatchAuthority {
    fn submit(&mut self, command: Command, actor: PlayerId) -> CommandOutcome {
        // Network retry after "accepted, response lost": return evidence of the
        // original commit and a current full frame; never apply twice.
        if let Some(receipt) = self
            .accepted_commands
            .get(&(actor, command.command_id.clone()))
        {
            return CommandOutcome::Duplicate {
                receipt: receipt.clone(),
                recovery: self.recovery(RecoveryReason::DuplicateCommand, actor),
            };
        }

        if command.match_id != self.match_id {
            return self.reject(command, RejectCode::WrongMatch, actor, false);
        }

        let published = match &self.published_prompt {
            Some(published) => published,
            None => {
                return self.reject(command, RejectCode::AuthorityBusy, actor, true);
            }
        };

        if command.expected_revision != published.revision {
            return self.reject(command, RejectCode::StaleRevision, actor, true);
        }
        if command.prompt_id != published.prompt_id {
            return self.reject(command, RejectCode::StalePrompt, actor, true);
        }

        let Some(offer) = published.internal_offers.get(&command.offer_id) else {
            return self.reject(command, RejectCode::UnknownOffer, actor, true);
        };
        if offer.actor != actor {
            return self.reject(command, RejectCode::NotActor, actor, false);
        }

        // Resolves CandidateIds through the server-owned offer, checks every
        // cardinality/dependency/payment constraint, and returns one atomic
        // rules command. It does not trust CandidateValue echoed by the client.
        let rules_command = match offer.lower(&command.answers, &self.game) {
            Ok(command) => command,
            Err(message) => {
                return self.reject_with_message(
                    command,
                    RejectCode::InvalidSelection,
                    message,
                    actor,
                    false,
                );
            }
        };

        let base_revision = self.revision;

        // apply_atomic guarantees failure is a no-op. Auto-passes and the AI
        // may advance through more internal revisions before the next surfaced
        // player decision.
        let domain_events = match self.game.apply_atomic(rules_command) {
            Ok(events) => events,
            Err(error) => {
                return self.reject_with_message(
                    command,
                    RejectCode::InvalidSelection,
                    error.to_string(),
                    actor,
                    true,
                );
            }
        };

        self.advance_until_surface();
        self.revision = self.revision.next();
        self.publish_prompt();

        let frame = self.project_frame(actor);
        let presentation = self.project_presentation(
            actor,
            base_revision,
            frame.revision,
            Some(command.command_id.clone()),
            domain_events,
        );
        let receipt = CommandReceipt {
            command_id: command.command_id.clone(),
            actor,
            accepted_at: base_revision,
            resulting_revision: frame.revision,
            resulting_frame_hash: frame.frame_hash.clone(),
        };

        self.accepted_commands
            .insert((actor, command.command_id), receipt.clone());
        self.trim_idempotency_window();

        CommandOutcome::Accepted {
            update: FrameUpdate {
                base_revision,
                frame,
                presentation,
                receipt: Some(receipt),
            },
        }
    }
}
```

`apply_atomic` can initially mean “clone compact state, apply, replace on
success.” Later it may use the mark/rollback transaction contract. The wire
semantics do not change.

## Incremental bridge over today's positional actions

The first production slice does not need the full choice grammar. Wrap each
current `ActionOption.index` in a revision-bound server-side offer and accept an
empty-answer `Command`. This immediately removes the stale-index/double-click
bug while preserving the current engine and UI.

```python
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PublishedPrompt:
    revision: int
    prompt_id: int
    action_by_offer: dict[int, int]
    offers: list[dict[str, Any]]


class GameSession:
    def __init__(self, trace_dir=None):
        # Existing initialization omitted.
        self.revision = 0
        self.next_prompt_id = 1
        self.published_prompt: PublishedPrompt | None = None
        self.accepted_commands: dict[str, dict[str, Any]] = {}

    def _publish_current_prompt(self) -> PublishedPrompt:
        # Call only when a new authoritative player decision is surfaced.
        actions = describe_actions(self.obs)
        prompt_id = self.next_prompt_id
        self.next_prompt_id += 1

        action_by_offer = {offer_id: offer_id for offer_id in range(len(actions))}
        offers = [
            {
                "id": offer_id,
                "actor": 0,
                "verb": action["type"].lower(),
                "source": None,
                "label": action["description"],
                "help": None,
                "choices": [],
                "confirm_label": action["description"],
                # focus becomes the first direct-manipulation bridge.
                "focus": action["focus"],
            }
            for offer_id, action in enumerate(actions)
        ]
        self.published_prompt = PublishedPrompt(
            revision=self.revision,
            prompt_id=prompt_id,
            action_by_offer=action_by_offer,
            offers=offers,
        )
        return self.published_prompt

    def current_recovery(self, reason: str) -> dict[str, Any]:
        prompt = self.published_prompt or self._publish_current_prompt()
        return {
            "protocol": 1,
            "reason": reason,
            "frame": self._experience_frame(prompt),
            "presentation_tail": [],
            "accepted_commands": list(self.accepted_commands.values())[-64:],
            "replay_cursor": len(self.trace.events),
            "checkpoint": None,
        }

    def hero_command(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("command must be an object")

        command_id = raw.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("command_id must be a non-empty string")

        # Idempotent retry: report, never execute again.
        if command_id in self.accepted_commands:
            return {
                "type": "command_outcome",
                "status": "duplicate",
                "receipt": self.accepted_commands[command_id],
                "recovery": self.current_recovery("duplicate_command"),
            }

        prompt = self.published_prompt
        if prompt is None:
            return self._reject(raw, "authority_busy", recover=True)
        if raw.get("expected_revision") != prompt.revision:
            return self._reject(raw, "stale_revision", recover=True)
        if raw.get("prompt_id") != prompt.prompt_id:
            return self._reject(raw, "stale_prompt", recover=True)

        offer_id = raw.get("offer_id")
        action_index = prompt.action_by_offer.get(offer_id)
        if action_index is None:
            return self._reject(raw, "unknown_offer", recover=True)
        if raw.get("answers") not in (None, []):
            return self._reject(raw, "invalid_selection", recover=False)

        base_revision = self.revision
        actions = describe_actions(self.obs)
        self._step_and_record("hero", action_index, actions)
        self._advance()

        # One wire revision per newly published authoritative surface. Internal
        # engine steps may be more numerous.
        self.revision += 1
        self.published_prompt = None
        next_prompt = self._publish_current_prompt()
        frame = self._experience_frame(next_prompt)
        receipt = {
            "command_id": command_id,
            "actor": 0,
            "accepted_at": base_revision,
            "resulting_revision": self.revision,
            "resulting_frame_hash": frame["frame_hash"],
        }
        self.accepted_commands[command_id] = receipt

        return {
            "type": "command_outcome",
            "status": "accepted",
            "update": {
                "base_revision": base_revision,
                "frame": frame,
                "presentation": [],
                "receipt": receipt,
            },
        }
```

Important migration detail: `current_message()` and reconnect must reuse the
existing `published_prompt`; they must not mint a new prompt ID for the same
decision. Commands are never placed in the current offline outbound queue.
Reconnect obtains recovery first, after which the player may select again.

## Generated TypeScript surface and client use

The TypeScript declarations should be generated from the protocol schema, not
hand-maintained. The generated `Command` shape is conceptually:

```ts
export interface Command {
  command_id: string;
  match_id: string;
  expected_revision: number;
  prompt_id: number;
  offer_id: number;
  answers: ChoiceAnswer[];
}

export type CommandOutcome =
  | { status: 'accepted'; update: FrameUpdate }
  | {
      status: 'duplicate';
      receipt: CommandReceipt;
      recovery: RecoveryEnvelope;
    }
  | {
      status: 'rejected';
      rejection: CommandRejection;
      recovery?: RecoveryEnvelope;
    };
```

The client has one authoritative frame writer and does not queue gameplay
commands while disconnected:

```ts
class ExperienceProtocolClient {
  private frame: ExperienceFrame | null = null;
  private inFlightCommand: string | null = null;

  submit(offer: InteractionOffer, answers: ChoiceAnswer[]): void {
    const frame = this.frame;
    const prompt = frame?.prompt;
    if (!frame || !prompt || this.inFlightCommand !== null) return;

    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      // Never replay a game command selected from a stale frame. Reconnect and
      // recover first; settings/telemetry may use a separate safe queue.
      this.reconnect();
      return;
    }

    const command: Command = {
      command_id: crypto.randomUUID(),
      match_id: frame.match_id,
      expected_revision: frame.revision,
      prompt_id: prompt.id,
      offer_id: offer.id,
      answers,
    };

    this.inFlightCommand = command.command_id;
    this.socket.send(JSON.stringify({ type: 'command', command }));
  }

  private applyOutcome(outcome: CommandOutcome): void {
    if (outcome.status === 'accepted') {
      this.inFlightCommand = null;
      this.applyUpdate(outcome.update);
      return;
    }

    if (outcome.status === 'duplicate') {
      this.inFlightCommand = null;
      this.applyRecovery(outcome.recovery);
      return;
    }

    this.inFlightCommand = null;
    if (outcome.recovery) {
      this.applyRecovery(outcome.recovery);
    }
    this.showCommandError(outcome.rejection);
  }

  private applyUpdate(update: FrameUpdate): void {
    const current = this.frame;

    // Sequence gate: old responses cannot overwrite a newer recovery/frame.
    if (current && update.frame.revision <= current.revision) return;

    // A gap is not patched optimistically. Ask for the complete truth.
    if (current && update.base_revision !== current.revision) {
      this.requestRecovery('revision_gap');
      return;
    }

    // One atomic store commit. The frame already contains its matching offers.
    this.frame = update.frame;
    gameStore.commitFrame(update.frame);

    // Presentation is optional theater over committed truth. It may play,
    // accelerate, skip, or be discarded on recovery.
    presentationStore.enqueue(update.presentation, update.frame.revision);
  }

  private applyRecovery(recovery: RecoveryEnvelope): void {
    const current = this.frame;
    if (current && recovery.frame.revision < current.revision) return;

    presentationStore.cancelAll();
    this.frame = recovery.frame;
    gameStore.commitFrame(recovery.frame);
    presentationStore.resumeTail(
      recovery.presentation_tail,
      recovery.frame.revision,
    );
  }
}
```

Direct board manipulation and the inspector both call `submit` with the same
`InteractionOffer`. A drag gesture never manufactures a separate rules action.

## Concrete Lightning Bolt exchange

The relevant excerpt of the frame for “cast Lightning Bolt, target opponent”
looks like:

```json
{
  "protocol": 1,
  "match_id": "match_01K...",
  "revision": 42,
  "prompt": {
    "id": 19,
    "actor": 0,
    "kind": "priority",
    "title": "Your priority",
    "instruction": "Choose an action"
  },
  "offers": [
    {
      "id": 7,
      "actor": 0,
      "verb": "cast",
      "source": {
        "kind": "object",
        "id": { "entity": 31, "incarnation": 0 }
      },
      "label": "Cast Lightning Bolt",
      "choices": [
        {
          "kind": "select",
          "role": 1,
          "label": "Target",
          "candidates": {
            "id": 3,
            "depends_on": [],
            "initial": [
              {
                "id": 11,
                "value": {
                  "kind": "subject",
                  "subject": { "kind": "player", "id": 1 }
                },
                "label": "Opponent"
              }
            ]
          },
          "min": 1,
          "max": 1,
          "ordered": false,
          "distinct": true
        }
      ],
      "confirm_label": "Cast"
    }
  ]
}
```

The client submits only IDs minted by that prompt:

```json
{
  "command_id": "0194f7d8-...",
  "match_id": "match_01K...",
  "expected_revision": 42,
  "prompt_id": 19,
  "offer_id": 7,
  "answers": [
    { "kind": "candidates", "role": 1, "candidates": [11] }
  ]
}
```

If revision 42 is still current, the authority resolves offer 7/candidate 11
through its server-owned map and applies one atomic cast command. If revision 43
already exists, the command is rejected as stale and includes a recovery
envelope. It is never reinterpreted as “whatever action 7 means now.”

## What belongs where

| Concern | Canonical rules state | Experience frame | Client-only | Research sidecar |
|---|---:|---:|---:|---:|
| Object incarnation/zone/controller | yes | viewer-safe projection | no | optional |
| Current prompt and legal offers | semantic continuation | yes | draft only | yes |
| Partial target/payment selection | no | no | yes | policy decoder trace |
| Domain events and trigger facts | yes/bounded ledger | no | no | optional |
| Presentation timing/particles/layout | no | semantic events only | yes | no |
| Search scores/policy logits | no | optional safe summary | analysis mode | yes |
| Hidden opponent cards/RNG | yes | never | never | privileged, access-controlled |
| Replay command/checkpoint cursor | no | recovery metadata | playback cursor | yes |

## Recommended implementation slices

1. Add revision, stable prompt, stable offer, command ID, and command dedupe
   around current positional actions. Stop queueing gameplay commands offline.
2. Make one atomic `ExperienceFrame` the live/replay client input and generate
   its bindings/schema.
3. Add `PresentationEvent` for move/cast/damage/die/tap/reveal while continuing
   to send complete recovery frames.
4. Replace current one-step offers with structured `ChoiceStep`s for targets,
   attackers/blockers, modes, and payments. Keep completed engine commands
   atomic.
5. Persist `RecoveryEnvelope` plus command/checkpoint replay records outside the
   process-local session registry.

This order gives immediate correctness before requiring the new semantic kernel.
The later kernel can implement the same protocol without a client rewrite.
