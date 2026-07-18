// decision.rs
// Shared semantic match contract: a revision-bound DecisionFrame, an atomic
// revision-bound Command, a composite viewer-safe Observation, and a
// fail-closed TransitionReceipt. This is the R1 vertical slice of
// docs/ARCHITECTURE.md: one common contract projected from the authoritative
// Game, with positional action indices kept private. It does not own match
// identity (Etude-owned), replay bookkeeping, or possible-world semantics
// (RUL-8 / R3). Legal-action identity comes from `structured_search_offers`,
// which covers every action-space kind as one offer per legal action.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    agent::{
        observation::{EventData, Observation as AgentObservation},
        structured_offer::{
            ChoiceAnswer, InteractionOffer, OfferId, OfferSubmission, PublicCommitment,
        },
    },
    flow::game::Game,
    state::{
        game_object::{ObjectLookupError, PlayerId},
        zone::ZoneType,
    },
};

/// Canonical serialization and digest contract for the semantic decision
/// slice. Increment when DecisionFrame/Observation field inclusion, ordering,
/// or digest algorithm changes.
pub const SEMANTIC_DECISION_VERSION: u16 = 4;

/// Stable digest of one complete legal offer set at a revision. Any change to
/// the legal actions, their order, or their binding revision changes it.
pub type DecisionFingerprint = String;
/// Stable digest of one committed viewer-visible event.
pub type EventIdentity = String;

/// Public authorization address for one object-bearing candidate. Only opaque
/// candidate identity crosses the wire; the exact ObjectRef remains bound in
/// the match authority.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ObjectCandidateAddress {
    pub decision_fingerprint: DecisionFingerprint,
    pub offer_id: u32,
    pub role: u16,
    pub candidate_id: u32,
}

/// One revision-bound semantic decision: the actor, a fingerprint of the full
/// legal offer set, and the offers themselves. Offers are projected from the
/// authoritative action space; positional indices are not part of this
/// contract.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecisionFrame {
    pub schema_version: u16,
    pub revision: u64,
    pub actor: u8,
    pub fingerprint: DecisionFingerprint,
    pub offers: Vec<InteractionOffer>,
    #[serde(default)]
    pub object_candidates: Vec<ObjectCandidateAddress>,
}

/// One atomic semantic commitment. `expected_revision` binds it to one exact
/// decision frame; a stale or cross-decision command fails closed before any
/// mutation. `answers` carries every choice knowable at commitment time.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Command {
    pub command_id: String,
    pub expected_revision: u64,
    pub offer_id: u32,
    pub answers: Vec<ChoiceAnswer>,
    #[serde(default)]
    pub object_preconditions: Vec<ObjectCandidateAddress>,
}

/// Typed identity for one composite Observation. `viewer_state_hash` digests
/// the viewer-safe projection only; it never includes opponent-private state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ObservationIdentity {
    pub schema_version: u16,
    pub revision: u64,
    pub viewer: u8,
    pub viewer_state_hash: String,
}

/// The complete viewer-safe legal input at one revision: current visible
/// state, the ordered newly-visible event increment, and the current semantic
/// decision (only when this viewer is the actor). Model tensors and histories
/// are derived from this, not the reverse.
#[derive(Clone, Debug, Serialize)]
pub struct Observation {
    pub identity: ObservationIdentity,
    /// Viewer-safe projection from `agent::Observation::for_player`. Opponent
    /// private hands and a non-acting viewer's candidates are suppressed.
    pub viewer_state: Value,
    pub events: Vec<EventIdentity>,
    pub decision: Option<DecisionFrame>,
}

/// Fail-closed record of one accepted transition. Produced only after a
/// command validates against `expected_revision`, decodes against the current
/// offer set, and applies atomically. `next_decision` is the fingerprint of
/// the following decision frame, or `None` at a terminal state.
#[derive(Clone, Debug, Serialize)]
pub struct TransitionReceipt {
    pub schema_version: u16,
    pub before_revision: u64,
    pub after_revision: u64,
    pub command_id: String,
    pub public_commitment: Option<PublicCommitment>,
    pub events: Vec<EventIdentity>,
    pub next_decision: Option<DecisionFingerprint>,
}

/// The outcome of one semantic execution: the fail-closed receipt plus the
/// next composite Observation for the command's actor.
#[derive(Clone, Debug, Serialize)]
pub struct SemanticTransition {
    pub receipt: TransitionReceipt,
    pub observation: Observation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticError {
    NoMatch,
    GameOver,
    NoActiveDecision,
    ViewerOutOfBounds(usize),
    ViewerSafetyViolation,
    StaleRevision { expected: u64, current: u64 },
    UnknownOffer(u32),
    UnknownObjectCandidate,
    StaleObject,
    IllegalCommand(String),
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMatch => write!(f, "no active match"),
            Self::GameOver => write!(f, "match is over"),
            Self::NoActiveDecision => write!(f, "no active decision to project"),
            Self::ViewerOutOfBounds(viewer) => {
                write!(f, "viewer {viewer} is out of bounds")
            }
            Self::ViewerSafetyViolation => {
                write!(f, "viewer-safe projection leaked opponent-private state")
            }
            Self::StaleRevision { expected, current } => {
                write!(
                    f,
                    "stale_revision: expected revision {expected}, authority at {current}"
                )
            }
            Self::UnknownOffer(offer) => {
                write!(f, "offer {offer} is absent from the current frame")
            }
            Self::UnknownObjectCandidate => write!(
                f,
                "unknown_object_candidate: candidate address is not bound by this authority"
            ),
            Self::StaleObject => write!(f, "stale_object: bound object is no longer current"),
            Self::IllegalCommand(message) => write!(f, "illegal command: {message}"),
        }
    }
}

impl std::error::Error for SemanticError {}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Canonical digest of one complete decision frame projection.
fn decision_fingerprint(revision: u64, projection_value: &Value) -> String {
    let payload = json!({
        "schema_version": SEMANTIC_DECISION_VERSION,
        "revision": revision,
        "projection": projection_value,
    });
    sha256_hex(&serde_json::to_vec(&payload).expect("fingerprint payload serializes"))
}

#[derive(Serialize)]
struct CanonicalEventIdentity {
    event_type: i32,
    source_kind: i32,
    source_id: i32,
    target_kind: i32,
    target_id: i32,
    amount: i32,
    controller_id: i32,
    from_zone: i32,
    to_zone: i32,
    source_incarnation: i32,
    target_incarnation: i32,
}

impl From<&EventData> for CanonicalEventIdentity {
    fn from(event: &EventData) -> Self {
        let mut canonical = Self {
            event_type: event.event_type,
            source_kind: event.source_kind,
            source_id: event.source_id,
            target_kind: event.target_kind,
            target_id: event.target_id,
            amount: event.amount,
            controller_id: event.controller_id,
            from_zone: event.from_zone,
            to_zone: event.to_zone,
            source_incarnation: event.source_incarnation,
            target_incarnation: event.target_incarnation,
        };
        let moves_into_hidden_zone = event.from_zone >= 0
            && matches!(
                event.to_zone,
                zone if zone == ZoneType::Hand as i32 || zone == ZoneType::Library as i32
            );
        if moves_into_hidden_zone {
            canonical.source_kind = 0;
            canonical.source_id = -1;
            canonical.source_incarnation = -1;
        }
        canonical
    }
}

fn event_identity(event: &EventData) -> EventIdentity {
    let canonical = CanonicalEventIdentity::from(event);
    sha256_hex(&serde_json::to_vec(&canonical).expect("event identity serializes"))
}

impl Game {
    /// Project the current semantic decision frame from the authoritative
    /// action space. Fails closed on a terminal state or when no external
    /// decision is published.
    pub fn semantic_decision_frame(&self) -> Result<DecisionFrame, SemanticError> {
        if self.is_game_over() {
            return Err(SemanticError::GameOver);
        }
        let offers = self
            .structured_search_offers()
            .map_err(|error| match error {
                crate::agent::structured_offer::StructuredOfferError::NoActiveActionSpace
                | crate::agent::structured_offer::StructuredOfferError::GameOver => {
                    SemanticError::NoActiveDecision
                }
                other => SemanticError::IllegalCommand(other.to_string()),
            })?;
        let projection = offers.projection();
        let projection_value = serde_json::to_value(projection).expect("projection serializes");
        let fingerprint = decision_fingerprint(self.decision_epoch, &projection_value);
        let mut object_candidates = Vec::new();
        if let Ok(presentation) = self.structured_offers() {
            let mut private = self.semantic_object_candidates.borrow_mut();
            for (offer_id, role, candidate_id, object_ref) in
                presentation.object_candidate_bindings()
            {
                let address = ObjectCandidateAddress {
                    decision_fingerprint: fingerprint.clone(),
                    offer_id: offer_id.0,
                    role: role.0,
                    candidate_id: candidate_id.0,
                };
                private.insert(address.clone(), object_ref);
                object_candidates.push(address);
            }
        }
        Ok(DecisionFrame {
            schema_version: SEMANTIC_DECISION_VERSION,
            revision: self.decision_epoch,
            actor: projection.actor,
            fingerprint,
            offers: projection_value
                .get("offers")
                .and_then(Value::as_array)
                .map(|array| {
                    array
                        .iter()
                        .map(|value| {
                            serde_json::from_value(value.clone())
                                .expect("offer round-trips through canonical value")
                        })
                        .collect()
                })
                .expect("projection carries offers"),
            object_candidates,
        })
    }

    /// Project the composite viewer-safe Observation at the current revision.
    /// `decision` is present only when `viewer` is the acting player, so a
    /// non-acting viewer never sees the actor's legal offers.
    pub fn semantic_observation(&self, viewer: PlayerId) -> Result<Observation, SemanticError> {
        if viewer.0 >= self.state.players.len() {
            return Err(SemanticError::ViewerOutOfBounds(viewer.0));
        }
        let agent_obs = AgentObservation::for_player(self, viewer);
        if !agent_obs
            .opponent_cards
            .iter()
            .all(|card| card.zone != ZoneType::Hand)
        {
            return Err(SemanticError::ViewerSafetyViolation);
        }
        let viewer_state_json = agent_obs.to_json();
        let viewer_state: Value =
            serde_json::from_str(&viewer_state_json).expect("agent observation is valid json");
        let viewer_state_hash = sha256_hex(viewer_state_json.as_bytes());

        let decision = if self.is_game_over() {
            None
        } else {
            let actor_is_viewer = self
                .current_action_space
                .as_ref()
                .and_then(|space| space.player)
                == Some(viewer);
            if actor_is_viewer {
                self.semantic_decision_frame().ok()
            } else {
                None
            }
        };

        Ok(Observation {
            identity: ObservationIdentity {
                schema_version: SEMANTIC_DECISION_VERSION,
                revision: self.decision_epoch,
                viewer: viewer.0 as u8,
                viewer_state_hash,
            },
            viewer_state,
            events: Vec::new(),
            decision,
        })
    }

    /// Validate one revision-bound Command against the current authority,
    /// apply it atomically, and return the fail-closed receipt plus the next
    /// composite Observation for the command's actor. No mutation occurs on
    /// any error path.
    pub fn execute_semantic_command(
        &mut self,
        command: &Command,
    ) -> Result<SemanticTransition, SemanticError> {
        self.execute_semantic_command_with_observation(command)
            .map(|(transition, _, _)| transition)
    }

    pub(crate) fn execute_semantic_command_with_observation(
        &mut self,
        command: &Command,
    ) -> Result<(SemanticTransition, AgentObservation, bool), SemanticError> {
        if self.is_game_over() {
            return Err(SemanticError::GameOver);
        }
        if command.expected_revision != self.decision_epoch {
            return Err(SemanticError::StaleRevision {
                expected: command.expected_revision,
                current: self.decision_epoch,
            });
        }

        let offers = self
            .structured_search_offers()
            .map_err(|error| match error {
                crate::agent::structured_offer::StructuredOfferError::NoActiveActionSpace
                | crate::agent::structured_offer::StructuredOfferError::GameOver => {
                    SemanticError::NoActiveDecision
                }
                other => SemanticError::IllegalCommand(other.to_string()),
            })?;
        let projection = offers.projection();
        let actor = PlayerId(projection.actor as usize);
        let public_commitment = projection
            .offers
            .iter()
            .find(|offer| offer.id.0 == command.offer_id)
            .ok_or(SemanticError::UnknownOffer(command.offer_id))?
            .public_commitment
            .clone();

        let submission = OfferSubmission {
            offer_id: OfferId(command.offer_id),
            answers: command.answers.clone(),
        };

        let atomic = offers
            .decode(&submission)
            .map_err(|error| SemanticError::IllegalCommand(error.to_string()))?;
        for address in &command.object_preconditions {
            let object_ref = self
                .semantic_object_candidates
                .borrow()
                .get(address)
                .copied()
                .ok_or(SemanticError::UnknownObjectCandidate)?;
            match self.lookup_current_permanent(object_ref) {
                Ok(_) => {}
                Err(ObjectLookupError::StaleIncarnation | ObjectLookupError::WrongZone) => {
                    return Err(SemanticError::StaleObject)
                }
                Err(ObjectLookupError::MissingEntity) => {
                    return Err(SemanticError::UnknownObjectCandidate)
                }
            }
        }

        let before_revision = self.decision_epoch;
        let done = self
            .apply_atomic_command(&atomic)
            .map_err(|error| SemanticError::IllegalCommand(error.to_string()))?;
        let after_revision = self.decision_epoch;

        let events = self.take_observation_events();
        let event_identities = events
            .iter()
            .flat_map(AgentObservation::event_data)
            .map(|event| event_identity(&event))
            .collect();

        let agent_observation = AgentObservation::new(self, &events);
        let observation = self.semantic_observation(actor)?;
        let next_decision = if done {
            None
        } else {
            self.semantic_decision_frame()
                .ok()
                .map(|frame| frame.fingerprint)
        };

        let transition = SemanticTransition {
            receipt: TransitionReceipt {
                schema_version: SEMANTIC_DECISION_VERSION,
                before_revision,
                after_revision,
                command_id: command.command_id.clone(),
                public_commitment,
                events: event_identities,
                next_decision,
            },
            observation,
        };
        Ok((transition, agent_observation, done))
    }

    /// Read-only cursor over committed rules events. Rejected Commands must
    /// leave this and the deterministic state witness unchanged.
    pub fn semantic_event_cursor(&self) -> u64 {
        self.state.events.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::structured_offer::{CandidateId, ChoiceAnswer, OfferVerb, RoleId},
        state::player::PlayerConfig,
        state::zone::ZoneType,
    };
    use std::collections::BTreeMap;

    fn interactive_deck() -> BTreeMap<String, usize> {
        BTreeMap::from([
            ("Island".to_string(), 12),
            ("Mountain".to_string(), 12),
            ("Gray Ogre".to_string(), 6),
            ("Wind Drake".to_string(), 6),
            ("Man-o'-War".to_string(), 4),
            ("Raging Goblin".to_string(), 4),
            ("Lightning Bolt".to_string(), 6),
            ("Counterspell".to_string(), 4),
            ("Ancestral Recall".to_string(), 3),
            ("Pyroclasm".to_string(), 3),
        ])
    }

    fn configs() -> Vec<PlayerConfig> {
        vec![
            PlayerConfig::new("hero", interactive_deck()),
            PlayerConfig::new("villain", interactive_deck()),
        ]
    }

    fn first_game() -> Game {
        Game::new(configs(), 11, true)
    }

    fn pass_command(frame: &DecisionFrame, id: &str) -> Command {
        let pass = frame
            .offers
            .iter()
            .find(|offer| offer.verb == OfferVerb::PassPriority)
            .expect("a pass priority offer is always legal");
        Command {
            command_id: id.to_string(),
            expected_revision: frame.revision,
            offer_id: pass.id.0,
            answers: Vec::new(),
            object_preconditions: Vec::new(),
        }
    }

    #[test]
    fn decision_frame_is_revision_bound_with_stable_fingerprint() {
        let game = first_game();
        let frame = game
            .semantic_decision_frame()
            .expect("first decision frame");
        assert_eq!(frame.schema_version, SEMANTIC_DECISION_VERSION);
        assert_eq!(frame.revision, game.decision_epoch);
        assert!(!frame.offers.is_empty());
        assert_eq!(frame.fingerprint.len(), 64);
        assert!(frame
            .fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));

        let again = game.semantic_decision_frame().expect("re-project");
        assert_eq!(frame.fingerprint, again.fingerprint);
    }

    #[test]
    fn receipt_carries_the_provider_owned_public_commitment() {
        let mut game = first_game();
        let frame = game.semantic_decision_frame().expect("first frame");
        let land = frame
            .offers
            .iter()
            .find_map(|offer| match &offer.public_commitment {
                Some(PublicCommitment::PlayLand { card }) => Some((offer.id.0, card.clone())),
                _ => None,
            })
            .expect("seeded opener has a public land commitment");
        let command = Command {
            command_id: "public-land".to_string(),
            expected_revision: frame.revision,
            offer_id: land.0,
            answers: Vec::new(),
            object_preconditions: Vec::new(),
        };

        let transition = game
            .execute_semantic_command(&command)
            .expect("land commitment executes");

        assert_eq!(
            transition.receipt.public_commitment,
            Some(PublicCommitment::PlayLand { card: land.1 })
        );
    }

    #[test]
    fn event_identity_masks_cards_moving_into_hidden_zones() {
        let hidden_move = |source_id| EventData {
            event_type: 1,
            source_kind: 1,
            source_id,
            target_kind: 0,
            target_id: -1,
            amount: 0,
            controller_id: 1,
            from_zone: ZoneType::Library as i32,
            to_zone: ZoneType::Hand as i32,
            source_incarnation: source_id,
            target_incarnation: -1,
        };
        assert_eq!(
            event_identity(&hidden_move(17)),
            event_identity(&hidden_move(29)),
        );

        let mut public_move_a = hidden_move(17);
        public_move_a.to_zone = ZoneType::Graveyard as i32;
        let mut public_move_b = hidden_move(29);
        public_move_b.to_zone = ZoneType::Graveyard as i32;
        assert_ne!(
            event_identity(&public_move_a),
            event_identity(&public_move_b)
        );
    }

    #[test]
    fn observation_is_viewer_safe_and_decision_only_for_actor() {
        let game = first_game();
        let actor = game
            .current_action_space
            .as_ref()
            .and_then(|space| space.player)
            .expect("an actor holds the first decision");

        let acting = game
            .semantic_observation(actor)
            .expect("acting viewer observation");
        assert_eq!(acting.identity.schema_version, SEMANTIC_DECISION_VERSION);
        assert_eq!(acting.identity.viewer, actor.0 as u8);
        assert!(acting.decision.is_some(), "acting viewer sees the decision");
        let hand_zone = (ZoneType::Hand as i32) as i64;
        let opponent_hand: Vec<&Value> = acting.viewer_state["opponent_cards"]
            .as_array()
            .expect("opponent_cards array")
            .iter()
            .filter(|card| card["zone"].as_i64() == Some(hand_zone))
            .collect();
        assert!(
            opponent_hand.is_empty(),
            "viewer state must not expose opponent-private hand"
        );

        let other = PlayerId((actor.0 + 1) % 2);
        let nonacting = game
            .semantic_observation(other)
            .expect("non-acting viewer observation");
        assert!(
            nonacting.decision.is_none(),
            "non-acting viewer must not see the actor's decision"
        );
    }

    #[test]
    fn execute_advances_revision_and_emits_fail_closed_receipt() {
        let mut game = first_game();
        let frame = game.semantic_decision_frame().expect("frame");
        let before = game.decision_epoch;
        let command = pass_command(&frame, "cmd-1");
        let SemanticTransition {
            receipt,
            observation,
        } = game
            .execute_semantic_command(&command)
            .expect("command applies");

        assert_eq!(receipt.schema_version, SEMANTIC_DECISION_VERSION);
        assert_eq!(receipt.before_revision, before);
        assert!(
            receipt.after_revision > before,
            "publishing a new decision advances the revision"
        );
        assert_eq!(receipt.command_id, "cmd-1");
        assert_eq!(
            receipt.public_commitment,
            Some(PublicCommitment::PassPriority)
        );
        assert_eq!(observation.identity.revision, receipt.after_revision);
        assert!(receipt.next_decision.is_some(), "match is not terminal");
    }

    #[test]
    fn stale_unknown_and_illegal_commands_fail_closed_without_mutation() {
        let mut game = first_game();
        let frame = game.semantic_decision_frame().expect("frame");
        let stale_revision = game.decision_epoch;

        let advance = pass_command(&frame, "cmd-advance");
        game.execute_semantic_command(&advance)
            .expect("advance past the first decision");
        let current_revision = game.decision_epoch;
        assert!(current_revision > stale_revision);

        let stale = Command {
            command_id: "stale".to_string(),
            expected_revision: stale_revision,
            offer_id: frame.offers[0].id.0,
            answers: Vec::new(),
            object_preconditions: Vec::new(),
        };
        let error = game.execute_semantic_command(&stale).unwrap_err();
        assert!(matches!(
            error,
            SemanticError::StaleRevision { expected, current }
            if expected == stale_revision && current == current_revision
        ));
        assert_eq!(game.decision_epoch, current_revision);

        let current_frame = game.semantic_decision_frame().expect("current frame");
        let unknown = Command {
            command_id: "unknown".to_string(),
            expected_revision: current_frame.revision,
            offer_id: u32::MAX,
            answers: Vec::new(),
            object_preconditions: Vec::new(),
        };
        let error = game.execute_semantic_command(&unknown).unwrap_err();
        assert!(matches!(error, SemanticError::UnknownOffer(u32::MAX)));
        assert_eq!(game.decision_epoch, current_frame.revision);

        let illegal = Command {
            command_id: "illegal".to_string(),
            expected_revision: current_frame.revision,
            offer_id: current_frame.offers[0].id.0,
            answers: vec![ChoiceAnswer::Candidates {
                role: RoleId(1),
                candidates: vec![CandidateId(0)],
            }],
            object_preconditions: Vec::new(),
        };
        let error = game.execute_semantic_command(&illegal).unwrap_err();
        assert!(matches!(error, SemanticError::IllegalCommand(_)));
        assert_eq!(game.decision_epoch, current_frame.revision);
    }

    #[test]
    fn fingerprint_changes_when_the_legal_offer_set_changes() {
        let mut game = first_game();
        let first = game.semantic_decision_frame().expect("frame").fingerprint;
        let pass = pass_command(&game.semantic_decision_frame().expect("frame"), "c");
        game.execute_semantic_command(&pass).expect("advance");
        let next = game
            .semantic_decision_frame()
            .expect("next frame")
            .fingerprint;
        assert_ne!(first, next, "a new decision has a different fingerprint");
    }

    #[test]
    fn authored_match_parity_exact_object_rejections_are_atomic() {
        let mut game = first_game();
        game.scenario_clear_hand(PlayerId(0));
        game.scenario_clear_hand(PlayerId(1));
        game.scenario_force_card_in_hand(PlayerId(0), "Lightning Bolt")
            .expect("bolt in hand");
        game.scenario_force_battlefield(PlayerId(0), "Mountain", true)
            .expect("casting mana");
        let permanent = game
            .scenario_force_battlefield(PlayerId(1), "Gray Ogre", true)
            .expect("object candidate");
        game.scenario_refresh_priority()
            .expect("refreshed priority");

        let object_ref = game
            .permanent_object_ref(permanent)
            .expect("exact current object");
        let frame = game.semantic_decision_frame().expect("semantic frame");
        let address = frame
            .object_candidates
            .iter()
            .find(|address| {
                game.semantic_object_candidates
                    .borrow()
                    .get(*address)
                    .is_some_and(|bound| *bound == object_ref)
            })
            .cloned()
            .expect("candidate privately binds exact object");
        let rendered = game.structured_offers().expect("structured offer");
        let rendered_json = serde_json::to_value(rendered.projection()).expect("projection json");
        assert!(rendered_json
            .to_string()
            .contains(&format!("\"incarnation\":{}", object_ref.incarnation.0)));

        let card = game.state.permanents[permanent]
            .as_ref()
            .expect("permanent")
            .card;
        game.move_card(card, ZoneType::Graveyard);
        let current = game.semantic_decision_frame().expect("current frame");
        let pass = current
            .offers
            .iter()
            .find(|offer| offer.verb == OfferVerb::PassPriority)
            .expect("pass offer");
        let stale_object = Command {
            command_id: "stale-object".to_string(),
            expected_revision: current.revision,
            offer_id: pass.id.0,
            answers: Vec::new(),
            object_preconditions: vec![address],
        };
        let witness = game.state.deterministic_hash_value();
        let cursor = game.semantic_event_cursor();
        assert!(matches!(
            game.execute_semantic_command(&stale_object),
            Err(SemanticError::StaleObject)
        ));
        assert_eq!(game.state.deterministic_hash_value(), witness);
        assert_eq!(game.semantic_event_cursor(), cursor);

        let stale_revision = stale_object.expected_revision;
        let advance = Command {
            object_preconditions: Vec::new(),
            command_id: "advance".to_string(),
            ..stale_object.clone()
        };
        game.execute_semantic_command(&advance).expect("advance");
        let stale = Command {
            command_id: "stale-revision".to_string(),
            ..advance
        };
        let witness = game.state.deterministic_hash_value();
        let cursor = game.semantic_event_cursor();
        assert!(matches!(
            game.execute_semantic_command(&stale),
            Err(SemanticError::StaleRevision { expected, current })
                if expected == stale_revision && current > expected
        ));
        assert_eq!(game.state.deterministic_hash_value(), witness);
        assert_eq!(game.semantic_event_cursor(), cursor);
    }
}
