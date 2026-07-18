//! Experimental structured legal offers over the current positional action ABI.
//!
//! The current `ActionSpace` and target legality queries remain authoritative.
//! This module projects a narrow, uncapped typed view for priority pass,
//! single `CreatureOrPlayer` target casts, and complete attacker declarations,
//! then lowers accepted IDs through the existing rules executor. It
//! intentionally does not own match revisions, prompt persistence, recovery,
//! or policy decoding.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    agent::action::{Action, ActionSpaceKind},
    flow::game::Game,
    state::{
        ability::TargetSpec,
        game_object::{CardId, ObjectId, ObjectRef, PermanentId, PlayerId, Target},
        target::Target as ActionTarget,
    },
};

macro_rules! numeric_id {
    ($name:ident, $inner:ty) => {
        #[derive(
            Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub $inner);
    };
}

numeric_id!(OfferId, u32);
numeric_id!(CandidateId, u32);
numeric_id!(CandidateSourceId, u32);
numeric_id!(RoleId, u16);

/// Private authority minted with one published engine decision.
///
/// This is not the Game protocol's match revision or prompt ID. Its only job
/// is to make a decoded command unusable after the engine publishes another
/// decision, even when that later decision has superficially identical legal
/// actions. Search forks cloned from the offered state intentionally retain
/// the same binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OfferSetBinding(u64);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ObjectRenderId {
    pub entity: u32,
    pub incarnation: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubjectRef {
    Object { id: ObjectRenderId },
    Player { id: u8 },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptKind {
    Priority,
    DeclareAttackers,
    DeclareBlockers,
    ChooseTarget,
    Scry,
    LookAndSelect,
    PayOrNot,
    Modal,
    DiscardThenDraw,
    Waterbend,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OfferVerb {
    ActivateAbility,
    Cast,
    ChooseMode,
    ChooseTarget,
    DeclareAttacker,
    DeclareAttackers,
    DeclareBlocker,
    Decline,
    PassPriority,
    PayCost,
    PlayLand,
    ScryBottom,
    ScryKeep,
    SelectCard,
    WaterbendTap,
}

/// Viewer-observable identity of an admitted public commitment. This is
/// intentionally narrower than `Action`: private object IDs, positional
/// indexes, and unresolved prompt structure are excluded.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PublicCommitment {
    PassPriority,
    Cast { card: String },
    PlayLand { card: String },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateValue {
    Subject { subject: SubjectRef },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Candidate {
    pub id: CandidateId,
    pub value: CandidateValue,
    pub label: String,
    pub help: Option<String>,
    pub preview: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CandidateSource {
    pub id: CandidateSourceId,
    pub depends_on: Vec<RoleId>,
    pub initial: Option<Vec<Candidate>>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InteractionOffer {
    pub id: OfferId,
    pub actor: u8,
    pub verb: OfferVerb,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_commitment: Option<PublicCommitment>,
    pub source: Option<SubjectRef>,
    pub label: String,
    pub help: Option<String>,
    pub choices: Vec<ChoiceStep>,
    pub confirm_label: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructuredOfferProjection {
    pub actor: u8,
    pub kind: PromptKind,
    pub offers: Vec<InteractionOffer>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChoiceAnswer {
    Candidates {
        role: RoleId,
        candidates: Vec<CandidateId>,
    },
}

impl ChoiceAnswer {
    fn role(&self) -> RoleId {
        match self {
            Self::Candidates { role, .. } => *role,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OfferSubmission {
    pub offer_id: OfferId,
    pub answers: Vec<ChoiceAnswer>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AtomicCommand {
    PassPriority {
        binding: OfferSetBinding,
        player: PlayerId,
    },
    CastSpell {
        binding: OfferSetBinding,
        player: PlayerId,
        card: CardId,
        targets: Vec<BoundTarget>,
    },
    DeclareAttackers {
        binding: OfferSetBinding,
        player: PlayerId,
        attackers: Vec<ObjectRef>,
    },
    SearchAction {
        binding: OfferSetBinding,
        action: Action,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundTarget {
    Player(PlayerId),
    Permanent(ObjectRef),
    StackSpell(CardId),
}

#[derive(Clone, Debug)]
enum InternalOffer {
    PassPriority {
        player: PlayerId,
    },
    CastSingleTarget {
        player: PlayerId,
        card: CardId,
        role: RoleId,
        targets: BTreeMap<CandidateId, BoundTarget>,
    },
    DeclareAttackers {
        player: PlayerId,
        role: RoleId,
        attackers: BTreeMap<CandidateId, ObjectRef>,
        declaration_order: Vec<ObjectRef>,
    },
    SearchAction {
        action: Action,
    },
}

#[derive(Clone, Debug)]
pub struct StructuredOfferSet {
    projection: StructuredOfferProjection,
    binding: OfferSetBinding,
    internal: BTreeMap<OfferId, InternalOffer>,
}

impl StructuredOfferSet {
    pub fn projection(&self) -> &StructuredOfferProjection {
        &self.projection
    }

    pub(crate) fn object_candidate_bindings(
        &self,
    ) -> Vec<(OfferId, RoleId, CandidateId, ObjectRef)> {
        let mut bindings = Vec::new();
        for (offer_id, offer) in &self.internal {
            match offer {
                InternalOffer::CastSingleTarget { role, targets, .. } => {
                    for (candidate_id, target) in targets {
                        if let BoundTarget::Permanent(object_ref) = target {
                            bindings.push((*offer_id, *role, *candidate_id, *object_ref));
                        }
                    }
                }
                InternalOffer::DeclareAttackers {
                    role, attackers, ..
                } => {
                    bindings.extend(attackers.iter().map(|(candidate_id, object_ref)| {
                        (*offer_id, *role, *candidate_id, *object_ref)
                    }));
                }
                InternalOffer::PassPriority { .. } | InternalOffer::SearchAction { .. } => {}
            }
        }
        bindings
    }

    /// Resolve only IDs minted by this offer set. Public candidate values are
    /// presentation data and are never trusted during decoding.
    pub fn decode(
        &self,
        submission: &OfferSubmission,
    ) -> Result<AtomicCommand, StructuredOfferError> {
        let offer = self
            .internal
            .get(&submission.offer_id)
            .ok_or(StructuredOfferError::UnknownOffer(submission.offer_id))?;

        match offer {
            InternalOffer::PassPriority { player } => {
                if !submission.answers.is_empty() {
                    return Err(StructuredOfferError::UnexpectedAnswers);
                }
                Ok(AtomicCommand::PassPriority {
                    binding: self.binding,
                    player: *player,
                })
            }
            InternalOffer::CastSingleTarget {
                player,
                card,
                role,
                targets,
            } => {
                let candidates = candidate_answer(&submission.answers, *role)?;
                if candidates.len() != 1 {
                    return Err(StructuredOfferError::WrongCardinality {
                        role: *role,
                        expected: 1,
                        actual: candidates.len(),
                    });
                }

                let candidate = candidates[0];
                let target = targets
                    .get(&candidate)
                    .copied()
                    .ok_or(StructuredOfferError::UnknownCandidate(candidate))?;
                Ok(AtomicCommand::CastSpell {
                    binding: self.binding,
                    player: *player,
                    card: *card,
                    targets: vec![target],
                })
            }
            InternalOffer::DeclareAttackers {
                player,
                role,
                attackers,
                declaration_order,
            } => {
                let candidates = candidate_answer(&submission.answers, *role)?;
                if candidates.len() > attackers.len() {
                    return Err(StructuredOfferError::CardinalityOutOfRange {
                        role: *role,
                        min: 0,
                        max: attackers.len(),
                        actual: candidates.len(),
                    });
                }

                let mut selected = BTreeSet::new();
                let mut selected_objects = BTreeSet::new();
                for candidate in candidates {
                    if !selected.insert(*candidate) {
                        return Err(StructuredOfferError::DuplicateCandidate(*candidate));
                    }
                    let object_ref = attackers
                        .get(candidate)
                        .copied()
                        .ok_or(StructuredOfferError::UnknownCandidate(*candidate))?;
                    selected_objects.insert(object_ref);
                }

                // The wire choice is unordered, but rules events must retain
                // the legacy CR 508 declaration order for deterministic
                // traces. Reorder the selected set against the authoritative
                // prompt order before producing the command.
                let selected_attackers = declaration_order
                    .iter()
                    .copied()
                    .filter(|object_ref| selected_objects.contains(object_ref))
                    .collect();

                Ok(AtomicCommand::DeclareAttackers {
                    binding: self.binding,
                    player: *player,
                    attackers: selected_attackers,
                })
            }
            InternalOffer::SearchAction { action } => {
                if !submission.answers.is_empty() {
                    return Err(StructuredOfferError::UnexpectedAnswers);
                }
                Ok(AtomicCommand::SearchAction {
                    binding: self.binding,
                    action: action.clone(),
                })
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredOfferError {
    GameOver,
    NoActiveActionSpace,
    WrongDecision,
    MissingActor,
    IdentityOverflow,
    InvalidCurrentTarget,
    Invariant(String),
    UnknownOffer(OfferId),
    UnexpectedAnswers,
    MissingAnswer(RoleId),
    DuplicateRole(RoleId),
    UnexpectedRole(RoleId),
    WrongCardinality {
        role: RoleId,
        expected: usize,
        actual: usize,
    },
    CardinalityOutOfRange {
        role: RoleId,
        min: usize,
        max: usize,
        actual: usize,
    },
    DuplicateCandidate(CandidateId),
    UnknownCandidate(CandidateId),
    StaleOrIllegal(String),
}

impl std::fmt::Display for StructuredOfferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GameOver => write!(f, "game is over"),
            Self::NoActiveActionSpace => write!(f, "no active action space"),
            Self::WrongDecision => write!(f, "current decision has no structured offer support"),
            Self::MissingActor => write!(f, "priority action space has no actor"),
            Self::IdentityOverflow => write!(f, "offer identity exceeds the wire domain"),
            Self::InvalidCurrentTarget => write!(f, "a current legal target cannot be projected"),
            Self::Invariant(message) => write!(f, "structured offer invariant failed: {message}"),
            Self::UnknownOffer(offer) => write!(f, "unknown offer {}", offer.0),
            Self::UnexpectedAnswers => write!(f, "offer received unexpected answers"),
            Self::MissingAnswer(role) => write!(f, "missing answer for role {}", role.0),
            Self::DuplicateRole(role) => write!(f, "duplicate answer for role {}", role.0),
            Self::UnexpectedRole(role) => write!(f, "unexpected answer role {}", role.0),
            Self::WrongCardinality {
                role,
                expected,
                actual,
            } => write!(
                f,
                "role {} requires {expected} candidate(s), got {actual}",
                role.0
            ),
            Self::CardinalityOutOfRange {
                role,
                min,
                max,
                actual,
            } => write!(
                f,
                "role {} requires {min}..={max} candidate(s), got {actual}",
                role.0
            ),
            Self::DuplicateCandidate(candidate) => {
                write!(f, "candidate {} was selected more than once", candidate.0)
            }
            Self::UnknownCandidate(candidate) => {
                write!(f, "unknown candidate {}", candidate.0)
            }
            Self::StaleOrIllegal(message) => write!(f, "stale or illegal command: {message}"),
        }
    }
}

impl std::error::Error for StructuredOfferError {}

impl Game {
    /// Project the currently supported structured decision without changing
    /// state. Unsupported legacy decisions remain available through
    /// `ActionSpace`.
    pub fn structured_offers(&self) -> Result<StructuredOfferSet, StructuredOfferError> {
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }
        match self.current_action_space.as_ref().map(|space| space.kind) {
            Some(ActionSpaceKind::Priority) => self.structured_priority_offers(),
            Some(ActionSpaceKind::DeclareAttacker) => self.structured_attacker_offers(),
            Some(_) => Err(StructuredOfferError::WrongDecision),
            None => Err(StructuredOfferError::NoActiveActionSpace),
        }
    }

    /// Project one stable structured offer for every action in the current
    /// learner-facing policy surface.
    ///
    /// Unlike the presentation-oriented projection above, this projection is
    /// deliberately action-aligned: one policy lookup row decodes to one
    /// exact cloned `Action`. The decoded action is still applied by the
    /// atomic structured executor, never by passing the policy index to
    /// `Game::step`.
    pub fn structured_search_offers(&self) -> Result<StructuredOfferSet, StructuredOfferError> {
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }
        let action_space = self
            .current_action_space
            .as_ref()
            .ok_or(StructuredOfferError::NoActiveActionSpace)?;
        let actor = action_space
            .player
            .ok_or(StructuredOfferError::MissingActor)?;
        let wire_actor = wire_player_id(actor)?;
        if action_space.actions.is_empty() {
            return Err(StructuredOfferError::Invariant(
                "search action space is empty".to_string(),
            ));
        }

        let mut offers = Vec::with_capacity(action_space.actions.len());
        let mut internal = BTreeMap::new();
        for action in &action_space.actions {
            let id = next_offer_id(offers.len())?;
            let label = format!("{:?}", action.action_type());
            let public_commitment = match action {
                Action::PassPriority { .. } => Some(PublicCommitment::PassPriority),
                Action::CastSpell { card, .. } => Some(PublicCommitment::Cast {
                    card: self.state.cards[*card].name.clone(),
                }),
                Action::PlayLand { card, .. } => Some(PublicCommitment::PlayLand {
                    card: self.state.cards[*card].name.clone(),
                }),
                _ => None,
            };
            offers.push(InteractionOffer {
                id,
                actor: wire_actor,
                verb: search_offer_verb(action),
                public_commitment,
                source: None,
                label: label.clone(),
                help: None,
                choices: Vec::new(),
                confirm_label: label,
            });
            internal.insert(
                id,
                InternalOffer::SearchAction {
                    action: action.clone(),
                },
            );
        }

        Ok(StructuredOfferSet {
            projection: StructuredOfferProjection {
                actor: wire_actor,
                kind: search_prompt_kind(action_space.kind)?,
                offers,
            },
            binding: OfferSetBinding(self.decision_epoch),
            internal,
        })
    }

    /// Project the currently covered priority actions without changing state.
    ///
    /// Unsupported legacy actions remain available through `ActionSpace`; this
    /// experimental projection currently covers pass and non-kicked casts with
    /// exactly one required `CreatureOrPlayer` target.
    pub fn structured_priority_offers(&self) -> Result<StructuredOfferSet, StructuredOfferError> {
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }
        let action_space = self
            .current_action_space
            .as_ref()
            .ok_or(StructuredOfferError::NoActiveActionSpace)?;
        if action_space.kind != ActionSpaceKind::Priority {
            return Err(StructuredOfferError::WrongDecision);
        }
        let actor = action_space
            .player
            .ok_or(StructuredOfferError::MissingActor)?;
        let wire_actor = wire_player_id(actor)?;

        let mut offers = Vec::new();
        let mut internal = BTreeMap::new();

        for action in &action_space.actions {
            match action {
                Action::PassPriority { player } => {
                    let id = next_offer_id(offers.len())?;
                    offers.push(InteractionOffer {
                        id,
                        actor: wire_player_id(*player)?,
                        verb: OfferVerb::PassPriority,
                        public_commitment: None,
                        source: None,
                        label: "Pass priority".to_string(),
                        help: None,
                        choices: Vec::new(),
                        confirm_label: "Pass".to_string(),
                    });
                    internal.insert(id, InternalOffer::PassPriority { player: *player });
                }
                Action::CastSpell { player, card } => {
                    let card_ref = &self.state.cards[*card];
                    if card_ref.kicker.is_some() {
                        continue;
                    }
                    let requirements = card_ref.target_requirements();
                    if requirements.len() != 1 {
                        continue;
                    }
                    let requirement = &requirements[0];
                    if requirement.min != 1
                        || requirement.max != 1
                        || requirement.spec != TargetSpec::CreatureOrPlayer
                    {
                        continue;
                    }

                    let legal_targets =
                        self.legal_targets_for_requirement(*player, *card, requirement);
                    if legal_targets.is_empty() {
                        return Err(StructuredOfferError::Invariant(format!(
                            "castable card {} has no legal target",
                            card_ref.name
                        )));
                    }

                    let role = RoleId(1);
                    let mut candidates = Vec::with_capacity(legal_targets.len());
                    let mut target_by_candidate = BTreeMap::new();
                    for target in legal_targets {
                        let candidate_id = next_candidate_id(candidates.len())?;
                        let bound_target = self.bind_target(target)?;
                        let (subject, label) = self.subject_and_label(bound_target)?;
                        candidates.push(Candidate {
                            id: candidate_id,
                            value: CandidateValue::Subject { subject },
                            label,
                            help: None,
                            preview: None,
                        });
                        target_by_candidate.insert(candidate_id, bound_target);
                    }

                    let id = next_offer_id(offers.len())?;
                    offers.push(InteractionOffer {
                        id,
                        actor: wire_player_id(*player)?,
                        verb: OfferVerb::Cast,
                        public_commitment: None,
                        source: Some(SubjectRef::Object {
                            id: object_render_id(
                                card_ref.id,
                                self.current_object_ref(*card)
                                    .ok_or(StructuredOfferError::InvalidCurrentTarget)?
                                    .incarnation
                                    .0,
                            ),
                        }),
                        label: format!("Cast {}", card_ref.name),
                        help: None,
                        choices: vec![ChoiceStep::Select {
                            role,
                            label: "Target".to_string(),
                            candidates: CandidateSource {
                                id: CandidateSourceId(0),
                                depends_on: Vec::new(),
                                initial: Some(candidates),
                            },
                            min: 1,
                            max: 1,
                            ordered: false,
                            distinct: true,
                        }],
                        confirm_label: "Cast".to_string(),
                    });
                    internal.insert(
                        id,
                        InternalOffer::CastSingleTarget {
                            player: *player,
                            card: *card,
                            role,
                            targets: target_by_candidate,
                        },
                    );
                }
                _ => {}
            }
        }

        Ok(StructuredOfferSet {
            projection: StructuredOfferProjection {
                actor: wire_actor,
                kind: PromptKind::Priority,
                offers,
            },
            binding: OfferSetBinding(self.decision_epoch),
            internal,
        })
    }

    /// Project one complete CR 508 attacker declaration as an unordered
    /// multi-select offer. N eligible creatures produce N candidates and
    /// represent all 2^N declarations without enumerating those subsets.
    pub fn structured_attacker_offers(&self) -> Result<StructuredOfferSet, StructuredOfferError> {
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }
        let (actor, declaration_order) = self.current_attacker_declaration()?;
        let wire_actor = wire_player_id(actor)?;
        let role = RoleId(1);

        let mut candidates = Vec::with_capacity(declaration_order.len());
        let mut attacker_by_candidate = BTreeMap::new();
        for attacker in &declaration_order {
            let permanent = self
                .state
                .permanents
                .get(attacker.0)
                .and_then(|permanent| permanent.as_ref())
                .ok_or_else(|| {
                    StructuredOfferError::Invariant(
                        "legacy attacker prompt names a missing permanent".to_string(),
                    )
                })?;
            let candidate_id = next_candidate_id(candidates.len())?;
            let object_ref = self
                .permanent_object_ref(*attacker)
                .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
            candidates.push(Candidate {
                id: candidate_id,
                value: CandidateValue::Subject {
                    subject: SubjectRef::Object {
                        id: object_render_id(permanent.id, object_ref.incarnation.0),
                    },
                },
                label: self.state.cards[permanent.card].name.clone(),
                help: None,
                preview: None,
            });
            attacker_by_candidate.insert(candidate_id, object_ref);
        }

        let max =
            u16::try_from(candidates.len()).map_err(|_| StructuredOfferError::IdentityOverflow)?;
        let id = OfferId(0);
        let offer = InteractionOffer {
            id,
            actor: wire_actor,
            verb: OfferVerb::DeclareAttackers,
            public_commitment: None,
            source: None,
            label: "Declare attackers".to_string(),
            help: None,
            choices: vec![ChoiceStep::Select {
                role,
                label: "Attackers".to_string(),
                candidates: CandidateSource {
                    id: CandidateSourceId(0),
                    depends_on: Vec::new(),
                    initial: Some(candidates),
                },
                min: 0,
                max,
                ordered: false,
                distinct: true,
            }],
            confirm_label: "Declare attackers".to_string(),
        };

        Ok(StructuredOfferSet {
            projection: StructuredOfferProjection {
                actor: wire_actor,
                kind: PromptKind::DeclareAttackers,
                offers: vec![offer],
            },
            binding: OfferSetBinding(self.decision_epoch),
            internal: BTreeMap::from([(
                id,
                InternalOffer::DeclareAttackers {
                    player: actor,
                    role,
                    attackers: attacker_by_candidate,
                    declaration_order: declaration_order
                        .into_iter()
                        .map(|permanent| {
                            self.permanent_object_ref(permanent)
                                .ok_or(StructuredOfferError::InvalidCurrentTarget)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                },
            )]),
        })
    }

    /// Decode IDs against one exact offer set and apply the resulting command
    /// through the existing rules path as one transactional call.
    pub fn apply_offer_submission(
        &mut self,
        offers: &StructuredOfferSet,
        submission: &OfferSubmission,
    ) -> Result<bool, StructuredOfferError> {
        let command = offers.decode(submission)?;
        self.apply_atomic_command(&command)
    }

    /// Decode one structured submission, then execute the same semantic
    /// choice through the preserved positional action ABI.
    ///
    /// This is an experiment-only differential oracle. It deliberately walks
    /// the public legacy prompts instead of sharing the atomic executor.
    pub fn apply_legacy_offer_submission(
        &mut self,
        offers: &StructuredOfferSet,
        submission: &OfferSubmission,
    ) -> Result<(bool, usize), StructuredOfferError> {
        let command = offers.decode(submission)?;
        let checkpoint = self.clone();
        match self.apply_legacy_command_inner(&command) {
            Ok(result) => Ok(result),
            Err(error) => {
                *self = checkpoint;
                Err(error)
            }
        }
    }

    fn apply_legacy_command_inner(
        &mut self,
        command: &AtomicCommand,
    ) -> Result<(bool, usize), StructuredOfferError> {
        let binding = match command {
            AtomicCommand::PassPriority { binding, .. }
            | AtomicCommand::CastSpell { binding, .. }
            | AtomicCommand::DeclareAttackers { binding, .. }
            | AtomicCommand::SearchAction { binding, .. } => *binding,
        };
        if binding != OfferSetBinding(self.decision_epoch) {
            return Err(StructuredOfferError::StaleOrIllegal(
                "offer set no longer names the current decision".to_string(),
            ));
        }
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }

        match command {
            AtomicCommand::PassPriority { player, .. } => {
                let action = self
                    .action_space()
                    .filter(|space| space.kind == ActionSpaceKind::Priority)
                    .and_then(|space| {
                        space.actions.iter().position(|action| {
                            matches!(
                                action,
                                Action::PassPriority { player: legal } if legal == player
                            )
                        })
                    })
                    .ok_or(StructuredOfferError::WrongDecision)?;
                let done = self
                    .step(action)
                    .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                Ok((done, 1))
            }
            AtomicCommand::CastSpell {
                player,
                card,
                targets,
                ..
            } => {
                if targets.len() != 1 {
                    return Err(StructuredOfferError::StaleOrIllegal(format!(
                        "single-target cast requires one target, got {}",
                        targets.len()
                    )));
                }
                let cast = self
                    .action_space()
                    .filter(|space| space.kind == ActionSpaceKind::Priority)
                    .and_then(|space| {
                        space.actions.iter().position(|action| {
                            matches!(
                                action,
                                Action::CastSpell {
                                    player: legal_player,
                                    card: legal_card,
                                } if legal_player == player && legal_card == card
                            )
                        })
                    })
                    .ok_or(StructuredOfferError::WrongDecision)?;
                let mut done = self
                    .step(cast)
                    .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                let mut actions = 1;

                // With skip_trivial enabled, a sole legal target may already
                // have been collapsed by tick(). Otherwise walk the explicit
                // ChooseTarget prompt exactly as the legacy caller does.
                if self
                    .action_space()
                    .is_some_and(|space| space.kind == ActionSpaceKind::ChooseTarget)
                {
                    let wanted = self.resolve_bound_target(targets[0])?;
                    let target = self
                        .action_space()
                        .and_then(|space| {
                            space.actions.iter().position(|action| {
                                matches!(
                                    action,
                                    Action::ChooseTarget {
                                        player: legal_player,
                                        target: legal_target,
                                    } if legal_player == player && legal_target == &wanted
                                )
                            })
                        })
                        .ok_or_else(|| {
                            StructuredOfferError::StaleOrIllegal(
                                "structured target is absent from the legacy prompt".to_string(),
                            )
                        })?;
                    done = self
                        .step(target)
                        .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                    actions += 1;
                }
                Ok((done, actions))
            }
            AtomicCommand::DeclareAttackers {
                player, attackers, ..
            } => {
                let selected: BTreeSet<_> = attackers
                    .iter()
                    .map(|object_ref| {
                        self.lookup_current_permanent(*object_ref).map_err(|_| {
                            StructuredOfferError::StaleOrIllegal(format!(
                                "attacker object {}@{} is no longer current",
                                object_ref.entity.0, object_ref.incarnation.0
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let mut actions = 0;
                let mut done = false;
                while self
                    .action_space()
                    .is_some_and(|space| space.kind == ActionSpaceKind::DeclareAttacker)
                {
                    let space = self.action_space().expect("checked attacker prompt");
                    if space.player != Some(*player) {
                        return Err(StructuredOfferError::WrongDecision);
                    }
                    let permanent = space
                        .actions
                        .iter()
                        .find_map(|action| match action {
                            Action::DeclareAttacker { permanent, .. } => Some(*permanent),
                            _ => None,
                        })
                        .ok_or_else(|| {
                            StructuredOfferError::Invariant(
                                "legacy attacker prompt is empty".to_string(),
                            )
                        })?;
                    let attack = selected.contains(&permanent);
                    let action = space
                        .actions
                        .iter()
                        .position(|action| {
                            matches!(
                                action,
                                Action::DeclareAttacker {
                                    player: legal_player,
                                    permanent: legal_permanent,
                                    attack: legal_attack,
                                } if legal_player == player
                                    && *legal_permanent == permanent
                                    && *legal_attack == attack
                            )
                        })
                        .ok_or_else(|| {
                            StructuredOfferError::Invariant(
                                "legacy attacker prompt lacks a binary choice".to_string(),
                            )
                        })?;
                    done = self
                        .step(action)
                        .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                    actions += 1;
                }
                Ok((done, actions))
            }
            AtomicCommand::SearchAction { .. } => Err(StructuredOfferError::WrongDecision),
        }
    }

    /// Apply one decoded structured command as a single rules mutation.
    ///
    /// Presentation Commands snapshot and restore on failure because the
    /// current engine has no transaction journal. Action-aligned search
    /// Commands validate every fallible precondition before their commit point.
    pub fn apply_atomic_command(
        &mut self,
        command: &AtomicCommand,
    ) -> Result<bool, StructuredOfferError> {
        if matches!(command, AtomicCommand::SearchAction { .. }) {
            return self.apply_search_action(command);
        }
        let checkpoint = self.clone();
        match self.apply_atomic_command_inner(command) {
            Ok(game_over) => Ok(game_over),
            Err(error) => {
                *self = checkpoint;
                Err(error)
            }
        }
    }

    /// Commit one action-aligned search Command after every fallible check.
    ///
    /// `structured_search_offers` stores an exact clone of a currently legal
    /// action. Binding and equality are revalidated before `current_action_space`
    /// is touched, so no rollback snapshot is needed on this hot path. A legal
    /// action becoming unexecutable without a new decision epoch is an engine
    /// invariant failure, not a recoverable Command error.
    fn apply_search_action(
        &mut self,
        command: &AtomicCommand,
    ) -> Result<bool, StructuredOfferError> {
        let AtomicCommand::SearchAction { binding, action } = command else {
            return Err(StructuredOfferError::WrongDecision);
        };
        if *binding != OfferSetBinding(self.decision_epoch) {
            return Err(StructuredOfferError::StaleOrIllegal(
                "offer set no longer names the current decision".to_string(),
            ));
        }
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }
        let offered = self
            .current_action_space
            .as_ref()
            .ok_or(StructuredOfferError::NoActiveActionSpace)?
            .actions
            .iter()
            .any(|legal| legal == action);
        if !offered {
            return Err(StructuredOfferError::StaleOrIllegal(
                "search action is no longer offered".to_string(),
            ));
        }

        let action = action.clone();
        self.current_action_space.take();
        self.execute_action(&action).unwrap_or_else(|error| {
            panic!("structured search offer admitted an unexecutable action: {error}")
        });
        Ok(self.finish_action_step())
    }

    fn apply_atomic_command_inner(
        &mut self,
        command: &AtomicCommand,
    ) -> Result<bool, StructuredOfferError> {
        let binding = match command {
            AtomicCommand::PassPriority { binding, .. }
            | AtomicCommand::CastSpell { binding, .. }
            | AtomicCommand::DeclareAttackers { binding, .. }
            | AtomicCommand::SearchAction { binding, .. } => *binding,
        };
        if binding != OfferSetBinding(self.decision_epoch) {
            return Err(StructuredOfferError::StaleOrIllegal(
                "offer set no longer names the current decision".to_string(),
            ));
        }
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }
        let action_space = self
            .current_action_space
            .as_ref()
            .ok_or(StructuredOfferError::NoActiveActionSpace)?;
        match command {
            AtomicCommand::PassPriority { player, .. } => {
                if action_space.kind != ActionSpaceKind::Priority {
                    return Err(StructuredOfferError::WrongDecision);
                }
                let action = action_space
                    .actions
                    .iter()
                    .find(|action| {
                        matches!(action, Action::PassPriority { player: legal } if legal == player)
                    })
                    .cloned()
                    .ok_or_else(|| {
                        StructuredOfferError::StaleOrIllegal(
                            "priority pass is no longer offered".to_string(),
                        )
                    })?;
                self.current_action_space.take();
                self.execute_action(&action)
                    .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                Ok(self.finish_action_step())
            }
            AtomicCommand::CastSpell {
                player,
                card,
                targets,
                ..
            } => {
                if action_space.kind != ActionSpaceKind::Priority {
                    return Err(StructuredOfferError::WrongDecision);
                }
                if targets.len() != 1 {
                    return Err(StructuredOfferError::StaleOrIllegal(format!(
                        "single-target cast requires one target, got {}",
                        targets.len()
                    )));
                }
                let action = action_space
                    .actions
                    .iter()
                    .find(|action| {
                        matches!(
                            action,
                            Action::CastSpell {
                                player: legal_player,
                                card: legal_card,
                            } if legal_player == player && legal_card == card
                        )
                    })
                    .cloned()
                    .ok_or_else(|| {
                        StructuredOfferError::StaleOrIllegal(
                            "spell is no longer offered".to_string(),
                        )
                    })?;

                self.current_action_space.take();
                self.execute_action(&action)
                    .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                let target_action = Action::ChooseTarget {
                    player: *player,
                    target: self.resolve_bound_target(targets[0])?,
                };
                self.execute_action(&target_action)
                    .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                Ok(self.finish_action_step())
            }
            AtomicCommand::DeclareAttackers {
                player, attackers, ..
            } => {
                let (legal_player, declaration_order) = self.current_attacker_declaration()?;
                if player != &legal_player {
                    return Err(StructuredOfferError::StaleOrIllegal(
                        "attacker declaration actor is no longer current".to_string(),
                    ));
                }

                let legal: BTreeSet<_> = declaration_order.iter().copied().collect();
                let selected: BTreeSet<_> = attackers
                    .iter()
                    .map(|object_ref| {
                        self.lookup_current_permanent(*object_ref).map_err(|_| {
                            StructuredOfferError::StaleOrIllegal(format!(
                                "attacker object {}@{} is no longer current",
                                object_ref.entity.0, object_ref.incarnation.0
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                if selected.len() != attackers.len() {
                    return Err(StructuredOfferError::StaleOrIllegal(
                        "attacker declaration contains a duplicate permanent".to_string(),
                    ));
                }
                if let Some(attacker) = selected.iter().find(|attacker| !legal.contains(attacker)) {
                    return Err(StructuredOfferError::StaleOrIllegal(format!(
                        "permanent {} is not offered as an attacker",
                        attacker.0
                    )));
                }

                self.current_action_space.take();
                let combat = self.state.combat.as_mut().ok_or_else(|| {
                    StructuredOfferError::StaleOrIllegal(
                        "attacker declaration has no combat state".to_string(),
                    )
                })?;
                combat.attackers_to_declare.clear();
                for attacker in declaration_order {
                    self.declare_attacker(attacker, selected.contains(&attacker))
                        .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                }
                Ok(self.finish_action_step())
            }
            AtomicCommand::SearchAction { action, .. } => {
                let offered = action_space.actions.iter().any(|legal| legal == action);
                if !offered {
                    return Err(StructuredOfferError::StaleOrIllegal(
                        "search action is no longer offered".to_string(),
                    ));
                }
                let action = action.clone();
                self.current_action_space.take();
                self.execute_action(&action)
                    .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))?;
                Ok(self.finish_action_step())
            }
        }
    }

    /// Recover the complete, still-undecided attacker batch from the current
    /// sequential legacy prompt. The first permanent was popped while the
    /// prompt was built; the remainder will be popped from the back.
    fn current_attacker_declaration(
        &self,
    ) -> Result<(PlayerId, Vec<PermanentId>), StructuredOfferError> {
        let action_space = self
            .current_action_space
            .as_ref()
            .ok_or(StructuredOfferError::NoActiveActionSpace)?;
        if action_space.kind != ActionSpaceKind::DeclareAttacker {
            return Err(StructuredOfferError::WrongDecision);
        }
        let actor = action_space
            .player
            .ok_or(StructuredOfferError::MissingActor)?;

        let mut current = None;
        let mut attacks = BTreeSet::new();
        for action in &action_space.actions {
            let Action::DeclareAttacker {
                player,
                permanent,
                attack,
            } = action
            else {
                return Err(StructuredOfferError::Invariant(
                    "declare-attacker prompt contains another action kind".to_string(),
                ));
            };
            if *player != actor {
                return Err(StructuredOfferError::Invariant(
                    "declare-attacker prompt contains another actor".to_string(),
                ));
            }
            if current
                .replace(*permanent)
                .is_some_and(|seen| seen != *permanent)
            {
                return Err(StructuredOfferError::Invariant(
                    "declare-attacker prompt names multiple permanents".to_string(),
                ));
            }
            attacks.insert(*attack);
        }
        let current = current.ok_or_else(|| {
            StructuredOfferError::Invariant("declare-attacker prompt is empty".to_string())
        })?;
        if attacks != BTreeSet::from([false, true]) {
            return Err(StructuredOfferError::Invariant(
                "declare-attacker prompt must offer attack and decline".to_string(),
            ));
        }

        let combat = self.state.combat.as_ref().ok_or_else(|| {
            StructuredOfferError::Invariant(
                "declare-attacker prompt has no combat state".to_string(),
            )
        })?;
        let mut declaration_order = Vec::with_capacity(combat.attackers_to_declare.len() + 1);
        declaration_order.push(current);
        declaration_order.extend(combat.attackers_to_declare.iter().rev().copied());
        if declaration_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != declaration_order.len()
        {
            return Err(StructuredOfferError::Invariant(
                "declare-attacker prompt repeats a permanent".to_string(),
            ));
        }
        Ok((actor, declaration_order))
    }

    fn bind_target(&self, target: Target) -> Result<BoundTarget, StructuredOfferError> {
        match target {
            Target::Player(player) => Ok(BoundTarget::Player(player)),
            Target::Permanent(permanent) => self
                .permanent_object_ref(permanent)
                .map(BoundTarget::Permanent)
                .ok_or(StructuredOfferError::InvalidCurrentTarget),
            Target::StackSpell(card) => Ok(BoundTarget::StackSpell(card)),
        }
    }

    fn resolve_bound_target(
        &self,
        target: BoundTarget,
    ) -> Result<ActionTarget, StructuredOfferError> {
        match target {
            BoundTarget::Player(player) => Ok(ActionTarget::Player(player)),
            BoundTarget::Permanent(object_ref) => self
                .lookup_current_permanent(object_ref)
                .map(ActionTarget::Permanent)
                .map_err(|_| {
                    StructuredOfferError::StaleOrIllegal(format!(
                        "target object {}@{} is no longer current",
                        object_ref.entity.0, object_ref.incarnation.0
                    ))
                }),
            BoundTarget::StackSpell(card) => Ok(ActionTarget::StackSpell(card)),
        }
    }

    fn subject_and_label(
        &self,
        target: BoundTarget,
    ) -> Result<(SubjectRef, String), StructuredOfferError> {
        match target {
            BoundTarget::Player(player) => Ok((
                SubjectRef::Player {
                    id: wire_player_id(player)?,
                },
                self.state.players[player.0].name.clone(),
            )),
            BoundTarget::Permanent(object_ref) => {
                let permanent = self
                    .lookup_current_permanent(object_ref)
                    .map_err(|_| StructuredOfferError::InvalidCurrentTarget)?;
                let permanent_ref = self
                    .state
                    .permanents
                    .get(permanent.0)
                    .and_then(|permanent| permanent.as_ref())
                    .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
                Ok((
                    SubjectRef::Object {
                        id: object_render_id(permanent_ref.id, object_ref.incarnation.0),
                    },
                    self.state.cards[permanent_ref.card].name.clone(),
                ))
            }
            BoundTarget::StackSpell(_) => Err(StructuredOfferError::InvalidCurrentTarget),
        }
    }
}

fn candidate_answer(
    answers: &[ChoiceAnswer],
    role: RoleId,
) -> Result<&[CandidateId], StructuredOfferError> {
    if answers.is_empty() {
        return Err(StructuredOfferError::MissingAnswer(role));
    }

    let mut seen_roles = BTreeSet::new();
    for answer in answers {
        if !seen_roles.insert(answer.role()) {
            return Err(StructuredOfferError::DuplicateRole(answer.role()));
        }
    }
    if answers.len() != 1 {
        return Err(StructuredOfferError::UnexpectedAnswers);
    }

    let ChoiceAnswer::Candidates {
        role: answer_role,
        candidates,
    } = &answers[0];
    if *answer_role != role {
        return Err(StructuredOfferError::UnexpectedRole(*answer_role));
    }
    Ok(candidates)
}

fn object_render_id(id: ObjectId, incarnation: u32) -> ObjectRenderId {
    ObjectRenderId {
        entity: id.0,
        incarnation,
    }
}

fn wire_player_id(player: PlayerId) -> Result<u8, StructuredOfferError> {
    u8::try_from(player.0).map_err(|_| StructuredOfferError::IdentityOverflow)
}

fn next_offer_id(len: usize) -> Result<OfferId, StructuredOfferError> {
    u32::try_from(len)
        .map(OfferId)
        .map_err(|_| StructuredOfferError::IdentityOverflow)
}

fn next_candidate_id(len: usize) -> Result<CandidateId, StructuredOfferError> {
    u32::try_from(len)
        .map(CandidateId)
        .map_err(|_| StructuredOfferError::IdentityOverflow)
}

fn search_prompt_kind(kind: ActionSpaceKind) -> Result<PromptKind, StructuredOfferError> {
    match kind {
        ActionSpaceKind::Priority => Ok(PromptKind::Priority),
        ActionSpaceKind::DeclareAttacker => Ok(PromptKind::DeclareAttackers),
        ActionSpaceKind::DeclareBlocker => Ok(PromptKind::DeclareBlockers),
        ActionSpaceKind::ChooseTarget => Ok(PromptKind::ChooseTarget),
        ActionSpaceKind::Scry => Ok(PromptKind::Scry),
        ActionSpaceKind::LookAndSelect => Ok(PromptKind::LookAndSelect),
        ActionSpaceKind::PayOrNot => Ok(PromptKind::PayOrNot),
        ActionSpaceKind::Modal => Ok(PromptKind::Modal),
        ActionSpaceKind::DiscardThenDraw => Ok(PromptKind::DiscardThenDraw),
        ActionSpaceKind::Waterbend => Ok(PromptKind::Waterbend),
        ActionSpaceKind::GameOver => Err(StructuredOfferError::GameOver),
    }
}

fn search_offer_verb(action: &Action) -> OfferVerb {
    match action {
        Action::PlayLand { .. } => OfferVerb::PlayLand,
        Action::CastSpell { .. } => OfferVerb::Cast,
        Action::ActivateAbility { .. } => OfferVerb::ActivateAbility,
        Action::PassPriority { .. } => OfferVerb::PassPriority,
        Action::DeclareAttacker { .. } => OfferVerb::DeclareAttacker,
        Action::DeclareBlocker { .. } => OfferVerb::DeclareBlocker,
        Action::ChooseTarget { .. } => OfferVerb::ChooseTarget,
        Action::ScryCard {
            to_bottom: false, ..
        } => OfferVerb::ScryKeep,
        Action::ScryCard {
            to_bottom: true, ..
        } => OfferVerb::ScryBottom,
        Action::SelectCard { .. } => OfferVerb::SelectCard,
        Action::Decline { .. } => OfferVerb::Decline,
        Action::PayCost { .. } => OfferVerb::PayCost,
        Action::ChooseMode { .. } => OfferVerb::ChooseMode,
        Action::WaterbendTap { .. } => OfferVerb::WaterbendTap,
    }
}
