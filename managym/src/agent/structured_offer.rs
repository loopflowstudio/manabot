//! Experimental structured legal offers over the current positional action ABI.
//!
//! The current `ActionSpace` and target legality queries remain authoritative.
//! This module projects a narrow, uncapped typed view for priority pass,
//! single `AnyTarget` target casts, complete combat declarations, and
//! commuting waterbend payments,
//! then lowers accepted IDs through the existing rules executor. It
//! intentionally does not own match revisions, prompt persistence, recovery,
//! or policy decoding. `compound_offers` covers the full legal surface and
//! `compound_commands` lowers its atomic selections to canonical Commands.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    agent::action::{Action, ActionSpaceKind},
    flow::{
        declaration::BlockerAssignment,
        game::{Game, PendingChoice},
    },
    state::{
        ability::TargetSpec,
        game_object::{ObjectId, PermanentId, PlayerId, PublicId, Target},
        zone::ZoneType,
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
    Learn,
    Waterbend,
    Discard,
    LegendRule,
    OrderTriggers,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OfferVerb {
    ActivateAbility,
    Cast,
    ChooseMode,
    ChooseTrigger,
    ChooseTarget,
    DeclareAttacker,
    DeclareAttackers,
    DeclareBlocker,
    DeclareBlockers,
    Decline,
    PassPriority,
    PayCost,
    PayWaterbend,
    PlayLand,
    ScryBottom,
    ScryKeep,
    SelectCard,
    LearnTakeLesson,
    LearnDiscard,
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
    Discard { card: String },
    LearnTakeLesson { card: String },
    DeclineLearn,
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
        object: ObjectId,
        targets: Vec<BoundTarget>,
    },
    DeclareAttackers {
        binding: OfferSetBinding,
        player: PlayerId,
        attackers: Vec<ObjectId>,
    },
    DeclareBlockers {
        binding: OfferSetBinding,
        player: PlayerId,
        assignments: Vec<BlockerAssignment>,
    },
    DeclarationSequence {
        binding: OfferSetBinding,
        kind: ActionSpaceKind,
        actions: Vec<Action>,
    },
    SearchAction {
        binding: OfferSetBinding,
        action: Action,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundTarget {
    Player(PlayerId),
    Permanent(ObjectId),
    StackSpell(ObjectId),
    StackAbility(PublicId),
    GraveyardCard(ObjectId),
}

#[derive(Clone, Debug)]
enum InternalOffer {
    PassPriority {
        player: PlayerId,
    },
    CastSingleTarget {
        player: PlayerId,
        object: ObjectId,
        role: RoleId,
        targets: BTreeMap<CandidateId, BoundTarget>,
    },
    DeclareAttackers {
        player: PlayerId,
        role: RoleId,
        attackers: BTreeMap<CandidateId, ObjectId>,
        declaration_order: Vec<ObjectId>,
    },
    DeclareBlockers {
        player: PlayerId,
        choices: Vec<BlockerChoice>,
    },
    Waterbend {
        player: PlayerId,
        role: RoleId,
        taps: BTreeMap<CandidateId, Action>,
        min: usize,
        max: usize,
        remaining_generic: usize,
    },
    SearchAction {
        action: Action,
    },
}

#[derive(Clone, Debug)]
struct BlockerChoice {
    role: RoleId,
    blocker: ObjectId,
    targets: BTreeMap<CandidateId, ObjectId>,
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
    ) -> Vec<(OfferId, RoleId, CandidateId, ObjectId)> {
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
                // Compound-only bindings lower through exact canonical actions;
                // they are not projected into canonical DecisionFrame offers.
                InternalOffer::DeclareBlockers { .. }
                | InternalOffer::Waterbend { .. }
                | InternalOffer::PassPriority { .. }
                | InternalOffer::SearchAction { .. } => {}
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
                object,
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
                    object: *object,
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
            InternalOffer::DeclareBlockers { player, choices } => {
                let mut answers = BTreeMap::new();
                for ChoiceAnswer::Candidates { role, candidates } in &submission.answers {
                    if answers.insert(*role, candidates).is_some() {
                        return Err(StructuredOfferError::DuplicateRole(*role));
                    }
                }
                let mut assignments = Vec::with_capacity(choices.len());
                for choice in choices {
                    let selected = answers
                        .remove(&choice.role)
                        .ok_or(StructuredOfferError::MissingAnswer(choice.role))?;
                    let attacker = match selected.as_slice() {
                        [] => None,
                        [candidate] => Some(
                            choice
                                .targets
                                .get(candidate)
                                .cloned()
                                .ok_or(StructuredOfferError::UnknownCandidate(*candidate))?,
                        ),
                        _ => {
                            return Err(StructuredOfferError::CardinalityOutOfRange {
                                role: choice.role,
                                min: 0,
                                max: 1,
                                actual: selected.len(),
                            })
                        }
                    };
                    assignments.push(BlockerAssignment {
                        blocker: choice.blocker,
                        attacker,
                    });
                }
                if let Some(role) = answers.keys().next() {
                    return Err(StructuredOfferError::UnexpectedRole(*role));
                }
                Ok(AtomicCommand::DeclareBlockers {
                    binding: self.binding,
                    player: *player,
                    assignments,
                })
            }
            InternalOffer::Waterbend {
                player,
                role,
                taps,
                min,
                max,
                remaining_generic,
            } => {
                let selected = candidate_answer(&submission.answers, *role)?;
                if selected.len() < *min || selected.len() > *max {
                    return Err(StructuredOfferError::CardinalityOutOfRange {
                        role: *role,
                        min: *min,
                        max: *max,
                        actual: selected.len(),
                    });
                }
                let mut ids = BTreeSet::new();
                for candidate in selected {
                    if !ids.insert(*candidate) {
                        return Err(StructuredOfferError::DuplicateCandidate(*candidate));
                    }
                    if !taps.contains_key(candidate) {
                        return Err(StructuredOfferError::UnknownCandidate(*candidate));
                    }
                }
                // Canonical candidate order removes permutation aliases. Payment
                // with mana is the final action unless taps exhaust generic cost.
                let mut actions: Vec<_> = ids.iter().map(|id| taps[id].clone()).collect();
                if selected.len() < *remaining_generic {
                    actions.push(Action::PayCost { player: *player });
                }
                Ok(AtomicCommand::DeclarationSequence {
                    binding: self.binding,
                    kind: ActionSpaceKind::Waterbend,
                    actions,
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
    Execution(crate::agent::action::AgentError),
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

impl From<crate::decision::SemanticError> for StructuredOfferError {
    fn from(error: crate::decision::SemanticError) -> Self {
        match error {
            crate::decision::SemanticError::Execution(error) => Self::Execution(error),
            other => Self::Invariant(other.to_string()),
        }
    }
}

impl From<StructuredOfferError> for crate::agent::action::AgentError {
    fn from(error: StructuredOfferError) -> Self {
        match error {
            StructuredOfferError::Execution(error) => error,
            other => Self::Rules(other.to_string()),
        }
    }
}

impl std::fmt::Display for StructuredOfferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution(error) => error.fmt(f),
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
        self.ensure_playable()
            .map_err(StructuredOfferError::Execution)?;
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
            let label = self
                .action_description(action)
                .unwrap_or_else(|| format!("{:?}", action.action_type()));
            let public_commitment = match (action_space.kind, action) {
                (_, Action::PassPriority { .. }) => Some(PublicCommitment::PassPriority),
                (_, Action::CastSpell { object, .. }) => Some(PublicCommitment::Cast {
                    card: self.state[*object].name.clone(),
                }),
                (_, Action::PlayLand { object, .. }) => Some(PublicCommitment::PlayLand {
                    card: self.state[*object].name.clone(),
                }),
                (ActionSpaceKind::Learn, Action::LearnDiscard { object, .. })
                | (ActionSpaceKind::Discard, Action::SelectCard { object, .. }) => {
                    Some(PublicCommitment::Discard {
                        card: self.state[*object].name.clone(),
                    })
                }
                (ActionSpaceKind::Learn, Action::LearnTakeLesson { object, .. }) => {
                    Some(PublicCommitment::LearnTakeLesson {
                        card: self.state.source_base(*object).1.name.clone(),
                    })
                }
                (ActionSpaceKind::Learn, Action::Decline { .. }) => {
                    Some(PublicCommitment::DeclineLearn)
                }
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

    /// Complete legal policy surface. Only engine-owned atomic choices group;
    /// every other decision retains its ordinary action-aligned offer.
    pub fn compound_offers(&self) -> Result<StructuredOfferSet, StructuredOfferError> {
        let mut complete = self.structured_search_offers()?;
        match self.current_action_space.as_ref().map(|space| space.kind) {
            Some(ActionSpaceKind::DeclareAttacker) => self.structured_attacker_offers(),
            Some(ActionSpaceKind::DeclareBlocker) => {
                Ok(self.structured_blocker_offers()?.unwrap_or(complete))
            }
            Some(ActionSpaceKind::Waterbend) => {
                Ok(self.structured_waterbend_offers()?.unwrap_or(complete))
            }
            Some(ActionSpaceKind::Priority) => {
                let presentation = self.structured_priority_offers()?;
                for offer in &mut complete.projection.offers {
                    let Some(InternalOffer::SearchAction {
                        action: Action::CastSpell { object, .. },
                    }) = complete.internal.get(&offer.id)
                    else {
                        continue;
                    };
                    let replacement =
                        presentation
                            .internal
                            .iter()
                            .find_map(|(id, inner)| match inner {
                                InternalOffer::CastSingleTarget {
                                    object: target_card,
                                    ..
                                } if target_card == object => Some((*id, inner.clone())),
                                _ => None,
                            });
                    if let Some((id, inner)) = replacement {
                        let original_id = offer.id;
                        *offer = presentation
                            .projection
                            .offers
                            .iter()
                            .find(|candidate| candidate.id == id)
                            .expect("bound offer")
                            .clone();
                        offer.id = original_id;
                        complete.internal.insert(original_id, inner);
                    }
                }
                Ok(complete)
            }
            _ => Ok(complete),
        }
    }

    /// Pending blocker factors expose every pair; full-plan validation owns
    /// coupled restrictions and required blocks before any factor executes.
    fn structured_blocker_offers(
        &self,
    ) -> Result<Option<StructuredOfferSet>, StructuredOfferError> {
        let space = self
            .action_space()
            .ok_or(StructuredOfferError::NoActiveActionSpace)?;
        let player = space
            .actions
            .first()
            .map(|action| action.player())
            .ok_or(StructuredOfferError::MissingActor)?;
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or(StructuredOfferError::WrongDecision)?;
        let first = space
            .actions
            .iter()
            .find_map(|action| match action {
                Action::DeclareBlocker { blocker, .. } => Some(*blocker),
                _ => None,
            })
            .ok_or(StructuredOfferError::WrongDecision)?;
        let mut choices = Vec::new();
        let mut bound = Vec::new();
        for blocker in
            std::iter::once(first).chain(combat.blockers_to_declare.iter().rev().copied())
        {
            let targets: Vec<_> = combat
                .attackers
                .iter()
                .copied()
                .filter(|attacker| self.blocker_pair_eligible(blocker, *attacker))
                .collect();
            let role = RoleId(
                u16::try_from(choices.len()).map_err(|_| StructuredOfferError::IdentityOverflow)?,
            );
            let mut candidates = Vec::new();
            let mut targets_by_id = BTreeMap::new();
            for attacker in targets {
                let id = next_candidate_id(candidates.len())?;
                candidates.push(self.compound_permanent_candidate(attacker, id)?);
                targets_by_id.insert(
                    id,
                    self.permanent_object(attacker)
                        .ok_or(StructuredOfferError::InvalidCurrentTarget)?,
                );
            }
            let name = self
                .compound_permanent_candidate(blocker, CandidateId(0))?
                .label;
            let max = u16::from(!candidates.is_empty());
            choices.push(ChoiceStep::Select {
                role,
                label: format!("Block with {name}"),
                candidates: CandidateSource {
                    id: CandidateSourceId(u32::from(role.0)),
                    depends_on: Vec::new(),
                    initial: Some(candidates),
                },
                min: 0,
                max,
                ordered: false,
                distinct: true,
            });
            bound.push(BlockerChoice {
                role,
                blocker: self
                    .permanent_object(blocker)
                    .ok_or(StructuredOfferError::InvalidCurrentTarget)?,
                targets: targets_by_id,
            });
        }
        let offer = InteractionOffer {
            id: OfferId(0),
            actor: wire_player_id(space.player.ok_or(StructuredOfferError::MissingActor)?)?,
            verb: OfferVerb::DeclareBlockers,
            public_commitment: None,
            source: None,
            label: "Declare blockers".into(),
            help: None,
            choices,
            confirm_label: "Declare blockers".into(),
        };
        Ok(Some(StructuredOfferSet {
            projection: StructuredOfferProjection {
                actor: offer.actor,
                kind: PromptKind::DeclareBlockers,
                offers: vec![offer],
            },
            binding: OfferSetBinding(self.decision_epoch),
            internal: BTreeMap::from([(
                OfferId(0),
                InternalOffer::DeclareBlockers {
                    player,
                    choices: bound,
                },
            )]),
        }))
    }

    /// Group a commuting subset of waterbend taps and the remaining mana
    /// payment. Taps with triggered effects or competing mana production stay
    /// sequential: their support and information boundary can depend on order.
    fn structured_waterbend_offers(
        &self,
    ) -> Result<Option<StructuredOfferSet>, StructuredOfferError> {
        let Some(PendingChoice::Waterbend {
            player,
            payment,
            remaining_generic,
            tapped,
        }) = &self.pending_choice
        else {
            return Err(StructuredOfferError::WrongDecision);
        };
        if !self.state.delayed_triggers.is_empty()
            || self.state.permanents.iter().flatten().any(|permanent| {
                let object = &self.state[permanent.object];
                !object.abilities.is_empty() || !object.triggered_mana_abilities.is_empty()
            })
        {
            return Ok(None);
        }
        let player = *player;
        let legal_taps = self.waterbend_payment_candidates(player, payment);
        if legal_taps.iter().any(|id| {
            self.state.permanents[*id]
                .as_ref()
                .is_some_and(|permanent| !self.state[permanent.object].mana_abilities.is_empty())
        }) {
            return Ok(None);
        }
        let (cost, remaining_generic) =
            self.waterbend_owed(player, payment, *remaining_generic, *tapped);
        let max = legal_taps.len().min(usize::from(remaining_generic));
        let min = (0..=max)
            .find(|count| self.can_pay_mana_cost(player, &cost.reduced_generic(*count as u8)))
            .ok_or_else(|| {
                StructuredOfferError::Invariant("waterbend has no payable completion".into())
            })?;
        let role = RoleId(0);
        let mut candidates = Vec::new();
        let mut taps = BTreeMap::new();
        for permanent in legal_taps {
            let id = next_candidate_id(candidates.len())?;
            candidates.push(self.compound_permanent_candidate(permanent, id)?);
            taps.insert(id, Action::WaterbendTap { player, permanent });
        }
        let offer = InteractionOffer {
            id: OfferId(0),
            actor: wire_player_id(
                self.action_space()
                    .and_then(|space| space.player)
                    .ok_or(StructuredOfferError::MissingActor)?,
            )?,
            verb: OfferVerb::PayWaterbend,
            public_commitment: None,
            source: None,
            label: "Pay waterbend".into(),
            help: None,
            choices: vec![ChoiceStep::Select {
                role,
                label: "Tap for waterbend".into(),
                candidates: CandidateSource {
                    id: CandidateSourceId(0),
                    depends_on: Vec::new(),
                    initial: Some(candidates),
                },
                min: min as u16,
                max: max as u16,
                ordered: false,
                distinct: true,
            }],
            confirm_label: "Pay waterbend".into(),
        };
        Ok(Some(StructuredOfferSet {
            projection: StructuredOfferProjection {
                actor: offer.actor,
                kind: PromptKind::Waterbend,
                offers: vec![offer],
            },
            binding: OfferSetBinding(self.decision_epoch),
            internal: BTreeMap::from([(
                OfferId(0),
                InternalOffer::Waterbend {
                    player,
                    role,
                    taps,
                    min,
                    max,
                    remaining_generic: usize::from(remaining_generic),
                },
            )]),
        }))
    }

    fn compound_permanent_candidate(
        &self,
        permanent: PermanentId,
        id: CandidateId,
    ) -> Result<Candidate, StructuredOfferError> {
        let object = self
            .permanent_object(permanent)
            .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
        let permanent = self.state.permanents[permanent]
            .as_ref()
            .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
        Ok(Candidate {
            id,
            value: CandidateValue::Subject {
                subject: SubjectRef::Object {
                    id: object_render_id(permanent.id, object.generation()),
                },
            },
            label: self.state[permanent.object].name.clone(),
            help: None,
            preview: None,
        })
    }

    /// Lower a complete submission on an exact fork, never mutate the root.
    /// Combat children stop once their bound declaration completes; the last
    /// child's ordinary continuation may already have resolved its consequences.
    /// No planner or policy observes intermediate states during this operation.
    pub fn compound_commands(
        &self,
        offers: &StructuredOfferSet,
        submission: &OfferSubmission,
    ) -> Result<Vec<crate::decision::Command>, StructuredOfferError> {
        let command = offers.decode(submission)?;
        if matches!(
            command,
            AtomicCommand::DeclareAttackers { .. } | AtomicCommand::DeclareBlockers { .. }
        ) {
            return Ok(self
                .prepare_declaration(&command)?
                .children
                .into_iter()
                .map(|child| child.command)
                .collect());
        }
        let mut fork = self.clone();
        let mut commands = Some(Vec::new());
        fork.apply_legacy_command_inner(&command, &mut commands)?;
        Ok(commands.expect("recording enabled"))
    }

    fn record_compound_step(
        &mut self,
        action: usize,
        commands: &mut Option<Vec<crate::decision::Command>>,
    ) -> Result<bool, StructuredOfferError> {
        if let Some(commands) = commands {
            let frame = self
                .semantic_decision_frame()
                .map_err(StructuredOfferError::from)?;
            commands.push(crate::decision::Command {
                command_id: format!("compound-{}-{}", frame.revision, commands.len()),
                expected_revision: frame.revision,
                offer_id: frame.offers[action].id.0,
                answers: Vec::new(),
                object_preconditions: Vec::new(),
            });
        }
        self.step(action).map_err(StructuredOfferError::Execution)
    }

    /// Project the currently covered priority actions without changing state.
    ///
    /// Unsupported legacy actions remain available through `ActionSpace`; this
    /// experimental projection currently covers pass and non-kicked casts with
    /// exactly one required `AnyTarget` target.
    pub fn structured_priority_offers(&self) -> Result<StructuredOfferSet, StructuredOfferError> {
        self.ensure_playable()
            .map_err(StructuredOfferError::Execution)?;
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
                        actor: wire_actor,
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
                Action::CastSpell { player, object } => {
                    let card_ref = &self.state[*object];
                    if card_ref.optional_cost().is_some() {
                        continue;
                    }
                    let requirements = card_ref.target_requirements();
                    if requirements.len() != 1 {
                        continue;
                    }
                    let requirement = &requirements[0];
                    if requirement.min != 1
                        || requirement.max != 1
                        || requirement.spec != TargetSpec::AnyTarget
                    {
                        continue;
                    }

                    let legal_targets =
                        self.legal_targets_for_requirement(*player, *object, requirement);
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
                        actor: wire_actor,
                        verb: OfferVerb::Cast,
                        public_commitment: None,
                        source: Some(SubjectRef::Object {
                            id: object_render_id(
                                card_ref.id,
                                self.state
                                    .live(*object)
                                    .ok_or(StructuredOfferError::InvalidCurrentTarget)?
                                    .generation(),
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
                            object: *object,
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
        self.ensure_playable()
            .map_err(StructuredOfferError::Execution)?;
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }
        let (player, declaration_order) = self.current_attacker_declaration()?;
        let wire_actor = wire_player_id(
            self.action_space()
                .and_then(|space| space.player)
                .ok_or(StructuredOfferError::MissingActor)?,
        )?;
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
                .permanent_object(*attacker)
                .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
            candidates.push(Candidate {
                id: candidate_id,
                value: CandidateValue::Subject {
                    subject: SubjectRef::Object {
                        id: object_render_id(permanent.id, object_ref.generation()),
                    },
                },
                label: self.state[permanent.object].name.clone(),
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
                    player,
                    role,
                    attackers: attacker_by_candidate,
                    declaration_order: declaration_order
                        .into_iter()
                        .map(|permanent| {
                            self.permanent_object(permanent)
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
        #[cfg(feature = "benchmark")]
        let decode_timer = crate::benchmark_search::Timer::new("command_validation");
        let command = offers.decode(submission)?;
        #[cfg(feature = "benchmark")]
        drop(decode_timer);
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
        match self.apply_legacy_command_inner(&command, &mut None) {
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
        commands: &mut Option<Vec<crate::decision::Command>>,
    ) -> Result<(bool, usize), StructuredOfferError> {
        let binding = match command {
            AtomicCommand::PassPriority { binding, .. }
            | AtomicCommand::CastSpell { binding, .. }
            | AtomicCommand::DeclareAttackers { binding, .. }
            | AtomicCommand::DeclareBlockers { binding, .. }
            | AtomicCommand::DeclarationSequence { binding, .. }
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
                let done = self.record_compound_step(action, commands)?;
                Ok((done, 1))
            }
            AtomicCommand::CastSpell {
                player,
                object,
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
                                    object: legal_card,
                                } if legal_player == player && legal_card == object
                            )
                        })
                    })
                    .ok_or(StructuredOfferError::WrongDecision)?;
                let mut done = self.record_compound_step(cast, commands)?;
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
                    done = self.record_compound_step(target, commands)?;
                    actions += 1;
                }
                Ok((done, actions))
            }
            AtomicCommand::DeclareAttackers {
                player, attackers, ..
            } => {
                let (actor, declaration_order) = self.current_attacker_declaration()?;
                if actor != *player {
                    return Err(StructuredOfferError::WrongDecision);
                }
                let selected: BTreeSet<_> = attackers
                    .iter()
                    .map(|object_ref| {
                        self.lookup_current_permanent(*object_ref).map_err(|_| {
                            StructuredOfferError::StaleOrIllegal(format!(
                                "attacker object {}@{} is no longer current",
                                object_ref.slot(),
                                object_ref.generation()
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                if selected.len() != attackers.len() {
                    return Err(StructuredOfferError::StaleOrIllegal(
                        "attacker declaration contains a duplicate permanent".to_string(),
                    ));
                }
                let legal: BTreeSet<_> = declaration_order.iter().copied().collect();
                if !selected.is_subset(&legal) {
                    return Err(StructuredOfferError::StaleOrIllegal(
                        "attacker declaration contains a permanent outside the pending batch"
                            .to_string(),
                    ));
                }
                let mut actions = 0;
                let mut done = false;
                // The final choice may skip trivial windows all the way to a
                // later turn's declaration, even by this same actor. Lower only
                // the batch bound at the root, never a fresh prompt of this kind.
                for permanent in declaration_order {
                    let space = self
                        .action_space()
                        .filter(|space| {
                            space.kind == ActionSpaceKind::DeclareAttacker
                                && space.actions.first().map(|action| action.player())
                                    == Some(*player)
                        })
                        .ok_or(StructuredOfferError::WrongDecision)?;
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
                    done = self.record_compound_step(action, commands)?;
                    actions += 1;
                }
                Ok((done, actions))
            }
            AtomicCommand::DeclareBlockers {
                player,
                assignments,
                ..
            } => {
                let mut done = false;
                let mut executed = 0;
                for assignment in assignments {
                    let blocker = self
                        .lookup_current_permanent(assignment.blocker)
                        .map_err(|_| StructuredOfferError::InvalidCurrentTarget)?;
                    let attacker = assignment
                        .attacker
                        .map(|object| {
                            self.lookup_current_permanent(object)
                                .map_err(|_| StructuredOfferError::InvalidCurrentTarget)
                        })
                        .transpose()?;
                    let wanted = Action::DeclareBlocker {
                        player: *player,
                        blocker,
                        attacker,
                    };
                    let index = self
                        .action_space()
                        .filter(|space| space.kind == ActionSpaceKind::DeclareBlocker)
                        .and_then(|space| space.actions.iter().position(|action| action == &wanted))
                        .ok_or(StructuredOfferError::WrongDecision)?;
                    done = self.record_compound_step(index, commands)?;
                    executed += 1;
                }
                Ok((done, executed))
            }
            AtomicCommand::DeclarationSequence { kind, actions, .. } => {
                let mut done = false;
                let mut executed = 0;
                for action in actions {
                    // tick() may have consumed forced payment suffix factors.
                    // Only these known, effect-free taps/remainder can be omitted.
                    if *kind == ActionSpaceKind::Waterbend && self.skip_trivial {
                        match action {
                            Action::WaterbendTap { permanent, .. }
                                if self.state.permanents[*permanent]
                                    .as_ref()
                                    .is_some_and(|permanent| permanent.tapped) =>
                            {
                                continue
                            }
                            Action::PayCost { .. }
                                if !matches!(
                                    self.pending_choice,
                                    Some(PendingChoice::Waterbend { .. })
                                ) =>
                            {
                                continue
                            }
                            _ => {}
                        }
                    }
                    // Exact action equality includes its rules player. Keep
                    // the decision kind too: never cross into another window.
                    let index = self
                        .action_space()
                        .filter(|space| space.kind == *kind)
                        .and_then(|space| space.actions.iter().position(|legal| legal == action))
                        .ok_or(StructuredOfferError::WrongDecision)?;
                    done = self.record_compound_step(index, commands)?;
                    executed += 1;
                }
                Ok((done, executed))
            }
            AtomicCommand::SearchAction { action, .. } => {
                let index = self
                    .action_space()
                    .and_then(|space| space.actions.iter().position(|legal| legal == action))
                    .ok_or(StructuredOfferError::WrongDecision)?;
                Ok((self.record_compound_step(index, commands)?, 1))
            }
        }
    }

    fn normalized_attacker_declaration(
        &self,
        command: &AtomicCommand,
    ) -> Result<(crate::flow::declaration::DeclarationCompletion, Vec<Action>), StructuredOfferError>
    {
        use crate::flow::declaration::{DeclarationAssignments, DeclarationCompletion};
        let AtomicCommand::DeclareAttackers {
            binding,
            player,
            attackers,
        } = command
        else {
            return Err(StructuredOfferError::WrongDecision);
        };
        if *binding != OfferSetBinding(self.decision_epoch) {
            return Err(StructuredOfferError::StaleOrIllegal(
                "offer set no longer names the current decision".into(),
            ));
        }
        let (actor, order) = self.current_attacker_declaration()?;
        if actor != *player {
            return Err(StructuredOfferError::WrongDecision);
        }
        let selected = attackers.iter().copied().collect::<BTreeSet<_>>();
        if selected.len() != attackers.len() {
            return Err(StructuredOfferError::StaleOrIllegal(
                "attacker declaration contains a duplicate permanent".into(),
            ));
        }
        let pending = order
            .iter()
            .map(|id| {
                self.permanent_object(*id)
                    .ok_or(StructuredOfferError::InvalidCurrentTarget)
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        if !selected.is_subset(&pending) {
            return Err(StructuredOfferError::StaleOrIllegal(
                "attacker declaration contains a permanent outside the pending batch".into(),
            ));
        }
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or(StructuredOfferError::WrongDecision)?;
        let mut expected_attackers = combat
            .attackers
            .iter()
            .map(|id| {
                self.permanent_object(*id)
                    .ok_or(StructuredOfferError::InvalidCurrentTarget)
            })
            .collect::<Result<Vec<_>, _>>()?;
        expected_attackers.extend(order.iter().filter_map(|id| {
            self.permanent_object(*id)
                .filter(|object| selected.contains(object))
        }));
        let expected = DeclarationCompletion {
            player: *player,
            assignments: DeclarationAssignments::Attackers {
                destination: self.non_active_player(),
                attackers: expected_attackers,
            },
        };
        let planned = order
            .into_iter()
            .map(|permanent| Action::DeclareAttacker {
                player: *player,
                permanent,
                attack: self
                    .permanent_object(permanent)
                    .is_some_and(|object| selected.contains(&object)),
            })
            .collect();
        Ok((expected, planned))
    }

    fn normalized_blocker_declaration(
        &self,
        command: &AtomicCommand,
    ) -> Result<(crate::flow::declaration::DeclarationCompletion, Vec<Action>), StructuredOfferError>
    {
        use crate::flow::{
            block_declaration::BlockProblem,
            declaration::{DeclarationAssignments, DeclarationCompletion},
        };
        let AtomicCommand::DeclareBlockers {
            binding,
            player: actor,
            assignments,
        } = command
        else {
            return Err(StructuredOfferError::WrongDecision);
        };
        if *binding != OfferSetBinding(self.decision_epoch) {
            return Err(StructuredOfferError::StaleOrIllegal(
                "offer set no longer names the current decision".into(),
            ));
        }
        let space = self
            .action_space()
            .filter(|space| space.kind == ActionSpaceKind::DeclareBlocker)
            .ok_or(StructuredOfferError::WrongDecision)?;
        let current = space
            .actions
            .iter()
            .find_map(|action| match action {
                Action::DeclareBlocker { blocker, .. } => Some(*blocker),
                _ => None,
            })
            .ok_or(StructuredOfferError::WrongDecision)?;
        let player = self.non_active_player();
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or(StructuredOfferError::WrongDecision)?;
        let order: Vec<_> = std::iter::once(current)
            .chain(combat.blockers_to_declare.iter().rev().copied())
            .collect();
        let mut submitted = Vec::new();
        if *actor != player {
            return Err(StructuredOfferError::WrongDecision);
        }
        for assignment in assignments {
            let blocker = self
                .lookup_current_permanent(assignment.blocker)
                .map_err(|_| StructuredOfferError::InvalidCurrentTarget)?;
            let attacker = assignment
                .attacker
                .map(|object| {
                    self.lookup_current_permanent(object)
                        .map_err(|_| StructuredOfferError::InvalidCurrentTarget)
                })
                .transpose()?;
            if !order.contains(&blocker) {
                return Err(StructuredOfferError::WrongDecision);
            }
            submitted.push((blocker, attacker));
        }
        BlockProblem::from_game(self)
            .and_then(|problem| problem.validate(self, Some(current), &submitted))
            .map_err(StructuredOfferError::Execution)?;
        let mut assigned = combat.attacker_to_blockers.clone();
        let mut planned = Vec::new();
        for blocker in order {
            let attacker = submitted
                .iter()
                .find(|(id, _)| *id == blocker)
                .ok_or(StructuredOfferError::WrongDecision)?
                .1;
            if let Some(attacker) = attacker {
                assigned.entry(attacker).or_default().push(blocker);
            }
            planned.push(Action::DeclareBlocker {
                player,
                blocker,
                attacker,
            });
        }
        let mut assignments = Vec::new();
        for (attacker, blockers) in assigned {
            for blocker in blockers {
                assignments.push((
                    self.permanent_object(blocker)
                        .ok_or(StructuredOfferError::InvalidCurrentTarget)?,
                    self.permanent_object(attacker)
                        .ok_or(StructuredOfferError::InvalidCurrentTarget)?,
                ));
            }
        }
        Ok((
            DeclarationCompletion {
                player,
                assignments: DeclarationAssignments::Blockers { assignments },
            },
            planned,
        ))
    }

    /// Dedicated combat surface. Nonacting viewers receive no draft authority.
    /// Fixed rows recover the actual committed prefix, never a browser snapshot.
    pub fn declaration_projection(
        &self,
        viewer: PlayerId,
    ) -> Result<Option<crate::flow::declaration::DeclarationProjection>, StructuredOfferError> {
        use crate::{
            experience::{PlayerId as WirePlayer, SubjectRef as WireSubject},
            flow::declaration::{
                DeclarationBinding, DeclarationProjection, DeclarationStage, DeclarationSubject,
            },
        };
        self.ensure_playable()
            .map_err(StructuredOfferError::Execution)?;
        let Some(space) = self.action_space() else {
            return Ok(None);
        };
        if space.player != Some(viewer)
            || !matches!(
                space.kind,
                ActionSpaceKind::DeclareAttacker | ActionSpaceKind::DeclareBlocker
            )
        {
            return Ok(None);
        }
        let _scan = self.board_scan();
        let frame = self.semantic_decision_frame()?;
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or(StructuredOfferError::WrongDecision)?;
        let (stage, pending, domain, destinations) = match space.kind {
            ActionSpaceKind::DeclareAttacker => {
                let (player, pending) = self.current_attacker_declaration()?;
                let mut domain = self.eligible_attackers(player);
                domain.extend(pending.iter().copied());
                domain.extend(combat.attackers.iter().copied());
                let destination = self
                    .decision_target(&Target::Player(self.non_active_player()))
                    .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
                (
                    DeclarationStage::Attackers,
                    pending,
                    domain,
                    vec![destination],
                )
            }
            ActionSpaceKind::DeclareBlocker => {
                let current = space
                    .actions
                    .iter()
                    .find_map(|action| match action {
                        Action::DeclareBlocker { blocker, .. } => Some(*blocker),
                        _ => None,
                    })
                    .ok_or(StructuredOfferError::WrongDecision)?;
                let pending: Vec<_> = std::iter::once(current)
                    .chain(combat.blockers_to_declare.iter().rev().copied())
                    .collect();
                let destinations = combat
                    .attackers
                    .iter()
                    .map(|attacker| self.declaration_object(*attacker))
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    DeclarationStage::Blockers,
                    pending,
                    self.eligible_blockers(self.non_active_player()),
                    destinations,
                )
            }
            _ => return Ok(None),
        };
        let domain: BTreeSet<_> = domain.into_iter().collect();
        let subjects = domain
            .into_iter()
            .map(|permanent| {
                let fixed = !pending.contains(&permanent);
                let assigned_to = match stage {
                    DeclarationStage::Attackers => {
                        combat
                            .attackers
                            .contains(&permanent)
                            .then_some(WireSubject::Player {
                                id: WirePlayer(self.non_active_player().0 as u8),
                            })
                    }
                    DeclarationStage::Blockers => combat
                        .attacker_to_blockers
                        .iter()
                        .find_map(|(attacker, blockers)| {
                            blockers.contains(&permanent).then_some(*attacker)
                        })
                        .map(|attacker| self.declaration_object(attacker).map(|view| view.subject))
                        .transpose()?,
                };
                let supports = if fixed {
                    Vec::new()
                } else {
                    match stage {
                        DeclarationStage::Attackers => {
                            destinations.iter().map(|view| view.subject).collect()
                        }
                        DeclarationStage::Blockers => combat
                            .attackers
                            .iter()
                            .filter(|attacker| self.blocker_pair_eligible(permanent, **attacker))
                            .map(|attacker| {
                                self.declaration_object(*attacker).map(|view| view.subject)
                            })
                            .collect::<Result<_, _>>()?,
                    }
                };
                Ok(DeclarationSubject {
                    object: self.declaration_object(permanent)?,
                    power: self.characteristics(permanent).power,
                    destinations: supports,
                    fixed,
                    assigned_to,
                })
            })
            .collect::<Result<_, StructuredOfferError>>()?;
        let mut constraints = Vec::new();
        if stage == DeclarationStage::Blockers {
            for attacker in &combat.attackers {
                let label = self.declaration_object(*attacker)?.label;
                if self
                    .characteristics(*attacker)
                    .keywords
                    .has(crate::state::card::Keyword::Menace)
                {
                    constraints.push(format!("{label} needs at least two blockers if blocked."));
                }
                if self.has_restriction(
                    *attacker,
                    crate::state::card::Restriction::CantBeBlockedByMoreThanOne,
                ) {
                    constraints.push(format!("{label} can be blocked by at most one creature."));
                }
                if self.must_be_blocked_by_all_able(*attacker) {
                    constraints.push(format!("Creatures able to block {label} must do so. The full declaration must meet as many blocking requirements as legally possible."));
                }
            }
        }
        Ok(Some(DeclarationProjection {
            binding: DeclarationBinding {
                revision: frame.revision,
                fingerprint: frame.fingerprint,
            },
            stage,
            actor: frame.actor,
            rules_player: wire_player_id(
                space
                    .actions
                    .first()
                    .ok_or(StructuredOfferError::MissingActor)?
                    .player(),
            )?,
            subjects,
            destinations,
            constraints,
        }))
    }

    /// Effective battlefield characteristics supply public names, including
    /// copies. `characteristics_set` marks an override, not visibility.
    fn declaration_object(
        &self,
        permanent: PermanentId,
    ) -> Result<crate::experience::DecisionObjectView, StructuredOfferError> {
        let object = self
            .permanent_object(permanent)
            .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
        let mut view = self
            .decision_object(object)
            .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
        let characteristics = self.characteristics(permanent);
        view.label = characteristics.name.to_string();
        Ok(view)
    }

    fn decode_declaration_plan(
        &self,
        plan: &crate::flow::declaration::DeclarationPlan,
    ) -> Result<AtomicCommand, StructuredOfferError> {
        use crate::{experience::SubjectRef as WireSubject, flow::declaration::DeclarationStage};
        self.ensure_playable()
            .map_err(StructuredOfferError::Execution)?;
        let viewer = self
            .action_space()
            .and_then(|space| space.player)
            .ok_or(StructuredOfferError::MissingActor)?;
        let projection = self
            .declaration_projection(viewer)?
            .ok_or(StructuredOfferError::WrongDecision)?;
        if plan.binding != projection.binding {
            return Err(StructuredOfferError::StaleOrIllegal(
                "declaration binding is stale".into(),
            ));
        }
        let pending: Vec<_> = projection
            .subjects
            .iter()
            .filter(|subject| !subject.fixed)
            .collect();
        if plan.assignments.len() != pending.len() {
            return Err(StructuredOfferError::StaleOrIllegal(
                "declaration must answer every pending creature exactly once".into(),
            ));
        }
        let exact_object = |reference: crate::experience::ObjectRenderId| {
            self.state
                .permanents
                .iter()
                .flatten()
                .find_map(|permanent| {
                    (permanent.id.0 == reference.entity
                        && permanent.object.generation() == reference.incarnation)
                        .then_some(permanent.object)
                })
                .ok_or(StructuredOfferError::InvalidCurrentTarget)
        };
        let mut seen = BTreeSet::new();
        let mut attackers = Vec::new();
        let mut assignments = Vec::new();
        for answer in &plan.assignments {
            let row = pending
                .iter()
                .find(|row| row.object.subject == WireSubject::Object { id: answer.subject })
                .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
            if !seen.insert((answer.subject.entity, answer.subject.incarnation)) {
                return Err(StructuredOfferError::StaleOrIllegal(
                    "duplicate creature in declaration".into(),
                ));
            }
            if answer
                .destination
                .0
                .is_some_and(|destination| !row.destinations.contains(&destination))
            {
                return Err(StructuredOfferError::StaleOrIllegal(format!(
                    "{} has no such destination",
                    row.object.label
                )));
            }
            let object = exact_object(answer.subject)?;
            match projection.stage {
                DeclarationStage::Attackers => {
                    if answer.destination.0.is_some() {
                        attackers.push(object);
                    }
                }
                DeclarationStage::Blockers => {
                    let attacker = match answer.destination.0 {
                        None => None,
                        Some(WireSubject::Object { id }) => Some(exact_object(id)?),
                        _ => return Err(StructuredOfferError::InvalidCurrentTarget),
                    };
                    assignments.push(BlockerAssignment {
                        blocker: object,
                        attacker,
                    });
                }
            }
        }
        let player = self
            .action_space()
            .and_then(|space| space.actions.first())
            .map(Action::player)
            .ok_or(StructuredOfferError::MissingActor)?;
        Ok(match projection.stage {
            DeclarationStage::Attackers => AtomicCommand::DeclareAttackers {
                binding: OfferSetBinding(self.decision_epoch),
                player,
                attackers,
            },
            DeclarationStage::Blockers => AtomicCommand::DeclareBlockers {
                binding: OfferSetBinding(self.decision_epoch),
                player,
                assignments,
            },
        })
    }

    /// Read-only legality; no taps, costs, triggers, RNG or capture writes.
    pub fn validate_declaration_plan(
        &self,
        plan: &crate::flow::declaration::DeclarationPlan,
    ) -> Result<(), StructuredOfferError> {
        let command = self.decode_declaration_plan(plan)?;
        match command {
            AtomicCommand::DeclareAttackers { .. } => {
                self.normalized_attacker_declaration(&command).map(|_| ())
            }
            AtomicCommand::DeclareBlockers { .. } => {
                self.normalized_blocker_declaration(&command).map(|_| ())
            }
            _ => Err(StructuredOfferError::WrongDecision),
        }
    }

    pub fn prepare_declaration_plan(
        &self,
        plan: &crate::flow::declaration::DeclarationPlan,
    ) -> Result<crate::flow::declaration::DeclarationTransaction, StructuredOfferError> {
        self.prepare_declaration(&self.decode_declaration_plan(plan)?)
    }

    /// Validate and execute the bound combat declaration without publishing anything.
    /// The last ordinary choice may finish combat; its completion evidence is
    /// checked before consequences, independently of the returned board state.
    pub fn prepare_declaration(
        &self,
        command: &AtomicCommand,
    ) -> Result<crate::flow::declaration::DeclarationTransaction, StructuredOfferError> {
        use crate::flow::declaration::{
            DeclarationAssignments, DeclarationChild, DeclarationCollector, DeclarationStage,
            DeclarationTransaction,
        };

        self.ensure_playable()
            .map_err(StructuredOfferError::Execution)?;
        let (expected, planned) = match command {
            AtomicCommand::DeclareAttackers { .. } => {
                self.normalized_attacker_declaration(command)?
            }
            AtomicCommand::DeclareBlockers { .. } => {
                self.normalized_blocker_declaration(command)?
            }
            _ => return Err(StructuredOfferError::WrongDecision),
        };
        let (stage, kind) = match expected.assignments {
            DeclarationAssignments::Attackers { .. } => (
                DeclarationStage::Attackers,
                ActionSpaceKind::DeclareAttacker,
            ),
            DeclarationAssignments::Blockers { .. } => {
                (DeclarationStage::Blockers, ActionSpaceKind::DeclareBlocker)
            }
        };
        let mut candidate = self.clone_with_card_capture();
        let mut collector = DeclarationCollector::new(stage);
        let mut children = Vec::new();
        // Preserve the positional API's accumulated event batch while each
        // child receipt consumes exactly the events scalar replay consumes.
        let mut events = Vec::new();
        let mut done = false;
        while collector.completion.is_none() {
            if collector.closed {
                return Err(StructuredOfferError::Invariant(
                    "bound declaration ended without completion evidence".into(),
                ));
            }
            let index = candidate
                .action_space()
                .filter(|space| space.kind == kind)
                .and_then(|space| {
                    space
                        .actions
                        .iter()
                        .position(|action| planned.contains(action))
                })
                .ok_or(StructuredOfferError::WrongDecision)?;
            let frame = candidate
                .semantic_decision_frame()
                .map_err(StructuredOfferError::from)?;
            let offer = &frame.offers[index];
            let child = crate::decision::Command {
                command_id: format!("compound-{}-{}", frame.revision, children.len()),
                expected_revision: frame.revision,
                offer_id: offer.id.0,
                answers: Vec::new(),
                object_preconditions: Vec::new(),
            };
            let root = candidate.clone_with_card_capture();
            done = candidate
                .step_with_completion(index, Some(&mut collector))
                .map_err(StructuredOfferError::Execution)?;
            events.extend(candidate.state.observation_events.iter().cloned());
            let (transition, observation, _) = candidate
                .finish_semantic_transition(
                    &child,
                    frame.revision,
                    PlayerId(frame.actor as usize),
                    offer.public_commitment.clone(),
                    done,
                )
                .map_err(StructuredOfferError::from)?;
            children.push(DeclarationChild {
                root,
                command: child,
                transition,
                observation,
            });
        }
        let completion = collector.completion.ok_or_else(|| {
            StructuredOfferError::Invariant("bound declaration did not complete".into())
        })?;
        if completion != expected {
            return Err(StructuredOfferError::StaleOrIllegal(
                "completed declaration differs from the submitted plan".into(),
            ));
        }
        candidate.state.observation_events_mut().extend(events);
        Ok(DeclarationTransaction {
            game: candidate,
            completion,
            children,
            done,
        })
    }

    /// Apply one decoded structured command as a single rules mutation.
    ///
    /// Declarations execute their ordinary choices on a private clone and
    /// publish it only after the whole sequence succeeds. Other presentation
    /// Commands restore on failure. Action-aligned search Commands validate
    /// every fallible precondition before their commit point.
    pub fn apply_atomic_command(
        &mut self,
        command: &AtomicCommand,
    ) -> Result<bool, StructuredOfferError> {
        self.ensure_playable()
            .map_err(StructuredOfferError::Execution)?;
        if matches!(command, AtomicCommand::SearchAction { .. }) {
            return self.apply_search_action(command);
        }
        if matches!(
            command,
            AtomicCommand::DeclareAttackers { .. } | AtomicCommand::DeclareBlockers { .. }
        ) {
            let transaction = self.prepare_declaration(command)?;
            let done = transaction.done;
            *self = transaction.game;
            return Ok(done);
        }
        if let AtomicCommand::DeclarationSequence { kind, .. } = command {
            if *kind != ActionSpaceKind::Waterbend {
                return Err(StructuredOfferError::WrongDecision);
            }
            let mut fork = self.clone_with_card_capture();
            let (done, _) = fork.apply_legacy_command_inner(command, &mut None)?;
            *self = fork;
            return Ok(done);
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
    /// is touched, so no rollback snapshot is needed on this hot path.
    /// Progression failures propagate and quarantine the in-place stream.
    fn apply_search_action(
        &mut self,
        command: &AtomicCommand,
    ) -> Result<bool, StructuredOfferError> {
        #[cfg(feature = "benchmark")]
        let validation_timer = crate::benchmark_search::Timer::new("command_validation");
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
        let index = self
            .current_action_space
            .as_ref()
            .ok_or(StructuredOfferError::NoActiveActionSpace)?
            .actions
            .iter()
            .position(|legal| legal == action)
            .ok_or_else(|| {
                StructuredOfferError::StaleOrIllegal("search action is no longer offered".into())
            })?;

        #[cfg(feature = "benchmark")]
        drop(validation_timer);
        #[cfg(feature = "benchmark")]
        let _rules_timer = crate::benchmark_search::Timer::new("rules_execution");
        self.step(index).map_err(StructuredOfferError::Execution)
    }

    fn apply_atomic_command_inner(
        &mut self,
        command: &AtomicCommand,
    ) -> Result<bool, StructuredOfferError> {
        let binding = match command {
            AtomicCommand::PassPriority { binding, .. }
            | AtomicCommand::CastSpell { binding, .. }
            | AtomicCommand::DeclareAttackers { binding, .. }
            | AtomicCommand::DeclareBlockers { binding, .. }
            | AtomicCommand::DeclarationSequence { binding, .. }
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
                    .map_err(StructuredOfferError::Execution)?;
                self.finish_action_step()
                    .map_err(StructuredOfferError::Execution)
            }
            AtomicCommand::CastSpell {
                player,
                object,
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
                                object: legal_card,
                            } if legal_player == player && legal_card == object
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
                    .map_err(StructuredOfferError::Execution)?;
                let target_action = Action::ChooseTarget {
                    player: *player,
                    target: self.resolve_bound_target(targets[0])?,
                };
                self.execute_action(&target_action)
                    .map_err(StructuredOfferError::Execution)?;
                self.finish_action_step()
                    .map_err(StructuredOfferError::Execution)
            }
            AtomicCommand::DeclareAttackers { .. }
            | AtomicCommand::DeclareBlockers { .. }
            | AtomicCommand::DeclarationSequence { .. } => {
                unreachable!("declaration routed to canonical lowering")
            }
            AtomicCommand::SearchAction { .. } => {
                unreachable!("search command routed to apply_search_action")
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
        let subject = action_space
            .actions
            .first()
            .map(|action| action.player())
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
            if *player != subject {
                return Err(StructuredOfferError::Invariant(
                    "declare-attacker prompt contains another rules player".to_string(),
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
        Ok((subject, declaration_order))
    }

    fn bind_target(&self, target: Target) -> Result<BoundTarget, StructuredOfferError> {
        match target {
            Target::Player(player) => Ok(BoundTarget::Player(player)),
            Target::Permanent(permanent) => self
                .permanent_object(permanent)
                .map(BoundTarget::Permanent)
                .ok_or(StructuredOfferError::InvalidCurrentTarget),
            Target::StackSpell(object_id) => Ok(BoundTarget::StackSpell(object_id)),
            Target::StackAbility(id) => Ok(BoundTarget::StackAbility(id)),
            Target::GraveyardCard(object) => Ok(BoundTarget::GraveyardCard(object)),
        }
    }

    fn resolve_bound_target(&self, target: BoundTarget) -> Result<Target, StructuredOfferError> {
        match target {
            BoundTarget::Player(player) => Ok(Target::Player(player)),
            BoundTarget::Permanent(object_ref) => self
                .lookup_current_permanent(object_ref)
                .map(Target::Permanent)
                .map_err(|_| {
                    StructuredOfferError::StaleOrIllegal(format!(
                        "target object {}@{} is no longer current",
                        object_ref.slot(),
                        object_ref.generation()
                    ))
                }),
            BoundTarget::StackSpell(object_id) => Ok(Target::StackSpell(object_id)),
            BoundTarget::StackAbility(id) => self
                .find_ability_on_stack_index(id)
                .map(|_| Target::StackAbility(id))
                .ok_or(StructuredOfferError::InvalidCurrentTarget),
            BoundTarget::GraveyardCard(object) => {
                if self.state.live(object).is_some()
                    && self.state.zones.zone_of(object) == Some(ZoneType::Graveyard)
                {
                    Ok(Target::GraveyardCard(object))
                } else {
                    Err(StructuredOfferError::InvalidCurrentTarget)
                }
            }
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
                        id: object_render_id(permanent_ref.id, object_ref.generation()),
                    },
                    self.state[permanent_ref.object].name.clone(),
                ))
            }
            BoundTarget::GraveyardCard(object) => {
                if self.state.live(object).is_none() {
                    return Err(StructuredOfferError::InvalidCurrentTarget);
                }
                Ok((
                    SubjectRef::Object {
                        id: object_render_id(self.state[object].id, object.generation()),
                    },
                    self.state[object].name.clone(),
                ))
            }
            BoundTarget::StackSpell(_) | BoundTarget::StackAbility(_) => {
                Err(StructuredOfferError::InvalidCurrentTarget)
            }
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

fn object_render_id(id: PublicId, incarnation: u32) -> ObjectRenderId {
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
        ActionSpaceKind::Learn => Ok(PromptKind::Learn),
        ActionSpaceKind::Waterbend => Ok(PromptKind::Waterbend),
        ActionSpaceKind::Discard => Ok(PromptKind::Discard),
        ActionSpaceKind::LegendRule => Ok(PromptKind::LegendRule),
        ActionSpaceKind::OrderTriggers => Ok(PromptKind::OrderTriggers),
        ActionSpaceKind::GameOver => Err(StructuredOfferError::GameOver),
    }
}

fn search_offer_verb(action: &Action) -> OfferVerb {
    match action {
        Action::PlayLand { .. } => OfferVerb::PlayLand,
        Action::CastSpell { .. } => OfferVerb::Cast,
        Action::ActivateAbility { .. }
        | Action::ActivateCardAbility { .. }
        | Action::Foretell { .. } => OfferVerb::ActivateAbility,
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
        Action::LearnTakeLesson { .. } => OfferVerb::LearnTakeLesson,
        Action::LearnDiscard { .. } => OfferVerb::LearnDiscard,
        Action::SelectCard { .. } => OfferVerb::SelectCard,
        Action::Decline { .. } => OfferVerb::Decline,
        Action::PayCost { .. } => OfferVerb::PayCost,
        Action::ChooseMode { .. } => OfferVerb::ChooseMode,
        Action::ChooseTrigger { .. } => OfferVerb::ChooseTrigger,
        Action::WaterbendTap { .. } => OfferVerb::WaterbendTap,
    }
}
