//! Experimental structured legal offers over the current positional action ABI.
//!
//! The current `ActionSpace` and target legality queries remain authoritative.
//! This module projects a narrow, uncapped typed view for priority pass,
//! single-target casts, complete combat declarations, and
//! commuting waterbend payments,
//! then lowers accepted IDs through the existing rules executor. It
//! intentionally does not own match revisions, prompt persistence, recovery,
//! or policy decoding. `compound_offers` covers the full legal surface and
//! `compound_commands` lowers its atomic selections to canonical Commands.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    agent::action::{Action, ActionSpaceKind},
    cards::content_pack::default_content_pack,
    flow::{cast_permission::PlayRoute, game::Game, play::CastAuthority},
    state::{
        ability::TargetSpec,
        card::SpellProfile,
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
    TapForMana,
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
        route: Option<PlayRoute>,
        profile: SpellProfile,
        targets: Vec<BoundTarget>,
    },
    DeclareAttackers {
        binding: OfferSetBinding,
        player: PlayerId,
        attackers: Vec<ObjectId>,
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
        route: Option<PlayRoute>,
        profile: SpellProfile,
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
    targets: BTreeMap<CandidateId, Action>,
    decline: Action,
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

    pub(crate) fn into_projection(self) -> StructuredOfferProjection {
        self.projection
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
                route,
                profile,
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
                    route: *route,
                    profile: *profile,
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
            InternalOffer::DeclareBlockers { choices } => {
                let mut answers = BTreeMap::new();
                for ChoiceAnswer::Candidates { role, candidates } in &submission.answers {
                    if answers.insert(*role, candidates).is_some() {
                        return Err(StructuredOfferError::DuplicateRole(*role));
                    }
                }
                let mut actions = Vec::with_capacity(choices.len());
                for choice in choices {
                    let selected = answers
                        .remove(&choice.role)
                        .ok_or(StructuredOfferError::MissingAnswer(choice.role))?;
                    let action = match selected.as_slice() {
                        [] => choice.decline.clone(),
                        [candidate] => choice
                            .targets
                            .get(candidate)
                            .cloned()
                            .ok_or(StructuredOfferError::UnknownCandidate(*candidate))?,
                        _ => {
                            return Err(StructuredOfferError::CardinalityOutOfRange {
                                role: choice.role,
                                min: 0,
                                max: 1,
                                actual: selected.len(),
                            })
                        }
                    };
                    actions.push(action);
                }
                if let Some(role) = answers.keys().next() {
                    return Err(StructuredOfferError::UnexpectedRole(*role));
                }
                Ok(AtomicCommand::DeclarationSequence {
                    binding: self.binding,
                    kind: ActionSpaceKind::DeclareBlocker,
                    actions,
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
            let label = self
                .action_description(action)
                .unwrap_or_else(|| format!("{:?}", action.action_type()));
            let public_commitment = match (action_space.kind, action) {
                (_, Action::PassPriority { .. }) => Some(PublicCommitment::PassPriority),
                (_, Action::CastSpell { object, .. } | Action::PlayLand { object, .. }) => {
                    // The chosen face's name: a modal back or an Adventure.
                    let card = default_content_pack()
                        .spell_definition(&self.state[*object], action.spell_profile())
                        .expect("offered face")
                        .name
                        .clone();
                    Some(if matches!(action, Action::PlayLand { .. }) {
                        PublicCommitment::PlayLand { card }
                    } else {
                        PublicCommitment::Cast { card }
                    })
                }
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
            Some(ActionSpaceKind::DeclareAttacker) if self.attacker_declaration_is_required() => {
                Ok(complete)
            }
            Some(ActionSpaceKind::DeclareAttacker) => self.structured_attacker_offers(),
            Some(ActionSpaceKind::DeclareBlocker) => {
                Ok(self.structured_blocker_offers()?.unwrap_or(complete))
            }
            Some(ActionSpaceKind::Waterbend) => {
                Ok(self.structured_waterbend_offers()?.unwrap_or(complete))
            }
            Some(ActionSpaceKind::Priority) => {
                for offer in &mut complete.projection.offers {
                    let Some(InternalOffer::SearchAction { action }) =
                        complete.internal.get(&offer.id)
                    else {
                        continue;
                    };
                    if let Some((projection, binding)) = self.single_target_offer(
                        action,
                        offer.id,
                        offer.actor,
                        self.compound_targets_for_action(action),
                    )? {
                        complete.internal.insert(offer.id, binding);
                        *offer = projection;
                    }
                }
                Ok(complete)
            }
            _ => Ok(complete),
        }
    }

    /// Coupled block declarations use the sequential offers. A product of
    /// independent roles would admit plans the next decision must reject.
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
        if combat
            .attackers
            .iter()
            .any(|&attacker| self.blocks_are_coupled(attacker))
        {
            return Ok(None);
        }
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
            let targets: Vec<_> = self
                .block_options(blocker, &[])
                .into_iter()
                .flatten()
                .collect();
            // Optionless later declarations are auto-resolved by tick(). Keep
            // them explicit only when that engine facility is disabled.
            if targets.is_empty() && self.skip_trivial && blocker != first {
                continue;
            }
            let role = RoleId(
                u16::try_from(choices.len()).map_err(|_| StructuredOfferError::IdentityOverflow)?,
            );
            let mut candidates = Vec::new();
            let mut actions = BTreeMap::new();
            for attacker in targets {
                let id = next_candidate_id(candidates.len())?;
                candidates.push(self.compound_permanent_candidate(attacker, id)?);
                actions.insert(
                    id,
                    Action::DeclareBlocker {
                        player,
                        blocker,
                        attacker: Some(attacker),
                    },
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
                targets: actions,
                decline: Action::DeclareBlocker {
                    player,
                    blocker,
                    attacker: None,
                },
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
                InternalOffer::DeclareBlockers { choices: bound },
            )]),
        }))
    }

    /// Group a commuting subset of waterbend taps and the remaining mana
    /// payment. Taps with triggered effects or competing mana production stay
    /// sequential: their support and information boundary can depend on order.
    fn structured_waterbend_offers(
        &self,
    ) -> Result<Option<StructuredOfferSet>, StructuredOfferError> {
        let Some((player, payment, _)) = self
            .paying()
            .filter(|(_, payment, _)| payment.paying_mana() && payment.waterbends())
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
        let legal_taps = self.help_tap_candidates(player, &payment.operation);
        if legal_taps.iter().any(|id| {
            self.state.permanents[*id]
                .as_ref()
                .is_some_and(|permanent| !self.state[permanent.object].mana_abilities.is_empty())
        }) {
            return Ok(None);
        }
        let (cost, remaining_generic) = payment.mana();
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

    /// The render id a viewer knows `object` by (`Game::object_event_ref`).
    fn object_render_ref(&self, object: ObjectId) -> Result<ObjectRenderId, StructuredOfferError> {
        let event = self
            .object_event_ref(object)
            .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
        Ok(object_render_id(event.entity, event.incarnation))
    }

    fn compound_permanent_candidate(
        &self,
        permanent: PermanentId,
        id: CandidateId,
    ) -> Result<Candidate, StructuredOfferError> {
        let object = self
            .permanent_object(permanent)
            .ok_or(StructuredOfferError::InvalidCurrentTarget)?;
        Ok(Candidate {
            id,
            value: CandidateValue::Subject {
                subject: SubjectRef::Object {
                    id: self.object_render_ref(object)?,
                },
            },
            label: self.observed_object_name(object),
            help: None,
            preview: None,
        })
    }

    /// Lower a complete submission on an exact fork, never mutate the root.
    /// Multi-command offers stop at the end of the native declaration/payment.
    /// No planner or policy observes intermediate states during this operation.
    pub fn compound_commands(
        &self,
        offers: &StructuredOfferSet,
        submission: &OfferSubmission,
    ) -> Result<Vec<crate::decision::Command>, StructuredOfferError> {
        let command = offers.decode(submission)?;
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
                .map_err(|error| StructuredOfferError::Invariant(error.to_string()))?;
            commands.push(crate::decision::Command {
                command_id: format!("compound-{}-{}", frame.revision, commands.len()),
                expected_revision: frame.revision,
                offer_id: frame.offers[action].id.0,
                answers: Vec::new(),
                object_preconditions: Vec::new(),
            });
        }
        self.step(action)
            .map_err(|error| StructuredOfferError::StaleOrIllegal(error.0))
    }

    /// Project the currently covered priority actions without changing state.
    ///
    /// Unsupported legacy actions remain available through `ActionSpace`; this
    /// experimental projection covers pass and casts with a single target
    /// that can be chosen immediately.
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
                Action::CastSpell { .. } => {
                    let id = next_offer_id(offers.len())?;
                    if let Some((offer, binding)) = self.single_target_offer(
                        action,
                        id,
                        wire_actor,
                        self.priority_precondition_targets(action),
                    )? {
                        offers.push(offer);
                        internal.insert(id, binding);
                    }
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

    // Recorded Commands address these candidates by their position in the compact
    // priority projection. Keep its front-face AnyTarget ordering stable.
    fn priority_precondition_targets(&self, action: &Action) -> Option<Vec<Target>> {
        let Action::CastSpell {
            player,
            object,
            profile: SpellProfile::Front,
            ..
        } = action
        else {
            return None;
        };
        let card = &self.state[*object];
        let requirements = card.target_requirements();
        let [requirement] = requirements.as_slice() else {
            return None;
        };
        (card.optional_cost().is_none()
            && requirement.spec == TargetSpec::AnyTarget
            && requirement.min == 1
            && requirement.max == 1)
            .then(|| {
                self.legal_spell_targets(*player, *object, card, requirement)
                    .collect()
            })
    }

    fn single_target_offer(
        &self,
        action: &Action,
        id: OfferId,
        wire_actor: u8,
        targets: Option<Vec<Target>>,
    ) -> Result<Option<(InteractionOffer, InternalOffer)>, StructuredOfferError> {
        let Action::CastSpell {
            player,
            object,
            route,
            profile,
        } = action
        else {
            return Ok(None);
        };
        let Some(legal_targets) = targets else {
            return Ok(None);
        };

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

        let offer = InteractionOffer {
            id,
            actor: wire_actor,
            verb: OfferVerb::Cast,
            public_commitment: None,
            source: Some(SubjectRef::Object {
                id: object_render_id(
                    self.state[*object].id,
                    self.state
                        .live(*object)
                        .ok_or(StructuredOfferError::InvalidCurrentTarget)?
                        .generation(),
                ),
            }),
            label: self
                .action_description(action)
                .unwrap_or_else(|| format!("Cast {}", self.state[*object].name)),
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
        };
        Ok(Some((
            offer,
            InternalOffer::CastSingleTarget {
                player: *player,
                object: *object,
                route: *route,
                profile: *profile,
                role,
                targets: target_by_candidate,
            },
        )))
    }

    /// Bind a target only when it is the next announcement choice. Other
    /// announcements retain their action-aligned offer and native prompts.
    fn compound_targets_for_action(&self, action: &Action) -> Option<Vec<Target>> {
        let Action::CastSpell {
            player,
            object,
            route,
            profile,
        } = *action
        else {
            return None;
        };
        let spell = default_content_pack().spell_definition(&self.state[object], profile)?;
        let requirements = spell.target_requirements();
        let [requirement] = requirements.as_slice() else {
            return None;
        };
        if requirement.min != 1
            || requirement.max != 1
            || requirement.max_x
            || requirement.total_mana_value_x
            || requirement.opponent_chooses
            || !matches!(
                requirement.spec,
                TargetSpec::AnyTarget | TargetSpec::Creature | TargetSpec::Permanent { .. }
            )
            || spell.cast_option().is_some()
            || spell.spell_modes().is_some()
        {
            return None;
        }
        let cast = self.cast_choices(
            player,
            object,
            profile,
            CastAuthority::Priority(
                self.selected_play_route(player, object, profile, route)
                    .ok()?,
            ),
        )?;
        if self.cast_needs_x_announcement(object, cast) {
            return None;
        }
        Some(
            self.legal_spell_targets(player, object, spell, requirement)
                .filter(|target| {
                    self.target_leaves_cost_payable(player, object, cast, &[], *target)
                })
                .collect(),
        )
    }

    /// Project one complete CR 508 attacker declaration as an unordered
    /// multi-select offer. N eligible creatures produce N candidates and
    /// represent all 2^N declarations without enumerating those subsets.
    pub fn structured_attacker_offers(&self) -> Result<StructuredOfferSet, StructuredOfferError> {
        if self.is_game_over() {
            return Err(StructuredOfferError::GameOver);
        }
        // `Select` cannot require one candidate, so a creature that attacks
        // each combat (CR 508.1d) keeps the sequential path.
        if self.attacker_declaration_is_required() {
            return Err(StructuredOfferError::WrongDecision);
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
            let object_ref = self.permanent_object(*attacker).ok_or_else(|| {
                StructuredOfferError::Invariant(
                    "legacy attacker prompt names a missing permanent".to_string(),
                )
            })?;
            let candidate_id = next_candidate_id(candidates.len())?;
            candidates.push(Candidate {
                id: candidate_id,
                value: CandidateValue::Subject {
                    subject: SubjectRef::Object {
                        id: self.object_render_ref(object_ref)?,
                    },
                },
                label: self.observed_object_name(object_ref),
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
        let checkpoint = self.rollback_snapshot();
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
                route,
                profile,
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
                let action = Action::CastSpell {
                    player: *player,
                    object: *object,
                    route: *route,
                    profile: *profile,
                };
                let cast = self
                    .action_space()
                    .filter(|space| space.kind == ActionSpaceKind::Priority)
                    .and_then(|space| self.resolve_action_index(space, &action))
                    .ok_or(StructuredOfferError::WrongDecision)?;
                // Trivial-decision skipping may commit the sole target during the cast step.
                let wanted = self.resolve_bound_target(targets[0])?;
                if !self
                    .compound_targets_for_action(&action)
                    .is_some_and(|legal| legal.contains(&wanted))
                {
                    return Err(StructuredOfferError::StaleOrIllegal(
                        "structured target is no longer legal".into(),
                    ));
                }
                let mut done = self.record_compound_step(cast, commands)?;
                let mut actions = 1;

                // With skip_trivial enabled, a sole legal target may already
                // have been collapsed by tick(). Otherwise walk the explicit
                // ChooseTarget prompt, resolving folded target aliases.
                if self
                    .action_space()
                    .is_some_and(|space| space.kind == ActionSpaceKind::ChooseTarget)
                {
                    let target_action = Action::ChooseTarget {
                        player: *player,
                        target: wanted,
                    };
                    let target = self
                        .action_space()
                        .and_then(|space| self.resolve_action_index(space, &target_action))
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
                let legal: BTreeSet<_> = declaration_order.iter().copied().collect();
                if selected.len() != attackers.len() || !selected.is_subset(&legal) {
                    return Err(StructuredOfferError::StaleOrIllegal(
                        "attacker declaration contains a duplicate or unoffered permanent".into(),
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
            AtomicCommand::DeclarationSequence { kind, actions, .. } => {
                let mut done = false;
                let mut executed = 0;
                if *kind == ActionSpaceKind::Waterbend {
                    let mut remaining_taps: Vec<PermanentId> = Vec::new();
                    let mut want_pay_cost = false;
                    for action in actions {
                        match *action {
                            Action::WaterbendTap { permanent, .. } => {
                                remaining_taps.push(permanent);
                            }
                            Action::PayCost { .. } => {
                                want_pay_cost = true;
                            }
                            _ => return Err(StructuredOfferError::WrongDecision),
                        }
                    }
                    let player = actions
                        .first()
                        .map(Action::player)
                        .ok_or(StructuredOfferError::WrongDecision)?;
                    while let Some(permanent) = remaining_taps.first().copied() {
                        remaining_taps.remove(0);
                        // tick() with skip_trivial may have consumed forced
                        // tap suffix factors (either this permanent or an
                        // interchangeable representative).
                        if self.skip_trivial {
                            if !self
                                .paying()
                                .is_some_and(|(_, payment, _)| payment.waterbends())
                            {
                                break;
                            }
                            if self.state.permanents[permanent]
                                .as_ref()
                                .is_some_and(|perm| perm.tapped)
                            {
                                continue;
                            }
                        }
                        let generic_before = self.paying().map(|(_, payment, _)| payment.mana().1);
                        let wanted = Action::WaterbendTap { player, permanent };
                        let index = self
                            .action_space()
                            .filter(|space| space.kind == ActionSpaceKind::Waterbend)
                            .and_then(|space| self.resolve_action_index(space, &wanted))
                            .ok_or(StructuredOfferError::WrongDecision)?;
                        done = self.record_compound_step(index, commands)?;
                        executed += 1;
                        if self.skip_trivial {
                            let Some(before) = generic_before else {
                                continue;
                            };
                            let Some((_, payment, _)) =
                                self.paying().filter(|(_, payment, _)| payment.waterbends())
                            else {
                                remaining_taps.clear();
                                break;
                            };
                            let after = payment.mana().1;
                            let auto_taps =
                                usize::from(before.saturating_sub(1).saturating_sub(after));
                            if auto_taps > 0 {
                                let mut retired = 0usize;
                                remaining_taps.retain(|&id| {
                                    if retired < auto_taps
                                        && self.state.permanents[id]
                                            .as_ref()
                                            .is_some_and(|perm| perm.tapped)
                                    {
                                        retired += 1;
                                        false
                                    } else {
                                        true
                                    }
                                });
                                let extra = (auto_taps - retired).min(remaining_taps.len());
                                remaining_taps.drain(0..extra);
                            }
                        }
                    }
                    if want_pay_cost
                        && (!self.skip_trivial
                            || self
                                .paying()
                                .is_some_and(|(_, payment, _)| payment.waterbends()))
                    {
                        let wanted = Action::PayCost { player };
                        let index = self
                            .action_space()
                            .filter(|space| space.kind == ActionSpaceKind::Waterbend)
                            .and_then(|space| self.resolve_action_index(space, &wanted))
                            .ok_or(StructuredOfferError::WrongDecision)?;
                        done = self.record_compound_step(index, commands)?;
                        executed += 1;
                    }
                    return Ok((done, executed));
                }
                for action in actions {
                    // Exact action equality includes its rules player. Keep
                    // the decision kind too: never cross into another window.
                    let index = self
                        .action_space()
                        .filter(|space| space.kind == *kind)
                        .and_then(|space| self.resolve_action_index(space, action))
                        .ok_or(StructuredOfferError::WrongDecision)?;
                    done = self.record_compound_step(index, commands)?;
                    executed += 1;
                }
                Ok((done, executed))
            }
            AtomicCommand::SearchAction { action, .. } => {
                let index = self
                    .action_space()
                    .and_then(|space| self.resolve_action_index(space, action))
                    .ok_or(StructuredOfferError::WrongDecision)?;
                Ok((self.record_compound_step(index, commands)?, 1))
            }
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
        let checkpoint = self.rollback_snapshot();
        if matches!(command, AtomicCommand::DeclarationSequence { .. }) {
            return match self.apply_legacy_command_inner(command, &mut None) {
                Ok((done, _)) => Ok(done),
                Err(error) => {
                    *self = checkpoint;
                    Err(error)
                }
            };
        }
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

        #[cfg(feature = "benchmark")]
        drop(validation_timer);
        #[cfg(feature = "benchmark")]
        let _rules_timer = crate::benchmark_search::Timer::new("rules_execution");
        let pending_memory = crate::agent::decision_memory::capture(
            self,
            self.current_action_space.as_ref().unwrap(),
            action,
        );
        let action = action.clone();
        let exposure = self.exposure_for_action(&action);
        self.current_action_space.take();
        self.execute_action(&action).unwrap_or_else(|error| {
            panic!("structured search offer admitted an unexecutable action: {error}")
        });
        crate::agent::decision_memory::commit(self, pending_memory);
        if let Some(choice) = exposure {
            self.commit_capture_choice(choice);
        }
        Ok(self.finish_action_step())
    }

    fn apply_atomic_command_inner(
        &mut self,
        command: &AtomicCommand,
    ) -> Result<bool, StructuredOfferError> {
        // One canonical choice sequence for ordinary and compound commits.
        self.apply_legacy_command_inner(command, &mut None)
            .map(|(done, _)| done)
    }

    /// Whether a creature still to be declared in this combat must attack.
    fn attacker_declaration_is_required(&self) -> bool {
        let current = self
            .current_action_space
            .as_ref()
            .into_iter()
            .flat_map(|space| &space.actions)
            .filter_map(|action| match action {
                Action::DeclareAttacker { permanent, .. } => Some(*permanent),
                _ => None,
            });
        let pending = self
            .state
            .combat
            .as_ref()
            .into_iter()
            .flat_map(|combat| combat.attackers_to_declare.iter().copied());
        current
            .chain(pending)
            .any(|attacker| self.must_attack(attacker))
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
                .stack_index_of(Target::StackAbility(id))
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
                self.lookup_current_permanent(object_ref)
                    .map_err(|_| StructuredOfferError::InvalidCurrentTarget)?;
                Ok((
                    SubjectRef::Object {
                        id: self.object_render_ref(object_ref)?,
                    },
                    self.observed_object_name(object_ref),
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
        | Action::TurnFaceUp { .. }
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
        Action::TapForMana { .. } => OfferVerb::TapForMana,
    }
}
