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

numeric_id!(DecisionOptionId, u32);
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
pub struct DecisionBinding(u64);

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
pub enum DecisionVerb {
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
pub struct DecisionOption {
    pub id: DecisionOptionId,
    pub actor: u8,
    pub verb: DecisionVerb,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_commitment: Option<PublicCommitment>,
    pub source: Option<SubjectRef>,
    pub label: String,
    pub help: Option<String>,
    pub choices: Vec<ChoiceStep>,
    pub confirm_label: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DecisionProjection {
    pub actor: u8,
    pub kind: PromptKind,
    pub offers: Vec<DecisionOption>,
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
pub struct DecisionSubmission {
    pub offer_id: DecisionOptionId,
    pub answers: Vec<ChoiceAnswer>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AtomicCommand {
    PassPriority {
        binding: DecisionBinding,
        player: PlayerId,
    },
    CastSpell {
        binding: DecisionBinding,
        player: PlayerId,
        object: ObjectId,
        route: Option<PlayRoute>,
        profile: SpellProfile,
        targets: Vec<BoundTarget>,
    },
    DeclareAttackers {
        binding: DecisionBinding,
        player: PlayerId,
        attackers: Vec<ObjectId>,
    },
    DeclarationSequence {
        binding: DecisionBinding,
        kind: ActionSpaceKind,
        actions: Vec<Action>,
    },
    SearchAction {
        binding: DecisionBinding,
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
        roles: Vec<RoleSelection<ObjectId>>,
        declaration_order: Vec<ObjectId>,
    },
    CombatAssignments {
        kind: ActionSpaceKind,
        choices: Vec<CombatChoice>,
    },
    Waterbend {
        player: PlayerId,
        roles: Vec<RoleSelection<Action>>,
        min: usize,
        max: usize,
        remaining_generic: usize,
    },
    SearchAction {
        action: Action,
    },
}

#[derive(Clone, Debug)]
struct RoleSelection<T> {
    role: RoleId,
    /// Maps each candidate to the item(s) it selects: one element for subset
    /// roles, or the canonical 1..=k prefix for count roles.
    items: BTreeMap<CandidateId, Vec<T>>,
    min: usize,
    max: usize,
}

impl<T: Clone> RoleSelection<T> {
    fn decode(
        &self,
        answers: &mut BTreeMap<RoleId, &[CandidateId]>,
        out: &mut Vec<T>,
    ) -> Result<(), DecisionSurfaceError> {
        let selected = answers
            .remove(&self.role)
            .ok_or(DecisionSurfaceError::MissingAnswer(self.role))?;
        let mut ids = BTreeSet::new();
        for candidate in selected {
            if !ids.insert(*candidate) {
                return Err(DecisionSurfaceError::DuplicateCandidate(*candidate));
            }
        }
        if selected.len() < self.min || selected.len() > self.max {
            return Err(DecisionSurfaceError::CardinalityOutOfRange {
                role: self.role,
                min: self.min,
                max: self.max,
                actual: selected.len(),
            });
        }
        // Canonical candidate order removes permutation aliases.
        for id in ids {
            let chosen = self
                .items
                .get(&id)
                .ok_or(DecisionSurfaceError::UnknownCandidate(id))?;
            out.extend(chosen.iter().cloned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct CombatChoice {
    role: RoleId,
    targets: BTreeMap<CandidateId, Vec<Action>>,
    decline: Vec<Action>,
}

#[derive(Clone, Debug)]
pub struct DecisionSurface {
    projection: DecisionProjection,
    binding: DecisionBinding,
    internal: BTreeMap<DecisionOptionId, InternalOffer>,
}

impl DecisionSurface {
    pub fn projection(&self) -> &DecisionProjection {
        &self.projection
    }

    pub(crate) fn into_projection(self) -> DecisionProjection {
        self.projection
    }

    pub(crate) fn object_candidate_bindings(
        &self,
    ) -> Vec<(DecisionOptionId, RoleId, CandidateId, ObjectId)> {
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
                InternalOffer::DeclareAttackers { roles, .. } => {
                    for role_entry in roles {
                        for (candidate_id, objects) in &role_entry.items {
                            if let Some(object_ref) = objects.last() {
                                bindings.push((
                                    *offer_id,
                                    role_entry.role,
                                    *candidate_id,
                                    *object_ref,
                                ));
                            }
                        }
                    }
                }
                InternalOffer::CombatAssignments {
                    kind: ActionSpaceKind::DeclareAttacker,
                    choices,
                } => {
                    for choice in choices {
                        for (candidate_id, actions) in &choice.targets {
                            if let Some(Action::DeclareAttacker {
                                attack: Some(crate::flow::combat::AttackTarget::Permanent(defender)),
                                ..
                            }) = actions.first()
                            {
                                bindings.push((*offer_id, choice.role, *candidate_id, *defender));
                            }
                        }
                    }
                }
                // Compound-only bindings lower through exact canonical actions;
                // they are not projected into canonical DecisionFrame offers.
                InternalOffer::CombatAssignments { .. }
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
        submission: &DecisionSubmission,
    ) -> Result<AtomicCommand, DecisionSurfaceError> {
        let offer = self
            .internal
            .get(&submission.offer_id)
            .ok_or(DecisionSurfaceError::UnknownOffer(submission.offer_id))?;

        match offer {
            InternalOffer::PassPriority { player } => {
                if !submission.answers.is_empty() {
                    return Err(DecisionSurfaceError::UnexpectedAnswers);
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
                    return Err(DecisionSurfaceError::WrongCardinality {
                        role: *role,
                        expected: 1,
                        actual: candidates.len(),
                    });
                }

                let candidate = candidates[0];
                let target = targets
                    .get(&candidate)
                    .copied()
                    .ok_or(DecisionSurfaceError::UnknownCandidate(candidate))?;
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
                roles,
                declaration_order,
            } => {
                let mut answers = collect_role_answers(&submission.answers)?;
                let mut decoded = Vec::new();
                for role_entry in roles {
                    role_entry.decode(&mut answers, &mut decoded)?;
                }
                if let Some(role) = answers.keys().next() {
                    return Err(DecisionSurfaceError::UnexpectedRole(*role));
                }
                let selected_objects: BTreeSet<ObjectId> = decoded.into_iter().collect();

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
            InternalOffer::CombatAssignments { kind, choices } => {
                let mut answers = collect_role_answers(&submission.answers)?;
                let mut actions = Vec::with_capacity(choices.len());
                for choice in choices {
                    let selected = answers
                        .remove(&choice.role)
                        .ok_or(DecisionSurfaceError::MissingAnswer(choice.role))?;
                    match selected {
                        [] => actions.extend(choice.decline.iter().cloned()),
                        [candidate] => {
                            let chosen = choice
                                .targets
                                .get(candidate)
                                .ok_or(DecisionSurfaceError::UnknownCandidate(*candidate))?;
                            actions.extend(chosen.iter().cloned());
                        }
                        _ => {
                            return Err(DecisionSurfaceError::CardinalityOutOfRange {
                                role: choice.role,
                                min: 0,
                                max: 1,
                                actual: selected.len(),
                            });
                        }
                    }
                }
                if let Some(role) = answers.keys().next() {
                    return Err(DecisionSurfaceError::UnexpectedRole(*role));
                }
                Ok(AtomicCommand::DeclarationSequence {
                    binding: self.binding,
                    kind: *kind,
                    actions,
                })
            }
            InternalOffer::Waterbend {
                player,
                roles,
                min,
                max,
                remaining_generic,
            } => {
                let mut answers = collect_role_answers(&submission.answers)?;
                let mut actions = Vec::new();
                for role_entry in roles {
                    role_entry.decode(&mut answers, &mut actions)?;
                }
                if let Some(role) = answers.keys().next() {
                    return Err(DecisionSurfaceError::UnexpectedRole(*role));
                }
                let total_taps = actions.len();
                if total_taps < *min || total_taps > *max {
                    return Err(DecisionSurfaceError::CardinalityOutOfRange {
                        role: roles.first().map_or(RoleId(0), |entry| entry.role),
                        min: *min,
                        max: *max,
                        actual: total_taps,
                    });
                }
                if total_taps < *remaining_generic {
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
                    return Err(DecisionSurfaceError::UnexpectedAnswers);
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
pub enum DecisionSurfaceError {
    GameOver,
    NoActiveActionSpace,
    WrongDecision,
    MissingActor,
    IdentityOverflow,
    InvalidCurrentTarget,
    Invariant(String),
    UnknownOffer(DecisionOptionId),
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

impl std::fmt::Display for DecisionSurfaceError {
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

impl std::error::Error for DecisionSurfaceError {}

impl Game {
    /// Project object-bearing candidate bindings for the current priority or
    /// attacker decision without changing state.
    pub fn decision_candidate_surface(&self) -> Result<DecisionSurface, DecisionSurfaceError> {
        if self.is_game_over() {
            return Err(DecisionSurfaceError::GameOver);
        }
        match self.action_space().map(|space| space.kind) {
            Some(ActionSpaceKind::Priority) => self.structured_priority_offers(),
            Some(ActionSpaceKind::DeclareAttacker) => self.structured_attacker_offers(),
            Some(_) => Err(DecisionSurfaceError::WrongDecision),
            None => Err(DecisionSurfaceError::NoActiveActionSpace),
        }
    }

    /// Deprecated compatibility alias for [`Self::decision_candidate_surface`] (removed in 0.3.0).
    pub fn structured_offers(&self) -> Result<DecisionSurface, DecisionSurfaceError> {
        self.decision_candidate_surface()
    }

    /// Project one stable action-aligned decision option for every action in
    /// the current authoritative policy surface.
    pub fn decision_surface(&self) -> Result<DecisionSurface, DecisionSurfaceError> {
        if self.is_game_over() {
            return Err(DecisionSurfaceError::GameOver);
        }
        let action_space = self
            .action_space()
            .ok_or(DecisionSurfaceError::NoActiveActionSpace)?;
        let actor = action_space
            .player
            .ok_or(DecisionSurfaceError::MissingActor)?;
        let wire_actor = wire_player_id(actor)?;
        if action_space.actions.is_empty() {
            return Err(DecisionSurfaceError::Invariant(
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
            offers.push(DecisionOption {
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

        Ok(DecisionSurface {
            projection: DecisionProjection {
                actor: wire_actor,
                kind: search_prompt_kind(action_space.kind)?,
                offers,
            },
            binding: DecisionBinding(self.decision_epoch),
            internal,
        })
    }

    /// Deprecated compatibility alias for [`Self::decision_surface`] (removed in 0.3.0).
    pub fn structured_search_offers(&self) -> Result<DecisionSurface, DecisionSurfaceError> {
        self.decision_surface()
    }

    /// Complete legal policy surface. Only engine-owned atomic choices group;
    /// every other decision retains its ordinary action-aligned offer.
    pub fn compound_offers(&self) -> Result<DecisionSurface, DecisionSurfaceError> {
        let mut complete = self.decision_surface()?;
        match self.action_space().map(|space| space.kind) {
            Some(ActionSpaceKind::DeclareAttacker)
                if self.attacker_declaration_is_constrained() =>
            {
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
    fn structured_blocker_offers(&self) -> Result<Option<DecisionSurface>, DecisionSurfaceError> {
        let space = self
            .action_space()
            .ok_or(DecisionSurfaceError::NoActiveActionSpace)?;
        let player = space
            .actions
            .first()
            .map(|action| action.player())
            .ok_or(DecisionSurfaceError::MissingActor)?;
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or(DecisionSurfaceError::WrongDecision)?;
        if combat
            .attackers()
            .any(|attacker| self.blocks_are_coupled(attacker))
        {
            return Ok(None);
        }
        let order = self.current_blocker_order(player)?;
        if !combat.declined_blockers.is_empty()
            && self.any_interchangeable_permanent(&combat.declined_blockers, &order)
        {
            return Ok(None);
        }
        let first = order[0];
        let mut blocker_targets: Vec<(PermanentId, Vec<PermanentId>)> = Vec::new();
        let mut all_attackers: Vec<PermanentId> = Vec::new();
        for blocker in order {
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
            for &attacker in &targets {
                if !all_attackers.contains(&attacker) {
                    all_attackers.push(attacker);
                }
            }
            blocker_targets.push((blocker, targets));
        }

        let attacker_classes = self.partition_interchangeable_permanents(&all_attackers);
        let attacker_class_of = class_index(&attacker_classes);
        let blocker_ids: Vec<PermanentId> = blocker_targets.iter().map(|(b, _)| *b).collect();
        let blocker_classes = self.partition_interchangeable_permanents(&blocker_ids);
        let targets_of: BTreeMap<PermanentId, Vec<PermanentId>> =
            blocker_targets.into_iter().collect();

        // Track how many prior blocker roles could already block each attacker
        // class so blocker #k receives up to `min(class_size, k + 1)` distinct
        // representatives (preserving split-block vs double-block outcomes).
        let mut attacker_class_seen = vec![0usize; attacker_classes.len()];
        let mut choices = Vec::new();
        let mut bound = Vec::new();

        for bclass in blocker_classes {
            let first_blocker = bclass[0];
            let legal_targets = &targets_of[&first_blocker];
            let blocker_name = self
                .compound_permanent_candidate(first_blocker, CandidateId(0))?
                .label;

            // Count-based role when multiple identical blockers share either no
            // legal targets or a single attacker.
            if bclass.len() >= 2 && legal_targets.len() <= 1 {
                let role = next_role_id(choices.len())?;
                let mut candidates = Vec::new();
                let mut actions = BTreeMap::new();
                if let [attacker] = legal_targets.as_slice() {
                    let attacker = *attacker;
                    let attacker_name = self
                        .compound_permanent_candidate(attacker, CandidateId(0))?
                        .label;
                    candidates.reserve(bclass.len());
                    for count in 1..=bclass.len() {
                        let id = next_candidate_id(candidates.len())?;
                        let mut candidate = self.compound_permanent_candidate(attacker, id)?;
                        candidate.label = format!("{count} × {blocker_name} → {attacker_name}");
                        candidates.push(candidate);
                        let seq = bclass
                            .iter()
                            .enumerate()
                            .map(|(idx, &blocker)| Action::DeclareBlocker {
                                player,
                                blocker,
                                attacker: (idx < count).then_some(attacker),
                            })
                            .collect();
                        actions.insert(id, seq);
                    }
                }
                let max = usize::from(!candidates.is_empty());
                choices.push(select_step(
                    role,
                    choices.len(),
                    format!("Block with {blocker_name}"),
                    candidates,
                    0,
                    max,
                )?);
                bound.push(CombatChoice {
                    role,
                    targets: actions,
                    decline: bclass
                        .iter()
                        .map(|&blocker| Action::DeclareBlocker {
                            player,
                            blocker,
                            attacker: None,
                        })
                        .collect(),
                });
                continue;
            }

            for blocker in bclass {
                let targets = &targets_of[&blocker];
                let role = next_role_id(choices.len())?;
                let mut per_class_allowed = vec![0usize; attacker_classes.len()];
                let mut per_class_emitted = vec![0usize; attacker_classes.len()];
                for &attacker in targets {
                    let class_idx = attacker_class_of[&attacker];
                    if per_class_allowed[class_idx] == 0 {
                        attacker_class_seen[class_idx] += 1;
                        per_class_allowed[class_idx] = attacker_classes[class_idx]
                            .len()
                            .min(attacker_class_seen[class_idx]);
                    }
                }
                let mut candidates = Vec::new();
                let mut actions = BTreeMap::new();
                for &attacker in targets {
                    let class_idx = attacker_class_of[&attacker];
                    if per_class_emitted[class_idx] >= per_class_allowed[class_idx] {
                        continue;
                    }
                    per_class_emitted[class_idx] += 1;
                    let id = next_candidate_id(candidates.len())?;
                    candidates.push(self.compound_permanent_candidate(attacker, id)?);
                    actions.insert(
                        id,
                        vec![Action::DeclareBlocker {
                            player,
                            blocker,
                            attacker: Some(attacker),
                        }],
                    );
                }
                let name = self
                    .compound_permanent_candidate(blocker, CandidateId(0))?
                    .label;
                let max = usize::from(!candidates.is_empty());
                choices.push(select_step(
                    role,
                    choices.len(),
                    format!("Block with {name}"),
                    candidates,
                    0,
                    max,
                )?);
                bound.push(CombatChoice {
                    role,
                    targets: actions,
                    decline: vec![Action::DeclareBlocker {
                        player,
                        blocker,
                        attacker: None,
                    }],
                });
            }
        }
        let offer = DecisionOption {
            id: DecisionOptionId(0),
            actor: wire_player_id(space.player.ok_or(DecisionSurfaceError::MissingActor)?)?,
            verb: DecisionVerb::DeclareBlockers,
            public_commitment: None,
            source: None,
            label: "Declare blockers".into(),
            help: None,
            choices,
            confirm_label: "Declare blockers".into(),
        };
        Ok(Some(DecisionSurface {
            projection: DecisionProjection {
                actor: offer.actor,
                kind: PromptKind::DeclareBlockers,
                offers: vec![offer],
            },
            binding: DecisionBinding(self.decision_epoch),
            internal: BTreeMap::from([(
                DecisionOptionId(0),
                InternalOffer::CombatAssignments {
                    kind: ActionSpaceKind::DeclareBlocker,
                    choices: bound,
                },
            )]),
        }))
    }

    /// Group a commuting subset of waterbend taps and the remaining mana
    /// payment. Taps with triggered effects or competing mana production stay
    /// sequential: their support and information boundary can depend on order.
    fn structured_waterbend_offers(&self) -> Result<Option<DecisionSurface>, DecisionSurfaceError> {
        let Some((player, payment, _)) = self
            .paying()
            .filter(|(_, payment, _)| payment.paying_mana() && payment.waterbends())
        else {
            // Resolving waterbend costs (e.g. OptionalPayment / Ward) live in
            // Decision::Payment rather than PendingChoice::Pay; fall back to
            // the folded sequential space.
            return Ok(None);
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
                DecisionSurfaceError::Invariant("waterbend has no payable completion".into())
            })?;
        let classes = self.partition_interchangeable_permanents(&legal_taps);
        let total_legal = legal_taps.len();
        let mut choices = Vec::new();
        let mut roles = Vec::new();
        if classes.iter().all(|class| class.len() == 1) || max <= 1 {
            let role = RoleId(0);
            let mut candidates = Vec::new();
            let mut items = BTreeMap::new();
            for class in &classes {
                let permanent = class[0];
                let id = next_candidate_id(candidates.len())?;
                candidates.push(self.compound_permanent_candidate(permanent, id)?);
                items.insert(id, vec![Action::WaterbendTap { player, permanent }]);
            }
            choices.push(select_step(
                role,
                0,
                "Tap for waterbend".into(),
                candidates,
                min,
                max,
            )?);
            roles.push(RoleSelection {
                role,
                items,
                min,
                max,
            });
        } else if classes.len() == 1 || min == total_legal || (min == 0 && total_legal <= max) {
            for class in &classes {
                let role = next_role_id(choices.len())?;
                let others = total_legal - class.len();
                let class_min = min.saturating_sub(others);
                let class_max = class.len().min(max);
                let name = self
                    .compound_permanent_candidate(class[0], CandidateId(0))?
                    .label;
                let (candidates, items, role_min, role_max) = if class.len() == 1 {
                    let permanent = class[0];
                    let id = CandidateId(0);
                    let candidate = self.compound_permanent_candidate(permanent, id)?;
                    (
                        vec![candidate],
                        BTreeMap::from([(id, vec![Action::WaterbendTap { player, permanent }])]),
                        class_min,
                        class_max,
                    )
                } else {
                    let mut candidates = Vec::new();
                    let mut counts = BTreeMap::new();
                    for count in class_min.max(1)..=class_max {
                        let id = next_candidate_id(candidates.len())?;
                        let mut candidate =
                            self.compound_permanent_candidate(class[count - 1], id)?;
                        if count > 1 {
                            candidate.label = format!("{count} × {name}");
                        }
                        candidates.push(candidate);
                        let seq = class[..count]
                            .iter()
                            .map(|&permanent| Action::WaterbendTap { player, permanent })
                            .collect();
                        counts.insert(id, seq);
                    }
                    let role_min = usize::from(class_min > 0);
                    let role_max = usize::from(!candidates.is_empty());
                    (candidates, counts, role_min, role_max)
                };
                choices.push(select_step(
                    role,
                    choices.len(),
                    format!("Tap {name} for waterbend"),
                    candidates,
                    role_min,
                    role_max,
                )?);
                roles.push(RoleSelection {
                    role,
                    items,
                    min: role_min,
                    max: role_max,
                });
            }
        } else {
            // Multi-class count roles with a coupled total bound (min..=max)
            // admit schema-valid role products outside that range. Fall back to
            // the sequential ActionSpace, which fold_action_space already folds
            // to one tap offer per equivalence class.
            return Ok(None);
        }
        let offer = DecisionOption {
            id: DecisionOptionId(0),
            actor: wire_player_id(
                self.action_space()
                    .and_then(|space| space.player)
                    .ok_or(DecisionSurfaceError::MissingActor)?,
            )?,
            verb: DecisionVerb::PayWaterbend,
            public_commitment: None,
            source: None,
            label: "Pay waterbend".into(),
            help: None,
            choices,
            confirm_label: "Pay waterbend".into(),
        };
        Ok(Some(DecisionSurface {
            projection: DecisionProjection {
                actor: offer.actor,
                kind: PromptKind::Waterbend,
                offers: vec![offer],
            },
            binding: DecisionBinding(self.decision_epoch),
            internal: BTreeMap::from([(
                DecisionOptionId(0),
                InternalOffer::Waterbend {
                    player,
                    roles,
                    min,
                    max,
                    remaining_generic: usize::from(remaining_generic),
                },
            )]),
        }))
    }

    /// The render id a viewer knows `object` by (`Game::object_event_ref`).
    fn object_render_ref(&self, object: ObjectId) -> Result<ObjectRenderId, DecisionSurfaceError> {
        let event = self
            .object_event_ref(object)
            .ok_or(DecisionSurfaceError::InvalidCurrentTarget)?;
        Ok(object_render_id(event.entity, event.incarnation))
    }

    fn compound_permanent_candidate(
        &self,
        permanent: PermanentId,
        id: CandidateId,
    ) -> Result<Candidate, DecisionSurfaceError> {
        let object = self
            .permanent_object(permanent)
            .ok_or(DecisionSurfaceError::InvalidCurrentTarget)?;
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
        offers: &DecisionSurface,
        submission: &DecisionSubmission,
    ) -> Result<Vec<crate::decision::Command>, DecisionSurfaceError> {
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
    ) -> Result<bool, DecisionSurfaceError> {
        if let Some(commands) = commands {
            let frame = self
                .semantic_decision_frame()
                .map_err(|error| DecisionSurfaceError::Invariant(error.to_string()))?;
            commands.push(crate::decision::Command {
                command_id: format!("compound-{}-{}", frame.revision, commands.len()),
                expected_revision: frame.revision,
                offer_id: frame.offers[action].id.0,
                answers: Vec::new(),
                object_preconditions: Vec::new(),
            });
        }
        self.step(action)
            .map_err(|error| DecisionSurfaceError::StaleOrIllegal(error.0))
    }

    /// Project the currently covered priority actions without changing state.
    ///
    /// Unsupported legacy actions remain available through `ActionSpace`; this
    /// experimental projection covers pass and casts with a single target
    /// that can be chosen immediately.
    pub fn structured_priority_offers(&self) -> Result<DecisionSurface, DecisionSurfaceError> {
        if self.is_game_over() {
            return Err(DecisionSurfaceError::GameOver);
        }
        let action_space = self
            .action_space()
            .ok_or(DecisionSurfaceError::NoActiveActionSpace)?;
        if action_space.kind != ActionSpaceKind::Priority {
            return Err(DecisionSurfaceError::WrongDecision);
        }
        let actor = action_space
            .player
            .ok_or(DecisionSurfaceError::MissingActor)?;
        let wire_actor = wire_player_id(actor)?;

        let mut offers = Vec::new();
        let mut internal = BTreeMap::new();

        for action in &action_space.actions {
            match action {
                Action::PassPriority { player } => {
                    let id = next_offer_id(offers.len())?;
                    offers.push(DecisionOption {
                        id,
                        actor: wire_actor,
                        verb: DecisionVerb::PassPriority,
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

        Ok(DecisionSurface {
            projection: DecisionProjection {
                actor: wire_actor,
                kind: PromptKind::Priority,
                offers,
            },
            binding: DecisionBinding(self.decision_epoch),
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
        id: DecisionOptionId,
        wire_actor: u8,
        targets: Option<Vec<Target>>,
    ) -> Result<Option<(DecisionOption, InternalOffer)>, DecisionSurfaceError> {
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
        let legal_targets = self.deduplicate_interchangeable_targets(legal_targets);

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

        let offer = DecisionOption {
            id,
            actor: wire_actor,
            verb: DecisionVerb::Cast,
            public_commitment: None,
            source: Some(SubjectRef::Object {
                id: object_render_id(
                    self.state[*object].id,
                    self.state
                        .live(*object)
                        .ok_or(DecisionSurfaceError::InvalidCurrentTarget)?
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
    pub fn structured_attacker_offers(&self) -> Result<DecisionSurface, DecisionSurfaceError> {
        if self.is_game_over() {
            return Err(DecisionSurfaceError::GameOver);
        }
        // `Select` cannot require one candidate, honor a dependent defender
        // restriction, or forbid a creature suppressed by an earlier
        // interchangeable decline, so keep the sequential path.
        if self.attacker_declaration_is_constrained() {
            return Err(DecisionSurfaceError::WrongDecision);
        }
        let (player, declaration_order) = self.current_attacker_declaration()?;
        let wire_actor = wire_player_id(
            self.action_space()
                .and_then(|space| space.player)
                .ok_or(DecisionSurfaceError::MissingActor)?,
        )?;
        let defenders: Vec<_> = self.declaration_targets(player).collect();
        if defenders.len() > 1 {
            return self.structured_attack_assignments(
                player,
                wire_actor,
                &declaration_order,
                &defenders,
            );
        }
        let classes = self.partition_interchangeable_permanents(&declaration_order);
        let mut choices = Vec::new();
        let mut roles = Vec::new();
        let (singletons, multi_classes): (Vec<_>, Vec<_>) =
            classes.into_iter().partition(|class| class.len() == 1);

        if !singletons.is_empty() {
            let role = next_role_id(choices.len() + 1)?;
            let mut candidates = Vec::with_capacity(singletons.len());
            let mut items = BTreeMap::new();
            for class in singletons {
                let attacker = class[0];
                let object_ref = self.permanent_object(attacker).ok_or_else(|| {
                    DecisionSurfaceError::Invariant(
                        "legacy attacker prompt names a missing permanent".to_string(),
                    )
                })?;
                let candidate_id = next_candidate_id(candidates.len())?;
                candidates.push(self.compound_permanent_candidate(attacker, candidate_id)?);
                items.insert(candidate_id, vec![object_ref]);
            }
            let max = candidates.len();
            choices.push(select_step(
                role,
                choices.len(),
                "Attackers".to_string(),
                candidates,
                0,
                max,
            )?);
            roles.push(RoleSelection {
                role,
                items,
                min: 0,
                max,
            });
        }

        for class in multi_classes {
            let role = next_role_id(choices.len() + 1)?;
            let class_objects: Vec<ObjectId> = class
                .iter()
                .map(|&permanent| {
                    self.permanent_object(permanent).ok_or_else(|| {
                        DecisionSurfaceError::Invariant(
                            "legacy attacker prompt names a missing permanent".to_string(),
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            let name = self.observed_object_name(class_objects[0]);
            let mut candidates = Vec::with_capacity(class_objects.len());
            let mut items = BTreeMap::new();
            for count in 1..=class_objects.len() {
                let candidate_id = next_candidate_id(candidates.len())?;
                let mut candidate =
                    self.compound_permanent_candidate(class[count - 1], candidate_id)?;
                if count > 1 {
                    candidate.label = format!("{count} × {name}");
                }
                candidates.push(candidate);
                items.insert(candidate_id, class_objects[..count].to_vec());
            }
            choices.push(select_step(
                role,
                choices.len(),
                format!("Attackers ({name})"),
                candidates,
                0,
                1,
            )?);
            roles.push(RoleSelection {
                role,
                items,
                min: 0,
                max: 1,
            });
        }

        let id = DecisionOptionId(0);
        let offer = DecisionOption {
            id,
            actor: wire_actor,
            verb: DecisionVerb::DeclareAttackers,
            public_commitment: None,
            source: None,
            label: "Declare attackers".to_string(),
            help: None,
            choices,
            confirm_label: "Declare attackers".to_string(),
        };

        Ok(DecisionSurface {
            projection: DecisionProjection {
                actor: wire_actor,
                kind: PromptKind::DeclareAttackers,
                offers: vec![offer],
            },
            binding: DecisionBinding(self.decision_epoch),
            internal: BTreeMap::from([(
                id,
                InternalOffer::DeclareAttackers {
                    player,
                    roles,
                    declaration_order: declaration_order
                        .into_iter()
                        .map(|permanent| {
                            self.permanent_object(permanent)
                                .ok_or(DecisionSurfaceError::InvalidCurrentTarget)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                },
            )]),
        })
    }

    /// Each attacker chooses at most one defender. Dependent restrictions
    /// (Tomik) keep the sequential choices, as do dependent blocks.
    fn structured_attack_assignments(
        &self,
        player: PlayerId,
        actor: u8,
        attackers: &[PermanentId],
        defenders: &[crate::flow::combat::AttackTarget],
    ) -> Result<DecisionSurface, DecisionSurfaceError> {
        use crate::flow::combat::AttackTarget;
        let defender_permanents: Vec<PermanentId> = defenders
            .iter()
            .filter_map(|&defender| match defender {
                AttackTarget::Permanent(object) => self.lookup_current_permanent(object).ok(),
                AttackTarget::Player(_) => None,
            })
            .collect();
        let defender_classes = self.partition_interchangeable_permanents(&defender_permanents);
        let defender_class_of = class_index(&defender_classes);
        let mut choices = Vec::new();
        let mut bound = Vec::new();
        for (attacker_idx, &permanent) in attackers.iter().enumerate() {
            let role = next_role_id(bound.len())?;
            let mut per_class_emitted = vec![0usize; defender_classes.len()];
            let mut candidates = Vec::new();
            let mut targets = BTreeMap::new();
            for &defender in defenders {
                let candidate = match defender {
                    AttackTarget::Player(seat) => {
                        let id = next_candidate_id(candidates.len())?;
                        Candidate {
                            id,
                            value: CandidateValue::Subject {
                                subject: SubjectRef::Player {
                                    id: wire_player_id(seat)?,
                                },
                            },
                            label: format!("Seat {}", seat.0 + 1),
                            help: None,
                            preview: None,
                        }
                    }
                    AttackTarget::Permanent(object) => {
                        let perm = self
                            .lookup_current_permanent(object)
                            .map_err(|_| DecisionSurfaceError::InvalidCurrentTarget)?;
                        if let Some(&class_idx) = defender_class_of.get(&perm) {
                            let limit = defender_classes[class_idx].len().min(attacker_idx + 1);
                            if per_class_emitted[class_idx] >= limit {
                                continue;
                            }
                            per_class_emitted[class_idx] += 1;
                        }
                        let id = next_candidate_id(candidates.len())?;
                        self.compound_permanent_candidate(perm, id)?
                    }
                };
                let id = candidate.id;
                candidates.push(candidate);
                targets.insert(
                    id,
                    vec![Action::DeclareAttacker {
                        player,
                        permanent,
                        attack: Some(defender),
                    }],
                );
            }
            let object = self
                .permanent_object(permanent)
                .ok_or(DecisionSurfaceError::InvalidCurrentTarget)?;
            choices.push(select_step(
                role,
                choices.len(),
                format!("Attack with {}", self.observed_object_name(object)),
                candidates,
                0,
                1,
            )?);
            bound.push(CombatChoice {
                role,
                targets,
                decline: vec![Action::DeclareAttacker {
                    player,
                    permanent,
                    attack: None,
                }],
            });
        }
        let id = DecisionOptionId(0);
        Ok(DecisionSurface {
            projection: DecisionProjection {
                actor,
                kind: PromptKind::DeclareAttackers,
                offers: vec![DecisionOption {
                    id,
                    actor,
                    verb: DecisionVerb::DeclareAttackers,
                    public_commitment: None,
                    source: None,
                    label: "Declare attackers".into(),
                    help: None,
                    choices,
                    confirm_label: "Declare attackers".into(),
                }],
            },
            binding: DecisionBinding(self.decision_epoch),
            internal: BTreeMap::from([(
                id,
                InternalOffer::CombatAssignments {
                    kind: ActionSpaceKind::DeclareAttacker,
                    choices: bound,
                },
            )]),
        })
    }

    /// Decode IDs against one exact offer set and apply the resulting command
    /// through the existing rules path as one transactional call.
    pub fn apply_offer_submission(
        &mut self,
        offers: &DecisionSurface,
        submission: &DecisionSubmission,
    ) -> Result<bool, DecisionSurfaceError> {
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
        offers: &DecisionSurface,
        submission: &DecisionSubmission,
    ) -> Result<(bool, usize), DecisionSurfaceError> {
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
    ) -> Result<(bool, usize), DecisionSurfaceError> {
        let binding = match command {
            AtomicCommand::PassPriority { binding, .. }
            | AtomicCommand::CastSpell { binding, .. }
            | AtomicCommand::DeclareAttackers { binding, .. }
            | AtomicCommand::DeclarationSequence { binding, .. }
            | AtomicCommand::SearchAction { binding, .. } => *binding,
        };
        if binding != DecisionBinding(self.decision_epoch) {
            return Err(DecisionSurfaceError::StaleOrIllegal(
                "offer set no longer names the current decision".to_string(),
            ));
        }
        if self.is_game_over() {
            return Err(DecisionSurfaceError::GameOver);
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
                    .ok_or(DecisionSurfaceError::WrongDecision)?;
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
                    return Err(DecisionSurfaceError::StaleOrIllegal(format!(
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
                    .ok_or(DecisionSurfaceError::WrongDecision)?;
                // Trivial-decision skipping may commit the sole target during the cast step.
                let wanted = self.resolve_bound_target(targets[0])?;
                if !self
                    .compound_targets_for_action(&action)
                    .is_some_and(|legal| legal.contains(&wanted))
                {
                    return Err(DecisionSurfaceError::StaleOrIllegal(
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
                            DecisionSurfaceError::StaleOrIllegal(
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
                    return Err(DecisionSurfaceError::WrongDecision);
                }
                let selected: BTreeSet<_> = attackers
                    .iter()
                    .map(|object_ref| {
                        self.lookup_current_permanent(*object_ref).map_err(|_| {
                            DecisionSurfaceError::StaleOrIllegal(format!(
                                "attacker object {}@{} is no longer current",
                                object_ref.slot(),
                                object_ref.generation()
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let legal: BTreeSet<_> = declaration_order.iter().copied().collect();
                if selected.len() != attackers.len() || !selected.is_subset(&legal) {
                    return Err(DecisionSurfaceError::StaleOrIllegal(
                        "attacker declaration contains a duplicate or unoffered permanent".into(),
                    ));
                }
                let defender = crate::flow::combat::AttackTarget::Player(self.non_active_player());
                let planned: Vec<crate::flow::decision_group::PlannedAttackEntry> = selected
                    .into_iter()
                    .map(|attacker| crate::flow::decision_group::PlannedAttackEntry {
                        attacker,
                        defender,
                        raw_defender: None,
                        committed: false,
                    })
                    .collect();
                let mut normalizer = crate::flow::decision_group::AttackPlanNormalizer::new(
                    declaration_order,
                    planned,
                );
                let mut actions = 0;
                let mut done = false;
                // The final choice may skip trivial windows all the way to a
                // later turn's declaration, even by this same actor. Lower only
                // the batch bound at the root, never a fresh prompt of this kind.
                while let Some(step) = normalizer.next_step(self, *player) {
                    let action = step.action_index.ok_or_else(|| {
                        DecisionSurfaceError::Invariant(
                            "legacy attacker prompt lacks a binary choice".to_string(),
                        )
                    })?;
                    done = self.record_compound_step(action, commands)?;
                    actions += 1;
                }
                if !self.skip_trivial && !normalizer.all_asked() {
                    return Err(DecisionSurfaceError::WrongDecision);
                }
                if !normalizer.all_committed() {
                    return Err(DecisionSurfaceError::StaleOrIllegal(
                        "attacker declaration could not commit every planned attacker".into(),
                    ));
                }
                Ok((done, actions))
            }
            AtomicCommand::DeclarationSequence { kind, actions, .. } => match *kind {
                ActionSpaceKind::DeclareAttacker => self.lower_attacker_sequence(actions, commands),
                ActionSpaceKind::DeclareBlocker => self.lower_blocker_sequence(actions, commands),
                ActionSpaceKind::Waterbend => self.lower_waterbend_sequence(actions, commands),
                _ => {
                    let mut done = false;
                    let mut executed = 0;
                    for action in actions {
                        // Exact action equality includes its rules player. Keep
                        // the decision kind too: never cross into another window.
                        let index = self
                            .action_space()
                            .filter(|space| space.kind == *kind)
                            .and_then(|space| self.resolve_action_index(space, action))
                            .ok_or(DecisionSurfaceError::WrongDecision)?;
                        done = self.record_compound_step(index, commands)?;
                        executed += 1;
                    }
                    Ok((done, executed))
                }
            },
            AtomicCommand::SearchAction { action, .. } => {
                let index = self
                    .action_space()
                    .and_then(|space| self.resolve_action_index(space, action))
                    .ok_or(DecisionSurfaceError::WrongDecision)?;
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
    ) -> Result<bool, DecisionSurfaceError> {
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
    ) -> Result<bool, DecisionSurfaceError> {
        #[cfg(feature = "benchmark")]
        let validation_timer = crate::benchmark_search::Timer::new("command_validation");
        let AtomicCommand::SearchAction { binding, action } = command else {
            return Err(DecisionSurfaceError::WrongDecision);
        };
        if *binding != DecisionBinding(self.decision_epoch) {
            return Err(DecisionSurfaceError::StaleOrIllegal(
                "offer set no longer names the current decision".to_string(),
            ));
        }
        if self.is_game_over() {
            return Err(DecisionSurfaceError::GameOver);
        }
        let offered = self
            .action_space()
            .ok_or(DecisionSurfaceError::NoActiveActionSpace)?
            .actions
            .iter()
            .any(|legal| legal == action);
        if !offered {
            return Err(DecisionSurfaceError::StaleOrIllegal(
                "search action is no longer offered".to_string(),
            ));
        }

        #[cfg(feature = "benchmark")]
        drop(validation_timer);
        #[cfg(feature = "benchmark")]
        let _rules_timer = crate::benchmark_search::Timer::new("rules_execution");
        let pending_memory =
            crate::agent::decision_memory::capture(self, self.action_space().unwrap(), action);
        let action = action.clone();
        let exposure = self.exposure_for_action(&action);
        self.clear_published_decision();
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
    ) -> Result<bool, DecisionSurfaceError> {
        // One canonical choice sequence for ordinary and compound commits.
        self.apply_legacy_command_inner(command, &mut None)
            .map(|(done, _)| done)
    }

    fn lower_attacker_sequence(
        &mut self,
        actions: &[Action],
        commands: &mut Option<Vec<crate::decision::Command>>,
    ) -> Result<(bool, usize), DecisionSurfaceError> {
        let player = actions
            .first()
            .map(Action::player)
            .ok_or(DecisionSurfaceError::WrongDecision)?;
        let mut order = Vec::with_capacity(actions.len());
        let mut planned = Vec::new();
        for action in actions {
            match *action {
                Action::DeclareAttacker {
                    player: action_player,
                    permanent,
                    attack,
                } if action_player == player => {
                    order.push(permanent);
                    if let Some(defender) = attack {
                        planned.push(crate::flow::decision_group::PlannedAttackEntry {
                            attacker: permanent,
                            defender,
                            raw_defender: None,
                            committed: false,
                        });
                    }
                }
                _ => return Err(DecisionSurfaceError::WrongDecision),
            }
        }
        let mut normalizer = crate::flow::decision_group::AttackPlanNormalizer::new(order, planned);
        let mut done = false;
        let mut executed = 0;
        while let Some(step) = normalizer.next_step(self, player) {
            let index = step
                .action_index
                .ok_or(DecisionSurfaceError::WrongDecision)?;
            done = self.record_compound_step(index, commands)?;
            executed += 1;
        }
        if executed == 0
            || (!self.skip_trivial && !normalizer.all_asked())
            || !normalizer.all_committed()
        {
            return Err(DecisionSurfaceError::WrongDecision);
        }
        Ok((done, executed))
    }

    fn current_blocker_order(
        &self,
        player: PlayerId,
    ) -> Result<Vec<PermanentId>, DecisionSurfaceError> {
        let first = self
            .action_space()
            .filter(|space| space.kind == ActionSpaceKind::DeclareBlocker)
            .and_then(|space| {
                space.actions.iter().find_map(|action| match action {
                    Action::DeclareBlocker {
                        player: action_player,
                        blocker,
                        ..
                    } if *action_player == player => Some(*blocker),
                    _ => None,
                })
            })
            .ok_or(DecisionSurfaceError::WrongDecision)?;
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or(DecisionSurfaceError::WrongDecision)?;
        Ok(std::iter::once(first)
            .chain(combat.blockers_to_declare.iter().rev().copied())
            .collect())
    }

    fn lower_blocker_sequence(
        &mut self,
        actions: &[Action],
        commands: &mut Option<Vec<crate::decision::Command>>,
    ) -> Result<(bool, usize), DecisionSurfaceError> {
        let player = actions
            .first()
            .map(Action::player)
            .ok_or(DecisionSurfaceError::WrongDecision)?;
        let order = self.current_blocker_order(player)?;
        let mut planned = Vec::new();
        for action in actions {
            match *action {
                Action::DeclareBlocker {
                    player: action_player,
                    blocker,
                    attacker,
                } if action_player == player => {
                    if attacker.is_some() {
                        planned.push(crate::flow::decision_group::PlannedBlockEntry {
                            blocker,
                            attacker,
                            raw_attacker: None,
                            committed: false,
                        });
                    }
                }
                _ => return Err(DecisionSurfaceError::WrongDecision),
            }
        }
        let mut normalizer = crate::flow::decision_group::BlockPlanNormalizer::new(order, planned);
        let mut done = false;
        let mut executed = 0;
        while let Some(step) = normalizer.next_step(self, player) {
            let index = step
                .action_index
                .ok_or(DecisionSurfaceError::WrongDecision)?;
            done = self.record_compound_step(index, commands)?;
            executed += 1;
        }
        if executed == 0
            || (!self.skip_trivial && !normalizer.all_asked())
            || !normalizer.all_committed()
        {
            return Err(DecisionSurfaceError::WrongDecision);
        }
        Ok((done, executed))
    }

    fn lower_waterbend_sequence(
        &mut self,
        actions: &[Action],
        commands: &mut Option<Vec<crate::decision::Command>>,
    ) -> Result<(bool, usize), DecisionSurfaceError> {
        let player = actions
            .first()
            .map(Action::player)
            .ok_or(DecisionSurfaceError::WrongDecision)?;
        let mut remaining_taps = std::collections::VecDeque::new();
        let mut want_pay_cost = false;
        for action in actions {
            match *action {
                Action::WaterbendTap { permanent, .. } => remaining_taps.push_back(permanent),
                Action::PayCost { .. } => want_pay_cost = true,
                _ => return Err(DecisionSurfaceError::WrongDecision),
            }
        }
        let is_waterbending = |game: &Game| {
            game.paying()
                .is_some_and(|(_, payment, _)| payment.waterbends())
        };
        let mut done = false;
        let mut executed = 0;
        while let Some(permanent) = remaining_taps.pop_front() {
            // tick() with skip_trivial may have consumed forced tap suffix
            // factors (either this permanent or an interchangeable copy).
            if self.skip_trivial {
                if !is_waterbending(self) {
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
                .ok_or(DecisionSurfaceError::WrongDecision)?;
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
                let auto_taps =
                    usize::from(before.saturating_sub(1).saturating_sub(payment.mana().1));
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
        if want_pay_cost && (!self.skip_trivial || is_waterbending(self)) {
            let wanted = Action::PayCost { player };
            let index = self
                .action_space()
                .filter(|space| space.kind == ActionSpaceKind::Waterbend)
                .and_then(|space| self.resolve_action_index(space, &wanted))
                .ok_or(DecisionSurfaceError::WrongDecision)?;
            done = self.record_compound_step(index, commands)?;
            executed += 1;
        }
        Ok((done, executed))
    }

    /// Whether a creature still to be declared in this combat has a dependent
    /// attack target restriction, must attack, or is forced to decline because
    /// an interchangeable creature declined earlier.
    fn attacker_declaration_is_constrained(&self) -> bool {
        if self.attack_targets(self.active_player()).any(|t| match t {
            crate::flow::combat::AttackTarget::Permanent(object) => {
                self.state.permanent_of(object).is_some_and(|id| {
                    self.has_restriction(
                        id,
                        crate::state::card::Restriction::CantBeAttackedByMoreThanOne,
                    )
                })
            }
            _ => false,
        }) {
            return true;
        }
        let current = self
            .action_space()
            .into_iter()
            .flat_map(|space| &space.actions)
            .find_map(|action| match action {
                Action::DeclareAttacker { permanent, .. } => Some(*permanent),
                _ => None,
            });
        let pending = self
            .state
            .combat
            .as_ref()
            .into_iter()
            .flat_map(|combat| combat.attackers_to_declare.iter().copied());
        if current
            .into_iter()
            .chain(pending.clone())
            .any(|attacker| self.must_attack(attacker))
        {
            return true;
        }
        let Some(combat) = self.state.combat.as_deref() else {
            return false;
        };
        if combat.declined_attackers.is_empty() {
            return false;
        }
        let candidates: Vec<_> = current.into_iter().chain(pending).collect();
        self.any_interchangeable_permanent(&combat.declined_attackers, &candidates)
    }

    /// Recover the complete, still-undecided attacker batch from the current
    /// sequential legacy prompt. The first permanent was popped while the
    /// prompt was built; the remainder will be popped from the back.
    fn current_attacker_declaration(
        &self,
    ) -> Result<(PlayerId, Vec<PermanentId>), DecisionSurfaceError> {
        let action_space = self
            .action_space()
            .ok_or(DecisionSurfaceError::NoActiveActionSpace)?;
        if action_space.kind != ActionSpaceKind::DeclareAttacker {
            return Err(DecisionSurfaceError::WrongDecision);
        }
        let subject = action_space
            .actions
            .first()
            .map(|action| action.player())
            .ok_or(DecisionSurfaceError::MissingActor)?;

        let mut current = None;
        let mut attacks = BTreeSet::new();
        for action in &action_space.actions {
            let Action::DeclareAttacker {
                player,
                permanent,
                attack,
            } = action
            else {
                return Err(DecisionSurfaceError::Invariant(
                    "declare-attacker prompt contains another action kind".to_string(),
                ));
            };
            if *player != subject {
                return Err(DecisionSurfaceError::Invariant(
                    "declare-attacker prompt contains another rules player".to_string(),
                ));
            }
            if current
                .replace(*permanent)
                .is_some_and(|seen| seen != *permanent)
            {
                return Err(DecisionSurfaceError::Invariant(
                    "declare-attacker prompt names multiple permanents".to_string(),
                ));
            }
            attacks.insert(attack.is_some());
        }
        let current = current.ok_or_else(|| {
            DecisionSurfaceError::Invariant("declare-attacker prompt is empty".to_string())
        })?;
        if attacks != BTreeSet::from([false, true]) {
            return Err(DecisionSurfaceError::Invariant(
                "declare-attacker prompt must offer attack and decline".to_string(),
            ));
        }

        let combat = self.state.combat.as_ref().ok_or_else(|| {
            DecisionSurfaceError::Invariant(
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
            return Err(DecisionSurfaceError::Invariant(
                "declare-attacker prompt repeats a permanent".to_string(),
            ));
        }
        Ok((subject, declaration_order))
    }

    fn bind_target(&self, target: Target) -> Result<BoundTarget, DecisionSurfaceError> {
        match target {
            Target::Player(player) => Ok(BoundTarget::Player(player)),
            Target::Permanent(permanent) => self
                .permanent_object(permanent)
                .map(BoundTarget::Permanent)
                .ok_or(DecisionSurfaceError::InvalidCurrentTarget),
            Target::StackSpell(object_id) => Ok(BoundTarget::StackSpell(object_id)),
            Target::StackAbility(id) => Ok(BoundTarget::StackAbility(id)),
            Target::GraveyardCard(object) => Ok(BoundTarget::GraveyardCard(object)),
        }
    }

    fn resolve_bound_target(&self, target: BoundTarget) -> Result<Target, DecisionSurfaceError> {
        match target {
            BoundTarget::Player(player) => Ok(Target::Player(player)),
            BoundTarget::Permanent(object_ref) => self
                .lookup_current_permanent(object_ref)
                .map(Target::Permanent)
                .map_err(|_| {
                    DecisionSurfaceError::StaleOrIllegal(format!(
                        "target object {}@{} is no longer current",
                        object_ref.slot(),
                        object_ref.generation()
                    ))
                }),
            BoundTarget::StackSpell(object_id) => Ok(Target::StackSpell(object_id)),
            BoundTarget::StackAbility(id) => self
                .stack_index_of(Target::StackAbility(id))
                .map(|_| Target::StackAbility(id))
                .ok_or(DecisionSurfaceError::InvalidCurrentTarget),
            BoundTarget::GraveyardCard(object) => {
                if self.state.live(object).is_some()
                    && self.state.zones.zone_of(object) == Some(ZoneType::Graveyard)
                {
                    Ok(Target::GraveyardCard(object))
                } else {
                    Err(DecisionSurfaceError::InvalidCurrentTarget)
                }
            }
        }
    }

    fn subject_and_label(
        &self,
        target: BoundTarget,
    ) -> Result<(SubjectRef, String), DecisionSurfaceError> {
        match target {
            BoundTarget::Player(player) => Ok((
                SubjectRef::Player {
                    id: wire_player_id(player)?,
                },
                self.state.players[player.0].name.clone(),
            )),
            BoundTarget::Permanent(object_ref) => {
                self.lookup_current_permanent(object_ref)
                    .map_err(|_| DecisionSurfaceError::InvalidCurrentTarget)?;
                Ok((
                    SubjectRef::Object {
                        id: self.object_render_ref(object_ref)?,
                    },
                    self.observed_object_name(object_ref),
                ))
            }
            BoundTarget::GraveyardCard(object) => {
                if self.state.live(object).is_none() {
                    return Err(DecisionSurfaceError::InvalidCurrentTarget);
                }
                Ok((
                    SubjectRef::Object {
                        id: object_render_id(self.state[object].id, object.generation()),
                    },
                    self.state[object].name.clone(),
                ))
            }
            BoundTarget::StackSpell(_) | BoundTarget::StackAbility(_) => {
                Err(DecisionSurfaceError::InvalidCurrentTarget)
            }
        }
    }
}

fn next_role_id(index: usize) -> Result<RoleId, DecisionSurfaceError> {
    u16::try_from(index)
        .map(RoleId)
        .map_err(|_| DecisionSurfaceError::IdentityOverflow)
}

fn select_step(
    role: RoleId,
    source_index: usize,
    label: String,
    candidates: Vec<Candidate>,
    min: usize,
    max: usize,
) -> Result<ChoiceStep, DecisionSurfaceError> {
    Ok(ChoiceStep::Select {
        role,
        label,
        candidates: CandidateSource {
            id: CandidateSourceId(
                u32::try_from(source_index).map_err(|_| DecisionSurfaceError::IdentityOverflow)?,
            ),
            depends_on: Vec::new(),
            initial: Some(candidates),
        },
        min: u16::try_from(min).map_err(|_| DecisionSurfaceError::IdentityOverflow)?,
        max: u16::try_from(max).map_err(|_| DecisionSurfaceError::IdentityOverflow)?,
        ordered: false,
        distinct: true,
    })
}

fn class_index(classes: &[Vec<PermanentId>]) -> BTreeMap<PermanentId, usize> {
    let mut map = BTreeMap::new();
    for (class_idx, class) in classes.iter().enumerate() {
        for &perm in class {
            map.insert(perm, class_idx);
        }
    }
    map
}

fn collect_role_answers(
    answers: &[ChoiceAnswer],
) -> Result<BTreeMap<RoleId, &[CandidateId]>, DecisionSurfaceError> {
    let mut map = BTreeMap::new();
    for ChoiceAnswer::Candidates { role, candidates } in answers {
        if map.insert(*role, candidates.as_slice()).is_some() {
            return Err(DecisionSurfaceError::DuplicateRole(*role));
        }
    }
    Ok(map)
}

fn candidate_answer(
    answers: &[ChoiceAnswer],
    role: RoleId,
) -> Result<&[CandidateId], DecisionSurfaceError> {
    if answers.is_empty() {
        return Err(DecisionSurfaceError::MissingAnswer(role));
    }

    let mut seen_roles = BTreeSet::new();
    for answer in answers {
        if !seen_roles.insert(answer.role()) {
            return Err(DecisionSurfaceError::DuplicateRole(answer.role()));
        }
    }
    if answers.len() != 1 {
        return Err(DecisionSurfaceError::UnexpectedAnswers);
    }

    let ChoiceAnswer::Candidates {
        role: answer_role,
        candidates,
    } = &answers[0];
    if *answer_role != role {
        return Err(DecisionSurfaceError::UnexpectedRole(*answer_role));
    }
    Ok(candidates)
}

fn object_render_id(id: PublicId, incarnation: u32) -> ObjectRenderId {
    ObjectRenderId {
        entity: id.0,
        incarnation,
    }
}

fn wire_player_id(player: PlayerId) -> Result<u8, DecisionSurfaceError> {
    u8::try_from(player.0).map_err(|_| DecisionSurfaceError::IdentityOverflow)
}

fn next_offer_id(len: usize) -> Result<DecisionOptionId, DecisionSurfaceError> {
    u32::try_from(len)
        .map(DecisionOptionId)
        .map_err(|_| DecisionSurfaceError::IdentityOverflow)
}

fn next_candidate_id(len: usize) -> Result<CandidateId, DecisionSurfaceError> {
    u32::try_from(len)
        .map(CandidateId)
        .map_err(|_| DecisionSurfaceError::IdentityOverflow)
}

fn search_prompt_kind(kind: ActionSpaceKind) -> Result<PromptKind, DecisionSurfaceError> {
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
        ActionSpaceKind::GameOver => Err(DecisionSurfaceError::GameOver),
    }
}

fn search_offer_verb(action: &Action) -> DecisionVerb {
    match action {
        Action::PlayLand { .. } => DecisionVerb::PlayLand,
        Action::CastSpell { .. } => DecisionVerb::Cast,
        Action::ActivateAbility { .. }
        | Action::ActivateCardAbility { .. }
        | Action::TurnFaceUp { .. }
        | Action::Foretell { .. } => DecisionVerb::ActivateAbility,
        Action::PassPriority { .. } => DecisionVerb::PassPriority,
        Action::DeclareAttacker { .. } => DecisionVerb::DeclareAttacker,
        Action::DeclareBlocker { .. } => DecisionVerb::DeclareBlocker,
        Action::ChooseTarget { .. } => DecisionVerb::ChooseTarget,
        Action::ScryCard {
            to_bottom: false, ..
        } => DecisionVerb::ScryKeep,
        Action::ScryCard {
            to_bottom: true, ..
        } => DecisionVerb::ScryBottom,
        Action::LearnTakeLesson { .. } => DecisionVerb::LearnTakeLesson,
        Action::LearnDiscard { .. } => DecisionVerb::LearnDiscard,
        Action::SelectCard { .. } => DecisionVerb::SelectCard,
        Action::Decline { .. } => DecisionVerb::Decline,
        Action::PayCost { .. } => DecisionVerb::PayCost,
        Action::ChooseMode { .. } => DecisionVerb::ChooseMode,
        Action::ChooseTrigger { .. } => DecisionVerb::ChooseTrigger,
        Action::WaterbendTap { .. } => DecisionVerb::WaterbendTap,
        Action::TapForMana { .. } => DecisionVerb::TapForMana,
    }
}
