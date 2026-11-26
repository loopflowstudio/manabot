//! Unified interactive decision API over sequential native `Action`s.
//!
//! Multi-part decisions (combat declarations, Scry, Surveil, OrderLibrary,
//! PartitionPiles) stay flat and sequential in the core engine (`Game::step`)
//! so RL policies, search, observations and replay digests never change. This
//! module reads the current decision (`Game::decision`, returning either
//! `Single` or a grouped variant above its domain threshold), previews a single
//! or grouped choice on a clone (`Game::preview`), and applies a valid choice
//! atomically (`Game::submit`) by stepping those exact sequential actions.
//!
//! Every grouped kind shares one clone-and-step driver and one inter-step
//! purity boundary: between step `i` and step `i+1` (while the same group
//! continues), the clone must not draw or reveal cards, advance the RNG, or
//! expose any hidden-zone object that was not already visible to the viewer at
//! step 0.
use std::collections::BTreeSet;

use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::agent::action::{Action, ActionSpaceKind, AgentError};
use crate::agent::structured_offer::OfferVerb;
use crate::cards::content_pack::default_content_pack;
use crate::experience::{ObjectRenderId, OfferSubjectView, SubjectRef};
use crate::flow::decision::Decision;
use crate::flow::event::GameEvent;
use crate::flow::game::Game;
use crate::state::ability::LibraryEnd as StateLibraryEnd;
use crate::state::game_object::{ObjectId, PlayerId, Target};
use crate::state::zone::ZoneType;

mod combat;
mod library;

pub(crate) use combat::CombatGroup;
use library::{OrderLibraryGroup, PartitionPilesGroup, ScryGroup, SurveilGroup};

/// Which end of a library receives ordered cards (`snake_case` wire spelling).
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LibraryEnd {
    Top,
    Bottom,
}

impl From<StateLibraryEnd> for LibraryEnd {
    fn from(end: StateLibraryEnd) -> Self {
        match end {
            StateLibraryEnd::Top => Self::Top,
            StateLibraryEnd::Bottom => Self::Bottom,
        }
    }
}

/// One proposed block: `blocker` blocks `attacker`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedBlock {
    pub blocker: ObjectRenderId,
    pub attacker: ObjectRenderId,
}

/// A card item in a library arrangement group (remaining or already committed).
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GroupCard {
    pub item: ObjectRenderId,
    pub label: String,
}

/// A declarable attacker in the engine's asking order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AttackerItem {
    pub item: ObjectRenderId,
    pub label: String,
    pub must_attack: bool,
}

/// A declarable blocker in the engine's asking order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BlockerItem {
    pub item: ObjectRenderId,
    pub label: String,
    pub can_block: Vec<ObjectRenderId>,
    pub must_block: bool,
}

/// Optimistic-concurrency address binding a decision to `(viewer, revision, fingerprint)`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionAddress {
    pub viewer: usize,
    pub revision: u64,
    pub fingerprint: String,
}

/// One raw single-step legal offer available at the current decision.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecisionOffer {
    pub id: usize,
    pub verb: OfferVerb,
    pub label: String,
    pub source: Option<SubjectRef>,
}

/// The active decision a viewer faces now, carrying its bound address, raw
/// single-step legal offers, and either `Single` or a grouped descriptor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ActiveDecision {
    pub address: DecisionAddress,
    pub offers: Vec<DecisionOffer>,
    #[serde(flatten)]
    pub spec: DecisionSpec,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionSpec {
    Single,
    DeclareAttackers {
        declarable: Vec<AttackerItem>,
        committed: Vec<ObjectRenderId>,
    },
    DeclareBlockers {
        declarable: Vec<BlockerItem>,
        committed: Vec<PlannedBlock>,
    },
    Scry {
        items: Vec<GroupCard>,
        committed_top: Vec<GroupCard>,
        committed_bottom: Vec<GroupCard>,
    },
    Surveil {
        items: Vec<GroupCard>,
        committed_graveyard: Vec<GroupCard>,
    },
    OrderLibrary {
        end: LibraryEnd,
        items: Vec<GroupCard>,
        committed: Vec<GroupCard>,
    },
    PartitionPiles {
        items: Vec<GroupCard>,
        committed_face_up: Vec<GroupCard>,
        committed_face_down: Vec<GroupCard>,
    },
}

impl DecisionSpec {
    fn matches_choice(&self, choice: &DecisionChoice) -> bool {
        matches!(
            (self, choice),
            (Self::Single, DecisionChoice::Single { .. })
                | (
                    Self::DeclareAttackers { .. },
                    DecisionChoice::DeclareAttackers { .. }
                )
                | (
                    Self::DeclareBlockers { .. },
                    DecisionChoice::DeclareBlockers { .. }
                )
                | (Self::Scry { .. }, DecisionChoice::Scry { .. })
                | (Self::Surveil { .. }, DecisionChoice::Surveil { .. })
                | (
                    Self::OrderLibrary { .. },
                    DecisionChoice::OrderLibrary { .. }
                )
                | (
                    Self::PartitionPiles { .. },
                    DecisionChoice::PartitionPiles { .. }
                )
        )
    }
}

/// Either a single-step offer id or a complete grouped choice.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionChoice {
    Single {
        offer_id: usize,
    },
    DeclareAttackers {
        #[serde(default)]
        attackers: Vec<ObjectRenderId>,
    },
    DeclareBlockers {
        #[serde(default)]
        blocks: Vec<PlannedBlock>,
    },
    /// `top` is drawn-first at index 0. `bottom` is top-to-bottom of the bottom
    /// group (`bottom[0]` drawn first among bottomed cards; `bottom.last()` on
    /// the very bottom), matching `OrderLibrary { end: Bottom }`.
    Scry {
        #[serde(default)]
        top: Vec<ObjectRenderId>,
        #[serde(default)]
        bottom: Vec<ObjectRenderId>,
    },
    /// Cards to put into the graveyard; all remaining looked-at cards stay on top.
    Surveil {
        #[serde(default)]
        graveyard: Vec<ObjectRenderId>,
    },
    /// Remaining cards in desired order, drawn-first at index 0.
    OrderLibrary {
        #[serde(default)]
        order: Vec<ObjectRenderId>,
    },
    /// Cards to put in the face-up pile; all remaining looked-at cards go face-down.
    PartitionPiles {
        #[serde(default)]
        face_up: Vec<ObjectRenderId>,
    },
}

/// One sequential native step compiled from a decision choice.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecisionStep {
    pub revision: u64,
    pub fingerprint: String,
    pub action_index: usize,
    /// The object answered at this step, or `None` for a single offer or
    /// group-ending `Decline` (Surveil).
    pub item: Option<ObjectRenderId>,
}

/// Why the engine refused a choice.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecisionRefusal {
    pub item: Option<ObjectRenderId>,
    pub label: String,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionOutcome {
    Single {
        offer_id: usize,
    },
    DeclareAttackers {
        attackers: Vec<ObjectRenderId>,
    },
    DeclareBlockers {
        blocks: Vec<PlannedBlock>,
    },
    Scry {
        top: Vec<ObjectRenderId>,
        bottom: Vec<ObjectRenderId>,
    },
    Surveil {
        graveyard: Vec<ObjectRenderId>,
        kept: Vec<ObjectRenderId>,
    },
    OrderLibrary {
        end: LibraryEnd,
        order: Vec<ObjectRenderId>,
    },
    PartitionPiles {
        face_up: Vec<ObjectRenderId>,
        face_down: Vec<ObjectRenderId>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecisionPreview {
    pub decision: ActiveDecision,
    pub steps: Vec<DecisionStep>,
    pub refused: Option<DecisionRefusal>,
    pub complete: bool,
    pub outcome: Option<DecisionOutcome>,
}

impl DecisionPreview {
    fn incomplete(decision: ActiveDecision, steps: Vec<DecisionStep>) -> Self {
        Self {
            decision,
            steps,
            refused: None,
            complete: false,
            outcome: None,
        }
    }

    fn refused(
        decision: ActiveDecision,
        steps: Vec<DecisionStep>,
        refusal: DecisionRefusal,
    ) -> Self {
        Self {
            decision,
            steps,
            refused: Some(refusal),
            complete: false,
            outcome: None,
        }
    }

    pub(crate) fn require_submittable(&self) -> Result<(), AgentError> {
        if let Some(refusal) = &self.refused {
            return Err(AgentError(format!("decision refused: {}", refusal.reason)));
        }
        if !self.complete || self.steps.is_empty() {
            return Err(AgentError("decision choice is incomplete".into()));
        }
        Ok(())
    }
}

pub(crate) enum ChoiceCheck {
    Ready,
    Incomplete,
    Refused(DecisionRefusal),
}

pub(crate) enum NextStep {
    Step {
        action_index: usize,
        item: Option<ObjectRenderId>,
    },
    Done,
    Refuse(DecisionRefusal),
}

pub(crate) trait GroupKind {
    type Progress;

    fn describe(&self, game: &Game, viewer: PlayerId) -> Option<DecisionSpec>;
    fn validate(&self, game: &Game, spec: &DecisionSpec, choice: &DecisionChoice) -> ChoiceCheck;
    fn start(
        &self,
        game: &Game,
        viewer: PlayerId,
        choice: &DecisionChoice,
    ) -> Result<Self::Progress, AgentError>;
    fn continues(
        &self,
        position: &Game,
        viewer: PlayerId,
        spec: &DecisionSpec,
        progress: &Self::Progress,
    ) -> bool;
    fn next_step(
        &self,
        live: &Game,
        position: &Game,
        viewer: PlayerId,
        choice: &DecisionChoice,
        progress: &mut Self::Progress,
    ) -> NextStep;
    fn finish(
        &self,
        live: &Game,
        position: &Game,
        obs_events_from: usize,
        spec: &DecisionSpec,
        choice: &DecisionChoice,
        progress: &Self::Progress,
    ) -> (DecisionOutcome, Option<DecisionRefusal>);
}

fn active_spec(game: &Game, viewer: PlayerId) -> DecisionSpec {
    if game.is_game_over() {
        return DecisionSpec::Single;
    }
    let Some(space) = game.action_space() else {
        return DecisionSpec::Single;
    };
    if space.player != Some(viewer) {
        return DecisionSpec::Single;
    }
    let spec = if let Some(suspended) = &game.state.suspended_decision {
        match &suspended.decision {
            Decision::Scry { player, .. } if *player == viewer => ScryGroup.describe(game, viewer),
            Decision::Surveil { player, .. } if *player == viewer => {
                SurveilGroup.describe(game, viewer)
            }
            Decision::OrderLibrary { player, .. } if *player == viewer => {
                OrderLibraryGroup.describe(game, viewer)
            }
            Decision::PartitionPiles { player, .. } if *player == viewer => {
                PartitionPilesGroup.describe(game, viewer)
            }
            _ => None,
        }
    } else {
        match space.kind {
            ActionSpaceKind::DeclareAttacker => CombatGroup::attackers().describe(game, viewer),
            ActionSpaceKind::DeclareBlocker => CombatGroup::blockers().describe(game, viewer),
            _ => None,
        }
    };
    spec.unwrap_or(DecisionSpec::Single)
}

/// Every hidden-zone object `viewer` can currently see: hands of seats sharing
/// information with `viewer`, plus `inspected_library_objects` across the
/// suspended decision continuation.
fn visible_hidden_objects(game: &Game, viewer: PlayerId) -> BTreeSet<ObjectId> {
    let mut visible = BTreeSet::new();
    for seat in [PlayerId(0), PlayerId(1)] {
        if game.shares_information(viewer, seat) {
            for card in game.state.zone_cards(ZoneType::Hand, seat) {
                visible.insert(game.state.object_of(card));
            }
        }
    }
    if let Some(suspended) = &game.state.suspended_decision {
        for (player, objects) in suspended
            .decisions()
            .filter_map(Decision::inspected_library_objects)
        {
            if game.shares_information(viewer, player) {
                visible.extend(objects.iter().copied());
            }
        }
    }
    visible
}

/// True when stepping from step `i` into step `i+1` crossed a hidden-info
/// boundary (draw, reveal, RNG advance/shuffle, or newly visible hidden object).
fn crossed_hidden_boundary(
    position: &Game,
    viewer: PlayerId,
    events_before: usize,
    rng_before: &ChaCha8Rng,
    visible_at_start: &BTreeSet<ObjectId>,
) -> bool {
    if position.state.rng != *rng_before {
        return true;
    }
    if (events_before..position.state.events.len()).any(|index| {
        matches!(
            position.state.events.get(index),
            Some(GameEvent::CardDrawn { .. } | GameEvent::CardRevealed { .. })
        )
    }) {
        return true;
    }
    let visible_now = visible_hidden_objects(position, viewer);
    visible_now.iter().any(|id| !visible_at_start.contains(id))
}

fn drive_preview<K: GroupKind>(
    live: &Game,
    kind: &K,
    viewer: PlayerId,
    decision: ActiveDecision,
    choice: &DecisionChoice,
) -> Result<DecisionPreview, AgentError> {
    if !decision.spec.matches_choice(choice) {
        return Err(AgentError(
            "choice kind does not match active decision".into(),
        ));
    }
    match kind.validate(live, &decision.spec, choice) {
        ChoiceCheck::Incomplete => return Ok(DecisionPreview::incomplete(decision, Vec::new())),
        ChoiceCheck::Refused(refusal) => {
            return Ok(DecisionPreview::refused(decision, Vec::new(), refusal));
        }
        ChoiceCheck::Ready => {}
    }

    let mut progress = kind.start(live, viewer, choice)?;
    let mut position = live.clone();
    let obs_events_from = position.state.observation_events.len();
    let visible_at_start = visible_hidden_objects(&position, viewer);
    let mut steps = Vec::new();

    while !position.is_game_over() && kind.continues(&position, viewer, &decision.spec, &progress) {
        match kind.next_step(live, &position, viewer, choice, &mut progress) {
            NextStep::Done => break,
            NextStep::Refuse(refusal) => {
                return Ok(DecisionPreview::refused(decision, steps, refusal));
            }
            NextStep::Step { action_index, item } => {
                let frame = position
                    .semantic_decision_frame()
                    .map_err(|error| AgentError(error.to_string()))?;
                let events_before = position.state.events.len();
                let rng_before = position.state.rng.clone();
                steps.push(DecisionStep {
                    revision: frame.revision,
                    fingerprint: frame.fingerprint,
                    action_index,
                    item,
                });
                position.step(action_index)?;
                if !position.is_game_over()
                    && kind.continues(&position, viewer, &decision.spec, &progress)
                    && crossed_hidden_boundary(
                        &position,
                        viewer,
                        events_before,
                        &rng_before,
                        &visible_at_start,
                    )
                {
                    return Ok(DecisionPreview::incomplete(decision, steps));
                }
            }
        }
    }

    let (outcome, refused) = kind.finish(
        live,
        &position,
        obs_events_from,
        &decision.spec,
        choice,
        &progress,
    );
    let complete = refused.is_none();
    Ok(DecisionPreview {
        decision,
        steps,
        refused,
        complete,
        outcome: Some(outcome),
    })
}

fn primary_subject(subjects: &[OfferSubjectView]) -> Option<SubjectRef> {
    subjects
        .iter()
        .find(|view| view.role == "source")
        .or_else(|| subjects.first())
        .map(|view| view.object.subject)
}

fn format_offer_label(
    game: &Game,
    space_kind: ActionSpaceKind,
    action: &Action,
    subjects: &[OfferSubjectView],
) -> String {
    let spell_name = |object: ObjectId| -> String {
        default_content_pack()
            .spell_definition(&game.state[object], action.spell_profile())
            .map(|def| def.name.clone())
            .unwrap_or_else(|| game.observed_object_name(object))
    };
    let perm_name = |permanent| -> String {
        game.permanent_object(permanent)
            .map(|object| game.observed_object_name(object))
            .unwrap_or_else(|| "permanent".to_string())
    };
    match action {
        Action::PlayLand { object, .. } => format!("Play {}", spell_name(*object)),
        Action::CastSpell { object, .. } => format!("Cast {}", spell_name(*object)),
        Action::PassPriority { .. } => "Pass priority".to_string(),
        Action::DeclareAttacker {
            permanent, attack, ..
        } => {
            let name = perm_name(*permanent);
            if *attack {
                format!("Attack with {name}")
            } else {
                format!("Don't attack with {name}")
            }
        }
        Action::DeclareBlocker {
            blocker, attacker, ..
        } => {
            let blocker_name = perm_name(*blocker);
            match attacker {
                Some(target) => format!("Block {} with {blocker_name}", perm_name(*target)),
                None => format!("Don't block with {blocker_name}"),
            }
        }
        Action::ChooseTarget { target, .. } => {
            if let Some(subject) = subjects.first() {
                format!("Target {}", subject.object.label)
            } else {
                match target {
                    Target::Player(player) => format!("Target Seat {}", player.0 + 1),
                    Target::Permanent(permanent) => format!("Target {}", perm_name(*permanent)),
                    Target::StackSpell(object) | Target::GraveyardCard(object) => {
                        format!("Target {}", game.observed_object_name(*object))
                    }
                    Target::StackAbility(id) => {
                        let name = game
                            .stack_ability_source(*id)
                            .map(|object| game.observed_object_name(object))
                            .unwrap_or_else(|| "ability".to_string());
                        format!("Target {name}")
                    }
                }
            }
        }
        Action::ActivateAbility { permanent, .. } => format!("Activate {}", perm_name(*permanent)),
        Action::ActivateCardAbility { object, .. } => {
            format!("Activate {}", game.observed_object_name(*object))
        }
        Action::TurnFaceUp { object, .. } => {
            format!("Turn {} face up", game.observed_object_name(*object))
        }
        Action::Foretell { object, .. } => {
            format!("Foretell {}", game.observed_object_name(*object))
        }
        Action::ScryCard {
            object, to_bottom, ..
        } => {
            let name = game.observed_object_name(*object);
            if *to_bottom {
                format!("Put {name} on bottom")
            } else {
                format!("Keep {name} on top")
            }
        }
        Action::SelectCard { object, .. } => {
            let name = game.observed_object_name(*object);
            match space_kind {
                ActionSpaceKind::Discard => format!("Discard {name}"),
                ActionSpaceKind::LegendRule => format!("Keep {name}"),
                _ => format!("Select {name}"),
            }
        }
        Action::LearnTakeLesson { object, .. } => {
            format!("Take lesson {}", game.state.source_base(*object).1.name)
        }
        Action::LearnDiscard { object, .. } => {
            format!("Discard {} to draw", game.observed_object_name(*object))
        }
        Action::PayCost { .. } => "Pay cost".to_string(),
        Action::Decline { .. } => "Decline".to_string(),
        Action::ChooseMode { mode, .. } => format!("Mode {}", mode + 1),
        Action::WaterbendTap { permanent, .. } | Action::TapForMana { permanent, .. } => {
            format!("Tap {}", perm_name(*permanent))
        }
        Action::ChooseTrigger { .. } => "Choose trigger".to_string(),
    }
}

fn project_offers(
    game: &Game,
    actor: PlayerId,
    frame: &crate::decision::DecisionFrame,
) -> Vec<DecisionOffer> {
    let subjects = game.decision_offer_subjects(actor);
    let space = game.action_space();
    frame
        .offers
        .iter()
        .enumerate()
        .map(|(index, offer)| {
            let offer_subjects = subjects.get(index).map(Vec::as_slice).unwrap_or(&[]);
            let source = primary_subject(offer_subjects);
            let label = space
                .and_then(|s| {
                    s.actions.get(index).map(|action| {
                        game.action_description(action).unwrap_or_else(|| {
                            format_offer_label(game, s.kind, action, offer_subjects)
                        })
                    })
                })
                .unwrap_or_else(|| offer.label.clone());
            DecisionOffer {
                id: offer.id.0 as usize,
                verb: offer.verb,
                label,
                source,
            }
        })
        .collect()
}

impl Game {
    fn build_decision(&self, frame: crate::decision::DecisionFrame) -> ActiveDecision {
        let actor = PlayerId(frame.actor as usize);
        let offers = project_offers(self, actor, &frame);
        let address = DecisionAddress {
            viewer: actor.0,
            revision: frame.revision,
            fingerprint: frame.fingerprint,
        };
        let spec = active_spec(self, actor);
        ActiveDecision {
            address,
            offers,
            spec,
        }
    }

    pub(crate) fn verify_step(&self, step: &DecisionStep) -> Result<(), AgentError> {
        let frame = self
            .semantic_decision_frame()
            .map_err(|error| AgentError(error.to_string()))?;
        if frame.revision != step.revision || frame.fingerprint != step.fingerprint {
            return Err(AgentError(
                "decision execution diverged from preview".into(),
            ));
        }
        Ok(())
    }

    /// Describe the active decision for `viewer` (defaults to the current
    /// acting player). Returns `None` when the game is over, there is no
    /// active chooser, or `viewer` is specified and is not the current actor.
    pub fn decision(&self, viewer: Option<PlayerId>) -> Option<ActiveDecision> {
        if self.is_game_over() {
            return None;
        }
        let frame = self.semantic_decision_frame().ok()?;
        let actor = PlayerId(frame.actor as usize);
        if viewer.is_some_and(|requested| requested != actor) {
            return None;
        }
        Some(self.build_decision(frame))
    }

    /// Read-only preview of `choice`, optionally bound to `at` for optimistic
    /// concurrency. Accepts either a single offer id (`DecisionChoice::Single`)
    /// or a grouped choice matching the active decision.
    pub fn preview(
        &self,
        at: Option<&DecisionAddress>,
        choice: &DecisionChoice,
    ) -> Result<DecisionPreview, AgentError> {
        let frame = self
            .semantic_decision_frame()
            .map_err(|error| AgentError(error.to_string()))?;
        let actor = PlayerId(frame.actor as usize);
        if let Some(expected) = at {
            if expected.viewer != actor.0
                || expected.revision != frame.revision
                || expected.fingerprint != frame.fingerprint
            {
                return Err(AgentError("stale or unauthorized decision".into()));
            }
        }
        let decision = self.build_decision(frame);
        if let &DecisionChoice::Single { offer_id } = choice {
            if offer_id >= decision.offers.len() {
                return Ok(DecisionPreview::refused(
                    decision,
                    Vec::new(),
                    DecisionRefusal {
                        item: None,
                        label: format!("Offer {offer_id}"),
                        reason: format!("Offer {offer_id} is not legal here."),
                    },
                ));
            }
            let step = DecisionStep {
                revision: decision.address.revision,
                fingerprint: decision.address.fingerprint.clone(),
                action_index: offer_id,
                item: None,
            };
            return Ok(DecisionPreview {
                decision,
                steps: vec![step],
                refused: None,
                complete: true,
                outcome: Some(DecisionOutcome::Single { offer_id }),
            });
        }
        match &decision.spec {
            DecisionSpec::Single => Err(AgentError("no active grouped decision".into())),
            DecisionSpec::DeclareAttackers { .. } => {
                drive_preview(self, &CombatGroup::attackers(), actor, decision, choice)
            }
            DecisionSpec::DeclareBlockers { .. } => {
                drive_preview(self, &CombatGroup::blockers(), actor, decision, choice)
            }
            DecisionSpec::Scry { .. } => drive_preview(self, &ScryGroup, actor, decision, choice),
            DecisionSpec::Surveil { .. } => {
                drive_preview(self, &SurveilGroup, actor, decision, choice)
            }
            DecisionSpec::OrderLibrary { .. } => {
                drive_preview(self, &OrderLibraryGroup, actor, decision, choice)
            }
            DecisionSpec::PartitionPiles { .. } => {
                drive_preview(self, &PartitionPilesGroup, actor, decision, choice)
            }
        }
    }

    /// Validate `choice` on a clone, then execute all compiled sequential steps
    /// atomically against the live game. Refused, incomplete, or stale choices
    /// return an error without mutating the live game.
    pub fn submit(
        &mut self,
        at: Option<&DecisionAddress>,
        choice: &DecisionChoice,
    ) -> Result<DecisionPreview, AgentError> {
        let preview = self.preview(at, choice)?;
        preview.require_submittable()?;
        for step in &preview.steps {
            self.verify_step(step)?;
            self.step(step.action_index)?;
        }
        Ok(preview)
    }

    pub(crate) fn render_card_label(&self, id: ObjectRenderId) -> String {
        self.state
            .latest_objects()
            .find(|(object, _)| {
                self.object_event_ref(*object)
                    .is_some_and(|event| ObjectRenderId::from(event) == id)
            })
            .map_or_else(
                || "That card".to_string(),
                |(object, _)| self.observed_object_name(object),
            )
    }
}

#[cfg(test)]
mod tests {
    use rand::RngCore;

    use super::*;
    use crate::state::card::CardDefId;
    use crate::state::player::PlayerConfig;

    fn sample_game() -> Game {
        let mut deck = std::collections::BTreeMap::new();
        deck.insert("Island".to_string(), 20);
        deck.insert("Mountain".to_string(), 20);
        let players = vec![
            PlayerConfig::new("P0", deck.clone()),
            PlayerConfig::new("P1", deck),
        ];
        Game::new(players, 42, true)
    }

    #[test]
    fn purity_boundary_detects_rng_draw_reveal_and_new_hidden_object() {
        let base = sample_game();
        let viewer = PlayerId(0);
        let events_before = base.state.events.len();
        let rng_before = base.state.rng.clone();
        let visible_at_start = visible_hidden_objects(&base, viewer);

        assert!(!crossed_hidden_boundary(
            &base,
            viewer,
            events_before,
            &rng_before,
            &visible_at_start
        ));

        // 1. RNG advance (e.g. shuffle).
        let mut rng_moved = base.clone();
        let _ = rng_moved.state.rng_mut().next_u64();
        assert!(crossed_hidden_boundary(
            &rng_moved,
            viewer,
            events_before,
            &rng_before,
            &visible_at_start
        ));

        // 2. CardDrawn event.
        let mut drew = base.clone();
        drew.state.events.push(GameEvent::CardDrawn {
            player: viewer,
            nth_this_turn: 1,
        });
        assert!(crossed_hidden_boundary(
            &drew,
            viewer,
            events_before,
            &rng_before,
            &visible_at_start
        ));

        // 3. CardRevealed event.
        let mut revealed = base.clone();
        revealed.state.events.push(GameEvent::CardRevealed {
            owner: PlayerId(1),
            definition: CardDefId(0),
        });
        assert!(crossed_hidden_boundary(
            &revealed,
            viewer,
            events_before,
            &rng_before,
            &visible_at_start
        ));

        // 4. Unseen hidden-zone object added to viewer's hand without an event.
        let mut gained_hand = base.clone();
        let unseen = gained_hand
            .state
            .zones
            .zone_objects(ZoneType::Library, viewer)[0];
        assert!(!visible_at_start.contains(&unseen));
        gained_hand
            .state
            .place_object(unseen, viewer, ZoneType::Hand);
        assert!(crossed_hidden_boundary(
            &gained_hand,
            viewer,
            events_before,
            &rng_before,
            &visible_at_start
        ));
    }
}
