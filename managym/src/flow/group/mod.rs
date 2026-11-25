//! First-class grouped-input compiler over sequential native `Action`s.
//!
//! Multi-part decisions (combat declarations, Scry, Surveil, OrderLibrary)
//! stay flat and sequential in the core engine (`Game::step`) so RL policies,
//! search, observations and replay digests never change. This module reads an
//! active multi-part decision (`Game::group`), previews a whole drafted plan
//! on a clone (`Game::preview_group`), and applies a valid plan atomically
//! (`Game::apply_group`) by stepping those exact sequential actions.
//!
//! Every kind shares one clone-and-step driver and one inter-step purity
//! boundary: between step `i` and step `i+1` (while the same group continues),
//! the clone must not draw or reveal cards, advance the RNG, or expose any
//! hidden-zone object that was not already visible to the viewer at step 0.
use std::collections::BTreeSet;

use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::agent::action::{ActionSpaceKind, AgentError};
use crate::experience::ObjectRenderId;
use crate::flow::decision::Decision;
use crate::flow::event::GameEvent;
use crate::flow::game::Game;
use crate::state::ability::LibraryEnd as StateLibraryEnd;
use crate::state::game_object::{ObjectId, PlayerId};
use crate::state::zone::ZoneType;

mod combat;
mod library;

pub(crate) use combat::CombatGroup;
use library::{OrderLibraryGroup, ScryGroup, SurveilGroup};

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

/// A card item in a library arrangement group.
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

/// The grouped decision `viewer` faces now, bound to its native semantic address.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Group {
    pub revision: u64,
    pub fingerprint: String,
    #[serde(flatten)]
    pub spec: GroupSpec,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GroupSpec {
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
        committed_top: Vec<ObjectRenderId>,
        committed_bottom: Vec<ObjectRenderId>,
    },
    Surveil {
        items: Vec<GroupCard>,
        committed_graveyard: Vec<ObjectRenderId>,
    },
    OrderLibrary {
        end: LibraryEnd,
        items: Vec<GroupCard>,
        committed: Vec<ObjectRenderId>,
    },
}

/// The whole plan a caller drafts for the current group.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GroupPlan {
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
}

/// One sequential native step compiled from a grouped plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GroupStep {
    pub revision: u64,
    pub fingerprint: String,
    pub action_index: usize,
    /// The object answered at this step, or `None` for a group-ending `Decline` (Surveil).
    pub item: Option<ObjectRenderId>,
}

/// Why the engine refused a drafted plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GroupRefusal {
    pub item: Option<ObjectRenderId>,
    pub label: String,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GroupOutcome {
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
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GroupPreview {
    pub group: Group,
    pub steps: Vec<GroupStep>,
    pub refused: Option<GroupRefusal>,
    pub complete: bool,
    pub outcome: Option<GroupOutcome>,
}

impl GroupPreview {
    fn incomplete(group: Group, steps: Vec<GroupStep>) -> Self {
        Self {
            group,
            steps,
            refused: None,
            complete: false,
            outcome: None,
        }
    }

    fn refused(group: Group, steps: Vec<GroupStep>, refusal: GroupRefusal) -> Self {
        Self {
            group,
            steps,
            refused: Some(refusal),
            complete: false,
            outcome: None,
        }
    }
}

pub(crate) enum PlanCheck {
    Ready,
    Incomplete,
    Refused(GroupRefusal),
}

pub(crate) enum NextStep {
    Step {
        action_index: usize,
        item: Option<ObjectRenderId>,
    },
    Done,
    Refuse(GroupRefusal),
}

pub(crate) trait GroupKind {
    type Progress;

    fn describe(&self, game: &Game, viewer: PlayerId) -> Option<GroupSpec>;
    fn matches_plan(&self, plan: &GroupPlan) -> bool;
    fn validate(&self, game: &Game, spec: &GroupSpec, plan: &GroupPlan) -> PlanCheck;
    fn start(
        &self,
        game: &Game,
        viewer: PlayerId,
        spec: &GroupSpec,
        plan: &GroupPlan,
    ) -> Result<Self::Progress, AgentError>;
    fn continues(&self, position: &Game, viewer: PlayerId, progress: &Self::Progress) -> bool;
    fn next_step(
        &self,
        live: &Game,
        position: &Game,
        viewer: PlayerId,
        plan: &GroupPlan,
        progress: &mut Self::Progress,
    ) -> NextStep;
    fn finish(
        &self,
        live: &Game,
        position: &Game,
        obs_events_from: usize,
        plan: &GroupPlan,
        progress: &Self::Progress,
    ) -> (GroupOutcome, Option<GroupRefusal>);
}

#[derive(Clone, Copy, Debug)]
enum ActiveKind {
    DeclareAttackers,
    DeclareBlockers,
    Scry,
    Surveil,
    OrderLibrary,
}

impl ActiveKind {
    fn describe(self, game: &Game, viewer: PlayerId) -> Option<GroupSpec> {
        match self {
            Self::DeclareAttackers => CombatGroup::attackers().describe(game, viewer),
            Self::DeclareBlockers => CombatGroup::blockers().describe(game, viewer),
            Self::Scry => ScryGroup.describe(game, viewer),
            Self::Surveil => SurveilGroup.describe(game, viewer),
            Self::OrderLibrary => OrderLibraryGroup.describe(game, viewer),
        }
    }

    fn preview(
        self,
        live: &Game,
        viewer: PlayerId,
        revision: u64,
        fingerprint: String,
        plan: &GroupPlan,
    ) -> Result<GroupPreview, AgentError> {
        match self {
            Self::DeclareAttackers => drive_preview(
                live,
                &CombatGroup::attackers(),
                viewer,
                revision,
                fingerprint,
                plan,
            ),
            Self::DeclareBlockers => drive_preview(
                live,
                &CombatGroup::blockers(),
                viewer,
                revision,
                fingerprint,
                plan,
            ),
            Self::Scry => drive_preview(live, &ScryGroup, viewer, revision, fingerprint, plan),
            Self::Surveil => {
                drive_preview(live, &SurveilGroup, viewer, revision, fingerprint, plan)
            }
            Self::OrderLibrary => {
                drive_preview(live, &OrderLibraryGroup, viewer, revision, fingerprint, plan)
            }
        }
    }
}

fn active_kind(game: &Game, viewer: PlayerId) -> Option<ActiveKind> {
    if game.is_game_over() {
        return None;
    }
    let space = game.action_space()?;
    if space.player != Some(viewer) {
        return None;
    }
    if let Some(suspended) = &game.state.suspended_decision {
        return match &suspended.decision {
            Decision::Scry { player, .. } if *player == viewer => Some(ActiveKind::Scry),
            Decision::Surveil { player, .. } if *player == viewer => Some(ActiveKind::Surveil),
            Decision::OrderLibrary { player, .. } if *player == viewer => {
                Some(ActiveKind::OrderLibrary)
            }
            _ => None,
        };
    }
    match space.kind {
        ActionSpaceKind::DeclareAttacker => Some(ActiveKind::DeclareAttackers),
        ActionSpaceKind::DeclareBlocker => Some(ActiveKind::DeclareBlockers),
        _ => None,
    }
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
    revision: u64,
    fingerprint: String,
    plan: &GroupPlan,
) -> Result<GroupPreview, AgentError> {
    let spec = kind
        .describe(live, viewer)
        .ok_or_else(|| AgentError("no active group".into()))?;
    if !kind.matches_plan(plan) {
        return Err(AgentError("plan kind does not match active group".into()));
    }
    let group = Group {
        revision,
        fingerprint,
        spec,
    };
    match kind.validate(live, &group.spec, plan) {
        PlanCheck::Incomplete => return Ok(GroupPreview::incomplete(group, Vec::new())),
        PlanCheck::Refused(refusal) => {
            return Ok(GroupPreview::refused(group, Vec::new(), refusal));
        }
        PlanCheck::Ready => {}
    }

    let mut progress = kind.start(live, viewer, &group.spec, plan)?;
    let mut position = live.clone();
    let obs_events_from = position.state.observation_events.len();
    let visible_at_start = visible_hidden_objects(&position, viewer);
    let mut steps = Vec::new();

    while !position.is_game_over() && kind.continues(&position, viewer, &progress) {
        match kind.next_step(live, &position, viewer, plan, &mut progress) {
            NextStep::Done => break,
            NextStep::Refuse(refusal) => return Ok(GroupPreview::refused(group, steps, refusal)),
            NextStep::Step { action_index, item } => {
                let frame = position
                    .semantic_decision_frame()
                    .map_err(|error| AgentError(error.to_string()))?;
                let events_before = position.state.events.len();
                let rng_before = position.state.rng.clone();
                steps.push(GroupStep {
                    revision: frame.revision,
                    fingerprint: frame.fingerprint,
                    action_index,
                    item,
                });
                position.step(action_index)?;
                if !position.is_game_over()
                    && kind.continues(&position, viewer, &progress)
                    && crossed_hidden_boundary(
                        &position,
                        viewer,
                        events_before,
                        &rng_before,
                        &visible_at_start,
                    )
                {
                    return Ok(GroupPreview::incomplete(group, steps));
                }
            }
        }
    }

    let (outcome, refused) = kind.finish(live, &position, obs_events_from, plan, &progress);
    let complete = refused.is_none();
    Ok(GroupPreview {
        group,
        steps,
        refused,
        complete,
        outcome: Some(outcome),
    })
}

impl Game {
    /// Describe the grouped decision `viewer` faces now, or `None` when the
    /// game is over, `viewer` is not the chooser, or the pending decision is
    /// not a supported group above its domain activation threshold.
    pub fn group(&self, viewer: PlayerId) -> Option<Group> {
        let spec = active_kind(self, viewer)?.describe(self, viewer)?;
        let frame = self.semantic_decision_frame().ok()?;
        if frame.actor as usize != viewer.0 {
            return None;
        }
        Some(Group {
            revision: frame.revision,
            fingerprint: frame.fingerprint,
            spec,
        })
    }

    /// Read-only preview of `plan` bound to `(viewer, revision, fingerprint)`.
    /// Errors on a stale/non-actor address, no active group, or a plan whose
    /// `kind` does not match the active group. A plan refused by engine rules
    /// returns `refused: Some(...)`, and a structurally incomplete plan returns
    /// `complete: false, refused: None` without cloning.
    pub fn preview_group(
        &self,
        viewer: PlayerId,
        revision: u64,
        fingerprint: &str,
        plan: &GroupPlan,
    ) -> Result<GroupPreview, AgentError> {
        let frame = self
            .semantic_decision_frame()
            .map_err(|error| AgentError(error.to_string()))?;
        if frame.actor as usize != viewer.0
            || frame.revision != revision
            || frame.fingerprint != fingerprint
        {
            return Err(AgentError("stale or unauthorized group preview".into()));
        }
        let active =
            active_kind(self, viewer).ok_or_else(|| AgentError("no active group".into()))?;
        active.preview(self, viewer, frame.revision, frame.fingerprint, plan)
    }

    /// Validate `plan` on a clone, then execute its compiled sequential steps
    /// atomically against the live game. Refused or incomplete plans return an
    /// error without mutating the live game.
    pub fn apply_group(
        &mut self,
        viewer: PlayerId,
        revision: u64,
        fingerprint: &str,
        plan: &GroupPlan,
    ) -> Result<GroupPreview, AgentError> {
        let preview = self.preview_group(viewer, revision, fingerprint, plan)?;
        if let Some(refusal) = &preview.refused {
            return Err(AgentError(format!("group refused: {}", refusal.reason)));
        }
        if !preview.complete {
            return Err(AgentError("group plan is incomplete".into()));
        }
        for step in &preview.steps {
            let frame = self
                .semantic_decision_frame()
                .map_err(|error| AgentError(error.to_string()))?;
            if frame.revision != step.revision || frame.fingerprint != step.fingerprint {
                return Err(AgentError("group execution diverged from preview".into()));
            }
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
