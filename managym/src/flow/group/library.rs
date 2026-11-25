//! Library arrangement kinds (`Scry`, `Surveil`, `OrderLibrary`) under the
//! shared grouped-input driver.
use std::collections::{HashMap, HashSet};

use crate::agent::action::{Action, AgentError};
use crate::experience::ObjectRenderId;
use crate::flow::decision::Decision;
use crate::flow::game::Game;
use crate::flow::group::{
    GroupCard, GroupKind, GroupOutcome, GroupPlan, GroupRefusal, GroupSpec, LibraryEnd, NextStep,
    PlanCheck,
};
use crate::state::game_object::{ObjectId, PlayerId};

fn card_render_id(game: &Game, object: ObjectId) -> ObjectRenderId {
    game.object_event_ref(object)
        .expect("inspected library card has an observed identity")
        .into()
}

fn group_card(game: &Game, object: ObjectId) -> GroupCard {
    GroupCard {
        item: card_render_id(game, object),
        label: game.observed_object_name(object),
    }
}

fn render_to_object_map(game: &Game, objects: &[ObjectId]) -> HashMap<ObjectRenderId, ObjectId> {
    objects
        .iter()
        .copied()
        .map(|object| (card_render_id(game, object), object))
        .collect()
}

fn refuse_card(game: &Game, id: ObjectRenderId, reason: &str) -> PlanCheck {
    let label = game.render_card_label(id);
    PlanCheck::Refused(GroupRefusal {
        item: Some(id),
        reason: format!("{label} {reason}"),
        label,
    })
}

// -----------------------------------------------------------------------------
// Scry
// -----------------------------------------------------------------------------

pub(crate) struct ScryGroup;

pub(crate) struct ScryProgress {
    planned: Vec<(ObjectId, ObjectRenderId, bool)>,
    next_index: usize,
    committed_top: Vec<ObjectRenderId>,
    committed_bottom: Vec<ObjectRenderId>,
}

impl GroupKind for ScryGroup {
    type Progress = ScryProgress;

    fn describe(&self, game: &Game, viewer: PlayerId) -> Option<GroupSpec> {
        let Decision::Scry {
            player,
            remaining,
            top,
            bottom,
            ..
        } = &game.state.suspended_decision.as_ref()?.decision
        else {
            return None;
        };
        if *player != viewer || remaining.len() < 2 {
            return None;
        }
        Some(GroupSpec::Scry {
            items: remaining.iter().map(|&o| group_card(game, o)).collect(),
            committed_top: top.iter().map(|&o| card_render_id(game, o)).collect(),
            // `Decision::Scry` commits `bottom.iter().rev()` to index 0, so
            // `bottom.last()` sits above `bottom[0]`. Expose top-of-bottom first.
            committed_bottom: bottom
                .iter()
                .rev()
                .map(|&o| card_render_id(game, o))
                .collect(),
        })
    }

    fn matches_plan(&self, plan: &GroupPlan) -> bool {
        matches!(plan, GroupPlan::Scry { .. })
    }

    fn validate(&self, game: &Game, spec: &GroupSpec, plan: &GroupPlan) -> PlanCheck {
        let (
            GroupSpec::Scry {
                items,
                committed_top,
                committed_bottom,
            },
            GroupPlan::Scry { top, bottom },
        ) = (spec, plan)
        else {
            unreachable!("matches_plan already verified variant");
        };
        let remaining: HashSet<_> = items.iter().map(|card| card.item).collect();
        let mut seen = HashSet::new();
        for &id in top.iter().chain(bottom.iter()) {
            if !remaining.contains(&id) {
                let reason = if committed_top.contains(&id) || committed_bottom.contains(&id) {
                    "has already been placed."
                } else {
                    "is not in the pending scry."
                };
                return refuse_card(game, id, reason);
            }
            if !seen.insert(id) {
                return refuse_card(game, id, "can only be placed once.");
            }
        }
        if seen.len() < remaining.len() {
            return PlanCheck::Incomplete;
        }
        PlanCheck::Ready
    }

    fn start(
        &self,
        game: &Game,
        _viewer: PlayerId,
        spec: &GroupSpec,
        plan: &GroupPlan,
    ) -> Result<Self::Progress, AgentError> {
        let (
            GroupSpec::Scry {
                committed_top,
                committed_bottom,
                ..
            },
            GroupPlan::Scry { top, bottom },
        ) = (spec, plan)
        else {
            unreachable!("matches_plan already verified variant");
        };
        let Some(Decision::Scry { remaining, .. }) =
            game.state.suspended_decision.as_ref().map(|s| &s.decision)
        else {
            return Err(AgentError("not a pending scry".into()));
        };
        let lookup = render_to_object_map(game, remaining);
        let mut planned = Vec::with_capacity(top.len() + bottom.len());
        for &id in top {
            let object = *lookup
                .get(&id)
                .ok_or_else(|| AgentError("card is not in the pending scry".into()))?;
            planned.push((object, id, false));
        }
        // `Decision::Scry` repositions `bottom.iter().rev()` to index 0, so
        // emitting `plan.bottom` in reverse commits `plan.bottom[0]` above
        // `plan.bottom.last()`, matching `OrderLibrary { end: Bottom }`.
        for &id in bottom.iter().rev() {
            let object = *lookup
                .get(&id)
                .ok_or_else(|| AgentError("card is not in the pending scry".into()))?;
            planned.push((object, id, true));
        }
        Ok(ScryProgress {
            planned,
            next_index: 0,
            committed_top: committed_top.clone(),
            committed_bottom: committed_bottom.clone(),
        })
    }

    fn continues(&self, position: &Game, viewer: PlayerId, progress: &Self::Progress) -> bool {
        progress.next_index < progress.planned.len()
            && matches!(
                position.state.suspended_decision.as_ref().map(|s| &s.decision),
                Some(Decision::Scry { player, .. }) if *player == viewer
            )
    }

    fn next_step(
        &self,
        live: &Game,
        position: &Game,
        _viewer: PlayerId,
        _plan: &GroupPlan,
        progress: &mut Self::Progress,
    ) -> NextStep {
        let Some(&(object, render_id, to_bottom)) = progress.planned.get(progress.next_index)
        else {
            return NextStep::Done;
        };
        progress.next_index += 1;
        let Some(space) = position.action_space() else {
            return NextStep::Done;
        };
        let index = space.actions.iter().position(|action| {
            matches!(
                action,
                Action::ScryCard {
                    object: offered,
                    to_bottom: offered_bottom,
                    ..
                } if *offered == object && *offered_bottom == to_bottom
            )
        });
        match index {
            Some(action_index) => NextStep::Step {
                action_index,
                item: Some(render_id),
            },
            None => {
                let label = live.observed_object_name(object);
                NextStep::Refuse(GroupRefusal {
                    item: Some(render_id),
                    reason: format!("{label} cannot be placed there."),
                    label,
                })
            }
        }
    }

    fn finish(
        &self,
        _live: &Game,
        _position: &Game,
        _obs_events_from: usize,
        plan: &GroupPlan,
        progress: &Self::Progress,
    ) -> (GroupOutcome, Option<GroupRefusal>) {
        let GroupPlan::Scry { top, bottom } = plan else {
            unreachable!("matches_plan already verified variant");
        };
        let mut final_top = progress.committed_top.clone();
        final_top.extend_from_slice(top);
        // Earlier raw bottom picks sit underneath later bottom picks once
        // `Decision::Scry` reverses its bottom vec, so `plan.bottom` sits
        // above `committed_bottom` in top-of-bottom-first reading order.
        let mut final_bottom = bottom.clone();
        final_bottom.extend_from_slice(&progress.committed_bottom);
        (
            GroupOutcome::Scry {
                top: final_top,
                bottom: final_bottom,
            },
            None,
        )
    }
}

// -----------------------------------------------------------------------------
// Surveil
// -----------------------------------------------------------------------------

pub(crate) struct SurveilGroup;

enum SurveilStep {
    Graveyard(ObjectId, ObjectRenderId),
    FinishKeep,
}

pub(crate) struct SurveilProgress {
    planned: Vec<SurveilStep>,
    next_index: usize,
    committed_graveyard: Vec<ObjectRenderId>,
    remaining: Vec<ObjectRenderId>,
}

impl GroupKind for SurveilGroup {
    type Progress = SurveilProgress;

    fn describe(&self, game: &Game, viewer: PlayerId) -> Option<GroupSpec> {
        let Decision::Surveil {
            player,
            looked,
            graveyard,
        } = &game.state.suspended_decision.as_ref()?.decision
        else {
            return None;
        };
        if *player != viewer {
            return None;
        }
        let remaining: Vec<_> = looked
            .iter()
            .copied()
            .filter(|object| !graveyard.contains(object))
            .collect();
        if remaining.len() < 2 {
            return None;
        }
        Some(GroupSpec::Surveil {
            items: remaining.iter().map(|&o| group_card(game, o)).collect(),
            committed_graveyard: graveyard.iter().map(|&o| card_render_id(game, o)).collect(),
        })
    }

    fn matches_plan(&self, plan: &GroupPlan) -> bool {
        matches!(plan, GroupPlan::Surveil { .. })
    }

    fn validate(&self, game: &Game, spec: &GroupSpec, plan: &GroupPlan) -> PlanCheck {
        let (
            GroupSpec::Surveil {
                items,
                committed_graveyard,
            },
            GroupPlan::Surveil { graveyard },
        ) = (spec, plan)
        else {
            unreachable!("matches_plan already verified variant");
        };
        let remaining: HashSet<_> = items.iter().map(|card| card.item).collect();
        let mut seen = HashSet::new();
        for &id in graveyard {
            if !remaining.contains(&id) {
                let reason = if committed_graveyard.contains(&id) {
                    "is already marked for the graveyard."
                } else {
                    "is not in the pending surveil."
                };
                return refuse_card(game, id, reason);
            }
            if !seen.insert(id) {
                return refuse_card(game, id, "can only be chosen once.");
            }
        }
        PlanCheck::Ready
    }

    fn start(
        &self,
        game: &Game,
        _viewer: PlayerId,
        spec: &GroupSpec,
        plan: &GroupPlan,
    ) -> Result<Self::Progress, AgentError> {
        let (
            GroupSpec::Surveil {
                items,
                committed_graveyard,
            },
            GroupPlan::Surveil { graveyard },
        ) = (spec, plan)
        else {
            unreachable!("matches_plan already verified variant");
        };
        let Some(Decision::Surveil {
            looked,
            graveyard: already_gy,
            ..
        }) = game.state.suspended_decision.as_ref().map(|s| &s.decision)
        else {
            return Err(AgentError("not a pending surveil".into()));
        };
        let remaining_objects: Vec<_> = looked
            .iter()
            .copied()
            .filter(|object| !already_gy.contains(object))
            .collect();
        let lookup = render_to_object_map(game, &remaining_objects);
        let mut planned = Vec::with_capacity(graveyard.len() + 1);
        for &id in graveyard {
            let object = *lookup
                .get(&id)
                .ok_or_else(|| AgentError("card is not in the pending surveil".into()))?;
            planned.push(SurveilStep::Graveyard(object, id));
        }
        // When every remaining card goes to the graveyard, the final SelectCard
        // finishes Surveil automatically without an explicit Decline step.
        if graveyard.len() < remaining_objects.len() {
            planned.push(SurveilStep::FinishKeep);
        }
        Ok(SurveilProgress {
            planned,
            next_index: 0,
            committed_graveyard: committed_graveyard.clone(),
            remaining: items.iter().map(|card| card.item).collect(),
        })
    }

    fn continues(&self, position: &Game, viewer: PlayerId, progress: &Self::Progress) -> bool {
        progress.next_index < progress.planned.len()
            && matches!(
                position.state.suspended_decision.as_ref().map(|s| &s.decision),
                Some(Decision::Surveil { player, .. }) if *player == viewer
            )
    }

    fn next_step(
        &self,
        live: &Game,
        position: &Game,
        _viewer: PlayerId,
        _plan: &GroupPlan,
        progress: &mut Self::Progress,
    ) -> NextStep {
        let Some(step) = progress.planned.get(progress.next_index) else {
            return NextStep::Done;
        };
        progress.next_index += 1;
        let Some(space) = position.action_space() else {
            return NextStep::Done;
        };
        match *step {
            SurveilStep::Graveyard(object, render_id) => {
                let index = space.actions.iter().position(|action| {
                    matches!(action, Action::SelectCard { object: offered, .. } if *offered == object)
                });
                match index {
                    Some(action_index) => NextStep::Step {
                        action_index,
                        item: Some(render_id),
                    },
                    None => {
                        let label = live.observed_object_name(object);
                        NextStep::Refuse(GroupRefusal {
                            item: Some(render_id),
                            reason: format!("{label} cannot be put into the graveyard."),
                            label,
                        })
                    }
                }
            }
            SurveilStep::FinishKeep => {
                let index = space
                    .actions
                    .iter()
                    .position(|action| matches!(action, Action::Decline { .. }));
                match index {
                    Some(action_index) => NextStep::Step {
                        action_index,
                        item: None,
                    },
                    None => NextStep::Refuse(GroupRefusal {
                        item: None,
                        label: "Surveil".to_string(),
                        reason: "Cannot finish surveil now.".to_string(),
                    }),
                }
            }
        }
    }

    fn finish(
        &self,
        _live: &Game,
        _position: &Game,
        _obs_events_from: usize,
        plan: &GroupPlan,
        progress: &Self::Progress,
    ) -> (GroupOutcome, Option<GroupRefusal>) {
        let GroupPlan::Surveil { graveyard } = plan else {
            unreachable!("matches_plan already verified variant");
        };
        let mut final_graveyard = progress.committed_graveyard.clone();
        final_graveyard.extend_from_slice(graveyard);
        let kept = progress
            .remaining
            .iter()
            .copied()
            .filter(|id| !graveyard.contains(id))
            .collect();
        (
            GroupOutcome::Surveil {
                graveyard: final_graveyard,
                kept,
            },
            None,
        )
    }
}

// -----------------------------------------------------------------------------
// OrderLibrary
// -----------------------------------------------------------------------------

pub(crate) struct OrderLibraryGroup;

pub(crate) struct OrderLibraryProgress {
    end: LibraryEnd,
    planned: Vec<(ObjectId, ObjectRenderId)>,
    next_index: usize,
    committed: Vec<ObjectRenderId>,
}

impl GroupKind for OrderLibraryGroup {
    type Progress = OrderLibraryProgress;

    fn describe(&self, game: &Game, viewer: PlayerId) -> Option<GroupSpec> {
        let Decision::OrderLibrary {
            player,
            object_ids,
            next,
            end,
        } = &game.state.suspended_decision.as_ref()?.decision
        else {
            return None;
        };
        let remaining = &object_ids[*next..];
        if *player != viewer || remaining.len() < 2 {
            return None;
        }
        Some(GroupSpec::OrderLibrary {
            end: (*end).into(),
            items: remaining.iter().map(|&o| group_card(game, o)).collect(),
            committed: object_ids[..*next]
                .iter()
                .map(|&o| card_render_id(game, o))
                .collect(),
        })
    }

    fn matches_plan(&self, plan: &GroupPlan) -> bool {
        matches!(plan, GroupPlan::OrderLibrary { .. })
    }

    fn validate(&self, game: &Game, spec: &GroupSpec, plan: &GroupPlan) -> PlanCheck {
        let (
            GroupSpec::OrderLibrary {
                items, committed, ..
            },
            GroupPlan::OrderLibrary { order },
        ) = (spec, plan)
        else {
            unreachable!("matches_plan already verified variant");
        };
        let remaining: HashSet<_> = items.iter().map(|card| card.item).collect();
        let mut seen = HashSet::new();
        for &id in order {
            if !remaining.contains(&id) {
                let reason = if committed.contains(&id) {
                    "has already been ordered."
                } else {
                    "is not in the pending library order."
                };
                return refuse_card(game, id, reason);
            }
            if !seen.insert(id) {
                return refuse_card(game, id, "can only be ordered once.");
            }
        }
        if seen.len() < remaining.len() {
            return PlanCheck::Incomplete;
        }
        PlanCheck::Ready
    }

    fn start(
        &self,
        game: &Game,
        _viewer: PlayerId,
        spec: &GroupSpec,
        plan: &GroupPlan,
    ) -> Result<Self::Progress, AgentError> {
        let (GroupSpec::OrderLibrary { end, committed, .. }, GroupPlan::OrderLibrary { order }) =
            (spec, plan)
        else {
            unreachable!("matches_plan already verified variant");
        };
        let Some(Decision::OrderLibrary {
            object_ids, next, ..
        }) = game.state.suspended_decision.as_ref().map(|s| &s.decision)
        else {
            return Err(AgentError("not a pending library order".into()));
        };
        let lookup = render_to_object_map(game, &object_ids[*next..]);
        let mut planned = Vec::with_capacity(order.len());
        for &id in order {
            let object = *lookup
                .get(&id)
                .ok_or_else(|| AgentError("card is not in the pending library order".into()))?;
            planned.push((object, id));
        }
        Ok(OrderLibraryProgress {
            end: *end,
            planned,
            next_index: 0,
            committed: committed.clone(),
        })
    }

    fn continues(&self, position: &Game, viewer: PlayerId, progress: &Self::Progress) -> bool {
        // Under `skip_trivial`, the final 1-card pick executes automatically and
        // exits `OrderLibrary` before `next_index` reaches `planned.len()`.
        progress.next_index < progress.planned.len()
            && matches!(
                position.state.suspended_decision.as_ref().map(|s| &s.decision),
                Some(Decision::OrderLibrary { player, end, .. })
                    if *player == viewer && LibraryEnd::from(*end) == progress.end
            )
    }

    fn next_step(
        &self,
        live: &Game,
        position: &Game,
        _viewer: PlayerId,
        _plan: &GroupPlan,
        progress: &mut Self::Progress,
    ) -> NextStep {
        let Some(&(object, render_id)) = progress.planned.get(progress.next_index) else {
            return NextStep::Done;
        };
        progress.next_index += 1;
        let Some(space) = position.action_space() else {
            return NextStep::Done;
        };
        let index = space.actions.iter().position(|action| {
            matches!(action, Action::SelectCard { object: offered, .. } if *offered == object)
        });
        match index {
            Some(action_index) => NextStep::Step {
                action_index,
                item: Some(render_id),
            },
            None => {
                let label = live.observed_object_name(object);
                NextStep::Refuse(GroupRefusal {
                    item: Some(render_id),
                    reason: format!("{label} cannot be ordered next."),
                    label,
                })
            }
        }
    }

    fn finish(
        &self,
        _live: &Game,
        _position: &Game,
        _obs_events_from: usize,
        plan: &GroupPlan,
        progress: &Self::Progress,
    ) -> (GroupOutcome, Option<GroupRefusal>) {
        let GroupPlan::OrderLibrary { order } = plan else {
            unreachable!("matches_plan already verified variant");
        };
        let mut final_order = progress.committed.clone();
        final_order.extend_from_slice(order);
        (
            GroupOutcome::OrderLibrary {
                end: progress.end,
                order: final_order,
            },
            None,
        )
    }
}
