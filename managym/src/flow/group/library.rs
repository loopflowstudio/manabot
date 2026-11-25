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

fn group_cards(game: &Game, objects: &[ObjectId]) -> Vec<GroupCard> {
    objects
        .iter()
        .map(|&object| GroupCard {
            item: card_render_id(game, object),
            label: game.observed_object_name(object),
        })
        .collect()
}

fn render_ids(game: &Game, objects: impl IntoIterator<Item = ObjectId>) -> Vec<ObjectRenderId> {
    objects
        .into_iter()
        .map(|object| card_render_id(game, object))
        .collect()
}

fn render_to_object_map(game: &Game, objects: &[ObjectId]) -> HashMap<ObjectRenderId, ObjectId> {
    objects
        .iter()
        .copied()
        .map(|object| (card_render_id(game, object), object))
        .collect()
}

fn resolve_card(
    lookup: &HashMap<ObjectRenderId, ObjectId>,
    id: ObjectRenderId,
    missing_err: &str,
) -> Result<ObjectId, AgentError> {
    lookup
        .get(&id)
        .copied()
        .ok_or_else(|| AgentError(missing_err.into()))
}

struct CardPlanReasons {
    committed: &'static str,
    missing: &'static str,
    duplicate: &'static str,
}

fn validate_card_plan<'a>(
    game: &Game,
    items: &[GroupCard],
    committed: impl IntoIterator<Item = &'a ObjectRenderId>,
    planned: impl IntoIterator<Item = &'a ObjectRenderId>,
    require_all: bool,
    reasons: CardPlanReasons,
) -> PlanCheck {
    let remaining: HashSet<_> = items.iter().map(|card| card.item).collect();
    let committed_set: HashSet<_> = committed.into_iter().copied().collect();
    let mut seen = HashSet::new();
    let refuse = |id: ObjectRenderId, suffix: &str| {
        let label = game.render_card_label(id);
        PlanCheck::Refused(GroupRefusal {
            item: Some(id),
            reason: format!("{label} {suffix}"),
            label,
        })
    };
    for &id in planned {
        if !remaining.contains(&id) {
            let reason = if committed_set.contains(&id) {
                reasons.committed
            } else {
                reasons.missing
            };
            return refuse(id, reason);
        }
        if !seen.insert(id) {
            return refuse(id, reasons.duplicate);
        }
    }
    if require_all && seen.len() < remaining.len() {
        return PlanCheck::Incomplete;
    }
    PlanCheck::Ready
}

#[derive(Clone, Copy)]
enum LibraryStep {
    Scry {
        object: ObjectId,
        render_id: ObjectRenderId,
        to_bottom: bool,
    },
    Select {
        object: ObjectId,
        render_id: ObjectRenderId,
        refusal_suffix: &'static str,
    },
    DeclineSurveil,
}

fn step_library(
    planned: &[LibraryStep],
    next_index: &mut usize,
    live: &Game,
    position: &Game,
) -> NextStep {
    let Some(&step) = planned.get(*next_index) else {
        return NextStep::Done;
    };
    *next_index += 1;
    let Some(space) = position.action_space() else {
        return NextStep::Done;
    };
    match step {
        LibraryStep::Scry {
            object,
            render_id,
            to_bottom,
        } => match space.actions.iter().position(|action| {
            matches!(
                action,
                Action::ScryCard {
                    object: offered,
                    to_bottom: offered_bottom,
                    ..
                } if *offered == object && *offered_bottom == to_bottom
            )
        }) {
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
        },
        LibraryStep::Select {
            object,
            render_id,
            refusal_suffix,
        } => match space.actions.iter().position(|action| {
            matches!(action, Action::SelectCard { object: offered, .. } if *offered == object)
        }) {
            Some(action_index) => NextStep::Step {
                action_index,
                item: Some(render_id),
            },
            None => {
                let label = live.observed_object_name(object);
                NextStep::Refuse(GroupRefusal {
                    item: Some(render_id),
                    reason: format!("{label} {refusal_suffix}"),
                    label,
                })
            }
        },
        LibraryStep::DeclineSurveil => match space
            .actions
            .iter()
            .position(|action| matches!(action, Action::Decline { .. }))
        {
            Some(action_index) => NextStep::Step {
                action_index,
                item: None,
            },
            None => NextStep::Refuse(GroupRefusal {
                item: None,
                label: "Surveil".to_string(),
                reason: "Cannot finish surveil now.".to_string(),
            }),
        },
    }
}

// -----------------------------------------------------------------------------
// Scry
// -----------------------------------------------------------------------------

pub(crate) struct ScryGroup;

pub(crate) struct ScryProgress {
    planned: Vec<LibraryStep>,
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
            items: group_cards(game, remaining),
            committed_top: render_ids(game, top.iter().copied()),
            // `Decision::Scry` commits `bottom.iter().rev()` to index 0, so
            // `bottom.last()` sits above `bottom[0]`. Expose top-of-bottom first.
            committed_bottom: render_ids(game, bottom.iter().rev().copied()),
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
        validate_card_plan(
            game,
            items,
            committed_top.iter().chain(committed_bottom.iter()),
            top.iter().chain(bottom.iter()),
            true,
            CardPlanReasons {
                committed: "has already been placed.",
                missing: "is not in the pending scry.",
                duplicate: "can only be placed once.",
            },
        )
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
        for &render_id in top {
            let object = resolve_card(&lookup, render_id, "card is not in the pending scry")?;
            planned.push(LibraryStep::Scry {
                object,
                render_id,
                to_bottom: false,
            });
        }
        // `Decision::Scry` repositions `bottom.iter().rev()` to index 0, so
        // emitting `plan.bottom` in reverse commits `plan.bottom[0]` above
        // `plan.bottom.last()`, matching `OrderLibrary { end: Bottom }`.
        for &render_id in bottom.iter().rev() {
            let object = resolve_card(&lookup, render_id, "card is not in the pending scry")?;
            planned.push(LibraryStep::Scry {
                object,
                render_id,
                to_bottom: true,
            });
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
        step_library(&progress.planned, &mut progress.next_index, live, position)
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

pub(crate) struct SurveilProgress {
    planned: Vec<LibraryStep>,
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
            items: group_cards(game, &remaining),
            committed_graveyard: render_ids(game, graveyard.iter().copied()),
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
        validate_card_plan(
            game,
            items,
            committed_graveyard,
            graveyard,
            false,
            CardPlanReasons {
                committed: "is already marked for the graveyard.",
                missing: "is not in the pending surveil.",
                duplicate: "can only be chosen once.",
            },
        )
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
        for &render_id in graveyard {
            let object = resolve_card(&lookup, render_id, "card is not in the pending surveil")?;
            planned.push(LibraryStep::Select {
                object,
                render_id,
                refusal_suffix: "cannot be put into the graveyard.",
            });
        }
        // When every remaining card goes to the graveyard, the final SelectCard
        // finishes Surveil automatically without an explicit Decline step.
        if graveyard.len() < remaining_objects.len() {
            planned.push(LibraryStep::DeclineSurveil);
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
        step_library(&progress.planned, &mut progress.next_index, live, position)
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
    planned: Vec<LibraryStep>,
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
            items: group_cards(game, remaining),
            committed: render_ids(game, object_ids[..*next].iter().copied()),
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
        validate_card_plan(
            game,
            items,
            committed,
            order,
            true,
            CardPlanReasons {
                committed: "has already been ordered.",
                missing: "is not in the pending library order.",
                duplicate: "can only be ordered once.",
            },
        )
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
        for &render_id in order {
            let object = resolve_card(
                &lookup,
                render_id,
                "card is not in the pending library order",
            )?;
            planned.push(LibraryStep::Select {
                object,
                render_id,
                refusal_suffix: "cannot be ordered next.",
            });
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
        step_library(&progress.planned, &mut progress.next_index, live, position)
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
