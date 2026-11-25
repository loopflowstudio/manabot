//! Variable row axes and the object joins a batch archive validates.
use super::observation_layout::{
    effect, fact, history, member, permanent, relation, stack, ArrayLayout,
};

/// Columns that hold object coordinates.
#[derive(Clone, Copy)]
pub(crate) enum Join {
    Objects,              // every element in action_focus
    Object(usize, usize), // column, coordinate base
}

pub(crate) fn joins(key: &str, policy_history: bool) -> &'static [Join] {
    use Join::*;
    match key {
        "action_focus" => &[Objects],
        "facts" => &[Object(fact::OBJECT, 1)],
        "effects" => &[
            Object(effect::CONTROLLER, 1),
            Object(effect::SOURCE, 1),
        ],
        "effect_members" => &[Object(member::OBJECT, 1)],
        "relations" => &[
            Object(relation::SOURCE, 1),
            Object(relation::REFERENCE, 1),
        ],
        "agent_permanents" | "opponent_permanents" => &[
            Object(permanent::ATTACHED_TO, 1),
            Object(permanent::BLOCKING, 1),
        ],
        "stack" => &[Object(stack::SOURCE, 1)],
        "events" if policy_history => &[
            Object(history::SOURCE_CONTEXT, 0),
            Object(history::TARGET_CONTEXT, 0),
        ],
        _ => &[],
    }
}

/// Extent key and row dimension, including the leading batch dimension.
pub(crate) fn row_axis(key: &str, array: &ArrayLayout) -> Option<(&'static str, usize)> {
    let seat = usize::from(array.axes[0] == "viewer_relative_seat");
    let axis = match *array.axes.get(seat)? {
        "fact" => "facts",
        "effect" => "effects",
        "detail" => "details",
        "membership" => "members",
        "relation" => "relations",
        "action" => "actions",
        "event" => "events",
        "card" => "cards",
        "definition" if key == "known_hand" => "known",
        "definition" => "decklists",
        "permanent" => "permanents",
        "stack" => "stack",
        _ => return None,
    };
    Some((axis, seat + 1))
}
