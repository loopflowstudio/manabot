//! Variable row axes and the joins that can keep otherwise empty rows alive.
use super::observation_layout::{
    action, effect, fact, history, member, permanent, relation, stack, ArrayLayout,
};

/// A zero-based object coordinate or a one-past-end extent on another axis.
#[derive(Clone, Copy)]
pub(crate) enum Join {
    Objects,                          // every element in action_focus
    Object(usize, usize),             // column, coordinate base
    Extent(&'static str, usize),      // destination and one-based reference column
    Span(&'static str, usize, usize), // destination, offset and count columns
}

pub(crate) fn joins(
    key: &str,
    policy_history: bool,
    layout: super::observation_encoder::InputLayout,
) -> &'static [Join] {
    use Join::*;
    match key {
        "action_focus" => &[Objects],
        "facts" => &[Object(fact::OBJECT, 1)],
        "effects" => &[
            Object(effect::CONTROLLER, 1),
            Object(effect::SOURCE, 1),
            Span("details", effect::DETAIL_OFFSET, effect::DETAIL_COUNT),
            Span("members", effect::MEMBER_OFFSET, effect::MEMBER_COUNT),
        ],
        "effect_members" => &[Object(member::OBJECT, 1)],
        "relations" => &[
            Object(relation::SOURCE, 1),
            Object(relation::REFERENCE, 1),
            Extent("relations", relation::FRAME),
        ],
        "actions" if layout == super::observation_encoder::InputLayout::OneToken => &[
            Extent("relations", action::CONTINUATION),
            Object(action::ATTACK_TARGET, 1),
        ],
        "agent_permanents" | "opponent_permanents"
            if layout == super::observation_encoder::InputLayout::OneToken =>
        {
            &[
                Object(permanent::ATTACHED_TO, 1),
                Object(permanent::BLOCKING, 1),
                Object(permanent::ATTACKED_PERMANENT, 1),
            ]
        }
        "actions" => &[Extent("relations", action::CONTINUATION)],
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
