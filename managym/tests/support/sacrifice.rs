use crate::support::{
    elemental::{card, cast, settle},
    helpers::Scenario,
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

pub(crate) fn enter(s: &mut Scenario, name: &str) -> CardId {
    let id = card(s, 0, name);
    if s.game().state.cards[id].types.is_land() {
        s.game_mut().move_card(id, ZoneType::Battlefield);
        s.game_mut().scenario_refresh_priority().unwrap();
    } else {
        cast(s, name);
    }
    settle(s);
    id
}

pub(crate) fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let i = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(i);
}

pub(crate) fn activate(s: &mut Scenario, id: PermanentId) {
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(
        s,
        |a| matches!(a,Action::ActivateAbility {permanent,..} if *permanent==id),
    );
}

pub(crate) fn target_opponent(s: &mut Scenario) {
    choose(s, |a| {
        matches!(
            a,
            Action::ChooseTarget {
                target: Target::Player(PlayerId(1)),
                ..
            }
        )
    });
}

/// Resolve damage triggers with a fixed target; fail if the stack never settles.
pub(crate) fn settle_targeting_opponent(s: &mut Scenario) {
    for _ in 0..20 {
        if s.action_space().kind == ActionSpaceKind::ChooseTarget {
            target_opponent(s);
        } else if s.action_space().kind == ActionSpaceKind::OrderTriggers {
            s.step_action(0);
        } else if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
        {
            return;
        } else {
            s.pass_priority();
        }
    }
    panic!("damage triggers did not settle");
}

pub(crate) fn select(s: &mut Scenario, card: CardId) {
    choose(
        s,
        |a| matches!(a,Action::SelectCard {card: selected,..} if *selected==card),
    );
}
