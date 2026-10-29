use crate::support::{
    elemental::{card, cast, pass_until, permanent, settle, setup},
    helpers::Scenario,
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{game_object::CardId, zone::ZoneType},
};

pub(crate) fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let index = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(index);
}

pub(crate) fn enter(s: &mut Scenario, name: &str) -> CardId {
    let id = card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    if !s.game().state.pending_triggers.is_empty() {
        s.pass_priority();
    }
    settle(s);
    id
}

pub(crate) fn activate(s: &mut Scenario, name: &str, index: usize) {
    let source = permanent(s, name);
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(
        s,
        |action| matches!(action, Action::ActivateAbility { permanent, ability_index, .. } if *permanent == source && *ability_index == index),
    );
}

pub(crate) fn ready(s: &mut Scenario, name: &str) {
    let source = permanent(s, name);
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.game_mut().scenario_refresh_priority().unwrap();
}

pub(crate) fn can_cast(s: &Scenario, id: CardId) -> bool {
    s.action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::CastSpell { card, .. } if *card == id))
}

pub(crate) fn lesson_payment(names: &[&str], pool: &str) -> (Scenario, CardId, usize) {
    let mut s = setup(names);
    for name in names.iter().filter(|name| **name != "Waterbending Lesson") {
        enter(&mut s, name);
        if *name == "Great Divide Guide" {
            ready(&mut s, name);
        }
    }
    let before = s.zone_size(0, ZoneType::Hand);
    let spell = cast(&mut s, "Waterbending Lesson");
    if s.action_space().kind == ActionSpaceKind::PayOrNot {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    s.set_player_mana_pool(0, pool);
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 3);
    (s, spell, before)
}
