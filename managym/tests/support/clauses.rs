use crate::support::{elemental::card, helpers::Scenario};
use managym::{
    agent::action::Action,
    state::{
        game_object::{CardId, PermanentId},
        zone::ZoneType,
    },
};

pub(crate) fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let i = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(i);
}

pub(crate) fn activate(s: &mut Scenario, source: PermanentId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
        if *permanent == source && *ability_index == index)
    });
}

pub(crate) fn enter(s: &mut Scenario, name: &str) -> PermanentId {
    let id = card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.game().state.card_to_permanent[id].unwrap()
}

pub(crate) fn tapped(s: &Scenario, id: PermanentId) -> bool {
    s.game().state.permanents[id].as_ref().unwrap().tapped
}

pub(crate) fn keywords(s: &Scenario, id: PermanentId) -> managym::state::card::Keywords {
    let permanent = s.game().state.permanents[id].as_ref().unwrap();
    permanent.effective_keywords(&s.game().state.cards[permanent.card])
}

pub(crate) fn can_cast(s: &Scenario, id: CardId) -> bool {
    s.action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::CastSpell { card, .. } if *card == id))
}
