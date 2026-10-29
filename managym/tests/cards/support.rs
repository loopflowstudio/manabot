//! Directed drivers shared by the per-set card tests.
use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        zone::ZoneType,
    },
};

use super::helpers::Scenario;

/// Both players have 16 of each basic plus four of each named card, player 0
/// is in their first main phase with empty hands and an empty battlefield.
pub fn setup(names: &[&str]) -> Scenario {
    let mut deck: BTreeMap<String, usize> = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        .into_iter()
        .map(|name| (name.into(), 16))
        .collect();
    for name in names {
        deck.insert((*name).into(), 4);
    }
    let mut s = Scenario::new(deck.clone(), deck, 161);
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().scenario_clear_hand(PlayerId(0));
    s.game_mut().scenario_clear_hand(PlayerId(1));
    s
}

/// A library copy of `name` owned by `player`.
pub fn library_card(s: &Scenario, player: usize, name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(i, c)| {
            (c.name == name
                && c.owner == PlayerId(player)
                && s.game().state.zones.zone_of(CardId(i)) == Some(ZoneType::Library))
            .then_some(CardId(i))
        })
        .unwrap_or_else(|| panic!("no {name} in library"))
}

pub fn has_action(s: &Scenario, predicate: impl Fn(&Action) -> bool) -> bool {
    s.action_space().actions.iter().any(predicate)
}

pub fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let i = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(i);
}

/// Move a library copy of `name` to player 0's hand and refresh priority.
pub fn to_hand(s: &mut Scenario, name: &str) -> CardId {
    let id = library_card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    id
}

/// Play a library copy of `name` with the ordinary land action.
pub fn play_land(s: &mut Scenario, name: &str) -> PermanentId {
    let id = to_hand(s, name);
    choose(
        s,
        |a| matches!(a, Action::PlayLand { card, .. } if *card == id),
    );
    s.game().state.card_to_permanent[id].expect("land entered")
}

/// Cast a library copy of `name`, leaving it on the stack.
pub fn cast(s: &mut Scenario, name: &str) -> CardId {
    let id = to_hand(s, name);
    choose(
        s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == id),
    );
    id
}

pub fn can_activate(s: &mut Scenario, source: PermanentId, index: usize) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    has_action(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
            if *permanent == source && *ability_index == index)
    })
}

pub fn activate(s: &mut Scenario, source: PermanentId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
            if *permanent == source && *ability_index == index)
    });
}

/// Pass and decline until the stack is empty and player priority returns.
pub fn settle(s: &mut Scenario) {
    for _ in 0..100 {
        if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.suspended_decision.is_none()
        {
            return;
        }
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. } | Action::Decline { .. }))
            .unwrap_or(0);
        s.step_action(index);
    }
    panic!("resolution did not settle")
}

/// Pass priority until the given decision is offered.
pub fn pass_until(s: &mut Scenario, kind: ActionSpaceKind) {
    for _ in 0..10 {
        if s.action_space().kind == kind {
            return;
        }
        s.pass_priority();
    }
    assert_eq!(s.action_space().kind, kind);
}

pub fn tapped(s: &Scenario, id: PermanentId) -> bool {
    s.game().state.permanents[id].as_ref().unwrap().tapped
}

pub fn counters(s: &Scenario, id: PermanentId) -> i32 {
    s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .plus1_counters
}
