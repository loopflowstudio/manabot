//! Shared drivers for per-card tests: a five-color fixture with ample mana and
//! helpers that take legal actions rather than mutating game state.
use std::collections::BTreeMap;

use crate::support::helpers::Scenario;
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

/// Both players hold sixteen of each basic plus four of each named card, start
/// in player 0's main phase with empty hands and ample floating mana.
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
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
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
        .unwrap_or_else(|| panic!("no {name}"))
}

/// Put a library copy of `name` into player 0's hand and cast it.
pub fn cast(s: &mut Scenario, name: &str) -> CardId {
    let id = library_card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Hand);
    cast_card(s, id);
    id
}

/// Cast `id` from whichever zone currently permits it.
pub fn cast_card(s: &mut Scenario, id: CardId) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == id),
    );
}

pub fn can_cast(s: &mut Scenario, id: CardId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    s.action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::CastSpell { card, .. } if *card == id))
}

/// Pass or decline until priority returns with an empty stack.
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

/// Pass priority until the action space is `kind`.
pub fn pass_until(s: &mut Scenario, kind: ActionSpaceKind) {
    for _ in 0..10 {
        if s.action_space().kind == kind {
            return;
        }
        s.pass_priority();
    }
    assert_eq!(s.action_space().kind, kind);
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

pub fn mode(s: &mut Scenario, wanted: usize) {
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    choose(
        s,
        |a| matches!(a, Action::ChooseMode { mode, .. } if *mode == wanted),
    );
}

pub fn can_activate(s: &mut Scenario, source: PermanentId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    s.action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == source))
}

pub fn activate(s: &mut Scenario, source: PermanentId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
        if *permanent == source && *ability_index == index)
    });
}

/// Abilities of `card` that function from its current hand or graveyard.
pub fn can_activate_card(s: &mut Scenario, source: CardId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    s.action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateCardAbility { card, .. } if *card == source))
}

pub fn activate_card(s: &mut Scenario, source: CardId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(s, |a| {
        matches!(a, Action::ActivateCardAbility { card, ability_index, .. }
        if *card == source && *ability_index == index)
    });
}

pub fn offered_targets(s: &Scenario) -> Vec<Target> {
    s.action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::ChooseTarget { target, .. } => Some(*target),
            _ => None,
        })
        .collect()
}

pub fn counters(s: &Scenario, id: PermanentId) -> i32 {
    s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .plus1_counters
}

pub fn ready(s: &mut Scenario, id: PermanentId) {
    s.game_mut().state.permanents[id]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
}
