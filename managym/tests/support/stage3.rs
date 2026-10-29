use crate::support::helpers::*;
use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};
use std::collections::BTreeMap;

pub(crate) fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

pub(crate) fn force_lands(s: &mut Scenario, player: usize, land: &str, count: usize) {
    for _ in 0..count {
        s.force_permanent_on_battlefield(player, land);
    }
}

pub(crate) fn permanent(s: &Scenario, id: PermanentId) -> &managym::state::permanent::Permanent {
    s.game().state.permanents[id]
        .as_ref()
        .expect("permanent should exist")
}

pub(crate) fn hand(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(player))
        .clone()
}

pub(crate) fn clear_hand(s: &mut Scenario, player: usize) {
    for card in hand(s, player) {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(player), ZoneType::Library);
    }
}

pub(crate) fn cast_only(s: &mut Scenario) {
    assert!(
        s.take_action_by_type(ActionType::PriorityCastSpell),
        "cast action should be available"
    );
}

pub(crate) fn cast_and_resolve(s: &mut Scenario) {
    cast_only(s);
    s.pass_priority();
    s.pass_priority();
}

/// Move an owned card straight into a player's graveyard (silent test
/// setup — e.g. planting a Lesson).
pub(crate) fn plant_in_graveyard(s: &mut Scenario, player: usize, card_name: &str) -> CardId {
    let card = find_owned_card(s, player, card_name);
    s.game_mut()
        .state
        .zones
        .move_card(card, PlayerId(player), ZoneType::Graveyard);
    card
}

pub(crate) fn find_owned_card(s: &Scenario, player: usize, card_name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(index, card)| {
            (card.owner == PlayerId(player)
                && card.name == card_name
                && matches!(
                    s.game().state.zones.zone_of(CardId(index)),
                    Some(ZoneType::Hand) | Some(ZoneType::Library)
                ))
            .then_some(CardId(index))
        })
        .unwrap_or_else(|| panic!("card {card_name} not found for player {player}"))
}

/// Put a fresh library copy into the hand — the stock helper re-finds
/// the same copy when called twice.
pub(crate) fn force_new_card_in_hand(s: &mut Scenario, player: usize, card_name: &str) -> CardId {
    let card = s
        .game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(index, card)| {
            (card.owner == PlayerId(player)
                && card.name == card_name
                && s.game().state.zones.zone_of(CardId(index)) == Some(ZoneType::Library))
            .then_some(CardId(index))
        })
        .unwrap_or_else(|| panic!("no library copy of {card_name} for player {player}"));
    s.game_mut()
        .state
        .zones
        .move_card(card, PlayerId(player), ZoneType::Hand);
    card
}

/// Advance until the stack, pending triggers, and decisions are drained.
pub(crate) fn resolve_everything(s: &mut Scenario) {
    s.advance_until(
        |sc| {
            sc.game().state.pending_triggers.is_empty()
                && sc.game().state.trigger_placement.is_none()
                && sc.game().state.stack_objects.is_empty()
                && sc.game().state.suspended_decision.is_none()
        },
        "stack and triggers should drain".to_string(),
    );
}

pub(crate) fn clear_summoning_sickness(s: &mut Scenario, id: PermanentId) {
    s.game_mut().state.permanents[id]
        .as_mut()
        .expect("permanent")
        .summoning_sick = false;
}

// ---------------------------------------------------------------------------
// Earthbend — Badgermole Cub.
// ---------------------------------------------------------------------------

pub(crate) fn cub_deck() -> BTreeMap<String, usize> {
    deck(&[("Forest", 24), ("Badgermole Cub", 16)])
}

/// Cast a Cub with lands out; answer the earthbend target choice with the
/// first offered land. Returns the animated land's permanent id.
pub(crate) fn earthbend_a_land(s: &mut Scenario) -> PermanentId {
    cast_and_resolve(s);
    // ETB trigger needs a target: land you control.
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    let target = s
        .action_space()
        .actions
        .iter()
        .find_map(|action| match action {
            Action::ChooseTarget {
                target: managym::state::target::Target::Permanent(p),
                ..
            } => Some(*p),
            _ => None,
        })
        .expect("land target should be offered");
    assert!(s.choose_target(Target::Permanent(target)));
    s.pass_priority();
    s.pass_priority(); // trigger resolves
    target
}

// ---------------------------------------------------------------------------
// Exile-until-this-leaves — Earth Kingdom Jailer.
// ---------------------------------------------------------------------------

pub(crate) fn jailer_scenario(seed: u64) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Earth Kingdom Jailer", 16)]),
        deck(&[
            ("Mountain", 16),
            ("Gray Ogre", 8),
            ("Raging Goblin", 8),
            ("Lightning Bolt", 8),
        ]),
        seed,
    );
    force_lands(&mut s, 0, "Plains", 3);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "Earth Kingdom Jailer");
    s
}

// ---------------------------------------------------------------------------
// Until-end-of-combat mana — Fire Nation Cadets (firebending 2).
// ---------------------------------------------------------------------------

pub(crate) fn cadets_scenario(seed: u64, with_lesson: bool) -> (Scenario, PermanentId) {
    let mut s = Scenario::new(
        deck(&[
            ("Mountain", 12),
            ("Fire Nation Cadets", 8),
            ("Firebending Lesson", 12),
            ("It'll Quench Ya!", 8),
        ]),
        deck(&[("Plains", 24), ("Wall of Stone", 16)]),
        seed,
    );
    let cadets = s.force_permanent_on_battlefield(0, "Fire Nation Cadets");
    clear_summoning_sickness(&mut s, cadets);
    clear_hand(&mut s, 0);
    if with_lesson {
        plant_in_graveyard(&mut s, 0, "It'll Quench Ya!");
    }
    (s, cadets)
}

pub(crate) fn attack_with(s: &mut Scenario, attacker: PermanentId) {
    let mut attacked = false;
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let space = s.action_space().clone();
        let index = space.actions.iter().position(|action| {
            matches!(action, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == attacker)
        });
        match index {
            Some(index) => {
                s.step_action(index);
                attacked = true;
            }
            None => s.decline_attack(),
        }
    }
    assert!(attacked, "expected to attack");
}
