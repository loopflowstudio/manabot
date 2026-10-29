use crate::support::helpers::*;
use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    state::{
        ability::Ability,
        game_object::{CardId, PermanentId, PlayerId},
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

pub(crate) fn find_owned_card_off_battlefield(s: &Scenario, player: usize, name: &str) -> CardId {
    let game = s.game();
    for (index, card) in game.state.cards.iter().enumerate() {
        if card.owner != PlayerId(player) || card.name != name {
            continue;
        }
        let card_id = CardId(index);
        if game.state.zones.zone_of(card_id) != Some(ZoneType::Battlefield) {
            return card_id;
        }
    }
    panic!("no off-battlefield {name} found for player {player}");
}

/// Move a card onto the battlefield through the real zone-change path so its
/// enters-the-battlefield triggers fire.
pub(crate) fn enters_battlefield_with_triggers(
    s: &mut Scenario,
    player: usize,
    name: &str,
) -> PermanentId {
    let card_id = find_owned_card_off_battlefield(s, player, name);
    s.game_mut().move_card(card_id, ZoneType::Battlefield);
    s.game().state.card_to_permanent[card_id].expect("permanent should exist")
}

pub(crate) fn permanent(s: &Scenario, id: PermanentId) -> &managym::state::permanent::Permanent {
    s.game().state.permanents[id]
        .as_ref()
        .expect("permanent should exist")
}

pub(crate) fn counters(s: &Scenario, id: PermanentId) -> i32 {
    permanent(s, id).plus1_counters
}

/// Advance with default actions until the stack, pending events, and pending
/// triggers are empty (all triggered abilities have resolved).
pub(crate) fn resolve_all(s: &mut Scenario) {
    s.advance_until(
        |s| {
            s.game().state.stack_objects.is_empty()
                && s.game().state.pending_events.is_empty()
                && s.game().state.pending_triggers.is_empty()
                && s.action_space().kind == ActionSpaceKind::Priority
        },
        "stack should empty out".to_string(),
    );
}

/// Move every card in a player's hand back to their library (test setup —
/// removes castable noise so `PriorityCastSpell` is unambiguous).
pub(crate) fn clear_hand(s: &mut Scenario, player: usize) {
    let hand: Vec<CardId> = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(player))
        .clone();
    for card in hand {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(player), ZoneType::Library);
    }
}

/// Move a specific card from the library into the hand (avoids colliding
/// with same-named cards already on the battlefield or in hand).
pub(crate) fn force_from_library(s: &mut Scenario, player: usize, name: &str) -> CardId {
    let game = s.game();
    let card_id = game
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(player))
        .iter()
        .copied()
        .find(|card_id| game.state.cards[card_id].name == name)
        .unwrap_or_else(|| panic!("no {name} in player {player}'s library"));
    s.game_mut()
        .state
        .zones
        .move_card(card_id, PlayerId(player), ZoneType::Hand);
    card_id
}

pub(crate) fn force_lands(s: &mut Scenario, player: usize, land: &str, count: usize) {
    for _ in 0..count {
        s.force_permanent_on_battlefield(player, land);
    }
}

pub(crate) fn cast_and_resolve(s: &mut Scenario) {
    assert!(
        s.take_action_by_type(ActionType::PriorityCastSpell),
        "cast action should be available"
    );
    s.pass_priority();
    s.pass_priority();
}

// ---------------------------------------------------------------------------
// Avatar Enthusiasts — "another Ally you control enters" filtering.
// ---------------------------------------------------------------------------

pub(crate) fn enthusiasts_scenario() -> (Scenario, PermanentId) {
    let list = deck(&[
        ("Plains", 20),
        ("Avatar Enthusiasts", 8),
        ("South Pole Voyager", 8),
        ("Gray Ogre", 4),
    ]);
    let mut s = Scenario::new(list.clone(), list, 11);
    let enthusiasts = s.force_permanent_on_battlefield(0, "Avatar Enthusiasts");
    (s, enthusiasts)
}

// ---------------------------------------------------------------------------
// Attack triggers, tapped-and-attacking tokens, vigilance & haste.
// ---------------------------------------------------------------------------

pub(crate) fn inject_ability(s: &mut Scenario, card_id: CardId, ability: Ability) {
    s.game_mut().state.cards[card_id].abilities.push(ability);
}

pub(crate) fn battlefield_card(s: &Scenario, permanent_id: PermanentId) -> CardId {
    permanent(s, permanent_id).card
}
