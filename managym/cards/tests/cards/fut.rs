//! Tests for cards filed in `src/fut.rs`, in card order.

use crate::support::{
    game::{five_color_setup, move_to_hand, Scenario},
    inspect::can_cast,
    play::{cast_card, cast_owned, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{game_object::PlayerId, zone::ZoneType},
};

fn on_opponents_upkeep(s: &mut Scenario) {
    s.advance_to_active_step(1, StepKind::Upkeep);
    until_my_priority(s);
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
}

/// The active opponent receives priority after each resolution; pass it back.
fn until_my_priority(s: &mut Scenario) {
    while s.action_space().player != Some(PlayerId(0)) {
        s.pass_priority();
    }
}

/// The grant gives the next creature card flash timing and draws a card; a
/// noncreature spell leaves it in place, and the first creature cast ends it.
#[test]
fn scout_s_warning_grants_flash_to_exactly_the_next_creature_card() {
    let mut s = five_color_setup(&["Scout's Warning", "Gray Ogre", "Sol Ring"]);
    let first = move_to_hand(&mut s, "Gray Ogre");
    let second = move_to_hand(&mut s, "Gray Ogre");
    let ring = move_to_hand(&mut s, "Sol Ring");
    on_opponents_upkeep(&mut s);
    assert!(!can_cast(&s, first));

    cast_owned(&mut s, "Scout's Warning");
    let hand = s.zone_size(0, ZoneType::Hand);
    settle(&mut s);
    until_my_priority(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
    assert!(can_cast(&s, first) && can_cast(&s, second));
    assert!(!can_cast(&s, ring), "only creature cards gain flash");

    cast_card(&mut s, first);
    settle(&mut s);
    until_my_priority(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre").len(), 1);
    assert!(
        !can_cast(&s, second),
        "the grant ended with the first creature cast"
    );
}

/// Casting a creature at ordinary timing still consumes the grant (CR 601.2i).
#[test]
fn scout_s_warning_is_consumed_by_a_creature_cast_at_sorcery_timing() {
    let mut s = five_color_setup(&["Scout's Warning", "Gray Ogre"]);
    let first = move_to_hand(&mut s, "Gray Ogre");
    let second = move_to_hand(&mut s, "Gray Ogre");
    cast_owned(&mut s, "Scout's Warning");
    settle(&mut s);
    cast_card(&mut s, first);
    settle(&mut s);
    on_opponents_upkeep(&mut s);
    assert!(!can_cast(&s, second));
}

/// The grant lasts only this turn.
#[test]
fn scout_s_warning_ends_with_the_turn() {
    let mut s = five_color_setup(&["Scout's Warning", "Gray Ogre"]);
    let ogre = move_to_hand(&mut s, "Gray Ogre");
    cast_owned(&mut s, "Scout's Warning");
    settle(&mut s);
    on_opponents_upkeep(&mut s);
    assert!(!can_cast(&s, ogre));
}

#[test]
fn rites_of_flourishing_draws_for_each_player_and_grants_a_land_play() {
    use crate::support::{
        inspect::has_action,
        play::{play_land, resolve_everything},
    };
    use managym::agent::action::Action;
    let mut s = five_color_setup(&["Rites of Flourishing"]);
    s.force_permanent_on_battlefield(0, "Rites of Flourishing");
    s.game_mut().scenario_refresh_priority().unwrap();
    play_land(&mut s, "Forest");
    play_land(&mut s, "Island");
    move_to_hand(&mut s, "Plains");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!has_action(&s, |a| matches!(a, Action::PlayLand { .. })));
    // The opponent's draw step: their turn-based draw and the additional one.
    s.advance_to_active_step(1, StepKind::Draw);
    let hand = s.zone_size(1, ZoneType::Hand);
    resolve_everything(&mut s);
    s.assert_zone_size(1, ZoneType::Hand, hand + 1);
}
