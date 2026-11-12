//! Tests for cards filed in `src/soi.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{assert_choice_contract, library_card, permanent_named},
    play::{activate_first, pass_until, play_and_settle, select_card, settle},
};
use managym::agent::action::ActionSpaceKind;

#[test]
fn fetch_land_counts_the_found_land_and_triggers_tracker_twice() {
    let mut s = five_color_setup(&["Fabled Passage", "Tireless Tracker"]);
    for _ in 0..3 {
        s.force_permanent_on_battlefield(0, "Forest");
    }
    play_and_settle(&mut s, "Tireless Tracker");
    play_and_settle(&mut s, "Fabled Passage");
    s.pass_priority();
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    let fetch = permanent_named(&s, "Fabled Passage");
    activate_first(&mut s, fetch);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_choice_contract(&s);
    let found = library_card(&s, 0, "Forest");
    select_card(&mut s, found);
    settle(&mut s);
    let id = s.game().state.card_to_permanent[found].unwrap();
    assert!(!s.game().state.permanents[id].as_ref().unwrap().tapped);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 2);
}

#[test]
fn fevered_visions_draws_each_end_step_and_burns_an_opponent_with_four_cards() {
    use crate::support::play::resolve_everything;
    use managym::{
        flow::turn::StepKind,
        state::{game_object::PlayerId, zone::ZoneType},
    };
    for (held, life) in [(3, 18), (2, 20)] {
        let mut s = five_color_setup(&["Fevered Visions"]);
        s.force_permanent_on_battlefield(0, "Fevered Visions");
        s.advance_to_active_step(0, StepKind::End);
        let hand = s.zone_size(0, ZoneType::Hand);
        resolve_everything(&mut s);
        // Your own end step: you draw and take no damage, whatever your hand.
        s.assert_zone_size(0, ZoneType::Hand, hand + 1);
        s.assert_life(0, 20);
        s.advance_to_active_step(1, StepKind::End);
        s.game_mut().scenario_clear_hand(PlayerId(1));
        for _ in 0..held {
            crate::support::game::move_from_library(&mut s, 1, "Island", ZoneType::Hand);
        }
        resolve_everything(&mut s);
        s.assert_zone_size(1, ZoneType::Hand, held + 1);
        s.assert_life(1, life);
    }
}
