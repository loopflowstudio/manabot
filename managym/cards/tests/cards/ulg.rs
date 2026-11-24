//! Tests for cards filed in `src/ulg.rs`, in card order.

use crate::support::inspect::permanent_of;
use crate::support::{
    game::{five_color_setup, Scenario},
    inspect::tapped,
    play::{activate, enter_and_settle, settle},
};
use managym::{
    agent::{action::ActionSpaceKind, observation::Observation},
    flow::turn::StepKind,
    state::{card::Keyword, game_object::PlayerId, mana::Color},
};

#[test]
fn treetop_village_enters_tapped_and_becomes_a_3_3_green_ape_until_end_of_turn() {
    let mut s = five_color_setup(&["Treetop Village"]);
    let object = enter_and_settle(&mut s, "Treetop Village");
    let village = permanent_of(&s, object).unwrap();
    assert!(tapped(&s, village));
    activate(&mut s, village, 0);
    settle(&mut s);
    let view = s.game().characteristics(village);
    assert!(view.is_creature() && view.is_land());
    assert_eq!((view.power, view.toughness), (3, 3));
    assert!(view.creature_types.iter().any(|t| t == "Ape"));
    assert!(view.keywords.has(Keyword::Trample));
    assert_eq!(
        view.colors.iter().copied().collect::<Vec<_>>(),
        vec![Color::Green]
    );
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!s.game().characteristics(village).is_creature());
}

/// CR 302.6: a land animated on the turn it came under its controller's
/// control cannot attack; one controlled since the turn began can. The
/// observation reports sickness only while it is a creature without haste.
#[test]
fn treetop_village_animated_the_turn_it_enters_cannot_attack() {
    let mut s = five_color_setup(&["Treetop Village"]);
    let object = enter_and_settle(&mut s, "Treetop Village");
    let village = permanent_of(&s, object).unwrap();
    s.game_mut().state.permanents[village]
        .as_mut()
        .unwrap()
        .tapped = false;
    let id = s.game().state.permanents[village].as_ref().unwrap().id.0 as i32;
    let sick = |s: &Scenario| {
        Observation::for_player(s.game(), PlayerId(0))
            .agent_permanents
            .iter()
            .find(|row| row.id == id)
            .map(|row| row.is_summoning_sick)
    };
    assert_eq!(sick(&s), Some(false), "a land is never sick");
    activate(&mut s, village, 0);
    settle(&mut s);
    assert_eq!(sick(&s), Some(true), "an Ape that entered this turn");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_ne!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);

    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "1G");
    activate(&mut s, village, 0);
    settle(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
}
