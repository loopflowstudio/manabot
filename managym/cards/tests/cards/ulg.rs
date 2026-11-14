//! Tests for cards filed in `src/ulg.rs`, in card order.

use crate::support::inspect::permanent_of;
use crate::support::{
    game::five_color_setup,
    inspect::tapped,
    play::{activate, enter_and_settle, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{card::Keyword, mana::Color},
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
