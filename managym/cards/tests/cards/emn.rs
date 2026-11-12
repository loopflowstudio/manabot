//! Tests for cards filed in `src/emn.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    play::{cast, settle},
};
use managym::state::{card::CardType, game_object::Target};

#[test]
fn imprisoned_in_the_moon_makes_a_colorless_land_that_taps_for_colorless() {
    let mut s = five_color_setup(&["Imprisoned in the Moon", "Agent of Treachery"]);
    let agent = s.force_permanent_on_battlefield(1, "Agent of Treachery");
    clear_summoning_sickness(&mut s, agent);
    cast(&mut s, "Imprisoned in the Moon");
    assert!(s.choose_target(Target::Permanent(agent)));
    settle(&mut s);
    let view = s.game().characteristics(agent);
    assert!(view.types.types.contains(&CardType::Land));
    assert!(!view.types.types.contains(&CardType::Creature));
    assert!(view.colors.is_empty());
    assert!(view.subtypes.is_empty());
    assert!(view.triggered_abilities.is_empty());
    let [ability] = view.activated_abilities.as_slice() else {
        panic!("one granted ability: {:?}", view.activated_abilities);
    };
    assert_eq!(
        ability.mana_choices(),
        Some(vec![managym::state::mana::Mana::parse("C")])
    );
}
