//! Tests for cards filed in `src/emn.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::observed,
    play::{cast, settle, target_opponent},
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
    let row = observed(&s, agent);
    assert_eq!(
        row.creature_types,
        Some(vec![]),
        "its creature types are gone"
    );
    assert!(row.temp_subtypes.is_empty());
    let [ability] = view.activated_abilities.as_slice() else {
        panic!("one granted ability: {:?}", view.activated_abilities);
    };
    assert_eq!(
        ability.mana_choices(),
        Some(vec![managym::state::mana::Mana::parse("C")])
    );
}

/// A basic land keeps its land subtype, which still corresponds to its one
/// card type (CR 205.1a, 205.3d), but loses its intrinsic `{T}: Add {G}`
/// (CR 305.6) with its other abilities: it taps only for `{C}`.
#[test]
fn imprisoned_in_the_moon_keeps_a_forest_a_forest_that_taps_only_for_colorless() {
    let mut s = five_color_setup(&["Imprisoned in the Moon"]);
    let forest = s.force_permanent_on_battlefield(1, "Forest");
    cast(&mut s, "Imprisoned in the Moon");
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    let view = s.game().characteristics(forest);
    assert_eq!(view.subtypes, vec!["Forest".to_string()]);
    let row = observed(&s, forest);
    assert_eq!(row.creature_types, None, "it had none to replace");
    assert!(row.temp_subtypes.is_empty(), "Forest is printed");
    let choices: Vec<_> = view
        .activated_abilities
        .iter()
        .map(|ability| ability.mana_choices())
        .collect();
    assert_eq!(
        choices,
        vec![Some(vec![managym::state::mana::Mana::parse("C")])],
        "only the granted colorless ability"
    );
}

/// A spell aimed only at the Dragon is copied onto each other creature its
/// caster controls; the Dragon's controller's other creatures are spared,
/// and a spell aimed at a player is not copied.
#[test]
fn mirrorwing_dragon_redirects_copies_onto_the_casters_creatures() {
    let mut s = five_color_setup(&["Mirrorwing Dragon", "Lightning Bolt", "Gray Ogre"]);
    let dragon = s.force_permanent_on_battlefield(1, "Mirrorwing Dragon");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let mine = s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(dragon)));
    settle(&mut s);
    assert!(s.game().state.permanents[mine].is_none());
    assert!(s.game().state.permanents[theirs].is_some());
    assert_eq!(
        s.game().state.permanents[dragon].as_ref().unwrap().damage,
        3
    );

    cast(&mut s, "Lightning Bolt");
    target_opponent(&mut s);
    settle(&mut s);
    s.assert_life(1, 17);
}
