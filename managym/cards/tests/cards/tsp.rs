//! Time Spiral card behavior.

#[test]
fn foundations_spell_cancel() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{now, zone_of};
    use crate::support::play::{cast, settle, spend_colorless};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Cancel", "Gray Ogre"]);
    let victim = cast(&mut s, "Gray Ogre");
    spend_colorless(&mut s);
    cast(&mut s, "Cancel");
    assert!(s.choose_target(Target::StackSpell(now(&s, victim))));
    settle(&mut s);
    assert_eq!(zone_of(&s, victim), Some(ZoneType::Graveyard));
}

#[test]
fn mystical_teachings_finds_instant_or_flash_and_flashbacks() {
    use crate::support::game::{clear_hand, five_color_setup, move_to_hand};
    use crate::support::inspect::{offered_cards, zone_of};
    use crate::support::play::{cast_card, pass_until, select_card, settle, spend_colorless};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&[
        "Mystical Teachings",
        "Lightning Bolt",
        "High Fae Trickster",
        "Gray Ogre",
        "Seismic Rupture",
    ]);
    clear_hand(&mut s, 0);
    let teachings = move_to_hand(&mut s, "Mystical Teachings");
    cast_card(&mut s, teachings);
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);

    let names: Vec<_> = offered_cards(&s)
        .into_iter()
        .map(|id| s.game().state[id].name.clone())
        .collect();
    assert!(names.iter().any(|n| n == "Lightning Bolt"));
    assert!(names.iter().any(|n| n == "High Fae Trickster"));
    assert!(!names.iter().any(|n| n == "Gray Ogre"));
    assert!(!names.iter().any(|n| n == "Seismic Rupture"));

    let trickster = offered_cards(&s)
        .into_iter()
        .find(|id| s.game().state[*id].name == "High Fae Trickster")
        .unwrap();
    select_card(&mut s, trickster);
    settle(&mut s);
    assert_eq!(zone_of(&s, trickster), Some(ZoneType::Hand));
    assert_eq!(zone_of(&s, teachings), Some(ZoneType::Graveyard));

    // Flashback from graveyard for {5}{B}, then exiled.
    let gy_teachings = crate::support::inspect::now(&s, teachings);
    cast_card(&mut s, gy_teachings);
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let bolt = offered_cards(&s)
        .into_iter()
        .find(|id| s.game().state[*id].name == "Lightning Bolt")
        .unwrap();
    select_card(&mut s, bolt);
    settle(&mut s);
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Hand));
    assert_eq!(zone_of(&s, gy_teachings), Some(ZoneType::Exile));
}
