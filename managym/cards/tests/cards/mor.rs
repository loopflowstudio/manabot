//! Morningtide card behavior.

#[test]
fn foundations_spell_negate() {
    crate::check_counterspell("Negate", "Sol Ring", "Gray Ogre");
}

#[test]
fn foundations_5e_kindled_fury() {
    use managym::state::card::Keyword;
    crate::assert_targeted_pump(
        "Kindled Fury",
        "Gray Ogre",
        0,
        (3, 2),
        &[Keyword::FirstStrike],
        0,
        0,
    );
}

#[test]
fn ballyrush_banneret_reduces_kithkin_or_soldier_once() {
    use crate::support::game::{five_color_unfunded_setup, move_to_hand};
    use crate::support::inspect::can_cast;
    use crate::support::play::{cast_card, settle};

    let mut s = five_color_unfunded_setup(&["Ballyrush Banneret", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Ballyrush Banneret");
    // Second Ballyrush Banneret is both Kithkin and Soldier ({1}{W}): costs {W}, not free!
    let second = move_to_hand(&mut s, "Ballyrush Banneret");
    let ogre = move_to_hand(&mut s, "Gray Ogre");

    s.set_player_mana_pool(0, "");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, second), "Kithkin Soldier is not free");

    s.set_player_mana_pool(0, "W");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, second), "Kithkin Soldier costs 1 less");
    assert!(!can_cast(&s, ogre), "Gray Ogre gets no discount");

    cast_card(&mut s, second);
    settle(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(0, "Ballyrush Banneret")
            .len(),
        2
    );
}
