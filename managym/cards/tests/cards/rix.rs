//! Tests for cards filed in `src/rix.rs`, in card order.

/// Fanatical Firebrand has haste, so it pays its own {T} the turn it arrives,
/// and is sacrificed to deal 1 damage.
#[test]
fn fanatical_firebrand_taps_and_sacrifices_itself_for_one_damage() {
    use crate::support::{
        game::five_color_unfunded_setup,
        inspect::{card_of, zone_of},
        play::{activate, settle},
    };
    use managym::state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    };
    let mut s = five_color_unfunded_setup(&["Fanatical Firebrand"]);
    let firebrand = s.force_permanent_on_battlefield(0, "Fanatical Firebrand");
    s.game_mut().state.permanents[firebrand]
        .as_mut()
        .unwrap()
        .summoning_sick = true;
    let firebrand_card = card_of(&s, firebrand);
    activate(&mut s, firebrand, 0);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    assert_eq!(zone_of(&s, firebrand_card), Some(ZoneType::Graveyard));
    settle(&mut s);
    s.assert_life(1, 19);
}

/// Thrashing Brontodon pays {1} and itself to destroy an artifact or
/// enchantment, never a creature.
#[test]
fn thrashing_brontodon_sacrifices_itself_to_destroy_an_artifact() {
    crate::support::play::sacrifices_itself_to_destroy_an_artifact("Thrashing Brontodon", "Forest");
}
