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
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands},
        inspect::{card_of, has_action, zone_of},
        play::{activate, settle},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::Target, player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_unfunded_setup(&["Thrashing Brontodon", "Sol Ring", "Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, "Thrashing Brontodon");
    let source_card = card_of(&s, source);
    force_lands(&mut s, 0, "Forest", 1);
    let target = s.force_permanent_on_battlefield(1, "Sol Ring");
    let target_card = card_of(&s, target);
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    activate(&mut s, source, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(!has_action(&s, |a| {
        matches!(a, Action::ChooseTarget { target: Target::Permanent(id), .. } if *id == other)
    }));
    assert!(s.choose_target(Target::Permanent(target)));
    assert_eq!(zone_of(&s, source_card), Some(ZoneType::Graveyard));
    assert_eq!(s.untapped_permanents_named(0, "Forest"), 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, target_card), Some(ZoneType::Graveyard));
    assert!(s.game().state.permanents[other].is_some());
}
