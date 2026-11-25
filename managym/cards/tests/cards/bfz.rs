//! Tests for cards filed in `src/bfz.rs`, in card order.

/// Felidar Cub is sacrificed, with no mana, to destroy an enchantment; an
/// artifact is no target.
#[test]
fn felidar_cub_sacrifices_itself_to_destroy_an_enchantment() {
    use crate::support::{
        game::five_color_unfunded_setup,
        inspect::{card_of, has_action, zone_of},
        play::{activate, settle},
    };
    use managym::{
        agent::action::Action,
        state::{game_object::Target, zone::ZoneType},
    };
    let mut s = five_color_unfunded_setup(&["Felidar Cub", "Vampiric Rites", "Sol Ring"]);
    let cub = s.force_permanent_on_battlefield(0, "Felidar Cub");
    let cub_card = card_of(&s, cub);
    let rites = s.force_permanent_on_battlefield(1, "Vampiric Rites");
    let rites_card = card_of(&s, rites);
    let ring = s.force_permanent_on_battlefield(1, "Sol Ring");
    activate(&mut s, cub, 0);
    assert!(!has_action(&s, |a| {
        matches!(a, Action::ChooseTarget { target: Target::Permanent(id), .. } if *id == ring)
    }));
    assert!(s.choose_target(Target::Permanent(rites)));
    assert_eq!(zone_of(&s, cub_card), Some(ZoneType::Graveyard));
    settle(&mut s);
    assert_eq!(zone_of(&s, rites_card), Some(ZoneType::Graveyard));
    assert!(s.game().state.permanents[ring].is_some());
}

/// Hedron Archive taps for {C}{C}, or pays {2}, its tap and itself to draw two
/// cards; its own mana cannot pay for the ability that taps it.
#[test]
fn hedron_archive_taps_for_two_colorless_or_is_sacrificed_to_draw_two() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::{card_of, has_action, tapped, top_spell, zone_of},
        play::{activate, cast_card, settle},
    };
    use managym::{
        agent::action::Action,
        state::{player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_unfunded_setup(&["Hedron Archive", "Prophetic Prism"]);
    let archive = s.force_permanent_on_battlefield(0, "Hedron Archive");
    let archive_card = card_of(&s, archive);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!has_action(&s, |a| {
        matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == archive)
    }));
    let prism = force_new_card_in_hand(&mut s, 0, "Prophetic Prism");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_card(&mut s, prism);
    assert_eq!(top_spell(&s).spent.total(), 2);
    assert!(tapped(&s, archive));
    settle(&mut s);

    s.game_mut().state.permanents[archive]
        .as_mut()
        .unwrap()
        .tapped = false;
    force_lands(&mut s, 0, "Mountain", 2);
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, archive, 0);
    assert_eq!(zone_of(&s, archive_card), Some(ZoneType::Graveyard));
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 2);
}

/// Vampiric Rites pays {1}{B} and a sacrificed creature of the player's choice
/// for 1 life and a card; with no creature it cannot be activated.
#[test]
fn vampiric_rites_sacrifices_a_chosen_creature_for_a_life_and_a_card() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands},
        inspect::{card_of, has_action, zone_of},
        play::{activate, select_card, settle},
    };
    use managym::{
        agent::action::Action,
        state::{player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_unfunded_setup(&["Vampiric Rites", "Gray Ogre", "Hill Giant"]);
    let rites = s.force_permanent_on_battlefield(0, "Vampiric Rites");
    force_lands(&mut s, 0, "Swamp", 4);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.game_mut().scenario_refresh_priority().unwrap();
    let offered = |s: &crate::support::game::Scenario| {
        has_action(
            s,
            |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == rites),
        )
    };
    assert!(!offered(&s), "no creature to sacrifice");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let giant_card = card_of(&s, giant);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(offered(&s));
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, rites, 0);
    select_card(&mut s, giant_card);
    assert_eq!(zone_of(&s, giant_card), Some(ZoneType::Graveyard));
    settle(&mut s);
    s.assert_life(0, 21);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
    assert!(s.game().state.permanents[ogre].is_some());
    assert_eq!(s.untapped_permanents_named(0, "Swamp"), 2);
}

#[test]
fn foundations_5e_sure_strike() {
    use managym::state::card::Keyword;
    crate::assert_targeted_pump(
        "Sure Strike",
        "Gray Ogre",
        0,
        (5, 2),
        &[Keyword::FirstStrike],
        0,
        0,
    );
}
