//! Tests for cards filed in `src/mrd.rs`, in card order.

/// Gilded Lotus adds three mana of one chosen color: alone it casts a spell
/// with two pips of one color, and never one that needs three colors.
#[test]
fn gilded_lotus_adds_three_mana_of_one_color() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::tapped,
        play::{cast_card, castable, settle},
    };
    use managym::state::player::ManaPayment;
    let mut s =
        five_color_unfunded_setup(&["Gilded Lotus", "Wall of Stone", "Teval, Arbiter of Virtue"]);
    let lotus = s.force_permanent_on_battlefield(0, "Gilded Lotus");
    force_lands(&mut s, 0, "Plains", 1);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    // {1}{R}{R} from the Lotus alone; {2}{B}{G}{U} needs three colors.
    let wall = force_new_card_in_hand(&mut s, 0, "Wall of Stone");
    let teval = force_new_card_in_hand(&mut s, 0, "Teval, Arbiter of Virtue");
    assert!(castable(&mut s, wall));
    assert!(!castable(&mut s, teval));
    cast_card(&mut s, wall);
    settle(&mut s);
    assert!(tapped(&s, lotus));
    assert_eq!(s.battlefield_permanents_named(0, "Wall of Stone").len(), 1);
    assert_eq!(s.untapped_permanents_named(0, "Plains"), 1);
}

#[test]
fn printed_creature_leonin_skyhunter() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat("Leonin Skyhunter", 2, 2, &[Keyword::Flying]);
}

#[test]
fn foundations_grant_fireshrieker() {
    use managym::state::card::Keyword;
    crate::check_attachment_grant("Fireshrieker", Some("2"), (2, 2), &[Keyword::DoubleStrike]);
}

#[test]
fn foundations_5d_solemn_simulacrum() {
    use crate::support::game::{five_color_setup, move_card, put_onto_battlefield};
    use crate::support::inspect::{
        card_of, library, library_card, library_top, offered_cards, permanent_of, tapped, zone_of,
    };
    use crate::support::play::{choose_mode, pass_until, select_card, settle};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::PlayerId;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Solemn Simulacrum", "Gray Ogre"]);
    let golem = put_onto_battlefield(&mut s, 0, "Solemn Simulacrum");
    pass_until(&mut s, ActionSpaceKind::Modal);
    let order = library(&s, 0);
    let mut decline = s.clone();
    choose_mode(&mut decline, 1);
    settle(&mut decline);
    assert_eq!(library(&decline, 0), order);
    choose_mode(&mut s, 0);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    let forest = library_card(&s, 0, "Forest");
    let ogre = library_card(&s, 0, "Gray Ogre");
    assert!(!offered_cards(&s).contains(&ogre));
    select_card(&mut s, forest);
    settle(&mut s);
    assert!(tapped(&s, permanent_of(&s, forest).unwrap()));
    let card = card_of(&s, golem);
    move_card(&mut s, card, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::Modal);
    let top = library_top(&s, 0);
    let mut keep = s.clone();
    choose_mode(&mut keep, 1);
    settle(&mut keep);
    assert_eq!(library_top(&keep, 0), top);
    choose_mode(&mut s, 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, top), Some(ZoneType::Hand));
}
