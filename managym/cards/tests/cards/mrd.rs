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
    crate::printed_creature_combat(
        "Leonin Skyhunter",
        2,
        2,
        managym::state::card::Keywords::of(&[Keyword::Flying]),
    );
}
