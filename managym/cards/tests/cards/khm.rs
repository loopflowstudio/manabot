//! Tests for cards filed in `src/khm.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::keywords,
    play::{activate, offers_ability, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{
        card::Keyword,
        game_object::{PlayerId, Target},
    },
};

fn coils(s: &crate::support::game::Scenario) -> usize {
    s.game()
        .state
        .permanents
        .iter()
        .flatten()
        .filter(|permanent| s.game().state[permanent.object].name == "Koma's Coil")
        .count()
}

/// Koma makes a Coil at each player's upkeep. Sacrificing one taps a target
/// whose other abilities then stay unusable until the turn ends; the other
/// mode makes Koma indestructible.
#[test]
fn koma_coils_each_upkeep_and_sacrifices_them_to_lock_or_protect() {
    let mut s = five_color_setup(&["Koma, Cosmos Serpent"]);
    let koma = s.force_permanent_on_battlefield(0, "Koma, Cosmos Serpent");
    let clue = s
        .game_mut()
        .create_token("token.clue", PlayerId(0), false)
        .unwrap();
    s.game_mut()
        .create_token("token.komas_coil", PlayerId(0), false);
    assert!(offers_ability(&mut s, clue, 0));

    activate(&mut s, koma, 0);
    assert!(s.choose_target(Target::Permanent(clue)));
    settle(&mut s);
    assert_eq!(coils(&s), 0);
    assert!(s.game().state.permanents[clue].as_ref().unwrap().tapped);
    assert!(s.game().activations_prohibited(clue));
    assert!(!offers_ability(&mut s, clue, 0));
    assert!(
        !offers_ability(&mut s, koma, 0),
        "no other Serpent to sacrifice"
    );

    s.game_mut()
        .create_token("token.komas_coil", PlayerId(0), false);
    activate(&mut s, koma, 1);
    settle(&mut s);
    assert!(keywords(&s, koma).has(Keyword::Indestructible));

    s.advance_to_active_step(1, StepKind::Draw);
    assert_eq!(coils(&s), 1, "the opponent's upkeep makes a Coil");
    assert!(!s.game().activations_prohibited(clue));
    assert!(!keywords(&s, koma).has(Keyword::Indestructible));
}

/// Seize the Spoils discards a chosen card as it is cast, then draws two and
/// makes a Treasure.
#[test]
fn seize_the_spoils_discards_a_chosen_card_then_draws_two_and_makes_a_treasure() {
    use managym::state::zone::ZoneType;
    for auto in [false, true] {
        let s = crate::support::play::cast_discarding_a_chosen_card("Seize the Spoils", 3, auto);
        // The Ogre and two drawn cards.
        assert_eq!(s.zone_size(0, ZoneType::Hand), 3);
        assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);
    }
}

#[test]
fn foundations_spell_snakeskin_veil() {
    use crate::support::game::five_color_setup;
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Snakeskin Veil", "Hill Giant"]);
    let subject = s.force_permanent_on_battlefield(0, "Hill Giant");
    cast(&mut s, "Snakeskin Veil");
    assert!(s.choose_target(Target::Permanent(subject)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(subject).pt(), (4, 4));
    assert!(s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Hexproof));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(subject).pt(), (4, 4));
    assert!(!s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Hexproof));
}
