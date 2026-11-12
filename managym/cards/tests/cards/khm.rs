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
        .filter(|permanent| s.game().state.cards[permanent.card].name == "Koma's Coil")
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
