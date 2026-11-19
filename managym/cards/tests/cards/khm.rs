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
    use crate::support::{
        game::{clear_hand, five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::{has_action, offered_cards, top_spell, zone_of},
        play::{cast_card, castable, fork_selecting, select_card, settle},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{
            payment::{How, Toward},
            player::ManaPayment,
            zone::ZoneType,
        },
    };
    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&["Seize the Spoils", "Gray Ogre", "Hill Giant"]);
        force_lands(&mut s, 0, "Mountain", 3);
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        clear_hand(&mut s, 0);
        let spell = force_new_card_in_hand(&mut s, 0, "Seize the Spoils");
        assert!(!castable(&mut s, spell), "no other card to discard");
        let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
        let giant = force_new_card_in_hand(&mut s, 0, "Hill Giant");
        assert!(castable(&mut s, spell));
        cast_card(&mut s, spell);
        // The discard is the player's pick under either payment setting,
        // and never the spell being cast.
        assert_eq!(
            s.action_space().kind,
            ActionSpaceKind::Discard,
            "auto={auto}"
        );
        let mut offered = offered_cards(&s);
        offered.sort();
        let mut expected = vec![ogre, giant];
        expected.sort();
        assert_eq!(offered, expected);
        assert!(!has_action(&s, |a| matches!(a, Action::PayCost { .. })));
        let fork = fork_selecting(&s, ogre);
        assert_eq!(zone_of(&fork, ogre), Some(ZoneType::Graveyard));
        assert_eq!(zone_of(&fork, giant), Some(ZoneType::Hand));
        // The Mountains pay the rest without a further choice.
        select_card(&mut s, giant);
        assert_eq!(zone_of(&s, giant), Some(ZoneType::Graveyard));
        assert_eq!(zone_of(&s, ogre), Some(ZoneType::Hand));
        let cast = top_spell(&s);
        assert_eq!(cast.spent.total(), 3);
        assert_eq!(
            cast.spent
                .contributed(How::Discarded, Toward::Printed)
                .count(),
            1
        );
        settle(&mut s);
        // The Ogre and two drawn cards.
        assert_eq!(s.zone_size(0, ZoneType::Hand), 3);
        assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);
    }
}
