//! Tests for cards filed in `src/tdm.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, force_new_card_in_hand, plant_in_graveyard},
    inspect::{can_cast, card_of, now, offered_cards, top_spell},
    play::{cast_card, select_card, settle},
};
use managym::state::{
    game_object::PlayerId,
    mana::Color,
    payment::{How, Toward},
    player::ManaPayment,
    zone::ZoneType,
};

/// Teval gives every spell its controller casts delve. Each graveyard card
/// exiled pays {1} and never a colored pip; the receipt names the exiled
/// cards; Teval's life loss is the spell's mana value, not the mana paid.
#[test]
fn teval_grants_delve_and_takes_life_equal_to_mana_value() {
    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&[
            "Teval, Arbiter of Virtue",
            "Gray Ogre",
            "Hill Giant",
            "Savannah Lions",
            "Llanowar Elves",
        ]);
        s.force_permanent_on_battlefield(0, "Teval, Arbiter of Virtue");
        let mountain = s.force_permanent_on_battlefield(0, "Mountain");
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
        // {2}{R} with one Mountain: not castable until the graveyard can
        // pay the generic part.
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!can_cast(&s, ogre), "auto={auto}");
        let graveyard: Vec<_> = ["Hill Giant", "Savannah Lions", "Llanowar Elves"]
            .into_iter()
            .map(|name| {
                let card = plant_in_graveyard(&mut s, 0, name);
                now(&s, card)
            })
            .collect();
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(can_cast(&s, ogre), "auto={auto}");

        cast_card(&mut s, ogre);
        // Every graveyard card is offered, and paying is not: the
        // Mountain alone cannot cover the cost. Which cards to exile is
        // the player's pick under the automatic payer too.
        let mut offered = offered_cards(&s);
        offered.sort();
        let mut expected = graveyard.clone();
        expected.sort();
        assert_eq!(offered, expected);
        let frame = s.game().semantic_decision_frame().unwrap();
        assert!(
            frame
                .offers
                .iter()
                .any(|offer| offer.label.ends_with("to pay {1} (delve)")),
            "{:?}",
            frame.offers
        );
        select_card(&mut s, graveyard[0]);
        // A fork at the second delve choice: the card not chosen stays.
        for (chosen, kept) in [(1, 2), (2, 1)] {
            let fork = crate::support::play::fork_selecting(&s, graveyard[chosen]);
            assert_eq!(
                top_spell(&fork)
                    .spent
                    .contributed(How::Exiled, Toward::Delve)
                    .count(),
                2
            );
            let in_graveyard = |object| {
                fork.game()
                    .state
                    .zones
                    .contains(object, ZoneType::Graveyard, PlayerId(0))
            };
            assert!(in_graveyard(graveyard[kept]) && !in_graveyard(graveyard[chosen]));
        }
        select_card(&mut s, graveyard[2]);
        // With {R} left and one Mountain, the payment finishes itself.
        let spell = top_spell(&s);
        assert_eq!(spell.spent.total(), 1, "auto={auto}");
        assert_eq!(spell.spent.amounts()[Color::Red as usize], 1);
        assert_eq!(
            spell.spent.mana()[0].source.unwrap().object,
            card_of(&s, mountain)
        );
        let exiled: Vec<_> = spell
            .spent
            .contributed(How::Exiled, Toward::Delve)
            .collect();
        assert_eq!(exiled.len(), 2, "auto={auto}");
        assert_eq!(spell.spent.contributions().len(), 2);
        for object in &exiled {
            assert!(s
                .game()
                .state
                .zones
                .contains(*object, ZoneType::Exile, PlayerId(0)));
        }
        s.assert_zone_size(0, ZoneType::Graveyard, 1);
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert!(
            context
                .pending
                .iter()
                .any(|line| line.starts_with("Paid with: exiled ") && line.ends_with(" to delve.")),
            "{:?}",
            context.pending
        );
        settle(&mut s);
        // Mana value three, although one mana was paid.
        s.assert_life(0, 17);

        if auto {
            // With lands for the whole cost the automatic payer pays with
            // mana: nothing is asked and the last graveyard card stays.
            for _ in 0..3 {
                s.force_permanent_on_battlefield(0, "Mountain");
            }
            let second = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
            s.game_mut().scenario_refresh_priority().unwrap();
            cast_card(&mut s, second);
            let spell = top_spell(&s);
            assert_eq!(spell.spent.total(), 3);
            assert!(spell.spent.contributions().is_empty());
            s.assert_zone_size(0, ZoneType::Graveyard, 1);
        }
    }
}

/// Delve pays generic mana only: no number of graveyard cards casts a spell
/// whose colored mana the caster cannot produce.
#[test]
fn teval_delve_never_pays_a_colored_pip() {
    let mut s = five_color_unfunded_setup(&["Teval, Arbiter of Virtue", "Gray Ogre", "Hill Giant"]);
    s.force_permanent_on_battlefield(0, "Teval, Arbiter of Virtue");
    for _ in 0..4 {
        plant_in_graveyard(&mut s, 0, "Hill Giant");
    }
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, ogre));
}

/// A spell cast from the graveyard is not a card its own delve can exile
/// (CR 601.2a: it is already on the stack as costs are paid).
#[test]
fn teval_delve_never_exiles_the_spell_being_cast_from_the_graveyard() {
    let mut s = five_color_unfunded_setup(&[
        "Teval, Arbiter of Virtue",
        "Join the Dance",
        "Hill Giant",
        "Gray Ogre",
    ]);
    s.force_permanent_on_battlefield(0, "Teval, Arbiter of Virtue");
    s.force_permanent_on_battlefield(0, "Forest");
    s.force_permanent_on_battlefield(0, "Plains");
    let dance = plant_in_graveyard(&mut s, 0, "Join the Dance");
    plant_in_graveyard(&mut s, 0, "Hill Giant");
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    // Flashback {3}{G}{W} with two lands and two other graveyard cards: one
    // generic short.
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, dance));

    let third = plant_in_graveyard(&mut s, 0, "Hill Giant");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, dance));
    cast_card(&mut s, dance);
    let offered = offered_cards(&s);
    assert_eq!(offered.len(), 3, "{offered:?}");
    assert!(!offered.contains(&now(&s, dance)));
    assert!(offered.contains(&now(&s, third)));
}
