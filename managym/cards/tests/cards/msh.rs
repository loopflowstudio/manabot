//! Tests for cards filed in `src/msh.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, force_new_card_in_hand},
    inspect::{can_cast, card_of, has_action, tapped, top_spell},
    play::{cast_card, choose, settle},
};
use managym::{
    agent::action::Action,
    state::{
        game_object::PlayerId,
        mana::Color,
        payment::{How, Toward},
        player::ManaPayment,
    },
};

/// Ironheart's improvise: each artifact tapped pays {1}, never the {U}. With
/// Ironheart on the battlefield a noncreature spell has improvise too, and a
/// creature spell does not.
#[test]
fn ironheart_improvises_with_artifacts_and_grants_it_to_noncreature_spells() {
    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&[
            "Ironheart, Clever Champion",
            "Glider Staff",
            "Dramatic Reversal",
            "Gray Ogre",
        ]);
        let staffs: Vec<_> = (0..4)
            .map(|_| s.force_permanent_on_battlefield(0, "Glider Staff"))
            .collect();
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let ironheart = force_new_card_in_hand(&mut s, 0, "Ironheart, Clever Champion");
        // Four artifacts pay {4}; nothing but blue mana pays the {U}.
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!can_cast(&s, ironheart), "auto={auto}");
        let island = s.force_permanent_on_battlefield(0, "Island");
        cast_card(&mut s, ironheart);
        // The Island pays only the {U}: which artifacts tap is the player's
        // pick, with the automatic payer as by hand.
        for staff in &staffs {
            assert!(has_action(&s, |a| {
                matches!(a, Action::WaterbendTap { permanent, .. } if permanent == staff)
            }));
        }
        assert!(!has_action(&s, |a| matches!(a, Action::PayCost { .. })));
        let frame = s.game().semantic_decision_frame().unwrap();
        assert!(
            frame
                .offers
                .iter()
                .any(|offer| offer.label == "Tap Glider Staff to pay {1} (improvise)"),
            "{:?}",
            frame.offers
        );
        // A fork at the first improvise tap: only the Staff chosen taps.
        for chosen in &staffs[..2] {
            let fork = crate::support::play::fork_choosing(
                &s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if permanent == chosen),
            );
            for staff in &staffs {
                assert_eq!(tapped(&fork, *staff), staff == chosen);
                assert!(!tapped(&s, *staff));
            }
        }
        for staff in &staffs {
            choose(
                &mut s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if permanent == staff),
            );
        }
        let spell = top_spell(&s);
        assert_eq!(spell.spent.total(), 1, "auto={auto}");
        assert_eq!(spell.spent.amounts()[Color::Blue as usize], 1);
        let mut improvised: Vec<_> = spell
            .spent
            .contributed(How::Tapped, Toward::Improvise)
            .collect();
        improvised.sort();
        let mut expected: Vec<_> = staffs.iter().map(|staff| card_of(&s, *staff)).collect();
        expected.sort();
        assert_eq!(improvised, expected, "auto={auto}");
        assert!(tapped(&s, island));
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert!(
            context
                .pending
                .iter()
                .any(|line| line.contains("tapped Glider Staff to improvise")),
            "{:?}",
            context.pending
        );
        settle(&mut s);

        // Ironheart is an artifact itself. It pays {1} of Dramatic Reversal
        // ({1}{U}) beside a second Island, and never helps a creature spell.
        let ironheart = crate::support::inspect::permanent_named(&s, "Ironheart, Clever Champion");
        let reversal = force_new_card_in_hand(&mut s, 0, "Dramatic Reversal");
        let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
        s.force_permanent_on_battlefield(0, "Mountain");
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!can_cast(&s, reversal), "no untapped blue source yet");
        assert!(
            !can_cast(&s, ogre),
            "a creature spell has no improvise: one Mountain for {{2}}{{R}}"
        );
        s.force_permanent_on_battlefield(0, "Island");
        cast_card(&mut s, reversal);
        if !auto {
            choose(
                &mut s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == ironheart),
            );
        }
        // The Mountain could pay the {1}. By hand the player improvises
        // instead; the automatic payer pays with mana and asks nothing.
        let spell = top_spell(&s);
        let improvised: Vec<_> = spell
            .spent
            .contributed(How::Tapped, Toward::Improvise)
            .collect();
        if auto {
            assert!(improvised.is_empty());
            assert_eq!(spell.spent.total(), 2);
        } else {
            assert_eq!(improvised, vec![card_of(&s, ironheart)]);
            assert_eq!(spell.spent.total(), 1);
        }
        // Reversal untaps its controller's nonland permanents as it resolves.
        settle(&mut s);
        assert!(!tapped(&s, ironheart));
    }
}

#[test]
fn quicksilver_power_up_discounts_only_normal_entry_and_is_spent_on_activation() {
    use crate::support::{game::enter_battlefield, play::activate};
    use managym::state::{card::Keyword, permanent::CounterKind};
    let name = "Quicksilver, Brash Blur";
    let mut s = five_color_unfunded_setup(&[name]);
    let permanent = enter_battlefield(&mut s, name);
    let view = s.game().characteristics(permanent);
    assert!(view.keywords.has(Keyword::Haste));
    assert_eq!(view.activated_abilities[0].mana_cost.mana_value, 4);
    s.set_player_mana_pool(0, "4");
    activate(&mut s, permanent, 0);
    settle(&mut s);
    let source = s.game().state.permanents[permanent].as_ref().unwrap();
    for kind in [CounterKind::PlusOnePlusOne, CounterKind::DoubleStrike] {
        assert_eq!(source.counter_count(kind), 1);
    }
    s.set_player_mana_pool(0, "4R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!has_action(
        &s,
        |action| matches!(action, Action::ActivateAbility { permanent: id,.. } if *id == permanent)
    ));
}
