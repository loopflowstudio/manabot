//! Tests for cards filed in `src/ecl.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, force_new_card_in_hand},
    inspect::{
        assert_choice_contract, can_cast, card_of, has_action, object_public_id, offer_labels,
        tap_pays_choice, tapped, top_spell,
    },
    play::{cast_card, choose, pass_until, settle, tap_for_mana},
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::Observation,
    },
    state::{
        game_object::{PermanentId, PlayerId, Target},
        mana::Color,
        payment::{How, Toward},
        player::ManaPayment,
    },
};

fn convokes(s: &crate::support::game::Scenario, creature: PermanentId) -> bool {
    has_action(
        s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == creature),
    )
}

/// Cast Appeal to Eirdu targeting `target` alone, up to its payment.
fn cast_appeal(s: &mut crate::support::game::Scenario, target: PermanentId) {
    let appeal = force_new_card_in_hand(s, 0, "Appeal to Eirdu");
    cast_card(s, appeal);
    pass_until(s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(target)));
    if s.action_space().kind == ActionSpaceKind::ChooseTarget {
        // One target is enough: decline the second.
        choose(s, |a| matches!(a, Action::Decline { .. }));
    }
}

/// Appeal to Eirdu ({3}{W}) with no lands: a white creature pays the {W},
/// any creature pays {1}, and a creature tapped for mana cannot also convoke.
/// With no white mana floating the Lions can only pay the {W}, so no tap-pays
/// choice is published here; the next test checks that choice's modes, focus
/// and Commands.
#[test]
fn appeal_to_eirdu_convokes_for_generic_and_for_white() {
    let mut s = five_color_unfunded_setup(&[
        "Appeal to Eirdu",
        "Savannah Lions",
        "Gray Ogre",
        "Hill Giant",
        "Llanowar Elves",
    ]);
    let lions = s.force_permanent_on_battlefield(0, "Savannah Lions");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    crate::support::game::clear_summoning_sickness(&mut s, elves);
    cast_appeal(&mut s, ogre);

    // Every creature may convoke; mana alone cannot pay.
    for creature in [lions, ogre, giant, elves] {
        assert!(convokes(&s, creature));
    }
    assert!(!has_action(&s, |a| matches!(a, Action::PayCost { .. })));
    let labels = offer_labels(&s);
    assert!(
        labels.contains(&"Tap Savannah Lions to pay {W} (convoke)".to_string())
            && labels.contains(&"Tap Gray Ogre to pay {1} (convoke)".to_string()),
        "{labels:?}"
    );

    // The Elves has two taps of one action type: the shared projection says
    // which pays the cost itself.
    let events = s.drain_events();
    let obs = Observation::new(s.game(), &events);
    let mut taps: Vec<_> = obs
        .action_space
        .actions
        .iter()
        .filter(|action| action.action_type == ActionType::TapForCost)
        .map(|action| action.pays_directly)
        .collect();
    taps.sort();
    assert_eq!(taps, [false, true, true, true, true]);

    // The Elves taps for {G} instead: it then cannot convoke as well.
    tap_for_mana(&mut s, elves);
    assert!(tapped(&s, elves));
    // A fork at the first convoke tap: only the creature chosen is tapped.
    for chosen in [lions, ogre, giant] {
        let fork = crate::support::play::fork_choosing(
            &s,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == chosen),
        );
        for creature in [lions, ogre, giant] {
            assert_eq!(tapped(&fork, creature), creature == chosen);
            assert!(!tapped(&s, creature));
        }
    }
    for creature in [lions, ogre, giant] {
        choose(
            &mut s,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == creature),
        );
        if creature == lions {
            // The committed list is in the order paid: the Elves's {G},
            // then the Lions.
            let paid: Vec<_> = s
                .game()
                .decision_context(PlayerId(0))
                .unwrap()
                .paid
                .into_iter()
                .map(|paid| paid.phrase)
                .collect();
            assert!(
                matches!(&paid[..], [elves, lions] if elves.contains("Llanowar Elves") && lions.contains("Savannah Lions")),
                "{paid:?}"
            );
        }
    }
    let spell = top_spell(&s);
    assert_eq!(spell.spent.total(), 1);
    assert_eq!(spell.spent.amounts()[Color::Green as usize], 1);
    assert_eq!(
        spell
            .spent
            .contributed(How::Tapped, Toward::Convoke)
            .collect::<Vec<_>>(),
        [lions, ogre, giant].map(|creature| card_of(&s, creature))
    );
    let context = s.game().decision_context(PlayerId(0)).unwrap();
    assert!(
        context
            .pending
            .iter()
            .any(|line| line.contains("tapped Savannah Lions to convoke")),
        "{:?}",
        context.pending
    );
    settle(&mut s);
    let view = s.game().characteristics(ogre);
    assert_eq!((view.power, view.toughness), (4, 3));
}

/// A white creature convoking for {3}{W} may pay the {W} or {1}: with {W}
/// already floating, both leave the rest payable, so the payer chooses. Each
/// choice is a native Command of its own, and the one taken is what the
/// receipt keeps.
#[test]
fn appeal_to_eirdu_can_convoke_a_white_creature_for_generic() {
    let mut s = five_color_unfunded_setup(&["Appeal to Eirdu", "Savannah Lions", "Gray Ogre"]);
    s.allow_priority_floats();
    let lions = s.force_permanent_on_battlefield(0, "Savannah Lions");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Forest");
    s.set_player_mana_pool(0, "W");
    cast_appeal(&mut s, ogre);
    assert!(convokes(&s, lions));
    assert!(
        offer_labels(&s).contains(&"Tap Savannah Lions to pay {W} or {1} (convoke)".to_string()),
        "{:?}",
        offer_labels(&s)
    );
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == lions),
    );
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    assert_eq!(
        offer_labels(&s),
        [
            "Savannah Lions pays {W} (convoke)",
            "Savannah Lions pays {1} (convoke)"
        ]
    );
    // The modes name the payments: {W} by convoke is 0, {1} by convoke 5.
    let choice = tap_pays_choice(&s);
    assert_eq!(choice.focus, [object_public_id(&s, lions)]);
    assert_eq!(choice.modes, [0, 5]);
    assert_choice_contract(&s);
    for (mode, white_spent) in [(0, 0), (5, 1)] {
        let mut fork = crate::support::play::fork_choosing(
            &s,
            |a| matches!(a, Action::ChooseMode { mode: m, .. } if *m == mode),
        );
        for creature in [ogre, other] {
            choose(
                &mut fork,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == creature),
            );
        }
        crate::support::play::pay_rest(&mut fork);
        let spell = top_spell(&fork);
        // Paid {1}, the Lions leave the {W} to the floating white mana.
        assert_eq!(spell.spent.total(), 1);
        assert!(spell.spent.amounts()[Color::White as usize] >= white_spent);
        assert_eq!(
            spell
                .spent
                .contributed(How::Tapped, Toward::Convoke)
                .collect::<Vec<_>>(),
            [lions, ogre, other].map(|creature| card_of(&fork, creature))
        );
    }
}

/// Only a white creature (or white mana) pays the {W}: red creatures alone
/// do not cast it. The automatic payer pays mana with mana: short of mana,
/// the player picks each creature to tap and the payer finishes once the
/// lands cover the rest; with enough lands nothing is asked.
#[test]
fn appeal_to_eirdu_needs_white_and_auto_leaves_convoke_to_the_player() {
    for lands in [1, 4] {
        let mut s = five_color_unfunded_setup(&["Appeal to Eirdu", "Gray Ogre", "Hill Giant"]);
        let ogres = [
            s.force_permanent_on_battlefield(0, "Gray Ogre"),
            s.force_permanent_on_battlefield(0, "Gray Ogre"),
        ];
        let giants = [
            s.force_permanent_on_battlefield(0, "Hill Giant"),
            s.force_permanent_on_battlefield(0, "Hill Giant"),
        ];
        let appeal = force_new_card_in_hand(&mut s, 0, "Appeal to Eirdu");
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!can_cast(&s, appeal), "four red creatures pay no {{W}}");

        let plains: Vec<_> = (0..lands)
            .map(|_| s.force_permanent_on_battlefield(0, "Plains"))
            .collect();
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(can_cast(&s, appeal));
        cast_card(&mut s, appeal);
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(s.choose_target(Target::Permanent(ogres[0])));
        if s.action_space().kind == ActionSpaceKind::ChooseTarget {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        let picked = [ogres[1], giants[0], giants[1]];
        if lands == 1 {
            // One Plains pays the {W}; three of the four creatures must
            // convoke, and which three is the player's pick.
            for creature in picked {
                assert!(!has_action(&s, |a| matches!(a, Action::PayCost { .. })));
                assert!(convokes(&s, ogres[0]));
                choose(
                    &mut s,
                    |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == creature),
                );
            }
        }
        let spell = top_spell(&s);
        assert_eq!(spell.spent.total(), lands);
        for unit in spell.spent.mana() {
            let source = unit.source.unwrap().object;
            assert!(plains.iter().any(|land| card_of(&s, *land) == source));
        }
        assert_eq!(
            spell
                .spent
                .contributed(How::Tapped, Toward::Convoke)
                .collect::<Vec<_>>(),
            if lands == 1 {
                picked.map(|creature| card_of(&s, creature)).to_vec()
            } else {
                Vec::new()
            }
        );
        assert!(!tapped(&s, ogres[0]));
        for creature in picked {
            assert_eq!(tapped(&s, creature), lands == 1);
        }
    }
}

/// Convoke taps creatures as the layers see them: an earthbent land is a
/// creature and may convoke.
#[test]
fn appeal_to_eirdu_convokes_with_a_land_that_became_a_creature() {
    let mut s = five_color_unfunded_setup(&[
        "Appeal to Eirdu",
        "Earthbending Lesson",
        "Gray Ogre",
        "Hill Giant",
    ]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.force_permanent_on_battlefield(0, "Plains");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Hill Giant");
    let lesson = force_new_card_in_hand(&mut s, 0, "Earthbending Lesson");
    s.set_player_mana_pool(0, "20WWWWGGGG");
    cast_card(&mut s, lesson);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert!(s.game().characteristics(forest).is_creature());
    s.set_player_mana_pool(0, "0");
    cast_appeal(&mut s, ogre);
    assert!(convokes(&s, forest));
    assert!(convokes(&s, ogre));
}
