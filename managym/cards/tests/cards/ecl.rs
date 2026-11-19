//! Tests for cards filed in `src/ecl.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, force_new_card_in_hand},
    inspect::{can_cast, card_of, has_action, tapped, top_spell},
    play::{cast_card, choose, pass_until, settle, tap_for_mana},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
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
    let labels: Vec<_> = s
        .game()
        .semantic_decision_frame()
        .unwrap()
        .offers
        .iter()
        .map(|offer| offer.label.clone())
        .collect();
    assert!(
        labels.contains(&"Tap Savannah Lions to pay {W} (convoke)".to_string())
            && labels.contains(&"Tap Gray Ogre to pay {1} (convoke)".to_string()),
        "{labels:?}"
    );

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

/// Only a white creature (or white mana) pays the {W}: red creatures alone
/// do not cast it. With a Plains, the automatic payer convokes just enough.
#[test]
fn appeal_to_eirdu_needs_white_and_auto_convokes_only_what_mana_cannot_pay() {
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

    let plains = s.force_permanent_on_battlefield(0, "Plains");
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, appeal));
    cast_card(&mut s, appeal);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogres[0])));
    if s.action_space().kind == ActionSpaceKind::ChooseTarget {
        choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    }
    let spell = top_spell(&s);
    assert_eq!(spell.spent.total(), 1);
    assert_eq!(
        spell.spent.mana()[0].source.unwrap().object,
        card_of(&s, plains)
    );
    assert_eq!(
        spell
            .spent
            .contributed(How::Tapped, Toward::Convoke)
            .count(),
        3
    );
    let untapped = ogres
        .iter()
        .chain(&giants)
        .filter(|creature| !tapped(&s, **creature))
        .count();
    assert_eq!(untapped, 1);
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
