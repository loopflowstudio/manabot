//! Tests for cards filed in `src/ice.rs`, in card order.

use crate::support::{
    game::{deck, deck_of, five_color_setup, force_lands, Scenario},
    inspect::{counters_of, hand},
    play::{cast, choose, choose_mode, pass_until, settle, spend_colorless},
};
use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::permanent::CounterKind,
};

/// Pyroclasm deals 2 damage to each creature on both sides; creatures with
/// toughness <= 2 die via state-based actions, tougher ones survive.
#[test]
fn pyroclasm_kills_small_creatures_on_both_sides() {
    let mut s = Scenario::new(
        deck(&[
            ("Mountain", 20),
            ("Pyroclasm", 8),
            ("Gray Ogre", 6),
            ("Wall of Stone", 6),
        ]),
        deck_of("Mountain", "Gray Ogre"),
        1213,
    );

    // Two mountains over two turns to afford 1R.
    s.advance_to_active_step(0, StepKind::Main);
    s.force_cards_in_hand(0, "Mountain", 2);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    // Board: 2/2 ogres on both sides, one 0/8 wall for player 0.
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Wall of Stone");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre").len(), 1);
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);

    s.force_card_in_hand(0, "Pyroclasm");
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    // Both ogres (toughness 2) die; the wall (toughness 8) survives.
    assert!(s.battlefield_permanents_named(0, "Gray Ogre").is_empty());
    assert!(s.battlefield_permanents_named(1, "Gray Ogre").is_empty());
    assert_eq!(s.battlefield_permanents_named(0, "Wall of Stone").len(), 1);
}

/// The caster of a noncreature spell may pay {4}; unpaid, Remora's
/// controller may draw. A creature spell doesn't trigger it.
#[test]
fn mystic_remora_draws_unless_the_caster_pays_four() {
    for pay in [false, true] {
        let mut s = five_color_setup(&["Mystic Remora", "Pyroclasm", "Gray Ogre"]);
        s.force_permanent_on_battlefield(1, "Mystic Remora");
        let before = hand(&s, 1).len();
        cast(&mut s, "Gray Ogre");
        settle(&mut s);
        assert_eq!(hand(&s, 1).len(), before);
        cast(&mut s, "Pyroclasm");
        spend_colorless(&mut s);
        pass_until(&mut s, ActionSpaceKind::PayOrNot);
        assert_eq!(s.action_space().player.unwrap().0, 0, "the caster decides");
        let pool = s.game().state.players[0].mana_pool.total();
        if pay {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
            spend_colorless(&mut s);
            assert_eq!(s.game().state.players[0].mana_pool.total(), pool - 4);
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
            assert_eq!(s.action_space().player.unwrap().0, 1);
            choose_mode(&mut s, 0);
        }
        settle(&mut s);
        assert_eq!(hand(&s, 1).len(), before + usize::from(!pay));
    }
}

/// Each upkeep adds an age counter and asks {1} per counter: paid once, then
/// declined, which sacrifices it.
#[test]
fn mystic_remora_cumulative_upkeep_grows_until_it_is_sacrificed() {
    let mut s = five_color_setup(&["Mystic Remora"]);
    let remora = s.force_permanent_on_battlefield(0, "Mystic Remora");
    force_lands(&mut s, 0, "Island", 2);
    s.advance_to_active_step(1, StepKind::Upkeep);
    s.advance_to_active_step(0, StepKind::Upkeep);
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    assert_eq!(counters_of(&s, remora, CounterKind::Age), 1);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(s.game().state.permanents[remora].is_some());
    s.advance_to_active_step(1, StepKind::Upkeep);
    s.advance_to_active_step(0, StepKind::Upkeep);
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    assert_eq!(counters_of(&s, remora, CounterKind::Age), 2);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert!(s.game().state.permanents[remora].is_none());
}
