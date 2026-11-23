//! Tests for cards filed in `src/ori.rs`, in card order.

use crate::support::{game::five_color_setup, play::enter_and_settle};
use managym::state::zone::ZoneType;

#[test]
fn elemental_bond_draws_for_a_creature_with_power_three_or_more() {
    let mut s = five_color_setup(&["Elemental Bond", "Gray Ogre", "Craw Wurm"]);
    s.force_permanent_on_battlefield(0, "Elemental Bond");
    s.game_mut().scenario_refresh_priority().unwrap();
    let hand = s.zone_size(0, ZoneType::Hand);
    enter_and_settle(&mut s, "Gray Ogre");
    s.assert_zone_size(0, ZoneType::Hand, hand);
    enter_and_settle(&mut s, "Craw Wurm");
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
}

/// Pyromancer's Goggles copies a red instant its mana paid for, once, with
/// the copy trigger above the spell; mana tapped at priority first carries
/// the rider through the pool.
#[test]
fn pyromancers_goggles_copies_a_red_instant_its_mana_paid_for() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::card_of,
        play::{cast_card, pay_rest, settle, tap_for_mana, target_opponent},
    };
    use managym::state::{
        game_object::PlayerId, mana::SpendEffect, player::ManaPayment, stack_object::StackObject,
    };

    for (auto, floated) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut s = five_color_unfunded_setup(&["Pyromancer's Goggles", "Lightning Bolt"]);
        let goggles = s.force_permanent_on_battlefield(0, "Pyromancer's Goggles");
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
        if floated {
            s.allow_priority_floats();
            tap_for_mana(&mut s, goggles);
        }
        cast_card(&mut s, bolt);
        target_opponent(&mut s);
        if !auto && !floated {
            tap_for_mana(&mut s, goggles);
        }
        if s.game().state.stack_objects.is_empty() {
            pay_rest(&mut s);
        }
        let case = format!("auto={auto} floated={floated}");
        // The trigger is the Goggles's, above the spell it will copy.
        let [StackObject::Spell(spell), StackObject::TriggeredAbility(trigger)] =
            &s.game().state.stack_objects[..]
        else {
            panic!("{case}: {:?}", s.game().state.stack_objects)
        };
        let [unit] = spell.spent.mana() else {
            panic!("one mana was spent, {case}")
        };
        assert_eq!(unit.source.unwrap().object, card_of(&s, goggles), "{case}");
        assert!(matches!(unit.effect, Some(SpendEffect::CopySpell { .. })));
        assert_eq!(trigger.controller, PlayerId(0));
        assert_eq!(
            trigger.source_identity.object(),
            card_of(&s, goggles),
            "{case}"
        );
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert!(
            context
                .pending
                .iter()
                .any(|line| line.contains("that mana copies it 1 time(s)")),
            "{case}: {:?}",
            context.pending
        );
        // The copy keeps its target unless a new one is chosen.
        settle(&mut s);
        s.assert_life(1, 14);
    }
}

#[test]
fn pyromancers_goggles_does_not_copy_other_spells_or_without_its_mana() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::{card_of, tapped, top_spell},
        play::{cast_card, pay_rest, settle_targeting_opponent, tap_for_mana, target_opponent},
    };
    use managym::state::player::ManaPayment;

    for auto in [false, true] {
        // A blue instant paid for with the Goggles's red: no trigger.
        let mut s = five_color_unfunded_setup(&["Pyromancer's Goggles", "Dramatic Reversal"]);
        let goggles = s.force_permanent_on_battlefield(0, "Pyromancer's Goggles");
        s.force_permanent_on_battlefield(0, "Island");
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let reversal = force_new_card_in_hand(&mut s, 0, "Dramatic Reversal");
        cast_card(&mut s, reversal);
        if !auto {
            tap_for_mana(&mut s, goggles);
            if s.game().state.stack_objects.is_empty() {
                pay_rest(&mut s);
            }
        }
        assert!(tapped(&s, goggles), "auto={auto}");
        assert_eq!(s.game().state.stack_objects.len(), 1, "auto={auto}");
        assert!(top_spell(&s)
            .spent
            .mana()
            .iter()
            .all(|unit| unit.effect.is_none()));
    }

    // The Goggles's unit floats; a Mountain pays for the Bolt instead. The
    // unit stays in the pool and nothing is copied.
    let mut s = five_color_unfunded_setup(&["Pyromancer's Goggles", "Lightning Bolt"]);
    let goggles = s.force_permanent_on_battlefield(0, "Pyromancer's Goggles");
    let mountain = s.force_permanent_on_battlefield(0, "Mountain");
    let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
    s.allow_priority_floats();
    tap_for_mana(&mut s, goggles);
    cast_card(&mut s, bolt);
    target_opponent(&mut s);
    tap_for_mana(&mut s, mountain);
    while s.game().state.stack_objects.is_empty() {
        let frame = s.game().semantic_decision_frame().unwrap();
        let plain = frame
            .offers
            .iter()
            .position(|offer| offer.label.starts_with("Spend") && !offer.label.contains("copy"))
            .unwrap_or_else(|| panic!("the Mountain's red is offered: {:?}", frame.offers));
        s.step_action(plain);
    }
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(
        top_spell(&s).spent.mana()[0].source.unwrap().object,
        card_of(&s, mountain)
    );
    assert_eq!(s.game().state.players[0].mana_pool.total(), 1);
    settle_targeting_opponent(&mut s);
    s.assert_life(1, 17);

    // Both units float before the cast, or the Mountain's floats and the
    // Goggles are tapped while paying: either way the payment asks which to
    // spend, and the Goggles's unit is not committed as it is made. A fork
    // at that choice: the Goggles's unit copies, the Mountain's does not.
    for goggles_first in [true, false] {
        let mut s = five_color_unfunded_setup(&["Pyromancer's Goggles", "Lightning Bolt"]);
        let goggles = s.force_permanent_on_battlefield(0, "Pyromancer's Goggles");
        let mountain = s.force_permanent_on_battlefield(0, "Mountain");
        let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
        s.allow_priority_floats();
        if goggles_first {
            tap_for_mana(&mut s, goggles);
        }
        tap_for_mana(&mut s, mountain);
        cast_card(&mut s, bolt);
        target_opponent(&mut s);
        if !goggles_first {
            tap_for_mana(&mut s, goggles);
            // Neither unit was committed; with no source left the payment
            // goes on to spending.
        }
        let frame = s.game().semantic_decision_frame().unwrap();
        for (copies, source) in [(true, goggles), (false, mountain)] {
            let spend = frame
                .offers
                .iter()
                .position(|offer| {
                    offer.label.starts_with("Spend") && offer.label.contains("copy") == copies
                })
                .unwrap_or_else(|| panic!("both units are offered: {:?}", frame.offers));
            let fork = crate::support::play::fork_stepping(&s, spend);
            assert_eq!(
                fork.game().state.stack_objects.len(),
                1 + usize::from(copies)
            );
            assert_eq!(
                top_spell(&fork).spent.mana()[0].source.unwrap().object,
                card_of(&fork, source)
            );
            assert_eq!(fork.game().state.players[0].mana_pool.total(), 1);
        }
        assert!(s.game().state.stack_objects.is_empty());
    }
}

#[test]
fn demonic_pact_loss_can_be_answered_with_herald() {
    use crate::support::{
        game::force_new_card_in_hand,
        play::{cast_card, command, settle},
    };
    use managym::{agent::action::Action, flow::turn::StepKind};

    let mut s = five_color_setup(&["Demonic Pact", "Herald of Eternal Dawn"]);
    s.force_permanent_on_battlefield(0, "Demonic Pact");
    let herald = force_new_card_in_hand(&mut s, 0, "Herald of Eternal Dawn");
    s.advance_to_active_step(0, StepKind::Upkeep);
    s.set_player_mana_pool(0, "4WWW");
    assert_eq!(s.action_space().actions.len(), 4);
    let root = s.game().clone();
    for protected in [false, true] {
        *s.game_mut() = root.clone();
        command(&mut s, |action| {
            matches!(action, Action::ChooseMode { mode: 3, .. })
        });
        assert_eq!(s.game().state.stack_objects.len(), 1);
        if protected {
            cast_card(&mut s, herald);
            settle(&mut s);
            let permanent = s.battlefield_permanents_named(0, "Herald of Eternal Dawn")[0];
            let object = s.game().state.permanents[permanent]
                .as_ref()
                .unwrap()
                .object;
            s.game_mut().move_object(object, ZoneType::Exile);
            s.pass_priority();
            assert!(!s.game().is_game_over());
            s.advance_to_active_step(1, StepKind::Upkeep);
            s.advance_to_active_step(0, StepKind::Upkeep);
            assert_eq!(s.action_space().actions.len(), 3);
            assert!(!s
                .action_space()
                .actions
                .iter()
                .any(|action| matches!(action, Action::ChooseMode { mode: 3, .. })));
        } else {
            s.pass_priority();
            s.pass_priority();
            s.assert_winner(1);
            assert!(s.game().state.loss(0).unwrap().instructed);
        }
    }
}

#[test]
fn demonic_pact_resolves_each_resource_mode() {
    use crate::support::{
        game::force_new_card_in_hand,
        play::{command, settle, target_opponent},
    };
    use managym::{agent::action::Action, flow::turn::StepKind};

    for mode in 0..3 {
        let mut s = five_color_setup(&["Demonic Pact"]);
        s.force_permanent_on_battlefield(0, "Demonic Pact");
        s.advance_to_active_step(0, StepKind::Upkeep);
        for name in ["Forest", "Swamp", "Island"] {
            force_new_card_in_hand(&mut s, 1, name);
        }
        let hands = [0, 1].map(|seat| s.zone_size(seat, ZoneType::Hand));
        command(
            &mut s,
            |a| matches!(a, Action::ChooseMode {mode: choice, ..} if *choice == mode),
        );
        if mode != 2 {
            target_opponent(&mut s);
        }
        settle(&mut s);
        s.assert_life(0, if mode == 0 { 24 } else { 20 });
        s.assert_life(1, if mode == 0 { 16 } else { 20 });
        s.assert_zone_size(0, ZoneType::Hand, hands[0] + if mode == 2 { 2 } else { 0 });
        s.assert_zone_size(1, ZoneType::Hand, hands[1] - if mode == 1 { 2 } else { 0 });
    }
}

#[test]
fn exhausted_pact_keeps_its_history_until_it_leaves() {
    use crate::support::{
        inspect::card_of,
        play::{choose_mode, settle, target_opponent},
    };
    use managym::{agent::action::ActionSpaceKind, flow::turn::StepKind};

    let mut s = five_color_setup(&["Demonic Pact", "Herald of Eternal Dawn"]);
    let original = s.force_permanent_on_battlefield(0, "Demonic Pact");
    let object = card_of(&s, original);
    s.force_permanent_on_battlefield(0, "Herald of Eternal Dawn");
    for mode in 0..4 {
        s.advance_to_active_step(0, StepKind::Upkeep);
        assert_eq!(s.action_space().actions.len(), 4 - mode);
        choose_mode(&mut s, mode);
        if mode < 2 {
            target_opponent(&mut s);
        }
        settle(&mut s);
        s.advance_to_active_step(1, StepKind::Upkeep);
    }
    s.advance_to_active_step(0, StepKind::Upkeep);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert!(s.game().state.stack_objects.is_empty());

    let exhausted = s.game().clone();
    for blink in [false, true] {
        *s.game_mut() = exhausted.clone();
        if blink {
            let exiled = s.game_mut().move_object(object, ZoneType::Exile);
            s.game_mut().move_object(exiled, ZoneType::Battlefield);
        } else {
            let second = s.force_permanent_on_battlefield(0, "Demonic Pact");
            assert_ne!(original, second);
        }
        s.advance_to_active_step(1, StepKind::Upkeep);
        s.advance_to_active_step(0, StepKind::Upkeep);
        s.advance_until(
            |s| s.action_space().kind == ActionSpaceKind::Modal,
            "fresh Pact has no modes".into(),
        );
        assert_eq!(s.action_space().actions.len(), 4);
    }
}

#[test]
fn countered_pact_mode_stays_used_when_the_permanent_is_stolen() {
    use crate::support::{
        game::force_new_card_in_hand,
        play::{cast_card, choose_mode, counter_stacked_ability, resolve_top, settle},
    };
    use managym::{
        agent::action::Action,
        flow::turn::StepKind,
        state::{
            game_object::{PlayerId, Target},
            player::ManaPayment,
        },
    };

    let mut s = five_color_setup(&["Demonic Pact", "Gray Ogre", "Agent of Treachery"]);
    let pact = s.force_permanent_on_battlefield(0, "Demonic Pact");
    s.advance_to_active_step(0, StepKind::Upkeep);
    choose_mode(&mut s, 2);
    let hand = s.zone_size(0, ZoneType::Hand);
    s.pass_priority();
    counter_stacked_ability(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand);

    s.advance_to_active_step(1, StepKind::Main);
    let agent = force_new_card_in_hand(&mut s, 1, "Agent of Treachery");
    s.game_mut().mana_payment[1] = ManaPayment::Auto;
    s.set_player_mana_pool(1, "5UU");
    cast_card(&mut s, agent);
    resolve_top(&mut s);
    assert!(s.choose_target(Target::Permanent(pact)));
    settle(&mut s);
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    let modes: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::ChooseMode { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    assert_eq!(modes, [0, 1, 3]);
}
