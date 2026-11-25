//! Tests for cards first printed in Scars of Mirrodin.
use crate::support::{
    game::{five_color_setup, force_new_card_in_hand},
    play::{cast, cast_card, choose_mode, settle},
};
use managym::flow::turn::StepKind;

#[test]
fn exsanguinate_gains_the_life_actually_lost() {
    for (protected, x, lost) in [(false, 3, 3), (true, 3, 0), (false, 0, 0)] {
        let mut s = five_color_setup(&["Exsanguinate", "Teferi's Protection"]);
        if protected {
            cast(&mut s, "Teferi's Protection");
            settle(&mut s);
        }
        s.advance_to_active_step(1, StepKind::Main);
        s.set_player_mana_pool(1, "3BB");
        let spell = force_new_card_in_hand(&mut s, 1, "Exsanguinate");
        cast_card(&mut s, spell);
        choose_mode(&mut s, x);
        settle(&mut s);
        s.assert_life(0, 20 - lost);
        s.assert_life(1, 20 + lost);
    }
}

#[test]
fn steel_hellkite_spends_its_own_activation_and_destroys_exact_mana_value() {
    use crate::support::{
        game::clear_summoning_sickness,
        play::{activate, attack_with, pass_until},
    };
    use managym::{
        agent::{action::ActionSpaceKind, observation::Observation},
        state::game_object::PlayerId,
    };
    let mut s = five_color_setup(&["Steel Hellkite", "Hill Giant"]);
    let hit = s.force_permanent_on_battlefield(0, "Steel Hellkite");
    let other = s.force_permanent_on_battlefield(0, "Steel Hellkite");
    assert_ne!(hit, other);
    clear_summoning_sickness(&mut s, hit);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, hit);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    s.assert_life(1, 15);
    let land = s.force_permanent_on_battlefield(1, "Forest");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let own = s
        .game_mut()
        .create_token("token.food", PlayerId(0), false)
        .unwrap();
    let target = s
        .game_mut()
        .create_token("token.food", PlayerId(1), false)
        .unwrap();
    s.set_player_mana_pool(0, "2");
    let saved = s.game().clone();
    for source in [other, hit] {
        *s.game_mut() = saved.clone();
        activate(&mut s, source, 1);
        choose_mode(&mut s, 0);
        let view = Observation::for_player(s.game(), PlayerId(0));
        for (id, used) in [
            (source, true),
            (if source == hit { other } else { hit }, false),
        ] {
            let object = s.game().state.permanents[id].as_ref().unwrap();
            let row = view
                .agent_permanents
                .iter()
                .find(|row| row.id == object.id.0 as i32)
                .unwrap();
            assert_eq!(row.used_abilities.contains(&1), used);
        }
        settle(&mut s);
        assert_eq!(s.game().state.permanents[target].is_none(), source == hit);
        for kept in [own, land, giant] {
            assert!(s.game().state.permanents[kept].is_some());
        }
    }
    *s.game_mut() = saved.clone();
    s.set_player_mana_pool(0, "4");
    activate(&mut s, hit, 1);
    choose_mode(&mut s, 4);
    settle(&mut s);
    assert!(s.game().state.permanents[giant].is_none());
    assert!(s.game().state.permanents[target].is_some());

    *s.game_mut() = saved;
    activate(&mut s, hit, 1);
    choose_mode(&mut s, 0);
    settle(&mut s);
    s.advance_to_active_step(1, StepKind::Main);
    let view = Observation::for_player(s.game(), PlayerId(0));
    assert!(view
        .agent_permanents
        .iter()
        .all(|p| p.used_abilities.is_empty() && p.combat_damage_to_players.is_empty()));
    s.pass_priority();
    s.set_player_mana_pool(0, "1");
    activate(&mut s, hit, 1);
    choose_mode(&mut s, 0);
    let fresh = s
        .game_mut()
        .create_token("token.food", PlayerId(1), false)
        .unwrap();
    settle(&mut s);
    assert!(s.game().state.permanents[fresh].is_some());
}

#[test]
fn steel_hellkite_offers_only_payable_x_values() {
    use crate::support::{game::five_color_unfunded_setup, play::activate};
    use managym::agent::action::Action;

    for with_mountain in [false, true] {
        let mut s = five_color_unfunded_setup(&["Steel Hellkite"]);
        let source = s.force_permanent_on_battlefield(0, "Steel Hellkite");
        let mountain = with_mountain.then(|| s.force_permanent_on_battlefield(0, "Mountain"));
        activate(&mut s, source, 1);
        let offered: Vec<_> = s
            .action_space()
            .actions
            .iter()
            .map(|action| match action {
                Action::ChooseMode { mode, .. } => *mode,
                _ => panic!("expected an X announcement"),
            })
            .collect();
        assert_eq!(offered, if with_mountain { vec![0, 1] } else { vec![0] });
        let decision = s.game().clone();
        for x in offered {
            *s.game_mut() = decision.clone();
            choose_mode(&mut s, x);
            settle(&mut s);
            let hellkite = s.game().state.permanents[source].as_ref().unwrap();
            assert!(!hellkite.tapped);
            if let Some(land) = mountain {
                assert_eq!(
                    s.game().state.permanents[land].as_ref().unwrap().tapped,
                    x == 1
                );
            }
        }
    }
}

/// Painful Quandary's trigger is the casting opponent's choice: discard a
/// card or lose 5 life, restored from one pending choice. With an empty hand
/// only the life loss remains.
#[test]
fn painful_quandary_makes_the_caster_discard_or_lose_five_life() {
    use crate::support::{
        game::move_card,
        inspect::{hand, now, zone_of},
        play::select_card,
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{
            game_object::{PlayerId, Target},
            player::ManaPayment,
            zone::ZoneType,
        },
    };
    let mut s = five_color_setup(&["Painful Quandary", "Lightning Bolt", "Gray Ogre"]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    s.force_permanent_on_battlefield(0, "Painful Quandary");
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "RRR");
    s.game_mut().scenario_clear_hand(PlayerId(1));
    let ogre = force_new_card_in_hand(&mut s, 1, "Gray Ogre");
    let bolt = force_new_card_in_hand(&mut s, 1, "Lightning Bolt");
    cast_card(&mut s, bolt);
    assert!(s.choose_target(Target::Player(PlayerId(0))));
    while s.action_space().kind == ActionSpaceKind::Priority {
        s.pass_priority();
    }
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    let choice = s.game().clone();
    select_card(&mut s, ogre);
    settle(&mut s);
    s.assert_life(1, 20);
    assert!(hand(&s, 1).is_empty());
    assert_eq!(zone_of(&s, now(&s, ogre)), Some(ZoneType::Graveyard));
    *s.game_mut() = choice;
    settle(&mut s);
    s.assert_life(1, 15);
    assert_eq!(hand(&s, 1), vec![ogre]);
    // Empty-handed, the caster just loses 5 life.
    let bolt = force_new_card_in_hand(&mut s, 1, "Lightning Bolt");
    s.set_player_mana_pool(1, "R");
    move_card(&mut s, ogre, ZoneType::Graveyard);
    cast_card(&mut s, bolt);
    assert!(s.choose_target(Target::Player(PlayerId(0))));
    settle(&mut s);
    s.assert_life(1, 10);
}

/// A trigger reading "that spell's mana value" after its spell is countered
/// reads the spell as it last existed on the stack (CR 608.2h), its
/// announced X included: X = 3 Exsanguinate has mana value 5. Both seats see
/// the departed spell on the pending trigger.
#[test]
fn a_countered_x_spell_keeps_its_stack_mana_value_for_its_cast_trigger() {
    use crate::support::game::inject_ability;
    use crate::support::inspect::{card_of, stack_spell_card};
    use managym::{
        agent::observation::Observation,
        state::{
            ability::{
                Ability, Actor, Aggregate, CompareOp, Condition, Effect, EffectValue,
                ObjectProperty, ObjectSelector, PlayerSelector, TriggerCondition,
            },
            game_object::{PlayerId, Target},
            player::ManaPayment,
            predicate::CardPredicate,
        },
    };
    let mut s = five_color_setup(&["Exsanguinate", "Counterspell", "Gray Ogre"]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    let watcher = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let watcher = card_of(&s, watcher);
    let mana_value = EffectValue::Property {
        of: Box::new(ObjectSelector::Triggering),
        property: ObjectProperty::ManaValue,
        aggregate: Aggregate::Sum,
    };
    inject_ability(
        &mut s,
        watcher,
        Ability::Triggered {
            condition: TriggerCondition::casts(Actor::Opponent, CardPredicate::default()),
            intervening_if: Some(Condition::compare(mana_value.clone(), CompareOp::Eq, 5)),
            effects: vec![Effect::LoseLife {
                player: PlayerSelector::TriggeringPlayer,
                amount: mana_value,
            }],
        },
    );
    let spell = force_new_card_in_hand(&mut s, 0, "Exsanguinate");
    cast_card(&mut s, spell);
    choose_mode(&mut s, 3);
    let spell = stack_spell_card(&s, "Exsanguinate");
    assert_eq!(s.game().state.stack_objects.len(), 2);
    s.pass_priority();
    let counter = force_new_card_in_hand(&mut s, 1, "Counterspell");
    cast_card(&mut s, counter);
    assert!(s.choose_target(Target::StackSpell(spell)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().state.stack_objects.len(), 1, "the trigger waits");
    let projected = [PlayerId(0), PlayerId(1)].map(|viewer| {
        Observation::for_player(s.game(), viewer)
            .continuation
            .into_iter()
            .find(|frame| frame.kind == 3)
            .expect("the cast trigger")
            .effect_execution
    });
    assert!(!projected[0].is_empty());
    assert_eq!(projected[0], projected[1], "public last known spell");
    let text = s
        .game()
        .decision_context(PlayerId(0))
        .unwrap()
        .pending
        .join(" ");
    assert!(
        text.contains("Last known spell Exsanguinate: mana value 5"),
        "{text}"
    );
    settle(&mut s);
    s.assert_life(0, 15);
    s.assert_life(1, 20);
}
