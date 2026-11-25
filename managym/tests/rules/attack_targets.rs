//! CR 508–510. Each attacking creature keeps its own declared defender.
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
    },
    flow::{combat::AttackTarget, turn::StepKind},
    state::{
        game_object::{PermanentId, PlayerId, Target},
        permanent::CounterKind,
        zone::ZoneType,
    },
};

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, move_card, set_counters, Scenario},
    inspect::{counters_of, offer_labels},
    play::{command, pass_until},
};

const JACE: &str = "The Theorist, Jace Beleren";

fn defending_planeswalker(s: &mut Scenario) -> PermanentId {
    let id = s.force_permanent_on_battlefield(1, JACE);
    set_counters(s, id, CounterKind::Loyalty, 3);
    id
}

#[test]
fn ordinary_trample_assigns_excess_to_the_chosen_planeswalker() {
    let mut s = five_color_setup(&["War Mammoth", JACE, "Llanowar Elves"]);
    let mammoth = s.force_permanent_on_battlefield(0, "War Mammoth");
    clear_summoning_sickness(&mut s, mammoth);
    let jace = defending_planeswalker(&mut s);
    let object = s.game().permanent_object(jace).unwrap();
    s.force_permanent_on_battlefield(1, "Llanowar Elves");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let life = s.life(1);
    assert_eq!(s.action_space().actions.len(), 3);
    assert!(matches!(
        s.action_space().actions[0],
        Action::DeclareAttacker {
            attack: Some(AttackTarget::Player(PlayerId(1))),
            ..
        }
    ));
    assert!(matches!(
        s.action_space().actions[1],
        Action::DeclareAttacker { attack: None, .. }
    ));
    use managym::agent::{
        observation_encoder::{encode, ObservationEncoderConfig},
        observation_layout::action,
    };
    let config = ObservationEncoderConfig::default();
    let encoded = encode(&Observation::for_player(s.game(), PlayerId(0)), &config).unwrap();
    let width = config.action_dim();
    assert_eq!(encoded.actions[action::ATTACK_TARGET], 0.0);
    assert_eq!(encoded.actions[width + action::ATTACK_TARGET], 0.0);
    assert_eq!(
        encoded.actions[2 * width + action::ATTACK_TARGET],
        (encoded.action_focus[2 * config.max_focus_objects + 1] + 1) as f32
    );
    assert!(encoded.actions[2 * width + action::ATTACK_TARGET] > 0.0);
    command(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { attack: Some(AttackTarget::Permanent(d)), .. } if *d == object),
    );
    for seat in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), seat);
        let defender = s.game().state.permanents[jace].as_ref().unwrap().id.0 as i32;
        assert!(obs
            .agent_permanents
            .iter()
            .chain(&obs.opponent_permanents)
            .any(|p| p.attacking && p.attacked_permanent == Some(defender)));
    }
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    pass_until(&mut s, ActionSpaceKind::Modal);
    assert!(offer_labels(&s)[0].contains(&format!("1 to Llanowar Elves, 2 to {JACE}")));
    command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 1);
    assert_eq!(s.life(1), life);
}

#[test]
fn unblocked_damage_kills_the_planeswalker_or_is_lost_if_it_left() {
    for removed in [false, true] {
        let mut s = five_color_setup(&["War Mammoth", JACE]);
        let mammoth = s.force_permanent_on_battlefield(0, "War Mammoth");
        clear_summoning_sickness(&mut s, mammoth);
        let jace = defending_planeswalker(&mut s);
        let object = s.game().permanent_object(jace).unwrap();
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        let life = s.life(1);
        command(
            &mut s,
            |a| matches!(a, Action::DeclareAttacker { attack: Some(AttackTarget::Permanent(d)), .. } if *d == object),
        );
        if removed {
            move_card(&mut s, object, ZoneType::Hand);
        }
        s.advance_to_active_step(0, StepKind::EndOfCombat);
        assert!(s.game().state.permanents[jace].is_none());
        assert_eq!(s.life(1), life);
    }
}

#[test]
fn doubled_boulders_choose_separate_defenders_before_either_arrives() {
    let mut s = five_color_setup(&[
        "Gray Ogre",
        JACE,
        "Fire Navy Trebuchet",
        "Doubling Season",
        "Tomik, Orzhov Lawmage",
    ]);
    for name in ["Fire Navy Trebuchet", "Doubling Season"] {
        s.force_permanent_on_battlefield(0, name);
    }
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    let jace = defending_planeswalker(&mut s);
    set_counters(&mut s, jace, CounterKind::Loyalty, 6);
    let object = s.game().permanent_object(jace).unwrap();
    let tomik = s.force_permanent_on_battlefield(1, "Tomik, Orzhov Lawmage");
    s.game_mut().state.permanents[tomik]
        .as_mut()
        .unwrap()
        .tapped = true;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    command(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { attack: Some(AttackTarget::Permanent(d)), .. } if *d == object),
    );
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    for target in [Target::Permanent(jace), Target::Player(PlayerId(1))] {
        assert!(s
            .battlefield_permanents_named(0, "Ballistic Boulder")
            .is_empty());
        command(
            &mut s,
            |a| matches!(a, Action::ChooseTarget { target: t, .. } if *t == target),
        );
    }
    let tokens = s.battlefield_permanents_named(0, "Ballistic Boulder");
    assert_eq!(tokens.len(), 2);
    let life = s.life(1);
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    assert_eq!(s.life(1), life - 2);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 2);
}

#[test]
fn attack_triggers_distinguish_you_and_your_planeswalker() {
    use crate::support::{game::inject_ability, inspect::card_of};
    use managym::state::{
        ability::{
            Ability, Actor, AttackSize, AttackedKind, Effect, Per, TriggerCondition, TriggerSubject,
        },
        predicate::CardPredicate,
    };
    for walker in [false, true] {
        let mut s = five_color_setup(&["Gray Ogre", JACE]);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        clear_summoning_sickness(&mut s, ogre);
        let jace = defending_planeswalker(&mut s);
        let object = card_of(&s, jace);
        for (attacked_kind, amount) in [
            (AttackedKind::PlayerOrPlaneswalker, 1),
            (AttackedKind::Player, 4),
        ] {
            inject_ability(
                &mut s,
                object,
                Ability::Triggered {
                    condition: TriggerCondition::Attacks {
                        subject: TriggerSubject::matching(
                            Actor::Opponent,
                            CardPredicate::creature(),
                            false,
                        ),
                        per: Per::Object,
                        attackers: AttackSize::Any,
                        attacked: Actor::You,
                        attacked_kind,
                        nth: None,
                    },
                    intervening_if: None,
                    effects: vec![Effect::GainLife {
                        amount: amount.into(),
                    }],
                },
            );
        }
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        let before = s.life(1);
        command(
            &mut s,
            |a| matches!(a,Action::DeclareAttacker{attack:Some(target),..} if matches!(target, AttackTarget::Permanent(_)) == walker),
        );
        s.advance_to_active_step(0, StepKind::EndOfCombat);
        // Attacking the player fires both triggers (+1 +4) and takes 2 combat damage (+3);
        // attacking the planeswalker fires only PlayerOrPlaneswalker (+1) and takes 0 damage.
        assert_eq!(s.life(1) - before, if walker { 1 } else { 3 });
    }
}

#[test]
fn ten_attackers_choose_independently_among_six_planeswalkers() {
    use crate::support::inspect::card_of;
    use managym::{
        agent::structured_offer::{ChoiceAnswer, ChoiceStep, OfferSubmission},
        state::card::{CardType, CardTypes},
    };
    let tomik = "Tomik, Orzhov Lawmage";
    let mut s = five_color_setup(&["Gray Ogre", "Llanowar Elves", "War Mammoth", tomik]);
    for (name, count) in [("Gray Ogre", 4), ("Llanowar Elves", 3), ("War Mammoth", 3)] {
        for _ in 0..count {
            let attacker = s.force_permanent_on_battlefield(0, name);
            clear_summoning_sickness(&mut s, attacker);
        }
    }
    for _ in 0..6 {
        let walker = s.force_permanent_on_battlefield(1, "Island");
        let object = card_of(&s, walker);
        s.game_mut().state[object].types = CardTypes::new([CardType::Planeswalker]);
        set_counters(&mut s, walker, CounterKind::Loyalty, 10);
    }
    let mut restricted = s.clone();
    restricted.force_permanent_on_battlefield(1, tomik);
    restricted.advance_to_active_step(0, StepKind::DeclareAttackers);
    for remaining in (1..=6).rev() {
        assert_eq!(restricted.action_space().actions.len(), 2 + remaining);
        command(&mut restricted, |a| {
            matches!(
                a,
                Action::DeclareAttacker {
                    attack: Some(AttackTarget::Permanent(_)),
                    ..
                }
            )
        });
    }
    assert_eq!(restricted.action_space().actions.len(), 2);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(s.action_space().actions.len(), 8);
    let offers = s.game().compound_offers().unwrap();
    let offer = &offers.projection().offers[0];
    assert_eq!(offer.choices.len(), 10);
    let mut answers = Vec::new();
    for (i, choice) in offer.choices.iter().enumerate() {
        let ChoiceStep::Select {
            role, candidates, ..
        } = choice;
        let options = candidates.initial.as_ref().unwrap();
        assert_eq!(options.len(), 7);
        answers.push(ChoiceAnswer::Candidates {
            role: *role,
            candidates: vec![options[i % 7].id],
        });
    }
    s.game_mut()
        .apply_offer_submission(
            &offers,
            &OfferSubmission {
                offer_id: offer.id,
                answers,
            },
        )
        .unwrap();
    let combat = s.game().state.combat.as_ref().unwrap();
    assert_eq!(combat.attack_targets.len(), 10);
    assert_eq!(
        combat
            .attack_targets
            .iter()
            .map(|(_, t)| t)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        7
    );
}

#[test]
fn regaining_planeswalker_type_does_not_restore_an_earlier_attack() {
    use managym::state::{
        card::CardType,
        effect::{Affected, EffectDuration, Modification},
    };
    let mut s = five_color_setup(&["Gray Ogre", JACE]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    let jace = defending_planeswalker(&mut s);
    let object = s.game().permanent_object(jace).unwrap();
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    command(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { attack: Some(AttackTarget::Permanent(d)), .. } if *d == object),
    );
    for kind in [CardType::Artifact, CardType::Planeswalker] {
        s.game_mut().apply_continuous(
            None,
            PlayerId(0),
            [Affected::Object(object)],
            Modification::SetTypes {
                types: vec![kind],
                subtypes: vec![],
            },
            EffectDuration::EndOfTurn,
        );
    }
    let before = s.life(1);
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    assert_eq!(s.life(1), before);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 3);
}

#[test]
fn planeswalker_defender_candidate_comes_from_structured_offers_and_rejects_stale_defender() {
    use managym::decision::{Command, SemanticError};
    let mut s = five_color_setup(&["Gray Ogre", JACE]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    let jace = defending_planeswalker(&mut s);
    let jace_object = s.game().permanent_object(jace).unwrap();
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let frame = s.game().semantic_decision_frame().unwrap();
    assert_eq!(frame.object_candidates.len(), 1);
    let address = frame.object_candidates[0].clone();
    assert_eq!(address.offer_id, 0);
    assert_eq!(address.role, 0);
    assert_eq!(address.candidate_id, 1);
    let attack_jace = s
        .action_space()
        .actions
        .iter()
        .position(|a| {
            matches!(
                a,
                Action::DeclareAttacker {
                    attack: Some(AttackTarget::Permanent(d)),
                    ..
                } if *d == jace_object
            )
        })
        .unwrap();
    let mut stale_branch = s.clone();
    move_card(&mut stale_branch, jace_object, ZoneType::Hand);
    let cmd = Command {
        command_id: "stale-jace".into(),
        expected_revision: frame.revision,
        offer_id: attack_jace as u32,
        answers: Vec::new(),
        object_preconditions: vec![address.clone()],
    };
    assert_eq!(
        stale_branch
            .game_mut()
            .execute_semantic_command(&cmd)
            .unwrap_err(),
        SemanticError::StaleObject
    );

    let live_cmd = Command {
        command_id: "attack-jace".into(),
        ..cmd
    };
    s.game_mut().execute_semantic_command(&live_cmd).unwrap();
    let combat = s.game().state.combat.as_ref().unwrap();
    assert_eq!(
        combat.defender(ogre),
        Some(AttackTarget::Permanent(jace_object))
    );
    move_card(&mut s, jace_object, ZoneType::Hand);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    let combat = s.game().state.combat.as_ref().unwrap();
    assert_eq!(combat.attack_targets, vec![(ogre, None)]);
    assert_eq!(combat.defender(ogre), None);
}
