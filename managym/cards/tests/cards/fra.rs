use super::sos::cast_prepared_insets;

#[test]
fn prepared_insets_cast_from_exile() {
    cast_prepared_insets(&[
        ("Prudent Fateseer", "Peer Review"),
        ("Diviner of Victory", "Unwind History"),
        ("Blossom-Blessed Angel", "Seed Suture"),
        ("Carnivorous Cultivator", "Enroot"),
        ("Emergency Phytomedic", "Seed Suture"),
        ("Fatehold Chronologist", "Peer Review"),
        ("Hallway Heckler", "Vicious Verse"),
        ("Konstrari Improviser", "Soul Tether"),
        ("Pompous Battlemage", "Improvised Act"),
        ("Semester Foreseer", "Peer Review"),
        ("Theorix Metamage", "Omit Variables"),
        ("Vigorbloom Vanguard", "Seed Suture"),
        ("Void Extrapolator", "Omit Variables"),
    ]);
}

/// Heartwood Crafter's colorless mana pays Soul Tether cast from its
/// prepared exile copy but no spell from hand. Codie copies the prepared
/// spell; every viewer sees which stack spell was prepared, and the mana's
/// restriction reaches the policy as a pooled-mana fact row.
#[test]
fn heartwood_mana_pays_its_prepared_spell_but_no_hand_spell() {
    use crate::support::{
        game::{clear_summoning_sickness, enter_battlefield, five_color_unfunded_setup},
        inspect::find_owned_card_off_battlefield,
        play::{cast_card, castable, resolve_top, settle, tap_for_mana},
    };
    use managym::agent::{
        observation::Observation,
        observation_encoder::{
            encode, unrepresentable_state, ObservationEncoderConfig, FACT_DIM, POOLED_MANA,
            STACK_DIM,
        },
        observation_layout::{fact, stack},
    };
    use managym::state::{game_object::PlayerId, zone::ZoneType};
    let mut s =
        five_color_unfunded_setup(&["Codie, Ravenous Codex", "Heartwood Crafter", "Gray Ogre"]);
    s.game_mut()
        .scenario_force_battlefield(PlayerId(0), "Codie, Ravenous Codex", true)
        .unwrap();
    let crafter = enter_battlefield(&mut s, "Heartwood Crafter");
    settle(&mut s);
    clear_summoning_sickness(&mut s, crafter);
    s.set_player_mana_pool(0, "RR");
    s.allow_priority_floats();
    tap_for_mana(&mut s, crafter);

    let ogre = crate::support::game::move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    assert!(
        !castable(&mut s, ogre),
        "the {{C}} cannot pay a spell from hand"
    );
    let obs = Observation::for_player(s.game(), PlayerId(0));
    assert!(obs.agent.restricted_mana.is_empty());
    let [pooled] = &obs.agent.pooled_mana_classes[..] else {
        panic!("one pooled class: {:?}", obs.agent.pooled_mana_classes)
    };
    assert_eq!(
        pooled.restriction.as_deref(),
        Some("anything but spells from hand")
    );
    assert_eq!(pooled.mana, [0, 0, 0, 0, 0, 1]);
    assert_eq!(obs.agent.lesson_mana, [0, 0, 0, 0, 0, 1]);
    assert_eq!(unrepresentable_state(&obs), None);
    let input = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
    let pooled_rows = input
        .facts
        .chunks_exact(FACT_DIM)
        .filter(|row| {
            let id = ((row[fact::ID_HIGH] as u32) << 16) | row[fact::ID_LOW] as u32;
            id == managym::cards::tokens::name_value(POOLED_MANA) as u32
        })
        .count();
    assert!(pooled_rows > 0, "the restricted class is a fact program");

    let tether = find_owned_card_off_battlefield(&s, 0, "Soul Tether");
    // The pool pays exactly, so the cast completes without a manual choice.
    cast_card(&mut s, tether);
    assert_eq!(
        s.game().state.players[0].mana_pool.total(),
        0,
        "Soul Tether spent the {{C}} and both red"
    );
    resolve_top(&mut s);
    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        let prepared: Vec<_> = obs
            .stack_objects
            .iter()
            .map(|object| (object.from_prepared, object.flashback))
            .collect();
        // Top first: Codie's copy above the prepared original.
        assert_eq!(prepared, [(false, false), (true, false)]);
        let input = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
        let rows: Vec<_> = input
            .stack
            .chunks_exact(STACK_DIM)
            .take(2)
            .map(|row| row[stack::FROM_PREPARED])
            .collect();
        assert_eq!(rows, [0.0, 1.0]);
        let restored: Observation = serde_json::from_str(&obs.to_json()).unwrap();
        assert_eq!(restored.to_json(), obs.to_json());
    }
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Heartwood").len(), 2);
}

/// Codie copies a spell cast from a prepared copy. Its copy is not a prepared
/// spell, an ordinary cast does not trigger it, and its activation prepares
/// each unprepared creature with a prepare spell without duplicating a copy.
#[test]
fn codie_copies_prepared_spells_and_prepares_each_creature() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup},
        play::{activate, cast_card, pay_rest, resolve_top, settle},
    };
    use managym::state::{
        game_object::{ObjectId, PlayerId},
        stack_object::StackObject,
        zone::ZoneType,
    };
    let prepared_spells = |s: &crate::support::game::Scenario| -> Vec<bool> {
        s.game()
            .state
            .stack_objects
            .iter()
            .filter_map(|object| match object {
                StackObject::Spell(spell) => Some(spell.from_prepared),
                _ => None,
            })
            .collect()
    };
    let tethers = |s: &crate::support::game::Scenario| -> Vec<ObjectId> {
        s.game()
            .state
            .zones
            .zone_objects(ZoneType::Exile, PlayerId(0))
            .iter()
            .copied()
            .filter(|id| s.game().state[*id].name == "Soul Tether")
            .collect()
    };
    let mut s = five_color_setup(&["Codie, Ravenous Codex", "Konstrari Improviser", "Gray Ogre"]);
    let codie = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Codie, Ravenous Codex", true)
        .unwrap();
    enter_battlefield(&mut s, "Konstrari Improviser");
    settle(&mut s);
    let [copy] = tethers(&s)[..] else {
        panic!("one prepared Soul Tether")
    };
    cast_card(&mut s, copy);
    pay_rest(&mut s);
    assert_eq!(prepared_spells(&s), [true]);
    resolve_top(&mut s);
    assert_eq!(prepared_spells(&s), [true, false]);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Heartwood").len(), 2);
    assert!(tethers(&s).is_empty());

    let ogre = crate::support::game::move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    cast_card(&mut s, ogre);
    pay_rest(&mut s);
    assert_eq!(prepared_spells(&s), [false]);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);

    enter_battlefield(&mut s, "Konstrari Improviser");
    settle(&mut s);
    let kept = tethers(&s);
    assert_eq!(kept.len(), 1);
    s.set_player_mana_pool(0, "WUBRG");
    activate(&mut s, codie, 0);
    settle(&mut s);
    let prepared = tethers(&s);
    assert_eq!(prepared.len(), 2, "one copy for each Improviser");
    assert!(prepared.contains(&kept[0]));
}

#[test]
fn commons_and_annex_lands_check_their_controllers_planeswalkers_and_pay_either_color() {
    use crate::support::{
        game::{five_color_unfunded_setup, set_counters},
        inspect::tapped,
        play::{play_land, tapped_dual_land_pays_either_color},
    };
    use managym::state::permanent::CounterKind;

    for (name, colors) in [
        ("Dedicated Commons", "RW"),
        ("Fatehold Annex", "WU"),
        ("Formidable Commons", "BG"),
        ("Innovative Commons", "UR"),
        ("Konstrari Annex", "RG"),
        ("Meticulous Commons", "WB"),
        ("Stingerquill Annex", "BR"),
        ("Theorix Annex", "UB"),
        ("Transformative Commons", "GU"),
        ("Vigorbloom Annex", "GW"),
    ] {
        tapped_dual_land_pays_either_color(name, colors, |_, _| {});
        for controller in [0, 1] {
            let mut s = five_color_unfunded_setup(&[name, "Gray Ogre"]);
            let walker = s.force_permanent_on_battlefield(controller, "Gray Ogre");
            crate::make_planeswalker(&mut s, walker);
            set_counters(&mut s, walker, CounterKind::Loyalty, 3);
            let land = play_land(&mut s, name);
            assert_eq!(
                tapped(&s, land),
                controller != 0,
                "{name}, planeswalker seat {controller}"
            );
        }
    }
}

#[test]
fn theorist_loyalty_is_paid_before_resolution_and_shared_across_abilities() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup},
        inspect::counters_of,
        play::{activate, offers_ability, settle},
    };
    use managym::{
        agent::{
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig},
            observation_layout::permanent,
        },
        flow::turn::StepKind,
        state::{game_object::PlayerId, permanent::CounterKind},
    };
    let name = "The Theorist, Jace Beleren";
    let mut s = five_color_setup(&["Doubling Season", name]);
    enter_battlefield(&mut s, "Doubling Season");
    let jace = enter_battlefield(&mut s, name);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 6);
    activate(&mut s, jace, 0);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 7);
    assert!(s.battlefield_permanents_named(0, "Illusion").is_empty());
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Illusion").len(), 2);
    assert!(!(0..3).any(|index| offers_ability(&mut s, jace, index)));
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let row = obs
        .agent_permanents
        .iter()
        .position(|p| p.characteristics.name == name)
        .unwrap();
    let input = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
    let offset = row * permanent::WIDTH;
    assert_eq!(
        input.agent_permanents[offset + permanent::PRINTED_LOYALTY],
        3.0
    );
    assert_eq!(
        input.agent_permanents[offset + permanent::LOYALTY_ACTIVATED_THIS_TURN],
        1.0
    );
    let restored: Observation = serde_json::from_str(&obs.to_json()).unwrap();
    assert_eq!(restored.to_json(), obs.to_json());
    s.advance_to_active_step(1, StepKind::Main);
    s.pass_priority();
    assert!(!offers_ability(&mut s, jace, 0));
    s.advance_to_active_step(0, StepKind::Main);
    assert!(offers_ability(&mut s, jace, 2));
}

#[test]
fn theorist_loyalty_timing_permission_preserves_quota_and_targets() {
    use crate::support::{
        game::{
            enter_battlefield, five_color_setup, player_effect, put_onto_battlefield, set_counters,
        },
        inspect::{counters_of, hand},
        play::{activate, offers_ability, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{
            effect::{EffectDuration, Modification},
            game_object::Target,
            permanent::CounterKind,
            predicate::CardPredicate,
        },
    };
    let name = "The Theorist, Jace Beleren";
    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    let jace = enter_battlefield(&mut s, name);
    let ogre = put_onto_battlefield(&mut s, 1, "Gray Ogre");
    assert!(!offers_ability(&mut s, jace, 2));
    let hand_size = hand(&s, 0).len();
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(hand(&s, 0).len(), hand_size + 1);
    s.pass_priority();
    assert!(!offers_ability(&mut s, jace, 1));
    for (subtype, allowed) in [("Chandra", false), ("Jace", true)] {
        player_effect(
            &mut s,
            0,
            Modification::LoyaltyAtInstantSpeed(Box::new(CardPredicate {
                subtype: Some(subtype.into()),
                ..Default::default()
            })),
            EffectDuration::EndOfTurn,
        );
        assert_eq!(offers_ability(&mut s, jace, 1), allowed);
    }
    activate(&mut s, jace, 1);
    assert!(s.choose_target(Target::Permanent(ogre)));
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 1);
    settle(&mut s);
    assert!(s.game().state.permanents[ogre].is_none());
    assert!(!offers_ability(&mut s, jace, 0));
    s.advance_to_active_step(0, StepKind::Main);
    let mine = enter_battlefield(&mut s, "Gray Ogre");
    set_counters(&mut s, jace, CounterKind::Loyalty, 6);
    activate(&mut s, jace, 2);
    assert!(s.game().state.permanents[jace].is_none());
    settle(&mut s);
    let cards = hand(&s, 0).len();
    assert_eq!(
        counters_of(&s, mine, CounterKind::PlusOnePlusOne),
        cards as i32
    );
}

/// A loyalty symbol makes even a mana-producing activation use the stack.
#[test]
fn theorist_loyalty_with_zero_cost_and_mana_output_uses_the_stack() {
    use crate::support::{
        game::{enter_battlefield, five_color_unfunded_setup},
        inspect::{card_of, counters_of},
        play::{activate, offers_ability, settle},
    };
    use managym::state::{
        ability::Effect,
        card::{ActivatedAbilityDefinition, LoyaltyCost},
        mana::Mana,
        permanent::CounterKind,
    };
    let name = "The Theorist, Jace Beleren";
    let mut s = five_color_unfunded_setup(&[name]);
    let jace = enter_battlefield(&mut s, name);
    let object = card_of(&s, jace);
    s.game_mut().state[object]
        .activated_abilities
        .push(ActivatedAbilityDefinition {
            loyalty: Some(LoyaltyCost::Fixed(0)),
            effects: vec![Effect::add_mana(Mana::parse("RR"))],
            ..Default::default()
        });
    activate(&mut s, jace, 3);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 3);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 2);
    assert!(!offers_ability(&mut s, jace, 0));
}

#[test]
fn minus_x_loyalty_survives_target_selection_payment_and_resolution() {
    use crate::support::{
        game::{enter_battlefield, five_color_unfunded_setup, set_counters},
        inspect::{card_of, counters_of},
        play::{activate, choose, choose_mode, settle},
    };
    use managym::{
        agent::{action::Action, observation::Observation},
        state::{
            ability::{Effect, EffectValue, ObjectSelector, TargetRequirement, TargetSpec},
            card::{ActivatedAbilityDefinition, LoyaltyCost},
            game_object::{PlayerId, Target},
            permanent::CounterKind,
        },
    };

    let name = "The Theorist, Jace Beleren";
    for take_target in [true, false] {
        let mut s = five_color_unfunded_setup(&[name, "Gray Ogre"]);
        let jace = enter_battlefield(&mut s, name);
        let ogre = enter_battlefield(&mut s, "Gray Ogre");
        let object = card_of(&s, jace);
        // A separate source-counter cost reserves its counter before pricing X.
        let reserved = u16::from(!take_target);
        s.game_mut().state[object]
            .activated_abilities
            .push(ActivatedAbilityDefinition {
                loyalty: Some(LoyaltyCost::MinusX),
                remove_counters: (!take_target).then_some((CounterKind::Loyalty, reserved)),
                effects: vec![
                    Effect::TargetClause {
                        requirement: TargetRequirement::up_to(1, TargetSpec::Creature),
                        effects: vec![Effect::return_to_hand(ObjectSelector::Target(
                            TargetSpec::Creature,
                        ))],
                    },
                    Effect::GainLife {
                        amount: EffectValue::X,
                    },
                ],
                ..Default::default()
            });
        let loyalty = 256 + i32::from(reserved);
        set_counters(&mut s, jace, CounterKind::Loyalty, loyalty);
        let before = s.life(0);
        activate(&mut s, jace, 3);
        let offered: Vec<_> = s
            .action_space()
            .actions
            .iter()
            .map(|a| match a {
                Action::ChooseMode { mode, .. } => *mode,
                _ => panic!("unexpected X offer {a:?}"),
            })
            .collect();
        assert_eq!(offered, (0..=256).collect::<Vec<_>>());
        let frame = s.game().semantic_decision_frame().unwrap();
        assert_eq!(frame.offers[256].label, "X = 256");
        choose_mode(&mut s, 256);
        assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), loyalty);
        let obs = Observation::for_player(s.game(), PlayerId(0));
        assert!(obs.continuation.iter().any(|f| f.x == 256));
        if take_target {
            assert!(s.choose_target(Target::Permanent(ogre)));
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        let stacked = s.game().state.stack_objects.last().unwrap();
        assert_eq!(stacked.x(), 256);
        assert_eq!(
            stacked.targets(),
            if take_target {
                vec![Target::Permanent(ogre)]
            } else {
                vec![]
            }
        );
        assert!(s.game().state.permanents[jace].is_none());
        settle(&mut s);
        assert_eq!(s.life(0), before + 256);
        assert_eq!(s.game().state.permanents[ogre].is_none(), take_target);
    }
}

#[test]
fn tomik_grants_a_restriction_to_each_planeswalker_and_flying_to_a_countered_creature() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup, ready, set_counters},
        inspect::keywords,
        play::{activate, command, settle},
    };
    use managym::{
        agent::action::Action,
        flow::{combat::AttackTarget, turn::StepKind},
        state::{card::Keyword, game_object::Target, permanent::CounterKind},
    };
    let tomik = "Tomik, Orzhov Lawmage";
    let walker = "The Theorist, Jace Beleren";
    let mut s = five_color_setup(&[tomik, walker, "Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, tomik);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let jace = s.force_permanent_on_battlefield(1, walker);
    s.force_permanent_on_battlefield(1, tomik);
    set_counters(&mut s, jace, CounterKind::Loyalty, 3);
    set_counters(&mut s, first, CounterKind::PlusOnePlusOne, 1);
    ready(&mut s, tomik);
    activate(&mut s, source, 0);
    assert!(s.action_space().actions.iter().all(
        |a| matches!(a, Action::ChooseTarget {target: Target::Permanent(id), ..} if *id == first)
    ));
    assert!(s.choose_target(Target::Permanent(first)));
    settle(&mut s);
    assert!(keywords(&s, first).has(Keyword::Flying));
    for id in [first, second] {
        clear_summoning_sickness(&mut s, id);
    }
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    command(&mut s, |a| {
        matches!(
            a,
            Action::DeclareAttacker {
                attack: Some(AttackTarget::Permanent(_)),
                ..
            }
        )
    });
    assert_eq!(s.action_space().actions.len(), 2);
    assert!(matches!(
        s.action_space().actions[0],
        Action::DeclareAttacker {
            attack: Some(AttackTarget::Player(_)),
            ..
        }
    ));
    s.declare_attack();
}

#[test]
fn empower_choices_survive_doubling_and_spend_loyalty() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup},
        inspect::{card_of, counters_of, hand},
        play::{activate, cast_owned, pay_rest, resolve_top, select_card, settle},
    };
    use managym::state::permanent::CounterKind;

    let mut s = five_color_setup(&["Doubling Season", "Protege's Awakening"]);
    enter_battlefield(&mut s, "Doubling Season");
    enter_battlefield(&mut s, "Doubling Season");
    cast_owned(&mut s, "Protege's Awakening");
    if s.game().state.stack_objects.is_empty() {
        pay_rest(&mut s);
    }
    resolve_top(&mut s);
    while s.action_space().kind == managym::agent::action::ActionSpaceKind::Modal {
        s.step_action(0);
    }
    let candidates = s.battlefield_permanents_named(0, "Jace");
    assert_eq!(candidates.len(), 4);
    assert_eq!(s.action_space().actions.len(), 4);
    for id in candidates {
        let mut fork = s.clone();
        let object = card_of(&fork, id);
        select_card(&mut fork, object);
        settle(&mut fork);
        assert_eq!(fork.battlefield_permanents_named(0, "Jace"), [id]);
        assert_eq!(counters_of(&fork, id, CounterKind::Loyalty), 24);
        let before = hand(&fork, 0).len();
        activate(&mut fork, id, 1);
        settle(&mut fork);
        assert_eq!(hand(&fork, 0).len(), before + 1);
        assert_eq!(counters_of(&fork, id, CounterKind::Loyalty), 21);
        cast_owned(&mut fork, "Protege's Awakening");
        settle(&mut fork);
        assert_eq!(fork.battlefield_permanents_named(0, "Jace"), [id]);
        assert_eq!(counters_of(&fork, id, CounterKind::Loyalty), 45);
    }
}

#[test]
fn empower_measures_exiled_creatures_and_planeswalker_excess() {
    use crate::support::{
        game::{five_color_setup, put_onto_battlefield},
        inspect::counters_of,
        play::{activate, cast_owned, settle},
    };
    use managym::state::{game_object::Target, permanent::CounterKind};
    let walker = "The Theorist, Jace Beleren";
    let mut s = five_color_setup(&[
        walker,
        "Gray Ogre",
        "Overwrite the Multiverse",
        "Violent Echoes",
    ]);
    let opposing = put_onto_battlefield(&mut s, 1, walker);
    let mine = put_onto_battlefield(&mut s, 0, walker);
    activate(&mut s, mine, 0);
    settle(&mut s);
    for seat in [0, 1] {
        put_onto_battlefield(&mut s, seat, "Gray Ogre");
    }
    cast_owned(&mut s, "Overwrite the Multiverse");
    settle(&mut s);
    let jace = s.battlefield_permanents_named(0, "Jace")[0];
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 3);
    assert_eq!(counters_of(&s, mine, CounterKind::Loyalty), 4);
    cast_owned(&mut s, "Violent Echoes");
    assert!(s.choose_target(Target::Permanent(opposing)));
    settle(&mut s);
    assert!(s.game().state.permanents[opposing].is_none());
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 6);
}

#[test]
fn empower_resolves_after_lethal_damage_but_requires_legal_targets() {
    use crate::support::{
        game::{five_color_setup, put_onto_battlefield},
        inspect::{card_of, counters_of},
        play::{cast_owned, pay_rest, settle},
    };
    use managym::state::{game_object::Target, permanent::CounterKind, zone::ZoneType};
    let mut s = five_color_setup(&["Gray Ogre", "Academic Ascent", "No Admittance"]);
    let gone = put_onto_battlefield(&mut s, 1, "Gray Ogre");
    cast_owned(&mut s, "Academic Ascent");
    assert!(s.choose_target(Target::Permanent(gone)));
    if s.game().state.stack_objects.is_empty() {
        pay_rest(&mut s);
    }
    let card = card_of(&s, gone);
    s.game_mut().move_object(card, ZoneType::Hand);
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Jace").is_empty());
    let ogre = put_onto_battlefield(&mut s, 1, "Gray Ogre");
    cast_owned(&mut s, "No Admittance");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert!(s.game().state.permanents[ogre].is_none());
    let jace = s.battlefield_permanents_named(0, "Jace")[0];
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 1);
}

#[test]
fn empower_entry_and_way_triggers_put_counters_on_the_same_jace() {
    use crate::support::{
        game::{five_color_setup, put_onto_battlefield},
        inspect::counters_of,
        play::{cast_owned, settle},
    };
    use managym::state::{game_object::Target, permanent::CounterKind};
    let cards = [
        ("Arcane Amphisbaena", 2),
        ("Keeper of the Quiet Hour", 2),
        ("Mindseeker Oculus", 4),
        ("Theorist's Proxy", 3),
        ("Way of the Mentor", 5),
        ("Way of the Necromancer", 2),
    ];
    let mut names: Vec<_> = cards.iter().map(|(name, _)| *name).collect();
    names.extend([
        "Inspired Tethermage",
        "Inspiring Overseer",
        "Gray Ogre",
        "Lightning Bolt",
    ]);
    let mut s = five_color_setup(&names);
    let mage = put_onto_battlefield(&mut s, 0, "Inspired Tethermage");
    let mut total = 0;
    for (i, (name, amount)) in cards.iter().enumerate() {
        s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
        cast_owned(&mut s, name);
        settle(&mut s);
        total += amount;
        let walkers = s.battlefield_permanents_named(0, "Jace");
        assert_eq!(walkers.len(), 1, "{name}");
        assert_eq!(
            counters_of(&s, walkers[0], CounterKind::Loyalty),
            total,
            "{name}"
        );
        assert_eq!(
            counters_of(&s, mage, CounterKind::PlusOnePlusOne),
            i as i32 + 1,
            "{name}"
        );
    }
    let jace = s.battlefield_permanents_named(0, "Jace")[0];
    let ogre = put_onto_battlefield(&mut s, 0, "Gray Ogre");
    s.set_player_mana_pool(0, "R");
    cast_owned(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), total + 1);
    s.set_player_mana_pool(0, "2W");
    cast_owned(&mut s, "Inspiring Overseer");
    settle(&mut s);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), total + 2);
}

#[test]
fn proxy_protects_the_next_cast_and_plan_counts_only_noncreature_casts() {
    use crate::support::{
        game::{five_color_setup, put_onto_battlefield},
        inspect::{counters_of, now, zone_of},
        play::{activate, cast_owned, settle},
    };
    use managym::{
        agent::{
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig},
        },
        state::{
            game_object::{PlayerId, Target},
            permanent::CounterKind,
            player::ManaPayment,
            zone::ZoneType,
        },
    };
    let mut s = five_color_setup(&[
        "Plan for All Outcomes",
        "Theorist's Proxy",
        "Counterspell",
        "Gray Ogre",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let proxy = put_onto_battlefield(&mut s, 0, "Theorist's Proxy");
    s.game_mut().scenario_refresh_priority().unwrap();
    settle(&mut s);
    put_onto_battlefield(&mut s, 0, "Plan for All Outcomes");
    s.game_mut().scenario_refresh_priority().unwrap();
    settle(&mut s);
    activate(&mut s, proxy, 0);
    settle(&mut s);
    let waiting = Observation::for_player(s.game(), PlayerId(0));
    let permission = waiting
        .agent
        .effects
        .iter()
        .find(|effect| effect.modification == "CantBeCountered")
        .unwrap();
    assert!(permission
        .description
        .as_deref()
        .unwrap()
        .contains("next spell you cast"));
    encode(&waiting, &ObservationEncoderConfig::default()).unwrap();
    for expected in [ZoneType::Battlefield, ZoneType::Graveyard] {
        s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
        let ogre = cast_owned(&mut s, "Gray Ogre");
        let spell = now(&s, ogre);
        let obs = Observation::for_player(s.game(), PlayerId(1));
        let protected = obs
            .shared_effects
            .as_deref()
            .unwrap_or_default()
            .iter()
            .find(|effect| effect.effect.modification == "CantBeCountered");
        if expected == ZoneType::Battlefield {
            encode(&obs, &ObservationEncoderConfig::default()).unwrap();
            assert_eq!(
                protected.unwrap().members[0].target.object_id,
                Some(obs.stack_objects[0].stack_object_id)
            );
        } else {
            assert!(protected.is_none());
        }
        cast_owned(&mut s, "Counterspell");
        assert!(s.choose_target(Target::StackSpell(spell)));
        settle(&mut s);
        assert_eq!(zone_of(&s, ogre), Some(expected));
        let jace = s.battlefield_permanents_named(0, "Jace")[0];
        assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 4);
    }
}

#[test]
fn empower_spells_and_graveyard_activation_keep_their_other_instructions() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::{counters_of, now, zone_of},
        play::{activate_card, cast_owned, choose_mode, settle},
    };
    use managym::state::{
        game_object::{PlayerId, Target},
        permanent::CounterKind,
        player::ManaPayment,
        zone::ZoneType,
    };
    let mut s = five_color_setup(&[
        "Campus Crier",
        "Hexhaven Battalion",
        "Rewrite Regrets",
        "Gray Ogre",
        "Vraska's Final Mercy",
        "Fatehold Charm",
        "Solve for Disappointment",
        "Tam's Resistance",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let crier = move_from_library(&mut s, 0, "Campus Crier", ZoneType::Graveyard);
    activate_card(&mut s, crier, 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, crier), Some(ZoneType::Exile));
    let jace = s.battlefield_permanents_named(0, "Jace")[0];
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 2);

    s.set_player_mana_pool(0, "4WW");
    cast_owned(&mut s, "Hexhaven Battalion");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Cadet").len(), 3);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 4);

    let ogre = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
    s.set_player_mana_pool(0, "3B");
    cast_owned(&mut s, "Rewrite Regrets");
    assert!(s.choose_target(Target::GraveyardCard(ogre)));
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Battlefield));
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 6);

    let life = s.life(0);
    s.set_player_mana_pool(0, "BB");
    cast_owned(&mut s, "Vraska's Final Mercy");
    choose_mode(&mut s, 1);
    settle(&mut s);
    assert_eq!(s.life(0), life - 2);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 12);

    s.set_player_mana_pool(0, "WU");
    cast_owned(&mut s, "Fatehold Charm");
    choose_mode(&mut s, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 14);

    let discard = move_from_library(&mut s, 1, "Gray Ogre", ZoneType::Hand);
    s.set_player_mana_pool(0, "1B");
    cast_owned(&mut s, "Solve for Disappointment");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(zone_of(&s, discard), Some(ZoneType::Graveyard));
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 15);

    let returned = s.game().state.permanent_of(now(&s, ogre)).unwrap();
    s.set_player_mana_pool(0, "1G");
    cast_owned(&mut s, "Tam's Resistance");
    assert!(s.choose_target(Target::Permanent(returned)));
    settle(&mut s);
    assert_eq!(counters_of(&s, returned, CounterKind::PlusOnePlusOne), 1);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 19);
}

#[test]
fn enforcer_counts_creatures_on_resolution_and_machinations_allows_one_activation() {
    use crate::support::{
        game::{
            clear_summoning_sickness, enter_battlefield, five_color_setup, put_onto_battlefield,
        },
        inspect::{card_of, counters_of},
        play::{activate, attack_with, cast_owned, offers_ability, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{permanent::CounterKind, player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Repurposed Enforcer", "Gray Ogre", "Jace's Machinations"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let enforcer = enter_battlefield(&mut s, "Repurposed Enforcer");
    clear_summoning_sickness(&mut s, enforcer);
    let ogre = put_onto_battlefield(&mut s, 0, "Gray Ogre");
    s.advance_to_step(StepKind::DeclareAttackers);
    attack_with(&mut s, enforcer);
    let removed = card_of(&s, ogre);
    s.game_mut().move_object(removed, ZoneType::Hand);
    settle(&mut s);
    let jace = s.battlefield_permanents_named(0, "Jace")[0];
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 1);
    s.advance_to_active_step(1, StepKind::Main);
    s.pass_priority();
    assert!(!offers_ability(&mut s, jace, 0));
    s.set_player_mana_pool(0, "2U");
    cast_owned(&mut s, "Jace's Machinations");
    settle(&mut s);
    if s.action_space().player == Some(managym::state::game_object::PlayerId(1)) {
        s.pass_priority();
    }
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 9);
    activate(&mut s, jace, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 8);
    assert!(!offers_ability(&mut s, jace, 1));
}

/// Each ultimate spends the last counter before its emblem enters. The emblem
/// stays on the original side when another copy or an opponent casts a spell.
#[test]
fn emblems_buff_only_their_controllers_creatures_after_the_walker_dies() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, set_counters},
        inspect::card_of,
        play::{activate, settle},
    };
    use managym::{
        agent::{
            observation::Observation,
            observation_encoder::{encode, FactKind, ObservationEncoderConfig, FACT_DIM},
            observation_layout::fact,
        },
        state::{game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
    };

    let mut s = five_color_setup(&["Ajani Resolute", "Vivien Reid", "Gray Ogre"]);
    let ogre = enter_battlefield(&mut s, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for (index, (name, cost)) in [
        ("Ajani Resolute", 10),
        ("Vivien Reid", 8),
        ("Vivien Reid", 8),
    ]
    .into_iter()
    .enumerate()
    {
        let walker = enter_battlefield(&mut s, name);
        set_counters(&mut s, walker, CounterKind::Loyalty, cost);
        activate(&mut s, walker, 2);
        settle(&mut s);
        assert!(s.battlefield_permanents_named(0, name).is_empty());
        let total = 2 + 2 * (index as i32 + 1);
        assert_eq!(s.game().object_pt(card_of(&s, ogre)), (total, total));
        assert_eq!(s.game().object_pt(card_of(&s, other)), (2, 2));
        for seat in [0, 1] {
            let obs = Observation::for_player(s.game(), PlayerId(seat));
            let (cards, permanents) = if seat == 0 {
                (&obs.agent_cards, &obs.agent_permanents)
            } else {
                (&obs.opponent_cards, &obs.opponent_permanents)
            };
            let emblems: Vec<_> = cards
                .iter()
                .filter(|row| row.zone == ZoneType::Command)
                .collect();
            assert_eq!(emblems.len(), index + 1);
            assert!(emblems
                .iter()
                .all(|e| !e.is_token && e.text_box.contains("+2/+2")));
            let view = permanents
                .iter()
                .find(|p| p.characteristics.name == "Gray Ogre")
                .unwrap();
            assert_eq!(view.characteristics.keywords.indestructible, index > 0);
            let rows = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
            assert_eq!(
                rows.facts
                    .chunks_exact(FACT_DIM)
                    .filter(|r| r[fact::KIND] == FactKind::EmblemAbility as u8 as f32)
                    .count(),
                index + 1
            );
        }
    }
}

#[test]
fn kaito_emblem_creates_ninjas_for_spells_cast_by_either_player() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, move_from_library, set_counters},
        play::{activate, cast_card, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{permanent::CounterKind, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Kaito, Cunning Infiltrator", "Gray Ogre"]);
    let kaito = enter_battlefield(&mut s, "Kaito, Cunning Infiltrator");
    set_counters(&mut s, kaito, CounterKind::Loyalty, 9);
    activate(&mut s, kaito, 2);
    settle(&mut s);
    for seat in [0, 1] {
        s.advance_to_active_step(seat, StepKind::Main);
        s.set_player_mana_pool(seat, "2R");
        let ogre = move_from_library(&mut s, seat, "Gray Ogre", ZoneType::Hand);
        cast_card(&mut s, ogre);
        settle(&mut s);
        assert_eq!(s.battlefield_permanents_named(0, "Ninja").len(), seat + 1);
        assert!(s.battlefield_permanents_named(1, "Ninja").is_empty());
    }
}

#[test]
fn chandra_offers_the_exiled_spell_at_full_cost_or_deals_two() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, pin_top},
        play::{activate, resolve_top, settle},
    };
    use managym::agent::action::Action;
    for (top, mana, cast) in [
        ("Gray Ogre", "2R", true),
        ("Gray Ogre", "2R", false),
        ("Gray Ogre", "", false),
        ("Mountain", "2R", false),
    ] {
        let mut s = five_color_setup(&["Chandra, Torch of Defiance", "Gray Ogre"]);
        let chandra = enter_battlefield(&mut s, "Chandra, Torch of Defiance");
        pin_top(&mut s, &[top]);
        s.set_player_mana_pool(0, mana);
        let life = s.life(1);
        activate(&mut s, chandra, 0);
        resolve_top(&mut s);
        let candidate = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::SelectCard { .. }));
        assert_eq!(candidate.is_some(), top == "Gray Ogre" && mana == "2R");
        if cast {
            s.step_action(candidate.unwrap());
        }
        settle(&mut s);
        assert_eq!(s.life(1), life - if cast { 0 } else { 2 });
        assert_eq!(
            s.battlefield_permanents_named(0, "Gray Ogre").len(),
            usize::from(cast)
        );
    }
}

#[test]
fn chandra_emblem_is_the_source_of_its_targeted_damage() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, set_counters},
        play::{activate, cast_owned, settle, settle_targeting_opponent},
    };
    use managym::{
        flow::event::GameEvent,
        state::{game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Chandra, Torch of Defiance", "Gray Ogre"]);
    let walker = enter_battlefield(&mut s, "Chandra, Torch of Defiance");
    set_counters(&mut s, walker, CounterKind::Loyalty, 7);
    activate(&mut s, walker, 3);
    settle(&mut s);
    let emblem = s
        .game()
        .state
        .zones
        .top(ZoneType::Command, PlayerId(0))
        .unwrap();
    let life = s.life(1);
    cast_owned(&mut s, "Gray Ogre");
    if s.game().state.stack_objects.is_empty() {
        crate::support::play::pay_rest(&mut s);
    }
    settle_targeting_opponent(&mut s);
    assert_eq!(s.life(1), life - 5);
    assert!(s.game().state.events.iter().any(|event| matches!(event, GameEvent::DamageDealt {source: Some(source), amount: 5, ..} if *source == emblem)));
}

#[test]
fn ajanis_pridemate_and_ajani_each_trigger_on_the_same_life_gain() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, set_counters},
        inspect::counters_of,
        play::{activate, settle},
    };
    use managym::{flow::turn::StepKind, state::permanent::CounterKind};
    let mut s = five_color_setup(&["Ajani Resolute"]);
    let ajani = enter_battlefield(&mut s, "Ajani Resolute");
    set_counters(&mut s, ajani, CounterKind::Loyalty, 5);
    activate(&mut s, ajani, 1);
    settle(&mut s);
    let cat = s.battlefield_permanents_named(0, "Ajani's Pridemate")[0];
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let life = s.life(0);
    activate(&mut s, ajani, 0);
    settle(&mut s);
    assert_eq!(s.life(0), life + 1);
    assert_eq!(counters_of(&s, ajani, CounterKind::Loyalty), 2);
    assert_eq!(counters_of(&s, cat, CounterKind::PlusOnePlusOne), 1);
}

#[test]
fn vivien_selects_a_creature_from_four_and_can_destroy_a_flyer() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, pin_top},
        inspect::{card_of, zone_of},
        play::{activate, choose, resolve_top, settle},
    };
    use managym::{
        agent::action::Action,
        flow::turn::StepKind,
        state::{game_object::Target, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Vivien Reid", "Gray Ogre", "Lightning Bolt", "Wind Drake"]);
    let vivien = enter_battlefield(&mut s, "Vivien Reid");
    let top = pin_top(
        &mut s,
        &["Lightning Bolt", "Gray Ogre", "Forest", "Mountain"],
    );
    activate(&mut s, vivien, 0);
    resolve_top(&mut s);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::SelectCard {object, ..} if *object == top[0])));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard {object, ..} if *object == top[1]),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, top[1]), Some(ZoneType::Hand));
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let ogre = enter_battlefield(&mut s, "Gray Ogre");
    let flyer = enter_battlefield(&mut s, "Wind Drake");
    let object = card_of(&s, flyer);
    activate(&mut s, vivien, 1);
    assert!(!s.choose_target(Target::Permanent(ogre)));
    assert!(s.choose_target(Target::Permanent(flyer)));
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
}

#[test]
fn kaitos_ninja_hits_a_player_and_adds_loyalty() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup},
        inspect::counters_of,
        play::{activate, attack_with, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{game_object::Target, permanent::CounterKind},
    };
    let mut s = five_color_setup(&["Kaito, Cunning Infiltrator"]);
    let kaito = enter_battlefield(&mut s, "Kaito, Cunning Infiltrator");
    activate(&mut s, kaito, 1);
    settle(&mut s);
    let ninja = s.battlefield_permanents_named(0, "Ninja")[0];
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    activate(&mut s, kaito, 0);
    assert!(s.choose_target(Target::Permanent(ninja)));
    settle(&mut s);
    s.advance_to_step(StepKind::DeclareAttackers);
    let life = s.life(1);
    attack_with(&mut s, ninja);
    s.advance_to_step(StepKind::End);
    assert_eq!(s.life(1), life - 2);
    assert_eq!(counters_of(&s, kaito, CounterKind::Loyalty), 3);
}

#[test]
fn chandras_red_mana_uses_the_stack_and_her_damage_targets_a_creature() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup},
        inspect::{card_of, zone_of},
        play::{activate, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{game_object::Target, mana::Mana, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Chandra, Torch of Defiance", "Gray Ogre"]);
    let chandra = enter_battlefield(&mut s, "Chandra, Torch of Defiance");
    s.set_player_mana_pool(0, "");
    activate(&mut s, chandra, 1);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.game().state.players[0].mana_pool.mana(), &Mana::default());
    settle(&mut s);
    assert_eq!(
        s.game().state.players[0].mana_pool.mana(),
        &Mana::parse("RR")
    );
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let ogre = enter_battlefield(&mut s, "Gray Ogre");
    let object = card_of(&s, ogre);
    activate(&mut s, chandra, 2);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
}
