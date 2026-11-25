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

/// The Way cycle's printed index for a planeswalker's first granted ability:
/// grants follow its own abilities in timestamp order.
fn printed_abilities(
    s: &crate::support::game::Scenario,
    walker: managym::state::game_object::PermanentId,
) -> usize {
    let object = crate::support::inspect::card_of(s, walker);
    s.game().state[object].activated_abilities.len()
}

/// A Way grants every planeswalker its controller controls a loyalty ability
/// that shares the holder's one loyalty activation each turn and leaves with
/// the Way. Four Ways over six planeswalkers offer every grant on each.
#[test]
fn way_grants_share_the_holders_loyalty_quota_and_leave_with_the_way() {
    use crate::support::{
        game::{five_color_setup, move_card, set_counters},
        inspect::{card_of, counters_of, keywords, offer_labels},
        play::{activate, offers_ability, settle},
    };
    use managym::{
        agent::action::Action,
        flow::turn::StepKind,
        state::{card::Keyword, permanent::CounterKind, zone::ZoneType},
    };
    let theorist = "The Theorist, Jace Beleren";
    let walkers = [
        theorist,
        "Ajani Resolute",
        "Vivien Reid",
        "Kaito, Cunning Infiltrator",
        "Chandra, Torch of Defiance",
    ];
    let ways = [
        "Way of the Wildspeaker",
        "Way of the Healer",
        "Way of the Warlord",
        "Way of the Deathbringer",
    ];
    let mut names = walkers.to_vec();
    names.extend(ways);
    names.push("Way of the Mentor");
    let mut s = five_color_setup(&names);
    let jace = s.force_permanent_on_battlefield(0, theorist);
    let wildspeaker = s.force_permanent_on_battlefield(0, ways[0]);
    set_counters(&mut s, jace, CounterKind::Loyalty, 6);
    let granted = printed_abilities(&s, jace);
    assert_eq!(granted, 3);
    assert!(offers_ability(&mut s, jace, granted));
    assert!(offer_labels(&s)
        .iter()
        .any(|label| label.contains("[−4]") && label.contains("Beast")));
    activate(&mut s, jace, granted);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 2);
    settle(&mut s);
    let beasts = s.battlefield_permanents_named(0, "Beast");
    assert_eq!(beasts.len(), 1);
    assert!(keywords(&s, beasts[0]).has(Keyword::Trample));
    assert!(!(0..=granted).any(|index| offers_ability(&mut s, jace, index)));

    // A printed activation spends the same quota as the granted one.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    set_counters(&mut s, jace, CounterKind::Loyalty, 6);
    activate(&mut s, jace, 0);
    settle(&mut s);
    assert!(!offers_ability(&mut s, jace, granted));

    // The grant ends when the Way leaves the battlefield.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(offers_ability(&mut s, jace, granted));
    let object = card_of(&s, wildspeaker);
    move_card(&mut s, object, ZoneType::Graveyard);
    assert!(!offers_ability(&mut s, jace, granted));
    assert_eq!(
        s.game().characteristics(jace).activated_abilities.len(),
        granted
    );

    // Four Ways over six planeswalkers: every holder offers every grant.
    for way in ways {
        s.force_permanent_on_battlefield(0, way);
    }
    let mut holders = vec![jace];
    for walker in &walkers[1..] {
        let holder = s.force_permanent_on_battlefield(0, walker);
        set_counters(&mut s, holder, CounterKind::Loyalty, 10);
        holders.push(holder);
    }
    // The sixth planeswalker is the Jace token Way of the Mentor empowers.
    crate::support::play::enter_and_settle(&mut s, "Way of the Mentor");
    holders.extend(s.battlefield_permanents_named(0, "Jace"));
    assert_eq!(holders.len(), 6);
    for walker in &holders {
        set_counters(&mut s, *walker, CounterKind::Loyalty, 10);
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    for walker in holders {
        let printed = printed_abilities(&s, walker);
        let offered: Vec<_> = s
            .action_space()
            .actions
            .iter()
            .filter_map(|action| match action {
                Action::ActivateAbility {
                    permanent,
                    ability_index,
                    ..
                } if *permanent == walker && *ability_index >= printed => Some(*ability_index),
                _ => None,
            })
            .collect();
        assert_eq!(offered, (printed..printed + 4).collect::<Vec<_>>());
    }
}

/// Way of the Warlord's two target clauses are independent: the player is
/// still dealt damage when no creature or planeswalker is chosen.
#[test]
fn warlord_grant_hits_the_player_with_or_without_a_creature_target() {
    use crate::support::{
        game::{five_color_setup, set_counters},
        play::{activate, choose, settle},
    };
    use managym::{
        agent::action::Action,
        state::{
            game_object::{PlayerId, Target},
            permanent::CounterKind,
        },
    };
    let theorist = "The Theorist, Jace Beleren";
    for target_ogre in [false, true] {
        let mut s = five_color_setup(&[theorist, "Way of the Warlord", "Gray Ogre"]);
        let jace = s.force_permanent_on_battlefield(0, theorist);
        s.force_permanent_on_battlefield(0, "Way of the Warlord");
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        set_counters(&mut s, jace, CounterKind::Loyalty, 4);
        let life = s.life(1);
        activate(&mut s, jace, 3);
        if target_ogre {
            assert!(s.choose_target(Target::Permanent(ogre)));
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        settle(&mut s);
        assert_eq!(s.life(1), life - 2);
        assert_eq!(s.game().state.permanents[ogre].is_none(), target_ogre);
        assert!(s.game().state.permanents[jace].is_none());
    }
}

/// Granted abilities whose instructions already compose: Healer's Cadet and
/// surveil, Pyromancer's {R} on the stack, Deathbringer's optional sacrifice,
/// Cryomancer's copy of the next instant and Avatar's counters per land.
#[test]
fn way_grants_resolve_through_their_holders() {
    use crate::support::{
        game::{five_color_setup, move_from_library, set_counters},
        inspect::{card_of, counters_of},
        play::{
            activate, cast_owned, choose, fork_choosing, fork_selecting, play_land, resolve_top,
            settle, target_opponent,
        },
    };
    use managym::{
        agent::action::Action,
        state::{permanent::CounterKind, zone::ZoneType},
    };
    let theorist = "The Theorist, Jace Beleren";
    let setup = |way: &str, extra: &[&str]| {
        let mut names = vec![theorist, way];
        names.extend(extra);
        let mut s = five_color_setup(&names);
        let jace = s.force_permanent_on_battlefield(0, theorist);
        s.force_permanent_on_battlefield(0, way);
        set_counters(&mut s, jace, CounterKind::Loyalty, 10);
        (s, jace)
    };

    let (mut s, jace) = setup("Way of the Healer", &[]);
    activate(&mut s, jace, 3);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Cadet").len(), 1);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 8);

    let (mut s, jace) = setup("Way of the Pyromancer", &[]);
    s.set_player_mana_pool(0, "");
    s.allow_priority_floats();
    activate(&mut s, jace, 3);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 1);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 11);

    let (mut s, jace) = setup("Way of the Deathbringer", &["Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    activate(&mut s, jace, 3);
    resolve_top(&mut s);
    let mut sacrificed = fork_selecting(&s, card_of(&s, ogre));
    settle(&mut sacrificed);
    assert!(sacrificed.game().state.permanents[ogre].is_none());
    assert_eq!(sacrificed.battlefield_permanents_named(0, "Beast").len(), 1);
    let mut kept = fork_choosing(&s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut kept);
    assert!(kept.game().state.permanents[ogre].is_some());
    assert!(kept.battlefield_permanents_named(0, "Beast").is_empty());

    let (mut s, jace) = setup("Way of the Cryomancer", &["Lightning Bolt"]);
    activate(&mut s, jace, 3);
    settle(&mut s);
    let total = s.life(0) + s.life(1);
    s.set_player_mana_pool(0, "R");
    cast_owned(&mut s, "Lightning Bolt");
    target_opponent(&mut s);
    settle(&mut s);
    assert_eq!(s.life(0) + s.life(1), total - 6, "the Bolt and its copy");

    let (mut s, jace) = setup("Avatar of Burgeoning Echoes", &[]);
    let avatar = s.battlefield_permanents_named(0, "Avatar of Burgeoning Echoes")[0];
    move_from_library(&mut s, 0, "Forest", ZoneType::Hand);
    play_land(&mut s, "Forest");
    settle(&mut s);
    let tokens = s.battlefield_permanents_named(0, "Jace");
    assert_eq!(tokens.len(), 1, "landfall empowers a Jace token");
    assert_eq!(counters_of(&s, tokens[0], CounterKind::Loyalty), 2);
    let lands: usize = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        .iter()
        .map(|land| s.battlefield_permanents_named(0, land).len())
        .sum();
    activate(&mut s, jace, 3);
    choose(
        &mut s,
        |a| matches!(a, Action::ChooseTarget { target: managym::state::game_object::Target::Permanent(p), .. } if *p == avatar),
    );
    settle(&mut s);
    assert_eq!(
        counters_of(&s, avatar, CounterKind::PlusOnePlusOne),
        lands as i32
    );
}

/// Sanctum Lurker keeps planeswalkers at 0 loyalty, grants them a +2 they
/// can still activate, and exempts nothing else: the legend rule applies,
/// and they die once the Lurker leaves.
#[test]
fn sanctum_lurker_keeps_zero_loyalty_planeswalkers_until_it_leaves() {
    use crate::support::{
        game::{five_color_setup, move_card},
        inspect::{card_of, counters_of},
        play::{activate, cast_owned, enter_and_settle, settle, target_opponent},
    };
    use managym::{
        flow::turn::StepKind,
        state::{permanent::CounterKind, zone::ZoneType},
    };
    let theorist = "The Theorist, Jace Beleren";
    let mut s = five_color_setup(&["Sanctum Lurker", theorist, "Lightning Bolt"]);
    enter_and_settle(&mut s, "Sanctum Lurker");
    let jace = s.battlefield_permanents_named(0, "Jace")[0];
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 1);
    activate(&mut s, jace, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 0);
    assert!(
        s.game().state.permanents[jace].is_some(),
        "0 loyalty is kept"
    );

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let life = (s.life(0), s.life(1));
    activate(&mut s, jace, 2);
    settle(&mut s);
    assert_eq!((s.life(0), s.life(1)), (life.0 + 1, life.1 - 1));
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 2);

    let first = s.force_permanent_on_battlefield(0, theorist);
    let second = s.force_permanent_on_battlefield(0, theorist);
    s.set_player_mana_pool(0, "R");
    cast_owned(&mut s, "Lightning Bolt");
    target_opponent(&mut s);
    settle(&mut s);
    assert_eq!(
        [first, second]
            .iter()
            .filter(|walker| s.game().state.permanents[**walker].is_some())
            .count(),
        1,
        "the legend rule still applies"
    );

    let lurker = s.battlefield_permanents_named(0, "Sanctum Lurker")[0];
    crate::support::game::set_counters(&mut s, jace, CounterKind::Loyalty, 0);
    let object = card_of(&s, lurker);
    move_card(&mut s, object, ZoneType::Graveyard);
    s.set_player_mana_pool(0, "R");
    cast_owned(&mut s, "Lightning Bolt");
    target_opponent(&mut s);
    settle(&mut s);
    assert!(s.game().state.permanents[jace].is_none());
}

/// Loyalty activations are events: Mind Sculptor draws only for one that
/// removed two or more counters, Paradox gains life and a land play for
/// each, printed or granted, and the turn's count reaches the policy.
#[test]
fn loyalty_activation_triggers_read_the_cost_paid() {
    use crate::support::{
        game::{five_color_setup, move_from_library, set_counters},
        inspect::hand,
        play::{activate, choose, settle},
    };
    use managym::{
        agent::{
            action::Action,
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig, TURN_STATE_DIM},
            observation_layout::turn_state,
        },
        flow::turn::StepKind,
        state::{game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
    };
    let theorist = "The Theorist, Jace Beleren";
    let mut s = five_color_setup(&[
        theorist,
        "Ajani Resolute",
        "Way of the Mind Sculptor",
        "Way of the Paradox",
        "Way of the Wildspeaker",
    ]);
    s.force_permanent_on_battlefield(0, "Way of the Mind Sculptor");
    s.force_permanent_on_battlefield(0, "Way of the Paradox");
    s.force_permanent_on_battlefield(0, "Way of the Wildspeaker");
    let jace = s.force_permanent_on_battlefield(0, theorist);
    let ajani = s.force_permanent_on_battlefield(0, "Ajani Resolute");
    set_counters(&mut s, jace, CounterKind::Loyalty, 6);
    set_counters(&mut s, ajani, CounterKind::Loyalty, 6);
    let lands: Vec<_> = (0..3)
        .map(|_| move_from_library(&mut s, 0, "Forest", ZoneType::Hand))
        .collect();
    let (cards, life) = (hand(&s, 0).len(), s.life(0));

    // +1 removes nothing: no draw, one life and one land play.
    activate(&mut s, jace, 0);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), cards);
    // Ajani's 0 gains a life too; its own trigger adds a loyalty counter.
    activate(&mut s, ajani, 0);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), cards);
    assert_eq!(s.life(0), life + 3);
    assert_eq!(
        s.game().state.turn.history[0].loyalty_abilities_activated,
        2
    );
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let input = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
    assert_eq!(
        input.player_turn_state[turn_state::LOYALTY_ABILITIES_ACTIVATED],
        2.0
    );
    assert_eq!(input.player_turn_state.len(), 2 * TURN_STATE_DIM);
    for land in &lands {
        choose(
            &mut s,
            |a| matches!(a, Action::PlayLand { object, .. } if object == land),
        );
    }
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::PlayLand { .. })));

    // A granted [−4] removes four: Mind Sculptor draws.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert_eq!(
        s.game().state.turn.history[0].loyalty_abilities_activated,
        0
    );
    let cards = hand(&s, 0).len();
    activate(&mut s, jace, 3);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), cards + 1);
}

/// Kiora's attack trigger checks the turn's loyalty activations; her grant
/// makes a hexproof Leviathan. Gideon punishes each opposing creature entry
/// and loyalty activation.
#[test]
fn kiora_and_gideon_read_loyalty_activations() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup, set_counters},
        inspect::{keywords, observed, tapped},
        play::{activate, attack_with, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{card::Keyword, permanent::CounterKind},
    };
    let theorist = "The Theorist, Jace Beleren";
    for activated in [false, true] {
        let mut s = five_color_setup(&[theorist, "Kiora of Salt and Sand", "Gray Ogre"]);
        s.force_permanent_on_battlefield(0, "Kiora of Salt and Sand");
        let jace = s.force_permanent_on_battlefield(0, theorist);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        clear_summoning_sickness(&mut s, ogre);
        set_counters(&mut s, jace, CounterKind::Loyalty, 8);
        if activated {
            activate(&mut s, jace, 3);
            settle(&mut s);
            let leviathans = s.battlefield_permanents_named(0, "Leviathan");
            assert_eq!(leviathans.len(), 1);
            assert!(keywords(&s, leviathans[0]).has(Keyword::Hexproof));
        }
        s.advance_to_step(StepKind::DeclareAttackers);
        attack_with(&mut s, ogre);
        settle(&mut s);
        assert_eq!(!tapped(&s, ogre), activated);
        assert_eq!(observed(&s, ogre).cant_be_blocked_this_turn, activated);
    }

    let mut s = five_color_setup(&[theorist, "Gideon the Oathless"]);
    s.force_permanent_on_battlefield(0, "Gideon the Oathless");
    let jace = s.force_permanent_on_battlefield(1, theorist);
    set_counters(&mut s, jace, CounterKind::Loyalty, 3);
    s.advance_to_active_step(1, StepKind::Main);
    let life = s.life(1);
    activate(&mut s, jace, 0);
    settle(&mut s);
    assert_eq!(
        s.life(1),
        life - 2,
        "one for the activation, one for the Illusion"
    );
}

#[test]
fn ajani_unrelenting_loyalty_abilities() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, move_to_hand},
        inspect::{counters_of, keywords},
        play::{activate, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{card::Keyword, game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
    };

    // +1 triggers Cadet creation, then pumps team +1/+0 and haste until end of turn.
    let mut s = five_color_setup(&["Ajani Unrelenting", "Gray Ogre", "Plains"]);
    let ajani = enter_battlefield(&mut s, "Ajani Unrelenting");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    assert_eq!(counters_of(&s, ajani, CounterKind::Loyalty), 5);
    activate(&mut s, ajani, 0);
    settle(&mut s);
    // Trigger resolves BEFORE +1 (stack LIFO), so the newly created Cadet also gets +1/+0 and haste!
    let cadets = s.battlefield_permanents_named(0, "Cadet");
    assert_eq!(cadets.len(), 1);
    assert_eq!(s.game().characteristics(ogre).pt(), (3, 2));
    assert!(keywords(&s, ogre).has(Keyword::Haste));
    assert_eq!(s.game().characteristics(cadets[0]).pt(), (3, 2));
    assert!(keywords(&s, cadets[0]).has(Keyword::Haste));

    // -2 discards your hand, then draws a card for each creature you control (including the new Cadet!).
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    move_to_hand(&mut s, "Plains");
    move_to_hand(&mut s, "Plains");
    let creatures_before = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Battlefield, PlayerId(0))
        .iter()
        .filter_map(|&obj| s.game().state.permanent_of(obj))
        .filter(|&id| s.game().characteristics(id).is_creature())
        .count();
    activate(&mut s, ajani, 1);
    settle(&mut s);
    // Cadet trigger resolved before -2, so creatures_before + 1 creatures are in play.
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        creatures_before + 1
    );

    // -3 deals 4 damage to each creature EXCEPT tokens you control:
    // own nontoken creature dies, opponent's token and nontoken die, own Cadet tokens survive!
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let opp_nontoken = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let opp_token = s
        .game_mut()
        .create_token("token.cadet", PlayerId(1), false)
        .unwrap();
    let own_cadets_before = s.battlefield_permanents_named(0, "Cadet").len();
    activate(&mut s, ajani, 2);
    settle(&mut s);
    assert!(
        s.game().state.permanents[ogre].is_none(),
        "own nontoken takes 4"
    );
    assert!(
        s.game().state.permanents[opp_nontoken].is_none(),
        "opponent nontoken takes 4"
    );
    assert!(
        s.game().state.permanents[opp_token].is_none(),
        "opponent token takes 4"
    );
    assert_eq!(
        s.battlefield_permanents_named(0, "Cadet").len(),
        own_cadets_before + 1,
        "own tokens are exempt from -3"
    );
}

#[test]
fn garruk_curse_breaker_loyalty_abilities() {
    use crate::support::{
        game::{clear_summoning_sickness, enter_battlefield, five_color_setup, set_counters},
        inspect::{keywords, tapped},
        play::{activate, attack_with, choose, settle},
    };
    use managym::{
        agent::action::Action,
        flow::{combat::AttackTarget, turn::StepKind},
        state::{
            card::Keyword, game_object::PlayerId, permanent::CounterKind, target::Target,
            zone::ZoneType,
        },
    };

    // Trigger: power >= 4 creature entering draws a card; power < 4 does not.
    // -3 creates a 4/4 green Beast with trample, which triggers Garruk's draw!
    let mut s = five_color_setup(&[
        "Garruk, Curse Breaker",
        "Gray Ogre",
        "Plains",
        "Ajani, Caller of the Pride",
    ]);
    let garruk = enter_battlefield(&mut s, "Garruk, Curse Breaker");
    let hand0 = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    let ogre = enter_battlefield(&mut s, "Gray Ogre");
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand0,
        "2/2 does not trigger power >= 4 draw"
    );
    activate(&mut s, garruk, 1); // -3 Beast
    settle(&mut s);
    let beasts = s.battlefield_permanents_named(0, "Beast");
    assert_eq!(beasts.len(), 1);
    assert!(keywords(&s, beasts[0]).has(Keyword::Trample));
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand0 + 1,
        "4/4 Beast triggers Garruk draw"
    );

    // +2 untaps up to two target lands.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let land1 = s.force_permanent_on_battlefield(0, "Plains");
    let land2 = s.force_permanent_on_battlefield(0, "Plains");
    s.game_mut().state.permanents[land1]
        .as_mut()
        .unwrap()
        .tapped = true;
    s.game_mut().state.permanents[land2]
        .as_mut()
        .unwrap()
        .tapped = true;
    activate(&mut s, garruk, 0);
    assert!(s.choose_target(Target::Permanent(land1)));
    assert!(s.choose_target(Target::Permanent(land2)));
    settle(&mut s);
    assert!(!tapped(&s, land1) && !tapped(&s, land2));

    // -4: until your next turn, whenever one or more creatures attack an opponent,
    // ONLY those attacking the opponent (not an opponent's planeswalker) get +2/+2 and trample until EOT.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    set_counters(&mut s, garruk, CounterKind::Loyalty, 6);
    let opp_pw = s.force_permanent_on_battlefield(1, "Ajani, Caller of the Pride");
    set_counters(&mut s, opp_pw, CounterKind::Loyalty, 4);
    let opp_pw_obj = s.game().permanent_object(opp_pw).unwrap();
    let other_ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    clear_summoning_sickness(&mut s, other_ogre);
    activate(&mut s, garruk, 2); // -4
    settle(&mut s);
    s.advance_to_step(StepKind::DeclareAttackers);
    // Attack opponent with ogre, attack opponent's planeswalker with other_ogre, decline any other creatures.
    while s.action_space().kind == managym::agent::action::ActionSpaceKind::DeclareAttacker {
        let offered = match s.action_space().actions[0] {
            Action::DeclareAttacker { permanent, .. } => permanent,
            _ => unreachable!(),
        };
        if offered == ogre {
            choose(
                &mut s,
                |a| matches!(a, Action::DeclareAttacker { permanent, attack: Some(AttackTarget::Player(p)), .. } if *permanent == ogre && *p == PlayerId(1)),
            );
        } else if offered == other_ogre {
            choose(
                &mut s,
                |a| matches!(a, Action::DeclareAttacker { permanent, attack: Some(AttackTarget::Permanent(obj)), .. } if *permanent == other_ogre && *obj == opp_pw_obj),
            );
        } else {
            choose(&mut s, |a| {
                matches!(a, Action::DeclareAttacker { attack: None, .. })
            });
        }
    }
    settle(&mut s);
    assert_eq!(
        s.game().characteristics(ogre).pt(),
        (4, 4),
        "attacker at opponent gets +2/+2"
    );
    assert!(keywords(&s, ogre).has(Keyword::Trample));
    assert_eq!(
        s.game().characteristics(other_ogre).pt(),
        (2, 2),
        "attacker at opponent's planeswalker does not trigger -4"
    );
    assert!(!keywords(&s, other_ogre).has(Keyword::Trample));

    // On your next turn, the delayed trigger has expired.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, ogre);
    settle(&mut s);
    assert_eq!(
        s.game().characteristics(ogre).pt(),
        (2, 2),
        "-4 expired at your next turn"
    );
}

#[test]
fn garruk_veiled_butcher_loyalty_abilities() {
    use crate::support::{
        game::{
            clear_hand, enter_battlefield, five_color_setup, force_new_card_in_hand, set_counters,
        },
        inspect::{card_of, now, zone_of},
        play::{activate, choose, settle},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        flow::turn::StepKind,
        state::{game_object::PlayerId, permanent::CounterKind, target::Target, zone::ZoneType},
    };

    // Static replacement: if an opponent's creature would die, exile it instead.
    // +2: up to one target creature gets -4/-1 until your next turn.
    let mut s = five_color_setup(&[
        "Garruk, Veiled Butcher",
        "Hill Giant",
        "Gray Ogre",
        "Plains",
    ]);
    let garruk = enter_battlefield(&mut s, "Garruk, Veiled Butcher");
    let opp_giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    activate(&mut s, garruk, 0); // +2
    assert!(s.choose_target(Target::Permanent(opp_giant)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(opp_giant).pt(), (-1, 2));
    // Still shrunk during opponent's turn.
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(opp_giant).pt(), (-1, 2));
    // Wears off when your next turn begins.
    s.advance_to_active_step(0, StepKind::Main);
    assert_eq!(s.game().characteristics(opp_giant).pt(), (3, 3));

    // -2: each player sacrifices a creature; Beast only if YOU sacrificed one.
    // First, with NO creature on seat 0: opponent sacrifices opp_giant (exiled by Garruk's static!), seat 0 gets NO Beast.
    let opp_giant_obj = card_of(&s, opp_giant);
    activate(&mut s, garruk, 1); // -2
    settle(&mut s);
    assert!(s.game().state.permanents[opp_giant].is_none());
    assert_eq!(
        zone_of(&s, now(&s, opp_giant_obj)),
        Some(ZoneType::Exile),
        "Garruk's static exiles opponent's dying creature"
    );
    assert!(
        s.battlefield_permanents_named(0, "Beast").is_empty(),
        "no Beast when you didn't sacrifice a creature"
    );

    // Next turn, seat 0 DOES have a creature to sacrifice -> creates 4/4 Beast (and own creature goes to graveyard, not exile).
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let own_ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let own_ogre_obj = card_of(&s, own_ogre);
    activate(&mut s, garruk, 1);
    settle(&mut s);
    assert_eq!(
        zone_of(&s, now(&s, own_ogre_obj)),
        Some(ZoneType::Graveyard),
        "Garruk's static only exiles opponents' creatures"
    );
    assert_eq!(s.battlefield_permanents_named(0, "Beast").len(), 1);

    // -3: each opponent discards 2 cards; draw a card unless they discarded 2 nonland cards.
    // Case A: opponent discards 2 nonlands -> 0 draws.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    set_counters(&mut s, garruk, CounterKind::Loyalty, 6);
    clear_hand(&mut s, 1);
    force_new_card_in_hand(&mut s, 1, "Gray Ogre");
    force_new_card_in_hand(&mut s, 1, "Hill Giant");
    let hand_before = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    activate(&mut s, garruk, 2);
    while s.action_space().kind == ActionSpaceKind::Priority {
        choose(&mut s, |a| matches!(a, Action::PassPriority { .. }));
    }
    while matches!(
        s.action_space().kind,
        ActionSpaceKind::Discard | ActionSpaceKind::LookAndSelect
    ) {
        choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    }
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand_before,
        "2 nonlands discarded -> no draw"
    );

    // Case B: opponent discards 1 land + 1 nonland -> 1 draw!
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    set_counters(&mut s, garruk, CounterKind::Loyalty, 6);
    clear_hand(&mut s, 1);
    force_new_card_in_hand(&mut s, 1, "Plains");
    force_new_card_in_hand(&mut s, 1, "Gray Ogre");
    let hand_before = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    activate(&mut s, garruk, 2);
    while s.action_space().kind == ActionSpaceKind::Priority {
        choose(&mut s, |a| matches!(a, Action::PassPriority { .. }));
    }
    while matches!(
        s.action_space().kind,
        ActionSpaceKind::Discard | ActionSpaceKind::LookAndSelect
    ) {
        choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    }
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand_before + 1,
        "1 land + 1 nonland discarded -> draw 1"
    );
}

#[test]
fn jace_reality_sculptor_loyalty_abilities() {
    use crate::support::{
        game::{clear_summoning_sickness, enter_battlefield, five_color_setup, set_counters},
        inspect::{counters_of, has_action},
        play::{activate, choose, settle},
    };
    use managym::{
        agent::action::Action,
        flow::{combat::AttackTarget, turn::StepKind},
        state::{game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
    };

    // +1: Empower Jace X where X = Islands you control.
    let mut s = five_color_setup(&["Jace, Reality Sculptor", "Island", "Hill Giant"]);
    let jace = enter_battlefield(&mut s, "Jace, Reality Sculptor");
    for _ in 0..3 {
        s.force_permanent_on_battlefield(0, "Island");
    }
    activate(&mut s, jace, 0); // +1
    settle(&mut s);
    let jace_tokens = s.battlefield_permanents_named(0, "Jace");
    assert_eq!(jace_tokens.len(), 1);
    assert_eq!(
        counters_of(&s, jace_tokens[0], CounterKind::Loyalty),
        3,
        "empowered by 3 Islands"
    );
    assert_eq!(counters_of(&s, jace, CounterKind::Loyalty), 6);

    // -3: until your next turn, whenever a creature attacks you or a planeswalker you control, -5/-0 until EOT.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let opp_a = s.force_permanent_on_battlefield(1, "Hill Giant");
    let opp_b = s.force_permanent_on_battlefield(1, "Hill Giant");
    clear_summoning_sickness(&mut s, opp_a);
    clear_summoning_sickness(&mut s, opp_b);
    let jace_obj = s.game().permanent_object(jace).unwrap();
    activate(&mut s, jace, 1); // -3
    settle(&mut s);
    s.advance_to_active_step(1, StepKind::DeclareAttackers);
    // Attack seat 0 with opp_a and Jace with opp_b.
    while s.action_space().kind == managym::agent::action::ActionSpaceKind::DeclareAttacker {
        let offered = match s.action_space().actions[0] {
            Action::DeclareAttacker { permanent, .. } => permanent,
            _ => unreachable!(),
        };
        if offered == opp_a {
            choose(
                &mut s,
                |a| matches!(a, Action::DeclareAttacker { permanent, attack: Some(AttackTarget::Player(p)), .. } if *permanent == opp_a && *p == PlayerId(0)),
            );
        } else if offered == opp_b {
            choose(
                &mut s,
                |a| matches!(a, Action::DeclareAttacker { permanent, attack: Some(AttackTarget::Permanent(obj)), .. } if *permanent == opp_b && *obj == jace_obj),
            );
        } else {
            choose(&mut s, |a| {
                matches!(a, Action::DeclareAttacker { attack: None, .. })
            });
        }
    }
    settle(&mut s);
    assert_eq!(
        s.game().characteristics(opp_a).pt(),
        (-2, 3),
        "attacking you gets -5/-0"
    );
    assert_eq!(
        s.game().characteristics(opp_b).pt(),
        (-2, 3),
        "attacking your planeswalker gets -5/-0"
    );

    // 0: gated on >= 25 loyalty across Jaces you control; exiles all but bottom card of opponent's library.
    s.advance_to_active_step(0, StepKind::Main);
    set_counters(&mut s, jace, CounterKind::Loyalty, 12);
    set_counters(&mut s, jace_tokens[0], CounterKind::Loyalty, 12);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        !has_action(
            &s,
            |a| matches!(a, Action::ActivateAbility { permanent, ability_index: 2, .. } if *permanent == jace)
        ),
        "24 total Jace loyalty is not enough for 0 ability"
    );
    set_counters(&mut s, jace_tokens[0], CounterKind::Loyalty, 13);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        has_action(
            &s,
            |a| matches!(a, Action::ActivateAbility { permanent, ability_index: 2, .. } if *permanent == jace)
        ),
        "25 total Jace loyalty unlocks 0 ability"
    );
    let opp_lib_before = s.game().state.zones.size(ZoneType::Library, PlayerId(1));
    assert!(opp_lib_before > 1);
    let bottom_card = *s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(1))
        .iter()
        .next()
        .unwrap();
    activate(&mut s, jace, 2); // 0
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Library, PlayerId(1)),
        1,
        "only the bottom card remains"
    );
    assert_eq!(
        *s.game()
            .state
            .zones
            .zone_objects(ZoneType::Library, PlayerId(1))
            .iter()
            .next()
            .unwrap(),
        bottom_card,
        "exact bottom card was preserved"
    );
    assert_eq!(
        s.game().state.zones.size(ZoneType::Exile, PlayerId(1)),
        opp_lib_before - 1
    );
}
