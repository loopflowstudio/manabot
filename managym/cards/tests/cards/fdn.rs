//! Foundations behavior through ordinary activation, payment and combat.
use crate::support::{
    game::{five_color_setup, move_card, Scenario},
    inspect::{card_of, permanent_named, permanent_of},
    play::{activate, attack_with, enter_and_settle, settle, spend_colorless},
};
use managym::{
    flow::turn::StepKind,
    state::{
        card::Keyword,
        game_object::{ObjectId, PlayerId},
        zone::ZoneType,
    },
};

#[test]
fn kellan_upgrades_replace_types_keep_trigger_and_pay_real_costs() {
    let mut s = five_color_setup(&["Kellan, Planar Trailblazer"]);
    enter_and_settle(&mut s, "Kellan, Planar Trailblazer");
    let k = permanent_named(&s, "Kellan, Planar Trailblazer");
    s.set_player_mana_pool(0, "3RR");
    activate(&mut s, k, 0);
    spend_colorless(&mut s);
    settle(&mut s);
    let first = s.game().characteristics(k);
    assert!(first.has_subtype("Detective") && !first.has_subtype("Scout"));
    assert_eq!(first.triggered_abilities.len(), 1);
    activate(&mut s, k, 1);
    settle(&mut s);
    let view = s.game().characteristics(k);
    assert!(view.has_subtype("Rogue") && !view.has_subtype("Detective"));
    assert_eq!(view.pt(), (3, 2));
    assert!(view.keywords.has(Keyword::DoubleStrike));
    assert_eq!(view.triggered_abilities.len(), 1);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let before = s.game().state.zones.size(ZoneType::Exile, PlayerId(0));
    attack_with(&mut s, k);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.players[1].life, 14);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Exile, PlayerId(0)),
        before + 2
    );
    let object = card_of(&s, k);
    let object = move_card(&mut s, object, ZoneType::Hand);
    let object = move_card(&mut s, object, ZoneType::Battlefield);
    let reentered = permanent_of(&s, object).unwrap();
    assert!(s.game().characteristics(reentered).has_subtype("Scout"));
    assert!(s
        .game()
        .characteristics(reentered)
        .triggered_abilities
        .is_empty());
}

/// "Ward—Pay 7 life." (CR 702.21a, 119.4): the opponent whose spell targets
/// Sire of Seven Deaths pays 7 life in one action or the spell is countered.
/// Paying is offered only at 7 life or more, and is the same single action
/// under manual and automatic payment. Paying at exactly 7 is legal and
/// leaves that player at 0.
#[test]
fn sire_of_seven_deaths_ward_is_paid_in_life_or_counters_the_spell() {
    use crate::support::{
        game::{clear_hand, five_color_unfunded_setup, force_lands},
        inspect::permanent,
        play::cast_only,
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind, ActionType},
        flow::turn::StepKind,
        state::player::ManaPayment,
    };

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Line {
        Decline,
        Pay,
    }
    for auto in [false, true] {
        for (life, line) in [
            (20, Line::Decline),
            (20, Line::Pay),
            (7, Line::Pay),
            (6, Line::Decline),
        ] {
            let case = format!("auto={auto} life={life} {line:?}");
            let mut s = five_color_unfunded_setup(&["Sire of Seven Deaths", "Lightning Strike"]);
            let sire = s.force_permanent_on_battlefield(0, "Sire of Seven Deaths");
            force_lands(&mut s, 1, "Mountain", 2);
            clear_hand(&mut s, 0);
            clear_hand(&mut s, 1);
            s.force_card_in_hand(1, "Lightning Strike");
            if auto {
                s.game_mut().mana_payment[1] = ManaPayment::Auto;
            }
            s.game_mut().state.players[1].life = life;
            s.advance_to_active_step(1, StepKind::Main);
            cast_only(&mut s);
            s.choose_target_named("Sire of Seven Deaths");
            // The ward trigger on the stack states its price in words.
            let stack = managym::agent::observation::Observation::for_player(s.game(), PlayerId(1))
                .stack_objects;
            assert!(
                stack.iter().any(|object| object
                    .description
                    .as_deref()
                    .is_some_and(|text| text.contains("unless its controller pays 7 life"))),
                "{case}: {:?}",
                stack
                    .iter()
                    .map(|object| &object.description)
                    .collect::<Vec<_>>()
            );
            s.pass_priority();
            s.pass_priority();

            let space = s.action_space();
            assert_eq!(space.kind, ActionSpaceKind::PayOrNot, "{case}");
            assert_eq!(space.player, Some(PlayerId(1)), "{case}");
            // Pay or decline and nothing else: no mana ability, no tap.
            // Below 7 life paying is not offered (this fixture shows the
            // forced decline; a game skips it).
            if life >= 7 {
                assert!(
                    matches!(
                        space.actions.as_slice(),
                        [Action::PayCost { .. }, Action::Decline { .. }]
                    ),
                    "{case}: {:?}",
                    space.actions
                );
            } else {
                assert!(
                    matches!(space.actions.as_slice(), [Action::Decline { .. }]),
                    "{case}: {:?}",
                    space.actions
                );
            }
            // The payer's observation focuses the warded permanent, whose
            // row states the price.
            let observation =
                managym::agent::observation::Observation::for_player(s.game(), PlayerId(1));
            let [focus] = observation.action_space.focus.as_slice() else {
                panic!("one focus, {case}: {:?}", observation.action_space.focus)
            };
            let warded = observation
                .opponent_permanents
                .iter()
                .find(|permanent| permanent.object_id == Some(*focus))
                .unwrap();
            assert_eq!(warded.characteristics.ward_life, 7, "{case}");
            let context = s.game().decision_context(PlayerId(1)).unwrap();
            assert!(
                context.instruction.contains("life"),
                "{}",
                context.instruction
            );
            assert!(context.mana_cost.0.is_none(), "a life cost owes no mana");
            assert!(s.take_action_by_type(match line {
                Line::Pay => ActionType::PayCost,
                Line::Decline => ActionType::DeclineChoice,
            }));
            match line {
                Line::Decline => {
                    s.assert_life(1, life);
                    s.pass_priority();
                    s.pass_priority();
                    assert_eq!(permanent(&s, sire).damage, 0, "countered, {case}");
                }
                Line::Pay if life == 7 => {
                    // One action paid it: no mana step followed.
                    s.assert_life(1, 0);
                    assert!(s.game().is_game_over(), "0 life loses, {case}");
                }
                Line::Pay => {
                    s.assert_life(1, life - 7);
                    assert!(s.game().state.suspended_decision.is_none(), "{case}");
                    s.pass_priority();
                    s.pass_priority();
                    assert_eq!(permanent(&s, sire).damage, 3, "resolved, {case}");
                    s.assert_life(1, life - 7);
                }
            }
        }
    }
}

/// A mana ability's own cost, paid inside Hungry Ghoul's payment, suspends a
/// decision of its own; the Ghoul's chosen sacrifice is still what has paid.
#[test]
fn hungry_ghoul_sacrifice_stays_named_while_a_mana_ability_is_being_paid() {
    use crate::support::{
        game::five_color_unfunded_setup,
        play::{choose, select_card},
    };
    use managym::agent::action::Action;

    let mut s = five_color_unfunded_setup(&["Hungry Ghoul", "Gray Ogre", "Prophetic Prism"]);
    let ghoul = s.force_permanent_on_battlefield(0, "Hungry Ghoul");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let prism = s.force_permanent_on_battlefield(0, "Prophetic Prism");
    s.force_permanent_on_battlefield(0, "Swamp");
    s.force_permanent_on_battlefield(0, "Forest");
    s.game_mut().scenario_refresh_priority().unwrap();
    let ogre_object = card_of(&s, ogre);

    activate(&mut s, ghoul, 0);
    select_card(&mut s, ogre_object);
    // The Prism's own {1} is a decision inside the Ghoul's payment.
    choose(
        &mut s,
        |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism),
    );
    assert!(s.game().state.suspended_decision.is_some());
    let context = serde_json::to_value(s.game().decision_context(PlayerId(0)).unwrap()).unwrap();
    assert_eq!(context["paid"].as_array().unwrap().len(), 1, "{context}");
    assert_eq!(context["paid"][0]["object"]["label"], "Gray Ogre");
}

/// With nothing left to tap, the Ghoul's {1} is paid from the pool, and which
/// floated unit pays is a choice. The sacrifice is already made at that
/// choice and stays named.
#[test]
fn hungry_ghoul_sacrifice_stays_named_while_a_pooled_unit_is_chosen() {
    use crate::support::{
        game::five_color_unfunded_setup,
        play::{select_card, tap_for_mana},
    };

    let mut s =
        five_color_unfunded_setup(&["Hungry Ghoul", "Gray Ogre", "Carnelian Orb of Dragonkind"]);
    let ghoul = s.force_permanent_on_battlefield(0, "Hungry Ghoul");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let orb = s.force_permanent_on_battlefield(0, "Carnelian Orb of Dragonkind");
    let mountain = s.force_permanent_on_battlefield(0, "Mountain");
    let ogre_object = card_of(&s, ogre);
    // The Orb's red carries a rider, so it is a unit apart from the land's.
    s.allow_priority_floats();
    tap_for_mana(&mut s, orb);
    tap_for_mana(&mut s, mountain);

    activate(&mut s, ghoul, 0);
    select_card(&mut s, ogre_object);
    let frame = s.game().semantic_decision_frame().unwrap();
    let spends = frame
        .offers
        .iter()
        .filter(|offer| offer.label.starts_with("Spend"))
        .count();
    assert_eq!(spends, 2, "{:?}", frame.offers);
    let context = serde_json::to_value(s.game().decision_context(PlayerId(0)).unwrap()).unwrap();
    assert_eq!(context["paid"].as_array().unwrap().len(), 1, "{context}");
    assert_eq!(context["paid"][0]["object"]["label"], "Gray Ogre");
}

/// The payment preview reads the cost being paid. Hungry Ghoul's "{1},
/// Sacrifice another creature" has no preview until the creature is chosen;
/// at the mana it forecasts the land that would be tapped, and the pending
/// decision names the creature already sacrificed and where it still is.
#[test]
fn hungry_ghoul_pending_payment_names_the_creature_already_sacrificed() {
    use crate::support::{game::five_color_unfunded_setup, play::select_card};
    use managym::agent::action::Action;

    let preview = |s: &Scenario, action: &dyn Fn(&Action) -> bool| {
        let frame = s.game().semantic_decision_frame().unwrap();
        let index = s.action_space().actions.iter().position(action).unwrap();
        let preview = s
            .game()
            .payment_preview(PlayerId(0), frame.revision, &frame.fingerprint, index)
            .unwrap();
        serde_json::to_value(&preview).unwrap()
    };
    let mut s = five_color_unfunded_setup(&["Hungry Ghoul", "Gray Ogre"]);
    let ghoul = s.force_permanent_on_battlefield(0, "Hungry Ghoul");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Swamp");
    s.force_permanent_on_battlefield(0, "Forest");
    s.game_mut().scenario_refresh_priority().unwrap();
    let ogre_object = card_of(&s, ogre);

    let announced = preview(
        &s,
        &|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == ghoul),
    );
    assert_eq!(announced["state"], "needs_choice");
    assert!(s
        .game()
        .decision_context(PlayerId(0))
        .unwrap()
        .paid
        .is_empty());

    activate(&mut s, ghoul, 0);
    assert!(s
        .game()
        .decision_context(PlayerId(0))
        .unwrap()
        .paid
        .is_empty());
    select_card(&mut s, ogre_object);
    let context = serde_json::to_value(s.game().decision_context(PlayerId(0)).unwrap()).unwrap();
    assert_eq!(context["paid"].as_array().unwrap().len(), 1, "{context}");
    // Chosen, and still there: it leaves as the payment completes.
    assert_eq!(
        context["paid"][0]["phrase"],
        "Gray Ogre, sacrificed when the payment completes"
    );
    assert!(
        context["instruction"]
            .as_str()
            .unwrap()
            .contains("A chosen sacrifice leaves when the payment completes"),
        "{context}"
    );
    // The sacrifice is reserved until the mana is paid: the Ogre is still
    // the battlefield permanent the board draws.
    assert_eq!(context["paid"][0]["object"]["label"], "Gray Ogre");
    assert_eq!(
        context["paid"][0]["object"]["display_id"],
        s.game().state.permanents[ogre].as_ref().unwrap().id.0
    );
    let paying = preview(&s, &|a| matches!(a, Action::PayCost { .. }));
    assert_eq!(paying["state"], "exact", "{paying}");
    assert_eq!(paying["sources"][0]["label"], "Swamp");
    assert_eq!(paying["sources"].as_array().unwrap().len(), 1);

    // Once the payment completes the Ogre has left, and the ability's frame
    // says so in the past tense.
    crate::support::play::choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    if s.action_space().kind == managym::agent::action::ActionSpaceKind::PayOrNot {
        crate::support::play::pay_rest(&mut s);
    }
    let context = s.game().decision_context(PlayerId(0)).unwrap();
    assert!(context.paid.is_empty(), "{:?}", context.paid);
    assert!(
        context
            .pending
            .iter()
            .any(|line| line.contains("Paid with: sacrificed Gray Ogre.")),
        "{:?}",
        context.pending
    );
}

/// CR 601.2h, 602.2b: the ability's receipt names the creature sacrificed
/// for it, and what that creature was stays readable after it has left.
/// The sacrifice is the player's choice under both payment settings; only
/// the mana is paid for them under auto.
#[test]
fn hungry_ghoul_receipt_names_the_sacrificed_creature_after_it_has_left() {
    use crate::support::{
        game::five_color_unfunded_setup,
        inspect::{counters, offered_cards, zone_of},
        play::{choose, select_card},
    };
    use managym::{
        agent::{action::Action, observation::Observation},
        state::{
            payment::{Contribution, How, Toward},
            player::ManaPayment,
            stack_object::StackObject,
        },
    };

    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&["Hungry Ghoul", "Gray Ogre", "Hill Giant"]);
        let ghoul = s.force_permanent_on_battlefield(0, "Hungry Ghoul");
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
        let swamp = s.force_permanent_on_battlefield(0, "Swamp");
        let forest = s.force_permanent_on_battlefield(0, "Forest");
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let (ogre_object, giant_object) = (card_of(&s, ogre), card_of(&s, giant));
        activate(&mut s, ghoul, 0);

        // "Another creature": each other creature is offered, never the Ghoul.
        let mut offered = offered_cards(&s);
        offered.sort();
        let mut expected = vec![ogre_object, giant_object];
        expected.sort();
        assert_eq!(offered, expected, "auto={auto}");

        // Either choice completes once in its own fork.
        for (index, chosen) in [ogre_object, giant_object].into_iter().enumerate() {
            let mut fork = s.game().clone();
            fork.mana_payment[0] = ManaPayment::Auto;
            let action = fork
                .action_space()
                .unwrap()
                .actions
                .iter()
                .position(|a| matches!(a, Action::SelectCard { object, .. } if *object == chosen))
                .unwrap();
            fork.step(action).unwrap();
            let [StackObject::ActivatedAbility(ability)] = fork.state.stack_objects.as_slice()
            else {
                panic!("fork {index}: one activation")
            };
            assert_eq!(ability.paid.sacrificed().collect::<Vec<_>>(), [chosen]);
        }

        select_card(&mut s, ogre_object);
        if !auto {
            // The chosen creature is committed, not yet gone, while the mana
            // is paid: the acting player sees it as the reserved sacrifice.
            assert_eq!(zone_of(&s, ogre_object), Some(ZoneType::Battlefield));
            let frame = Observation::for_player(s.game(), PlayerId(0))
                .continuation
                .into_iter()
                .find(|frame| frame.kind == 4)
                .expect("the announcement being paid");
            assert!(frame.sacrifice.is_some());
            assert!(!frame.effect_execution.is_empty());
            for land in [swamp, forest] {
                assert!(crate::support::inspect::has_action(&s, |a| {
                    matches!(a, Action::TapForMana { permanent, .. } if *permanent == land)
                }));
            }
            choose(
                &mut s,
                |a| matches!(a, Action::TapForMana { permanent, .. } if *permanent == swamp),
            );
        }

        let ability = crate::support::inspect::top_activation(&s);
        assert_eq!(
            ability.paid.contributions(),
            [Contribution {
                object: ogre_object,
                how: How::Sacrificed,
                toward: Toward::Printed,
                after_mana: 0,
            }],
            "auto={auto}"
        );
        let [unit] = ability.paid.mana() else {
            panic!("one mana was spent")
        };
        let land = unit.source.expect("a land produced it").object;
        if !auto {
            assert_eq!(land, card_of(&s, swamp));
        }
        assert!(tapped_lands(&s, &[swamp, forest]) == 1);

        // The Ogre is gone; the receipt still says what it was.
        assert!(s.game().state.live(ogre_object).is_none());
        assert_eq!(
            zone_of(&s, crate::support::inspect::now(&s, ogre_object)),
            Some(ZoneType::Graveyard)
        );
        let lki = s.game().object_lki(ogre_object).expect("last known Ogre");
        assert_eq!((lki.power, lki.toughness), (2, 2));
        assert_eq!(
            s.game()
                .object_lki_definition(ogre_object)
                .unwrap()
                .printed
                .name,
            "Gray Ogre"
        );
        assert_eq!(zone_of(&s, giant_object), Some(ZoneType::Battlefield));

        // Both players read the same public fact in text.
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert!(
            context
                .pending
                .contains(&"Paid with: sacrificed Gray Ogre.".to_string()),
            "{:?}",
            context.pending
        );

        settle(&mut s);
        assert_eq!(counters(&s, ghoul), 1);
    }
}

fn tapped_lands(s: &Scenario, lands: &[managym::state::game_object::PermanentId]) -> usize {
    lands
        .iter()
        .filter(|land| crate::support::inspect::tapped(s, **land))
        .count()
}

/// CR 601.2f: the announced target settles the total cost, and the receipt
/// shows what was actually spent. Onto a tapped creature Luminous Rebuke
/// costs {1}{W}: two units from two Plains, each named, under both payment
/// settings. With two mana an untapped creature is not a legal choice.
#[test]
fn luminous_rebuke_cost_is_settled_by_its_target_and_the_receipt_shows_what_was_spent() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_to_hand},
        inspect::{has_action, tapped, zone_of},
        play::{choose, pass_until},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::Target, mana::Color, player::ManaPayment, stack_object::StackObject},
    };

    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&["Luminous Rebuke", "Gray Ogre", "Hill Giant"]);
        let attacker = s.force_permanent_on_battlefield(1, "Gray Ogre");
        let untapped = s.force_permanent_on_battlefield(1, "Hill Giant");
        s.game_mut().state.permanents[attacker]
            .as_mut()
            .unwrap()
            .tapped = true;
        let plains = [
            s.force_permanent_on_battlefield(0, "Plains"),
            s.force_permanent_on_battlefield(0, "Plains"),
        ];
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let rebuke = move_to_hand(&mut s, "Luminous Rebuke");
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { object, .. } if *object == rebuke),
        );
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(!has_action(&s, |a| {
            matches!(a, Action::ChooseTarget { target, .. } if *target == Target::Permanent(untapped))
        }));
        assert!(s.choose_target(Target::Permanent(attacker)));

        let Some(StackObject::Spell(spell)) = s.game().state.stack_objects.last() else {
            panic!("Luminous Rebuke is on the stack, auto={auto}")
        };
        assert_eq!(spell.spent.total(), 2, "{{4}}{{W}} less {{3}}");
        assert_eq!(spell.spent.amounts()[Color::White as usize], 2);
        let mut sources: Vec<_> = spell
            .spent
            .mana()
            .iter()
            .map(|unit| unit.source.expect("a Plains").object)
            .collect();
        sources.sort();
        let mut expected = plains.map(|land| card_of(&s, land)).to_vec();
        expected.sort();
        assert_eq!(sources, expected);
        assert!(spell.spent.contributions().is_empty());
        assert!(plains.iter().all(|land| tapped(&s, *land)));

        settle(&mut s);
        assert!(s.game().state.permanents[attacker].is_none());
        assert_eq!(
            zone_of(&s, card_of(&s, untapped)),
            Some(ZoneType::Battlefield)
        );
    }
}

const CREATURES: [&str; 4] = [
    "Gray Ogre",
    "Hill Giant",
    "Savannah Lions",
    "Llanowar Elves",
];

fn creatures(s: &mut Scenario, player: usize, count: usize) {
    for index in 0..count {
        s.force_permanent_on_battlefield(player, CREATURES[index / 4]);
    }
}

fn creature_count(s: &Scenario, player: usize) -> usize {
    CREATURES
        .iter()
        .map(|name| s.battlefield_permanents_named(player, name).len())
        .sum()
}

/// Blasphemous Edict costs {B} only while thirteen creatures are on the
/// battlefield, whoever controls them, and its mana value stays five. A
/// player with thirteen or fewer creatures sacrifices them all.
#[test]
fn blasphemous_edict_costs_one_black_at_thirteen_creatures() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::{can_cast, top_spell},
        play::{cast_card, pay_rest, settle},
    };
    use managym::agent::action::ActionSpaceKind;
    let mut names = vec!["Blasphemous Edict"];
    names.extend(CREATURES);
    let mut s = five_color_unfunded_setup(&names);
    creatures(&mut s, 0, 7);
    creatures(&mut s, 1, 5);
    s.force_permanent_on_battlefield(0, "Swamp");
    let edict = force_new_card_in_hand(&mut s, 0, "Blasphemous Edict");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, edict), "twelve creatures");
    s.force_permanent_on_battlefield(1, CREATURES[1]);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, edict), "thirteen creatures");

    // Only the alternative is payable, so it is not asked.
    cast_card(&mut s, edict);
    if s.action_space().kind == ActionSpaceKind::PayOrNot {
        pay_rest(&mut s);
    }
    let spell = top_spell(&s);
    assert_eq!(spell.spent.total(), 1);
    let printed = s.game().state[spell.object].mana_cost.as_ref().unwrap();
    assert_eq!(printed.mana_value, 5);
    settle(&mut s);
    assert_eq!((creature_count(&s, 0), creature_count(&s, 1)), (0, 0));
}

/// With the full cost payable too, the caster chooses; a player with
/// more than thirteen creatures chooses which to keep and cannot keep
/// more or fewer than the rest.
#[test]
fn blasphemous_edict_asks_for_its_cost_and_for_the_creatures_kept() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::{has_action, offered_cards, top_spell},
        play::{cast_card, choose, pay_rest, settle},
    };
    use managym::agent::action::{Action, ActionSpaceKind};
    let mut names = vec!["Blasphemous Edict"];
    names.extend(CREATURES);
    let mut s = five_color_unfunded_setup(&names);
    creatures(&mut s, 0, 15);
    creatures(&mut s, 1, 2);
    for _ in 0..5 {
        s.force_permanent_on_battlefield(0, "Swamp");
    }
    let edict = force_new_card_in_hand(&mut s, 0, "Blasphemous Edict");
    cast_card(&mut s, edict);
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    let frame = s.game().semantic_decision_frame().unwrap();
    assert!(
        frame
            .offers
            .iter()
            .any(|offer| offer.label == "Pay {B} instead of the mana cost"),
        "{:?}",
        frame.offers
    );
    // The choice is the card's alternative cost, not a kicker.
    assert_eq!(
        s.game().decision_context(PlayerId(0)).unwrap().instruction,
        "Choose whether to pay {B} rather than the mana cost."
    );
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    if s.action_space().kind == ActionSpaceKind::PayOrNot {
        pay_rest(&mut s);
    }
    assert_eq!(top_spell(&s).spent.total(), 1);

    s.advance_until(
        |sc| sc.action_space().kind == ActionSpaceKind::LookAndSelect,
        "the caster chooses creatures to keep".into(),
    );
    for remaining in [15, 14] {
        assert_eq!(offered_cards(&s).len(), remaining);
        assert!(!has_action(&s, |a| matches!(a, Action::Decline { .. })));
        choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    }
    settle(&mut s);
    assert_eq!((creature_count(&s, 0), creature_count(&s, 1)), (2, 0));
}

/// Claws Out costs {1} less for each Cat its caster controls and gives that
/// player's creatures +2/+2 until end of turn.
#[test]
fn claws_out_costs_less_per_cat_and_pumps_its_casters_creatures() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::top_spell,
        play::{cast_card, castable, settle},
    };
    use managym::state::player::ManaPayment;
    let mut s = five_color_unfunded_setup(&["Claws Out", "Felidar Cub", "Gray Ogre"]);
    force_lands(&mut s, 0, "Plains", 3);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let claws = force_new_card_in_hand(&mut s, 0, "Claws Out");
    let first = s.force_permanent_on_battlefield(0, "Felidar Cub");
    assert!(!castable(&mut s, claws), "one Cat: {{2}}{{W}}{{W}}");
    let second = s.force_permanent_on_battlefield(0, "Felidar Cub");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert!(castable(&mut s, claws), "two Cats: {{1}}{{W}}{{W}}");
    cast_card(&mut s, claws);
    assert_eq!(top_spell(&s).spent.total(), 3);
    settle(&mut s);
    for cat in [first, second] {
        assert_eq!(s.game().characteristics(cat).pt(), (4, 4));
    }
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 4));
    assert_eq!(s.game().characteristics(theirs).pt(), (2, 2));
}

fn replicate(s: &mut Scenario, target: managym::state::game_object::PermanentId, kicked: bool) {
    use crate::support::play::{cast, choose};
    use managym::{agent::action::Action, state::game_object::Target};
    cast(s, "Rite of Replication");
    choose(s, |action| {
        if kicked {
            matches!(action, Action::PayCost { .. })
        } else {
            matches!(action, Action::Decline { .. })
        }
    });
    assert!(s.choose_target(Target::Permanent(target)));
    settle(s);
}

#[test]
fn copy_sacrifice_is_intrinsic_to_each_electroduplicate_driver() {
    use crate::support::{
        game::move_from_library,
        inspect::{offered_targets, zone_of},
        play::{cast_card, command, resolve_top},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::Target, player::ManaPayment},
    };
    use std::collections::BTreeSet;

    let driver = "Driver of the Dead";
    let mut s = five_color_setup(&[
        driver,
        "Electroduplicate",
        "Rite of Replication",
        "Doubling Season",
        "Llanowar Elves",
        "Gray Ogre",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.force_permanent_on_battlefield(0, "Doubling Season");
    let original = s.force_permanent_on_battlefield(0, driver);
    let eligible: BTreeSet<_> = (0..4)
        .map(|_| move_from_library(&mut s, 0, "Llanowar Elves", ZoneType::Graveyard))
        .collect();
    for (owner, name) in [(0, "Gray Ogre"), (1, "Llanowar Elves"), (0, "Forest")] {
        move_from_library(&mut s, owner, name, ZoneType::Graveyard);
    }
    let spell = move_from_library(&mut s, 0, "Electroduplicate", ZoneType::Graveyard);
    cast_card(&mut s, spell);
    assert!(s.choose_target(Target::Permanent(original)));
    settle(&mut s);
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Exile));
    let drivers = s.battlefield_permanents_named(0, driver);
    assert_eq!(drivers.len(), 3);
    replicate(
        &mut s,
        *drivers.iter().find(|id| **id != original).unwrap(),
        false,
    );
    let mut tokens = BTreeSet::new();
    for id in s.battlefield_permanents_named(0, driver) {
        assert_eq!(
            s.game().characteristics(id).keywords.has(Keyword::Haste),
            id != original
        );
        if id != original {
            tokens.insert(card_of(&s, id));
        }
    }
    assert_eq!(tokens.len(), 4);
    s.advance_to_step(StepKind::End);
    assert_eq!(s.action_space().kind, ActionSpaceKind::OrderTriggers);
    let snapshot = s.snapshot();
    let mut first_order = Vec::new();
    for reverse in [false, true] {
        s.restore(&snapshot);
        let mut tokens = tokens.clone();
        let mut eligible = eligible.clone();
        while s.action_space().kind == ActionSpaceKind::OrderTriggers {
            let actions = &s.action_space().actions;
            let next = if reverse {
                actions.last()
            } else {
                actions.first()
            }
            .unwrap()
            .clone();
            command(&mut s, |action| action == &next);
        }
        assert_eq!(s.game().state.stack_objects.len(), 4);
        let sources: Vec<_> = s
            .game()
            .state
            .stack_objects
            .iter()
            .map(|trigger| trigger.source_object())
            .collect();
        if reverse {
            assert!(sources.iter().rev().eq(first_order.iter()));
        } else {
            first_order = sources;
        }
        while let Some(trigger) = s.game().state.stack_objects.last() {
            assert_eq!(trigger.controller(), PlayerId(0));
            let source = trigger.source_object();
            assert!(tokens.remove(&source));
            resolve_top(&mut s);
            assert!(permanent_of(&s, source).is_none());
            assert_eq!(
                offered_targets(&s).into_iter().collect::<BTreeSet<_>>(),
                eligible
                    .iter()
                    .copied()
                    .map(Target::GraveyardCard)
                    .collect()
            );
            let chosen = eligible.pop_first().unwrap();
            command(
                &mut s,
                |action| matches!(action, Action::ChooseTarget { target: Target::GraveyardCard(card), .. } if *card == chosen),
            );
            assert_eq!(
                s.game().state.stack_objects.last().unwrap().source_object(),
                source
            );
            resolve_top(&mut s);
            assert_eq!(zone_of(&s, chosen), Some(ZoneType::Battlefield));
        }
        assert!(tokens.is_empty() && eligible.is_empty());
        assert_eq!(s.battlefield_permanents_named(0, driver), [original]);
    }
}

#[test]
fn copy_sacrifice_scheduled_by_the_fire_crystal_remembers_only_its_token() {
    use crate::support::{
        inspect::top_spell,
        play::{cast, resolve_top},
    };
    use managym::state::{game_object::Target, player::ManaPayment};

    let mut s = five_color_setup(&["The Fire Crystal", "Gray Ogre", "Rite of Replication"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let crystal = s.force_permanent_on_battlefield(0, "The Fire Crystal");
    cast(&mut s, "Gray Ogre");
    assert_eq!(top_spell(&s).spent.total(), 2);
    settle(&mut s);
    let ogre = permanent_named(&s, "Gray Ogre");
    activate(&mut s, crystal, 0);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    let ogres = s.battlefield_permanents_named(0, "Gray Ogre");
    assert_eq!(ogres.len(), 2);
    let token = *ogres.iter().find(|id| **id != ogre).unwrap();
    replicate(&mut s, token, true);
    let ogres = s.battlefield_permanents_named(0, "Gray Ogre");
    assert_eq!(ogres.len(), 7);
    assert!(ogres
        .iter()
        .all(|id| s.game().characteristics(*id).keywords.has(Keyword::Haste)));
    let crystal_object = card_of(&s, crystal);
    move_card(&mut s, crystal_object, ZoneType::Graveyard);
    assert!(ogres
        .iter()
        .all(|id| !s.game().characteristics(*id).keywords.has(Keyword::Haste)));
    s.advance_to_step(StepKind::End);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    resolve_top(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(0, "Gray Ogre"),
        ogres
            .into_iter()
            .filter(|id| *id != token)
            .collect::<Vec<_>>()
    );
}

#[test]
fn leyline_axe_moves_its_grants_when_equipped_again() {
    use crate::support::game::enters_battlefield_with_triggers;
    use managym::state::target::Target;
    let mut s = five_color_setup(&["Leyline Axe", "Gray Ogre"]);
    let axe = enters_battlefield_with_triggers(&mut s, 0, "Leyline Axe");
    let ogres = [0, 1].map(|_| enters_battlefield_with_triggers(&mut s, 0, "Gray Ogre"));
    assert!(s.game().state.permanents[axe]
        .as_ref()
        .unwrap()
        .attached_to
        .is_none());
    for ogre in ogres {
        s.set_player_mana_pool(0, "3");
        activate(&mut s, axe, 0);
        assert!(s.choose_target(Target::Permanent(ogre)));
        settle(&mut s);
        for id in ogres {
            let view = s.game().characteristics(id);
            assert_eq!(view.pt(), if ogre == id { (3, 3) } else { (2, 2) });
            for keyword in [Keyword::DoubleStrike, Keyword::Trample] {
                assert_eq!(view.keywords.has(keyword), ogre == id);
            }
        }
    }
}

#[test]
fn herald_protects_against_a_failed_draw_in_its_last_sba_batch() {
    let mut s = five_color_setup(&["Herald of Eternal Dawn"]);
    let herald = s.force_permanent_on_battlefield(0, "Herald of Eternal Dawn");
    s.game_mut().state.permanents[herald]
        .as_mut()
        .unwrap()
        .damage = 6;
    s.game_mut().state.players[0].drew_when_empty = true;
    s.pass_priority();
    assert!(s.game().state.permanents[herald].is_none());
    assert!(!s.game().is_game_over());
    assert!(!s.game().state.players[0].drew_when_empty);
    s.game_mut().state.players[0].drew_when_empty = true;
    s.pass_priority();
    assert_eq!(s.game().state.loss(0).unwrap().reason(), Some("deck_empty"));
}

/// Replay both legal orders from the same pending end-step decision.
fn check_both_trigger_orders(s: &mut Scenario, check: impl Fn(&mut Scenario, [ObjectId; 2])) {
    use crate::support::play::command;
    use managym::{agent::action::ActionSpaceKind, search_state::witness};

    s.advance_to_step(StepKind::End);
    assert_eq!(s.action_space().kind, ActionSpaceKind::OrderTriggers);
    let orders: [_; 2] = s.action_space().actions.clone().try_into().unwrap();
    let root = s.snapshot();
    let [first, second] = orders.map(|order| {
        s.restore(&root);
        let committed = command(s, |action| action == &order);
        let recorded = serde_json::to_vec(&committed).unwrap();
        let expected = witness(s.game());
        s.restore(&root);
        s.game_mut()
            .execute_semantic_command(&serde_json::from_slice(&recorded).unwrap())
            .unwrap();
        assert_eq!(witness(s.game()), expected);
        let [bottom, top] = s.game().state.stack_objects.as_slice() else {
            panic!("expected both ordered triggers on the stack");
        };
        let sources = [bottom.source_object(), top.source_object()];
        check(s, sources);
        sources
    });
    assert_eq!(first, [second[1], second[0]]);
}

#[test]
fn copy_sacrifice_remembers_clones_ability_after_it_returns_to_hand() {
    use crate::support::{
        inspect::zone_of,
        play::{cast, resolve_top},
    };
    use managym::{
        agent::observation::Observation,
        state::{game_object::Target, player::ManaPayment},
    };

    let mut s = five_color_setup(&["Clone", "Gray Ogre", "Electroduplicate"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Electroduplicate");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    let token = *s
        .battlefield_permanents_named(0, "Gray Ogre")
        .iter()
        .find(|id| **id != ogre)
        .unwrap();
    cast(&mut s, "Clone");
    resolve_top(&mut s);
    assert!(s.choose_target(Target::Permanent(token)));
    settle(&mut s);
    let clone = *s
        .battlefield_permanents_named(0, "Gray Ogre")
        .iter()
        .find(|id| **id != ogre && **id != token)
        .unwrap();
    let clone = card_of(&s, clone);
    let token = card_of(&s, token);
    check_both_trigger_orders(&mut s, |s, sources| {
        let hand = move_card(s, clone, ZoneType::Hand);
        assert!(sources.contains(&token) && sources.contains(&clone));
        for source in sources.iter().rev() {
            for viewer in [PlayerId(0), PlayerId(1)] {
                let view = Observation::for_player(s.game(), viewer);
                assert_eq!(view.stack_objects.len(), s.game().state.stack_objects.len());
                let programs: Vec<_> = view
                    .continuation
                    .iter()
                    .filter(|frame| frame.kind == 3)
                    .collect();
                assert_eq!(programs.len(), view.stack_objects.len());
                assert!(programs.iter().all(|frame| !frame.program.is_empty()
                    && frame
                        .program_description
                        .as_ref()
                        .unwrap()
                        .to_lowercase()
                        .contains("sacrifice")));
                for entry in view.stack_objects {
                    let text = entry.description.unwrap();
                    assert!(
                        text.contains("Gray Ogre") && text.to_lowercase().contains("sacrifice"),
                        "{text}"
                    );
                }
            }
            assert_eq!(
                s.game().state.stack_objects.last().unwrap().source_object(),
                *source
            );
            resolve_top(s);
            // Even the departed source's ability resolves once (CR 113.7a).
            assert_eq!(
                s.game()
                    .state
                    .turn
                    .ability_resolutions_this_turn
                    .get(&(source.slot(), 0)),
                Some(&1)
            );
            assert_eq!(zone_of(s, hand), Some(ZoneType::Hand));
        }
        assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre"), [ogre]);
    });
}

/// A temporary death trigger distinguishes the exact recipient from its copies.
fn grant_coil_death_trigger(s: &mut Scenario, coil: managym::state::game_object::PermanentId) {
    use managym::state::{
        ability::{Ability, Effect, TriggerCondition, TriggerSubject},
        effect::{Affected, EffectDuration, Modification},
    };
    let object = card_of(s, coil);
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        [Affected::Object(object)],
        Modification::GrantTriggered(Box::new(Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            intervening_if: None,
            effects: vec![Effect::gain_life(3)],
        })),
        EffectDuration::EndOfTurn,
    );
}

#[test]
fn named_coil_copies_inherit_electroduplicate_but_only_the_recipient_gains_life() {
    use crate::support::{
        game::move_from_library,
        inspect::{offered_targets, top_spell, zone_of},
        play::{cast, cast_card},
    };
    use managym::state::{game_object::Target, player::ManaPayment};
    use std::collections::BTreeSet;
    let mut s = five_color_setup(&[
        "Electroduplicate",
        "Rite of Replication",
        "Self-Reflection",
        "Relm's Sketching",
        "Gray Ogre",
        "Doubling Season",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let original = s
        .game_mut()
        .create_token("token.komas_coil", PlayerId(0), false)
        .unwrap();
    let opponent = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(1, "Island");
    let clue = s
        .game_mut()
        .create_token("token.clue", PlayerId(1), false)
        .unwrap();
    s.force_permanent_on_battlefield(1, "Doubling Season");
    cast(&mut s, "Electroduplicate");
    assert!(s.choose_target(Target::Permanent(original)));
    settle(&mut s);
    let recipient = *s
        .battlefield_permanents_named(0, "Koma's Coil")
        .iter()
        .find(|id| **id != original)
        .unwrap();
    grant_coil_death_trigger(&mut s, recipient);
    let main = s.snapshot();
    for (spell, flashback) in [
        ("Rite of Replication", false),
        ("Relm's Sketching", false),
        ("Self-Reflection", false),
        ("Self-Reflection", true),
    ] {
        s.restore(&main);
        if spell == "Rite of Replication" {
            replicate(&mut s, recipient, false);
        } else {
            let casting_card = if flashback {
                let card = move_from_library(&mut s, 0, spell, ZoneType::Graveyard);
                cast_card(&mut s, card);
                card
            } else {
                cast(&mut s, spell)
            };
            let mut legal = vec![original, recipient];
            if spell == "Relm's Sketching" {
                legal.extend([opponent, land, clue]);
            }
            assert_eq!(
                offered_targets(&s).into_iter().collect::<BTreeSet<_>>(),
                legal.into_iter().map(Target::Permanent).collect()
            );
            assert!(s.choose_target(Target::Permanent(recipient)));
            assert_eq!(
                top_spell(&s).spent.total(),
                if spell == "Self-Reflection" && !flashback {
                    6
                } else {
                    4
                }
            );
            settle(&mut s);
            assert_eq!(
                zone_of(&s, casting_card),
                Some(if flashback {
                    ZoneType::Exile
                } else {
                    ZoneType::Graveyard
                })
            );
        }
        check_coil_sacrifices(&mut s, original, recipient);
    }
}

fn check_coil_sacrifices(
    s: &mut Scenario,
    original: managym::state::game_object::PermanentId,
    recipient: managym::state::game_object::PermanentId,
) {
    use crate::support::play::resolve_top;
    use managym::{
        agent::observation::Observation,
        state::mana::{Color, Colors},
    };
    use std::collections::BTreeSet;
    let coils = s.battlefield_permanents_named(0, "Koma's Coil");
    assert_eq!(coils.len(), 3);
    for &id in &coils {
        let object = &s.game().state[card_of(s, id)];
        assert!(object.has_subtype("Serpent"));
        assert_eq!(object.colors(), Colors::from_iter([Color::Blue]));
        assert_eq!(s.game().characteristics(id).pt(), (3, 3));
    }
    let descendants: BTreeSet<_> = coils
        .into_iter()
        .filter(|id| *id != original)
        .map(|id| {
            assert!(s.game().characteristics(id).keywords.has(Keyword::Haste));
            card_of(s, id)
        })
        .collect();
    let grantee = card_of(s, recipient);
    check_both_trigger_orders(s, |s, sources| {
        assert_eq!(
            sources.iter().copied().collect::<BTreeSet<_>>(),
            descendants
        );
        for viewer in [PlayerId(0), PlayerId(1)] {
            let view = Observation::for_player(s.game(), viewer);
            let recipients: BTreeSet<_> = view
                .agent_permanents
                .iter()
                .chain(&view.opponent_permanents)
                .filter(|row| {
                    row.is_token
                        && row.characteristics.name == "Koma's Coil"
                        && row.characteristics.keywords.haste
                })
                .map(|row| row.id)
                .collect();
            assert_eq!(recipients.len(), 2);
            assert_eq!(
                view.continuation
                    .iter()
                    .filter(|frame| frame.kind == 3)
                    .map(|frame| frame.source.object.unwrap())
                    .collect::<BTreeSet<_>>(),
                recipients
            );
        }
        let before = s.game().state.players[0].life;
        for source in sources.into_iter().rev() {
            resolve_top(s);
            if source == grantee {
                assert_eq!(
                    s.game().state.stack_objects.last().unwrap().source_object(),
                    grantee
                );
                resolve_top(s);
            }
        }
        assert_eq!(s.game().state.players[0].life, before + 3);
        assert!(s.game().state.stack_objects.is_empty());
        assert_eq!(s.battlefield_permanents_named(0, "Koma's Coil"), [original]);
    });
}

#[test]
fn named_coil_upkeep_copies_only_another_controlled_nonland() {
    use crate::support::inspect::offered_targets;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Extravagant Replication", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Extravagant Replication");
    let coil = s
        .game_mut()
        .create_token("token.komas_coil", PlayerId(0), false)
        .unwrap();
    let clue = s
        .game_mut()
        .create_token("token.clue", PlayerId(0), false)
        .unwrap();
    s.force_permanent_on_battlefield(0, "Island");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(s.game().state.stack_objects.is_empty() && offered_targets(&s).is_empty());
    s.advance_to_active_step(0, StepKind::Upkeep);
    let targets = offered_targets(&s);
    assert_eq!(targets.len(), 2);
    for candidate in [coil, clue] {
        assert!(targets.contains(&Target::Permanent(candidate)));
    }
    assert!(s.choose_target(Target::Permanent(coil)));
    grant_coil_death_trigger(&mut s, coil);
    settle(&mut s);
    let coils = s.battlefield_permanents_named(0, "Koma's Coil");
    assert_eq!(coils.len(), 2);
    for id in coils {
        assert_eq!(
            s.game().characteristics(id).triggered_abilities.len(),
            usize::from(id == coil)
        );
    }
}

/// Flush real zone-move triggers from a constructed position before passing.
fn stack_departures(s: &mut Scenario) {
    s.game_mut().scenario_refresh_priority().unwrap();
    if !s.game().state.pending_triggers.is_empty() {
        s.pass_priority();
    }
}

#[test]
fn departure_returns_use_last_creature_types_and_exact_graveyard_cards() {
    use crate::support::inspect::{counters_of, zone_of};
    use managym::state::permanent::CounterKind;
    for (creature, added_type, count) in
        [("Gray Ogre", "Angel", 1), ("Infernal Vessel", "Demon", 2)]
    {
        let mut s = five_color_setup(&[creature, "Valkyrie's Call"]);
        if added_type == "Angel" {
            s.force_permanent_on_battlefield(0, "Valkyrie's Call");
        }
        let id = s.force_permanent_on_battlefield(0, creature);
        let original = card_of(&s, id);
        let grave = move_card(&mut s, original, ZoneType::Graveyard);
        stack_departures(&mut s);
        let root = s.snapshot();
        let hand = move_card(&mut s, grave, ZoneType::Hand);
        let later_grave = move_card(&mut s, hand, ZoneType::Graveyard);
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(later_grave),
            Some(ZoneType::Graveyard)
        );
        s.restore(&root);
        settle(&mut s);
        let returned = permanent_of(&s, original).unwrap();
        let view = s.game().characteristics(returned);
        assert!(view.has_subtype(added_type));
        assert_eq!(view.keywords.has(Keyword::Flying), added_type == "Angel");
        assert_eq!(
            counters_of(&s, returned, CounterKind::PlusOnePlusOne),
            count
        );
        move_card(&mut s, original, ZoneType::Graveyard);
        stack_departures(&mut s);
        assert!(s.game().state.stack_objects.is_empty());
        assert_eq!(zone_of(&s, original), Some(ZoneType::Graveyard));
    }
}

#[test]
fn departure_manifested_land_and_mooned_creature_use_battlefield_types() {
    use crate::support::{
        inspect::{library_card, zone_of},
        play::cast,
    };
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Valkyrie's Call", "Gray Ogre", "Imprisoned in the Moon"]);
    s.force_permanent_on_battlefield(0, "Valkyrie's Call");
    let forest = library_card(&s, 0, "Forest");
    let manifest = s.game_mut().manifest(forest, PlayerId(0));
    move_card(&mut s, manifest, ZoneType::Graveyard);
    stack_departures(&mut s);
    settle(&mut s);
    let returned = permanent_of(&s, forest).unwrap();
    assert!(s.game().characteristics(returned).is_land());
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let object = card_of(&s, ogre);
    cast(&mut s, "Imprisoned in the Moon");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert!(!s.game().characteristics(ogre).is_creature());
    move_card(&mut s, object, ZoneType::Graveyard);
    stack_departures(&mut s);
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
}

#[test]
fn departure_angelic_destiny_and_enchanted_creature_die_together() {
    use crate::support::{game::until_end_of_turn, inspect::zone_of, play::cast};
    use managym::state::{effect::Modification, game_object::Target};
    let mut s = five_color_setup(&["Angelic Destiny", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let aura = cast(&mut s, "Angelic Destiny");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    let enchanted = s.game().characteristics(ogre);
    assert_eq!(enchanted.pt(), (6, 6));
    assert!(
        enchanted.has_subtype("Angel")
            && enchanted.keywords.has(Keyword::FirstStrike)
            && enchanted.keywords.has(Keyword::Flying)
    );
    let attached = s.snapshot();
    // The animated Aura has zero toughness in the same SBA batch that
    // destroys the lethally damaged creature it enchants.
    let id = permanent_of(&s, aura).unwrap();
    until_end_of_turn(&mut s, id, Modification::BecomeCreature);
    s.game_mut().state.permanents[ogre].as_mut().unwrap().damage = 6;
    s.pass_priority();
    assert!(s.game().state.permanents[ogre].is_none());
    assert_eq!(zone_of(&s, aura), Some(ZoneType::Graveyard));
    assert_eq!(s.game().state.stack_objects.len(), 1);
    let stacked = s.snapshot();
    settle(&mut s);
    assert_eq!(zone_of(&s, aura), Some(ZoneType::Hand));

    // A later graveyard appearance of the Aura is a new object (CR 400.7).
    s.restore(&stacked);
    let in_hand = move_card(&mut s, aura, ZoneType::Hand);
    move_card(&mut s, in_hand, ZoneType::Graveyard);
    settle(&mut s);
    assert_eq!(zone_of(&s, aura), Some(ZoneType::Graveyard));

    // The Aura leaving alone does not trigger its enchanted-creature death.
    s.restore(&attached);
    move_card(&mut s, aura, ZoneType::Graveyard);
    stack_departures(&mut s);
    assert!(s.game().state.stack_objects.is_empty());
    assert!(s.game().state.permanents[ogre].is_some());
}

#[test]
fn departure_linked_enchantments_bind_target_counts_and_both_appearances() {
    use crate::support::{game::enter_battlefield, inspect::zone_of, play::command};
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::game_object::Target,
    };
    for name in ["Prayer of Binding", "Banishing Light", "Stasis Snare"] {
        let prayer = name == "Prayer of Binding";
        let mut s = five_color_setup(&[name, "Gray Ogre"]);
        let empty_board = s.snapshot();
        enter_battlefield(&mut s, name);
        stack_departures(&mut s);
        assert_eq!(s.game().state.stack_objects.len(), usize::from(prayer));
        settle(&mut s);
        s.assert_life(0, if prayer { 22 } else { 20 });
        s.restore(&empty_board);
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        let target = card_of(&s, ogre);
        let enchantment = enter_battlefield(&mut s, name);
        let source = card_of(&s, enchantment);
        if s.action_space().kind != ActionSpaceKind::ChooseTarget {
            s.pass_priority();
        }
        let choosing = s.snapshot();
        assert_eq!(
            s.action_space()
                .actions
                .iter()
                .any(|a| matches!(a, Action::Decline { .. })),
            prayer
        );
        if prayer {
            command(&mut s, |a| matches!(a, Action::Decline { .. }));
            settle(&mut s);
            s.assert_life(0, 22);
            assert_eq!(zone_of(&s, target), Some(ZoneType::Battlefield));
            s.restore(&choosing);
        }
        command(
            &mut s,
            |a| matches!(a, Action::ChooseTarget {target: Target::Permanent(id), ..} if *id == ogre),
        );
        let stacked = s.snapshot();
        move_card(&mut s, source, ZoneType::Hand);
        settle(&mut s);
        assert_eq!(zone_of(&s, target), Some(ZoneType::Battlefield));
        s.restore(&stacked);
        move_card(&mut s, target, ZoneType::Hand);
        settle(&mut s);
        s.assert_life(0, 20);
        s.restore(&stacked);
        settle(&mut s);
        assert_eq!(zone_of(&s, target), Some(ZoneType::Exile));
        let exiled = s.snapshot();
        move_card(&mut s, source, ZoneType::Hand);
        assert_eq!(zone_of(&s, target), Some(ZoneType::Battlefield));
        s.restore(&exiled);
        move_card(&mut s, target, ZoneType::Hand);
        let later = move_card(&mut s, target, ZoneType::Exile);
        move_card(&mut s, source, ZoneType::Hand);
        assert_eq!(s.game().state.zones.zone_of(later), Some(ZoneType::Exile));
    }
}

#[test]
fn departure_each_hoarding_dragon_recovers_only_its_exiled_artifact() {
    use crate::support::{
        game::enter_battlefield,
        inspect::{library_card, zone_of},
        play::{command, pass_until},
    };
    use managym::agent::action::{Action, ActionSpaceKind};
    let mut s = five_color_setup(&["Hoarding Dragon", "Hedron Archive", "Leyline Axe"]);
    let dragons = ["Hedron Archive", "Leyline Axe"].map(|name| {
        let artifact = library_card(&s, 0, name);
        let dragon = enter_battlefield(&mut s, "Hoarding Dragon");
        pass_until(&mut s, ActionSpaceKind::Modal);
        command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
        command(
            &mut s,
            |a| matches!(a, Action::SelectCard {object,..} if *object == artifact),
        );
        settle(&mut s);
        assert_eq!(zone_of(&s, artifact), Some(ZoneType::Exile));
        (card_of(&s, dragon), artifact)
    });
    let root = s.snapshot();
    for index in 0..2 {
        s.restore(&root);
        let (source, artifact) = dragons[index];
        move_card(&mut s, source, ZoneType::Graveyard);
        stack_departures(&mut s);
        pass_until(&mut s, ActionSpaceKind::Modal);
        command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
        settle(&mut s);
        assert_eq!(zone_of(&s, artifact), Some(ZoneType::Hand));
        assert_eq!(zone_of(&s, dragons[1 - index].1), Some(ZoneType::Exile));
    }

    // A bounced and replayed Dragon is a new object: its death cannot reach
    // the card the earlier appearance exiled.
    s.restore(&root);
    let (source, artifact) = dragons[0];
    let in_hand = move_card(&mut s, source, ZoneType::Hand);
    settle(&mut s);
    move_card(&mut s, in_hand, ZoneType::Battlefield);
    stack_departures(&mut s);
    if s.action_space().kind == ActionSpaceKind::Modal {
        command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    }
    settle(&mut s);
    move_card(&mut s, source, ZoneType::Graveyard);
    stack_departures(&mut s);
    settle(&mut s);
    assert_eq!(zone_of(&s, artifact), Some(ZoneType::Exile));
}

#[test]
fn departure_valkyries_call_sees_a_creature_dying_beside_it() {
    use crate::support::{
        game::until_end_of_turn,
        inspect::{counters_of, zone_of},
        play::command,
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{effect::Modification, permanent::CounterKind},
    };
    let mut s = five_color_setup(&["Valkyrie's Call", "Gray Ogre"]);
    let call = s.force_permanent_on_battlefield(0, "Valkyrie's Call");
    let call_card = card_of(&s, call);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = card_of(&s, ogre);
    // The animated Call (0/0) and the lethally damaged Ogre leave in one
    // state-based batch. Each trigger looks back (CR 603.10a), and the Call
    // was itself a nontoken, non-Angel creature, so both cards return.
    until_end_of_turn(&mut s, call, Modification::BecomeCreature);
    s.game_mut().state.permanents[ogre].as_mut().unwrap().damage = 2;
    s.pass_priority();
    assert!(s.game().state.permanents[call].is_none());
    assert_eq!(s.action_space().kind, ActionSpaceKind::OrderTriggers);
    command(&mut s, |_| true);
    assert_eq!(s.game().state.stack_objects.len(), 2);
    settle(&mut s);
    let returned = permanent_of(&s, card).unwrap();
    let view = s.game().characteristics(returned);
    assert!(view.has_subtype("Angel") && view.keywords.has(Keyword::Flying));
    assert_eq!(counters_of(&s, returned, CounterKind::PlusOnePlusOne), 1);
    assert_eq!(zone_of(&s, call_card), Some(ZoneType::Battlefield));
}

#[test]
fn departure_familiar_freezes_revival_count_and_remembers_exact_card() {
    use crate::support::{
        game::set_counters,
        inspect::{counters_of, zone_of},
        play::{cast, command},
    };
    use managym::{
        agent::action::Action,
        state::{permanent::CounterKind, player::ManaPayment},
    };
    let mut s = five_color_setup(&["Nine-Lives Familiar"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let familiar = cast(&mut s, "Nine-Lives Familiar");
    settle(&mut s);
    let id = permanent_of(&s, familiar).unwrap();
    assert_eq!(counters_of(&s, id, CounterKind::Revival), 8);
    set_counters(&mut s, id, CounterKind::Revival, 4);
    let grave = move_card(&mut s, familiar, ZoneType::Graveyard);
    stack_departures(&mut s);
    settle(&mut s);
    assert_eq!(s.game().state.delayed_triggers.len(), 1);
    s.advance_to_step(StepKind::End);
    let root = s.snapshot();
    command(&mut s, |a| matches!(a, Action::PassPriority { .. }));
    settle(&mut s);
    let returned = permanent_of(&s, familiar).unwrap();
    assert_eq!(counters_of(&s, returned, CounterKind::Revival), 3);
    s.restore(&root);
    move_card(&mut s, grave, ZoneType::Hand);
    move_card(&mut s, grave, ZoneType::Graveyard);
    settle(&mut s);
    assert_eq!(zone_of(&s, grave), Some(ZoneType::Graveyard));
    let blink = move_card(&mut s, grave, ZoneType::Battlefield);
    assert_eq!(
        counters_of(&s, permanent_of(&s, blink).unwrap(), CounterKind::Revival),
        0
    );
    move_card(&mut s, blink, ZoneType::Graveyard);
    stack_departures(&mut s);
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn departure_kykar_modes_blink_another_creature_or_create_a_flying_spirit() {
    use crate::support::{
        inspect::{offered_targets, zone_of},
        play::{cast, command, pass_until},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::Target, mana::Color, player::ManaPayment},
    };
    let mut s = five_color_setup(&["Kykar, Zephyr Awakener", "Gray Ogre", "Hedron Archive"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.force_permanent_on_battlefield(0, "Kykar, Zephyr Awakener");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = card_of(&s, ogre);
    cast(&mut s, "Hedron Archive");
    pass_until(&mut s, ActionSpaceKind::Modal);
    let root = s.snapshot();
    command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    settle(&mut s);
    let spirit = s.game().characteristics(permanent_named(&s, "Spirit"));
    assert!(spirit.keywords.has(Keyword::Flying) && spirit.colors.contains(&Color::White));
    assert_eq!(spirit.pt(), (1, 1));
    s.restore(&root);
    command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    assert_eq!(offered_targets(&s), vec![Target::Permanent(ogre)]);
    command(
        &mut s,
        |a| matches!(a,Action::ChooseTarget {target:Target::Permanent(id),..} if *id==ogre),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    let exiled = s.snapshot();
    s.advance_to_step(StepKind::End);
    settle(&mut s);
    assert_ne!(permanent_of(&s, card).unwrap(), ogre);
    s.restore(&exiled);
    move_card(&mut s, card, ZoneType::Hand);
    let later = move_card(&mut s, card, ZoneType::Exile);
    s.advance_to_step(StepKind::End);
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(later), Some(ZoneType::Exile));
}

#[test]
fn departure_granted_returns_follow_owner_controller_and_one_incarnation() {
    use crate::support::{
        game::enter_battlefield,
        inspect::{counters_of, tapped, zone_of},
        play::{cast, command},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::Target, permanent::CounterKind},
    };
    let mut s = five_color_setup(&[
        "Fake Your Own Death",
        "Undying Malice",
        "Gray Ogre",
        "Agent of Treachery",
    ]);
    let root = s.snapshot();

    // A stolen creature returns tapped to its owner; the trigger's controller
    // gets the Treasure.
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    enter_battlefield(&mut s, "Agent of Treachery");
    if s.action_space().kind != ActionSpaceKind::ChooseTarget {
        s.pass_priority();
    }
    command(
        &mut s,
        |a| matches!(a, Action::ChooseTarget { target: Target::Permanent(id), .. } if *id == ogre),
    );
    settle(&mut s);
    let controller = |s: &Scenario, id: managym::state::game_object::PermanentId| {
        s.game().state.permanents[id].as_ref().unwrap().controller
    };
    assert_eq!(controller(&s, ogre), PlayerId(0));
    let card = card_of(&s, ogre);
    cast(&mut s, "Fake Your Own Death");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 2));
    move_card(&mut s, card, ZoneType::Graveyard);
    stack_departures(&mut s);
    settle(&mut s);
    let returned = permanent_of(&s, card).unwrap();
    assert_eq!(controller(&s, returned), PlayerId(1));
    assert!(tapped(&s, returned));
    assert_eq!(s.game().characteristics(returned).pt(), (2, 2));
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);
    // The grant belonged to the departed appearance.
    move_card(&mut s, card, ZoneType::Graveyard);
    stack_departures(&mut s);
    assert!(s.game().state.stack_objects.is_empty());
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));

    // A dying token cannot return, but still makes Treasure.
    s.restore(&root);
    let token = s
        .game_mut()
        .create_token("token.flying_spirit", PlayerId(0), false)
        .unwrap();
    let token_object = card_of(&s, token);
    cast(&mut s, "Fake Your Own Death");
    assert!(s.choose_target(Target::Permanent(token)));
    settle(&mut s);
    s.game_mut().move_object(token_object, ZoneType::Graveyard);
    stack_departures(&mut s);
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Spirit").is_empty());
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);

    // Malice returns the owner's card tapped with one +1/+1 counter.
    s.restore(&root);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = card_of(&s, ogre);
    cast(&mut s, "Undying Malice");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    move_card(&mut s, card, ZoneType::Graveyard);
    stack_departures(&mut s);
    settle(&mut s);
    let returned = permanent_of(&s, card).unwrap();
    assert!(tapped(&s, returned));
    assert_eq!(counters_of(&s, returned, CounterKind::PlusOnePlusOne), 1);
    move_card(&mut s, card, ZoneType::Graveyard);
    stack_departures(&mut s);
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn kicked_healer_returns_only_cheap_nonland_permanents() {
    super::check_kicked_return("Sun-Blessed Healer", "2WW", "Sol Ring", "Gray Ogre");
}
