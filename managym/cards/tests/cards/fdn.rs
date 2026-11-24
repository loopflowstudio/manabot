//! Foundations behavior through ordinary activation, payment and combat.
use crate::support::game::move_card;
use crate::support::inspect::permanent_of;
use crate::support::{
    game::five_color_setup,
    inspect::{card_of, permanent_named},
    play::{activate, attack_with, enter_and_settle, settle, spend_colorless},
};
use managym::{
    flow::turn::StepKind,
    state::{card::Keyword, game_object::PlayerId, zone::ZoneType},
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
    use crate::support::{
        game::{five_color_unfunded_setup, Scenario},
        play::select_card,
    };
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

fn tapped_lands(
    s: &crate::support::game::Scenario,
    lands: &[managym::state::game_object::PermanentId],
) -> usize {
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

fn creatures(s: &mut crate::support::game::Scenario, player: usize, count: usize) {
    for index in 0..count {
        s.force_permanent_on_battlefield(player, CREATURES[index / 4]);
    }
}

fn creature_count(s: &crate::support::game::Scenario, player: usize) -> usize {
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

#[test]
fn lightshell_duo_orders_or_mills_and_gains_prowess() {
    use crate::support::{
        game::pin_top,
        inspect::{library_top, offered_cards, zone_of},
        play::{cast, choose, fork_choosing, fork_selecting, pass_until, resolve_top, select_card},
    };
    use managym::agent::action::{Action, ActionSpaceKind};

    let mut s = five_color_setup(&["Lightshell Duo", "Scout's Warning"]);
    let cards = pin_top(&mut s, &["Island", "Forest"]);
    cast(&mut s, "Lightshell Duo");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let duo = permanent_named(&s, "Lightshell Duo");
    assert_eq!(s.game().characteristics(duo).pt(), (3, 4));
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    assert_eq!(offered_cards(&s), cards);

    let order = fork_choosing(&s, |a| matches!(a, Action::Decline { .. }));
    assert_eq!(offered_cards(&order), cards);
    let (island, forest) = (cards[0], cards[1]);
    for (first, second) in [(island, forest), (forest, island)] {
        let mut mill = fork_selecting(&s, first);
        choose(&mut mill, |a| matches!(a, Action::Decline { .. }));
        assert_eq!(zone_of(&mill, first), Some(ZoneType::Graveyard));
        assert_eq!(library_top(&mill, 0), second);

        let mut keep = fork_selecting(&order, first);
        select_card(&mut keep, second);
        cast(&mut keep, "Scout's Warning");
        resolve_top(&mut keep);
        assert_eq!(keep.game().characteristics(duo).pt(), (4, 5));
        assert_eq!(
            library_top(&keep, 0),
            first,
            "prowess resolves before the draw"
        );
        resolve_top(&mut keep);
        assert_eq!(zone_of(&keep, first), Some(ZoneType::Hand));
        assert_eq!(library_top(&keep, 0), second);
        keep.advance_to_active_step(1, StepKind::Main);
        assert_eq!(keep.game().characteristics(duo).pt(), (3, 4));
    }
}

#[test]
fn arahbo_and_helpful_hunter_resolve_in_either_order() {
    use crate::support::{
        inspect::library_card,
        play::{fork_stepping, pass_until},
    };
    use managym::agent::action::ActionSpaceKind;
    let mut s = five_color_setup(&["Arahbo, the First Fang", "Helpful Hunter"]);
    enter_and_settle(&mut s, "Arahbo, the First Fang");
    let original = s.battlefield_permanents_named(0, "Cat");
    assert_eq!(original.len(), 1);
    let hand = s.zone_size(0, ZoneType::Hand);
    let hunter = library_card(&s, 0, "Helpful Hunter");
    s.game_mut().move_object(hunter, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::OrderTriggers);
    assert_eq!(s.action_space().actions.len(), 2);
    let mut after_first = Vec::new();
    for first in [0, 1] {
        let mut s = fork_stepping(&s, first);
        s.advance_until(
            |s| {
                s.game().state.stack_objects.len() == 1
                    && s.action_space().kind == ActionSpaceKind::Priority
            },
            "first trigger resolved".into(),
        );
        after_first.push((
            s.zone_size(0, ZoneType::Hand) - hand,
            s.battlefield_permanents_named(0, "Cat").len() - 1,
        ));
        settle(&mut s);
        assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
        let cats = s.battlefield_permanents_named(0, "Cat");
        assert_eq!(cats.len(), 2);
        assert!(cats.contains(&original[0]));
        for cat in cats {
            assert_eq!(s.game().characteristics(cat).pt(), (2, 2));
        }
        assert!(s.game().state.pending_triggers.is_empty());
    }
    after_first.sort();
    assert_eq!(after_first, [(0, 1), (1, 0)]);
}

#[test]
fn token_copy_of_arahbo_triggers_on_its_own_entry_once() {
    use crate::support::{
        game::plant_in_graveyard,
        inspect::now,
        play::{cast, choose, choose_mode},
    };
    use managym::{agent::action::Action, state::game_object::Target};
    let mut s = five_color_setup(&["Foggy Swamp Visions", "Arahbo, the First Fang"]);
    let grave = plant_in_graveyard(&mut s, 0, "Arahbo, the First Fang");
    cast(&mut s, "Foggy Swamp Visions");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::GraveyardCard(now(&s, grave))));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    let copy = permanent_named(&s, "Arahbo, the First Fang");
    assert!(!s.game().state[card_of(&s, copy)].is_card());
    let cats = s.battlefield_permanents_named(0, "Cat");
    assert_eq!(cats.len(), 1);
    assert_eq!(s.game().characteristics(cats[0]).pt(), (2, 2));
}

#[test]
fn anthem_bonuses_parent_and_its_cat_while_in_play() {
    let mut s = five_color_setup(&["Anthem of Champions", "Prideful Parent"]);
    let opponent = s
        .game_mut()
        .create_token("token.cat_1_1_w", PlayerId(1), false)
        .unwrap();
    enter_and_settle(&mut s, "Prideful Parent");
    let parent = permanent_named(&s, "Prideful Parent");
    let cats = s.battlefield_permanents_named(0, "Cat");
    assert_eq!(cats.len(), 1);
    let assert_board = |s: &crate::support::game::Scenario, bonus: i32| {
        let view = s.game().characteristics(parent);
        assert_eq!(view.pt(), (2 + bonus, 2 + bonus));
        assert!(view.keywords.has(Keyword::Vigilance));
        assert_eq!(
            s.game().characteristics(cats[0]).pt(),
            (1 + bonus, 1 + bonus)
        );
        assert_eq!(s.game().characteristics(opponent).pt(), (1, 1));
    };
    assert_board(&s, 0);
    let anthem = enter_and_settle(&mut s, "Anthem of Champions");
    assert_board(&s, 1);
    move_card(&mut s, anthem, ZoneType::Graveyard);
    assert_board(&s, 0);
}

#[test]
fn twinflame_tyrant_doubles_only_friendly_damage_to_opposing_recipients() {
    use crate::support::{
        inspect::{find_owned_card, permanent},
        play::cast_card,
    };
    use managym::state::game_object::Target;

    for caster in [0, 1] {
        for recipient in [0, 1] {
            for creature in [false, true] {
                let mut s =
                    five_color_setup(&["Twinflame Tyrant", "Lightning Bolt", "Wall of Stone"]);
                s.force_permanent_on_battlefield(0, "Twinflame Tyrant");
                let wall = s.force_permanent_on_battlefield(recipient, "Wall of Stone");
                s.advance_to_active_step(caster, StepKind::Main);
                let bolt = find_owned_card(&s, caster, "Lightning Bolt");
                let bolt = move_card(&mut s, bolt, ZoneType::Hand);
                s.set_player_mana_pool(caster, "R");
                cast_card(&mut s, bolt);
                let target = if creature {
                    Target::Permanent(wall)
                } else {
                    Target::Player(PlayerId(recipient))
                };
                assert!(s.choose_target(target));
                settle(&mut s);
                let damage = if creature {
                    permanent(&s, wall).damage
                } else {
                    20 - s.life(recipient)
                };
                assert_eq!(
                    damage,
                    if caster == 0 && recipient == 1 { 6 } else { 3 },
                    "caster {caster}, target {target:?}"
                );
            }
        }
    }
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
fn token_producer_guarded_heir() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Guarded Heir"]);
    enter_and_settle(&mut s, "Guarded Heir");
    crate::assert_creature_tokens(
        &s,
        "Knight",
        2,
        (3, 3),
        &[managym::state::mana::Color::White],
        &[],
    );
}

#[test]
fn token_producer_dragon_trainer() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Dragon Trainer"]);
    enter_and_settle(&mut s, "Dragon Trainer");
    crate::assert_creature_tokens(
        &s,
        "Dragon",
        1,
        (4, 4),
        &[managym::state::mana::Color::Red],
        &[managym::state::card::Keyword::Flying],
    );
}

#[test]
fn token_producer_rite_of_the_dragoncaller() {
    use crate::support::{
        game::five_color_setup,
        play::{enter_and_settle, play_and_settle},
    };
    let mut s = five_color_setup(&["Rite of the Dragoncaller", "Dragon Fodder"]);
    enter_and_settle(&mut s, "Rite of the Dragoncaller");
    play_and_settle(&mut s, "Dragon Fodder");
    crate::assert_creature_tokens(
        &s,
        "Dragon",
        1,
        (5, 5),
        &[managym::state::mana::Color::Red],
        &[managym::state::card::Keyword::Flying],
    );
}

#[test]
fn token_producer_elfsworn_giant() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Elfsworn Giant"]);
    enter_and_settle(&mut s, "Elfsworn Giant");
    enter_and_settle(&mut s, "Forest");
    crate::assert_creature_tokens(
        &s,
        "Elf Warrior",
        1,
        (1, 1),
        &[managym::state::mana::Color::Green],
        &[],
    );
}

#[test]
fn token_producer_infestation_sage() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Infestation Sage"]);
    let source = enter_and_settle(&mut s, "Infestation Sage");
    crate::support::play::move_and_settle(
        &mut s,
        source,
        managym::state::zone::ZoneType::Graveyard,
    );
    crate::assert_creature_tokens(
        &s,
        "Insect",
        1,
        (1, 1),
        &[
            managym::state::mana::Color::Black,
            managym::state::mana::Color::Green,
        ],
        &[managym::state::card::Keyword::Flying],
    );
}

#[test]
fn token_producer_eager_trufflesnout() {
    use crate::support::{
        game::five_color_setup,
        play::{attack_with, enter_and_settle},
    };
    let mut s = five_color_setup(&["Eager Trufflesnout"]);
    enter_and_settle(&mut s, "Eager Trufflesnout");
    let source = crate::support::inspect::permanent_named(&s, "Eager Trufflesnout");
    crate::support::game::clear_summoning_sickness(&mut s, source);
    s.advance_to_active_step(0, managym::flow::turn::StepKind::DeclareAttackers);
    attack_with(&mut s, source);
    s.advance_to_active_step(0, managym::flow::turn::StepKind::PostcombatMain);
    let tokens = s.battlefield_permanents_named(0, "Food");
    assert_eq!(tokens.len(), 1);
}

#[test]
fn printed_creature_quakestrider_ceratops() {
    crate::printed_creature_combat("Quakestrider Ceratops", 12, 8, &[]);
}

#[test]
fn foundations_spell_stab() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, zone_of};
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Stab", "Hill Giant", "Gray Ogre"]);
    let subject = s.force_permanent_on_battlefield(0, "Hill Giant");
    let victim = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let dead = card_of(&s, victim);
    cast(&mut s, "Stab");
    assert!(s.choose_target(Target::Permanent(subject)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(subject).pt(), (1, 1));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(subject).pt(), (3, 3));
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "B");
    cast(&mut s, "Stab");
    assert!(s.choose_target(Target::Permanent(victim)));
    settle(&mut s);
    assert_eq!(zone_of(&s, dead), Some(ZoneType::Graveyard));
}

#[test]
fn foundations_spell_goblin_surprise() {
    use crate::support::game::five_color_setup;
    use crate::support::play::{cast, choose_mode, settle};
    use managym::flow::turn::StepKind;
    let mut s = five_color_setup(&["Goblin Surprise", "Hill Giant"]);
    let subject = s.force_permanent_on_battlefield(0, "Hill Giant");
    cast(&mut s, "Goblin Surprise");
    choose_mode(&mut s, 0);
    settle(&mut s);
    assert_eq!(s.game().characteristics(subject).pt(), (5, 3));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(subject).pt(), (3, 3));
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "2R");
    cast(&mut s, "Goblin Surprise");
    choose_mode(&mut s, 1);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Goblin").len(), 2);
}

#[test]
fn foundations_spell_seekers_folly() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{offered_cards, owned_card, zone_of};
    use crate::support::play::{cast, choose_mode, pass_until, select_card, settle};
    use managym::agent::{action::ActionSpaceKind, observation::Observation};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Seeker's Folly", "Gray Ogre"]);
    let cards = ["Forest", "Island", "Mountain"].map(|name| {
        s.force_card_in_hand(1, name);
        owned_card(&s, 1, name)
    });
    let creature = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Seeker's Folly");
    choose_mode(&mut s, 0);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::Discard);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    let offers = offered_cards(&s);
    assert_eq!(offers.len(), cards.len());
    assert!(cards.iter().all(|card| offers.contains(card)));
    let caster = Observation::for_player(s.game(), PlayerId(0));
    assert!(caster.action_space.actions.is_empty());
    assert!(!caster
        .opponent_cards
        .iter()
        .any(|card| card.zone == ZoneType::Hand));
    for kept in cards {
        let mut branch = s.clone();
        for discarded in cards.into_iter().filter(|card| *card != kept) {
            assert_eq!(branch.action_space().player, Some(PlayerId(1)));
            select_card(&mut branch, discarded);
        }
        settle(&mut branch);
        assert_eq!(branch.zone_size(1, ZoneType::Hand), 1);
        assert_eq!(zone_of(&branch, kept), Some(ZoneType::Hand));
    }
    settle(&mut s);
    cast(&mut s, "Seeker's Folly");
    choose_mode(&mut s, 1);
    settle(&mut s);
    assert_eq!(s.game().characteristics(creature).pt(), (1, 1));
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

#[test]
fn foundations_permanent_clinquant_skymage() {
    use crate::support::game::{five_color_setup, ready};
    use crate::support::inspect::counters;
    use crate::support::play::{activate, settle};
    let mut s = five_color_setup(&["Clinquant Skymage", "Arcanis the Omnipotent"]);
    let source = s.force_permanent_on_battlefield(0, "Clinquant Skymage");
    let arcanis = s.force_permanent_on_battlefield(0, "Arcanis the Omnipotent");
    ready(&mut s, "Arcanis the Omnipotent");
    activate(&mut s, arcanis, 0);
    settle(&mut s);
    assert_eq!(counters(&s, source), 3);
}

#[test]
fn foundations_permanent_elvish_regrower() {
    use crate::support::game::{five_color_setup, plant_in_graveyard};
    use crate::support::inspect::zone_of;
    use crate::support::play::{cast, pass_until, settle, spend_colorless};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Elvish Regrower", "Gray Ogre", "Giant Growth"]);
    let wrong_type = plant_in_graveyard(&mut s, 0, "Giant Growth");
    let theirs = plant_in_graveyard(&mut s, 1, "Gray Ogre");
    let card = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    cast(&mut s, "Elvish Regrower");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(!s.choose_target(Target::GraveyardCard(wrong_type)));
    assert!(!s.choose_target(Target::GraveyardCard(theirs)));
    assert!(s.choose_target(Target::GraveyardCard(card)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Hand));
}

#[test]
fn foundations_grant_twinblade_blessing() {
    use managym::state::card::Keyword;
    crate::check_attachment_grant("Twinblade Blessing", None, (2, 2), &[Keyword::DoubleStrike]);
}

#[test]
fn foundations_5d_rune_sealed_wall() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::tapped;
    use crate::support::play::activate;
    let mut s = five_color_setup(&["Rune-Sealed Wall"]);
    let id = s.force_permanent_on_battlefield(0, "Rune-Sealed Wall");
    crate::support::game::clear_summoning_sickness(&mut s, id);
    activate(&mut s, id, 0);
    assert!(tapped(&s, id));
    crate::assert_surveil_one(&mut s);
}

#[test]
fn foundations_5d_strix_lookout() {
    use crate::support::game::{five_color_setup, move_from_library};
    use crate::support::inspect::{library_top, tapped};
    use crate::support::play::activate;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Strix Lookout", "Gray Ogre"]);
    let before = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    let id = s.force_permanent_on_battlefield(0, "Strix Lookout");
    crate::support::game::clear_summoning_sickness(&mut s, id);
    let top = library_top(&s, 0);
    s.set_player_mana_pool(0, "1U");
    activate(&mut s, id, 0);
    assert!(tapped(&s, id));
    crate::assert_draw_then_discard(&mut s, before, top);
}

#[test]
fn foundations_5d_icewind_elemental() {
    use crate::support::game::{five_color_setup, move_from_library, put_onto_battlefield};
    use crate::support::inspect::library_top;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Icewind Elemental", "Gray Ogre"]);
    let before = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    put_onto_battlefield(&mut s, 0, "Icewind Elemental");
    let top = library_top(&s, 0);
    crate::assert_draw_then_discard(&mut s, before, top);
}

#[test]
fn foundations_5d_billowing_shriekmass() {
    use crate::support::game::{five_color_setup, move_card, move_from_library};
    use crate::support::inspect::permanent_of;
    use crate::support::play::enter_and_settle;
    use managym::state::game_object::PlayerId;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Billowing Shriekmass"]);
    for _ in 0..4 {
        move_from_library(&mut s, 0, "Forest", ZoneType::Graveyard);
    }
    let before = s.zone_size(0, ZoneType::Library);
    let card = enter_and_settle(&mut s, "Billowing Shriekmass");
    let id = permanent_of(&s, card).unwrap();
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 7);
    assert_eq!(s.zone_size(0, ZoneType::Library), before - 4);
    assert_eq!(s.game().characteristics(id).pt(), (4, 4));
    let milled = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Graveyard, PlayerId(0))[0];
    move_card(&mut s, milled, ZoneType::Exile);
    assert_eq!(s.game().characteristics(id).pt(), (2, 3));
}

#[test]
fn foundations_5d_skyknight_squire() {
    use crate::support::game::{five_color_setup, set_counters};
    use crate::support::inspect::{counters, keywords};
    use crate::support::play::enter_and_settle;
    use managym::state::card::Keyword;
    use managym::state::permanent::CounterKind;

    let mut s = five_color_setup(&["Skyknight Squire", "Gray Ogre"]);
    let id = s.force_permanent_on_battlefield(0, "Skyknight Squire");
    for n in 1..=3 {
        enter_and_settle(&mut s, "Gray Ogre");
        assert_eq!(counters(&s, id), n);
        let face = s.game().characteristics(id);
        assert_eq!(face.keywords.has(Keyword::Flying), n == 3);
        assert_eq!(face.has_subtype("Knight"), n == 3);
        assert!(face.has_subtype("Cat") && face.has_subtype("Scout"));
    }
    set_counters(&mut s, id, CounterKind::PlusOnePlusOne, 2);
    assert!(!keywords(&s, id).has(Keyword::Flying));
    assert!(!s.game().characteristics(id).has_subtype("Knight"));
}

#[test]
fn foundations_5d_inspiring_paladin() {
    use crate::support::game::{five_color_setup, move_card, set_counters};
    use crate::support::inspect::{card_of, keywords};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::permanent::CounterKind;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Inspiring Paladin", "Gray Ogre"]);
    let id = s.force_permanent_on_battlefield(0, "Inspiring Paladin");
    let ally = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert!(!keywords(&s, ally).has(Keyword::FirstStrike));
    for target in [ally, enemy] {
        set_counters(&mut s, target, CounterKind::PlusOnePlusOne, 1);
    }
    for (turn, first) in [(0, true), (1, false), (0, true)] {
        s.advance_to_active_step(turn, StepKind::Main);
        assert_eq!(keywords(&s, id).has(Keyword::FirstStrike), first);
        assert_eq!(keywords(&s, ally).has(Keyword::FirstStrike), first);
        assert!(!keywords(&s, enemy).has(Keyword::FirstStrike));
    }
    let source = card_of(&s, id);
    move_card(&mut s, source, ZoneType::Graveyard);
    assert!(!keywords(&s, ally).has(Keyword::FirstStrike));
}

#[test]
fn foundations_5d_quick_draw_katana() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::keywords;
    use crate::support::play::{activate, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Quick-Draw Katana", "Gray Ogre", "Hill Giant"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let gear = s.force_permanent_on_battlefield(0, "Quick-Draw Katana");
    s.set_player_mana_pool(0, "2");
    activate(&mut s, gear, 0);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 2));
    assert!(keywords(&s, ogre).has(Keyword::FirstStrike));
    assert_eq!(s.game().characteristics(giant).pt(), (3, 3));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(ogre).pt(), (2, 2));
    assert!(!keywords(&s, ogre).has(Keyword::FirstStrike));
}

#[test]
fn foundations_5d_starlight_snare() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, permanent_named, tapped};
    use crate::support::play::{cast, resolve_top, settle, spend_colorless};
    use managym::flow::turn::StepKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Starlight Snare", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Starlight Snare");
    assert!(s.choose_target(Target::Permanent(ogre)));
    spend_colorless(&mut s);
    resolve_top(&mut s);
    let aura = permanent_named(&s, "Starlight Snare");
    let source = card_of(&s, aura);
    assert!(!tapped(&s, ogre));
    let mut removed = s.clone();
    move_card(&mut removed, source, ZoneType::Graveyard);
    let mut reentered = removed.clone();
    let wearer = card_of(&reentered, ogre);
    move_card(&mut reentered, wearer, ZoneType::Exile);
    let new_wearer = move_card(&mut reentered, wearer, ZoneType::Battlefield);
    let new_ogre = reentered
        .game()
        .lookup_current_permanent(new_wearer)
        .unwrap();
    settle(&mut reentered);
    assert!(!tapped(&reentered, new_ogre));

    for (branch, held) in [(&mut s, true), (&mut removed, false)] {
        settle(branch);
        assert!(tapped(branch, ogre));
        branch.advance_to_active_step(1, StepKind::Main);
        assert_eq!(tapped(branch, ogre), held);
    }
}

#[test]
fn foundations_5e_boltwave() {
    use crate::support::game::five_color_setup;
    use crate::support::play::{cast, settle};
    use managym::agent::action::ActionSpaceKind;
    let mut s = five_color_setup(&["Boltwave"]);
    cast(&mut s, "Boltwave");
    assert_ne!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    settle(&mut s);
    assert_eq!((s.life(0), s.life(1)), (20, 17));
}

#[test]
fn foundations_5e_faebloom_trick() {
    use crate::support::inspect::{offered_targets, owned_card, tapped};
    use crate::support::play::{cast, cast_card, pass_until};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    use managym::state::mana::Color;
    let mut s = five_color_setup(&["Faebloom Trick", "Gray Ogre", "Unsummon"]);
    s.force_card_in_hand(1, "Unsummon");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Faebloom Trick");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    // The reflexive trigger's target is chosen after both Faeries entered.
    crate::assert_creature_tokens(&s, "Faerie", 2, (1, 1), &[Color::Blue], &[Keyword::Flying]);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    let offers = offered_targets(&s);
    assert_eq!(offers, [Target::Permanent(theirs)]);
    assert!(s.choose_target(Target::Permanent(theirs)));

    let mut response = s.clone();
    assert_eq!(response.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(response.game().state.stack_objects.len(), 1);
    response.pass_priority();
    assert_eq!(response.action_space().player, Some(PlayerId(1)));
    let unsummon = owned_card(&response, 1, "Unsummon");
    cast_card(&mut response, unsummon);
    assert!(response.choose_target(Target::Permanent(theirs)));
    settle(&mut response);
    assert!(response.game().state.permanents[theirs].is_none());
    assert_eq!(response.zone_size(1, ZoneType::Hand), 1);
    crate::assert_creature_tokens(
        &response,
        "Faerie",
        2,
        (1, 1),
        &[Color::Blue],
        &[Keyword::Flying],
    );
    assert!(!tapped(&response, ours));

    settle(&mut s);
    assert!(tapped(&s, theirs));
    assert!(!tapped(&s, ours));
}

#[test]
fn foundations_5e_joust_through() {
    use crate::support::game::{clear_summoning_sickness, five_color_setup};
    use crate::support::inspect::{card_of, offered_targets, zone_of};
    use crate::support::play::{cast, settle};
    use managym::agent::action::ActionSpaceKind;
    use managym::flow::turn::StepKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Joust Through", "Hill Giant", "Gray Ogre"]);
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let idle = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let blockers = [
        s.force_permanent_on_battlefield(1, "Gray Ogre"),
        s.force_permanent_on_battlefield(1, "Gray Ogre"),
    ];
    clear_summoning_sickness(&mut s, giant);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    crate::support::play::attack_with(&mut s, giant);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        crate::support::play::choose(&mut s, |a| {
            matches!(
                a,
                managym::agent::action::Action::DeclareBlocker { attacker: None, .. }
            )
        });
    }
    s.set_player_mana_pool(0, "W");
    cast(&mut s, "Joust Through");
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(giant)));
    assert!(!offers.contains(&Target::Permanent(idle)));
    let in_combat: Vec<_> = blockers
        .into_iter()
        .filter(|id| offers.contains(&Target::Permanent(*id)))
        .collect();
    assert_eq!(in_combat.len(), 1, "only the blocking Ogre");
    let card = card_of(&s, in_combat[0]);
    assert!(s.choose_target(Target::Permanent(in_combat[0])));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
    assert_eq!(s.life(0), 21);
}

#[test]
fn foundations_5e_preposterous_proportions() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::keywords;
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Preposterous Proportions", "Gray Ogre", "Hill Giant"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Preposterous Proportions");
    settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (12, 12));
    assert_eq!(s.game().characteristics(giant).pt(), (13, 13));
    assert!(keywords(&s, ogre).has(Keyword::Vigilance));
    assert_eq!(s.game().characteristics(theirs).pt(), (2, 2));
    assert!(!keywords(&s, theirs).has(Keyword::Vigilance));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(giant).pt(), (3, 3));
    assert!(!keywords(&s, giant).has(Keyword::Vigilance));
}

#[test]
fn foundations_5e_refute() {
    use crate::support::game::move_to_hand;
    use crate::support::inspect::{library_top, now, zone_of};
    use crate::support::play::cast;
    use managym::state::game_object::Target;

    let mut s = five_color_setup(&["Refute", "Lightning Bolt"]);
    let island = move_to_hand(&mut s, "Island");
    let bolt = cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    let bolt = now(&s, bolt);
    cast(&mut s, "Refute");
    assert!(s.choose_target(Target::StackSpell(bolt)));
    spend_colorless(&mut s);
    let drawn = library_top(&s, 0);
    crate::assert_draw_then_discard(&mut s, island, drawn);
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Graveyard));
    s.assert_life(1, 20);
}

#[test]
fn foundations_5e_uncharted_voyage() {
    use crate::support::inspect::{card_of, library, library_top, mode_labels, now, zone_of};
    use crate::support::play::{cast, choose_mode, pass_until};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Uncharted Voyage", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = card_of(&s, ogre);
    // The owner has already looked at its top two cards (as after a scry 2).
    let seen: Vec<_> = library(&s, 1)
        .iter()
        .rev()
        .take(2)
        .map(|object| s.game().state[*object].definition_id)
        .collect();
    let ogre_definition = s.game().state[card].definition_id;
    s.game_mut().state.players[1]
        .known_library
        .own
        .extend([(0, seen[0]), (1, seen[1])]);
    cast(&mut s, "Uncharted Voyage");
    assert!(s.choose_target(Target::Permanent(ogre)));
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::Modal);
    // The owner chooses the end; then the caster surveils.
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(mode_labels(&s), ["On top", "On the bottom"]);
    for (mode, top) in [(0, true), (1, false)] {
        let mut fork = s.clone();
        choose_mode(&mut fork, mode);
        assert_eq!(zone_of(&fork, card), Some(ZoneType::Library));
        let placed = now(&fork, card);
        let size = library(&fork, 1).len();
        // The Ogre came from the battlefield, so both players know where it is;
        // the owner's earlier facts keep their places from the top.
        let known = &fork.game().state.players[1].known_library;
        if top {
            assert_eq!(library_top(&fork, 1), placed);
            assert_eq!(known.public.get(&0), Some(&ogre_definition));
            assert_eq!(known.own.get(&1), Some(&seen[0]));
            assert_eq!(known.own.get(&2), Some(&seen[1]));
        } else {
            assert_eq!(library(&fork, 1)[0], placed);
            assert_eq!(known.public.get(&(size - 1)), Some(&ogre_definition));
            assert_eq!(known.own.get(&0), Some(&seen[0]));
            assert_eq!(known.own.get(&1), Some(&seen[1]));
        }
        crate::assert_surveil_one(&mut fork);
    }
}

#[test]
fn foundations_5f_ambush_wolf() {
    use crate::support::game::plant_in_graveyard;
    use crate::support::inspect::{keywords, permanent_named, zone_of};
    use crate::support::play::enter_and_settle;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Ambush Wolf", "Giant Growth"]);
    // With every graveyard empty the "up to one" target asks nothing.
    let mut empty = s.clone();
    enter_and_settle(&mut empty, "Ambush Wolf");
    let wolf = permanent_named(&empty, "Ambush Wolf");
    assert!(keywords(&empty, wolf).has(Keyword::Flash));
    let ours = plant_in_graveyard(&mut s, 0, "Giant Growth");
    let theirs = plant_in_graveyard(&mut s, 1, "Giant Growth");
    crate::enter_choosing(
        &mut s,
        "Ambush Wolf",
        &[Target::GraveyardCard(ours), Target::GraveyardCard(theirs)],
        Target::GraveyardCard(theirs),
    );
    assert_eq!(zone_of(&s, theirs), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, ours), Some(ZoneType::Graveyard));
}

#[test]
fn foundations_5f_beast_kin_ranger() {
    use crate::support::inspect::library_card;
    use crate::support::play::{enter_and_settle, move_and_settle};
    let mut s = five_color_setup(&["Beast-Kin Ranger", "Gray Ogre"]);
    let card = enter_and_settle(&mut s, "Beast-Kin Ranger");
    let ranger = permanent_of(&s, card).unwrap();
    assert_eq!(s.game().characteristics(ranger).pt(), (3, 3));
    enter_and_settle(&mut s, "Gray Ogre");
    enter_and_settle(&mut s, "Gray Ogre");
    assert_eq!(s.game().characteristics(ranger).pt(), (5, 3));
    let enemy = library_card(&s, 1, "Gray Ogre");
    move_and_settle(&mut s, enemy, ZoneType::Battlefield);
    assert_eq!(s.game().characteristics(ranger).pt(), (5, 3));
    assert!(s
        .game()
        .characteristics(ranger)
        .keywords
        .has(Keyword::Trample));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(ranger).pt(), (3, 3));
}

#[test]
fn foundations_5f_bigfin_bouncer() {
    crate::bounces_an_opposing_creature("Bigfin Bouncer");
}

#[test]
fn foundations_5f_crackling_cyclops() {
    use crate::support::play::{cast, settle};
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Crackling Cyclops", "Lightning Bolt", "Gray Ogre"]);
    let cyclops = s.force_permanent_on_battlefield(0, "Crackling Cyclops");
    cast(&mut s, "Gray Ogre");
    spend_colorless(&mut s);
    settle(&mut s);
    assert_eq!(s.game().characteristics(cyclops).pt(), (0, 4));
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    // The trigger resolves before the Bolt it answered.
    settle(&mut s);
    assert_eq!(s.game().characteristics(cyclops).pt(), (3, 4));
    s.assert_life(1, 17);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(cyclops).pt(), (0, 4));
}

#[test]
fn foundations_5f_dauntless_veteran() {
    crate::attack_pumps_the_team("Dauntless Veteran", (2, 2), true);
}

#[test]
fn foundations_5f_dazzling_angel() {
    crate::gains_life_when_another_creature_enters("Dazzling Angel");
}

#[test]
fn foundations_5f_hinterland_sanctifier() {
    crate::gains_life_when_another_creature_enters("Hinterland Sanctifier");
}

#[test]
fn foundations_5f_elementalist_adept() {
    use crate::support::play::{cast, settle};
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Elementalist Adept", "Lightning Bolt"]);
    let adept = s.force_permanent_on_battlefield(0, "Elementalist Adept");
    assert!(s.game().characteristics(adept).keywords.has(Keyword::Flash));
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(s.game().characteristics(adept).pt(), (3, 2));
}

/// "Put a +1/+1 counter on each of up to two other target creatures you
/// control": Felidar Savior itself and the opponent's creature are never
/// offered, and a target that leaves before resolution takes nothing while
/// the other still gets its counter.
#[test]
fn foundations_5f_felidar_savior() {
    use crate::support::inspect::{counters, offered_targets};
    use crate::support::play::{choose, enter_until_target, settle};
    use managym::agent::action::Action;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Felidar Savior", "Gray Ogre"]);
    let mut empty = s.clone();
    crate::support::play::enter_and_settle(&mut empty, "Felidar Savior");
    let alone = empty.battlefield_permanents_named(0, "Felidar Savior")[0];
    assert_eq!(counters(&empty, alone), 0);
    let ogres = [
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
    ];
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let savior = enter_until_target(&mut s, "Felidar Savior");
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(ogres[0])));
    assert!(offers.contains(&Target::Permanent(ogres[1])));
    assert!(!offers.contains(&Target::Permanent(enemy)));
    let source = permanent_of(&s, savior).unwrap();
    assert!(!offers.contains(&Target::Permanent(source)));
    for chosen in 0..=2 {
        let mut fork = s.clone();
        for ogre in ogres.iter().take(chosen) {
            assert!(fork.choose_target(Target::Permanent(*ogre)));
        }
        if chosen < 2 {
            choose(&mut fork, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut fork);
        for (i, ogre) in ogres.iter().enumerate() {
            assert_eq!(counters(&fork, *ogre), i32::from(i < chosen));
        }
        assert_eq!(counters(&fork, source), 0);
        assert_eq!(counters(&fork, enemy), 0);
    }
    for ogre in ogres {
        assert!(s.choose_target(Target::Permanent(ogre)));
    }
    let departed = card_of(&s, ogres[0]);
    move_card(&mut s, departed, ZoneType::Hand);
    settle(&mut s);
    assert_eq!(counters(&s, ogres[1]), 1);
    assert_eq!(counters(&s, source), 0);
}

#[test]
fn foundations_5f_sanguine_syphoner() {
    let (mut s, ..) = crate::attack_and_resolve("Sanguine Syphoner");
    assert_eq!((s.life(0), s.life(1)), (21, 19));
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!((s.life(0), s.life(1)), (21, 18));
}

/// "{2}{R}: Target creature can't block this turn": the targeted creature
/// is not asked to block this turn and blocks again the next.
#[test]
fn foundations_5f_sower_of_chaos() {
    use crate::support::game::clear_summoning_sickness;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Sower of Chaos", "Gray Ogre"]);
    let sower = s.force_permanent_on_battlefield(0, "Sower of Chaos");
    clear_summoning_sickness(&mut s, sower);
    let stopped = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let mut next_turn = s.clone();
    s.set_player_mana_pool(0, "2R");
    activate(&mut s, sower, 0);
    assert!(s.choose_target(Target::Permanent(stopped)));
    spend_colorless(&mut s);
    settle(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, sower);
    let blocks = crate::walk_block_declaration(&mut s);
    assert_eq!(blocks, [(other, vec![sower])]);
    next_turn.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut next_turn, sower);
    let blocks = crate::walk_block_declaration(&mut next_turn);
    assert_eq!(blocks.len(), 2, "{blocks:?}");
}

/// "{4}{W}: Creatures you control get +1/+1 until end of turn": the bonus
/// reaches the Marshal and its controller's Ogre, not the opposing Ogre.
#[test]
fn foundations_5f_dawnwing_marshal() {
    use crate::support::play::{activate, settle, spend_colorless};
    let mut s = five_color_setup(&["Dawnwing Marshal", "Gray Ogre"]);
    let marshal = s.force_permanent_on_battlefield(0, "Dawnwing Marshal");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "4W");
    activate(&mut s, marshal, 0);
    spend_colorless(&mut s);
    settle(&mut s);
    assert_eq!(s.game().characteristics(marshal).pt(), (3, 3));
    assert_eq!(s.game().characteristics(ours).pt(), (3, 3));
    assert_eq!(s.game().characteristics(theirs).pt(), (2, 2));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(ours).pt(), (2, 2));
}

/// "{2}{G}: Put a +1/+1 counter on target creature you control. Activate
/// only as a sorcery": offered in the main phase with an empty stack, only
/// for the controller's creatures, and never during combat.
#[test]
fn foundations_5f_treetop_snarespinner() {
    use crate::support::{
        inspect::{can_activate, counters, offered_targets},
        play::{activate, settle, spend_colorless},
    };
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Treetop Snarespinner", "Gray Ogre"]);
    let spinner = s.force_permanent_on_battlefield(0, "Treetop Snarespinner");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let mut combat = s.clone();
    activate(&mut s, spinner, 0);
    eprintln!(
        "DBG spinner {spinner:?} ours {ours:?} space {:?}",
        s.action_space()
    );
    let mut offered = offered_targets(&s);
    offered.sort_by_key(|target| format!("{target:?}"));
    let mut wanted = vec![Target::Permanent(spinner), Target::Permanent(ours)];
    wanted.sort_by_key(|target| format!("{target:?}"));
    assert_eq!(offered, wanted);
    assert!(s.choose_target(Target::Permanent(ours)));
    spend_colorless(&mut s);
    settle(&mut s);
    assert_eq!(counters(&s, ours), 1);
    assert_eq!(s.game().characteristics(ours).pt(), (3, 3));

    combat.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(!can_activate(&combat, spinner));
}

/// Apothecary Stomper's entry asks for one mode: two +1/+1 counters on a
/// creature its controller controls, or 4 life.
#[test]
fn foundations_5f_apothecary_stomper() {
    use crate::support::{
        inspect::{counters, mode_labels, offered_targets},
        play::{choose_mode, enter_and_settle, fork_stepping, pass_until, settle},
    };
    use managym::{agent::action::ActionSpaceKind, state::game_object::Target};
    let mut s = five_color_setup(&["Apothecary Stomper", "Gray Ogre"]);
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let _ = enter_and_settle;
    crate::support::game::put_onto_battlefield(&mut s, 0, "Apothecary Stomper");
    pass_until(&mut s, ActionSpaceKind::Modal);
    assert_eq!(mode_labels(&s).len(), 2);
    let mut life = s.clone();
    choose_mode(&mut life, 1);
    settle(&mut life);
    assert_eq!(life.life(0), 24);
    assert_eq!(counters(&life, ours), 0);

    choose_mode(&mut s, 0);
    let mut offered = offered_targets(&s);
    offered.sort_by_key(|target| format!("{target:?}"));
    assert_eq!(offered.len(), 2, "{offered:?}");
    assert!(offered.contains(&Target::Permanent(ours)));
    let _ = fork_stepping;
    assert!(s.choose_target(Target::Permanent(ours)));
    settle(&mut s);
    assert_eq!(counters(&s, ours), 2);
    assert_eq!(s.life(0), 20);
}

/// Spinner of Souls with a friendly Gray Ogre that has just died, `top`
/// pinned on player 0's library, stopped at the "you may" choice.
fn spinner_of_souls_trigger(
    top: &[&str],
) -> (
    crate::support::game::Scenario,
    Vec<managym::state::game_object::ObjectId>,
) {
    use crate::support::{
        game::{move_card, pin_top},
        inspect::card_of,
        play::pass_until,
    };
    use managym::{agent::action::ActionSpaceKind, state::zone::ZoneType};
    let mut s = five_color_setup(&["Spinner of Souls", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Spinner of Souls");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let pinned = pin_top(&mut s, top);
    let card = card_of(&s, ogre);
    move_card(&mut s, card, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::Modal);
    (s, pinned)
}

fn revealed_since(
    s: &crate::support::game::Scenario,
    cursor: usize,
) -> Vec<managym::state::card::CardDefId> {
    use managym::flow::event::GameEvent;
    s.game()
        .state
        .events
        .iter()
        .skip(cursor)
        .filter_map(|event| match event {
            GameEvent::CardRevealed { definition, .. } => Some(*definition),
            _ => None,
        })
        .collect()
}

/// "Whenever another nontoken creature you control dies, you may reveal
/// cards from the top of your library until you reveal a creature card. Put
/// that card into your hand and the rest on the bottom of your library in a
/// random order." The cards stay in the library while revealed: a match on
/// top keeps every known position, one passed card keeps those below it and
/// is publicly known on the bottom, and two or more passed cards are put back
/// in an unknown order.
#[test]
fn foundations_5f_spinner_of_souls() {
    use crate::support::{
        inspect::{hand, library},
        play::{choose_mode, settle},
    };
    use std::collections::BTreeMap;
    let definition = |s: &crate::support::game::Scenario,
                      id: managym::state::game_object::ObjectId| {
        s.game().state[id].definition_id
    };

    // Declining reveals nothing.
    let (mut s, _) = spinner_of_souls_trigger(&["Forest", "Gray Ogre"]);
    let order = library(&s, 0);
    let cursor = s.game().state.events.len();
    choose_mode(&mut s, 1);
    settle(&mut s);
    assert_eq!(library(&s, 0), order);
    assert!(hand(&s, 0).is_empty());
    assert!(revealed_since(&s, cursor).is_empty());

    // A creature on top: positions below it shift up one.
    let (mut s, pinned) = spinner_of_souls_trigger(&["Gray Ogre", "Forest", "Island", "Plains"]);
    let (island, plains) = (definition(&s, pinned[2]), definition(&s, pinned[3]));
    let known = &mut s.game_mut().state.players[0].known_library;
    known.own = BTreeMap::from([(2, island), (3, plains)]);
    known.public = BTreeMap::from([(3, plains)]);
    let cursor = s.game().state.events.len();
    choose_mode(&mut s, 0);
    settle(&mut s);
    assert_eq!(
        hand(&s, 0),
        vec![crate::support::inspect::now(&s, pinned[0])]
    );
    assert_eq!(
        revealed_since(&s, cursor),
        vec![definition(&s, crate::support::inspect::now(&s, pinned[0]))]
    );
    let known = &s.game().state.players[0].known_library;
    assert_eq!(known.own, BTreeMap::from([(1, island), (2, plains)]));
    assert_eq!(known.public, BTreeMap::from([(2, plains)]));

    // One land passed: it goes to the bottom and the facts below it hold.
    let (mut s, pinned) = spinner_of_souls_trigger(&["Forest", "Gray Ogre", "Island", "Plains"]);
    let (forest, island, plains) = (
        definition(&s, pinned[0]),
        definition(&s, pinned[2]),
        definition(&s, pinned[3]),
    );
    s.game_mut().state.players[0].known_library.own = BTreeMap::from([(2, island), (3, plains)]);
    let cursor = s.game().state.events.len();
    choose_mode(&mut s, 0);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 1);
    assert_eq!(s.game().state[hand(&s, 0)[0]].name, "Gray Ogre");
    assert_eq!(library(&s, 0)[0], pinned[0], "the Forest is on the bottom");
    assert_eq!(revealed_since(&s, cursor)[0], forest);
    let bottom = library(&s, 0).len() - 1;
    let known = &s.game().state.players[0].known_library;
    assert_eq!(
        known.own,
        BTreeMap::from([(0, island), (1, plains), (bottom, forest)])
    );
    assert_eq!(
        known.public,
        BTreeMap::from([(bottom, forest)]),
        "the revealed Forest is publicly known on the bottom"
    );

    // Two lands passed: both go to the bottom in a random order.
    let (mut s, pinned) = spinner_of_souls_trigger(&["Forest", "Island", "Gray Ogre"]);
    choose_mode(&mut s, 0);
    settle(&mut s);
    assert_eq!(s.game().state[hand(&s, 0)[0]].name, "Gray Ogre");
    let library = library(&s, 0);
    let mut bottom = library[..2].to_vec();
    bottom.sort();
    let mut lands = pinned[..2].to_vec();
    lands.sort();
    assert_eq!(bottom, lands);
    assert!(s.game().state.players[0].known_library.is_empty());
}

/// With no creature card left in the library, every card is revealed and
/// none goes to hand; an opposing creature's death does not trigger it.
#[test]
fn foundations_5f_spinner_of_souls_without_a_creature() {
    use crate::support::{
        game::move_card,
        inspect::{card_of, hand, library},
        play::{choose_mode, settle},
    };
    use managym::{agent::action::ActionSpaceKind, state::zone::ZoneType};
    let (mut s, _) = spinner_of_souls_trigger(&[]);
    for object in library(&s, 0) {
        if s.game().state[object].types.is_creature() {
            move_card(&mut s, object, ZoneType::Exile);
        }
    }
    let mut cards = library(&s, 0);
    let cursor = s.game().state.events.len();
    choose_mode(&mut s, 0);
    settle(&mut s);
    assert!(hand(&s, 0).is_empty());
    assert_eq!(revealed_since(&s, cursor).len(), cards.len());
    let mut after = library(&s, 0);
    cards.sort();
    after.sort();
    assert_eq!(after, cards);

    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = card_of(&s, theirs);
    move_card(&mut s, card, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert_ne!(s.action_space().kind, ActionSpaceKind::Modal);
    assert!(s.game().state.pending_triggers.is_empty());
}

/// Player 0 attacks with a Gray Ogre (beside Twinflame Tyrant when
/// `tyrant`); player 1's `blocker` carries `shields` Fleeting Flights cast
/// by player 0 and blocks it. Stops at the first decision after blocks.
fn fleeting_flight_block(
    blocker: &str,
    shields: usize,
    tyrant: bool,
) -> (
    crate::support::game::Scenario,
    managym::state::game_object::PermanentId,
    managym::state::game_object::PermanentId,
) {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup},
        play::{attack_with, cast, settle},
    };
    use managym::{flow::turn::StepKind, state::game_object::Target};
    let mut s = five_color_setup(&[
        "Fleeting Flight",
        "Twinflame Tyrant",
        "Gray Ogre",
        "Fog Bank",
        "Lightning Bolt",
    ]);
    if tyrant {
        s.force_permanent_on_battlefield(0, "Twinflame Tyrant");
    }
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, attacker);
    let blocker = s.force_permanent_on_battlefield(1, blocker);
    for _ in 0..shields {
        s.set_player_mana_pool(0, "W");
        cast(&mut s, "Fleeting Flight");
        assert!(s.choose_target(Target::Permanent(blocker)));
        settle(&mut s);
    }
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, attacker);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    (s, attacker, blocker)
}

/// "Put a +1/+1 counter on target creature. It gains flying until end of
/// turn. Prevent all combat damage that would be dealt to it this turn." The
/// shield is a resolved record on that creature, offered beside Twinflame
/// Tyrant's doubling in either order and named for the spell that made it.
/// The shielded creature still deals combat damage.
#[test]
fn foundations_5g_fleeting_flight() {
    use crate::support::{
        inspect::{card_of, counters_of, keywords, offer_labels, permanent, zone_of},
        play::{command, pass_until},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        flow::turn::StepKind,
        state::{game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
    };
    let (mut s, attacker, blocker) = fleeting_flight_block("Gray Ogre", 1, true);
    assert_eq!(counters_of(&s, blocker, CounterKind::PlusOnePlusOne), 1);
    assert!(keywords(&s, blocker).has(managym::state::card::Keyword::Flying));
    pass_until(&mut s, ActionSpaceKind::Modal);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    let mut labels = offer_labels(&s);
    labels.sort();
    assert_eq!(
        labels,
        [
            "Fleeting Flight: Prevent all combat damage",
            "Twinflame Tyrant: Double the damage",
        ]
    );
    let attacker_card = card_of(&s, attacker);
    for mode in 0..2 {
        let mut branch = s.clone();
        let mut replay = command(
            &mut branch,
            |action| matches!(action, Action::ChooseMode { mode: choice, .. } if *choice == mode),
        );
        for game in [&mut branch, &mut replay] {
            game.advance_to_active_step(0, StepKind::EndOfCombat);
            assert_eq!(permanent(game, blocker).damage, 0, "mode {mode}");
            assert_eq!(
                zone_of(game, attacker_card),
                Some(ZoneType::Graveyard),
                "the shielded 3/3 still deals its damage"
            );
        }
    }
}

/// Two shields and Fog Bank's own prevention are three distinct
/// replacements: all are offered with Twinflame Tyrant's doubling, and after
/// the doubling all three prevention choices remain.
#[test]
fn foundations_5g_fleeting_flight_shields_do_not_collide() {
    use crate::support::{
        inspect::{offer_labels, permanent},
        play::{choose_mode, pass_until},
    };
    use managym::{agent::action::ActionSpaceKind, flow::turn::StepKind};
    let (mut s, _, fog) = fleeting_flight_block("Fog Bank", 2, true);
    pass_until(&mut s, ActionSpaceKind::Modal);
    let labels = offer_labels(&s);
    let mut sorted = labels.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        [
            "Fleeting Flight: Prevent all combat damage",
            "Fleeting Flight: Prevent all combat damage",
            "Fog Bank: Prevent all combat damage",
            "Twinflame Tyrant: Double the damage",
        ]
    );
    let double = labels
        .iter()
        .position(|label| label.starts_with("Twinflame Tyrant"))
        .unwrap();
    choose_mode(&mut s, double);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    let mut rest = offer_labels(&s);
    rest.sort();
    assert_eq!(
        rest,
        [
            "Fleeting Flight: Prevent all combat damage",
            "Fleeting Flight: Prevent all combat damage",
            "Fog Bank: Prevent all combat damage",
        ]
    );
    choose_mode(&mut s, 0);
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    assert_eq!(permanent(&s, fog).damage, 0);
}

/// The shield prevents only combat damage, survives losing all abilities
/// (it is an effect, not an ability) and ends in the cleanup step.
#[test]
fn foundations_5g_fleeting_flight_controls() {
    use crate::support::{
        game::{move_to_hand, until_end_of_turn},
        inspect::{card_of, counters_of, keywords, permanent, zone_of},
        play::{cast_card, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{
            effect::Modification, game_object::Target, permanent::CounterKind, zone::ZoneType,
        },
    };

    // Noncombat damage is dealt: Lightning Bolt kills the shielded 3/3.
    let (mut s, _, blocker) = fleeting_flight_block("Gray Ogre", 1, false);
    let card = card_of(&s, blocker);
    let bolt = move_to_hand(&mut s, "Lightning Bolt");
    s.set_player_mana_pool(0, "R");
    cast_card(&mut s, bolt);
    assert!(s.choose_target(Target::Permanent(blocker)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));

    // Losing all abilities leaves the shield in place.
    let (mut s, _, blocker) = fleeting_flight_block("Gray Ogre", 1, false);
    until_end_of_turn(&mut s, blocker, Modification::LoseAllAbilities);
    assert!(!keywords(&s, blocker).has(managym::state::card::Keyword::Flying));
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    assert_eq!(permanent(&s, blocker).damage, 0);

    // The shield and flying end at cleanup; the counter stays.
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!s
        .game()
        .effects_on(blocker)
        .any(|effect| effect.modification == Modification::PreventCombatDamage));
    assert!(!keywords(&s, blocker).has(managym::state::card::Keyword::Flying));
    assert_eq!(counters_of(&s, blocker, CounterKind::PlusOnePlusOne), 1);
}

/// "Landfall — Whenever a land you control enters, tap target creature an
/// opponent controls and put a stun counter on it." The stun counter
/// replaces its next untap, in the untap step or by an effect.
#[test]
fn foundations_5g_grappling_kraken() {
    use crate::support::{
        game::five_color_setup,
        inspect::{counters_of, offered_targets, tapped},
        play::{cast, play_land, settle},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        flow::turn::StepKind,
        state::{game_object::Target, permanent::CounterKind},
    };
    let mut s = five_color_setup(&["Grappling Kraken", "Gray Ogre", "Involuntary Employment"]);
    let kraken = s.force_permanent_on_battlefield(0, "Grappling Kraken");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    play_land(&mut s, "Forest");
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(ogre)));
    assert!(!offers.contains(&Target::Permanent(kraken)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert!(tapped(&s, ogre));
    assert_eq!(counters_of(&s, ogre, CounterKind::Stun), 1);

    // An effect's untap removes the stun counter instead.
    let mut effect = s.clone();
    effect.set_player_mana_pool(0, "3R");
    cast(&mut effect, "Involuntary Employment");
    assert!(effect.choose_target(Target::Permanent(ogre)));
    settle(&mut effect);
    assert!(tapped(&effect, ogre));
    assert_eq!(counters_of(&effect, ogre, CounterKind::Stun), 0);

    // So does the untap step; the next one untaps it.
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(tapped(&s, ogre));
    assert_eq!(counters_of(&s, ogre, CounterKind::Stun), 0);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!tapped(&s, ogre));
}

#[test]
fn zul_ashur_grants_one_exact_zombie_for_the_turn() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_from_library},
        inspect::{assert_choice_contract, can_cast, zone_of},
        play::{choose, fork_choosing},
    };
    use managym::{
        agent::{action::Action, observation::Observation},
        state::game_object::Target,
    };

    let mut s = five_color_unfunded_setup(&[
        "Zul Ashur, Lich Lord",
        "Hungry Ghoul",
        "Gray Ogre",
        "Counterspell",
    ]);
    let zul = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Zul Ashur, Lich Lord", true)
        .unwrap();
    let zul_card = card_of(&s, zul);
    let zombie = move_from_library(&mut s, 0, "Hungry Ghoul", ZoneType::Graveyard);
    let other = move_from_library(&mut s, 0, "Hungry Ghoul", ZoneType::Graveyard);
    move_from_library(&mut s, 1, "Hungry Ghoul", ZoneType::Graveyard);
    move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, zul, 0);
    let targets: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::ChooseTarget {
                target: Target::GraveyardCard(card),
                ..
            } => Some(*card),
            _ => None,
        })
        .collect();
    assert_eq!(targets.len(), 2);
    assert!(targets.contains(&zombie) && targets.contains(&other));
    assert_choice_contract(&s);
    assert!(s.choose_target(Target::GraveyardCard(zombie)));
    // The activated ability resolves independently of its source.
    move_card(&mut s, zul_card, ZoneType::Graveyard);
    settle(&mut s);
    assert!(!can_cast(&s, zombie)); // The printed mana cost is still owed.
    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        let permitted: Vec<_> = obs
            .agent_cards
            .iter()
            .chain(&obs.opponent_cards)
            .filter_map(|card| Some((card, card.cast_permissions.first()?)))
            .collect();
        let [(card, grant)] = permitted.as_slice() else {
            panic!("expected exactly one public grant")
        };
        assert_eq!(card.id, s.game().state[zombie].id.0 as i32);
        assert_eq!(card.zone, ZoneType::Graveyard);
        assert!(grant.this_turn);
        assert_eq!(grant.cost.as_ref().unwrap().mana_value, 2);
        assert_eq!(grant.player_id, s.game().state.players[0].id.0 as i32);
    }

    s.set_player_mana_pool(0, "1B");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, zombie));
    assert!(!can_cast(&s, other));
    assert_choice_contract(&s);
    let cast = |a: &Action| matches!(a, Action::CastSpell { object, .. } if *object == zombie);
    let mut resolved = fork_choosing(&s, cast);
    settle(&mut resolved);
    assert_eq!(zone_of(&resolved, zombie), Some(ZoneType::Battlefield));
    assert_eq!(resolved.game().state.players[0].mana_pool.total(), 0);

    let mut expired = fork_choosing(&s, |a| matches!(a, Action::PassPriority { .. }));
    expired.advance_to_active_step(1, StepKind::Main);
    expired.advance_to_active_step(0, StepKind::Main);
    expired.set_player_mana_pool(0, "1B");
    expired.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&expired, zombie));

    // Casting and countering creates two new appearances. Neither inherits
    // the grant, and an ordinary graveyard cast has no flashback replacement.
    let counter = move_from_library(&mut s, 1, "Counterspell", ZoneType::Hand);
    s.set_player_mana_pool(1, "UU");
    choose(&mut s, cast);
    let spell = crate::support::inspect::now(&s, zombie);
    s.pass_priority();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { object, .. } if *object == counter),
    );
    assert!(s.choose_target(Target::StackSpell(spell)));
    settle(&mut s);
    assert_eq!(zone_of(&s, zombie), Some(ZoneType::Graveyard));
    s.set_player_mana_pool(0, "1B");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, zombie));
}

#[test]
fn zul_ashur_ward_costs_the_opponent_two_life() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands},
        play::{cast_only, resolve_top},
    };
    use managym::agent::action::ActionType;

    let mut s = five_color_unfunded_setup(&["Zul Ashur, Lich Lord", "Lightning Bolt"]);
    let zul = s.force_permanent_on_battlefield(0, "Zul Ashur, Lich Lord");
    let zombie = card_of(&s, zul);
    force_lands(&mut s, 1, "Mountain", 1);
    s.force_card_in_hand(1, "Lightning Bolt");
    s.advance_to_active_step(1, StepKind::Main);
    cast_only(&mut s);
    s.choose_target_named("Zul Ashur, Lich Lord");
    resolve_top(&mut s);
    assert!(s.take_action_by_type(ActionType::PayCost));
    s.assert_life(1, 18);
    settle(&mut s);
    assert_eq!(
        crate::support::inspect::zone_of(&s, zombie),
        Some(ZoneType::Graveyard)
    );
}

#[test]
fn overlapping_grants_price_each_cast_in_both_orders() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_from_library},
        play::fork_choosing,
    };
    use managym::{
        agent::action::Action,
        flow::cast_permission::{PlayGrant, PlayRoute},
        state::mana::ManaCost,
    };
    for free_first in [false, true] {
        let mut s = five_color_unfunded_setup(&["Hungry Ghoul"]);
        let card = move_from_library(&mut s, 0, "Hungry Ghoul", ZoneType::Graveyard);
        let grant = PlayGrant::new(card, PlayerId(0));
        let free = PlayGrant {
            cost: Some(ManaCost::default()),
            ..grant.clone()
        };
        for terms in if free_first {
            [free, grant]
        } else {
            [grant, free]
        } {
            s.game_mut().state.cast_permissions.grant(terms);
        }
        s.game_mut().state.cast_permissions.grants.reverse();
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = managym::agent::observation::Observation::for_player(s.game(), viewer);
            let listed = obs
                .agent_cards
                .iter()
                .chain(&obs.opponent_cards)
                .find(|c| c.id == s.game().state[card].id.0 as i32)
                .unwrap();
            let mut costs: Vec<_> = listed
                .cast_permissions
                .iter()
                .map(|g| (g.player_id, g.cost.as_ref().unwrap().mana_value))
                .collect();
            costs.sort();
            let caster = s.game().state.players[0].id.0 as i32;
            assert_eq!(costs, [(caster, 0), (caster, 2)]);
            use managym::agent::{
                observation_encoder::{encode, ObservationEncoderConfig, FACT_DIM, PLAY_GRANT},
                observation_layout::fact,
            };
            let input = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
            let programs: Vec<_> = input
                .facts
                .chunks_exact(FACT_DIM)
                .filter(|row| {
                    let id = ((row[fact::ID_HIGH] as u32) << 16) | row[fact::ID_LOW] as u32;
                    id == managym::cards::tokens::name_value(PLAY_GRANT) as u32
                        && row[fact::VALUE_2] == 0.
                })
                .collect();
            assert_eq!(programs.len(), 2);
            assert!(programs[0][fact::OBJECT] > 0.);
            assert_eq!(programs[0][fact::OBJECT], programs[1][fact::OBJECT]);
            let restored: managym::agent::observation::Observation =
                serde_json::from_str(&obs.to_json()).unwrap();
            assert_eq!(restored.to_json(), obs.to_json());
        }
        for funded in [false, true] {
            if funded {
                s.set_player_mana_pool(0, "1B");
            }
            s.game_mut().scenario_refresh_priority().unwrap();
            let routes: Vec<_> = s
                .action_space()
                .actions
                .iter()
                .filter(|a| matches!(a,Action::CastSpell {object,..} if *object==card))
                .cloned()
                .collect();
            assert_eq!(routes.len(), if funded { 2 } else { 1 });
            for action in routes {
                let Action::CastSpell {
                    route: Some(PlayRoute::Grant(index)),
                    ..
                } = action
                else {
                    panic!("unbound grant")
                };
                let free = index == usize::from(!free_first);
                assert!(funded || free);
                let branch = fork_choosing(&s, |a| *a == action);
                assert_eq!(
                    branch.game().state.players[0].mana_pool.total(),
                    if free && funded { 2 } else { 0 }
                );
            }
        }
    }
}

#[test]
fn muldrotha_and_zul_offer_independent_casts() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_from_library},
        inspect::{can_cast, now, zone_of},
        play::{choose, fork_choosing},
    };
    use managym::{
        agent::action::Action, flow::cast_permission::PlayRoute, state::game_object::Target,
    };
    let mut s = five_color_unfunded_setup(&[
        "Muldrotha, the Gravetide",
        "Zul Ashur, Lich Lord",
        "Hungry Ghoul",
        "Counterspell",
    ]);
    let muldrotha = s.force_permanent_on_battlefield(0, "Muldrotha, the Gravetide");
    let source_id = s.game().state.permanents[muldrotha].as_ref().unwrap().id.0;
    let zul = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Zul Ashur, Lich Lord", true)
        .unwrap();
    let zombie = move_from_library(&mut s, 0, "Hungry Ghoul", ZoneType::Graveyard);
    let other = move_from_library(&mut s, 0, "Hungry Ghoul", ZoneType::Graveyard);
    let counter = move_from_library(&mut s, 1, "Counterspell", ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, zul, 0);
    assert!(s.choose_target(Target::GraveyardCard(zombie)));
    settle(&mut s);
    s.set_player_mana_pool(0, "1B");
    s.set_player_mana_pool(1, "UU");
    s.game_mut().scenario_refresh_priority().unwrap();
    let obs = managym::agent::observation::Observation::for_player(s.game(), PlayerId(0));
    use managym::agent::{
        observation_encoder::{encode, ObservationEncoderConfig, ACTION_DIM},
        observation_layout::action,
    };
    let encoded = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
    let mut labels = Vec::new();
    for (index, offer) in obs
        .action_space
        .actions
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            a.focus.first() == Some(&(s.game().state[zombie].id.0 as i32))
                && a.route_mana_value.is_some()
        })
    {
        let label = offer.description.as_deref().unwrap();
        let row = &encoded.actions[index * ACTION_DIM..][..ACTION_DIM];
        assert_eq!(row[action::ROUTE_MANA_VALUE], 0.2);
        assert_eq!(row[action::QUOTA_TYPE] > 0., label.contains("Muldrotha"));
        labels.push(label);
    }
    assert_eq!(labels.len(), 2);
    assert!(labels.iter().any(|label| label.contains("Zul Ashur")));
    for use_quota in [false, true] {
        let mut branch = fork_choosing(
            &s,
            |a| matches!(a,Action::CastSpell {object,route:Some(route),..} if *object==zombie && matches!(route,PlayRoute::GraveyardType {..})==use_quota),
        );
        let spell = now(&branch, zombie);
        branch.pass_priority();
        choose(
            &mut branch,
            |a| matches!(a,Action::CastSpell {object,..} if *object==counter),
        );
        assert!(branch.choose_target(Target::StackSpell(spell)));
        settle(&mut branch);
        assert_eq!(zone_of(&branch, zombie), Some(ZoneType::Graveyard));
        branch.set_player_mana_pool(0, "1B");
        branch.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(can_cast(&branch, other), !use_quota);
        assert_eq!(can_cast(&branch, zombie), !use_quota);
        for card in [zombie, other] {
            crate::support::game::move_card(&mut branch, card, ZoneType::Exile);
        }
        branch.game_mut().scenario_refresh_priority().unwrap();
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = managym::agent::observation::Observation::for_player(branch.game(), viewer);
            let owner = if viewer == PlayerId(0) {
                &obs.agent
            } else {
                &obs.opponent
            };
            assert!(obs
                .agent_cards
                .iter()
                .chain(&obs.opponent_cards)
                .all(|card| card.owner_id != owner.id || card.zone != ZoneType::Graveyard));
            assert_eq!(owner.spent_graveyard_types.len(), usize::from(use_quota));
            if use_quota {
                assert_eq!(
                    owner.spent_graveyard_types[0],
                    (source_id, managym::state::card::CardType::Creature)
                );
            }
        }
    }
}

#[test]
fn lone_graveyard_routes_keep_their_permission_in_the_observation() {
    use crate::support::game::{five_color_unfunded_setup, move_from_library};
    use managym::{
        agent::{action::Action, observation::Observation},
        state::game_object::Target,
    };
    // A single-route Command names no route, but the offer still shows which
    // permission it uses: Muldrotha's quota and focus, or a grant's price.
    let lone_offer = |s: &crate::support::game::Scenario, card| {
        let obs = Observation::for_player(s.game(), PlayerId(0));
        let index = s
            .action_space()
            .actions
            .iter()
            .position(
                |a| matches!(a, Action::CastSpell { object, route: None, .. } if *object == card),
            )
            .expect("a single-route cast of the graveyard card");
        obs.action_space.actions[index].clone()
    };

    let mut s = five_color_unfunded_setup(&["Muldrotha, the Gravetide", "Hungry Ghoul"]);
    let muldrotha = s.force_permanent_on_battlefield(0, "Muldrotha, the Gravetide");
    let ghoul = move_from_library(&mut s, 0, "Hungry Ghoul", ZoneType::Graveyard);
    s.set_player_mana_pool(0, "1B");
    s.game_mut().scenario_refresh_priority().unwrap();
    let offer = lone_offer(&s, ghoul);
    let source = s.game().state.permanents[muldrotha]
        .as_ref()
        .unwrap()
        .object;
    assert_eq!(
        offer.quota_type,
        Some(managym::state::card::CardType::Creature as u8 + 1)
    );
    assert_eq!(
        offer.focus,
        [
            s.game().state[ghoul].id.0 as i32,
            s.game().state[source].id.0 as i32
        ]
    );
    assert!(offer.description.unwrap().contains("Muldrotha"));

    let mut s = five_color_unfunded_setup(&["Zul Ashur, Lich Lord", "Hungry Ghoul"]);
    let zul = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Zul Ashur, Lich Lord", true)
        .unwrap();
    let ghoul = move_from_library(&mut s, 0, "Hungry Ghoul", ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, zul, 0);
    assert!(s.choose_target(Target::GraveyardCard(ghoul)));
    settle(&mut s);
    s.set_player_mana_pool(0, "1B");
    s.game_mut().scenario_refresh_priority().unwrap();
    let offer = lone_offer(&s, ghoul);
    assert_eq!(offer.route_mana_value, Some(2));
    assert_eq!(offer.quota_type, None);
    assert!(offer.description.unwrap().contains("Zul Ashur"));
}

#[test]
fn stale_ghoul_precondition_preserves_mana_and_muldrotha_allowance() {
    use crate::support::game::{five_color_unfunded_setup, move_from_library};
    use managym::{
        agent::action::Action,
        decision::{Command, SemanticError},
    };
    let mut s =
        five_color_unfunded_setup(&["Hungry Ghoul", "Lightning Bolt", "Muldrotha, the Gravetide"]);
    let ghoul = s.force_permanent_on_battlefield(0, "Hungry Ghoul");
    move_from_library(&mut s, 0, "Lightning Bolt", ZoneType::Hand);
    s.set_player_mana_pool(0, "R");
    s.game_mut().scenario_refresh_priority().unwrap();
    let frame = s.game().semantic_decision_frame().unwrap();
    let [address] = frame.object_candidates.as_slice() else {
        panic!("only the Ghoul can be targeted as a permanent")
    };
    let card = card_of(&s, ghoul);
    move_card(&mut s, card, ZoneType::Exile);
    let card = move_card(&mut s, card, ZoneType::Graveyard);
    s.force_permanent_on_battlefield(0, "Muldrotha, the Gravetide");
    s.set_player_mana_pool(0, "1B");
    s.game_mut().scenario_refresh_priority().unwrap();
    let frame = s.game().semantic_decision_frame().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::CastSpell { object, .. } if *object == card))
        .unwrap();
    let command = Command {
        command_id: "stale-ghoul".into(),
        expected_revision: frame.revision,
        offer_id: frame.offers[index].id.0,
        answers: vec![],
        object_preconditions: vec![address.clone()],
    };
    let before = managym::search_state::witness(s.game());
    assert!(matches!(
        s.game_mut().execute_semantic_command(&command),
        Err(SemanticError::StaleObject)
    ));
    assert_eq!(before, managym::search_state::witness(s.game()));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 2);
    assert!(s.game().state.cast_permissions.spent_types.is_empty());
}

#[test]
fn muldrotha_spends_the_chosen_type_for_this_incarnation_and_turn() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_from_library},
        inspect::can_cast,
        play::{choose, fork_choosing},
    };
    use managym::{agent::action::Action, flow::cast_permission::PlayRoute, state::card::CardType};
    let mut s = five_color_unfunded_setup(&[
        "Muldrotha, the Gravetide",
        "Scarecrow Guide",
        "Hungry Ghoul",
    ]);
    let source = s.force_permanent_on_battlefield(0, "Muldrotha, the Gravetide");
    let dual = move_from_library(&mut s, 0, "Scarecrow Guide", ZoneType::Graveyard);
    let ghoul = move_from_library(&mut s, 0, "Hungry Ghoul", ZoneType::Graveyard);
    let swamp = move_from_library(&mut s, 0, "Swamp", ZoneType::Graveyard);
    let forest = move_from_library(&mut s, 0, "Forest", ZoneType::Graveyard);
    s.set_player_mana_pool(0, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    for kind in [CardType::Artifact, CardType::Creature] {
        let mut branch = fork_choosing(&s, |action| {
            matches!(action,
            Action::CastSpell {object, route:Some(PlayRoute::GraveyardType {kind:chosen,..}),..}
            if *object==dual && *chosen==kind)
        });
        settle(&mut branch);
        branch.set_player_mana_pool(0, "1B");
        branch.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(can_cast(&branch, ghoul), kind == CardType::Artifact);
        choose(
            &mut branch,
            |action| matches!(action,Action::PlayLand {object,..} if *object==swamp),
        );
        assert!(!branch
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a,Action::PlayLand {object,..} if *object==forest)));
        if kind == CardType::Creature {
            let card = card_of(&branch, source);
            let exiled = move_card(&mut branch, card, ZoneType::Exile);
            move_card(&mut branch, exiled, ZoneType::Battlefield);
            settle(&mut branch);
            branch.game_mut().scenario_refresh_priority().unwrap();
            assert!(can_cast(&branch, ghoul));
        } else {
            branch.advance_to_active_step(1, StepKind::Main);
            assert!(!can_cast(&branch, ghoul));
            branch.advance_to_active_step(0, StepKind::Main);
            assert!(branch
                .action_space()
                .actions
                .iter()
                .any(|a| matches!(a,Action::PlayLand {object,..} if *object==forest)));
        }
    }
}

#[test]
fn omniscience_offers_paid_and_free_casts_with_optional_costs() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_from_library},
        inspect::top_spell,
        play::{choose, fork_choosing, spend_named},
    };
    use managym::{
        agent::{action::Action, observation::Observation},
        flow::cast_permission::PlayRoute,
        state::{mana::Color, player::ManaPayment, target::Target},
    };
    let mut s = five_color_unfunded_setup(&["Omniscience", "Burst Lightning"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    for _ in 0..2 {
        s.force_permanent_on_battlefield(0, "Omniscience");
    }
    let burst = move_from_library(&mut s, 0, "Burst Lightning", ZoneType::Hand);
    s.set_player_mana_pool(0, "4R");
    s.game_mut().scenario_refresh_priority().unwrap();
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let prices: Vec<_> = obs
        .action_space
        .actions
        .iter()
        .filter_map(|offer| offer.route_mana_value)
        .collect();
    assert_eq!(prices, [1, 0]);
    for (route, price) in [(PlayRoute::Hand, 1), (PlayRoute::FreeFromHand, 0)] {
        let root = fork_choosing(
            &s,
            |a| matches!(a, Action::CastSpell { object, route: Some(chosen), .. } if *object == burst && *chosen == route),
        );
        for kicked in [false, true] {
            let mut branch = fork_choosing(&root, |a| {
                if kicked {
                    matches!(a, Action::PayCost { .. })
                } else {
                    matches!(a, Action::Decline { .. })
                }
            });
            assert!(branch.choose_target(Target::Player(PlayerId(1))));
            if branch.action_space().kind == managym::agent::action::ActionSpaceKind::PayOrNot {
                choose(&mut branch, |a| matches!(a, Action::PayCost { .. }));
            }
            let spell = top_spell(&branch);
            assert_eq!(spell.spent.mana().len(), price + if kicked { 4 } else { 0 });
            assert_eq!(spell.kicked, kicked);
            settle(&mut branch);
            assert_eq!(
                branch.game().state.players[1].life,
                if kicked { 16 } else { 18 }
            );
        }
    }

    s.game_mut().mana_payment[0] = ManaPayment::Manual;
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { object, route: Some(PlayRoute::FreeFromHand), .. } if *object == burst),
    );
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    spend_named(&mut s, |label| label == "Spend {R} (expires at step end)");
    spend_colorless(&mut s);
    let spell = top_spell(&s);
    assert!(spell.kicked);
    assert_eq!(
        spell
            .spent
            .mana()
            .iter()
            .map(|unit| unit.color)
            .collect::<Vec<_>>(),
        [
            Color::Red as u8,
            Color::Colorless as u8,
            Color::Colorless as u8,
            Color::Colorless as u8,
        ]
    );
    settle(&mut s);
    s.assert_life(1, 16);
}

#[test]
fn high_fae_trickster_changes_timing_after_zone_permission() {
    use crate::support::{game::move_from_library, inspect::can_cast};
    use managym::flow::cast_permission::PlayGrant;
    let mut s = five_color_setup(&["High Fae Trickster", "Gray Ogre", "Drannith Magistrate"]);
    s.advance_to_active_step(0, StepKind::Upkeep);
    let ogre = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    s.set_player_mana_pool(0, "2R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, ogre));
    s.force_permanent_on_battlefield(0, "High Fae Trickster");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, ogre));
    let exile = move_card(&mut s, ogre, ZoneType::Exile);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, exile));
    s.game_mut()
        .state
        .cast_permissions
        .grant(PlayGrant::new(exile, PlayerId(0)));
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, exile));
    s.force_permanent_on_battlefield(1, "Drannith Magistrate");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, exile));
}

#[test]
fn think_twice_draws_from_hand_then_graveyard() {
    use crate::support::{
        game::move_from_library,
        inspect::{now, top_spell, zone_of},
        play::choose,
    };
    use managym::agent::action::Action;
    let mut s = five_color_setup(&["Think Twice"]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let card = move_from_library(&mut s, 0, "Think Twice", ZoneType::Hand);
    for (flashback, price, destination) in
        [(false, 2, ZoneType::Graveyard), (true, 3, ZoneType::Exile)]
    {
        s.set_player_mana_pool(0, if flashback { "2U" } else { "1U" });
        s.game_mut().scenario_refresh_priority().unwrap();
        let object = now(&s, card);
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { object: legal, .. } if *legal == object),
        );
        let spell = top_spell(&s);
        assert_eq!(spell.flashback, flashback);
        assert_eq!(spell.spent.mana().len(), price);
        let size = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
            size + 1
        );
        assert_eq!(zone_of(&s, card), Some(destination));
    }
}

#[test]
fn reassembling_skeleton_returns_only_its_activated_appearance_tapped() {
    use crate::support::{game::move_from_library, play::activate_card};
    let mut s = five_color_setup(&["Reassembling Skeleton"]);
    s.set_player_mana_pool(0, "1B");
    let skeleton = move_from_library(&mut s, 0, "Reassembling Skeleton", ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    activate_card(&mut s, skeleton, 0);
    let mut blink = s.clone();
    assert_eq!(blink.game().state.stack_objects.len(), 1);
    let exiled = move_card(&mut blink, skeleton, ZoneType::Exile);
    let mut exiled_only = blink.clone();
    settle(&mut exiled_only);
    assert_eq!(
        exiled_only.game().state.zones.zone_of(exiled),
        Some(ZoneType::Exile)
    );
    move_card(&mut blink, exiled, ZoneType::Graveyard);
    settle(&mut blink);
    assert!(permanent_of(&blink, skeleton).is_none());
    settle(&mut s);
    let permanent = permanent_of(&s, skeleton).unwrap();
    assert!(
        s.game().state.permanents[permanent]
            .as_ref()
            .unwrap()
            .tapped
    );
}

#[test]
fn suspicious_shambler_exiles_as_cost_and_creates_two_zombies() {
    use crate::support::{game::move_from_library, inspect::zone_of, play::activate_card};
    use managym::agent::action::Action;
    let mut s = five_color_setup(&["Suspicious Shambler"]);
    let card = move_from_library(&mut s, 0, "Suspicious Shambler", ZoneType::Graveyard);
    s.advance_to_active_step(0, StepKind::Upkeep);
    s.set_player_mana_pool(0, "4BB");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateCardAbility { object, .. } if *object == card)));
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "4BB");
    s.game_mut().scenario_refresh_priority().unwrap();
    activate_card(&mut s, card, 0);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    assert!(s.battlefield_permanents_named(0, "Zombie").is_empty());
    settle(&mut s);
    let zombies = s.battlefield_permanents_named(0, "Zombie");
    assert_eq!(zombies.len(), 2);
    for id in zombies {
        let view = s.game().characteristics(id);
        assert!(view.has_subtype("Zombie"));
        assert_eq!(view.pt(), (2, 2));
    }
}

#[test]
fn granted_foreign_land_enters_under_its_players_control() {
    use crate::support::{game::move_from_library, play::choose};
    use managym::{agent::action::Action, flow::cast_permission::PlayGrant};
    let mut s = five_color_setup(&[]);
    let land = move_from_library(&mut s, 1, "Swamp", ZoneType::Exile);
    s.game_mut().state.cast_permissions.grant(PlayGrant {
        play_lands: true,
        ..PlayGrant::new(land, PlayerId(0))
    });
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::PlayLand {object, ..} if *object == land),
    );
    let permanent = s.game().state.permanents[permanent_of(&s, land).unwrap()]
        .as_ref()
        .unwrap();
    assert_eq!(permanent.controller, PlayerId(0));
    assert_eq!(s.game().state[permanent.object].owner, PlayerId(1));
}

#[test]
fn omniscience_preserves_sacrifices_and_fixes_printed_x_at_zero() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_from_library},
        inspect::{can_cast, top_spell, zone_of},
        play::{choose, select_card},
    };
    use managym::{
        agent::action::Action,
        flow::cast_permission::PlayRoute,
        state::{
            card::SpellProfile,
            game_object::Target,
            payment::{How, Toward},
        },
    };
    let mut s = five_color_unfunded_setup(&[
        "Omniscience",
        "Archon's Glory",
        "Deadly Dispute",
        "Gray Ogre",
        "Scavenger Regent",
    ]);
    let omni = s.force_permanent_on_battlefield(0, "Omniscience");
    let dispute = move_from_library(&mut s, 0, "Deadly Dispute", ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, dispute));
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let glory = move_from_library(&mut s, 0, "Archon's Glory", ZoneType::Hand);
    let regent = move_from_library(&mut s, 0, "Scavenger Regent", ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    for (spell, payer, reason) in [
        (dispute, ogre, Toward::Printed),
        (glory, omni, Toward::Bargain),
    ] {
        let mut branch = s.clone();
        choose(
            &mut branch,
            |a| matches!(a, Action::CastSpell {object, route: Some(PlayRoute::FreeFromHand), ..} if *object == spell),
        );
        if spell == glory {
            choose(&mut branch, |a| matches!(a, Action::PayCost { .. }));
            assert!(branch.choose_target(Target::Permanent(ogre)));
        }
        let paid = card_of(&branch, payer);
        select_card(&mut branch, paid);
        let stack = top_spell(&branch);
        assert_eq!(stack.spent.total(), 0);
        assert_eq!(
            stack
                .spent
                .contributed(How::Sacrificed, reason)
                .collect::<Vec<_>>(),
            [paid]
        );
        assert_eq!(zone_of(&branch, paid), Some(ZoneType::Graveyard));
    }
    s.set_player_mana_pool(0, "5BB");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell {object, route: Some(PlayRoute::FreeFromHand), profile: SpellProfile::Omen, ..} if *object == regent),
    );
    let spell = top_spell(&s);
    assert_eq!(spell.x, 0);
    assert_eq!(spell.spent.total(), 0);
}

#[test]
fn sphinx_grants_its_target_flashback_alongside_the_printed_cost() {
    use crate::support::{
        game::{clear_summoning_sickness, move_from_library},
        inspect::{top_spell, zone_of},
        play::{choose, fork_choosing, pass_until},
    };
    use managym::{
        agent::{
            action::{Action, ActionSpaceKind},
            observation::Observation,
        },
        flow::cast_permission::PlayRoute,
        state::game_object::Target,
    };
    let mut s = five_color_setup(&["Sphinx of Forgotten Lore", "Think Twice"]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let sphinx = s.force_permanent_on_battlefield(0, "Sphinx of Forgotten Lore");
    clear_summoning_sickness(&mut s, sphinx);
    let object = move_from_library(&mut s, 0, "Think Twice", ZoneType::Graveyard);
    let other = move_from_library(&mut s, 0, "Think Twice", ZoneType::Graveyard);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, sphinx);
    assert!(s.choose_target(Target::GraveyardCard(object)));
    let source = card_of(&s, sphinx);
    move_card(&mut s, source, ZoneType::Graveyard);
    settle(&mut s);
    s.set_player_mana_pool(0, "2U");
    s.game_mut().scenario_refresh_priority().unwrap();
    let observer = Observation::for_player(s.game(), PlayerId(0));
    let granted = observer
        .agent_cards
        .iter()
        .find(|card| !card.cast_permissions.is_empty())
        .unwrap();
    assert!(granted.cast_permissions[0].flashback);
    assert!(granted.cast_permissions[0].this_turn);
    assert_eq!(
        s.action_space()
            .actions
            .iter()
            .filter(|a| matches!(a, Action::CastSpell {object: id, ..} if *id==other))
            .count(),
        1
    );
    for (route, price) in [(PlayRoute::PrintedFlashback, 3), (PlayRoute::Grant(0), 2)] {
        let mut branch = fork_choosing(
            &s,
            |a| matches!(a,Action::CastSpell{object:id,route:Some(chosen),..} if *id==object && route==*chosen),
        );
        if branch.action_space().kind == ActionSpaceKind::PayOrNot {
            choose(&mut branch, |a| matches!(a, Action::PayCost { .. }));
        }
        let spell = top_spell(&branch);
        assert_eq!(spell.spent.total(), price);
        assert!(spell.flashback);
        settle(&mut branch);
        assert_eq!(zone_of(&branch, object), Some(ZoneType::Exile));
    }
}

/// A resolving effect supplies the timing of its casts (CR 608.2g), so Etali's
/// free combat cast of Harbinger owes no flash surcharge even with {2} open.
#[test]
fn harbinger_cast_by_a_resolving_effect_owes_no_flash_surcharge() {
    use crate::support::{
        game::{five_color_unfunded_setup, pin_top},
        inspect::{now, top_spell},
        play::{attack_with, choose, pass_until},
    };
    use managym::agent::action::{Action, ActionSpaceKind};
    let mut s = five_color_unfunded_setup(&["Etali, Primal Storm", "Harbinger of the Tides"]);
    let etali = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Etali, Primal Storm", true)
        .unwrap();
    let harbinger = pin_top(&mut s, &["Harbinger of the Tides"])[0];
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, etali);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    s.set_player_mana_pool(0, "2");
    let harbinger = now(&s, harbinger);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { object, .. } if *object == harbinger),
    );
    assert_ne!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert_eq!(top_spell(&s).object, now(&s, harbinger));
    assert_eq!(top_spell(&s).spent.total(), 0);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 2);
}

#[test]
fn harbinger_requires_its_surcharge_only_for_its_own_flash_permission() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_from_library},
        inspect::{can_cast, top_spell, zone_of},
        play::{cast_profile, choose, fork_choosing, pass_until, select_card},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        flow::cast_permission::PlayRoute,
        state::{card::SpellProfile, game_object::Target},
    };
    let mut root = five_color_unfunded_setup(&[
        "Harbinger of the Tides",
        "Gray Ogre",
        "Omniscience",
        "High Fae Trickster",
    ]);
    root.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    root.advance_to_active_step(0, StepKind::Upkeep);
    let ogre = root.force_permanent_on_battlefield(1, "Gray Ogre");
    let other = root.force_permanent_on_battlefield(1, "Gray Ogre");
    root.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .tapped = true;
    let target = card_of(&root, ogre);
    let card = move_from_library(&mut root, 0, "Harbinger of the Tides", ZoneType::Hand);
    root.set_player_mana_pool(0, "UU");
    root.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&root, card));
    root.set_player_mana_pool(0, "2UU");
    let mut paid = root.clone();
    cast_profile(&mut paid, card, SpellProfile::Front);
    assert_eq!(top_spell(&paid).spent.total(), 4);
    pass_until(&mut paid, ActionSpaceKind::ChooseTarget);
    assert!(!paid.choose_target(Target::Permanent(other)));
    assert!(paid.choose_target(Target::Permanent(ogre)));
    crate::support::play::resolve_top(&mut paid);
    let mut declined = paid.clone();
    choose(&mut declined, |a| matches!(a, Action::Decline { .. }));
    settle(&mut declined);
    assert_eq!(zone_of(&declined, target), Some(ZoneType::Battlefield));
    select_card(&mut paid, target);
    settle(&mut paid);
    assert_eq!(zone_of(&paid, target), Some(ZoneType::Hand));

    root.force_permanent_on_battlefield(0, "Omniscience");
    root.set_player_mana_pool(0, "2");
    root.game_mut().scenario_refresh_priority().unwrap();
    let free = |a: &Action| matches!(a,Action::CastSpell{object,route:Some(PlayRoute::FreeFromHand),..} if *object==card);
    let forced = fork_choosing(&root, free);
    assert_eq!(top_spell(&forced).spent.total(), 2);

    // With another flash permission the player can choose either price.
    root.force_permanent_on_battlefield(0, "High Fae Trickster");
    root.game_mut().scenario_refresh_priority().unwrap();
    let optional = fork_choosing(&root, free);
    for (pay, price) in [(true, 2), (false, 0)] {
        let branch = fork_choosing(&optional, |a| {
            if pay {
                matches!(a, Action::PayCost { .. })
            } else {
                matches!(a, Action::Decline { .. })
            }
        });
        assert_eq!(top_spell(&branch).spent.total(), price);
        assert!(!top_spell(&branch).kicked);
    }
}

/// Tinybones exiles each card an opponent discards with a stash counter.
/// On its controller's turn they may play those cards they don't own: a
/// foreign land enters under their control, either printed spell of an
/// Adventure is offered, and mana of any type pays for it. Every viewer sees
/// the counter. The permission is Tinybones's, not the exile's: it ends when
/// Tinybones leaves, does not reach the opponent's turn even with flash from
/// High Fae Trickster, and yields to Drannith Magistrate for spells only.
#[test]
fn tinybones_stashes_discards_and_plays_them_on_its_turn() {
    use crate::support::{
        game::move_from_library,
        inspect::{now, zone_of},
        play::{activate, cast_profile, choose, resolve_top},
    };
    use managym::{
        agent::{
            action::Action,
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig, FACT_DIM},
            observation_layout::fact,
        },
        cards::tokens::name_value,
        state::{card::SpellProfile, game_object::ObjectId, permanent::CounterKind},
    };
    let mut s = five_color_setup(&[
        "Tinybones, Bauble Burglar",
        "Smaug, the Great Calamity",
        "Gray Ogre",
        "High Fae Trickster",
        "Drannith Magistrate",
    ]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let tinybones = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Tinybones, Bauble Burglar", true)
        .unwrap();
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let smaug = move_from_library(&mut s, 1, "Smaug, the Great Calamity", ZoneType::Hand);
    let forest = move_from_library(&mut s, 1, "Forest", ZoneType::Hand);
    move_from_library(&mut s, 1, "Gray Ogre", ZoneType::Hand);
    for card in [smaug, forest] {
        s.game_mut().state.permanents[tinybones]
            .as_mut()
            .unwrap()
            .tapped = false;
        activate(&mut s, tinybones, 0);
        resolve_top(&mut s);
        let object = now(&s, card);
        choose(
            &mut s,
            |a| matches!(a, Action::SelectCard { object: chosen, .. } if *chosen == object),
        );
        settle(&mut s);
        assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
        let exiled = now(&s, card);
        assert_eq!(
            s.game().state.object_counters[&exiled].get(CounterKind::Stash),
            1
        );
    }
    let stash = name_value("CounterKind::stash") as u32;
    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        let stashed: Vec<_> = obs
            .agent_cards
            .iter()
            .chain(&obs.opponent_cards)
            .filter(|card| card.counters.get("stash") == Some(&1))
            .map(|card| card.characteristics.name.as_str())
            .collect();
        assert_eq!(
            stashed,
            ["Smaug, the Great Calamity", "Forest"],
            "{viewer:?}"
        );
        let input = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
        let rows = input
            .facts
            .chunks_exact(FACT_DIM)
            .filter(|row| ((row[fact::ID_HIGH] as u32) << 16 | row[fact::ID_LOW] as u32) == stash)
            .count();
        assert_eq!(rows, 2, "a stash fact row per card for {viewer:?}");
    }

    let offers = |s: &mut crate::support::game::Scenario, card: ObjectId| -> Vec<Action> {
        s.game_mut().scenario_refresh_priority().unwrap();
        let object = now(s, card);
        s.action_space()
            .actions
            .iter()
            .filter(|a| {
                matches!(a, Action::CastSpell { object: o, .. } | Action::PlayLand { object: o, .. } if o == &object)
            })
            .cloned()
            .collect()
    };
    let profiles = |actions: &[Action]| -> Vec<SpellProfile> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::CastSpell { profile, .. } => Some(*profile),
                _ => None,
            })
            .collect()
    };
    assert_eq!(
        profiles(&offers(&mut s, smaug)),
        [SpellProfile::Front, SpellProfile::Adventure]
    );
    assert!(matches!(
        offers(&mut s, forest)[..],
        [Action::PlayLand { .. }]
    ));

    // Drannith Magistrate refuses the spells but not the land play.
    let mut hand_only = s.clone();
    hand_only.force_permanent_on_battlefield(1, "Drannith Magistrate");
    assert!(offers(&mut hand_only, smaug).is_empty());
    assert_eq!(offers(&mut hand_only, forest).len(), 1);

    // Without Tinybones, nothing; the counters stay.
    let mut gone = s.clone();
    let source = gone.game().state.permanents[tinybones]
        .as_ref()
        .unwrap()
        .object;
    move_card(&mut gone, source, ZoneType::Graveyard);
    assert!(offers(&mut gone, smaug).is_empty());
    assert!(offers(&mut gone, forest).is_empty());
    assert_eq!(gone.game().state.object_counters.len(), 2);

    // Not during the opponent's turn, even as though it had flash.
    let mut theirs = s.clone();
    theirs.force_permanent_on_battlefield(0, "High Fae Trickster");
    theirs.advance_to_active_step(1, StepKind::Upkeep);
    theirs.pass_priority();
    assert_eq!(theirs.action_space().player, Some(PlayerId(0)));
    assert!(offers(&mut theirs, smaug).is_empty());

    // Mana of any type pays the Adventure: five blue for {4}{R}.
    s.set_player_mana_pool(0, "UUUUU");
    cast_profile(&mut s, smaug, SpellProfile::Adventure);
    assert!(s.choose_target(managym::state::game_object::Target::Permanent(ogre)));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert!(
        s.game().state.permanents[ogre].is_none(),
        "Spew Flame resolved"
    );
    assert_eq!(zone_of(&s, smaug), Some(ZoneType::Exile));
    assert!(
        !s.game().state.object_counters.contains_key(&now(&s, smaug)),
        "the stash counter left with the stack appearance"
    );

    // The foreign land enters under Tinybones's controller.
    let exiled = now(&s, forest);
    choose(
        &mut s,
        |a| matches!(a, Action::PlayLand { object, .. } if *object == exiled),
    );
    settle(&mut s);
    let land = s.game().state.permanent_of(now(&s, forest)).unwrap();
    let land = s.game().state.permanents[land].as_ref().unwrap();
    assert_eq!(
        (land.controller, s.game().state[land.object].owner),
        (PlayerId(0), PlayerId(1))
    );
}

/// The stash permission belongs to whichever Tinybones its player controls
/// now, not to the one that exiled the card: a later Tinybones plays what an
/// earlier one stashed. An exact grant on the same card adds its own route,
/// and each (face, route) is offered once.
#[test]
fn a_later_tinybones_plays_an_earlier_stash_beside_an_exact_grant() {
    use crate::support::{
        game::move_from_library,
        inspect::{now, zone_of},
        play::{activate, choose, resolve_top},
    };
    use managym::{
        agent::{
            action::Action,
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig, ACTION_DIM},
            observation_layout::action as action_columns,
        },
        flow::cast_permission::{play_route_code, PlayGrant, PlayRoute},
        state::permanent::CounterKind,
    };
    let mut s = five_color_setup(&["Tinybones, Bauble Burglar"]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let tinybones = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Tinybones, Bauble Burglar", true)
        .unwrap();
    let forest = move_from_library(&mut s, 1, "Forest", ZoneType::Hand);
    move_from_library(&mut s, 1, "Island", ZoneType::Hand);
    activate(&mut s, tinybones, 0);
    resolve_top(&mut s);
    let object = now(&s, forest);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { object: chosen, .. } if *chosen == object),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, forest), Some(ZoneType::Exile));
    let routes = |s: &mut crate::support::game::Scenario| -> Vec<Option<PlayRoute>> {
        s.game_mut().scenario_refresh_priority().unwrap();
        let object = now(s, forest);
        s.action_space()
            .actions
            .iter()
            .filter_map(|a| match a {
                Action::PlayLand {
                    object: o, route, ..
                } if *o == object => Some(*route),
                _ => None,
            })
            .collect()
    };

    let source = s.game().state.permanents[tinybones]
        .as_ref()
        .unwrap()
        .object;
    move_card(&mut s, source, ZoneType::Graveyard);
    assert!(routes(&mut s).is_empty());
    s.game_mut()
        .scenario_force_battlefield(PlayerId(0), "Tinybones, Bauble Burglar", true)
        .unwrap();
    assert_eq!(routes(&mut s), [None], "the later Tinybones's lone route");

    let exiled = now(&s, forest);
    // A cast-only grant never plays a land (CR 305.9): only the stash route.
    s.game_mut()
        .state
        .cast_permissions
        .grant(PlayGrant::new(exiled, PlayerId(0)));
    assert_eq!(
        routes(&mut s),
        [None],
        "a cast-only grant adds no land play"
    );
    s.game_mut().state.cast_permissions.grant(PlayGrant {
        play_lands: true,
        ..PlayGrant::new(exiled, PlayerId(0))
    });
    let offered = routes(&mut s);
    assert_eq!(offered.len(), 2, "{offered:?}");
    assert!(matches!(offered[0], Some(PlayRoute::Grant(_))));
    assert_eq!(
        offered[1],
        Some(PlayRoute::ExileCounter(CounterKind::Stash))
    );
    let exile_offers = s
        .action_space()
        .actions
        .iter()
        .filter(|a| matches!(a, Action::PlayLand { object, .. } if *object == exiled))
        .count();
    assert_eq!(exile_offers, 2, "one card, listed once per route");

    // Both routes are free land plays; only `play_route` tells them apart
    // in the encoded action rows.
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let indices: Vec<usize> = s
        .action_space()
        .actions
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, Action::PlayLand { object, .. } if *object == exiled))
        .map(|(index, _)| index)
        .collect();
    let codes: Vec<_> = indices
        .iter()
        .map(|&index| observation.action_space.actions[index].play_route)
        .collect();
    assert_eq!(
        codes,
        [
            Some(play_route_code(offered[0].unwrap())),
            Some(play_route_code(offered[1].unwrap()))
        ]
    );
    assert_ne!(codes[0], codes[1]);
    let input = encode(&observation, &ObservationEncoderConfig::default()).unwrap();
    let rows: Vec<&[f32]> = indices
        .iter()
        .map(|&index| &input.actions[index * ACTION_DIM..(index + 1) * ACTION_DIM])
        .collect();
    assert_ne!(rows[0], rows[1]);
    let differing: Vec<usize> = (0..ACTION_DIM)
        .filter(|&column| rows[0][column] != rows[1][column])
        .collect();
    assert_eq!(differing, [action_columns::PLAY_ROUTE]);
}

/// Loot gives its controller, and only them, a second land play each turn.
/// Its activation looks at six cards and offers only creature cards with
/// mana value at most the lands its controller has as it resolves; the
/// chosen creature enters and the other five go to the bottom.
#[test]
fn loot_grants_its_controller_a_land_play_and_finds_a_creature_within_its_lands() {
    use crate::support::{
        game::{move_to_hand, pin_top},
        inspect::{has_action, now, offered_cards, zone_of},
        play::{play_land, resolve_top, select_card},
    };
    use managym::{agent::action::Action, state::player::ManaPayment};
    let mut s = five_color_setup(&[
        "Loot, Exuberant Explorer",
        "Gray Ogre",
        "Tinybones, Bauble Burglar",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let mut theirs = s.clone();
    theirs
        .game_mut()
        .scenario_force_battlefield(PlayerId(1), "Loot, Exuberant Explorer", true)
        .unwrap();
    play_land(&mut theirs, "Forest");
    move_to_hand(&mut theirs, "Island");
    theirs.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        !has_action(&theirs, |a| matches!(a, Action::PlayLand { .. })),
        "an opponent's Loot"
    );

    let loot = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Loot, Exuberant Explorer", true)
        .unwrap();
    play_land(&mut s, "Forest");
    play_land(&mut s, "Island");
    move_to_hand(&mut s, "Plains");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        !has_action(&s, |a| matches!(a, Action::PlayLand { .. })),
        "two land plays only"
    );

    let top = pin_top(
        &mut s,
        &[
            "Gray Ogre",
            "Tinybones, Bauble Burglar",
            "Forest",
            "Island",
            "Swamp",
            "Plains",
        ],
    );
    let names = |s: &crate::support::game::Scenario,
                 cards: Vec<managym::state::game_object::ObjectId>|
     -> Vec<String> {
        let mut names: Vec<_> = cards
            .into_iter()
            .map(|id| s.game().state[id].name.clone())
            .collect();
        names.sort();
        names
    };
    let mut three_lands = s.clone();
    three_lands.force_permanent_on_battlefield(0, "Mountain");
    activate(&mut three_lands, loot, 0);
    resolve_top(&mut three_lands);
    assert_eq!(
        names(&three_lands, offered_cards(&three_lands)),
        ["Gray Ogre", "Tinybones, Bauble Burglar"]
    );

    activate(&mut s, loot, 0);
    resolve_top(&mut s);
    assert_eq!(
        names(&s, offered_cards(&s)),
        ["Tinybones, Bauble Burglar"],
        "two lands: the three-mana Ogre is out of reach"
    );
    let found = now(&s, top[1]);
    select_card(&mut s, found);
    settle(&mut s);
    assert_eq!(zone_of(&s, top[1]), Some(ZoneType::Battlefield));
    let library: Vec<_> = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(0))
        .iter()
        .copied()
        .collect();
    for card in [top[0], top[2], top[3], top[4], top[5]] {
        let object = now(&s, card);
        assert_eq!(zone_of(&s, card), Some(ZoneType::Library));
        let position = library.iter().position(|id| *id == object).unwrap();
        assert!(position < 5, "{position}: on the bottom");
    }
}

/// Redcap Gutter-Dweller enters with two Rats that can't block. At its
/// controller's upkeep they may sacrifice another creature; if they do, it
/// gets a +1/+1 counter and the top card of their library is exiled,
/// playable this turn. Declining changes nothing.
#[test]
fn redcap_makes_nonblocking_rats_and_trades_one_for_a_counter_and_a_card() {
    use crate::support::{
        inspect::{has_action, permanent_named, zone_of},
        play::{choose, pass_until},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::ObjectId, permanent::CounterKind},
    };
    let mut s = five_color_setup(&["Redcap Gutter-Dweller", "Gray Ogre"]);
    enter_and_settle(&mut s, "Redcap Gutter-Dweller");
    let rats = s.battlefield_permanents_named(0, "Rat");
    assert_eq!(rats.len(), 2);
    let redcap = permanent_named(&s, "Redcap Gutter-Dweller");

    // The Rats can't block; Redcap can.
    let mut blocking = s.clone();
    let attacker = blocking.force_permanent_on_battlefield(1, "Gray Ogre");
    blocking.advance_to_active_step(1, StepKind::Main);
    crate::support::game::clear_summoning_sickness(&mut blocking, attacker);
    pass_until(&mut blocking, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut blocking, attacker);
    pass_until(&mut blocking, ActionSpaceKind::DeclareBlocker);
    // Each creature's block is asked in turn; collect who may block the Ogre.
    let mut blockers = Vec::new();
    while blocking.action_space().kind == ActionSpaceKind::DeclareBlocker {
        blockers.extend(
            blocking
                .action_space()
                .actions
                .iter()
                .filter_map(|a| match a {
                    Action::DeclareBlocker {
                        blocker,
                        attacker: Some(_),
                        ..
                    } => Some(*blocker),
                    _ => None,
                }),
        );
        choose(&mut blocking, |a| {
            matches!(a, Action::DeclareBlocker { attacker: None, .. })
        });
    }
    assert!(blockers.contains(&redcap), "{blockers:?}");
    assert!(
        rats.iter().all(|rat| !blockers.contains(rat)),
        "{blockers:?}"
    );

    let top = |s: &crate::support::game::Scenario| -> ObjectId {
        s.game()
            .state
            .zones
            .top(ZoneType::Library, PlayerId(0))
            .unwrap()
    };
    s.advance_to_active_step(0, StepKind::Upkeep);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let mut declined = s.clone();
    let card = top(&s);
    assert!(has_action(&declined, |a| matches!(
        a,
        Action::Decline { .. }
    )));
    choose(&mut declined, |a| matches!(a, Action::Decline { .. }));
    settle(&mut declined);
    assert_eq!(declined.battlefield_permanents_named(0, "Rat").len(), 2);
    assert_eq!(zone_of(&declined, card), Some(ZoneType::Library));

    let rat = s.game().state.permanents[rats[0]].as_ref().unwrap().object;
    assert!(
        !has_action(
            &s,
            |a| matches!(a, Action::SelectCard { object, .. } if s.game().state.permanent_of(*object) == Some(redcap))
        ),
        "another creature"
    );
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { object, .. } if *object == rat),
    );
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Rat").len(), 1);
    assert_eq!(
        s.game().state.permanents[redcap]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::PlusOnePlusOne),
        1
    );
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    let exiled = s
        .game()
        .state
        .zones
        .top(ZoneType::Exile, PlayerId(0))
        .unwrap();
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        has_action(
            &s,
            |a| matches!(a, Action::CastSpell { object, .. } | Action::PlayLand { object, .. } if *object == exiled)
        ),
        "{} is playable this turn",
        s.game().state[exiled].name
    );
}

/// Quilled Greatwurm's own graveyard permission removes six counters, of
/// any kinds, from among its caster's creatures, asking for the kind only
/// when a creature holds several. Muldrotha's creature permission and an
/// exact grant overlap it on the same card; neither removes counters, and
/// only Muldrotha's spends its quota. Too few counters closes only the
/// Greatwurm's own route.
#[test]
fn quilled_greatwurm_casts_from_the_graveyard_by_removing_six_counters() {
    use crate::support::{
        game::{five_color_unfunded_setup, move_from_library},
        inspect::{has_action, zone_of},
        play::{choose, fork_choosing},
    };
    use managym::{
        agent::{action::Action, observation::Observation},
        flow::cast_permission::{play_route_code, PlayGrant, PlayRoute},
        state::permanent::CounterKind,
    };
    let mut s =
        five_color_unfunded_setup(&["Quilled Greatwurm", "Muldrotha, the Gravetide", "Gray Ogre"]);
    let muldrotha = s.force_permanent_on_battlefield(0, "Muldrotha, the Gravetide");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let wurm = move_from_library(&mut s, 0, "Quilled Greatwurm", ZoneType::Graveyard);
    let add = |s: &mut crate::support::game::Scenario,
               id: managym::state::game_object::PermanentId,
               kind: CounterKind,
               n: i32| {
        s.game_mut().state.permanents[id]
            .as_mut()
            .unwrap()
            .counters
            .change(kind, n);
    };
    add(&mut s, muldrotha, CounterKind::PlusOnePlusOne, 4);
    add(&mut s, muldrotha, CounterKind::Stun, 1);
    add(&mut s, ogre, CounterKind::PlusOnePlusOne, 1);
    s.game_mut()
        .state
        .cast_permissions
        .grant(PlayGrant::new(wurm, PlayerId(0)));
    s.set_player_mana_pool(0, "4GG");
    s.game_mut().scenario_refresh_priority().unwrap();

    let routes = |s: &crate::support::game::Scenario| -> Vec<PlayRoute> {
        s.action_space()
            .actions
            .iter()
            .filter_map(|a| match a {
                Action::CastSpell {
                    object,
                    route: Some(route),
                    ..
                } if *object == wurm => Some(*route),
                _ => None,
            })
            .collect()
    };
    let offered = routes(&s);
    assert_eq!(offered.len(), 3, "{offered:?}");
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let mut codes: Vec<u8> = obs
        .action_space
        .actions
        .iter()
        .zip(&s.action_space().actions)
        .filter(|(_, a)| matches!(a, Action::CastSpell { object, .. } if *object == wurm))
        .filter_map(|(option, _)| option.play_route)
        .collect();
    codes.sort_unstable();
    assert_eq!(
        codes,
        vec![
            play_route_code(PlayRoute::Grant(0)),
            play_route_code(PlayRoute::GraveyardType {
                source: s.game().state.permanents[muldrotha]
                    .as_ref()
                    .unwrap()
                    .object,
                permanent_id: s.game().state.permanents[muldrotha].as_ref().unwrap().id,
                kind: managym::state::card::CardType::Creature,
            }),
            play_route_code(PlayRoute::PrintedGraveyard),
        ]
    );
    assert!(obs
        .action_space
        .actions
        .iter()
        .any(|option| option.description.as_deref()
            == Some("Cast Quilled Greatwurm from graveyard by removing six counters")
            || option.description.as_deref()
                == Some("Cast Quilled Greatwurm from graveyard by removing 6 counters")));

    let counters = |s: &crate::support::game::Scenario,
                    id: managym::state::game_object::PermanentId|
     -> Vec<(CounterKind, i32)> {
        s.game().state.permanents[id]
            .as_ref()
            .unwrap()
            .counters
            .iter()
            .collect()
    };
    let cast_by = |route: fn(&PlayRoute) -> bool| {
        fork_choosing(
            &s,
            |a| matches!(a, Action::CastSpell { object, route: Some(r), .. } if *object == wurm && route(r)),
        )
    };

    // The Greatwurm's own route: one counter at a time, the kind asked of
    // the creature holding two kinds.
    let mut own = cast_by(|r| *r == PlayRoute::PrintedGraveyard);
    // Progress is the count still owed, never a chosen mode.
    let owed = |s: &crate::support::game::Scenario, n: &str| {
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert_eq!(context.chosen_mode.0, None);
        assert!(
            context.pending.iter().any(|line| line == n),
            "{:?}",
            context.pending
        );
    };
    owed(&own, "6 counters left to remove.");
    let ogre_object = own.game().state.permanents[ogre].as_ref().unwrap().object;
    let muldrotha_object = own.game().state.permanents[muldrotha]
        .as_ref()
        .unwrap()
        .object;
    choose(
        &mut own,
        |a| matches!(a, Action::SelectCard { object, .. } if *object == ogre_object),
    );
    assert!(counters(&own, ogre).is_empty());
    owed(&own, "5 counters left to remove.");
    choose(
        &mut own,
        |a| matches!(a, Action::SelectCard { object, .. } if *object == muldrotha_object),
    );
    let modes: Vec<usize> = own
        .action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::ChooseMode { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    assert_eq!(
        modes,
        vec![
            CounterKind::PlusOnePlusOne as usize,
            CounterKind::Stun as usize
        ]
    );
    let obs = Observation::for_player(own.game(), PlayerId(0));
    assert!(obs
        .action_space
        .actions
        .iter()
        .any(|option| option.description.as_deref()
            == Some("Remove a stun counter from Muldrotha, the Gravetide")));
    choose(
        &mut own,
        |a| matches!(a, Action::ChooseMode { mode, .. } if *mode == CounterKind::Stun as usize),
    );
    for _ in 0..4 {
        choose(
            &mut own,
            |a| matches!(a, Action::SelectCard { object, .. } if *object == muldrotha_object),
        );
    }
    settle(&mut own);
    assert!(counters(&own, muldrotha).is_empty());
    assert!(own.game().state.cast_permissions.spent_types.is_empty());
    assert_eq!(
        own.battlefield_permanents_named(0, "Quilled Greatwurm")
            .len(),
        1
    );

    // Muldrotha's permission and the grant remove nothing; only Muldrotha's
    // spends the creature quota.
    for (muldrothas, spent) in [(true, 1), (false, 0)] {
        let mut other = if muldrothas {
            cast_by(|r| matches!(r, PlayRoute::GraveyardType { .. }))
        } else {
            cast_by(|r| matches!(r, PlayRoute::Grant(_)))
        };
        settle(&mut other);
        assert_eq!(counters(&other, muldrotha).len(), 2);
        assert_eq!(counters(&other, ogre).len(), 1);
        assert_eq!(other.game().state.cast_permissions.spent_types.len(), spent);
        assert_eq!(
            other
                .battlefield_permanents_named(0, "Quilled Greatwurm")
                .len(),
            1
        );
    }

    // Five counters close only the Greatwurm's own route.
    add(&mut s, ogre, CounterKind::PlusOnePlusOne, -1);
    s.game_mut().scenario_refresh_priority().unwrap();
    let left = routes(&s);
    assert_eq!(left.len(), 2, "{left:?}");
    assert!(!left.contains(&PlayRoute::PrintedGraveyard));
    assert!(has_action(
        &s,
        |a| matches!(a, Action::CastSpell { object, .. } if *object == wurm)
    ));
    assert_eq!(zone_of(&s, wurm), Some(ZoneType::Graveyard));
}

/// "Whenever a creature you control deals combat damage during your turn,
/// put that many +1/+1 counters on it": each creature triggers once with
/// its own total, a trample split across a blocker and the player included.
/// A creature that dies in combat gets nothing, and blocking on the
/// opponent's turn triggers nothing.
#[test]
fn quilled_greatwurm_counts_each_creatures_combat_damage_on_your_turn() {
    use crate::support::play::{choose, pass_until};
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::PermanentId, permanent::CounterKind},
    };
    let mut s = five_color_setup(&["Quilled Greatwurm", "Gray Ogre", "Muldrotha, the Gravetide"]);
    let force = |s: &mut crate::support::game::Scenario, player: usize, name: &str| {
        s.game_mut()
            .scenario_force_battlefield(PlayerId(player as _), name, true)
            .unwrap()
    };
    let wurm = force(&mut s, 0, "Quilled Greatwurm");
    let unblocked = force(&mut s, 0, "Gray Ogre");
    let doomed = force(&mut s, 0, "Gray Ogre");
    let chump = force(&mut s, 1, "Gray Ogre");
    let wall = force(&mut s, 1, "Muldrotha, the Gravetide");
    let plus = |s: &crate::support::game::Scenario, id: PermanentId| {
        s.game().state.permanents[id]
            .as_ref()
            .map(|p| p.counter_count(CounterKind::PlusOnePlusOne))
    };
    let life = s.life(1);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |a| {
            matches!(a, Action::DeclareAttacker { attack: true, .. })
        });
    }
    pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        let blocks: &[(PermanentId, PermanentId)] = &[(chump, wurm), (wall, doomed)];
        choose(&mut s, |a| match a {
            Action::DeclareBlocker {
                blocker,
                attacker: Some(attacker),
                ..
            } => blocks.contains(&(*blocker, *attacker)),
            _ => false,
        });
    }
    s.advance_to_step(StepKind::PostcombatMain);
    assert_eq!(s.life(1), life - 5 - 2);
    assert_eq!(plus(&s, wurm), Some(7));
    assert_eq!(plus(&s, unblocked), Some(2));
    assert_eq!(plus(&s, doomed), None, "died in combat");

    // On the opponent's turn the Greatwurm's controller blocks: no trigger.
    s.advance_to_active_step(1, StepKind::Main);
    let attacker = s.game().state.permanents[wall]
        .as_ref()
        .map(|_| wall)
        .unwrap();
    crate::support::game::clear_summoning_sickness(&mut s, attacker);
    s.game_mut().state.permanents[wurm].as_mut().unwrap().tapped = false;
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, attacker);
    pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        choose(&mut s, |a| match a {
            Action::DeclareBlocker {
                blocker,
                attacker: Some(_),
                ..
            } => *blocker == wurm,
            Action::DeclareBlocker { attacker: None, .. } => true,
            _ => false,
        });
    }
    s.advance_to_step(StepKind::PostcombatMain);
    assert!(
        s.game().state.permanents[wall].is_none(),
        "the Greatwurm dealt its damage"
    );
    assert_eq!(plus(&s, wurm), Some(7));
}

/// Each stacked Greatwurm trigger shows its own creature and amount: the
/// stack entry's X, its description ("that many", "event: ..."), and the
/// continuation frame that joins the event creature in the policy input.
#[test]
fn quilled_greatwurm_triggers_name_their_creature_and_damage_on_the_stack() {
    use crate::support::play::choose;
    use managym::{
        agent::{
            action::{Action, ActionSpaceKind},
            observation::{Observation, StackObjectKindData},
        },
        state::{game_object::ObjectId, stack_object::StackObject},
    };
    let mut s = five_color_setup(&["Quilled Greatwurm", "Gray Ogre"]);
    let wurm = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Quilled Greatwurm", true)
        .unwrap();
    let ogre = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
        .unwrap();
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |a| {
            matches!(a, Action::DeclareAttacker { attack: true, .. })
        });
    }
    let mut placement = Vec::new();
    s.advance_until(
        |s| {
            if s.action_space().kind == ActionSpaceKind::OrderTriggers {
                let obs = Observation::for_player(s.game(), PlayerId(0));
                placement.extend(
                    obs.action_space
                        .actions
                        .iter()
                        .filter_map(|option| option.description.clone()),
                );
            }
            s.game().state.stack_objects.len() == 2
                && s.action_space().kind == ActionSpaceKind::Priority
        },
        "both triggers stacked".into(),
    );
    assert!(
        placement
            .iter()
            .any(|label| label.contains("put that many (7) +1/+1 counters")
                || label.contains("Put that many (7) +1/+1 counters")),
        "{placement:?}"
    );

    let object = |id: managym::state::game_object::PermanentId| {
        s.game().state.permanents[id].as_ref().unwrap().object
    };
    let (wurm_object, ogre_object) = (object(wurm), object(ogre));
    let mut stacked: Vec<(u8, ObjectId)> = s
        .game()
        .state
        .stack_objects
        .iter()
        .map(|item| match item {
            StackObject::TriggeredAbility(trigger) => match trigger.context {
                Some(managym::state::game_object::Target::Permanent(id)) => (
                    trigger.x,
                    s.game().state.permanents[id].as_ref().unwrap().object,
                ),
                other => panic!("event creature: {other:?}"),
            },
            other => panic!("only triggers: {other:?}"),
        })
        .collect();
    stacked.sort();
    assert_eq!(stacked, vec![(2, ogre_object), (7, wurm_object)]);

    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        let mut entries: Vec<(i32, String)> = obs
            .stack_objects
            .iter()
            .inspect(|item| assert_eq!(item.kind, StackObjectKindData::TriggeredAbility))
            .map(|item| (item.x, item.description.clone().unwrap()))
            .collect();
        entries.sort();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, 2);
        assert!(
            entries[0].1.contains("Put that many (2) +1/+1 counters")
                && entries[0].1.contains(" — event: Gray Ogre"),
            "{entries:?}"
        );
        assert_eq!(entries[1].0, 7);
        assert!(
            entries[1].1.contains("Put that many (7) +1/+1 counters")
                && entries[1].1.contains(" — event: Quilled Greatwurm"),
            "{entries:?}"
        );
        assert!(!entries.iter().any(|(_, text)| text.contains("announced")));

        // The policy input joins each stacked trigger to its creature.
        let mut frames: Vec<(u8, Option<i32>)> = s
            .game()
            .continuation_frames(viewer)
            .iter()
            .filter(|frame| frame.kind == 3)
            .map(|frame| (frame.x, frame.context.and_then(|c| c.object)))
            .collect();
        frames.sort();
        assert_eq!(frames.len(), 2);
        assert_eq!((frames[0].0, frames[1].0), (2, 7));
        assert!(frames.iter().all(|(_, object)| object.is_some()));
        assert_ne!(frames[0].1, frames[1].1);
    }
}

/// Tinybones lets its controller spend mana as though it were mana of any
/// type on a stashed card, and that applies to the total cost after
/// reductions (CR 601.2f before 601.2h): an opponent's stashed Glamdring's
/// Gleam of Death, {3}{U} less an equipped Glamdring's 2, is paid with {R}{R}.
#[test]
fn glamdring_cast_from_a_tinybones_stash_spends_any_mana_after_its_reduction() {
    use crate::support::{
        game::move_from_library,
        inspect::{can_cast, zone_of},
        play::{cast_profile, choose},
    };
    use managym::{
        agent::action::Action,
        state::{
            card::SpellProfile,
            game_object::Target,
            permanent::{CounterKind, Counters},
        },
    };
    let mut s = five_color_setup(&[
        "Tinybones, Bauble Burglar",
        "Glamdring, Foe-hammer",
        "Gray Ogre",
    ]);
    s.game_mut()
        .scenario_force_battlefield(PlayerId(0), "Tinybones, Bauble Burglar", true)
        .unwrap();
    let stashed = move_from_library(&mut s, 1, "Glamdring, Foe-hammer", ZoneType::Exile);
    let mut stash = Counters::default();
    stash.change(CounterKind::Stash, 1);
    s.game_mut().state.object_counters.insert(stashed, stash);
    s.set_player_mana_pool(0, "RR");
    s.game_mut().scenario_refresh_priority().unwrap();
    let gleam = |s: &crate::support::game::Scenario| {
        s.action_space().actions.iter().any(|a| matches!(
            a,
            Action::CastSpell { object, profile: SpellProfile::Adventure, .. } if *object == stashed
        ))
    };
    assert!(can_cast(&s, stashed), "Glamdring's own {{2}} with any mana");
    assert!(!gleam(&s), "{{3}}{{U}} before the reduction");

    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let equipment = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Glamdring, Foe-hammer", true)
        .unwrap();
    s.set_player_mana_pool(0, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, equipment, 0);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    s.set_player_mana_pool(0, "RR");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(gleam(&s));
    let library = s.zone_size(0, ZoneType::Library);
    cast_profile(&mut s, stashed, SpellProfile::Adventure);
    if s.action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::PayCost { .. }))
    {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(
        zone_of(&s, stashed),
        Some(ZoneType::Exile),
        "on an adventure"
    );
    assert_eq!(
        s.zone_size(0, ZoneType::Library),
        library - 6,
        "Gleam of Death milled six"
    );
}
