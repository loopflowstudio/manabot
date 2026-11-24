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
            assert_eq!(
                ability.paid.paid_as(How::Sacrificed).collect::<Vec<_>>(),
                [chosen]
            );
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
fn cat_collectors_food_creates_a_cat_only_on_its_controllers_first_gain() {
    use crate::support::play::command;
    use managym::agent::action::Action;

    let mut s = five_color_setup(&["Cat Collector"]);
    for _ in 0..2 {
        enter_and_settle(&mut s, "Cat Collector");
        let food = permanent_named(&s, "Food");
        let before = s.life(0);
        command(
            &mut s,
            |action| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == food),
        );
        spend_colorless(&mut s);
        settle(&mut s);
        assert_eq!(s.life(0), before + 3);
        assert_eq!(s.battlefield_permanents_named(0, "Cat").len(), 1);
    }
}

#[test]
fn lunar_insight_counts_distinct_controlled_nonland_values_at_resolution() {
    use crate::support::{
        game::force_new_card_in_hand,
        inspect::{hand, top_spell},
        play::cast_card,
    };
    use managym::state::player::ManaPayment;

    let mut s = five_color_setup(&[
        "Lunar Insight",
        "Felidar Cub",
        "Kellan, Planar Trailblazer",
        "Gray Ogre",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.force_permanent_on_battlefield(0, "Island");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let spell = force_new_card_in_hand(&mut s, 0, "Lunar Insight");
    cast_card(&mut s, spell);
    assert_eq!(top_spell(&s).spent.total(), 3);
    let pending = s.game().clone();

    for (foods, creatures, expected) in [
        (2, &["Felidar Cub"][..], 2),
        (1, &["Kellan, Planar Trailblazer", "Felidar Cub"][..], 3),
        (0, &[][..], 0),
    ] {
        *s.game_mut() = pending.clone();
        for _ in 0..foods {
            s.game_mut()
                .create_token("token.food", PlayerId(0), false)
                .unwrap();
        }
        for creature in creatures {
            s.force_permanent_on_battlefield(0, creature);
        }
        let before = hand(&s, 0).len();
        settle(&mut s);
        assert_eq!(hand(&s, 0).len() - before, expected);
    }
}

#[test]
fn felling_blow_reads_power_after_the_counter_and_keeps_both_targets_bound() {
    use crate::support::{
        game::force_new_card_in_hand,
        inspect::{permanent, top_spell},
        play::cast_card,
    };
    use managym::{
        flow::event::{DamageTarget, GameEvent},
        state::{game_object::Target, permanent::CounterKind, player::ManaPayment},
    };

    let mut s = five_color_setup(&["Felling Blow", "Gray Ogre", "Steel Hellkite"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let victim = s.force_permanent_on_battlefield(1, "Steel Hellkite");
    let object = card_of(&s, source);
    let spell = force_new_card_in_hand(&mut s, 0, "Felling Blow");
    cast_card(&mut s, spell);
    s.choose_target_named("Gray Ogre");
    s.choose_target_named("Steel Hellkite");
    assert_eq!(
        top_spell(&s).targets,
        vec![Target::Permanent(source), Target::Permanent(victim)]
    );
    let pending = s.game().clone();
    for (blink, counters, damage) in [
        (None, 1, 3),
        (Some(victim), 1, 0),
        (Some(source), 0, 0),
        (None, 1, 3),
    ] {
        *s.game_mut() = pending.clone();
        if let Some(id) = blink {
            let card = card_of(&s, id);
            let exiled = move_card(&mut s, card, ZoneType::Exile);
            move_card(&mut s, exiled, ZoneType::Battlefield);
        }
        settle(&mut s);
        let ogre = permanent(&s, permanent_named(&s, "Gray Ogre"));
        let hellkite = permanent(&s, s.battlefield_permanents_named(1, "Steel Hellkite")[0]);
        assert_eq!(ogre.counters.get(CounterKind::PlusOnePlusOne), counters);
        assert_eq!(hellkite.counters.get(CounterKind::PlusOnePlusOne), 0);
        assert_eq!(hellkite.damage, damage);
        assert!(!s
            .game()
            .state
            .observation_events
            .iter()
            .any(|event| matches!(event, GameEvent::CombatDamageDealt { .. })));
        let mut hits = s
            .game()
            .state
            .events
            .iter()
            .skip(pending.state.events.len())
            .filter_map(|event| match event {
                GameEvent::DamageDealt {
                    source,
                    target,
                    amount,
                    ..
                } => Some((*source, target, *amount)),
                _ => None,
            });
        if damage > 0 {
            assert_eq!(
                hits.next(),
                Some((Some(object), &DamageTarget::Permanent(victim), 3))
            );
        }
        assert_eq!(hits.next(), None);
    }
}

/// Cast a constructed one-mana instant targeting the top pending ability.
fn counter_top_ability(s: &mut crate::support::game::Scenario) {
    let target = s.game().state.stack_objects.last().unwrap().id();
    counter_ability(s, target);
}

fn counter_ability(
    s: &mut crate::support::game::Scenario,
    target: managym::state::game_object::PublicId,
) {
    use crate::support::{game::force_new_card_in_hand, play::cast_card};
    use managym::state::{
        ability::{Effect, ObjectSelector, TargetRequirement, TargetSpec},
        game_object::Target,
    };

    let instant = force_new_card_in_hand(s, 0, "Lightning Bolt");
    s.game_mut().state[instant].targeting = vec![TargetRequirement::one(TargetSpec::Ability)];
    s.game_mut().state[instant].spell_effects = vec![Effect::CounterAbility {
        what: ObjectSelector::Target(TargetSpec::Ability),
    }];
    s.set_player_mana_pool(0, "R");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_card(s, instant);
    assert!(s.choose_target(Target::StackAbility(target)));
}

#[test]
fn drake_hatcher_binds_each_hit_and_spends_incubation_before_resolution() {
    use crate::support::{
        game::set_counters,
        inspect::{counters_of, has_action},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::permanent::CounterKind::{Incubation, PlusOnePlusOne},
    };

    let mut s = five_color_setup(&["Drake Hatcher", "Lightning Bolt"]);
    let first = enter_and_settle(&mut s, "Drake Hatcher");
    let second = enter_and_settle(&mut s, "Drake Hatcher");
    let first = permanent_of(&s, first).unwrap();
    let second = permanent_of(&s, second).unwrap();
    set_counters(&mut s, second, PlusOnePlusOne, 2);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        s.declare_attack();
    }
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 2,
        "two combat damage triggers".into(),
    );
    set_counters(&mut s, second, PlusOnePlusOne, 0);
    let pending = s.game().clone();
    for _ in 0..2 {
        *s.game_mut() = pending.clone();
        for viewer in [PlayerId(0), PlayerId(1)] {
            use managym::agent::{
                observation::Observation,
                observation_encoder::{encode, FactKind, ObservationEncoderConfig},
                observation_layout::fact,
            };
            let observation = Observation::for_player(s.game(), viewer);
            let encoding = encode(&observation, &ObservationEncoderConfig::default()).unwrap();
            let mut hits = Vec::new();
            for (index, frame) in observation
                .continuation
                .iter()
                .enumerate()
                .filter(|(_, frame)| frame.kind == 3)
            {
                let source = frame.source.object.unwrap();
                let row = encoding
                    .facts
                    .chunks_exact(fact::WIDTH)
                    .find(|row| {
                        row[fact::KIND] == FactKind::TriggeringAmount as u8 as f32
                            && row[fact::ID_LOW] == (index + 1) as f32
                    })
                    .unwrap();
                assert_eq!(row[fact::VALUE], frame.triggering_amount as f32);
                hits.push((source, frame.triggering_amount));
            }
            let mut expected = [(first, 1), (second, 3)].map(|(id, amount)| {
                let permanent = s.game().state.permanents[id].as_ref().unwrap();
                (permanent.id.0 as i32, amount)
            });
            hits.sort_unstable();
            expected.sort_unstable();
            assert_eq!(hits, expected);
        }
        settle(&mut s);
        assert_eq!(counters_of(&s, first, Incubation), 1);
        assert_eq!(counters_of(&s, second, Incubation), 3);
    }
    let offered = |id| move |action: &Action| matches!(action, Action::ActivateAbility { permanent, ability_index: 0, .. } if *permanent == id);
    set_counters(&mut s, first, Incubation, 2);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!has_action(&s, offered(first)));
    assert!(has_action(&s, offered(second)));
    activate(&mut s, second, 0);
    assert_eq!(counters_of(&s, second, Incubation), 0);
    assert!(s.battlefield_permanents_named(0, "Drake").is_empty());
    let paid = s.game().clone();
    settle(&mut s);
    let drakes = s.battlefield_permanents_named(0, "Drake");
    assert_eq!(drakes.len(), 1);
    let token = s.game().characteristics(drakes[0]);
    assert_eq!(token.pt(), (2, 2));
    assert!(token.keywords.has(Keyword::Flying));
    assert_eq!(counters_of(&s, second, Incubation), 0);

    *s.game_mut() = paid;
    counter_top_ability(&mut s);
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Drake").is_empty());
    assert_eq!(counters_of(&s, second, Incubation), 0);
}

#[test]
fn high_society_hunter_can_sacrifice_a_token_or_decline() {
    use crate::support::{
        game::clear_summoning_sickness,
        inspect::{counters, offered_cards},
        play::{choose, select_card},
    };
    use managym::agent::action::Action;

    let mut s = five_color_setup(&["High-Society Hunter", "Cackling Prowler"]);
    let hunter = s.force_permanent_on_battlefield(0, "High-Society Hunter");
    let prowler = s.force_permanent_on_battlefield(0, "Cackling Prowler");
    let token = s
        .game_mut()
        .create_token("token.cat", PlayerId(0), false)
        .unwrap();
    let cat = card_of(&s, token);
    clear_summoning_sickness(&mut s, hunter);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, hunter);
    s.advance_until(
        |s| !offered_cards(s).is_empty(),
        "Hunter sacrifice choice".into(),
    );
    let offered = offered_cards(&s);
    assert_eq!(offered.len(), 2);
    assert!(offered.contains(&cat) && offered.contains(&card_of(&s, prowler)));
    let pending = s.game().clone();
    let hand = s.zone_size(0, ZoneType::Hand);
    for sacrifice in [false, true] {
        *s.game_mut() = pending.clone();
        if sacrifice {
            select_card(&mut s, cat);
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut s);
        assert_eq!(s.game().state.permanents[token].is_none(), sacrifice);
        assert_eq!(counters(&s, hunter), i32::from(sacrifice));
        assert_eq!(s.zone_size(0, ZoneType::Hand), hand);
        let restored = s.game().clone();
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = managym::agent::observation::Observation::for_player(s.game(), viewer);
            assert_eq!(obs.turn.creature_died_this_turn, sacrifice);
        }
        s.advance_to_active_step(0, StepKind::End);
        let end_hash = s.game().state.deterministic_hash();
        *s.game_mut() = restored;
        s.advance_to_active_step(0, StepKind::End);
        assert_eq!(s.game().state.deterministic_hash(), end_hash);
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = managym::agent::observation::Observation::for_player(s.game(), viewer);
            assert_eq!(obs.stack_objects.len(), usize::from(sacrifice));
            if sacrifice {
                let key = s.game().state[card_of(&s, prowler)].definition_id;
                assert_eq!(obs.stack_objects[0].source_card_registry_key, key.0 as i32);
                assert_eq!(obs.stack_objects[0].ability_index, Some(0));
            }
        }
        settle(&mut s);
        assert_eq!(counters(&s, prowler), i32::from(sacrifice));
        s.advance_to_active_step(1, StepKind::Main);
        assert!(!s.game().state.turn.creature_died_this_turn());
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
fn high_society_hunter_draws_for_simultaneous_nontoken_deaths() {
    use crate::support::{inspect::hand, play::cast};
    use managym::{agent::observation::Observation, state::player::ManaPayment};
    let mut s = five_color_setup(&[
        "High-Society Hunter",
        "Gray Ogre",
        "Fumigate",
        "Lightning Bolt",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let hunter = s.force_permanent_on_battlefield(0, "High-Society Hunter");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let island = s.force_permanent_on_battlefield(1, "Island");
    let original = s.game().state[card_of(&s, ogre)].clone();
    let copy = card_of(&s, island);
    s.game_mut().state[copy].become_copy_of(&original, &[]);
    let hunter_key = s.game().state[card_of(&s, hunter)].definition_id;
    cast(&mut s, "Fumigate");
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 2,
        "Hunter's two death triggers".into(),
    );
    let pending = s.game().clone();
    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        assert_eq!(obs.stack_objects.len(), 2);
        assert!(obs.stack_objects.iter().all(|a| a.source_card_registry_key
            == hunter_key.0 as i32
            && a.ability_index == Some(1)));
    }
    for counter in [false, true] {
        *s.game_mut() = pending.clone();
        if counter {
            counter_top_ability(&mut s);
        }
        let before = hand(&s, 0);
        settle(&mut s);
        let drawn: Vec<_> = hand(&s, 0)
            .into_iter()
            .filter(|c| !before.contains(c))
            .collect();
        assert_eq!(drawn.len(), if counter { 1 } else { 2 });
        let own = Observation::for_player(s.game(), PlayerId(0));
        let foe = Observation::for_player(s.game(), PlayerId(1));
        for card in drawn {
            let object = &s.game().state[card];
            assert!(own.agent_cards.iter().any(|c| c.zone == ZoneType::Hand
                && c.id == object.id.0 as i32
                && c.characteristics.name == object.name));
        }
        assert!(foe.opponent_cards.iter().all(|c| c.zone != ZoneType::Hand));
    }
}

#[test]
fn cackling_prowler_ward_counters_a_spell_unless_two_is_paid() {
    use crate::support::{
        game::{clear_hand, five_color_unfunded_setup, force_lands},
        play::cast_only,
    };
    use managym::{
        agent::action::{ActionSpaceKind, ActionType},
        flow::turn::StepKind,
        state::game_object::PlayerId,
    };
    for pay in [false, true] {
        let mut s = five_color_unfunded_setup(&["Cackling Prowler", "Lightning Strike"]);
        let prowler = s.force_permanent_on_battlefield(0, "Cackling Prowler");
        force_lands(&mut s, 1, "Mountain", 4);
        clear_hand(&mut s, 0);
        clear_hand(&mut s, 1);
        s.force_card_in_hand(1, "Lightning Strike");
        s.advance_to_active_step(1, StepKind::Main);
        cast_only(&mut s);
        s.choose_target_named("Cackling Prowler");
        s.pass_priority();
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
        assert_eq!(s.action_space().player, Some(PlayerId(1)));
        assert!(s.take_action_by_type(if pay {
            ActionType::PayCost
        } else {
            ActionType::DeclineChoice
        }));
        s.pass_priority();
        s.pass_priority();
        assert_eq!(s.game().state.permanents[prowler].is_none(), pay);
    }
}

#[test]
fn incinerating_blast_offers_discard_only_after_damaging_a_legal_target() {
    use crate::support::{
        game::force_new_card_in_hand,
        inspect::{now, permanent},
        play::{cast_card, fork_choosing, resolve_top},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::player::ManaPayment,
    };

    let mut s = five_color_setup(&["Incinerating Blast", "Gray Ogre"]);
    let caster = PlayerId(0);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let victim = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let creature = card_of(&s, victim);
    let first = force_new_card_in_hand(&mut s, 0, "Forest");
    let second = force_new_card_in_hand(&mut s, 0, "Mountain");
    let spell = force_new_card_in_hand(&mut s, 0, "Incinerating Blast");
    cast_card(&mut s, spell);
    s.choose_target_named("Gray Ogre");
    let pending = s.game().clone();
    let library = pending.state.zones.size(ZoneType::Library, caster);

    resolve_top(&mut s);
    assert_eq!(permanent(&s, victim).damage, 6);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.action_space().player, Some(caster));
    assert_eq!(s.action_space().actions.len(), 3);
    for discard in [None, Some(first), Some(second)] {
        let action = discard.map_or(Action::Decline { player: caster }, |object| {
            Action::SelectCard {
                player: caster,
                object,
            }
        });
        let branch = fork_choosing(&s, |offered| *offered == action);
        let state = &branch.game().state;
        assert_eq!(
            state.zones.size(ZoneType::Library, caster),
            library - usize::from(discard.is_some())
        );
        assert_eq!(state.zones.size(ZoneType::Hand, caster), 2);
        for card in [first, second] {
            let zone = if Some(card) == discard {
                ZoneType::Graveyard
            } else {
                ZoneType::Hand
            };
            assert!(state.zones.contains(now(&branch, card), zone, caster));
        }
        assert!(state
            .zones
            .contains(now(&branch, creature), ZoneType::Graveyard, PlayerId(1)));
    }

    *s.game_mut() = pending;
    move_card(&mut s, creature, ZoneType::Exile);
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    let state = &s.game().state;
    assert!(state.stack_objects.is_empty());
    assert_eq!(
        state.zones.zone_objects(ZoneType::Hand, caster).to_vec(),
        [first, second]
    );
    assert_eq!(state.zones.size(ZoneType::Library, caster), library);
}

#[test]
fn second_draw_listeners_share_the_ordinal_and_resolve_independently() {
    use crate::support::{
        inspect::counters_of,
        play::{cast, resolve_top, resolve_until_priority},
    };
    use managym::{
        agent::{action::ActionSpaceKind, observation::Observation},
        state::{game_object::Target, permanent::CounterKind::PlusOnePlusOne},
    };

    let mut s = five_color_setup(&[
        "Clone",
        "Homunculus Horde",
        "Erudite Wizard",
        "Mischievous Mystic",
        "Lightning Bolt",
    ]);
    let original = s.force_permanent_on_battlefield(1, "Homunculus Horde");
    let clone = cast(&mut s, "Clone");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert!(s.choose_target(Target::Permanent(original)));
    settle(&mut s);
    let clone = card_of(&s, permanent_of(&s, clone).unwrap());
    let wizard = s.force_permanent_on_battlefield(0, "Erudite Wizard");
    s.force_permanent_on_battlefield(0, "Mischievous Mystic");
    s.game_mut().draw_cards(PlayerId(0), 3);
    let ordinals: Vec<_> = s
        .game()
        .state
        .observation_events
        .iter()
        .filter_map(|event| match event {
            managym::flow::event::GameEvent::CardDrawn {
                player: PlayerId(0),
                nth_this_turn,
            } => Some(*nth_this_turn),
            _ => None,
        })
        .collect();
    assert_eq!(ordinals, [1, 2, 3]);
    s.advance_until(
        |s| {
            s.action_space().kind == ActionSpaceKind::Priority
                && s.game().state.stack_objects.len() == 3
        },
        "expected three second-draw triggers".into(),
    );
    let root = s.game().clone();
    for (departed, countered) in [(false, false), (true, false), (false, true)] {
        *s.game_mut() = root.clone();
        if departed {
            move_card(&mut s, clone, ZoneType::Hand);
            let pending = s.game().clone();
            *s.game_mut() = pending;
        }
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = Observation::for_player(s.game(), viewer);
            let drawer = if viewer == PlayerId(0) {
                &obs.agent
            } else {
                &obs.opponent
            };
            assert_eq!(drawer.turn_history.cards_drawn, 3);
            assert!(obs
                .opponent_cards
                .iter()
                .all(|card| card.zone != ZoneType::Hand));
            for name in ["Erudite Wizard", "Homunculus Horde", "Mischievous Mystic"] {
                assert_eq!(
                    obs.stack_objects
                        .iter()
                        .filter(|entry| entry
                            .description
                            .as_deref()
                            .is_some_and(|text| text.starts_with(name)))
                        .count(),
                    1
                );
            }
        }
        if countered {
            let target = s
                .game()
                .state
                .stack_objects
                .iter()
                .find(|entry| entry.source_object() == clone)
                .unwrap()
                .id();
            counter_ability(&mut s, target);
        }
        settle(&mut s);
        assert_eq!(counters_of(&s, wizard, PlusOnePlusOne), 1);
        let faeries = s.battlefield_permanents_named(0, "Faerie");
        assert_eq!(faeries.len(), 1);
        let faerie = s.game().characteristics(faeries[0]);
        assert!(faerie.keywords.has(Keyword::Flying));
        assert_eq!(faerie.pt(), (1, 1));
        let hordes = s.battlefield_permanents_named(0, "Homunculus Horde");
        assert_eq!(
            hordes
                .iter()
                .filter(|id| !s.game().state[card_of(&s, **id)].is_card())
                .count(),
            usize::from(!countered)
        );
        assert_eq!(
            s.battlefield_permanents_named(1, "Homunculus Horde").len(),
            1
        );
    }
    let late = s.force_permanent_on_battlefield(0, "Erudite Wizard");
    s.game_mut().draw_cards(PlayerId(0), 1);
    s.game_mut().draw_cards(PlayerId(1), 2);
    resolve_until_priority(&mut s);
    assert_eq!(counters_of(&s, late, PlusOnePlusOne), 0);
    assert_eq!(
        s.battlefield_permanents_named(1, "Homunculus Horde").len(),
        2
    );
    s.advance_to_active_step(1, StepKind::Main);
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_until_priority(&mut s);
    assert_eq!(counters_of(&s, late, PlusOnePlusOne), 0);
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_until_priority(&mut s);
    assert_eq!(counters_of(&s, late, PlusOnePlusOne), 1);
    assert_eq!(counters_of(&s, wizard, PlusOnePlusOne), 2);
}

fn assert_raid(s: &crate::support::game::Scenario, attacker: Option<usize>) {
    use managym::agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
        observation_layout::player::ATTACKED,
    };
    for viewer in 0..2 {
        let obs = Observation::for_player(s.game(), PlayerId(viewer));
        let encoded = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
        assert_eq!(
            encoded.agent_player[ATTACKED],
            f32::from(attacker == Some(viewer))
        );
        assert_eq!(
            encoded.opponent_player[ATTACKED],
            f32::from(attacker == Some(1 - viewer))
        );
        assert_eq!(obs.agent.turn_history.attacked, attacker == Some(viewer));
        assert_eq!(
            obs.opponent.turn_history.attacked,
            attacker == Some(1 - viewer)
        );
    }
}

#[test]
fn linden_and_chart_read_the_completed_attack_declaration() {
    use crate::support::play::cast;
    use managym::agent::action::ActionSpaceKind;
    let queen = "Linden, the Steadfast Queen";
    let mut s = five_color_setup(&[queen, "Savannah Lions", "Chart a Course", "Lightning Bolt"]);
    let linden = s.force_permanent_on_battlefield(0, queen);
    let lions = s.force_permanent_on_battlefield(0, "Savannah Lions");
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    let partial = s.game().clone();
    let offer = s.action_space().clone();
    assert_raid(&s, None);
    s.decline_attack();
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_raid(&s, Some(0));

    *s.game_mut() = partial;
    assert_eq!(s.action_space(), &offer);
    let chosen = &s.game().state.combat.as_ref().unwrap().attackers;
    assert_eq!(chosen.len(), 1);
    assert_eq!(
        crate::support::inspect::permanent(&s, chosen[0]).tapped,
        chosen[0] == lions
    );
    assert_raid(&s, None);
    s.declare_attack();
    s.advance_until(
        |s| {
            s.action_space().kind == ActionSpaceKind::Priority
                && s.game().state.stack_objects.len() == 2
        },
        "two Linden triggers".into(),
    );
    assert_raid(&s, Some(0));
    let before = s.game().state.players[0].life;
    let target = s.game().state.stack_objects[0].id();
    counter_ability(&mut s, target);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, before + 1);
    s.game_mut()
        .state
        .turn
        .additional_combats
        .push(managym::flow::turn::AdditionalCombat { attackers: None });
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.decline_attack(); // Only vigilant Linden can attack again.
    assert_raid(&s, Some(0));
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    for id in [linden, lions] {
        let object = card_of(&s, id);
        move_card(&mut s, object, ZoneType::Graveyard);
    }
    s.set_player_mana_pool(0, "1U");
    let hand = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    cast(&mut s, "Chart a Course");
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand + 2
    );
    s.advance_to_active_step(1, StepKind::Main);
    assert_raid(&s, None);
}

#[test]
fn linden_and_chart_ignore_entry_attacking_and_empty_declarations() {
    use crate::support::{game::force_new_card_in_hand, play::cast_card};
    let queen = "Linden, the Steadfast Queen";
    let mut s = five_color_setup(&[queen, "Chart a Course", "Gray Ogre"]);
    s.force_permanent_on_battlefield(1, queen);
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.advance_to_active_step(1, StepKind::BeginningOfCombat);
    s.game_mut()
        .create_token("token.soldier", PlayerId(1), true)
        .unwrap();
    s.advance_to_active_step(1, StepKind::DeclareAttackers);
    let root = s.game().clone();
    s.declare_attack();
    s.declare_attack();
    assert_raid(&s, Some(1));
    assert_eq!(s.game().state.stack_objects.len(), 1);
    *s.game_mut() = root;
    s.decline_attack();
    s.decline_attack();
    assert_raid(&s, None);
    assert!(s.game().state.stack_objects.is_empty());
    s.advance_to_active_step(1, StepKind::PostcombatMain);
    s.set_player_mana_pool(1, "1U");
    let hand = s.game().state.zones.size(ZoneType::Hand, PlayerId(1));
    let chart = force_new_card_in_hand(&mut s, 1, "Chart a Course");
    cast_card(&mut s, chart);
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(1)),
        hand + 1
    );
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

/// One ordinary declaration drives the threshold trigger, two resolution-time
/// counts, the raid entry replacement and both kinds of raid listener.
#[test]
fn attack_batch_thresholds_counts_and_raid_listeners() {
    use crate::support::{
        game::{clear_summoning_sickness, enters_battlefield_with_triggers},
        inspect::{counters, offered_targets},
        play::{pass_until, settle_targeting_opponent},
    };
    use managym::{agent::action::ActionSpaceKind, state::target::Target};

    for declared in [0, 2, 3] {
        let mut s = five_color_setup(&[
            "Armasaur Guide",
            "Battlesong Berserker",
            "Ancestor Dragon",
            "Jazal Goldmane",
            "Skyship Buccaneer",
            "Storm Fleet Spy",
            "Goblin Boarders",
            "Gorehorn Raider",
            "Searslicer Goblin",
            "Midnight Snack",
            "Gray Ogre",
        ]);
        s.game_mut().scenario_set_life(PlayerId(1), 100);
        let guide = s.force_permanent_on_battlefield(0, "Armasaur Guide");
        let berserker = s.force_permanent_on_battlefield(0, "Battlesong Berserker");
        s.force_permanent_on_battlefield(0, "Ancestor Dragon");
        let jazal = s.force_permanent_on_battlefield(0, "Jazal Goldmane");
        s.force_permanent_on_battlefield(0, "Searslicer Goblin");
        s.force_permanent_on_battlefield(0, "Midnight Snack");
        let mut attackers = Vec::new();
        for _ in 0..3 {
            let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
            clear_summoning_sickness(&mut s, ogre);
            attackers.push(ogre);
        }
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        let before = s.game().state.players[0].life;
        for i in 0..3 {
            if i < declared {
                s.declare_attack();
            } else {
                s.decline_attack();
            }
        }
        let mut targets = 0;
        while matches!(
            s.action_space().kind,
            ActionSpaceKind::OrderTriggers | ActionSpaceKind::ChooseTarget
        ) {
            if s.action_space().kind == ActionSpaceKind::OrderTriggers {
                s.step_action(0);
            } else {
                assert!(offered_targets(&s).contains(&Target::Permanent(guide)));
                assert!(s.choose_target(Target::Permanent(berserker)));
                targets += 1;
            }
        }
        assert_eq!(
            targets,
            usize::from(declared > 0) + usize::from(declared >= 3)
        );
        settle(&mut s);
        assert_eq!(s.game().state.players[0].life - before, declared as i32);
        assert_eq!(counters(&s, berserker), i32::from(declared >= 3));
        assert_eq!(
            s.game()
                .characteristics(berserker)
                .keywords
                .has(Keyword::Menace),
            declared > 0
        );
        assert_eq!(
            s.game().characteristics(berserker).power,
            3 + i32::from(declared > 0) + i32::from(declared >= 3)
        );
        s.set_player_mana_pool(0, "3WW");
        activate(&mut s, jazal, 0);
        spend_colorless(&mut s);
        settle(&mut s);
        for &ogre in &attackers {
            let expected = if s.game().state.permanents[ogre].as_ref().unwrap().attacking {
                declared as i32
            } else {
                0
            };
            assert_eq!(
                s.game().characteristics(ogre).pt(),
                (2 + expected, 2 + expected)
            );
        }
        s.advance_to_active_step(0, StepKind::PostcombatMain);
        let boarders = enters_battlefield_with_triggers(&mut s, 0, "Goblin Boarders");
        assert_eq!(counters(&s, boarders), i32::from(declared > 0));
        for spy in ["Skyship Buccaneer", "Storm Fleet Spy"] {
            enters_battlefield_with_triggers(&mut s, 0, spy);
            let hand = s.zone_size(0, ZoneType::Hand);
            crate::support::play::resolve_everything(&mut s);
            assert_eq!(
                s.zone_size(0, ZoneType::Hand),
                hand + usize::from(declared > 0),
                "{spy}"
            );
        }
        let life = s.game().state.players[1].life;
        enters_battlefield_with_triggers(&mut s, 0, "Gorehorn Raider");
        s.game_mut().scenario_refresh_priority().unwrap();
        if declared > 0 {
            pass_until(&mut s, ActionSpaceKind::ChooseTarget);
            assert!(offered_targets(&s).contains(&Target::Permanent(guide)));
            settle_targeting_opponent(&mut s);
        } else {
            settle(&mut s);
        }
        assert_eq!(
            s.game().state.players[1].life,
            life - 2 * i32::from(declared > 0)
        );
        s.advance_to_active_step(0, StepKind::End);
        settle(&mut s);
        for token in ["Food", "Goblin"] {
            assert_eq!(
                s.battlefield_permanents_named(0, token).len(),
                usize::from(declared > 0),
                "{token}"
            );
        }
    }
}

#[test]
fn attack_batch_midnight_snack_reads_total_after_sacrifice() {
    use crate::support::play::target_opponent;
    use managym::agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig, TURN_STATE_DIM},
        observation_layout::turn_state,
    };
    let mut s = five_color_setup(&["Midnight Snack", "Ancestor Dragon", "Gray Ogre"]);
    let snack = s.force_permanent_on_battlefield(0, "Midnight Snack");
    s.force_permanent_on_battlefield(0, "Ancestor Dragon");
    let food = s
        .game_mut()
        .create_token("token.food", PlayerId(0), false)
        .unwrap();
    s.set_player_mana_pool(0, "2");
    activate(&mut s, food, 0);
    settle(&mut s);
    for _ in 0..2 {
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        crate::support::game::clear_summoning_sickness(&mut s, ogre);
    }
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.declare_attack();
    settle(&mut s);
    for viewer in 0..2 {
        let obs = Observation::for_player(s.game(), PlayerId(viewer));
        let encoded = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
        let row = &encoded.player_turn_state[viewer * TURN_STATE_DIM..];
        assert_eq!(
            (row[turn_state::LIFE_GAINS], row[turn_state::LIFE_GAINED]),
            (2., 5.)
        );
    }
    let life = s.game().state.players[1].life;
    s.set_player_mana_pool(0, "2B");
    activate(&mut s, snack, 0);
    if s.action_space().kind == managym::agent::action::ActionSpaceKind::ChooseTarget {
        target_opponent(&mut s);
    }
    spend_colorless(&mut s);
    assert!(s.game().state.permanents[snack].is_none());
    settle(&mut s);
    assert_eq!(s.game().state.players[1].life, life - 5);
}

#[test]
fn attack_batch_alesha_reanimates_with_departed_power() {
    use crate::support::{
        game::plant_in_graveyard,
        inspect::{counters, offered_targets},
        play::pass_until,
    };
    use managym::{
        agent::{
            action::ActionSpaceKind,
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig},
            observation_layout::stack,
        },
        state::target::Target,
    };
    let mut s = five_color_setup(&["Alesha, Who Laughs at Fate", "Gray Ogre", "Serra Angel"]);
    let alesha = s.force_permanent_on_battlefield(0, "Alesha, Who Laughs at Fate");
    let ogre = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    let angel = plant_in_graveyard(&mut s, 0, "Serra Angel");
    crate::support::game::clear_summoning_sickness(&mut s, alesha);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, alesha);
    settle(&mut s);
    assert_eq!(counters(&s, alesha), 1);
    s.advance_to_active_step(0, StepKind::End);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(!offered_targets(&s).contains(&Target::GraveyardCard(angel)));
    assert!(s.choose_target(Target::GraveyardCard(ogre)));
    let original = card_of(&s, alesha);
    move_card(&mut s, original, ZoneType::Exile);
    for viewer in 0..2 {
        let obs = Observation::for_player(s.game(), PlayerId(viewer));
        let ability = &obs.stack_objects[0];
        assert_eq!(ability.source_power, Some(3));
        assert!(ability
            .description
            .as_ref()
            .unwrap()
            .contains("source power 3"));
        let encoded = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
        assert_eq!(encoded.stack[stack::SOURCE_POWER], 0.3);
    }
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre").len(), 1);
}

#[test]
fn attack_batch_each_doubled_occurrence_keeps_its_attacker() {
    use crate::support::{
        game::{clear_summoning_sickness, until_end_of_turn},
        inspect::counters,
    };
    use managym::state::{
        ability::{
            Ability, Condition, Effect, ObjectSelector, Per, TriggerCondition, TriggerSubject,
        },
        effect::Modification,
        permanent::CounterKind,
        predicate::CardPredicate,
        stack_object::StackObject,
        target::Target,
    };
    let mut s = five_color_setup(&["Katara, the Fearless", "Gray Ogre"]);
    let katara = s.force_permanent_on_battlefield(0, "Katara, the Fearless");
    let listener = Ability::Triggered {
        condition: TriggerCondition::ActiveIf {
            active_if: Condition::YourTurn,
            condition: Box::new(TriggerCondition::attacks(
                TriggerSubject::you_control(CardPredicate::creature()),
                Per::Object,
            )),
        },
        intervening_if: None,
        effects: vec![Effect::put_counters(
            ObjectSelector::Triggering,
            CounterKind::PlusOnePlusOne,
            1,
        )],
    };
    until_end_of_turn(
        &mut s,
        katara,
        Modification::GrantTriggered(Box::new(listener)),
    );
    let ogres = [0, 1].map(|_| {
        let id = s.force_permanent_on_battlefield(0, "Gray Ogre");
        clear_summoning_sickness(&mut s, id);
        id
    });
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.declare_attack();
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 4,
        "four attack triggers".into(),
    );
    for id in ogres {
        let triggers = s.game().state.stack_objects.iter().filter(|entry| {
            matches!(entry, StackObject::TriggeredAbility(a) if a.context == Some(Target::Permanent(id)))
        });
        assert_eq!(triggers.count(), 2);
    }
    settle(&mut s);
    for id in ogres {
        assert_eq!(counters(&s, id), 2);
    }
}

#[test]
fn attack_batch_aurelia_first_declaration_tracks_each_appearance() {
    use crate::support::{game::clear_summoning_sickness, inspect::tapped};
    use managym::{
        agent::{action::ActionSpaceKind, observation::Observation},
        flow::turn::AdditionalCombat,
    };
    let name = "Aurelia, the Warleader";
    let mut s = five_color_setup(&[name, "Gray Ogre", "Honest Work"]);
    let aurelia = s.force_permanent_on_battlefield(0, name);
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let original = card_of(&s, aurelia);
    clear_summoning_sickness(&mut s, other);
    let main = s.game().clone();
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    let partial = s.game().clone();
    for viewer in 0..2 {
        let obs = Observation::for_player(s.game(), PlayerId(viewer));
        assert!(!obs
            .agent_permanents
            .iter()
            .chain(&obs.opponent_permanents)
            .any(|p| p.attacked_this_turn));
    }
    s.decline_attack();
    *s.game_mut() = partial;
    s.declare_attack();
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert!(tapped(&s, other));
    settle(&mut s);
    assert!(!tapped(&s, other));
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    let first = s.game().clone();
    for blink in [false, true] {
        *s.game_mut() = first.clone();
        let attacker = if blink {
            let away = move_card(&mut s, original, ZoneType::Exile);
            let back = move_card(&mut s, away, ZoneType::Battlefield);
            assert_eq!(s.game().state.turn.attack_count(back), 0);
            permanent_of(&s, back).unwrap()
        } else {
            aurelia
        };
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        attack_with(&mut s, attacker);
        assert_eq!(s.game().state.stack_objects.len(), usize::from(blink));
        assert_raid(&s, Some(0));
    }
    *s.game_mut() = main;
    clear_summoning_sickness(&mut s, aurelia);
    let aura = s.force_permanent_on_battlefield(1, "Honest Work");
    s.game_mut().state.permanents[aura]
        .as_mut()
        .unwrap()
        .attached_to = Some(original);
    assert!(s.game().characteristics(aurelia).abilities_lost);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, aurelia);
    assert!(s.game().state.stack_objects.is_empty());
    let aura_object = card_of(&s, aura);
    move_card(&mut s, aura_object, ZoneType::Graveyard);
    assert!(!s.game().characteristics(aurelia).abilities_lost);
    s.game_mut().state.permanents[aurelia]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.game_mut()
        .state
        .turn
        .additional_combats
        .push(AdditionalCombat { attackers: None });
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, aurelia);
    assert!(s.game().state.stack_objects.is_empty());
    assert_eq!(s.game().state.turn.attack_count(original), 2);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().state.turn.attack_count(original), 0);
    assert_raid(&s, None);
}

#[test]
fn attack_batch_aurelia_that_entered_attacking_still_gets_a_first_declaration() {
    use crate::support::inspect::library_card;
    use managym::flow::turn::AdditionalCombat;
    let name = "Aurelia, the Warleader";
    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    crate::support::game::clear_summoning_sickness(&mut s, ogre);
    let printed = s.game().state[library_card(&s, 0, name)].clone();
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    let token = s
        .game_mut()
        .create_token("token.soldier", PlayerId(0), true)
        .unwrap();
    let object = card_of(&s, token);
    s.game_mut().state[object].become_copy_of(&printed, &[]);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.decline_attack();
    assert_raid(&s, None);
    assert_eq!(s.game().state.turn.attack_count(object), 0);
    assert!(s.game().state.stack_objects.is_empty());
    s.game_mut()
        .state
        .turn
        .additional_combats
        .push(AdditionalCombat { attackers: None });
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    s.game_mut().state.permanents[token]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, token);
    assert_eq!(s.game().state.turn.attack_count(object), 1);
    assert_eq!(s.game().state.stack_objects.len(), 1);
}

#[test]
fn attack_batch_mana_ability_counts_live_board_objects() {
    use crate::support::game::until_end_of_turn;
    use managym::agent::action::ActionSpaceKind;
    use managym::state::{
        ability::{Effect, ObjectSelector},
        card::ActivatedAbilityDefinition,
        effect::Modification,
        mana::Mana,
        predicate::CardPredicate,
    };
    let mut s = five_color_setup(&["Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    until_end_of_turn(
        &mut s,
        source,
        Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::AddMana {
                mana: Mana::parse("G"),
                amount: ObjectSelector::each_you_control(CardPredicate::creature()).count(),
                until_end_of_combat: false,
            }],
            ..Default::default()
        })),
    );
    crate::support::game::clear_summoning_sickness(&mut s, source);
    s.allow_priority_floats();
    s.set_player_mana_pool(0, "");
    let root = s.game().clone();
    for remove in [false, true] {
        *s.game_mut() = root.clone();
        if remove {
            let object = card_of(&s, other);
            move_card(&mut s, object, ZoneType::Exile);
        }
        s.game_mut().scenario_refresh_priority().unwrap();
        activate(&mut s, source, 0);
        if s.action_space().kind == ActionSpaceKind::Modal {
            assert_eq!(s.action_space().actions.len(), 1);
            s.step_action(0);
        }
        assert_eq!(
            s.game().state.players[0].mana_pool.total(),
            if remove { 1 } else { 2 }
        );
        assert!(s.game().state.stack_objects.is_empty());
    }
}

#[test]
fn attack_batch_spell_payment_comparison_survives_countering() {
    use crate::support::{
        game::{force_new_card_in_hand, plant_in_graveyard},
        inspect::top_spell,
        play::{cast, cast_card},
    };
    use managym::{
        agent::{
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig},
            observation_layout::stack,
        },
        state::{
            ability::{CompareOp, Condition, Effect, EffectValue, ObjectSelector, TargetSpec},
            player::ManaPayment,
            target::Target,
        },
    };
    for discount in [false, true] {
        let mut s = five_color_setup(&["Tolarian Terror", "Counterspell", "Consider"]);
        s.game_mut().mana_payment = [ManaPayment::Auto; 2];
        if discount {
            plant_in_graveyard(&mut s, 0, "Consider");
        }
        let counter = force_new_card_in_hand(&mut s, 1, "Counterspell");
        let target = || Box::new(ObjectSelector::Target(TargetSpec::Spell));
        s.game_mut().state[counter].spell_effects.push(Effect::If {
            condition: Condition::compare(
                EffectValue::SpellManaSpent(target()),
                CompareOp::Lt,
                EffectValue::SpellManaValue(target()),
            ),
            then: vec![Effect::gain_life(1)],
            otherwise: Vec::new(),
        });
        cast(&mut s, "Tolarian Terror");
        let spell = top_spell(&s).object;
        for viewer in 0..2 {
            let obs = Observation::for_player(s.game(), PlayerId(viewer));
            let spent = 7 - u16::from(discount);
            assert_eq!(obs.stack_objects[0].mana_spent, spent);
            assert_eq!(
                encode(&obs, &ObservationEncoderConfig::default())
                    .unwrap()
                    .stack[stack::MANA_SPENT],
                f32::from(spent)
            );
        }
        s.pass_priority();
        cast_card(&mut s, counter);
        assert!(s.choose_target(Target::StackSpell(spell)));
        let before = s.game().state.players[1].life;
        settle(&mut s);
        assert!(s
            .battlefield_permanents_named(0, "Tolarian Terror")
            .is_empty());
        assert_eq!(s.game().state.players[1].life, before + i32::from(discount));
    }
}

#[test]
fn power_batch_goblin_and_ruby_qualify_at_declaration() {
    use crate::support::game::{clear_summoning_sickness, set_counters};
    use managym::state::permanent::CounterKind;
    for (name, qualified) in [
        ("Courageous Goblin", (3, 2)),
        ("Ruby, Daring Tracker", (3, 4)),
    ] {
        let mut s = five_color_setup(&[name, "Gray Ogre"]);
        let source = s.force_permanent_on_battlefield(0, name);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        clear_summoning_sickness(&mut s, source);
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        let base = s.game().characteristics(source).pt();
        let root = s.game().clone();
        for before in [false, true] {
            *s.game_mut() = root.clone();
            if before {
                set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 2);
            }
            attack_with(&mut s, source);
            assert_eq!(s.game().state.stack_objects.len(), usize::from(before));
            set_counters(
                &mut s,
                ogre,
                CounterKind::PlusOnePlusOne,
                if before { 0 } else { 2 },
            );
            settle(&mut s);
            let view = s.game().characteristics(source);
            assert_eq!(view.pt(), if before { qualified } else { base });
            assert_eq!(
                view.keywords.has(Keyword::Menace),
                before && name == "Courageous Goblin"
            );
        }
    }
}

#[test]
fn power_batch_ruby_taps_for_either_color() {
    use crate::support::play::choose_mode;
    use managym::state::mana::Mana;
    let mut s = five_color_setup(&["Ruby, Daring Tracker"]);
    let ruby = s.force_permanent_on_battlefield(0, "Ruby, Daring Tracker");
    s.set_player_mana_pool(0, "");
    s.allow_priority_floats();
    activate(&mut s, ruby, 0);
    let choice = s.game().clone();
    assert_eq!(s.action_space().actions.len(), 2);
    for (mode, color) in ["R", "G"].into_iter().enumerate() {
        *s.game_mut() = choice.clone();
        choose_mode(&mut s, mode);
        assert!(crate::support::inspect::tapped(&s, ruby));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 1);
        assert_eq!(s.game().state.players[0].mana_pool, Mana::parse(color));
    }
}

#[test]
fn power_batch_hornbeetle_excludes_itself_and_rechecks_other_power() {
    use crate::support::{game::set_counters, inspect::counters};
    use managym::state::permanent::CounterKind;
    let mut s = five_color_setup(&["Nessian Hornbeetle", "Gray Ogre"]);
    let beetle = s.force_permanent_on_battlefield(0, "Nessian Hornbeetle");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    set_counters(&mut s, beetle, CounterKind::PlusOnePlusOne, 2);
    let root = s.game().clone();
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(s.game().state.stack_objects.is_empty());
    *s.game_mut() = root;
    set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 2);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    let pending = s.game().clone();
    for stays in [true, false] {
        *s.game_mut() = pending.clone();
        if !stays {
            let obj = card_of(&s, ogre);
            move_card(&mut s, obj, ZoneType::Hand);
        }
        settle(&mut s);
        assert_eq!(counters(&s, beetle), 2 + i32::from(stays));
    }
}

#[test]
fn power_batch_surrak_sums_signed_power_and_rechecks_after_targeting() {
    use crate::support::{
        game::{set_counters, until_end_of_turn},
        inspect::offered_targets,
    };
    use managym::state::{effect::Modification, game_object::Target, permanent::CounterKind};
    let name = "Surrak, the Hunt Caller";
    let mut s = five_color_setup(&[name, "Hill Giant", "Gray Ogre"]);
    let surrak = s.force_permanent_on_battlefield(0, name);
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    until_end_of_turn(
        &mut s,
        ogre,
        Modification::ModifyPt {
            power: -3,
            toughness: 0,
        },
    );
    let root = s.game().clone(); // 5 + 3 - 1
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(s.game().state.stack_objects.is_empty());
    *s.game_mut() = root;
    set_counters(&mut s, giant, CounterKind::PlusOnePlusOne, 1);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    let offered = offered_targets(&s);
    assert_eq!(offered.len(), 3);
    assert!([surrak, giant, ogre]
        .iter()
        .all(|&id| offered.contains(&Target::Permanent(id))));
    s.choose_target_named("Gray Ogre");
    let pending = s.game().clone();
    for meets in [false, true] {
        *s.game_mut() = pending.clone();
        if !meets {
            set_counters(&mut s, giant, CounterKind::PlusOnePlusOne, 0);
        }
        settle(&mut s);
        assert_eq!(
            s.game().characteristics(ogre).keywords.has(Keyword::Haste),
            meets
        );
    }
}

#[test]
fn power_batch_ashroot_and_partners_read_live_or_departed_source() {
    use crate::support::{
        game::{clear_summoning_sickness, set_counters},
        inspect::counters,
    };
    use managym::{
        agent::observation::Observation,
        state::{game_object::Target, permanent::CounterKind},
    };
    for (name, base, keyword) in [
        ("Ashroot Animist", 4, Keyword::Trample),
        ("Halana and Alena, Partners", 2, Keyword::Haste),
    ] {
        let mut s = five_color_setup(&[name, "Gray Ogre", "Hill Giant", "Lightning Bolt"]);
        let source = s.force_permanent_on_battlefield(0, name);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        s.force_permanent_on_battlefield(0, "Hill Giant");
        // A cast offer at priority binds the Ogre's exact incarnation. Carry that
        // address into the targeted trigger to distinguish reentry at the same revision.
        s.force_card_in_hand(0, "Lightning Bolt");
        s.game_mut().scenario_refresh_priority().unwrap();
        let main = s.game().semantic_decision_frame().unwrap();
        let compound = s.game().structured_offers().unwrap();
        let cast = compound
            .projection()
            .offers
            .iter()
            .find(|o| o.label == "Cast Lightning Bolt")
            .unwrap();
        let managym::agent::structured_offer::ChoiceStep::Select { candidates, .. } =
            &cast.choices[0];
        let candidate = candidates
            .initial
            .as_ref()
            .unwrap()
            .iter()
            .find(|c| c.label.contains("Gray Ogre"))
            .unwrap();
        let address = main
            .object_candidates
            .iter()
            .find(|a| a.offer_id == cast.id.0 && a.candidate_id == candidate.id.0)
            .unwrap()
            .clone();
        if name == "Ashroot Animist" {
            clear_summoning_sickness(&mut s, source);
            s.advance_to_active_step(0, StepKind::DeclareAttackers);
            attack_with(&mut s, source);
        } else {
            s.advance_to_active_step(0, StepKind::BeginningOfCombat);
        }
        assert!(!s.choose_target(Target::Permanent(source)));
        assert_eq!(s.action_space().actions.len(), 2);
        let frame = s.game().semantic_decision_frame().unwrap();
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|action| {
                matches!(
                    action,
                    managym::agent::action::Action::ChooseTarget {
                        target: Target::Permanent(id),
                        ..
                    } if *id == ogre
                )
            })
            .unwrap();
        let offer = &frame.offers[index];
        let mut branch = s.clone();
        let card = card_of(&branch, ogre);
        let moved = move_card(&mut branch, card, ZoneType::Exile);
        move_card(&mut branch, moved, ZoneType::Battlefield);
        let stale = managym::decision::Command {
            command_id: "departed-target".into(),
            expected_revision: frame.revision,
            offer_id: offer.id.0,
            answers: vec![],
            object_preconditions: vec![address],
        };
        assert!(matches!(
            branch.game_mut().execute_semantic_command(&stale),
            Err(managym::decision::SemanticError::StaleObject)
        ));
        s.choose_target_named("Gray Ogre");
        let pending = s.game().clone();
        for (increase, blink) in [(0, false), (2, false), (2, true)] {
            *s.game_mut() = pending.clone();
            set_counters(&mut s, source, CounterKind::PlusOnePlusOne, increase);
            if blink {
                let object = card_of(&s, source);
                let moved = move_card(&mut s, object, ZoneType::Exile);
                move_card(&mut s, moved, ZoneType::Battlefield);
            }
            let power = base + increase;
            for viewer in [PlayerId(0), PlayerId(1)] {
                let obs = Observation::for_player(s.game(), viewer);
                assert_eq!(obs.stack_objects[0].source_power, Some(power));
            }
            settle(&mut s);
            let view = s.game().characteristics(ogre);
            assert_eq!(view.pt(), (2 + power, 2 + power));
            assert!(view.keywords.has(keyword));
            assert_eq!(
                counters(&s, ogre),
                if keyword == Keyword::Haste { power } else { 0 }
            );
        }
    }
}

#[test]
fn power_batch_bulk_up_doubles_signed_resolution_power_and_flashes_back() {
    use crate::support::{
        game::{force_new_card_in_hand, until_end_of_turn},
        inspect::now,
        play::cast_card,
    };
    use managym::state::{effect::Modification, player::ManaPayment};
    let mut s = five_color_setup(&["Bulk Up", "Gray Ogre"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let spell = force_new_card_in_hand(&mut s, 0, "Bulk Up");
    cast_card(&mut s, spell);
    s.choose_target_named("Gray Ogre");
    let pending = s.game().clone();
    for delta in [-4, 1] {
        *s.game_mut() = pending.clone();
        until_end_of_turn(
            &mut s,
            ogre,
            Modification::ModifyPt {
                power: delta,
                toughness: 0,
            },
        );
        settle(&mut s);
        let power = 2 * (2 + delta);
        assert_eq!(s.game().characteristics(ogre).pt(), (power, 2));
        assert_eq!(
            s.game().state.zones.zone_of(now(&s, spell)),
            Some(ZoneType::Graveyard)
        );
        s.set_player_mana_pool(0, "4RR");
        cast_card(&mut s, spell);
        s.choose_target_named("Gray Ogre");
        settle(&mut s);
        assert_eq!(s.game().characteristics(ogre).pt(), (2 * power, 2));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        assert_eq!(
            s.game().state.zones.zone_of(now(&s, spell)),
            Some(ZoneType::Exile)
        );
    }
}

#[test]
fn power_batch_bite_down_binds_both_targets_and_damages_loyalty() {
    use crate::support::{
        game::{force_new_card_in_hand, set_counters},
        inspect::counters_of,
        play::cast_card,
    };
    use managym::state::{
        card::{CardType, CardTypes},
        permanent::CounterKind,
        player::ManaPayment,
    };
    let mut s = five_color_setup(&["Bite Down", "Gray Ogre", "Hill Giant"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let walker = s.force_permanent_on_battlefield(1, "Hill Giant");
    let card = card_of(&s, walker);
    s.game_mut().state[card].types = CardTypes::new([CardType::Planeswalker]);
    set_counters(&mut s, walker, CounterKind::Loyalty, 5);
    let spell = force_new_card_in_hand(&mut s, 0, "Bite Down");
    cast_card(&mut s, spell);
    s.choose_target_named("Gray Ogre");
    s.choose_target_named("Hill Giant");
    let pending = s.game().clone();
    for blink in [None, Some(source), Some(walker)] {
        *s.game_mut() = pending.clone();
        set_counters(&mut s, source, CounterKind::PlusOnePlusOne, 1);
        let mut victim = walker;
        if let Some(id) = blink {
            let card = card_of(&s, id);
            let moved = move_card(&mut s, card, ZoneType::Exile);
            let returned = move_card(&mut s, moved, ZoneType::Battlefield);
            if id == walker {
                victim = permanent_of(&s, returned).unwrap();
                set_counters(&mut s, victim, CounterKind::Loyalty, 5);
            }
        }
        settle(&mut s);
        assert_eq!(
            counters_of(&s, victim, CounterKind::Loyalty),
            if blink.is_some() { 5 } else { 2 }
        );
    }
}

#[test]
fn power_batch_once_each_turn_checks_each_attacker_before_spending_the_grant() {
    use crate::support::{
        game::{clear_summoning_sickness, set_counters, until_end_of_turn},
        inspect::counters,
    };
    use managym::{
        flow::turn::AdditionalCombat,
        state::{
            ability::{
                Ability, Condition, Effect, ObjectSelector, Per, TriggerCondition, TriggerSubject,
            },
            effect::Modification,
            permanent::CounterKind,
            predicate::CardPredicate,
        },
    };
    let mut s = five_color_setup(&["Katara, the Fearless", "Savannah Lions", "Serra Angel"]);
    let katara = s.force_permanent_on_battlefield(0, "Katara, the Fearless");
    let lions = s.force_permanent_on_battlefield(0, "Savannah Lions");
    let angel = s.force_permanent_on_battlefield(0, "Serra Angel");
    for id in [lions, angel] {
        clear_summoning_sickness(&mut s, id);
    }
    set_counters(&mut s, angel, CounterKind::PlusOnePlusOne, 1);
    let listener = Ability::Triggered {
        condition: TriggerCondition::ActiveIf {
            active_if: Condition::YourTurn,
            condition: Box::new(TriggerCondition::OnceEachTurn {
                condition: Box::new(TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate::creature()),
                    Per::Object,
                )),
            }),
        },
        intervening_if: Some(Condition::TriggeringHadCounters),
        effects: vec![Effect::put_counters(
            ObjectSelector::Triggering,
            CounterKind::PlusOnePlusOne,
            1,
        )],
    };
    until_end_of_turn(
        &mut s,
        katara,
        Modification::GrantTriggered(Box::new(listener)),
    );
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.declare_attack();
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
    assert_eq!((counters(&s, lions), counters(&s, angel)), (0, 2));
    s.game_mut()
        .state
        .turn
        .additional_combats
        .push(AdditionalCombat { attackers: None });
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, angel);
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn power_batch_wrapped_hits_keep_the_creature_and_damage_of_each_hit() {
    use crate::support::{
        game::{clear_summoning_sickness, until_end_of_turn},
        inspect::counters,
    };
    use managym::state::{
        ability::{
            Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, Per, TriggerCondition,
            TriggerSubject,
        },
        effect::Modification,
        permanent::CounterKind,
        predicate::CardPredicate,
    };
    let mut s = five_color_setup(&["Katara, the Fearless", "Gray Ogre", "Hill Giant"]);
    let katara = s.force_permanent_on_battlefield(0, "Katara, the Fearless");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    for id in [ogre, giant] {
        clear_summoning_sickness(&mut s, id);
    }
    until_end_of_turn(
        &mut s,
        katara,
        Modification::GrantTriggered(Box::new(Ability::Triggered {
            condition: TriggerCondition::ActiveIf {
                active_if: Condition::YourTurn,
                condition: Box::new(TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::you_control(CardPredicate::creature()),
                    to: Actor::Opponent,
                    per: Per::Object,
                }),
            },
            intervening_if: None,
            effects: vec![Effect::put_counters(
                ObjectSelector::Triggering,
                CounterKind::PlusOnePlusOne,
                EffectValue::TriggeringAmount,
            )],
        })),
    );
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!((counters(&s, ogre), counters(&s, giant)), (4, 6));
}

#[test]
fn power_batch_suki_triggers_once_under_kataras_doubler() {
    let mut s = five_color_setup(&[
        "Katara, the Fearless",
        "Suki, Courageous Rescuer",
        "Gray Ogre",
    ]);
    s.force_permanent_on_battlefield(0, "Katara, the Fearless");
    s.force_permanent_on_battlefield(0, "Suki, Courageous Rescuer");
    let ogres = [0, 1].map(|_| s.force_permanent_on_battlefield(0, "Gray Ogre"));
    for ogre in ogres {
        let object = card_of(&s, ogre);
        move_card(&mut s, object, ZoneType::Graveyard);
        s.pass_priority();
        settle(&mut s);
        assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
    }
}

#[test]
fn power_batch_printed_intervening_condition_publishes_pending_power() {
    use crate::support::game::{clear_summoning_sickness, set_counters};
    use managym::{
        agent::observation::Observation,
        state::{
            ability::{
                Ability, Aggregate, CompareOp, Condition, Effect, ObjectProperty, ObjectSelector,
                Per, TriggerCondition, TriggerSubject,
            },
            permanent::CounterKind,
        },
    };
    let mut s = five_color_setup(&["Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = card_of(&s, ogre);
    s.game_mut().state[card].abilities = vec![Ability::Triggered {
        condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
        intervening_if: Some(Condition::compare(
            ObjectSelector::This.property(ObjectProperty::Power, Aggregate::Sum),
            CompareOp::Ge,
            3,
        )),
        effects: vec![Effect::gain_life(1)],
    }];
    clear_summoning_sickness(&mut s, ogre);
    set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 1);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, ogre);
    let life = s.game().state.players[0].life;
    for (count, power) in [(1, 3), (0, 2)] {
        set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, count);
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = Observation::for_player(s.game(), viewer);
            let stack = &obs.stack_objects[0];
            assert_eq!(stack.source_power, Some(power));
            assert!(stack
                .description
                .as_ref()
                .unwrap()
                .contains(&format!("source power {power}")));
        }
    }
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, life);
}

#[test]
fn power_batch_grantors_with_one_timestamp_have_separate_turn_limits() {
    use crate::support::game::clear_summoning_sickness;
    use managym::{
        agent::observation::Observation,
        state::{
            ability::{Ability, Effect, Per, TriggerCondition, TriggerSubject},
            card::{StaticAbility, StaticScope, StaticTriggeredGrant},
            effect::{EffectDuration, Modification},
            predicate::CardPredicate,
        },
    };
    let mut s = five_color_setup(&["Katara, the Fearless", "Gray Ogre"]);
    let katara = s.force_permanent_on_battlefield(0, "Katara, the Fearless");
    clear_summoning_sickness(&mut s, katara);
    let grantors = [0, 1].map(|_| {
        let id = s.force_permanent_on_battlefield(0, "Gray Ogre");
        s.game().affected_permanent(id).unwrap()
    });
    let ability = Ability::Triggered {
        condition: TriggerCondition::OnceEachTurn {
            condition: Box::new(TriggerCondition::attacks(TriggerSubject::This, Per::Object)),
        },
        intervening_if: None,
        effects: vec![Effect::gain_life(1)],
    };
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        grantors,
        Modification::GrantStatic(Box::new(StaticAbility::TriggeredGrant(
            StaticTriggeredGrant {
                scope: StaticScope::YouControl(CardPredicate::subtype("Ally")),
                ability: Box::new(ability),
            },
        ))),
        EffectDuration::EndOfTurn,
    );
    let life = s.game().state.players[0].life;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, katara);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, life + 2);
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let source = obs
        .agent_permanents
        .iter()
        .find(|p| p.id == crate::support::inspect::permanent(&s, katara).id.0 as i32)
        .unwrap();
    assert_eq!(source.once_each_turn_used, 2);
}

/// Create a Food for player 0 and eat it: one gain of 3 life.
fn eat_food(s: &mut crate::support::game::Scenario) {
    use crate::support::play::command;
    use managym::agent::action::Action;
    let food = s
        .game_mut()
        .create_token("token.food", PlayerId(0), false)
        .unwrap();
    if s.game().state.players[0].mana_pool.total() < 2 {
        s.set_player_mana_pool(0, "2");
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    command(
        s,
        |action| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == food),
    );
    spend_colorless(s);
    settle(s);
}

fn surveils(s: &mut crate::support::game::Scenario) -> usize {
    use managym::flow::event::GameEvent;
    s.drain_events()
        .iter()
        .filter(|event| matches!(event, GameEvent::Surveilled { .. }))
        .count()
}

#[test]
fn vanguard_seraph_surveils_on_the_first_gain_of_each_turn_only() {
    let mut s = five_color_setup(&["Vanguard Seraph"]);
    s.force_permanent_on_battlefield(0, "Vanguard Seraph");
    s.drain_events();
    eat_food(&mut s);
    assert_eq!(surveils(&mut s), 1, "first gain this turn");
    eat_food(&mut s);
    assert_eq!(surveils(&mut s), 0, "second gain this turn");

    // A Seraph arriving after the turn's first gain waits for the next turn.
    let mut late = five_color_setup(&["Vanguard Seraph"]);
    eat_food(&mut late);
    late.force_permanent_on_battlefield(0, "Vanguard Seraph");
    late.drain_events();
    eat_food(&mut late);
    assert_eq!(surveils(&mut late), 0, "late entry");
    late.advance_to_active_step(1, StepKind::Main);
    late.advance_to_active_step(0, StepKind::Main);
    late.drain_events();
    eat_food(&mut late);
    assert_eq!(surveils(&mut late), 1, "next turn resets the first gain");
}

#[test]
fn bloodthirsty_conqueror_gains_each_actual_loss_with_its_own_amount() {
    use crate::support::{
        game::force_new_card_in_hand,
        play::{cast_card, choose_mode, target_opponent},
    };
    use managym::state::stack_object::StackObject;

    let pending_amounts = |s: &crate::support::game::Scenario| -> Vec<i32> {
        s.game()
            .state
            .stack_objects
            .iter()
            .filter_map(|object| match object {
                StackObject::TriggeredAbility(trigger) => Some(trigger.triggering_amount),
                _ => None,
            })
            .collect()
    };

    // Life loss (Exsanguinate) and damage (Lightning Bolt) each trigger with
    // the amount lost; both wait together and survive the source leaving.
    let mut s = five_color_setup(&["Bloodthirsty Conqueror", "Exsanguinate", "Lightning Bolt"]);
    let conqueror = s.force_permanent_on_battlefield(0, "Bloodthirsty Conqueror");
    s.set_player_mana_pool(0, "2BB");
    let spell = force_new_card_in_hand(&mut s, 0, "Exsanguinate");
    cast_card(&mut s, spell);
    choose_mode(&mut s, 2);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(pending_amounts(&s), vec![2]);
    s.set_player_mana_pool(0, "R");
    let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
    cast_card(&mut s, bolt);
    target_opponent(&mut s);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(pending_amounts(&s), vec![2, 3]);
    let object = card_of(&s, conqueror);
    move_card(&mut s, object, ZoneType::Hand);
    settle(&mut s);
    s.assert_life(1, 15);
    // Exsanguinate's own gain (2) plus the two triggered gains.
    s.assert_life(0, 20 + 2 + 2 + 3);

    // Loss its total can't take leaves it unchanged: nothing triggers.
    let mut s = five_color_setup(&[
        "Bloodthirsty Conqueror",
        "Exsanguinate",
        "Teferi's Protection",
    ]);
    s.force_permanent_on_battlefield(0, "Bloodthirsty Conqueror");
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "2W");
    let protection = force_new_card_in_hand(&mut s, 1, "Teferi's Protection");
    cast_card(&mut s, protection);
    settle(&mut s);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "3BB");
    let spell = force_new_card_in_hand(&mut s, 0, "Exsanguinate");
    cast_card(&mut s, spell);
    choose_mode(&mut s, 3);
    s.pass_priority();
    s.pass_priority();
    assert!(pending_amounts(&s).is_empty());
    settle(&mut s);
    s.assert_life(1, 20);
    s.assert_life(0, 20);

    // A finite drain still ends the game at zero life.
    let mut s = five_color_setup(&["Bloodthirsty Conqueror", "Lightning Bolt"]);
    s.force_permanent_on_battlefield(0, "Bloodthirsty Conqueror");
    s.game_mut().scenario_set_life(PlayerId(1), 3);
    let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
    cast_card(&mut s, bolt);
    target_opponent(&mut s);
    s.pass_priority();
    s.pass_priority();
    s.assert_game_over();
    s.assert_winner(0);
}

/// A lifelink creature dealing damage to several recipients at once gains
/// life once (CR 119.9; Exemplar of Light ruling), so Exemplar places one
/// counter for Sire of Seven Deaths' trample through two blockers. Two
/// lifelink attackers dealing damage at the same time are two gains.
#[test]
fn exemplar_of_light_counts_one_gain_per_lifelink_source() {
    use crate::support::inspect::counters;
    use managym::agent::action::{Action, ActionSpaceKind};

    let mut s = five_color_setup(&[
        "Exemplar of Light",
        "Sire of Seven Deaths",
        "Savannah Lions",
    ]);
    let exemplar = s.force_permanent_on_battlefield(0, "Exemplar of Light");
    let sire = s.force_permanent_on_battlefield(0, "Sire of Seven Deaths");
    let lions = [
        s.force_permanent_on_battlefield(1, "Savannah Lions"),
        s.force_permanent_on_battlefield(1, "Savannah Lions"),
    ];
    crate::support::game::clear_summoning_sickness(&mut s, sire);
    crate::support::play::pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, sire);
    crate::support::play::pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
    // Menace: both Lions block the Sire.
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| {
                matches!(a, Action::DeclareBlocker { blocker, attacker: Some(at), .. }
                    if lions.contains(blocker) && *at == sire)
            })
            .expect("block the Sire");
        s.step_action(index);
    }
    s.advance_to_active_step(0, StepKind::End);
    s.assert_life(0, 27);
    s.assert_life(1, 15);
    assert_eq!(s.game().state.turn.history[0].life_gains, 1);
    assert_eq!(counters(&s, exemplar), 1);

    // Two lifelink sources in the same damage step are two gains.
    let mut s = five_color_setup(&["Exemplar of Light", "Healer's Hawk"]);
    let exemplar = s.force_permanent_on_battlefield(0, "Exemplar of Light");
    let hawks = [
        s.force_permanent_on_battlefield(0, "Healer's Hawk"),
        s.force_permanent_on_battlefield(0, "Healer's Hawk"),
    ];
    for hawk in hawks {
        crate::support::game::clear_summoning_sickness(&mut s, hawk);
    }
    crate::support::play::pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        match s.action_space().actions.iter().position(|a| {
            matches!(a, Action::DeclareAttacker { permanent, attack: true, .. }
                if hawks.contains(permanent))
        }) {
            Some(index) => s.step_action(index),
            None => s.decline_attack(),
        }
    }
    s.advance_to_active_step(0, StepKind::End);
    s.assert_life(0, 22);
    assert_eq!(s.game().state.turn.history[0].life_gains, 2);
    assert_eq!(counters(&s, exemplar), 2);
}

#[test]
fn exemplar_of_light_draws_once_each_turn_for_counters_on_itself() {
    use crate::support::inspect::{counters, hand};

    let mut s = five_color_setup(&["Exemplar of Light"]);
    let exemplar = s.force_permanent_on_battlefield(0, "Exemplar of Light");
    let before = hand(&s, 0).len();
    eat_food(&mut s);
    assert_eq!(counters(&s, exemplar), 1);
    assert_eq!(hand(&s, 0).len(), before + 1, "first placement draws");
    eat_food(&mut s);
    assert_eq!(counters(&s, exemplar), 2);
    assert_eq!(hand(&s, 0).len(), before + 1, "once each turn");

    // A spell's counter on another creature is not a counter on this one;
    // the same spell on this creature is.
    for (target, draws) in [("Gray Ogre", 0), ("Exemplar of Light", 1)] {
        let mut other = five_color_setup(&["Exemplar of Light", "Gray Ogre", "Felling Blow"]);
        other.force_permanent_on_battlefield(0, "Exemplar of Light");
        other.force_permanent_on_battlefield(0, "Gray Ogre");
        other.force_permanent_on_battlefield(1, "Gray Ogre");
        let before = hand(&other, 0).len();
        let blow = crate::support::game::force_new_card_in_hand(&mut other, 0, "Felling Blow");
        crate::support::play::cast_card(&mut other, blow);
        let mine = other.battlefield_permanents_named(0, target)[0];
        other.choose_target(managym::state::game_object::Target::Permanent(mine));
        let theirs = other.battlefield_permanents_named(1, "Gray Ogre")[0];
        other.choose_target(managym::state::game_object::Target::Permanent(theirs));
        settle(&mut other);
        assert_eq!(hand(&other, 0).len(), before + draws, "{target}");
    }
    // The next turn's first placement draws again.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let before = hand(&s, 0).len();
    eat_food(&mut s);
    assert_eq!(counters(&s, exemplar), 3);
    assert_eq!(hand(&s, 0).len(), before + 1);
}

/// Tragic Banshee reads whether a creature died this turn as its ability
/// resolves (CR 608.2h): a death between trigger and resolution counts.
#[test]
fn tragic_banshee_reads_death_history_as_it_resolves() {
    use crate::support::play::{bolt, cast, resolve_top};
    use managym::state::player::ManaPayment;
    for (before_entry, between, dies) in [
        (false, false, false),
        (true, false, true),
        (false, true, true),
    ] {
        let mut s = five_color_setup(&[
            "Tragic Banshee",
            "Hill Giant",
            "Savannah Lions",
            "Lightning Bolt",
        ]);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
        let lions = s.force_permanent_on_battlefield(0, "Savannah Lions");
        if before_entry {
            bolt(&mut s, lions);
            settle(&mut s);
        }
        cast(&mut s, "Tragic Banshee");
        // A card in hand keeps player 0's priority with the trigger stacked.
        crate::support::game::force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
        resolve_top(&mut s);
        // The only opposing creature is the target.
        assert!(s.choose_target(managym::state::game_object::Target::Permanent(giant)));
        assert_eq!(s.game().state.stack_objects.len(), 1);
        if between {
            bolt(&mut s, lions);
            assert_eq!(s.game().state.stack_objects.len(), 1, "Bolt resolved first");
        }
        settle(&mut s);
        assert_eq!(
            s.game().state.permanents[giant].is_none(),
            dies,
            "{before_entry} {between}"
        );
        if !dies {
            assert_eq!(s.game().characteristics(giant).pt(), (2, 2));
        }
    }
}

/// Vengeful Bloodwitch triggers once for each creature its controller
/// controlled that died, itself and a land copying a creature included, in one
/// simultaneous event; the opponent's creature does not trigger it.
#[test]
fn vengeful_bloodwitch_drains_for_each_of_its_controllers_simultaneous_deaths() {
    use managym::state::player::ManaPayment;
    let mut s = five_color_setup(&[
        "Vengeful Bloodwitch",
        "Gray Ogre",
        "Savannah Lions",
        "Fumigate",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.force_permanent_on_battlefield(0, "Vengeful Bloodwitch");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    // A noncreature card copying a creature dies as that creature.
    let island = s.force_permanent_on_battlefield(0, "Island");
    let original = s.game().state[card_of(&s, ogre)].clone();
    let copy = card_of(&s, island);
    s.game_mut().state[copy].become_copy_of(&original, &[]);
    s.force_permanent_on_battlefield(1, "Savannah Lions");
    crate::support::play::cast(&mut s, "Fumigate");
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 3,
        "three Bloodwitch triggers".into(),
    );
    settle(&mut s);
    // Fumigate gains one for each of four creatures; each drain one more.
    s.assert_life(0, 27);
    s.assert_life(1, 17);
}

/// Needletooth Pack's end-step trigger needs a creature to have died this
/// turn, on either side; a death on an earlier turn does not count.
#[test]
fn needletooth_pack_counts_this_turns_deaths_at_its_end_step() {
    use crate::support::{inspect::counters, play::bolt};
    for (when, counted) in [("none", false), ("opponent", true), ("earlier", false)] {
        let mut s = five_color_setup(&["Needletooth Pack", "Savannah Lions", "Lightning Bolt"]);
        let pack = s.force_permanent_on_battlefield(0, "Needletooth Pack");
        let lions = s.force_permanent_on_battlefield(1, "Savannah Lions");
        if when == "earlier" {
            // A death on the opponent's turn, before player 0's next turn.
            s.advance_to_active_step(1, StepKind::Main);
            crate::support::game::force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
            s.game_mut().scenario_refresh_priority().unwrap();
            s.advance_until(
                |s| s.action_space().player == Some(PlayerId(0)),
                "priority".into(),
            );
            bolt(&mut s, lions);
            settle(&mut s);
            s.advance_to_active_step(0, StepKind::Main);
        } else if when == "opponent" {
            bolt(&mut s, lions);
            settle(&mut s);
        }
        s.advance_to_active_step(0, StepKind::End);
        settle(&mut s);
        assert_eq!(counters(&s, pack), if counted { 2 } else { 0 }, "{when}");
    }
}

/// Slumbering Cerberus stays tapped through its controller's untap step and
/// untaps at any end step of a turn in which a creature died.
#[test]
fn slumbering_cerberus_untaps_only_at_an_end_step_after_a_death() {
    use crate::support::{inspect::tapped, play::bolt};
    let mut s = five_color_setup(&["Slumbering Cerberus", "Savannah Lions", "Lightning Bolt"]);
    let cerberus = s.force_permanent_on_battlefield(0, "Slumbering Cerberus");
    s.game_mut().state.permanents[cerberus]
        .as_mut()
        .unwrap()
        .tapped = true;
    s.advance_to_active_step(0, StepKind::End);
    settle(&mut s);
    assert!(tapped(&s, cerberus), "no death this turn");
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(tapped(&s, cerberus), "its untap step leaves it tapped");
    // A death on the opponent's turn untaps it at that turn's end step.
    s.advance_to_active_step(1, StepKind::Main);
    let lions = s.force_permanent_on_battlefield(1, "Savannah Lions");
    s.game_mut().scenario_refresh_priority().unwrap();
    s.advance_until(
        |s| s.action_space().player == Some(PlayerId(0)),
        "priority".into(),
    );
    bolt(&mut s, lions);
    settle(&mut s);
    assert!(tapped(&s, cerberus));
    s.advance_to_active_step(1, StepKind::End);
    settle(&mut s);
    assert!(!tapped(&s, cerberus), "each end step");
}

/// Wardens of the Cycle chooses its mode as the trigger is put on the stack.
/// The stacked ability keeps its printed identity: both seats see the chosen
/// mode in the stack row and continuation frame, and it resolves only that
/// mode.
#[test]
fn wardens_of_the_cycle_keeps_its_chosen_mode_on_the_stack() {
    use crate::support::{
        inspect::hand,
        play::{bolt, choose_mode},
    };
    use managym::agent::{action::ActionSpaceKind, observation::Observation};
    let mut s = five_color_setup(&["Wardens of the Cycle", "Savannah Lions", "Lightning Bolt"]);
    let wardens = s.force_permanent_on_battlefield(0, "Wardens of the Cycle");
    let key = s.game().state[card_of(&s, wardens)].definition_id;
    let lions = s.force_permanent_on_battlefield(1, "Savannah Lions");
    // No death: no trigger.
    let quiet = s.game().clone();
    s.advance_to_active_step(0, StepKind::End);
    assert!(s.game().state.stack_objects.is_empty());
    *s.game_mut() = quiet;
    bolt(&mut s, lions);
    settle(&mut s);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::Modal,
        "Wardens' mode".into(),
    );
    let modal = s.game().clone();
    for mode in [0, 1] {
        *s.game_mut() = modal.clone();
        choose_mode(&mut s, mode);
        assert_eq!(s.game().state.stack_objects.len(), 1);
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = Observation::for_player(s.game(), viewer);
            let row = &obs.stack_objects[0];
            assert_eq!(row.chosen_mode, Some(mode));
            assert_eq!(row.ability_index, Some(0));
            assert_eq!(row.source_card_registry_key, key.0 as i32);
            assert!(row
                .description
                .as_deref()
                .unwrap()
                .contains(&format!("chosen mode {}", mode + 1)));
            let frame = obs.continuation.iter().find(|f| f.kind == 3).unwrap();
            assert_eq!(frame.mode, Some(mode));
        }
        let before = hand(&s, 0).len();
        settle(&mut s);
        let (life, drawn) = if mode == 0 { (22, 0) } else { (19, 1) };
        s.assert_life(0, life);
        assert_eq!(hand(&s, 0).len(), before + drawn);
    }
}

/// CR 603.4: a printed modal trigger whose intervening "if" stops holding
/// before it resolves does nothing, though its mode was chosen.
#[test]
fn modal_trigger_rechecks_its_intervening_if_on_resolution() {
    use crate::support::{
        game::inject_ability,
        play::{bolt, choose_mode},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{
            ability::{Ability, Actor, Condition, Effect, TriggerCondition, TurnStep},
            predicate::CardPredicate,
        },
    };
    let mut s = five_color_setup(&["Gray Ogre", "Savannah Lions", "Lightning Bolt"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let lions = s.force_permanent_on_battlefield(0, "Savannah Lions");
    let ogre_card = card_of(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
            intervening_if: Some(Condition::ControlsAtLeast {
                distinct_names: false,
                count: 2,
                predicate: CardPredicate::creature(),
            }),
            effects: vec![Effect::TriggerModes {
                modes: vec![vec![Effect::gain_life(2)], vec![Effect::gain_life(5)]],
                up_to: None,
                unchosen: false,
            }],
        },
    );
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::Modal,
        "the injected mode".into(),
    );
    choose_mode(&mut s, 1);
    let pending = s.game().clone();
    settle(&mut s);
    s.assert_life(0, 25);
    *s.game_mut() = pending;
    bolt(&mut s, lions);
    settle(&mut s);
    s.assert_life(0, 20);
}

/// Scrawling Crawler's upkeep trigger has each player draw in APNAP order
/// (CR 101.4), and every opponent draw, including those inside that trigger
/// and the opponent's own draw step, costs them 1 life per Crawler. Two
/// Crawlers each queue a drain for each opponent draw.
#[test]
fn scrawling_crawler_draws_for_each_player_in_order_and_drains_opponent_draws() {
    use crate::support::play::settle;
    use managym::flow::event::GameEvent;
    for crawlers in [1_usize, 2] {
        let mut s = five_color_setup(&["Scrawling Crawler"]);
        for _ in 0..crawlers {
            s.force_permanent_on_battlefield(0, "Scrawling Crawler");
        }
        s.advance_to_active_step(1, StepKind::Main);
        s.assert_life(1, 20 - crawlers as i32);
        s.advance_to_active_step(0, StepKind::Upkeep);
        let hands = [
            s.zone_size(0, ZoneType::Hand),
            s.zone_size(1, ZoneType::Hand),
        ];
        let life = s.life(1);
        s.drain_events();
        settle(&mut s);
        let drawers: Vec<_> = s
            .drain_events()
            .into_iter()
            .filter_map(|event| match event {
                GameEvent::CardDrawn { player, .. } => Some(player),
                _ => None,
            })
            .collect();
        assert_eq!(drawers, [PlayerId(0), PlayerId(1)].repeat(crawlers));
        assert_eq!(s.zone_size(0, ZoneType::Hand), hands[0] + crawlers);
        assert_eq!(s.zone_size(1, ZoneType::Hand), hands[1] + crawlers);
        s.assert_life(1, life - (crawlers * crawlers) as i32);
        s.assert_life(0, 20);
    }
}

/// Firespitter Whelp pings each opponent once for each noncreature or Dragon
/// spell its controller casts: an instant and a Dragon creature, but not
/// another creature.
#[test]
fn firespitter_whelp_pings_once_per_noncreature_or_dragon_spell() {
    use crate::support::{
        game::force_new_card_in_hand,
        play::{cast_card, settle},
    };
    use managym::state::{game_object::Target, player::ManaPayment};
    let mut s = five_color_setup(&[
        "Firespitter Whelp",
        "Lightning Bolt",
        "Gray Ogre",
        "Sol Ring",
    ]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    s.force_permanent_on_battlefield(0, "Firespitter Whelp");
    let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
    cast_card(&mut s, bolt);
    assert!(s.choose_target(Target::Player(PlayerId(0))));
    settle(&mut s);
    s.assert_life(1, 19);
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    cast_card(&mut s, ogre);
    settle(&mut s);
    s.assert_life(1, 19);
    // A Dragon creature spell matches one of the two alternatives once.
    let whelp = force_new_card_in_hand(&mut s, 0, "Firespitter Whelp");
    cast_card(&mut s, whelp);
    assert_eq!(s.game().state.stack_objects.len(), 2, "one trigger");
    settle(&mut s);
    s.assert_life(1, 18);
}

/// Archmage of Runes takes {1} from instants and sorceries and draws for
/// each one cast; a creature spell pays full price and draws nothing.
#[test]
fn archmage_of_runes_discounts_and_draws_for_instants_and_sorceries() {
    use crate::support::{
        game::force_new_card_in_hand,
        inspect::{hand, top_spell},
        play::{cast_card, settle},
    };
    use managym::state::player::ManaPayment;
    let mut s = five_color_setup(&["Archmage of Runes", "Fumigate", "Gray Ogre"]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    s.force_permanent_on_battlefield(0, "Archmage of Runes");
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    cast_card(&mut s, ogre);
    assert_eq!(top_spell(&s).spent.total(), 3);
    settle(&mut s);
    assert!(hand(&s, 0).is_empty());
    let fumigate = force_new_card_in_hand(&mut s, 0, "Fumigate");
    cast_card(&mut s, fumigate);
    assert_eq!(top_spell(&s).spent.total(), 4);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 1);
}

/// Goblin Negotiation makes a Goblin for each excess damage its own damage
/// dealt (CR 120.4a): X = 5 at a 2/2 makes three, X = 2 none. A later damage
/// instruction with no excess (damage to a player) leaves none, never the
/// earlier instruction's excess.
#[test]
fn goblin_negotiation_creates_a_goblin_for_each_excess_damage() {
    use crate::support::{
        game::force_new_card_in_hand,
        play::{cast_card, choose_mode},
    };
    use managym::state::{
        ability::{Effect, ObjectSelector},
        game_object::Target,
    };
    for (x, later_player_damage, goblins) in [(5, false, 3), (2, false, 0), (5, true, 0)] {
        let mut s = five_color_setup(&["Goblin Negotiation", "Gray Ogre"]);
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        s.set_player_mana_pool(0, &format!("{x}RR"));
        let spell = force_new_card_in_hand(&mut s, 0, "Goblin Negotiation");
        if later_player_damage {
            s.game_mut().state[spell]
                .spell_effects
                .insert(1, Effect::deal_damage(1, ObjectSelector::You));
        }
        cast_card(&mut s, spell);
        choose_mode(&mut s, x);
        assert!(s.choose_target(Target::Permanent(ogre)));
        settle(&mut s);
        let made = s.battlefield_permanents_named(0, "Goblin");
        assert_eq!(made.len(), goblins, "X = {x}");
        assert!(s.game().state.permanents[ogre].is_none(), "X = {x}");
        if later_player_damage {
            s.assert_life(0, 19);
        }
    }
}

/// Niv-Mizzet, Visionary draws the noncombat damage a source its controller
/// controls deals to an opponent: a resolving spell, a source sacrificed to
/// pay for its ability (its last controller), and not a spell the opponent
/// now controls, damage to its controller or a creature, or combat damage.
#[test]
fn niv_mizzet_draws_noncombat_damage_its_controllers_sources_deal_to_an_opponent() {
    use crate::support::{
        game::{clear_summoning_sickness, force_new_card_in_hand, Scenario},
        inspect::hand,
        play::{cast_card, pass_until},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{game_object::Target, stack_object::StackObject},
    };
    let mut s = five_color_setup(&[
        "Niv-Mizzet, Visionary",
        "Lightning Bolt",
        "Fanatical Firebrand",
        "Gray Ogre",
    ]);
    let niv = s.force_permanent_on_battlefield(0, "Niv-Mizzet, Visionary");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let cards = |s: &Scenario| hand(s, 0).len();
    let bolt = |s: &mut Scenario, target: Target, stolen: bool| {
        s.set_player_mana_pool(0, "R");
        let bolt = force_new_card_in_hand(s, 0, "Lightning Bolt");
        cast_card(s, bolt);
        assert!(s.choose_target(target));
        if stolen {
            for object in s.game_mut().state.stack_objects.iter_mut() {
                if let StackObject::Spell(spell) = object {
                    spell.controller = PlayerId(1);
                }
            }
        }
        let before = cards(s);
        settle(s);
        cards(s) - before
    };
    assert_eq!(bolt(&mut s, Target::Player(PlayerId(1)), false), 3);
    s.assert_life(1, 17);
    assert_eq!(bolt(&mut s, Target::Player(PlayerId(1)), true), 0, "stolen");
    s.assert_life(1, 14);
    assert_eq!(
        bolt(&mut s, Target::Player(PlayerId(0)), false),
        0,
        "to you"
    );
    assert_eq!(
        bolt(&mut s, Target::Permanent(ogre), false),
        0,
        "to a creature"
    );

    let firebrand = s.force_permanent_on_battlefield(0, "Fanatical Firebrand");
    let before = cards(&s);
    activate(&mut s, firebrand, 0);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    assert!(s.game().state.permanents[firebrand].is_none(), "sacrificed");
    settle(&mut s);
    assert_eq!(cards(&s), before + 1, "departed source");
    s.assert_life(1, 13);

    clear_summoning_sickness(&mut s, niv);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, niv);
    let before = cards(&s);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    s.assert_life(1, 8);
    assert_eq!(cards(&s), before, "combat damage");
    assert!(s
        .game()
        .characteristics(niv)
        .static_abilities
        .contains(&managym::state::card::StaticAbility::NoMaximumHandSize));
}

/// Hare Apparent counts the other creatures its controller controls by name
/// as its trigger resolves: two others make two Rabbits, the opponent's copy
/// and the Rabbits themselves never count, and a fourth copy sees three.
#[test]
fn hare_apparent_makes_a_rabbit_for_each_other_hare_apparent_you_control() {
    use crate::support::play::enter_and_settle;
    let mut s = five_color_setup(&["Hare Apparent"]);
    s.force_permanent_on_battlefield(0, "Hare Apparent");
    s.force_permanent_on_battlefield(0, "Hare Apparent");
    s.force_permanent_on_battlefield(1, "Hare Apparent");
    enter_and_settle(&mut s, "Hare Apparent");
    assert_eq!(s.battlefield_permanents_named(0, "Rabbit").len(), 2);
    assert!(s.battlefield_permanents_named(1, "Rabbit").is_empty());
    enter_and_settle(&mut s, "Hare Apparent");
    assert_eq!(s.battlefield_permanents_named(0, "Rabbit").len(), 5);
}

/// Ingenious Leonin targets another attacking creature its controller
/// controls: itself, a creature that is not attacking and the opponent's
/// creatures are not offered. Every target gets a counter; only a Cat gains
/// first strike.
#[test]
fn ingenious_leonin_grows_another_attacker_and_gives_a_cat_first_strike() {
    use crate::support::{
        game::clear_summoning_sickness,
        inspect::{counters, keywords, offered_targets},
        play::{activate, pass_until},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{card::Keyword, game_object::Target},
    };
    let mut s = five_color_setup(&["Ingenious Leonin", "Leonin Vanguard", "Gray Ogre"]);
    let leonin = s.force_permanent_on_battlefield(0, "Ingenious Leonin");
    let cat = s.force_permanent_on_battlefield(0, "Leonin Vanguard");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let home = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for id in [leonin, cat, ogre, home] {
        clear_summoning_sickness(&mut s, id);
    }
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let index = s.action_space().actions.iter().position(|action| {
            matches!(action, Action::DeclareAttacker { permanent, attack, .. }
                if *attack == [leonin, cat, ogre].contains(permanent))
        });
        s.step_action(index.unwrap());
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    for target in [cat, ogre] {
        s.set_player_mana_pool(0, "3W");
        activate(&mut s, leonin, 0);
        let offered = offered_targets(&s);
        assert!(offered.contains(&Target::Permanent(cat)));
        assert!(offered.contains(&Target::Permanent(ogre)));
        for absent in [leonin, home, theirs] {
            assert!(!offered.contains(&Target::Permanent(absent)));
        }
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
    }
    assert_eq!((counters(&s, cat), counters(&s, ogre)), (1, 1));
    assert!(keywords(&s, cat).has(Keyword::FirstStrike));
    assert!(!keywords(&s, ogre).has(Keyword::FirstStrike));
}
