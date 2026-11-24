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
    use crate::support::play::{cast, choose_mode, settle};
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Seeker's Folly", "Gray Ogre"]);
    s.force_card_in_hand(1, "Forest");
    s.force_card_in_hand(1, "Island");
    let creature = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Seeker's Folly");
    choose_mode(&mut s, 0);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(s.zone_size(1, ZoneType::Hand), 0);
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
    use crate::support::inspect::{offered_targets, tapped};
    use crate::support::play::{cast, pass_until};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    use managym::state::mana::Color;
    let mut s = five_color_setup(&["Faebloom Trick", "Gray Ogre"]);
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
    use crate::support::inspect::{counters, library_card, offered_targets};
    use crate::support::play::{choose, pass_until, settle};
    use managym::agent::action::{Action, ActionSpaceKind};
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
    let savior = library_card(&s, 0, "Felidar Savior");
    move_card(&mut s, savior, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
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
