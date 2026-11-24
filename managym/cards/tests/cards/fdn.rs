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
    s.game_mut()
        .state
        .cast_permissions
        .grant(PlayGrant::new(land, PlayerId(0)));
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
