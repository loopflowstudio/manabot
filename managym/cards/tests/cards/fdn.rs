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
                        row[fact::KIND] == FactKind::TriggeringDamage as u8 as f32
                            && row[fact::ID_LOW] == (index + 1) as f32
                    })
                    .unwrap();
                assert_eq!(row[fact::VALUE], frame.triggering_damage as f32);
                hits.push((source, frame.triggering_damage));
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
        assert!(!s.game().state.turn.creature_died_this_turn);
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
