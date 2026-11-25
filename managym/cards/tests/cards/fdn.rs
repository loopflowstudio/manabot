//! Foundations behavior through ordinary activation, payment and combat.
use crate::support::game::{five_color_setup, move_card, until_end_of_turn, Scenario};
use crate::support::inspect::{card_of, keywords, observed, permanent_named, permanent_of};
use crate::support::play::{
    activate, attack_with, cast, enter_and_settle, settle, spend_colorless,
};
use managym::flow::priority::MandatoryLoopWatch;
use managym::flow::turn::StepKind;
use managym::state::card::Keyword;
use managym::state::effect::Modification;
use managym::state::game_object::{ObjectId, PlayerId, Target};
use managym::state::permanent::CounterKind;
use managym::state::zone::ZoneType;

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
        |s| s.action_space().kind == ActionSpaceKind::OrderTriggers,
        "two Hatcher triggers to order".into(),
    );
    let check_labels = |labels: Vec<&str>| {
        for amount in [1, 3] {
            let expected = format!("Put that many ({amount}) incubation counters");
            assert_eq!(
                labels
                    .iter()
                    .filter(|label| label.contains(&expected))
                    .count(),
                1,
                "{labels:?}"
            );
        }
    };
    let offers = crate::support::inspect::offer_labels(&s);
    check_labels(offers.iter().map(String::as_str).collect());
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
            check_labels(
                observation
                    .stack_objects
                    .iter()
                    .filter_map(|item| item.description.as_deref())
                    .collect(),
            );
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
        .create_token("token.cat_1_1_w", PlayerId(0), false)
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
    use managym::state::game_object::Target;
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

pub(crate) fn assert_raid(s: &crate::support::game::Scenario, attacker: Option<usize>) {
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
    let chosen: Vec<_> = s
        .game()
        .state
        .combat
        .as_ref()
        .unwrap()
        .attackers()
        .collect();
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
                    to: Some(Actor::Opponent),
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
    // Trample: lethal damage (1) to each Lion, the other five to the player.
    for _ in lions {
        crate::support::play::pass_until(&mut s, ActionSpaceKind::Modal);
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::ChooseMode { mode: 1, .. }))
            .expect("assign one damage to a Lion");
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
            matches!(a, Action::DeclareAttacker { permanent, attack: Some(_), .. }
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
                min: 0,
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
                if attack.is_some() == [leonin, cat, ogre].contains(permanent))
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

/// Fill `player`'s graveyard with Plains until it holds `count` cards.
fn graveyard_of(s: &mut crate::support::game::Scenario, player: usize, count: usize) {
    use crate::support::game::plant_in_graveyard;
    while s
        .game()
        .state
        .zones
        .size(ZoneType::Graveyard, PlayerId(player))
        < count
    {
        plant_in_graveyard(s, player, "Plains");
    }
}

/// Kiora loots two as it enters. Attacking with seven cards in its
/// controller's graveyard offers Scion of the Deep, a legendary 8/8; with
/// six it does not trigger.
#[test]
fn kiora_loots_and_offers_the_scion_at_threshold() {
    use crate::support::{game::clear_summoning_sickness, play::pass_until};
    use managym::agent::action::{ActionSpaceKind, ActionType};
    for cards in [6, 7] {
        let mut s = five_color_setup(&["Kiora, the Rising Tide"]);
        let kiora = enter_and_settle(&mut s, "Kiora, the Rising Tide");
        let kiora = permanent_of(&s, kiora).unwrap();
        assert!(s
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Hand, PlayerId(0))
            .is_empty());
        assert_eq!(
            s.game().state.zones.size(ZoneType::Graveyard, PlayerId(0)),
            2
        );
        graveyard_of(&mut s, 0, cards);
        clear_summoning_sickness(&mut s, kiora);
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, kiora);
        if cards == 7 {
            pass_until(&mut s, ActionSpaceKind::PayOrNot);
            assert!(s.take_action_by_type(ActionType::PayCost));
        }
        settle(&mut s);
        let scions = s.battlefield_permanents_named(0, "Scion of the Deep");
        assert_eq!(scions.len(), usize::from(cards == 7), "{cards} cards");
        if let Some(scion) = scions.first() {
            let view = s.game().characteristics(*scion);
            assert_eq!(view.pt(), (8, 8));
            assert!(view.is_legendary);
        }
    }
}

/// Crypt Feaster's threshold is an intervening "if" (CR 603.4): six cards
/// never trigger it, and a graveyard that drops below seven before the
/// trigger resolves gives no bonus.
#[test]
fn crypt_feaster_checks_threshold_on_attack_and_resolution() {
    use crate::support::{game::clear_summoning_sickness, play::pass_until};
    use managym::agent::action::ActionSpaceKind;
    for (cards, drop_before_resolution, power) in [(6, false, 3), (7, false, 5), (7, true, 3)] {
        let mut s = five_color_setup(&["Crypt Feaster"]);
        let feaster = s.force_permanent_on_battlefield(0, "Crypt Feaster");
        clear_summoning_sickness(&mut s, feaster);
        graveyard_of(&mut s, 0, cards);
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, feaster);
        if drop_before_resolution {
            assert!(!s.game().state.stack_objects.is_empty());
            let card = s
                .game()
                .state
                .zones
                .zone_objects(ZoneType::Graveyard, PlayerId(0))[0];
            move_card(&mut s, card, ZoneType::Exile);
        }
        settle(&mut s);
        assert_eq!(
            s.game().characteristics(feaster).pt(),
            (power, 4),
            "{cards} cards, dropped: {drop_before_resolution}"
        );
    }
}

/// Revenge of the Rats makes a tapped Rat for each creature card in its
/// caster's graveyard, and flashes back from there, then is exiled.
#[test]
fn revenge_of_the_rats_counts_creature_cards_and_flashes_back() {
    use crate::support::{
        game::plant_in_graveyard,
        inspect::{tapped, zone_of},
        play::{cast, cast_card},
    };
    let mut s = five_color_setup(&["Revenge of the Rats", "Gray Ogre"]);
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    plant_in_graveyard(&mut s, 0, "Plains");
    plant_in_graveyard(&mut s, 1, "Gray Ogre");
    let spell = cast(&mut s, "Revenge of the Rats");
    settle(&mut s);
    let rats = s.battlefield_permanents_named(0, "Rat");
    assert_eq!(rats.len(), 2);
    assert!(rats.iter().all(|rat| tapped(&s, *rat)));
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Graveyard));
    cast_card(&mut s, spell);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Rat").len(), 4);
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Exile));
}

#[test]
fn ravenous_amulet_drains_by_the_soul_counters_it_had_when_sacrificed() {
    use crate::support::{game::set_counters, inspect::counters_of, play::select_card};
    use managym::{agent::observation::Observation, state::permanent::CounterKind::Soul};

    let mut s = five_color_setup(&["Ravenous Amulet", "Gray Ogre"]);
    let amulet = s.force_permanent_on_battlefield(0, "Ravenous Amulet");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let hand = s.zone_size(0, ZoneType::Hand);
    s.set_player_mana_pool(0, "1");
    activate(&mut s, amulet, 0);
    let ogre_object = card_of(&s, ogre);
    select_card(&mut s, ogre_object);
    spend_colorless(&mut s);
    settle(&mut s);
    assert_eq!(permanent_of(&s, ogre_object), None);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
    assert_eq!(counters_of(&s, amulet, Soul), 1);

    set_counters(&mut s, amulet, Soul, 3);
    s.game_mut().state.permanents[amulet]
        .as_mut()
        .unwrap()
        .tapped = false;
    let card = card_of(&s, amulet);
    let life = s.life(1);
    s.set_player_mana_pool(0, "4");
    activate(&mut s, amulet, 1);
    spend_colorless(&mut s);
    assert_eq!(permanent_of(&s, card), None, "sacrificed as a cost");
    for viewer in [PlayerId(0), PlayerId(1)] {
        let observation = Observation::for_player(s.game(), viewer);
        let frame = observation
            .continuation
            .iter()
            .find(|frame| frame.kind == 2)
            .expect("stacked drain");
        assert!(
            !frame.effect_execution.is_empty(),
            "the stack keeps its source's counters"
        );
    }
    let capture = s.game().capture_decision(PlayerId(0), 0).unwrap();
    assert!(capture.frames.iter().any(|frame| frame
        .effect_execution_summary
        .iter()
        .any(|line| line.starts_with("Last known Ravenous Amulet") && line.contains("(Soul, 3)"))));
    settle(&mut s);
    assert_eq!(s.life(1), life - 3);
}

#[test]
fn fishing_pole_baits_by_tapping_itself_and_fishes_when_its_creature_untaps() {
    use crate::support::inspect::permanent;
    use crate::support::{
        game::set_counters,
        inspect::{counters_of, tapped},
        play::offers_ability,
    };
    use managym::state::{
        game_object::Target,
        permanent::CounterKind::{Bait, Stun},
    };

    let mut s = five_color_setup(&["Fishing Pole", "Gray Ogre"]);
    let pole = s.force_permanent_on_battlefield(0, "Fishing Pole");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    crate::support::game::clear_summoning_sickness(&mut s, ogre);
    s.set_player_mana_pool(0, "2");
    activate(&mut s, pole, 0);
    assert!(s.choose_target(Target::Permanent(ogre)));
    spend_colorless(&mut s);
    settle(&mut s);
    assert_eq!(permanent(&s, pole).attached_to, Some(card_of(&s, ogre)));

    // The granted ability taps the Pole as well as the creature.
    s.game_mut().state.permanents[pole].as_mut().unwrap().tapped = true;
    assert!(
        !offers_ability(&mut s, ogre, 0),
        "the Pole must be untapped"
    );
    s.game_mut().state.permanents[pole].as_mut().unwrap().tapped = false;
    s.set_player_mana_pool(0, "1");
    activate(&mut s, ogre, 0);
    spend_colorless(&mut s);
    assert!(tapped(&s, ogre) && tapped(&s, pole));
    settle(&mut s);
    assert_eq!(counters_of(&s, pole, Bait), 1);

    // Untapping in the untap step triggers, and the Fish arrives in upkeep.
    let pending = s.game().clone();
    s.advance_to_active_step(1, StepKind::Main);
    assert!(s.battlefield_permanents_named(0, "Fish").is_empty());
    s.advance_to_active_step(0, StepKind::Upkeep);
    settle(&mut s);
    let fish = s.battlefield_permanents_named(0, "Fish");
    assert_eq!(fish.len(), 1);
    assert_eq!(s.game().characteristics(fish[0]).pt(), (1, 1));
    assert_eq!(counters_of(&s, pole, Bait), 0);

    // A stun counter removed instead of untapping is no untapping.
    *s.game_mut() = pending.clone();
    set_counters(&mut s, ogre, Stun, 1);
    s.advance_to_active_step(0, StepKind::Upkeep);
    settle(&mut s);
    assert!(tapped(&s, ogre));
    assert!(s.battlefield_permanents_named(0, "Fish").is_empty());
    assert_eq!(counters_of(&s, pole, Bait), 1);

    // Without bait, untapping catches nothing.
    *s.game_mut() = pending;
    set_counters(&mut s, pole, Bait, 0);
    s.advance_to_active_step(0, StepKind::Upkeep);
    settle(&mut s);
    assert!(!tapped(&s, ogre));
    assert!(s.battlefield_permanents_named(0, "Fish").is_empty());
}

/// Two Fishing Poles on one creature grant two activations that differ only
/// in which Pole they tap: each names its own Pole in focus and text, and
/// paying one taps that Pole alone.
#[test]
fn two_fishing_poles_on_one_creature_offer_activations_naming_each_pole() {
    use crate::support::inspect::{counters_of, tapped};
    use managym::{
        agent::{action::Action, observation::Observation},
        state::{game_object::Target, permanent::CounterKind::Bait},
    };

    let mut s = five_color_setup(&["Fishing Pole", "Gray Ogre"]);
    let poles = [
        s.force_permanent_on_battlefield(0, "Fishing Pole"),
        s.force_permanent_on_battlefield(0, "Fishing Pole"),
    ];
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    crate::support::game::clear_summoning_sickness(&mut s, ogre);
    for pole in poles {
        s.set_player_mana_pool(0, "2");
        activate(&mut s, pole, 0);
        assert!(s.choose_target(Target::Permanent(ogre)));
        spend_colorless(&mut s);
        settle(&mut s);
    }
    s.set_player_mana_pool(0, "1");
    s.game_mut().scenario_refresh_priority().unwrap();
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let granted: Vec<_> = observation
        .action_space
        .actions
        .iter()
        .zip(&s.action_space().actions)
        .filter(|(_, action)| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == ogre))
        .map(|(option, action)| (option.clone(), action.clone()))
        .collect();
    assert_eq!(granted.len(), 2);
    let public = |id: managym::state::game_object::PermanentId| {
        s.game().state.permanents[id].as_ref().unwrap().id.0 as i32
    };
    for ((option, _), pole) in granted.iter().zip(poles) {
        assert_eq!(option.focus, [public(ogre), public(pole)]);
        let description = option.description.as_deref().expect("names its Pole");
        assert!(description.contains("tap Fishing Pole"), "{description}");
    }
    assert_ne!(granted[0].0.description, granted[1].0.description);

    let second = granted[1].1.clone();
    crate::support::play::choose(&mut s, |action| *action == second);
    spend_colorless(&mut s);
    assert!(tapped(&s, ogre) && tapped(&s, poles[1]) && !tapped(&s, poles[0]));
    settle(&mut s);
    assert_eq!(counters_of(&s, poles[1], Bait), 1);
    assert_eq!(counters_of(&s, poles[0], Bait), 0);
}

/// Bloodthirsty Conqueror beside a Gray Ogre that drains each opponent 1
/// whenever its controller gains life (and, with `poison`, gives each
/// opponent a poison counter): every loss is gained back and every gain is a
/// loss. The opponent has Herald of Eternal Dawn (can't lose) or
/// `opponent_life`. Hands are empty; Lightning Bolt is on the stack at the
/// opponent, and player 0 holds `spare_bolts` more with mana for them.
fn conqueror_drain_loop(
    herald: bool,
    opponent_life: i32,
    spare_bolts: usize,
    poison: bool,
) -> (Scenario, managym::state::game_object::PermanentId) {
    use crate::support::{
        game::{clear_hand, force_new_card_in_hand, inject_ability},
        play::{cast_card, target_opponent},
    };
    use managym::state::{
        ability::{Ability, Actor, Effect, EffectValue, PlayerSelector, TriggerCondition},
        permanent::CounterKind,
    };

    let mut s = five_color_setup(&[
        "Bloodthirsty Conqueror",
        "Gray Ogre",
        "Herald of Eternal Dawn",
        "Lightning Bolt",
    ]);
    s.force_permanent_on_battlefield(0, "Bloodthirsty Conqueror");
    let drain = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let drain_card = card_of(&s, drain);
    inject_ability(
        &mut s,
        drain_card,
        Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::GainsLife {
                who: Actor::You,
                nth: None,
            },
            effects: [Effect::LoseLife {
                player: PlayerSelector::EachOpponent,
                amount: EffectValue::Fixed(1),
            }]
            .into_iter()
            .chain(poison.then_some(Effect::GivePlayerCounters {
                player: PlayerSelector::EachOpponent,
                kind: CounterKind::Poison,
                count: 1,
            }))
            .collect(),
        },
    );
    if herald {
        s.force_permanent_on_battlefield(1, "Herald of Eternal Dawn");
    }
    s.game_mut().state.players[1].life = opponent_life;
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.set_player_mana_pool(0, &"R".repeat(1 + spare_bolts));
    let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
    for _ in 0..spare_bolts {
        force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
    }
    cast_card(&mut s, bolt);
    target_opponent(&mut s);
    (s, drain)
}

/// Pass priority, whoever holds it, until the game ends or `passes` run out.
fn pass_until_over(s: &mut Scenario, passes: usize) {
    for _ in 0..passes {
        if s.game().is_game_over() {
            return;
        }
        s.pass_priority();
    }
}

/// Against Herald of Eternal Dawn the drain can never end, so the game is a
/// draw (CR 104.4b); against a large but finite life total it drains to a
/// win.
#[test]
fn bloodthirsty_conqueror_draws_a_mandatory_loop_against_a_protected_opponent() {
    for herald in [true, false] {
        let (mut s, _) = conqueror_drain_loop(herald, if herald { 20 } else { 200 }, 0, false);
        s.game_mut().skip_trivial = true;
        s.pass_priority();
        assert!(s.game().is_game_over());
        assert_eq!(s.game().state.mandatory_loop_draw(), herald);
        if herald {
            assert_eq!(s.game().winner_index(), None);
            assert!(s.life(1) < 0);
        } else {
            assert_eq!(s.game().winner_index(), Some(0));
            assert_eq!(s.life(1), 0);
        }
    }
}

/// The loop is the players' passes, not one call into the engine: offered
/// every priority (no trivial skipping), or holding an instant they never
/// cast, players who only pass are still in the loop.
#[test]
fn bloodthirsty_conqueror_loop_draws_while_players_pass_with_choices() {
    let (mut s, _) = conqueror_drain_loop(true, 20, 0, false);
    s.game_mut().skip_trivial = false;
    pass_until_over(&mut s, 2_000);
    assert!(s.game().state.mandatory_loop_draw());

    let (mut s, _) = conqueror_drain_loop(true, 20, 1, false);
    s.game_mut().skip_trivial = true;
    pass_until_over(&mut s, 2_000);
    assert!(s.game().state.mandatory_loop_draw());
    assert_eq!(s.game().winner_index(), None);
}

/// Casting the held Bolt at the drain is a way to stop: it ends the watch,
/// and with the drain gone the chain runs out.
#[test]
fn bloodthirsty_conqueror_loop_ends_when_a_player_kills_the_drain() {
    use crate::support::play::choose;
    use managym::agent::action::Action;
    use managym::state::game_object::Target;

    let (mut s, drain) = conqueror_drain_loop(true, 20, 1, false);
    s.game_mut().skip_trivial = true;
    pass_until_over(&mut s, 40);
    assert!(!s.game().is_game_over());
    assert!(s.game().state.priority.loop_watch.is_some());
    while s.action_space().player != Some(PlayerId(0)) {
        s.pass_priority();
    }
    choose(&mut s, |action| matches!(action, Action::CastSpell { .. }));
    assert!(s.game().state.priority.loop_watch.is_none());
    assert!(s.choose_target(Target::Permanent(drain)));
    // With nothing else to do, the chain unwinds and the game moves on.
    assert!(!s.game().is_game_over());
    assert!(s.game().state.permanents[drain].is_none());
    assert!(s.game().state.stack_objects.is_empty());
    assert!(s.game().state.priority.loop_watch.is_none());
}

/// Resolutions the watch may take past its unwatched prefix to recognize a
/// loop of a few resolutions per cycle.
const LOOP_RECOGNIZED_WITHIN: u32 = MandatoryLoopWatch::UNWATCHED + 64;

fn assert_prompt_loop_draw(s: &Scenario) {
    assert!(s.game().state.mandatory_loop_draw());
    let watch = s.game().state.priority.loop_watch.as_ref().unwrap();
    assert!(
        watch.resolutions() <= LOOP_RECOGNIZED_WITHIN,
        "{}",
        watch.resolutions()
    );
}

/// Keep an instant available so a missed draw returns control to the fixture
/// once per resolution, even with forced decisions skipped.
fn pass_through_trigger_loop(s: &mut Scenario) {
    crate::support::game::force_new_card_in_hand(s, 0, "Lightning Bolt");
    s.set_player_mana_pool(0, "R");
    s.game_mut().skip_trivial = true;
    pass_until_over(s, 2 * LOOP_RECOGNIZED_WITHIN as usize + 2);
    assert_prompt_loop_draw(s);
}

#[test]
fn a_state_trigger_loop_draws_while_its_condition_stays_true() {
    use crate::support::game::{inject_ability, set_counters};
    use managym::state::ability::{Ability, Condition, Effect, TriggerCondition};

    let mut s = five_color_setup(&["Gray Ogre", "Lightning Bolt"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = card_of(&s, ogre);
    inject_ability(
        &mut s,
        card,
        Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::State(Condition::SourceCountersAtLeast {
                kind: CounterKind::Page,
                count: 4,
            }),
            effects: vec![Effect::gain_life(1)],
        },
    );
    set_counters(&mut s, ogre, CounterKind::Page, 4);
    pass_through_trigger_loop(&mut s);
}

#[test]
fn an_untap_trigger_loop_draws_when_the_trigger_untaps_its_source() {
    use crate::support::game::inject_ability;
    use managym::state::ability::{
        Ability, Effect, ObjectSelector, TriggerCondition, TriggerSubject,
    };

    let mut s = five_color_setup(&["Gray Ogre", "Lightning Bolt"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = card_of(&s, ogre);
    inject_ability(
        &mut s,
        card,
        Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::BecomesUntapped {
                subject: TriggerSubject::This,
            },
            effects: vec![
                Effect::Tap {
                    what: ObjectSelector::This,
                },
                Effect::Untap {
                    what: ObjectSelector::This,
                },
            ],
        },
    );
    s.game_mut().state.permanents[ogre].as_mut().unwrap().tapped = true;
    // The ordinary untap step starts the chain.
    s.advance_to_active_step(1, StepKind::Upkeep);
    pass_through_trigger_loop(&mut s);
}

/// Poison is a total the watch tracks, like life: a drain that also poisons
/// Herald of Eternal Dawn's controller repeats its state and draws soon after
/// the watch starts, instead of running to the cap.
#[test]
fn a_poison_loop_against_a_protected_player_draws_promptly() {
    let (mut s, _) = conqueror_drain_loop(true, 20, 0, true);
    s.game_mut().skip_trivial = true;
    s.pass_priority();
    assert_prompt_loop_draw(&s);
    assert!(s.game().state.players[1].counters.get(CounterKind::Poison) >= 10);
}

/// A rejected action at priority changes nothing, the loop watch included.
#[test]
fn a_rejected_action_leaves_the_loop_watch() {
    use managym::agent::action::Action;
    use managym::search_state::witness;

    let (mut s, _) = conqueror_drain_loop(true, 20, 1, false);
    s.game_mut().skip_trivial = true;
    pass_until_over(&mut s, 40);
    while s.action_space().player != Some(PlayerId(0)) {
        s.pass_priority();
    }
    assert!(s.game().state.priority.loop_watch.is_some());
    // Casting the Conqueror from the battlefield is refused at execution.
    let conqueror = permanent_named(&s, "Bloodthirsty Conqueror");
    let object = s.game().state.permanents[conqueror]
        .as_ref()
        .unwrap()
        .object;
    s.game_mut().current_action_space.as_mut().unwrap().actions = vec![Action::CastSpell {
        player: PlayerId(0),
        object,
        profile: Default::default(),
        route: None,
    }];
    let before = witness(s.game());
    assert!(s.game_mut().step(0).is_err());
    assert_eq!(witness(s.game()), before);
}

/// Conqueror beside a Gray Ogre with an injected "whenever you gain life"
/// trigger, empty hands, and Lightning Bolt on the stack at the opponent.
fn conqueror_gain_chain(
    intervening_if: Option<managym::state::ability::Condition>,
    effects: Vec<managym::state::ability::Effect>,
) -> (Scenario, managym::state::game_object::PermanentId) {
    use crate::support::{
        game::{clear_hand, force_new_card_in_hand, inject_ability},
        play::{cast_card, target_opponent},
    };
    use managym::state::ability::{Ability, Actor, TriggerCondition};

    let mut s = five_color_setup(&["Bloodthirsty Conqueror", "Gray Ogre", "Lightning Bolt"]);
    s.force_permanent_on_battlefield(0, "Bloodthirsty Conqueror");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ogre_card = card_of(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            intervening_if,
            condition: TriggerCondition::GainsLife {
                who: Actor::You,
                nth: None,
            },
            effects,
        },
    );
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.set_player_mana_pool(0, "R");
    let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
    cast_card(&mut s, bolt);
    target_opponent(&mut s);
    (s, ogre)
}

/// A spell or ability that stops for a decision is still resolving (CR
/// 608.2m): the watch counts no resolution for it. Bolt sets off the
/// Conqueror, whose gain sets off the Ogre's discard of the one card in hand.
#[test]
fn a_suspended_resolution_is_not_counted_toward_a_loop() {
    use crate::support::game::force_new_card_in_hand;
    use managym::state::ability::Effect;

    let (mut s, _) = conqueror_gain_chain(None, vec![Effect::discard()]);
    force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    // Offer the one-card discard rather than skipping it.
    s.game_mut().skip_trivial = false;
    for _ in 0..20 {
        if s.game().state.suspended_decision.is_some() {
            break;
        }
        s.pass_priority();
    }
    assert!(s.game().state.suspended_decision.is_some());
    assert_eq!(s.action_space().actions.len(), 1);
    // Bolt and the Conqueror's trigger resolved; the Ogre's has not.
    let watch = s.game().state.priority.loop_watch.as_ref().unwrap();
    assert_eq!(watch.resolutions(), 2);
}

/// Bolt starts a life-gain trigger that taps a chosen Ogre and gains life again.
fn suspending_gain_loop() -> Scenario {
    use managym::state::{
        ability::{Effect, ObjectSelector},
        predicate::CardPredicate,
    };

    conqueror_gain_chain(
        None,
        vec![
            Effect::Tap {
                what: ObjectSelector::chosen(CardPredicate {
                    named: Some("Gray Ogre".into()),
                    ..Default::default()
                }),
            },
            Effect::gain_life(1),
        ],
    )
    .0
}

/// A loop made only of resolutions that stop for a decision is still a loop:
/// each counts once it finishes. The Ogre's gain trigger taps a chosen Gray
/// Ogre, then gains 1 life, which triggers it again with no way to stop.
#[test]
fn a_loop_of_suspending_resolutions_draws() {
    let mut s = suspending_gain_loop();
    s.game_mut().skip_trivial = true;
    s.pass_priority();
    assert_prompt_loop_draw(&s);
}

#[test]
fn a_choice_during_resolution_clears_the_loop_watch() {
    use managym::agent::action::ActionSpaceKind;

    let mut s = suspending_gain_loop();
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().skip_trivial = true;
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.action_space().actions.len(), 2);
    assert!(s.game().state.priority.loop_watch.is_none());

    // Inspect the next priority, with the following gain trigger on the stack.
    s.game_mut().skip_trivial = false;
    s.step_action(0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert!(s.game().state.priority.loop_watch.is_none());
}

/// A long chain that uses something up is no loop (CR 104.4b asks for "no
/// way to stop"): each gain removes one of the Ogre's 100 +1/+1 counters to
/// gain 1 more, so the chain runs far past the watch's unwatched prefix and
/// ends on its own with every counter spent.
#[test]
fn a_finite_chain_over_counters_is_no_mandatory_loop() {
    use crate::support::{game::set_counters, inspect::counters_of};
    use managym::state::{
        ability::{Condition, Effect, ObjectSelector},
        permanent::CounterKind,
    };

    let (mut s, ogre) = conqueror_gain_chain(
        Some(Condition::SourceCountersAtLeast {
            kind: CounterKind::PlusOnePlusOne,
            count: 1,
        }),
        vec![
            Effect::RemoveCounters {
                from: ObjectSelector::This,
                up_to: 1,
                chosen: Some(CounterKind::PlusOnePlusOne),
            },
            Effect::gain_life(1),
        ],
    );
    set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 100);
    s.game_mut().skip_trivial = true;
    s.pass_priority();
    assert!(!s.game().is_game_over());
    assert_eq!(counters_of(&s, ogre, CounterKind::PlusOnePlusOne), 0);
    assert_eq!(s.life(0), 20 + 3 + 100);
    assert_eq!(s.life(1), 17);
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
        let (mut replay, _) = command(
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
            matches!(
                a,
                Action::DeclareAttacker {
                    attack: Some(_),
                    ..
                }
            )
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
    // Assign lethal damage to the blocker and trample over for five.
    pass_until(&mut s, ActionSpaceKind::Modal);
    choose(&mut s, |action| {
        matches!(action, Action::ChooseMode { mode: 2, .. })
    });
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
/// same combat total drives the offer, stack description, policy input and
/// counters placed, even when it exceeds the range of announced X.
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
    crate::support::game::set_counters(
        &mut s,
        ogre,
        managym::state::permanent::CounterKind::PlusOnePlusOne,
        258,
    );
    s.game_mut().state.players[1].life = 300;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |a| {
            matches!(
                a,
                Action::DeclareAttacker {
                    attack: Some(_),
                    ..
                }
            )
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
    let mut stacked: Vec<(i32, ObjectId)> = s
        .game()
        .state
        .stack_objects
        .iter()
        .map(|item| match item {
            StackObject::TriggeredAbility(trigger) => match trigger.context {
                Some(managym::state::game_object::Target::Permanent(id)) => {
                    assert_eq!(trigger.x, 0);
                    (
                        trigger.triggering_amount,
                        s.game().state.permanents[id].as_ref().unwrap().object,
                    )
                }
                other => panic!("event creature: {other:?}"),
            },
            other => panic!("only triggers: {other:?}"),
        })
        .collect();
    stacked.sort();
    assert_eq!(stacked, vec![(7, wurm_object), (260, ogre_object)]);

    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        assert_eq!(obs.stack_objects.len(), 2);
        for (amount, name) in [(7, "Quilled Greatwurm"), (260, "Gray Ogre")] {
            assert!(obs.stack_objects.iter().any(|item| {
                assert_eq!(item.kind, StackObjectKindData::TriggeredAbility);
                assert_eq!(item.x, 0);
                let description = item.description.as_deref().unwrap();
                description.contains(&format!("Put that many ({amount}) +1/+1 counters"))
                    && description.contains(&format!(" — event: {name}"))
            }));
        }

        // The policy input joins each stacked trigger to its creature.
        let mut frames: Vec<(i32, Option<i32>)> = s
            .game()
            .continuation_frames(viewer)
            .iter()
            .filter(|frame| frame.kind == 3)
            .map(|frame| {
                assert_eq!(frame.x, 0);
                (
                    frame.triggering_amount,
                    frame.context.and_then(|c| c.object),
                )
            })
            .collect();
        frames.sort();
        assert_eq!(frames.len(), 2);
        assert_eq!((frames[0].0, frames[1].0), (7, 260));
        assert!(frames.iter().all(|(_, object)| object.is_some()));
        assert_ne!(frames[0].1, frames[1].1);
    }
    settle(&mut s);
    assert_eq!(crate::support::inspect::counters(&s, ogre), 258 + 260);
    assert_eq!(crate::support::inspect::counters(&s, wurm), 7);
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

/// Replay both legal orders from the same pending end-step decision.
fn check_both_trigger_orders(s: &mut Scenario, check: impl Fn(&mut Scenario, [ObjectId; 2])) {
    use crate::support::play::command;
    use managym::{agent::action::ActionSpaceKind, search_state::witness};

    s.advance_to_step(StepKind::End);
    assert_eq!(s.action_space().kind, ActionSpaceKind::OrderTriggers);
    // The copy-granted ability is no catalog slot: each placement frame
    // carries the program the stacked trigger will carry.
    let view = managym::agent::observation::Observation::for_player(s.game(), PlayerId(0));
    let placements: Vec<_> = view
        .continuation
        .iter()
        .filter(|frame| matches!(frame.kind, 7 | 8))
        .collect();
    assert_eq!(placements.len(), 2);
    assert!(placements.iter().all(|frame| frame
        .program_description
        .as_ref()
        .is_some_and(|text| text.to_lowercase().contains("sacrifice"))));
    let orders: [_; 2] = s.action_space().actions.clone().try_into().unwrap();
    let root = s.snapshot();
    let [first, second] = orders.map(|order| {
        s.restore(&root);
        let (_, committed) = command(s, |action| action == &order);
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
            let waiting = s.game().state.stack_objects.len();
            resolve_top(s);
            // The copied ability finishes even when that exact source has left.
            assert_eq!(s.game().state.stack_objects.len(), waiting - 1);
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

/// Two creatures die in one state-based batch and each Call trigger returns
/// its own card. Blinking the first returned Angel while the second trigger
/// waits makes a new object (CR 400.7): it keeps neither the counter nor the
/// granted Angel type and flying, and the second return touches only its card.
#[test]
fn departure_valkyries_call_returns_simultaneous_deaths_and_a_blink_starts_fresh() {
    use crate::support::{
        inspect::{counters_of, zone_of},
        play::{command, resolve_top},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{game_object::ObjectId, permanent::CounterKind},
    };
    let mut s = five_color_setup(&["Valkyrie's Call", "Gray Ogre", "Hill Giant"]);
    s.force_permanent_on_battlefield(0, "Valkyrie's Call");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let cards = [card_of(&s, ogre), card_of(&s, giant)];
    s.game_mut().state.permanents[ogre].as_mut().unwrap().damage = 2;
    s.game_mut().state.permanents[giant]
        .as_mut()
        .unwrap()
        .damage = 3;
    s.pass_priority();
    assert!(cards
        .iter()
        .all(|&c| zone_of(&s, c) == Some(ZoneType::Graveyard)));
    assert_eq!(s.action_space().kind, ActionSpaceKind::OrderTriggers);
    command(&mut s, |_| true);
    assert_eq!(s.game().state.stack_objects.len(), 2);

    let angel = |s: &Scenario, card: ObjectId| {
        let id = permanent_of(s, card).unwrap();
        let view = s.game().characteristics(id);
        (
            view.has_subtype("Angel") && view.keywords.has(Keyword::Flying),
            counters_of(s, id, CounterKind::PlusOnePlusOne),
        )
    };
    resolve_top(&mut s);
    let (first, second) = match permanent_of(&s, cards[0]) {
        Some(_) => (cards[0], cards[1]),
        None => (cards[1], cards[0]),
    };
    assert_eq!(angel(&s, first), (true, 1));
    assert_eq!(zone_of(&s, second), Some(ZoneType::Graveyard));

    let exiled = move_card(&mut s, first, ZoneType::Exile);
    let blinked = move_card(&mut s, exiled, ZoneType::Battlefield);
    assert_eq!(angel(&s, blinked), (false, 0));
    settle(&mut s);
    assert_eq!(angel(&s, second), (true, 1));
    assert_eq!(angel(&s, blinked), (false, 0));
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

/// Clone dies as a 4-power Panda but is a non-Bear in the graveyard.
#[test]
fn fiendish_panda_and_driver_choose_exact_graveyard_targets() {
    use crate::support::{
        game::move_from_library,
        inspect::{counters, offered_targets, zone_of},
        play::{cast, cast_clone_as, command, pass_until, resolve_top},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::Target, player::ManaPayment},
    };

    let mut s = five_color_setup(&[
        "Fiendish Panda",
        "Clone",
        "Koala-Sheep",
        "Driver of the Dead",
        "Craw Wurm",
        "Llanowar Elves",
        "Gray Ogre",
        "Redirect Lightning",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let panda = s.force_permanent_on_battlefield(1, "Fiendish Panda");
    let clone = cast_clone_as(&mut s, panda);
    enter_and_settle(&mut s, "Koala-Sheep");
    assert_eq!(counters(&s, permanent_of(&s, clone).unwrap()), 1);
    assert_eq!(counters(&s, panda), 0);

    let mut grave = |owner, name| move_from_library(&mut s, owner, name, ZoneType::Graveyard);
    let elf = grave(0, "Llanowar Elves");
    let ogre = grave(0, "Gray Ogre");
    let driver = grave(0, "Driver of the Dead");
    grave(0, "Craw Wurm");
    grave(0, "Koala-Sheep");
    grave(0, "Forest");
    grave(1, "Llanowar Elves");

    let dead_clone = move_card(&mut s, clone, ZoneType::Graveyard);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert_eq!(
        offered_targets(&s),
        [elf, ogre, driver].map(Target::GraveyardCard)
    );
    assert!(s.choose_target(Target::GraveyardCard(ogre)));
    let trigger = s.game().state.stack_objects.last().unwrap().as_target();
    let root = s.snapshot();
    for reentered in [false, true] {
        s.restore(&root);
        let mut replacements = vec![elf, driver];
        if reentered {
            move_card(&mut s, dead_clone, ZoneType::Hand);
            replacements.push(move_card(&mut s, dead_clone, ZoneType::Graveyard));
        }
        cast(&mut s, "Redirect Lightning");
        command(&mut s, |a| matches!(a, Action::Decline { .. }));
        assert!(s.choose_target(trigger));
        resolve_top(&mut s);
        assert_eq!(
            offered_targets(&s),
            replacements
                .into_iter()
                .map(Target::GraveyardCard)
                .collect::<Vec<_>>()
        );
        command(
            &mut s,
            |a| matches!(a, Action::ChooseTarget { target: Target::GraveyardCard(card), .. } if *card == driver),
        );
        settle(&mut s);
        assert_eq!(zone_of(&s, driver), Some(ZoneType::Battlefield));
        assert_eq!(zone_of(&s, ogre), Some(ZoneType::Graveyard));

        move_card(&mut s, driver, ZoneType::Graveyard);
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert_eq!(offered_targets(&s), [Target::GraveyardCard(elf)]);
        assert!(s.choose_target(Target::GraveyardCard(elf)));
        settle(&mut s);
        assert_eq!(zone_of(&s, elf), Some(ZoneType::Battlefield));
        assert_eq!(zone_of(&s, clone), Some(ZoneType::Graveyard));
    }
}

/// Panda normally dies before choosing a target. Put its power condition on
/// a live modal source to check the mode offer and the resolution recheck.
#[test]
fn source_power_limits_trigger_modes_and_is_rechecked_after_savor() {
    use crate::support::{
        game::move_from_library,
        inspect::{offered_targets, zone_of},
        play::{cast, choose_mode, resolve_top},
    };
    use managym::{
        agent::action::Action,
        state::{
            ability::{Ability, Actor, Effect, ObjectSelector, TargetSpec, TriggerCondition},
            game_object::Target,
            player::ManaPayment,
            predicate::CardPredicate,
        },
    };
    let mut s = five_color_setup(&[
        "Hill Giant",
        "Gray Ogre",
        "Llanowar Elves",
        "Driver of the Dead",
        "Savor",
    ]);
    let source = s.force_permanent_on_battlefield(0, "Hill Giant");
    let object = card_of(&s, source);
    let return_creature = |min_mana_value| {
        Effect::return_to_battlefield(
            ObjectSelector::Target(TargetSpec::GraveyardCardYouOwnWithinSourcePower {
                predicate: CardPredicate {
                    min_mana_value,
                    ..CardPredicate::creature()
                },
            }),
            false,
        )
    };
    s.game_mut().state[object].abilities = vec![Ability::Triggered {
        condition: TriggerCondition::casts(Actor::You, CardPredicate::creature()),
        intervening_if: None,
        effects: vec![Effect::TriggerModes {
            modes: vec![
                vec![return_creature(Some(4))],
                vec![return_creature(None)],
                vec![Effect::gain_life(1)],
            ],
            min: 0,
            up_to: None,
            unchosen: false,
        }],
    }];
    let ogre = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
    move_from_library(&mut s, 0, "Driver of the Dead", ZoneType::Graveyard);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    cast(&mut s, "Llanowar Elves");
    let modes: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::ChooseMode { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    assert_eq!(modes, [1, 2]);
    choose_mode(&mut s, 1);
    assert_eq!(offered_targets(&s), [Target::GraveyardCard(ogre)]);
    assert!(s.choose_target(Target::GraveyardCard(ogre)));
    let root = s.snapshot();
    resolve_top(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Battlefield));

    s.restore(&root);
    cast(&mut s, "Savor");
    assert!(s.choose_target(Target::Permanent(source)));
    resolve_top(&mut s);
    assert_eq!(s.game().characteristics(source).pt(), (1, 1));
    resolve_top(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Graveyard));
}

#[test]
fn fiendish_panda_gains_one_counter_per_lifelink_source_in_combat() {
    use crate::support::{
        game::clear_summoning_sickness,
        inspect::counters,
        play::{attack_with_group, choose, pass_until},
    };
    use managym::agent::action::{Action, ActionSpaceKind};
    let mut s = five_color_setup(&["Fiendish Panda", "Sire of Seven Deaths", "Gray Ogre"]);
    let panda = s.force_permanent_on_battlefield(0, "Fiendish Panda");
    let split = s.force_permanent_on_battlefield(0, "Sire of Seven Deaths");
    let unblocked = s.force_permanent_on_battlefield(0, "Sire of Seven Deaths");
    for attacker in [split, unblocked] {
        clear_summoning_sickness(&mut s, attacker);
        s.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with_group(&mut s, &[split, unblocked]);
    pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
    for _ in 0..2 {
        choose(
            &mut s,
            |a| matches!(a, Action::DeclareBlocker { attacker: Some(id), .. } if *id == split),
        );
    }
    pass_until(&mut s, ActionSpaceKind::Modal);
    // Assign the first strike among the two Ogres, with enough to kill both.
    crate::support::play::choose_mode(&mut s, 2);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.life(0), 34);
    assert_eq!(s.life(1), 10);
    assert_eq!(counters(&s, panda), 2);
}

#[test]
fn raise_the_past_returns_small_creatures_together_so_momo_sees_the_sailor() {
    use crate::support::inspect::zone_of;
    let (s, moved) = crate::return_from_two_graveyards("Raise the Past");
    for object in [moved.momo, moved.sailor] {
        assert_eq!(zone_of(&s, object), Some(ZoneType::Battlefield));
    }
    assert_eq!(
        zone_of(&s, moved.drake),
        Some(ZoneType::Graveyard),
        "mana value 3"
    );
    assert_eq!(
        zone_of(&s, moved.bolt),
        Some(ZoneType::Graveyard),
        "not a creature"
    );
    assert_eq!(
        zone_of(&s, moved.their_sailor),
        Some(ZoneType::Graveyard),
        "only your graveyard"
    );
    let momo = permanent_of(&s, moved.momo).unwrap();
    assert_eq!(
        s.game().characteristics(momo).power,
        2,
        "the Sailor entered with Momo"
    );
}

/// CR 205.3m: the animated land has every creature type, indefinitely, so
/// Diligent Zookeeper's "non-Human" excludes it. Witness Protection's later
/// type change replaces them with Citizen alone (CR 613.7, layer 4).
#[test]
fn soulstone_sanctuary_has_all_creature_types_until_a_later_type_change() {
    let mut s = five_color_setup(&[
        "Soulstone Sanctuary",
        "Diligent Zookeeper",
        "Witness Protection",
    ]);
    s.force_permanent_on_battlefield(0, "Diligent Zookeeper");
    let object = enter_and_settle(&mut s, "Soulstone Sanctuary");
    let sanctuary = permanent_of(&s, object).unwrap();
    activate(&mut s, sanctuary, 0);
    settle(&mut s);
    let view = s.game().characteristics(sanctuary);
    assert!(view.is_creature() && view.is_land());
    assert!(view.creature_types.is_all());
    assert!(view.has_subtype("Ally") && view.has_subtype("Human"));
    assert!(!view.has_subtype("Forest"), "no land subtype");
    assert!(view.keywords.has(Keyword::Vigilance));
    assert_eq!(view.pt(), (3, 3), "a Human gets no Zookeeper bonus");
    assert!(observed(&s, sanctuary).is_ally);

    s.advance_to_active_step(0, StepKind::Upkeep);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(
        s.game().characteristics(sanctuary).creature_types.is_all(),
        "no duration"
    );
    s.set_player_mana_pool(0, "U");
    cast(&mut s, "Witness Protection");
    assert!(s.choose_target(Target::Permanent(sanctuary)));
    settle(&mut s);
    let view = s.game().characteristics(sanctuary);
    assert_eq!(view.creature_types.listed(), ["Citizen".to_string()]);
    assert!(!view.has_subtype("Human") && !view.is_land());
    assert_eq!(view.pt(), (2, 2), "a Citizen with one creature type");
}

/// A later grant of every creature type overrides an earlier set list, so
/// the permanent row reports no set list (the shared-effect row carries all
/// creature types); a noncreature granted every type is no Ally mid-pass
/// (CR 205.3d), so "Allies you control" skips it.
#[test]
fn all_creature_types_overrides_a_set_list_and_needs_a_creature() {
    let mut s = five_color_setup(&["Gray Ogre", "Allied Teamwork", "Forest"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    until_end_of_turn(
        &mut s,
        ogre,
        Modification::SetCreatureTypes(vec!["Citizen".into()]),
    );
    assert_eq!(
        observed(&s, ogre).creature_types,
        Some(vec!["Citizen".to_string()])
    );
    until_end_of_turn(&mut s, ogre, Modification::AllCreatureTypes);
    assert!(s.game().characteristics(ogre).creature_types.is_all());
    assert_eq!(observed(&s, ogre).creature_types, None);

    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.force_permanent_on_battlefield(0, "Allied Teamwork");
    until_end_of_turn(&mut s, forest, Modification::AllCreatureTypes);
    let view = s.game().characteristics(forest);
    assert!(!view.is_creature() && !view.has_subtype("Ally"));
    assert_eq!(view.pt(), (0, 0), "no Allied Teamwork bonus");
}

/// The granted "When this creature deals combat damage, sacrifice it" fires
/// on combat damage only (CR 510.2): biting with Rocky Rebuke keeps the
/// Goblin, hitting the opponent in combat sacrifices it.
#[test]
fn dropkick_bomber_sacrifices_its_goblin_after_combat_damage_only() {
    let mut s = five_color_setup(&[
        "Dropkick Bomber",
        "Raging Goblin",
        "Rocky Rebuke",
        "Gray Ogre",
    ]);
    let bomber = s.force_permanent_on_battlefield(0, "Dropkick Bomber");
    let goblin = s.force_permanent_on_battlefield(0, "Raging Goblin");
    let other = s.force_permanent_on_battlefield(0, "Raging Goblin");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert_eq!(s.game().characteristics(goblin).pt(), (2, 2));
    assert_eq!(
        s.game().characteristics(bomber).pt(),
        (2, 3),
        "other Goblins"
    );

    activate(&mut s, bomber, 0);
    assert!(
        !s.choose_target(Target::Permanent(bomber)),
        "another Goblin"
    );
    assert!(s.choose_target(Target::Permanent(goblin)));
    settle(&mut s);
    assert!(keywords(&s, goblin).has(Keyword::Flying));
    assert!(!keywords(&s, other).has(Keyword::Flying));

    cast(&mut s, "Rocky Rebuke");
    assert!(s.choose_target(Target::Permanent(goblin)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert!(s.game().state.permanents[ogre].is_none(), "the bite hit");
    assert!(
        s.game().state.permanents[goblin].is_some(),
        "noncombat damage"
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, goblin);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.players[1].life, 18);
    assert!(s.game().state.permanents[goblin].is_none(), "sacrificed");
    assert!(s.game().state.permanents[other].is_some());
}

/// Blocked combat damage also fires the granted sacrifice: the Goblin
/// survives the Ogre's 2 damage, so only the trigger removes it.
#[test]
fn dropkick_bomber_sacrifices_a_blocked_goblin_after_combat_damage() {
    use crate::support::game::until_end_of_turn;
    use managym::state::{card::Keywords, effect::Modification};
    let mut s = five_color_setup(&["Dropkick Bomber", "Raging Goblin", "Gray Ogre"]);
    let bomber = s.force_permanent_on_battlefield(0, "Dropkick Bomber");
    let goblin = s.force_permanent_on_battlefield(0, "Raging Goblin");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    until_end_of_turn(
        &mut s,
        ogre,
        Modification::AddKeywords(Keywords::of(&[Keyword::Reach])),
    );
    until_end_of_turn(
        &mut s,
        goblin,
        Modification::ModifyPt {
            power: 0,
            toughness: 3,
        },
    );
    activate(&mut s, bomber, 0);
    assert!(s.choose_target(Target::Permanent(goblin)));
    settle(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, goblin);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    assert_eq!(s.game().state.players[1].life, 20, "blocked");
    assert!(
        s.game().state.permanents[ogre].is_none(),
        "2 damage killed it"
    );
    assert!(s.game().state.permanents[goblin].is_none(), "sacrificed");
}

/// Its animation has no duration, and an animated land is summoning sick on
/// the turn it came under its controller's control (CR 302.6): it attacks
/// on the next turn, untapped thanks to vigilance.
#[test]
fn soulstone_sanctuary_stays_animated_and_attacks_from_the_next_turn() {
    use crate::support::{game::Scenario, inspect::tapped};
    use managym::agent::action::{Action, ActionSpaceKind};
    let mut s = five_color_setup(&["Soulstone Sanctuary"]);
    let object = enter_and_settle(&mut s, "Soulstone Sanctuary");
    let sanctuary = permanent_of(&s, object).unwrap();
    activate(&mut s, sanctuary, 0);
    settle(&mut s);
    let offered = |s: &Scenario| {
        s.action_space().actions.iter().any(
            |action| matches!(action, Action::DeclareAttacker { permanent, .. } if *permanent == sanctuary),
        )
    };
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(!offered(&s), "sick on its entry turn");

    s.advance_to_active_step(1, StepKind::Upkeep);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    assert!(offered(&s), "still a creature a turn later");
    attack_with(&mut s, sanctuary);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.players[1].life, 17);
    assert!(!tapped(&s, sanctuary), "vigilance");
}

/// The entry grant belongs to the creature it was given to (CR 611.2c):
/// equipping another creature moves only the Armor's +2/+0 and flying.
#[test]
fn celestial_armor_entry_protection_stays_when_the_armor_moves() {
    use crate::support::play::pass_until;
    use managym::agent::action::ActionSpaceKind;
    let mut s = five_color_setup(&["Celestial Armor", "Gray Ogre"]);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "2W");
    let armor = cast(&mut s, "Celestial Armor");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(first)));
    settle(&mut s);
    let armor = permanent_of(&s, armor).unwrap();
    let has = |s: &Scenario, id, keyword| keywords(s, id).has(keyword);
    assert_eq!(s.game().characteristics(first).pt(), (4, 2));
    for keyword in [Keyword::Flying, Keyword::Hexproof, Keyword::Indestructible] {
        assert!(has(&s, first, keyword), "{keyword:?}");
    }

    s.set_player_mana_pool(0, "3W");
    activate(&mut s, armor, 0);
    assert!(s.choose_target(Target::Permanent(second)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(first).pt(), (2, 2));
    assert_eq!(s.game().characteristics(second).pt(), (4, 2));
    assert!(!has(&s, first, Keyword::Flying) && has(&s, second, Keyword::Flying));
    for keyword in [Keyword::Hexproof, Keyword::Indestructible] {
        assert!(has(&s, first, keyword) && !has(&s, second, keyword));
    }
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!has(&s, first, Keyword::Hexproof), "until end of turn");
}

#[test]
fn chandra_flameshaper_loyalty_abilities() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, pin_top, set_counters},
        inspect::{counters_of, has_action, keywords},
        play::{activate, cast, choose, choose_mode, pass_until, select_card, settle},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        flow::turn::StepKind,
        state::{card::Keyword, game_object::PlayerId, permanent::CounterKind, target::Target},
    };

    // +2 adds {R}{R}{R}, exiles top 3, chooses one playable this turn; others stay unplayable.
    let mut s = five_color_setup(&[
        "Chandra, Flameshaper",
        "Gray Ogre",
        "Lightning Bolt",
        "Plains",
    ]);
    let chandra = enter_battlefield(&mut s, "Chandra, Flameshaper");
    assert_eq!(counters_of(&s, chandra, CounterKind::Loyalty), 6);
    let bolt = pin_top(&mut s, &["Lightning Bolt", "Gray Ogre", "Plains"])[0];
    s.set_player_mana_pool(0, "0");
    activate(&mut s, chandra, 0);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 3);
    assert_eq!(
        s.action_space().actions.len(),
        3,
        "choose 1 of 3 exiled cards"
    );
    // Choose Lightning Bolt from among the three exiled cards.
    select_card(&mut s, bolt);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert!(
        has_action(
            &s,
            |a| matches!(a, Action::CastSpell { object, .. } if s.game().state[*object].name == "Lightning Bolt")
        ),
        "chosen exiled card is playable this turn"
    );
    assert!(
        !has_action(
            &s,
            |a| matches!(a, Action::CastSpell { object, .. } if s.game().state[*object].name == "Gray Ogre")
        ),
        "unchosen exiled creature is not playable"
    );
    assert!(
        !has_action(
            &s,
            |a| matches!(a, Action::PlayLand { object, .. } if s.game().state[*object].name == "Plains")
        ),
        "unchosen exiled land is not playable"
    );

    // +1 creates a hasty token copy that sacrifices itself at the beginning of the end step.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    activate(&mut s, chandra, 1);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    let ogres = s.battlefield_permanents_named(0, "Gray Ogre");
    assert_eq!(ogres.len(), 2);
    let copy = ogres.into_iter().find(|&id| id != ogre).unwrap();
    assert!(keywords(&s, copy).has(Keyword::Haste));
    s.advance_to_step(StepKind::End);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre"), vec![ogre]);

    // -4 divided damage: 8 divided among 2 targets (6 and 2), linear sequential offers,
    // and CR 608.2b: if one target becomes illegal before resolution, the other still gets only its assigned share.
    let mut s = five_color_setup(&["Chandra, Flameshaper", "Hill Giant", "Lightning Bolt"]);
    let chandra = enter_battlefield(&mut s, "Chandra, Flameshaper");
    let g1 = s.force_permanent_on_battlefield(1, "Hill Giant");
    let g2 = s.force_permanent_on_battlefield(1, "Hill Giant");
    activate(&mut s, chandra, 2);
    assert!(s.choose_target(Target::Permanent(g1)));
    assert!(s.choose_target(Target::Permanent(g2)));
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    // Sequential division: 1..=7 offered for g1 (linear: 7 choices), g2 gets remainder automatically.
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    assert_eq!(s.action_space().actions.len(), 7);
    choose_mode(&mut s, 6); // 6 to g1, 2 to g2.
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    // Before Chandra -4 resolves, destroy g1 with Lightning Bolt so g1 becomes illegal.
    s.set_player_mana_pool(0, "R");
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(g1)));
    settle(&mut s);
    assert!(s.game().state.permanents[g1].is_none(), "g1 died to Bolt");
    assert_eq!(
        s.game().state.permanents[g2].as_ref().unwrap().damage,
        2,
        "CR 608.2b: g2 still takes only its assigned 2 damage, not 8"
    );

    // -4 stress state: 8 damage across 8 legal targets stays linear and auto-assigns 1 each.
    let mut s = five_color_setup(&["Chandra, Flameshaper"]);
    let chandra = enter_battlefield(&mut s, "Chandra, Flameshaper");
    set_counters(&mut s, chandra, CounterKind::Loyalty, 6);
    let beasts: Vec<_> = (0..8)
        .map(|_| {
            s.game_mut()
                .create_token("token.beast_4_4_g_trample", PlayerId(1), false)
                .unwrap()
        })
        .collect();
    activate(&mut s, chandra, 2);
    for (i, &beast) in beasts.iter().enumerate() {
        assert_eq!(
            s.action_space().actions.len(),
            (9 - i) + 1, // 8 Beasts + Chandra herself + decline
            "linear target offers at step {i}"
        );
        assert!(s.choose_target(Target::Permanent(beast)));
    }
    // Choosing 8 targets for 8 damage leaves only 1 each -> auto-assigned without combinatorial choices.
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    settle(&mut s);
    for &beast in &beasts {
        assert_eq!(s.game().state.permanents[beast].as_ref().unwrap().damage, 1);
    }
}
