//! Tests for cards filed in `src/eoe.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, move_card, set_counters, until_end_of_turn},
    inspect::{card_of, counters_of, has_action, offered_cards, tapped, top_activation},
    play::{activate, fork_selecting, select_card, settle},
};
use managym::{
    agent::{action::Action, observation::Observation},
    flow::turn::StepKind,
    state::{
        card::Keyword,
        effect::Modification,
        game_object::PlayerId,
        payment::{How, Toward},
        permanent::CounterKind,
        player::ManaPayment,
        zone::ZoneType,
    },
};

/// Station (CR 702.184a) taps one other untapped creature, whatever its
/// power and under either payment setting, and puts charge counters equal to
/// that creature's power on the Spacecraft as the ability resolves. The
/// receipt names the creature; each choice is its own line of play.
#[test]
fn lumen_class_frigate_stations_for_the_tapped_creatures_power() {
    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&[
            "Lumen-Class Frigate",
            "Gray Ogre",
            "Hill Giant",
            "Wall of Stone",
        ]);
        let frigate = s.force_permanent_on_battlefield(0, "Lumen-Class Frigate");
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
        let wall = s.force_permanent_on_battlefield(0, "Wall of Stone");
        until_end_of_turn(&mut s, wall, Modification::pump(-1, 0));
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        // Summoning-sick creatures can station: it is not a {T} cost.
        activate(&mut s, frigate, 0);
        let creatures = [(ogre, 2), (giant, 3), (wall, 0)];
        let mut offered = offered_cards(&s);
        offered.sort();
        let mut expected: Vec<_> = creatures.iter().map(|(id, _)| card_of(&s, *id)).collect();
        expected.sort();
        assert_eq!(offered, expected, "auto={auto}");
        let frame = s.game().semantic_decision_frame().unwrap();
        assert!(
            frame
                .offers
                .iter()
                .any(|offer| offer.label == "Tap Hill Giant to station"),
            "{:?}",
            frame.offers
        );

        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert!(
            context.instruction.contains("to station"),
            "{}",
            context.instruction
        );

        // A fork at the tap choice: the counters follow the creature chosen.
        for (creature, power) in creatures {
            let chosen = card_of(&s, creature);
            let mut fork = fork_selecting(&s, chosen);
            let ability = top_activation(&fork);
            assert_eq!(
                ability
                    .paid
                    .contributed(How::Tapped, Toward::Station)
                    .collect::<Vec<_>>(),
                [chosen]
            );
            for viewer in [PlayerId(0), PlayerId(1)] {
                let obs = Observation::for_player(fork.game(), viewer);
                assert_eq!(obs.stack_objects[0].paid_amount, Some(power));
            }
            let context = fork.game().decision_context(PlayerId(0)).unwrap();
            assert!(
                context
                    .pending
                    .iter()
                    .any(|line| line.starts_with("Paid with: tapped ")
                        && line.ends_with(" to station.")),
                "{:?}",
                context.pending
            );
            settle(&mut fork);
            assert_eq!(
                counters_of(&fork, frigate, CounterKind::Charge),
                power,
                "auto={auto}"
            );
            assert!(tapped(&fork, creature));
        }

        let giant_object = card_of(&s, giant);
        select_card(&mut s, giant_object);
        settle(&mut s);
        assert_eq!(counters_of(&s, frigate, CounterKind::Charge), 3);
        assert!(tapped(&s, giant) && !tapped(&s, ogre) && !tapped(&s, frigate));
    }
}

/// A creature that left the battlefield before the station ability resolves
/// counts for its last known power.
#[test]
fn lumen_class_frigate_stations_for_a_departed_creatures_last_known_power() {
    let mut s = five_color_unfunded_setup(&["Lumen-Class Frigate", "Hill Giant"]);
    let frigate = s.force_permanent_on_battlefield(0, "Lumen-Class Frigate");
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let giant_object = card_of(&s, giant);
    activate(&mut s, frigate, 0);
    if s.game().state.stack_objects.is_empty() {
        select_card(&mut s, giant_object);
    }
    // The tap identifies the creature. Its power can still change in response.
    until_end_of_turn(&mut s, giant, Modification::pump(2, 0));
    move_card(&mut s, giant_object, ZoneType::Graveyard);
    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        assert_eq!(obs.stack_objects[0].paid_amount, Some(5));
    }
    settle(&mut s);
    assert_eq!(counters_of(&s, frigate, CounterKind::Charge), 5);
}

/// The printed levels read the charge counters: other creatures get +1/+1 at
/// two, and at twelve the Frigate is a 3/5 artifact creature with flying and
/// lifelink that still stations, never by tapping itself. Station only as a
/// sorcery.
#[test]
fn lumen_class_frigate_levels_follow_its_charge_counters() {
    let mut s = five_color_unfunded_setup(&["Lumen-Class Frigate", "Gray Ogre", "Hill Giant"]);
    let frigate = s.force_permanent_on_battlefield(0, "Lumen-Class Frigate");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let stations = |a: &Action| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == frigate);

    for (charge, ogre_power, creature) in
        [(1, 2, false), (2, 3, false), (11, 3, false), (12, 3, true)]
    {
        set_counters(&mut s, frigate, CounterKind::Charge, charge);
        let view = s.game().characteristics(frigate);
        assert_eq!(view.is_creature(), creature, "charge={charge}");
        assert_eq!(view.keywords.has(Keyword::Flying), creature);
        assert_eq!(view.keywords.has(Keyword::Lifelink), creature);
        assert_eq!(s.game().characteristics(ogre).power, ogre_power);
        assert_eq!(s.game().characteristics(foe).power, 2);
    }
    let view = s.game().characteristics(frigate);
    assert_eq!((view.power, view.toughness), (3, 5));

    // A creature Frigate is not "another creature": only the Ogre is offered.
    activate(&mut s, frigate, 0);
    assert_eq!(offered_cards(&s), [card_of(&s, ogre)]);
    let ogre_object = card_of(&s, ogre);
    select_card(&mut s, ogre_object);
    settle(&mut s);
    // The Ogre is 3/3 under the Frigate's own bonus as the ability resolves.
    assert_eq!(counters_of(&s, frigate, CounterKind::Charge), 15);

    // With every other creature tapped there is nothing to station with.
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!has_action(&s, stations));
    s.game_mut().state.permanents[ogre].as_mut().unwrap().tapped = false;
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(has_action(&s, stations));
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(!has_action(&s, stations), "station only as a sorcery");
}

#[test]
fn silencer_and_fynn_count_artifact_creature_hits() {
    use crate::support::{
        game::{clear_summoning_sickness, until_end_of_turn},
        play::{attack_with, pass_until, resolve_top},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{card::Keywords, effect::Modification},
    };

    for token in [false, true] {
        let mut s = five_color_unfunded_setup(&["Virulent Silencer", "Fynn, the Fangbearer"]);
        let fynn = s.force_permanent_on_battlefield(0, "Fynn, the Fangbearer");
        let silencer = s.force_permanent_on_battlefield(0, "Virulent Silencer");
        let attacker = if token {
            s.game_mut()
                .create_token("token.ballistic_boulder", PlayerId(0), false)
                .unwrap()
        } else {
            silencer
        };
        clear_summoning_sickness(&mut s, attacker);
        until_end_of_turn(
            &mut s,
            attacker,
            Modification::AddKeywords(Keywords::of(&[Keyword::Deathtouch])),
        );
        s.game_mut().state.players[1]
            .counters
            .change(CounterKind::Poison, 6);
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, attacker);
        let triggers = if token { 1 } else { 2 };
        s.advance_until(
            |s| {
                s.game().state.stack_objects.len() == triggers
                    && s.action_space().kind == ActionSpaceKind::Priority
            },
            "missing poison trigger".into(),
        );
        let source = card_of(&s, fynn);
        s.game_mut().move_object(source, ZoneType::Exile);
        resolve_top(&mut s);
        assert_eq!(
            s.game().state.players[1].counters.get(CounterKind::Poison),
            8
        );
        if !token {
            resolve_top(&mut s);
        }
        s.assert_life(1, 18);
        assert_eq!(
            s.game().state.players[1].counters.get(CounterKind::Poison),
            if token { 8 } else { 10 }
        );
        assert_eq!(s.game().winner_index(), (!token).then_some(0));
    }
}

#[test]
fn silencer_requires_combat_damage_to_the_player() {
    use crate::support::{
        game::clear_summoning_sickness,
        play::{attack_with, cast_card, pass_until, settle_targeting_opponent},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{
            ability::Actor,
            card::{DamageTransform, ReplacementEffect},
        },
    };

    for prevent in [false, true] {
        let mut s = five_color_unfunded_setup(&["Virulent Silencer", "Gray Ogre"]);
        let attacker = s.force_permanent_on_battlefield(0, "Virulent Silencer");
        clear_summoning_sickness(&mut s, attacker);
        let defender = s.force_permanent_on_battlefield(1, "Gray Ogre");
        if prevent {
            let object = card_of(&s, defender);
            s.game_mut().state[object].replacement_effects =
                vec![ReplacementEffect::ModifyDamage {
                    creature_source: false,
                    source_by: Actor::AnyPlayer,
                    recipient_by: Actor::You,
                    transform: DamageTransform::Prevent(2),
                }];
        }
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, attacker);
        pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
        if prevent {
            s.decline_block();
        } else {
            s.declare_block();
        }
        s.advance_to_step(StepKind::EndOfCombat);
        s.assert_life(1, 20);
        assert_eq!(
            s.game().state.players[1].counters.get(CounterKind::Poison),
            0
        );
    }

    let mut s = five_color_unfunded_setup(&["Virulent Silencer", "Warstorm Surge"]);
    s.force_permanent_on_battlefield(0, "Warstorm Surge");
    let silencer = crate::support::game::force_new_card_in_hand(&mut s, 0, "Virulent Silencer");
    s.set_player_mana_pool(0, "3");
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    cast_card(&mut s, silencer);
    settle_targeting_opponent(&mut s);
    s.assert_life(1, 18);
    assert_eq!(
        s.game().state.players[1].counters.get(CounterKind::Poison),
        0
    );
}

#[test]
fn dominion_bracelet_exiles_the_chosen_grantor_at_the_equipped_price() {
    use crate::support::inspect::zone_of;
    use managym::state::target::Target;
    for selected in 0..2 {
        let mut s = five_color_unfunded_setup(&[
            "The Dominion Bracelet",
            "Sakashima of a Thousand Faces",
            "Gray Ogre",
        ]);
        s.force_permanent_on_battlefield(0, "Sakashima of a Thousand Faces");
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let bracelets = [0, 1].map(|_| {
            let bracelet = s.force_permanent_on_battlefield(0, "The Dominion Bracelet");
            s.set_player_mana_pool(0, "1");
            activate(&mut s, bracelet, 0);
            assert!(s.choose_target(Target::Permanent(ogre)));
            settle(&mut s);
            card_of(&s, bracelet)
        });
        let view = s.game().characteristics(ogre);
        assert_eq!(view.power, 4);
        let index = view
            .activated_abilities
            .iter()
            .position(|a| {
                a.granted_by
                    .is_some_and(|g| g.grantor == bracelets[selected])
            })
            .unwrap();
        assert_eq!(view.activated_abilities[index].mana_cost.generic(), 11);
        for (counters, price) in [(12, 0), (11, 0), (10, 1)] {
            set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, counters);
            assert_eq!(
                s.game().characteristics(ogre).activated_abilities[index]
                    .mana_cost
                    .generic(),
                price
            );
        }
        use managym::state::mana::{Mana, ManaRestriction};
        let restriction = ManaRestriction::ExceptNonartifactSpells;
        s.set_player_mana_pool(0, "U");
        s.game_mut().state.players[0].mana_pool.add(
            &Mana::restricted_to(restriction.clone(), [0, 0, 0, 0, 0, 1]),
            None,
        );
        if selected == 1 {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        s.game_mut().scenario_refresh_priority().unwrap();
        let obs = managym::agent::observation::Observation::for_player(s.game(), PlayerId(0));
        use managym::agent::observation_encoder::{encode, ObservationEncoderConfig};
        let config = ObservationEncoderConfig::default();
        let encoded = encode(&obs, &config).unwrap();
        let frame = s.game().semantic_decision_frame().unwrap();
        for bracelet in bracelets {
            let grantor = s.game().state.permanent_of(bracelet).unwrap();
            let expected = format!(
                "exile The Dominion Bracelet (permanent {})",
                s.game().state.permanents[grantor].as_ref().unwrap().id.0
            );
            let action_index = obs
                .action_space
                .actions
                .iter()
                .position(|a| {
                    a.description
                        .as_ref()
                        .is_some_and(|d| d.contains(&expected))
                })
                .unwrap();
            let offer = &obs.action_space.actions[action_index];
            let label = &frame.offers[action_index].label;
            assert!(label.starts_with("Activate ability of Gray Ogre:"));
            assert!(label.contains(&expected));
            let offset = action_index * config.max_focus_objects;
            for (slot, id) in offer.focus.iter().enumerate() {
                let row = obs
                    .agent_permanents
                    .iter()
                    .position(|p| p.id == *id)
                    .unwrap();
                assert_eq!(
                    encoded.action_focus[offset + slot] as usize,
                    2 + 2 * config.max_cards_per_player + row
                );
            }
            assert_eq!(
                offer.focus,
                [ogre, grantor]
                    .map(|id| s.game().state.permanents[id].as_ref().unwrap().id.0 as i32)
            );
        }
        let action_index = s.action_space().actions.iter().position(|action| matches!(action,
            Action::ActivateAbility { permanent, ability_index, .. } if *permanent == ogre && *ability_index == index)).unwrap();
        let mut indexed = crate::support::play::fork_stepping(&s, action_index);
        let command = managym::decision::Command {
            command_id: format!("bracelet-{selected}"),
            expected_revision: frame.revision,
            offer_id: frame.offers[action_index].id.0,
            answers: vec![],
            object_preconditions: vec![],
        };
        s.game_mut().execute_semantic_command(&command).unwrap();
        assert!(indexed.choose_target(Target::Player(PlayerId(1))));
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        if selected == 0 {
            let root = s.game().state.deterministic_hash();
            let choices = s.action_space().actions.len();
            assert_eq!(choices, 2, "{:?}", s.action_space());
            let mut colors = Vec::new();
            for choice in 0..choices {
                let fork = crate::support::play::fork_stepping(&s, choice);
                let replay = crate::support::play::fork_stepping(&s, choice);
                let paid = &top_activation(&fork).paid;
                assert_eq!(paid, &top_activation(&replay).paid);
                assert_eq!(
                    fork.game().state.deterministic_hash(),
                    replay.game().state.deterministic_hash()
                );
                assert_eq!(
                    paid.contributed(How::Exiled, Toward::Printed)
                        .collect::<Vec<_>>(),
                    [bracelets[selected]]
                );
                let color = paid.mana()[0].color;
                colors.push(color);
                let pool = &fork.game().state.players[0].mana_pool;
                assert_eq!(pool.total(), 1);
                assert_eq!(
                    pool.mana().restricted_of(&restriction)[5],
                    u8::from(color != 5)
                );
            }
            colors.sort();
            assert_eq!(colors, [1, 5]);
            assert_eq!(s.game().state.deterministic_hash(), root);
            s.step_action(0);
            indexed.step_action(0);
        } else {
            assert_eq!(top_activation(&s).paid.mana()[0].color, 5);
        }
        let ability = top_activation(&s);
        assert_eq!(ability.paid, top_activation(&indexed).paid);
        assert_eq!(
            s.game().state.players[0].mana_pool,
            indexed.game().state.players[0].mana_pool
        );
        for bracelet in bracelets {
            assert_eq!(zone_of(&s, bracelet), zone_of(&indexed, bracelet));
        }
        assert_eq!(
            ability
                .paid
                .contributed(How::Exiled, Toward::Printed)
                .collect::<Vec<_>>(),
            [bracelets[selected]]
        );
        assert_eq!(zone_of(&s, bracelets[selected]), Some(ZoneType::Exile));
        assert_eq!(
            zone_of(&s, bracelets[1 - selected]),
            Some(ZoneType::Battlefield)
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 1);
        if selected == 1 {
            crate::support::play::counter_stacked_ability(&mut s);
        } else {
            settle(&mut s);
        }
        s.advance_to_active_step(1, StepKind::Upkeep);
        assert_eq!(
            s.game().decision_controller(PlayerId(1)),
            PlayerId(selected)
        );
    }
}

/// Unravel counters a spell and draws when less mana was spent to cast it
/// than its mana value on the stack (CR 608.2h): a discounted Tolarian
/// Terror draws; a full-price one and an X spell paid in full do not.
#[test]
fn unravel_draws_when_the_countered_spell_cost_less_than_its_mana_value() {
    use crate::support::{
        game::{five_color_setup, force_new_card_in_hand, plant_in_graveyard, Scenario},
        inspect::{top_spell, zone_of},
        play::{cast_card, choose_mode},
    };
    use managym::state::{game_object::Target, player::ManaPayment, zone::ZoneType};
    let unravel = |discount: usize, name: &str, x: Option<usize>, spent: u16| -> bool {
        let mut s: Scenario =
            five_color_setup(&["Unravel", "Tolarian Terror", "Consider", "Exsanguinate"]);
        s.game_mut().mana_payment = [ManaPayment::Auto; 2];
        for _ in 0..discount {
            plant_in_graveyard(&mut s, 0, "Consider");
        }
        let card = force_new_card_in_hand(&mut s, 0, name);
        cast_card(&mut s, card);
        if let Some(x) = x {
            choose_mode(&mut s, x);
        }
        let spell = top_spell(&s).object;
        assert_eq!(top_spell(&s).spent.total(), spent);
        s.pass_priority();
        let unravel = force_new_card_in_hand(&mut s, 1, "Unravel");
        let hand = s.zone_size(1, ZoneType::Hand);
        cast_card(&mut s, unravel);
        assert!(s.choose_target(Target::StackSpell(spell)));
        settle(&mut s);
        assert_eq!(zone_of(&s, spell), Some(ZoneType::Graveyard), "countered");
        s.zone_size(1, ZoneType::Hand) == hand
    };
    assert!(unravel(4, "Tolarian Terror", None, 3));
    assert!(!unravel(0, "Tolarian Terror", None, 7));
    assert!(!unravel(0, "Exsanguinate", Some(2), 4));
}

/// Plasma Bolt deals 3 instead of 2 once a nonland permanent left the
/// battlefield this turn, whoever controlled it, or once any player warped
/// a spell; a departing land is not enough.
#[test]
fn plasma_bolt_deals_three_after_a_nonland_departure_or_a_warp() {
    use crate::support::{
        game::{five_color_setup, force_new_card_in_hand, move_card},
        play::{cast_card, settle},
    };
    use managym::state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    };
    let bolt = |s: &mut crate::support::game::Scenario| {
        let life = s.life(1);
        s.set_player_mana_pool(0, "R");
        let card = force_new_card_in_hand(s, 0, "Plasma Bolt");
        cast_card(s, card);
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        settle(s);
        life - s.life(1)
    };
    let mut s = five_color_setup(&["Plasma Bolt", "Gray Ogre"]);
    assert_eq!(bolt(&mut s), 2, "nothing left");
    let land = s.force_permanent_on_battlefield(0, "Mountain");
    let land = s.game().state.permanents[land].as_ref().unwrap().object;
    move_card(&mut s, land, ZoneType::Graveyard);
    assert_eq!(bolt(&mut s), 2, "a land left");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ogre = s.game().state.permanents[ogre].as_ref().unwrap().object;
    move_card(&mut s, ogre, ZoneType::Graveyard);
    assert_eq!(bolt(&mut s), 3, "the opponent's creature left");

    let mut s = five_color_setup(&["Plasma Bolt"]);
    s.game_mut().state.turn.history[1].warped_this_turn = true;
    assert_eq!(bolt(&mut s), 3, "the opponent warped");
}
