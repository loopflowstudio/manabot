//! Tests for cards filed in `src/kld.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::tapped,
    play::{cast_owned, settle, spend_colorless},
};

#[test]
fn dramatic_reversal_untaps_your_nonland_permanents_only() {
    let mut s = five_color_setup(&["Dramatic Reversal", "Gray Ogre", "Sol Ring"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ring = s.force_permanent_on_battlefield(0, "Sol Ring");
    let island = s.force_permanent_on_battlefield(0, "Island");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for id in [ogre, ring, island, foe] {
        s.game_mut().state.permanents[id].as_mut().unwrap().tapped = true;
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Dramatic Reversal");
    settle(&mut s);
    assert!(!tapped(&s, ogre));
    assert!(!tapped(&s, ring));
    assert!(tapped(&s, island));
    assert!(tapped(&s, foe));
}

#[test]
fn fumigate_counts_destroyed_creatures_even_when_their_destination_changes() {
    use crate::support::{
        game::set_counters,
        inspect::{card_of, zone_of},
        play::cast,
    };
    use managym::{
        flow::event::GameEvent,
        state::{game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
    };

    let mut s = five_color_setup(&["Fumigate", "Gray Ogre", "Predator Ooze"]);
    let survivor = s.force_permanent_on_battlefield(1, "Predator Ooze");
    let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    set_counters(&mut s, foe, CounterKind::Finality, 1);
    let (own, foe) = (card_of(&s, own), card_of(&s, foe));

    for expected_gains in [vec![(20, 22)], vec![]] {
        s.drain_events();
        s.set_player_mana_pool(0, "3WW");
        cast(&mut s, "Fumigate");
        settle(&mut s);
        let gains: Vec<_> = s
            .drain_events()
            .into_iter()
            .filter_map(|event| match event {
                GameEvent::LifeChanged {
                    player: PlayerId(0),
                    old,
                    new,
                    ..
                } => Some((old, new)),
                _ => None,
            })
            .collect();
        assert_eq!(gains, expected_gains);
        s.assert_life(0, 22);
        assert_eq!(zone_of(&s, own), Some(ZoneType::Graveyard));
        assert_eq!(zone_of(&s, foe), Some(ZoneType::Exile));
        assert!(s.game().state.permanents[survivor].is_some());
    }
}

#[test]
fn noxious_gearhulk_gains_the_destroyed_creatures_toughness_only_if_destroyed() {
    use crate::support::{
        game::until_end_of_turn,
        play::{cast, choose},
    };
    use managym::{
        agent::action::Action,
        state::{
            card::{Keyword, Keywords},
            effect::Modification,
            game_object::Target,
        },
    };
    // (accept, indestructible, expected life)
    for (accept, indestructible, life) in [(true, false, 23), (false, false, 20), (true, true, 20)]
    {
        let mut s = five_color_setup(&["Noxious Gearhulk", "Agent of Treachery"]);
        let victim = s.force_permanent_on_battlefield(1, "Agent of Treachery");
        if indestructible {
            until_end_of_turn(
                &mut s,
                victim,
                Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
            );
        }
        cast(&mut s, "Noxious Gearhulk");
        spend_colorless(&mut s);
        for _ in 0..10 {
            let space = s.action_space().clone();
            if space
                .actions
                .iter()
                .any(|a| matches!(a, Action::ChooseTarget { .. }))
            {
                assert!(s.choose_target(Target::Permanent(victim)));
            } else if space
                .actions
                .iter()
                .any(|a| matches!(a, Action::PayCost { .. }))
            {
                if accept {
                    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
                } else {
                    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
                }
            } else if s.game().state.stack_objects.is_empty()
                && s.game().state.pending_triggers.is_empty()
            {
                break;
            } else {
                s.pass_priority();
            }
        }
        settle(&mut s);
        let destroyed = accept && !indestructible;
        assert_eq!(s.game().state.permanents[victim].is_none(), destroyed);
        s.assert_life(0, life);
    }
}

/// Chosen-colour mana and crew on one permanent (CR 702.122a, 601.2h): the
/// mana the Caravan makes names the Caravan on the spell it pays for, and its
/// crew ability's receipt names the creatures tapped for it. Manual offers
/// every creature that can help, and so does auto: it pays only the mana.
#[test]
fn cultivators_caravan_mana_names_the_caravan_and_crew_names_the_tapped_creatures() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::{card_of, offered_cards, top_activation},
        play::{activate, cast_card, select_card},
    };
    use managym::{
        agent::action::Action,
        state::{
            game_object::PlayerId,
            mana::Color,
            payment::{Contribution, How, Toward},
            player::ManaPayment,
            stack_object::StackObject,
        },
    };

    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&[
            "Cultivator's Caravan",
            "Gray Ogre",
            "Hill Giant",
            "Llanowar Elves",
        ]);
        let caravan = s.force_permanent_on_battlefield(0, "Cultivator's Caravan");
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
        let spell = force_new_card_in_hand(&mut s, 0, "Llanowar Elves");
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        s.game_mut().scenario_refresh_priority().unwrap();
        let tapped_by = |s: &crate::support::game::Scenario, objects: &[_]| -> Vec<Contribution> {
            objects
                .iter()
                .map(|permanent| Contribution {
                    object: card_of(s, *permanent),
                    how: How::Tapped,
                    toward: Toward::Crew,
                    after_mana: 0,
                })
                .collect()
        };

        // {G} for Llanowar Elves can only come from the Caravan's green mode.
        cast_card(&mut s, spell);
        let Some(StackObject::Spell(cast)) = s.game().state.stack_objects.last() else {
            panic!("Llanowar Elves is on the stack, auto={auto}")
        };
        let [unit] = cast.spent.mana() else {
            panic!("one mana was spent")
        };
        assert_eq!(unit.color, Color::Green as u8);
        assert_eq!(unit.source.unwrap().object, card_of(&s, caravan));
        assert!(tapped(&s, caravan));
        settle(&mut s);
        // The Elves that resolved is summoning sick, and can still crew.
        let elves = crate::support::inspect::permanent_named(&s, "Llanowar Elves");

        // Crew 3 with a 2/2, a 3/3 and a 1/1: a tapped Vehicle can be crewed.
        activate(&mut s, caravan, 1);
        let mut offered = offered_cards(&s);
        offered.sort();
        let mut expected = vec![card_of(&s, ogre), card_of(&s, giant), card_of(&s, elves)];
        expected.sort();
        assert_eq!(offered, expected, "auto={auto}");
        let (ogre_object, elves_object) = (card_of(&s, ogre), card_of(&s, elves));
        select_card(&mut s, ogre_object);
        // One power is still owed; the tapped Ogre is no longer offered.
        let mut offered = offered_cards(&s);
        offered.sort();
        let mut expected = vec![card_of(&s, giant), elves_object];
        expected.sort();
        assert_eq!(offered, expected, "auto={auto}");
        // This step offers no payment to preview; the pending decision
        // itself says what has paid, and names the creature still there.
        assert!(!s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::PayCost { .. })));
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        let [paid] = context.paid.as_slice() else {
            panic!("one creature has crewed, auto={auto}: {:?}", context.paid)
        };
        assert_eq!(paid.phrase, "tapped Gray Ogre to crew");
        let seen = paid.object.as_ref().unwrap();
        assert_eq!(
            (seen.label.as_str(), seen.display_id),
            (
                "Gray Ogre",
                s.game().state.permanents[ogre].as_ref().unwrap().id.0
            )
        );
        assert!(
            s.game()
                .decision_context(PlayerId(1))
                .unwrap()
                .paid
                .is_empty(),
            "the opponent is not told the acting player's payment"
        );
        // A fork at the second crew choice: either creature left finishes it.
        for chosen in [card_of(&s, giant), elves_object] {
            let fork = crate::support::play::fork_selecting(&s, chosen);
            let ability = top_activation(&fork);
            assert_eq!(
                ability
                    .paid
                    .contributed(How::Tapped, Toward::Crew)
                    .collect::<Vec<_>>(),
                [ogre_object, chosen],
                "auto={auto}"
            );
        }
        select_card(&mut s, elves_object);
        let crew = [ogre, elves];
        let ability = top_activation(&s);
        assert_eq!(ability.paid.contributions(), tapped_by(&s, &crew));
        assert!(ability.paid.mana().is_empty());
        for creature in [ogre, giant, elves] {
            assert_eq!(
                tapped(&s, creature),
                crew.contains(&creature),
                "auto={auto}"
            );
        }
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert!(
            context
                .pending
                .iter()
                .any(|line| line.starts_with("Paid with: tapped ") && line.ends_with(" to crew.")),
            "{:?}",
            context.pending
        );
        settle(&mut s);
        assert!(s.game().characteristics(caravan).is_creature());
    }
}

/// Tapped at priority, the Caravan's chosen-colour unit waits in the pool and
/// still names the Caravan on the spell it later pays for, under either
/// payment setting.
#[test]
fn cultivators_caravan_mana_floated_first_still_names_the_caravan() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::{card_of, top_spell},
        play::{cast_card, choose},
    };
    use managym::{
        agent::action::Action,
        state::{mana::Color, player::ManaPayment},
    };
    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&["Cultivator's Caravan", "Llanowar Elves"]);
        let caravan = s.force_permanent_on_battlefield(0, "Cultivator's Caravan");
        let spell = force_new_card_in_hand(&mut s, 0, "Llanowar Elves");
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        s.allow_priority_floats();
        choose(
            &mut s,
            |a| matches!(a, Action::ActivateAbility { permanent, ability_index: 0, .. } if *permanent == caravan),
        );
        choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
        assert!(tapped(&s, caravan), "auto={auto}");
        assert_eq!(s.game().state.players[0].mana_pool.total(), 1);

        cast_card(&mut s, spell);
        if s.game().state.stack_objects.is_empty() {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
        let [unit] = top_spell(&s).spent.mana() else {
            panic!("one mana was spent, auto={auto}")
        };
        assert_eq!(unit.color, Color::Green as u8);
        assert_eq!(unit.source.unwrap().object, card_of(&s, caravan));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn authority_checks_entering_control() {
    use crate::support::{inspect::permanent_of, play::resolve_top};
    use managym::flow::event::GameEvent;
    use managym::state::{game_object::PlayerId, stack_object::StackObject};

    for (authority, entrant, should_trigger) in [
        (0, "Gray Ogre", true),
        (1, "Gray Ogre", false),
        (0, "Sol Ring", false),
    ] {
        let mut s = five_color_setup(&["Authority of the Consuls", entrant]);
        s.force_permanent_on_battlefield(authority, "Authority of the Consuls");
        let spell = cast_owned(&mut s, entrant);
        spend_colorless(&mut s);
        let Some(StackObject::Spell(top)) = s.game_mut().state.stack_objects.last_mut() else {
            panic!("expected a permanent spell");
        };
        // The other player takes control before the owner's spell resolves.
        top.controller = PlayerId(1);
        resolve_top(&mut s);
        let id = permanent_of(&s, spell).expect("spell entered the battlefield");
        assert_eq!(
            s.game().state.permanents[id].as_ref().unwrap().controller,
            PlayerId(1)
        );
        assert_eq!(tapped(&s, id), should_trigger);
        assert_eq!([s.life(0), s.life(1)], [20, 20]);
        if !should_trigger {
            assert!(s.game().state.stack_objects.is_empty());
            continue;
        }

        assert!(matches!(
            s.game().state.stack_objects.as_slice(),
            [StackObject::TriggeredAbility(_)]
        ));
        s.drain_events();
        let mut fork = s.clone();
        for branch in [&mut s, &mut fork] {
            resolve_top(branch);
            assert_eq!([branch.life(0), branch.life(1)], [21, 20]);
            assert!(branch.game().state.stack_objects.is_empty());
        }
        let events = s.drain_events();
        assert_eq!(events, fork.drain_events());
        let gains: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                GameEvent::LifeChanged {
                    player, old, new, ..
                } => Some((*player, *old, *new)),
                _ => None,
            })
            .collect();
        assert_eq!(gains, [(PlayerId(0), 20, 21)]);
    }
}

#[test]
fn printed_creature_brazen_scourge() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat("Brazen Scourge", 3, 3, &[Keyword::Haste]);
}

#[test]
fn foundations_5f_cloudblazer() {
    use crate::support::game::five_color_setup;
    use crate::support::play::enter_and_settle;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Cloudblazer"]);
    let hand = s.zone_size(0, ZoneType::Hand);
    enter_and_settle(&mut s, "Cloudblazer");
    s.assert_life(0, 22);
    s.assert_zone_size(0, ZoneType::Hand, hand + 2);
}
