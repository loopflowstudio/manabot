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
/// every creature that can help; auto taps the fewest that reach the number.
#[test]
fn cultivators_caravan_mana_names_the_caravan_and_crew_names_the_tapped_creatures() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::{card_of, offered_cards},
        play::{activate, cast_card, choose, select_card},
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
                })
                .collect()
        };

        // {G} for Llanowar Elves can only come from the Caravan's green mode.
        cast_card(&mut s, spell);
        if !auto {
            choose(
                &mut s,
                |a| matches!(a, Action::ActivateAbility { permanent, ability_index: 0, .. } if *permanent == caravan),
            );
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
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
        let crew = if auto {
            vec![giant]
        } else {
            let mut offered = offered_cards(&s);
            offered.sort();
            let mut expected = vec![card_of(&s, ogre), card_of(&s, giant), card_of(&s, elves)];
            expected.sort();
            assert_eq!(offered, expected);
            let (ogre_object, elves_object) = (card_of(&s, ogre), card_of(&s, elves));
            select_card(&mut s, ogre_object);
            // One power is still owed; the tapped Ogre is no longer offered.
            let mut offered = offered_cards(&s);
            offered.sort();
            let mut expected = vec![card_of(&s, giant), elves_object];
            expected.sort();
            assert_eq!(offered, expected);
            select_card(&mut s, elves_object);
            vec![ogre, elves]
        };
        let Some(StackObject::ActivatedAbility(ability)) = s.game().state.stack_objects.last()
        else {
            panic!("the crew ability is on the stack, auto={auto}")
        };
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
