//! Caradora's optional search and counter predicate use the shared rules paths.
use crate::support::inspect::{current_object, zone_of};
use crate::support::{
    game::{five_color_setup, put_onto_battlefield},
    inspect::offered_cards,
    play::{choose_mode, pass_until, select_card, settle},
};
use managym::{
    agent::action::ActionSpaceKind,
    flow::event::GameEvent,
    state::{game_object::PlayerId, zone::ZoneType},
};

#[test]
fn caradora_search_reveals_a_vehicle_and_declining_does_not_shuffle() {
    for search in [false, true] {
        let mut s = five_color_setup(&["Caradora, Heart of Alacria", "Phoenix Fleet Airship"]);
        let before = s
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Library, PlayerId(0))
            .to_vec();
        put_onto_battlefield(&mut s, 0, "Caradora, Heart of Alacria");
        pass_until(&mut s, ActionSpaceKind::Modal);
        choose_mode(&mut s, if search { 0 } else { 1 });
        if search {
            assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
            let offered = offered_cards(&s);
            assert!(!offered.is_empty());
            assert!(offered
                .iter()
                .all(|object| s.game().state[*object].name == "Phoenix Fleet Airship"));
            let selected = offered[0];
            select_card(&mut s, selected);
            settle(&mut s);
            assert_eq!(zone_of(&s, selected), Some(ZoneType::Hand));
            let definition = current_object(&s, selected).definition_id;
            assert!(s.game().state.events.iter().any(|event| matches!(event,
                GameEvent::CardRevealed { owner: PlayerId(0), definition: shown } if *shown == definition
            )));
            assert_eq!(
                s.game().state.players[0].known_hand.get(&definition),
                Some(&1)
            );
        } else {
            settle(&mut s);
            // The entry helper can take Caradora out of the library; every
            // remaining member keeps its order when the optional search is declined.
            let after = s
                .game()
                .state
                .zones
                .zone_objects(ZoneType::Library, PlayerId(0));
            let retained: Vec<_> = before
                .into_iter()
                .filter(|object| after.contains(*object))
                .collect();
            assert_eq!(retained, after.to_vec());
        }
    }
}

/// Saddle 1 taps other creatures with total power 1 or more, at sorcery
/// speed, and the ability's receipt names them. The Jaguar gets +2/+2 only
/// when it attacks while saddled.
#[test]
fn alacrian_jaguar_gets_plus_two_only_when_it_attacks_saddled() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_unfunded_setup},
        inspect::{card_of, has_action, tapped},
        play::{activate, attack_with},
    };
    use managym::{
        agent::action::Action,
        flow::turn::StepKind,
        state::{
            payment::{How, Toward},
            stack_object::StackObject,
        },
    };
    for saddled in [false, true] {
        let mut s = five_color_unfunded_setup(&["Alacrian Jaguar", "Llanowar Elves", "Gray Ogre"]);
        let jaguar = s.force_permanent_on_battlefield(0, "Alacrian Jaguar");
        let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        clear_summoning_sickness(&mut s, jaguar);
        let saddles = |a: &Action| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == jaguar);
        if saddled {
            activate(&mut s, jaguar, 0);
            let mut offered = offered_cards(&s);
            offered.sort();
            let mut expected = vec![card_of(&s, elves), card_of(&s, ogre)];
            expected.sort();
            assert_eq!(offered, expected, "the Jaguar cannot saddle itself");
            let frame = s.game().semantic_decision_frame().unwrap();
            assert!(
                frame
                    .offers
                    .iter()
                    .any(|offer| offer.label == "Tap Llanowar Elves to saddle"),
                "{:?}",
                frame.offers
            );
            let rider = card_of(&s, elves);
            select_card(&mut s, rider);
            let Some(StackObject::ActivatedAbility(ability)) = s.game().state.stack_objects.last()
            else {
                panic!("the saddle ability is on the stack")
            };
            assert_eq!(
                ability
                    .paid
                    .contributed(How::Tapped, Toward::Saddle)
                    .collect::<Vec<_>>(),
                [rider]
            );
            settle(&mut s);
            assert!(tapped(&s, elves) && !tapped(&s, ogre) && !tapped(&s, jaguar));
        }
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        attack_with(&mut s, jaguar);
        s.advance_until(
            |sc| sc.game().state.stack_objects.is_empty(),
            "the attack trigger resolves".into(),
        );
        // Saddle only as a sorcery: not during combat.
        assert!(!has_action(&s, saddles));
        let view = s.game().characteristics(jaguar);
        let expected = if saddled { (6, 6) } else { (4, 4) };
        assert_eq!((view.power, view.toughness), expected, "saddled={saddled}");
    }
}
