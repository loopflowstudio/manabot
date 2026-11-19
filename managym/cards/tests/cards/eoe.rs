//! Tests for cards filed in `src/eoe.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, set_counters},
    inspect::{card_of, counters_of, has_action, offered_cards, tapped},
    play::{activate, fork_selecting, select_card, settle},
};
use managym::{
    agent::action::Action,
    flow::turn::StepKind,
    state::{
        card::Keyword,
        game_object::PlayerId,
        payment::{How, Toward},
        permanent::CounterKind,
        player::ManaPayment,
        stack_object::StackObject,
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

        // A fork at the tap choice: the counters follow the creature chosen.
        for (creature, power) in creatures {
            let chosen = card_of(&s, creature);
            let mut fork = fork_selecting(&s, chosen);
            let [StackObject::ActivatedAbility(ability)] = fork.state.stack_objects.as_slice()
            else {
                panic!("one station activation, auto={auto}")
            };
            assert_eq!(
                ability
                    .paid
                    .contributed(How::Tapped, Toward::Station)
                    .collect::<Vec<_>>(),
                [chosen]
            );
            let context = fork.decision_context(PlayerId(0)).unwrap();
            assert!(
                context
                    .pending
                    .iter()
                    .any(|line| line.starts_with("Paid with: tapped ")
                        && line.ends_with(" to station.")),
                "{:?}",
                context.pending
            );
            for _ in 0..2 {
                let pass = fork
                    .action_space()
                    .unwrap()
                    .actions
                    .iter()
                    .position(|a| matches!(a, Action::PassPriority { .. }))
                    .unwrap();
                fork.step(pass).unwrap();
            }
            assert!(fork.state.stack_objects.is_empty());
            let stationed = fork.state.permanents[frigate].as_ref().unwrap();
            assert_eq!(
                stationed.counter_count(CounterKind::Charge),
                power,
                "auto={auto}"
            );
            assert!(fork.state.permanents[creature].as_ref().unwrap().tapped);
        }

        let giant_object = card_of(&s, giant);
        select_card(&mut s, giant_object);
        settle(&mut s);
        assert_eq!(counters_of(&s, frigate, CounterKind::Charge), 3);
        assert!(tapped(&s, giant) && !tapped(&s, ogre) && !tapped(&s, frigate));
    }
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
