//! Core Set 2021 card behavior.
use crate::support::{
    game::{five_color_setup, move_card},
    inspect::{card_of, permanent, zone_of},
    play::{activate, cast, settle},
};
use managym::state::{game_object::Target, zone::ZoneType};

#[test]
fn heartfire_immolator_deals_its_power_at_sacrifice_after_returning() {
    for expected in [2, 3] {
        let mut s = five_color_setup(&["Heartfire Immolator", "Craw Wurm", "Lightning Bolt"]);
        let source = s.force_permanent_on_battlefield(0, "Heartfire Immolator");
        let card = card_of(&s, source);
        let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
        if expected == 3 {
            cast(&mut s, "Lightning Bolt");
            assert!(s.choose_target(Target::Player(managym::state::game_object::PlayerId(1))));
            settle(&mut s);
        }
        assert_eq!(s.game().characteristics(source).power, expected);
        activate(&mut s, source, 0);
        assert!(crate::support::inspect::offered_targets(&s).contains(&Target::Permanent(source)));
        assert!(s.choose_target(Target::Permanent(wurm)));
        assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
        move_card(&mut s, card, ZoneType::Battlefield);
        let obs = managym::agent::observation::Observation::for_player(
            s.game(),
            managym::state::game_object::PlayerId(1),
        );
        let [ability] = obs.stack_objects.as_slice() else {
            panic!("expected pending Immolator activation")
        };
        assert_eq!(ability.source_power, Some(expected));
        assert!(ability
            .description
            .as_ref()
            .unwrap()
            .contains(&format!("source power {expected}")));
        settle(&mut s);
        assert_eq!(permanent(&s, wurm).damage, expected);
    }
}

/// The reduction reads the turn's total gain as the cost is determined
/// (CR 601.2f): several gains add up and a later loss does not undo them.
#[test]
fn sanguine_indulgence_costs_three_less_after_three_life_gained_this_turn() {
    use crate::support::{
        game::{force_new_card_in_hand, plant_in_graveyard},
        inspect::{now, top_spell},
        play::{cast_card, select_card},
    };
    use managym::state::{game_object::PlayerId, player::ManaPayment};
    // Each gain sacrifices a creature to Ayli for its toughness.
    for (gains, bolt_self, spent, returned) in [
        (&[][..], false, 4, 2),
        (&["Gray Ogre"][..], false, 4, 1),
        (&["Gray Ogre", "Savannah Lions"][..], true, 1, 2),
        (&["Hill Giant"][..], false, 1, 0),
    ] {
        let mut s = five_color_setup(&[
            "Sanguine Indulgence",
            "Ayli, Eternal Pilgrim",
            "Lightning Bolt",
            "Gray Ogre",
            "Hill Giant",
            "Savannah Lions",
        ]);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        let ayli = s.force_permanent_on_battlefield(0, "Ayli, Eternal Pilgrim");
        for name in gains {
            let creature = s.force_permanent_on_battlefield(0, name);
            let card = card_of(&s, creature);
            activate(&mut s, ayli, 0);
            select_card(&mut s, card);
            settle(&mut s);
        }
        if bolt_self {
            let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
            cast_card(&mut s, bolt);
            s.choose_target(Target::Player(PlayerId(0)));
            settle(&mut s);
        }
        let creatures: Vec<_> = ["Gray Ogre", "Hill Giant"][..returned]
            .iter()
            .map(|name| plant_in_graveyard(&mut s, 0, name))
            .collect();
        let spell = force_new_card_in_hand(&mut s, 0, "Sanguine Indulgence");
        cast_card(&mut s, spell);
        for creature in &creatures {
            let card = now(&s, *creature);
            assert!(s.choose_target(Target::GraveyardCard(card)));
        }
        if s.action_space().kind == managym::agent::action::ActionSpaceKind::ChooseTarget {
            // Up to two: finish with fewer.
            crate::support::play::choose(&mut s, |a| {
                matches!(a, managym::agent::action::Action::Decline { .. })
            });
        }
        assert_eq!(top_spell(&s).spent.total(), spent, "gains {gains:?}");
        settle(&mut s);
        for creature in creatures {
            assert_eq!(zone_of(&s, now(&s, creature)), Some(ZoneType::Hand));
        }
    }
}
