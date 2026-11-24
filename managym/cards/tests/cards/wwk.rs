//! Tests for cards filed in `src/wwk.rs`, in card order.

use crate::support::{
    game::{five_color_setup, until_end_of_turn},
    inspect::card_of,
    play::{bolt, settle},
};
use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    state::{effect::Modification, game_object::PlayerId, target::Target},
};

/// Place every Kalastria trigger, targeting the opponent; return how many.
fn place_targeting_opponent(s: &mut crate::support::game::Scenario) -> usize {
    for _ in 0..20 {
        match s.action_space().kind {
            ActionSpaceKind::ChooseTarget => {
                assert!(s.choose_target(Target::Player(PlayerId(1))));
            }
            ActionSpaceKind::OrderTriggers => s.step_action(0),
            _ => break,
        }
    }
    s.game().state.stack_objects.len()
}

/// Kalastria Highborn's target player is announced as the trigger is put on
/// the stack (CR 603.3d), before the optional {B} is offered on resolution.
#[test]
fn kalastria_highborn_targets_before_offering_its_payment() {
    let mut s = five_color_setup(&["Kalastria Highborn", "Lightning Bolt"]);
    let highborn = s.force_permanent_on_battlefield(0, "Kalastria Highborn");
    bolt(&mut s, highborn);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert_eq!(place_targeting_opponent(&mut s), 1);
    s.set_player_mana_pool(0, "B");
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    let pending = s.game().clone();
    for pay in [true, false] {
        *s.game_mut() = pending.clone();
        if pay {
            assert!(s.take_action_by_type(ActionType::PayCost));
        } else {
            crate::support::play::choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut s);
        s.assert_life(1, if pay { 18 } else { 20 });
        s.assert_life(0, if pay { 22 } else { 20 });
    }
}

/// The Vampire test reads the dying creature as it last existed: a copy of
/// Kalastria and a creature granted Vampire trigger it; a Vampire whose
/// creature types were replaced does not.
#[test]
fn kalastria_highborn_reads_a_dying_creatures_last_known_types() {
    for (case, expected) in [("copy", 2), ("granted", 1), ("stripped", 0)] {
        let mut s = five_color_setup(&[
            "Kalastria Highborn",
            "Gray Ogre",
            "Stromkirk Bloodthief",
            "Lightning Bolt",
        ]);
        let highborn = s.force_permanent_on_battlefield(0, "Kalastria Highborn");
        let victim = match case {
            "copy" => {
                let island = s.force_permanent_on_battlefield(0, "Island");
                let original = s.game().state[card_of(&s, highborn)].clone();
                let copy = card_of(&s, island);
                s.game_mut().state[copy].become_copy_of(&original, &[]);
                island
            }
            "granted" => {
                let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
                until_end_of_turn(
                    &mut s,
                    ogre,
                    Modification::AddSubtypes(vec!["Vampire".into()]),
                );
                ogre
            }
            _ => {
                let thief = s.force_permanent_on_battlefield(0, "Stromkirk Bloodthief");
                until_end_of_turn(
                    &mut s,
                    thief,
                    Modification::SetCreatureTypes(vec!["Citizen".into()]),
                );
                thief
            }
        };
        bolt(&mut s, victim);
        assert_eq!(place_targeting_opponent(&mut s), expected, "{case}");
    }
}

/// Dragonmaster Outcast makes a 5/5 flying Dragon at its controller's
/// upkeep only while that player controls six or more lands.
#[test]
fn dragonmaster_outcast_makes_a_dragon_with_six_lands() {
    use crate::support::game::{five_color_setup, force_lands};
    use managym::{flow::turn::StepKind, state::card::Keyword};
    for (lands, dragons) in [(5, 0), (6, 1)] {
        let mut s = five_color_setup(&["Dragonmaster Outcast"]);
        s.force_permanent_on_battlefield(0, "Dragonmaster Outcast");
        let have = s.zone_size(0, managym::state::zone::ZoneType::Battlefield) - 1;
        force_lands(&mut s, 0, "Mountain", lands - have.min(lands));
        force_lands(&mut s, 1, "Forest", 6);
        s.advance_to_active_step(1, StepKind::Main);
        s.advance_to_active_step(0, StepKind::Draw);
        let made = s.battlefield_permanents_named(0, "Dragon");
        assert_eq!(made.len(), dragons, "{lands} lands");
        if let Some(&dragon) = made.first() {
            let view = s.game().characteristics(dragon);
            assert_eq!((view.power, view.toughness), (5, 5));
            assert!(view.keywords.has(Keyword::Flying));
        }
    }
}
