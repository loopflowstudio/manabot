//! Tests for cards filed in `src/bro.rs`, in card order.

use crate::support::{
    game::{five_color_setup, five_color_unfunded_setup, plant_in_graveyard, Scenario},
    inspect::{can_cast, find_owned_card, offered_targets, permanent, permanent_named},
    play::{activate_card, attack_with, cast, pass_until},
};
use managym::{
    agent::action::ActionSpaceKind,
    flow::turn::StepKind,
    state::{
        card::Keyword,
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

/// Resolve to rest, choosing `target` whenever a target is asked.
fn settle_targeting(s: &mut Scenario, target: Target) {
    for _ in 0..40 {
        if s.action_space().kind == ActionSpaceKind::ChooseTarget {
            assert!(offered_targets(s).contains(&target), "{target:?} offered");
            assert!(s.choose_target(target));
        } else if s.action_space().kind == ActionSpaceKind::OrderTriggers {
            s.step_action(0);
        } else if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.pending_triggers.is_empty()
        {
            return;
        } else {
            s.pass_priority();
        }
    }
    panic!("stack did not settle");
}

fn powerstones(s: &Scenario, player: usize) -> Vec<bool> {
    s.battlefield_permanents_named(player, "Powerstone")
        .into_iter()
        .map(|id| permanent(s, id).tapped)
        .collect()
}

#[test]
fn cityscape_leveler_cast_trigger_destroys_and_pays_the_controller_a_tapped_powerstone() {
    let mut s = five_color_setup(&["Cityscape Leveler", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast(&mut s, "Cityscape Leveler");
    settle_targeting(&mut s, Target::Permanent(ogre));
    assert!(s.game().state.permanents[ogre].is_none());
    assert_eq!(powerstones(&s, 1), vec![true]);
    assert!(powerstones(&s, 0).is_empty());
    assert_eq!(
        s.battlefield_permanents_named(0, "Cityscape Leveler").len(),
        1
    );
}

#[test]
fn powerstone_mana_pays_for_artifact_spells_but_not_nonartifact_spells() {
    for (pool, name, castable) in [
        ("1R", "Gray Ogre", false),
        ("2R", "Gray Ogre", true),
        ("7", "Cityscape Leveler", true),
        ("6", "Cityscape Leveler", false),
    ] {
        let mut s = five_color_unfunded_setup(&["Cityscape Leveler", "Gray Ogre"]);
        s.force_card_in_hand(0, name);
        s.game_mut()
            .create_token("token.powerstone", PlayerId(0), false)
            .expect("powerstone");
        s.set_player_mana_pool(0, pool);
        s.game_mut().scenario_refresh_priority().unwrap();
        let object = find_owned_card(&s, 0, name);
        assert_eq!(
            can_cast(&s, object),
            castable,
            "{name} with {pool} and a Powerstone"
        );
    }
}

#[test]
fn unearthed_cityscape_leveler_has_haste_and_is_exiled_instead_of_destroyed() {
    let mut s = five_color_setup(&["Cityscape Leveler"]);
    let object = plant_in_graveyard(&mut s, 0, "Cityscape Leveler");
    activate_card(&mut s, object, 0);
    settle_targeting(&mut s, Target::Player(PlayerId(1)));
    let leveler = permanent_named(&s, "Cityscape Leveler");
    assert!(s
        .game()
        .characteristics(leveler)
        .keywords
        .has(Keyword::Haste));
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, leveler);
    // Its attack trigger destroys the Leveler itself: it is exiled instead,
    // and its controller still creates the Powerstone.
    settle_targeting(&mut s, Target::Permanent(leveler));
    assert_eq!(s.game().state.zones.zone_of(object), Some(ZoneType::Exile));
    assert_eq!(powerstones(&s, 0), vec![true]);
}

#[test]
fn unearthed_cityscape_leveler_is_exiled_at_the_next_end_step() {
    let mut s = five_color_setup(&["Cityscape Leveler"]);
    let object = plant_in_graveyard(&mut s, 0, "Cityscape Leveler");
    activate_card(&mut s, object, 0);
    settle_targeting(&mut s, Target::Player(PlayerId(1)));
    assert_eq!(
        s.game().state.zones.zone_of(object),
        Some(ZoneType::Battlefield)
    );
    s.advance_to_step(StepKind::End);
    settle_targeting(&mut s, Target::Player(PlayerId(1)));
    assert_eq!(s.game().state.zones.zone_of(object), Some(ZoneType::Exile));
}
