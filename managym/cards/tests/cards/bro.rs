//! Tests for cards filed in `src/bro.rs`, in card order.

use crate::support::inspect::zone_of;
use crate::support::{
    game::{five_color_setup, five_color_unfunded_setup, plant_in_graveyard, Scenario},
    inspect::{can_cast, find_owned_card, offered_targets, permanent, permanent_named},
    play::{activate_card, attack_with, cast, pass_until, spend_colorless},
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
    spend_colorless(&mut s);
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
    spend_colorless(&mut s);
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
    assert_eq!(zone_of(&s, object), Some(ZoneType::Exile));
    assert_eq!(powerstones(&s, 0), vec![true]);
}

#[test]
fn unearthed_cityscape_leveler_is_exiled_at_the_next_end_step() {
    let mut s = five_color_setup(&["Cityscape Leveler"]);
    let object = plant_in_graveyard(&mut s, 0, "Cityscape Leveler");
    activate_card(&mut s, object, 0);
    spend_colorless(&mut s);
    settle_targeting(&mut s, Target::Player(PlayerId(1)));
    assert_eq!(zone_of(&s, object), Some(ZoneType::Battlefield));
    s.advance_to_step(StepKind::End);
    settle_targeting(&mut s, Target::Player(PlayerId(1)));
    assert_eq!(zone_of(&s, object), Some(ZoneType::Exile));
}

/// Goblin Firebomb pays {7}, its tap and itself to destroy any permanent, a
/// land included, and has flash.
#[test]
fn goblin_firebomb_sacrifices_itself_to_destroy_any_permanent_and_has_flash() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands},
        inspect::{card_of, zone_of},
        play::{activate, settle},
    };
    use managym::state::{game_object::Target, player::ManaPayment, zone::ZoneType};
    let mut s = five_color_unfunded_setup(&["Goblin Firebomb"]);
    let bomb = s.force_permanent_on_battlefield(0, "Goblin Firebomb");
    let bomb_card = card_of(&s, bomb);
    force_lands(&mut s, 0, "Mountain", 7);
    let plains = s.force_permanent_on_battlefield(1, "Plains");
    let plains_card = card_of(&s, plains);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    activate(&mut s, bomb, 0);
    assert!(s.choose_target(Target::Permanent(plains)));
    assert_eq!(zone_of(&s, bomb_card), Some(ZoneType::Graveyard));
    assert_eq!(s.untapped_permanents_named(0, "Mountain"), 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, plains_card), Some(ZoneType::Graveyard));

    crate::support::play::casts_at_instant_speed("Goblin Firebomb");
}

#[test]
fn foundations_5d_bushwhack() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{
        card_of, library_card, offered_cards, offered_targets, permanent, zone_of,
    };
    use crate::support::play::{cast, choose_mode, pass_until, select_card, settle};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Bushwhack", "Gray Ogre", "Hill Giant"]);
    let ally = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Hill Giant");
    cast(&mut s, "Bushwhack");
    assert!(crate::support::inspect::mode_labels(&s)[1].contains("fights"));
    let mut search = s.clone();
    choose_mode(&mut search, 0);
    pass_until(&mut search, ActionSpaceKind::LookAndSelect);
    assert_eq!(search.action_space().player, Some(PlayerId(0)));
    let forest = library_card(&search, 0, "Forest");
    assert!(!offered_cards(&search).contains(&library_card(&search, 0, "Gray Ogre")));
    let definition = search.game().state[forest].definition_id;
    select_card(&mut search, forest);
    settle(&mut search);
    assert_eq!(zone_of(&search, forest), Some(ZoneType::Hand));
    assert!(search.game().state.events.iter().any(|event| matches!(
        event,
        managym::flow::event::GameEvent::CardRevealed { definition: shown, .. } if *shown == definition
    )));
    assert_eq!(
        search.game().state.players[0].known_hand.get(&definition),
        Some(&1)
    );
    choose_mode(&mut s, 1);
    assert!(!offered_targets(&s).contains(&Target::Permanent(enemy)));
    assert!(s.choose_target(Target::Permanent(ally)));
    assert!(!offered_targets(&s).contains(&Target::Permanent(ally)));
    assert!(s.choose_target(Target::Permanent(enemy)));
    let ogre = card_of(&s, ally);
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Graveyard));
    assert_eq!(permanent(&s, enemy).damage, 2);
}

#[test]
fn foundations_5e_deadly_riposte() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, offered_targets, zone_of};
    use crate::support::play::{cast, settle};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Deadly Riposte", "Gray Ogre"]);
    let untapped = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let tapped = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[tapped]
        .as_mut()
        .unwrap()
        .tapped = true;
    let card = card_of(&s, tapped);
    cast(&mut s, "Deadly Riposte");
    assert_eq!(offered_targets(&s), [Target::Permanent(tapped)]);
    assert!(s.choose_target(Target::Permanent(tapped)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
    assert!(s.game().state.permanents[untapped].is_some());
    assert_eq!(s.life(0), 22);
}
