//! Tests for cards filed in `src/khm.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::keywords,
    play::{activate, offers_ability, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{
        card::Keyword,
        game_object::{PlayerId, Target},
    },
};

fn coils(s: &crate::support::game::Scenario) -> usize {
    s.game()
        .state
        .permanents
        .iter()
        .flatten()
        .filter(|permanent| s.game().state[permanent.object].name == "Koma's Coil")
        .count()
}

/// Koma makes a Coil at each player's upkeep. Sacrificing one taps a target
/// whose other abilities then stay unusable until the turn ends; the other
/// mode makes Koma indestructible.
#[test]
fn koma_coils_each_upkeep_and_sacrifices_them_to_lock_or_protect() {
    let mut s = five_color_setup(&["Koma, Cosmos Serpent"]);
    let koma = s.force_permanent_on_battlefield(0, "Koma, Cosmos Serpent");
    let clue = s
        .game_mut()
        .create_token("token.clue", PlayerId(0), false)
        .unwrap();
    s.game_mut()
        .create_token("token.komas_coil", PlayerId(0), false);
    assert!(offers_ability(&mut s, clue, 0));

    activate(&mut s, koma, 0);
    assert!(s.choose_target(Target::Permanent(clue)));
    settle(&mut s);
    assert_eq!(coils(&s), 0);
    assert!(s.game().state.permanents[clue].as_ref().unwrap().tapped);
    assert!(s.game().activations_prohibited(clue));
    assert!(!offers_ability(&mut s, clue, 0));
    assert!(
        !offers_ability(&mut s, koma, 0),
        "no other Serpent to sacrifice"
    );

    s.game_mut()
        .create_token("token.komas_coil", PlayerId(0), false);
    activate(&mut s, koma, 1);
    settle(&mut s);
    assert!(keywords(&s, koma).has(Keyword::Indestructible));

    s.advance_to_active_step(1, StepKind::Draw);
    assert_eq!(coils(&s), 1, "the opponent's upkeep makes a Coil");
    assert!(!s.game().activations_prohibited(clue));
    assert!(!keywords(&s, koma).has(Keyword::Indestructible));
}

/// Seize the Spoils discards a chosen card as it is cast, then draws two and
/// makes a Treasure.
#[test]
fn seize_the_spoils_discards_a_chosen_card_then_draws_two_and_makes_a_treasure() {
    use managym::state::zone::ZoneType;
    for auto in [false, true] {
        let s = crate::support::play::cast_discarding_a_chosen_card("Seize the Spoils", 3, auto);
        // The Ogre and two drawn cards.
        assert_eq!(s.zone_size(0, ZoneType::Hand), 3);
        assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);
    }
}

#[test]
fn fynn_poison_and_herald_loss_protection() {
    use crate::support::{
        game::clear_summoning_sickness,
        play::{attack_with, pass_until},
    };
    use managym::{
        agent::{action::ActionSpaceKind, observation::Observation},
        state::{permanent::CounterKind, zone::ZoneType},
    };
    for protected in [false, true] {
        let mut s = five_color_setup(&["Fynn, the Fangbearer", "Herald of Eternal Dawn"]);
        let fynn = s.force_permanent_on_battlefield(0, "Fynn, the Fangbearer");
        clear_summoning_sickness(&mut s, fynn);
        s.game_mut().state.players[1]
            .counters
            .change(CounterKind::Poison, 8);
        let herald =
            protected.then(|| s.force_permanent_on_battlefield(1, "Herald of Eternal Dawn"));
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, fynn);
        while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
            s.decline_block();
        }
        s.advance_to_step(StepKind::EndOfCombat);
        for viewer in [0, 1] {
            let obs = Observation::for_player(s.game(), PlayerId(viewer));
            let poisoned = if viewer == 1 {
                &obs.agent
            } else {
                &obs.opponent
            };
            assert_eq!(poisoned.counters["poison"], 10);
        }
        assert_eq!(s.game().is_game_over(), !protected);
        if let Some(herald) = herald {
            s.game_mut().state.players[1].drew_when_empty = true;
            s.pass_priority();
            assert!(!s.game().state.players[1].drew_when_empty);
            let object = s.game().state.permanents[herald].as_ref().unwrap().object;
            s.game_mut().move_object(object, ZoneType::Exile);
            s.pass_priority();
        }
        s.assert_winner(0);
        assert_eq!(s.game().state.loss(1).unwrap().reason(), Some("poison"));
    }
}

#[test]
fn axe_equipped_fynn_poisons_in_both_damage_steps() {
    use crate::support::{
        game::clear_summoning_sickness,
        play::{attack_with, pass_until},
    };
    use managym::{agent::action::ActionSpaceKind, state::permanent::CounterKind};

    let mut s = five_color_setup(&["Leyline Axe", "Fynn, the Fangbearer"]);
    let axe = s.force_permanent_on_battlefield(0, "Leyline Axe");
    let fynn = s.force_permanent_on_battlefield(0, "Fynn, the Fangbearer");
    clear_summoning_sickness(&mut s, fynn);
    s.set_player_mana_pool(0, "3");
    activate(&mut s, axe, 0);
    assert!(s.choose_target(Target::Permanent(fynn)));
    settle(&mut s);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, fynn);
    s.advance_to_step(StepKind::EndOfCombat);
    assert_eq!(
        s.game().state.players[1].counters.get(CounterKind::Poison),
        4
    );
    s.assert_life(1, 16);
}

#[test]
fn immersturm_predator_grows_when_it_becomes_tapped_unless_its_one_target_is_gone() {
    use crate::support::{
        game::{move_card, plant_in_graveyard},
        inspect::{card_of, counters_of, offered_targets, tapped, zone_of},
        play::{choose, select_card},
    };
    use managym::state::zone::ZoneType;
    use managym::{agent::action::Action, state::permanent::CounterKind::PlusOnePlusOne};

    let mut s = five_color_setup(&["Immersturm Predator", "Gray Ogre", "Craw Wurm"]);
    let predator = s.force_permanent_on_battlefield(0, "Immersturm Predator");
    let fodder = [0, 1].map(|_| {
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        card_of(&s, ogre)
    });
    let wurm = plant_in_graveyard(&mut s, 1, "Craw Wurm");
    activate(&mut s, predator, 0);
    select_card(&mut s, fodder[0]);
    s.advance_until(
        |s| !offered_targets(s).is_empty(),
        "the tap trigger chooses up to one graveyard card".into(),
    );
    assert!(tapped(&s, predator));
    assert!(keywords(&s, predator).has(Keyword::Indestructible));
    let wurm_target = Target::GraveyardCard(wurm);
    assert!(offered_targets(&s).contains(&wurm_target));
    let pending = s.game().clone();

    assert!(s.choose_target(wurm_target));
    settle(&mut s);
    assert_eq!(zone_of(&s, wurm), Some(ZoneType::Exile));
    assert_eq!(counters_of(&s, predator, PlusOnePlusOne), 1);

    // CR 608.2b: with its one target gone, the ability doesn't resolve.
    *s.game_mut() = pending.clone();
    assert!(s.choose_target(wurm_target));
    move_card(&mut s, wurm, ZoneType::Hand);
    settle(&mut s);
    assert_eq!(counters_of(&s, predator, PlusOnePlusOne), 0);

    // Choosing no target, it still grows.
    *s.game_mut() = pending;
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(counters_of(&s, predator, PlusOnePlusOne), 1);
    assert_eq!(zone_of(&s, wurm), Some(ZoneType::Graveyard));

    // Already tapped, "tap it" doesn't make it become tapped again.
    activate(&mut s, predator, 0);
    select_card(&mut s, fodder[1]);
    settle(&mut s);
    assert_eq!(counters_of(&s, predator, PlusOnePlusOne), 1);
}
