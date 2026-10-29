//! Adventures in the Forgotten Realms (`src/cardsets/afr.rs`).
use crate::support::{
    activate_any, activate_named, can_cast_now, cast, choose, enter_or_cast, enter_settled,
    library_card, permanent, ready_named, select, settle, setup,
};
use managym::{
    agent::action::Action,
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

#[test]
fn selected_sacrifice_cannot_also_fund_the_announced_spell() {
    for mode in 0..3 {
        let mut s = setup(&[
            "Deadly Dispute",
            "Tireless Provisioner",
            "Callous Inspector",
        ]);
        if mode == 2 {
            enter_settled(&mut s, "Callous Inspector");
        }
        s.game_mut().create_token("Treasure", PlayerId(0), false);
        let treasure = permanent(&s, "Treasure");
        let treasure_card = s.game().state.permanents[treasure].as_ref().unwrap().card;
        s.set_player_mana_pool(0, if mode == 1 { "BB" } else { "B" });
        let dispute = library_card(&s, 0, "Deadly Dispute");
        s.game_mut().move_card(dispute, ZoneType::Hand);
        s.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(
            can_cast_now(&s, dispute),
            mode > 0,
            "one Treasure cannot be both mana and sacrifice"
        );
        if mode == 0 {
            continue;
        }
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { card, .. } if *card == dispute),
        );
        let offers_treasure = s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::SelectCard { card, .. } if *card == treasure_card));
        assert_eq!(offers_treasure, mode == 1);
        if mode == 1 {
            choose(
                &mut s,
                |a| matches!(a, Action::SelectCard { card, .. } if *card == treasure_card),
            );
        } else {
            let inspector = permanent(&s, "Callous Inspector");
            let inspector_card = s.game().state.permanents[inspector].as_ref().unwrap().card;
            choose(
                &mut s,
                |a| matches!(a, Action::SelectCard { card, .. } if *card == inspector_card),
            );
            activate_named(&mut s, "Treasure", 0);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
        assert!(s.game().state.permanents[treasure].is_none());
        assert_eq!(s.game().state.zones.zone_of(dispute), Some(ZoneType::Stack));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn a_mana_tap_can_precede_a_distinct_sacrifice_cost() {
    for pool in ["0", "B"] {
        let mut s = setup(&["Deadly Dispute", "Llanowar Elves"]);
        let elves_card = enter_settled(&mut s, "Llanowar Elves");
        ready_named(&mut s, "Llanowar Elves");
        let elves = permanent(&s, "Llanowar Elves");
        s.set_player_mana_pool(0, pool);
        let dispute = library_card(&s, 0, "Deadly Dispute");
        s.game_mut().move_card(dispute, ZoneType::Hand);
        s.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(can_cast_now(&s, dispute), pool == "B");
        if pool == "B" {
            choose(
                &mut s,
                |a| matches!(a, Action::CastSpell { card, .. } if *card == dispute),
            );
            choose(
                &mut s,
                |a| matches!(a, Action::SelectCard { card, .. } if *card == elves_card),
            );
            // Tap and sacrifice are different costs: G plus pooled B pays 1B,
            // then the still-present tapped creature pays the sacrifice.
            assert!(s.game().state.permanents[elves].is_none());
            assert_eq!(
                s.game().state.zones.zone_of(elves_card),
                Some(ZoneType::Graveyard)
            );
            assert_eq!(s.game().state.zones.zone_of(dispute), Some(ZoneType::Stack));
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        }
    }
}

#[test]
fn removal_respects_mana_value_and_excluded_creature_types() {
    for name in ["Power Word Kill", "Epic Downfall"] {
        let mut s = setup(&[name, "Eyetwitch", "Gnarled Professor"]);
        let small = s.force_permanent_on_battlefield(1, "Eyetwitch");
        let big = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        if name == "Power Word Kill" {
            let card = s.game().state.permanents[small].as_ref().unwrap().card;
            s.game_mut().state.cards[card]
                .subtypes
                .push("Dragon".into());
        }
        cast(&mut s, name);
        assert!(!s.action_space().actions.iter().any(|action| matches!(
            action,
            Action::ChooseTarget { target: managym::state::target::Target::Permanent(id), .. } if *id == small
        )));
        let target_card = s.game().state.permanents[big].as_ref().unwrap().card;
        assert!(s.choose_target(Target::Permanent(big)));
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(target_card),
            Some(if name == "Epic Downfall" {
                ZoneType::Exile
            } else {
                ZoneType::Graveyard
            })
        );
    }
}

#[test]
fn dispute_sacrifices_before_responses_draws_two_and_makes_live_treasure() {
    let mut s = setup(&["Deadly Dispute", "Unlucky Cabbage Merchant"]);
    enter_or_cast(&mut s, "Unlucky Cabbage Merchant");
    let food = permanent(&s, "Food");
    let food_card = s.game().state.permanents[food].as_ref().unwrap().card;
    cast(&mut s, "Deadly Dispute");
    let hand = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    select(&mut s, food_card);
    assert_ne!(
        s.game().state.zones.zone_of(food_card),
        Some(ZoneType::Battlefield)
    );
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand + 1
    );
    let treasure = permanent(&s, "Treasure");
    let stack = s.game().state.stack_objects.len();
    activate_any(&mut s, treasure);
    assert_eq!(s.game().state.stack_objects.len(), stack);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert!(s.battlefield_permanents_named(0, "Treasure").is_empty());
}
