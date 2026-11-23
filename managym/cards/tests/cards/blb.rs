//! Tests for cards filed in `src/blb.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::permanent_named,
    play::{cast, settle, spend_colorless},
};
use managym::{agent::action::Action, state::game_object::Target};

#[test]
fn savor_makes_food_with_a_real_tap_cost_and_cindermaw_prevents_its_life_gain() {
    for blocked in [false, true] {
        let mut s = five_color_setup(&["Savor", "Gnarled Professor", "Giant Cindermaw"]);
        let target = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        if blocked {
            s.force_permanent_on_battlefield(1, "Giant Cindermaw");
        }
        cast(&mut s, "Savor");
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert_eq!(s.game().characteristics(target).toughness, 2);
        let food = permanent_named(&s, "Food");
        s.game_mut().state.permanents[food].as_mut().unwrap().tapped = true;
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==food)));
        s.game_mut().state.permanents[food].as_mut().unwrap().tapped = false;
        s.game_mut().scenario_refresh_priority().unwrap();
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==food))
            .unwrap();
        let life = s.life(0);
        s.step_action(index);
        spend_colorless(&mut s);
        assert!(s.game().state.permanents[food].is_none());
        settle(&mut s);
        assert_eq!(s.life(0), life + if blocked { 0 } else { 3 });
    }
}

#[test]
fn diresight_surveils_two_draws_two_and_loses_two_life() {
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Diresight"]);
    cast(&mut s, "Diresight");
    spend_colorless(&mut s);
    let hand = s.zone_size(0, ZoneType::Hand);
    let library = s.zone_size(0, ZoneType::Library);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 2);
    let milled = s.zone_size(0, ZoneType::Graveyard) - 1;
    assert!(milled <= 2);
    s.assert_zone_size(0, ZoneType::Library, library - 2 - milled);
    s.assert_life(0, 18);
}

/// Feed the Cycle ({1}{B}) costs a forage or {B} more. The caster picks
/// as the cost is paid: sacrifice a Food, exile three graveyard cards
/// (the first commits to three), or pay {B}; the receipt says which. The
/// automatic payer leaves that choice to the player and pays only the mana.
#[test]
fn feed_the_cycle_forages_with_a_food_or_three_cards_or_pays_black() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand, plant_in_graveyard},
        inspect::{can_cast, has_action, now, offer_labels, offered_cards, top_spell},
        play::{
            cast_card, choose, fork_choosing, fork_selecting, pass_until, pay_rest, select_card,
            settle,
        },
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{
            game_object::{PlayerId, Target},
            payment::{How, Toward},
            player::ManaPayment,
            zone::ZoneType,
        },
    };
    for auto in [false, true] {
        for branch in ["food", "cards", "mana"] {
            let mut s = five_color_unfunded_setup(&["Feed the Cycle", "Gray Ogre", "Hill Giant"]);
            let victim = s.force_permanent_on_battlefield(1, "Gray Ogre");
            s.force_permanent_on_battlefield(0, "Swamp");
            s.force_permanent_on_battlefield(0, "Mountain");
            if auto {
                s.game_mut().mana_payment[0] = ManaPayment::Auto;
            }
            let feed = force_new_card_in_hand(&mut s, 0, "Feed the Cycle");
            // No Food, an empty graveyard and no third land: neither half.
            s.game_mut().scenario_refresh_priority().unwrap();
            assert!(!can_cast(&s, feed));

            s.force_permanent_on_battlefield(0, "Swamp");
            let food = s
                .game_mut()
                .create_token("token.food", PlayerId(0), false)
                .unwrap();
            let food = s.game().state.permanents[food].as_ref().unwrap().object;
            let cards: Vec<_> = (0..3)
                .map(|_| {
                    let card = plant_in_graveyard(&mut s, 0, "Hill Giant");
                    now(&s, card)
                })
                .collect();
            cast_card(&mut s, feed);
            pass_until(&mut s, ActionSpaceKind::ChooseTarget);
            assert!(s.choose_target(Target::Permanent(victim)));

            assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
            let mut offered = offered_cards(&s);
            offered.sort();
            let mut expected = cards.clone();
            expected.push(food);
            expected.sort();
            assert_eq!(offered, expected);
            assert!(has_action(&s, |a| matches!(a, Action::PayCost { .. })));
            let labels = offer_labels(&s);
            for label in [
                "Sacrifice Food (forage)",
                "Exile Hill Giant (forage)",
                "Pay {B} instead of foraging",
            ] {
                assert!(labels.iter().any(|offer| offer == label), "{labels:?}");
            }

            // A fork at the forage choice: each way to pay is its own line,
            // and the cost leaves when its payment completes.
            let finished = |mut fork: crate::support::game::Scenario| {
                if fork.action_space().kind == ActionSpaceKind::PayOrNot {
                    pay_rest(&mut fork);
                }
                (
                    fork.game().state.permanent_of(food).is_some(),
                    fork.zone_size(0, ZoneType::Exile),
                )
            };
            assert_eq!(finished(fork_selecting(&s, food)), (false, 0));
            let mut fork = fork_selecting(&s, cards[0]);
            select_card(&mut fork, cards[1]);
            select_card(&mut fork, cards[2]);
            assert_eq!(finished(fork), (true, 3));
            let pay = |a: &Action| matches!(a, Action::PayCost { .. });
            assert_eq!(finished(fork_choosing(&s, pay)), (true, 0));
            assert!(s.game().state.permanent_of(food).is_some());
            assert_eq!(s.zone_size(0, ZoneType::Exile), 0);

            match branch {
                "food" => select_card(&mut s, food),
                "cards" => {
                    select_card(&mut s, cards[0]);
                    // Committed: two more cards, and nothing else.
                    assert_eq!(offered_cards(&s).len(), 2);
                    assert!(!has_action(&s, |a| matches!(a, Action::PayCost { .. })));
                    select_card(&mut s, cards[1]);
                    select_card(&mut s, cards[2]);
                }
                _ => choose(&mut s, |a| matches!(a, Action::PayCost { .. })),
            }
            if s.action_space().kind == ActionSpaceKind::PayOrNot {
                pay_rest(&mut s);
            }
            let spell = top_spell(&s);
            assert_eq!(
                spell.spent.total(),
                if branch == "mana" { 3 } else { 2 },
                "auto={auto} {branch}"
            );
            assert_eq!(
                spell
                    .spent
                    .contributed(How::Sacrificed, Toward::Forage)
                    .collect::<Vec<_>>(),
                if branch == "food" { vec![food] } else { vec![] }
            );
            assert_eq!(
                spell.spent.contributed(How::Exiled, Toward::Forage).count(),
                if branch == "cards" { 3 } else { 0 }
            );
            assert_eq!(
                s.zone_size(0, ZoneType::Exile),
                if branch == "cards" { 3 } else { 0 }
            );
            assert_eq!(
                s.battlefield_permanents_named(0, "Food").len(),
                usize::from(branch != "food")
            );
            settle(&mut s);
            assert!(s.battlefield_permanents_named(1, "Gray Ogre").is_empty());
        }
    }
}

fn convocation_draw(s: &mut crate::support::game::Scenario) {
    use crate::support::play::activate_named;
    use managym::state::zone::ZoneType;
    let hand = s.zone_size(0, ZoneType::Hand);
    let life = s.life(0);
    activate_named(s, "Lunar Convocation", 0);
    spend_colorless(s);
    s.assert_life(0, life - 2);
    settle(s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
}

fn collector_food(s: &mut crate::support::game::Scenario) {
    use crate::support::play::{activate_first, enter_and_settle};
    enter_and_settle(s, "Cat Collector");
    let food = permanent_named(s, "Food");
    activate_first(s, food);
    spend_colorless(s);
    settle(s);
}

#[test]
fn lunar_convocation_checks_life_history_when_the_end_step_begins() {
    use crate::support::play::{fork_stepping, play_and_settle};
    use managym::{agent::action::ActionSpaceKind, flow::turn::StepKind, search_state::witness};

    let mut s = five_color_setup(&["Lunar Convocation", "Cat Collector"]);
    collector_food(&mut s);
    play_and_settle(&mut s, "Lunar Convocation");
    let root = s.game().clone();
    let root_witness = witness(&root);
    let frame = |s: &crate::support::game::Scenario| {
        serde_json::to_value(s.game().semantic_decision_frame().unwrap()).unwrap()
    };
    let root_frame = frame(&s);
    for early in [true, false] {
        *s.game_mut() = root.clone();
        assert_eq!(frame(&s), root_frame);
        assert_eq!(witness(s.game()), root_witness);
        if early {
            convocation_draw(&mut s);
        }
        s.advance_to_step(StepKind::End);
        if early {
            assert_eq!(s.action_space().kind, ActionSpaceKind::OrderTriggers);
            assert_eq!(s.action_space().actions.len(), 2);
            for order in 0..2 {
                let mut branch = fork_stepping(&s, order);
                settle(&mut branch);
                branch.assert_life(1, 19);
                assert_eq!(branch.battlefield_permanents_named(0, "Bat").len(), 1);
            }
        } else {
            assert_eq!(s.game().state.stack_objects.len(), 1);
            s.set_player_mana_pool(0, "1B");
            convocation_draw(&mut s);
            s.assert_life(1, 19);
            assert!(s.battlefield_permanents_named(0, "Bat").is_empty());
        }
    }
    assert_eq!(witness(&root), root_witness);
}

#[test]
fn lunar_convocation_counts_equal_gain_and_loss_then_resets() {
    use crate::support::play::play_and_settle;
    use managym::flow::turn::StepKind;
    for gained in [false, true] {
        let mut s = five_color_setup(&["Lunar Convocation", "Cat Collector"]);
        if gained {
            collector_food(&mut s);
            collector_food(&mut s);
        }
        play_and_settle(&mut s, "Lunar Convocation");
        for _ in 0..3 {
            convocation_draw(&mut s);
        }
        s.assert_life(0, if gained { 20 } else { 14 });
        s.advance_to_step(StepKind::End);
        settle(&mut s);
        let bats = usize::from(gained);
        let opponent_life = if gained { 19 } else { 20 };
        for active in [1, 0] {
            s.assert_life(1, opponent_life);
            assert_eq!(s.battlefield_permanents_named(0, "Bat").len(), bats);
            s.advance_to_active_step(active, StepKind::End);
            assert!(s.game().state.stack_objects.is_empty());
            assert_eq!(s.game().state.turn.history[0].life_gains, 0);
            assert_eq!(s.game().state.turn.history[0].life_lost, 0);
        }
    }
}
