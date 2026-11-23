use std::collections::BTreeMap;

use managym::{state::player::PlayerConfig, Game};

use crate::support::game::*;

/// CR 103.3, 103.5 — Each player shuffles and draws seven cards for opening hands.
#[test]
fn cr_103_5_opening_hands_have_seven_cards() {
    let s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Forest", 40)]), 7);

    s.assert_zone_size(0, managym::state::zone::ZoneType::Hand, 7);
    s.assert_zone_size(1, managym::state::zone::ZoneType::Hand, 7);
}

/// CR 103.4 — Players start an ordinary two-player game at 20 life.
#[test]
fn cr_103_4_players_start_at_twenty_life() {
    let s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Forest", 40)]), 13);

    s.assert_life(0, 20);
    s.assert_life(1, 20);
}

/// Negative: this engine stage supports exactly two players.
#[test]
#[should_panic(expected = "game supports exactly two players")]
fn cr_103_negative_rejects_non_two_player_game() {
    let p0 = PlayerConfig::new("solo", BTreeMap::from([("Mountain".to_string(), 40)]));
    let _ = Game::new(vec![p0], 1, false);
}

#[test]
fn cr_103_opening_commands_keep_private_bottom_order_and_enter_before_turn_one() {
    use crate::support::inspect::assert_choice_contract;
    use managym::{
        agent::{action::Action, observation::Observation},
        state::{game_object::PlayerId, zone::ZoneType},
    };
    let cards = deck(&[("Leyline Axe", 12), ("Quicksilver, Brash Blur", 12)]);
    let mut s = Scenario::opening(cards.clone(), cards, 103);
    for round in 1..=2 {
        for seat in 0..2 {
            assert_eq!(s.action_space().player, Some(PlayerId(seat)));
            assert_eq!(s.action_space().actions.len(), 2);
            s.step_action(1);
        }
        for seat in 0..2 {
            let player = PlayerId(seat);
            let mut bottom = Vec::new();
            for index in 0..round {
                assert_eq!(s.action_space().player, Some(player));
                assert_choice_contract(&s);
                let other = Observation::for_player(s.game(), PlayerId(1 - seat));
                assert!(other.action_space.actions.is_empty());
                let Action::SelectCard { object, .. } = s.action_space().actions[index] else {
                    panic!("expected a hand card")
                };
                bottom.push(object.slot());
                s.step_action(index);
            }
            let library = s.game().state.zones.zone_objects(ZoneType::Library, player);
            assert_eq!(
                library
                    .iter()
                    .take(round)
                    .map(|object| object.slot())
                    .collect::<Vec<_>>(),
                bottom
            );
            assert_eq!(
                Observation::for_player(s.game(), player)
                    .agent
                    .known_library
                    .len(),
                round
            );
            assert!(Observation::for_player(s.game(), PlayerId(1 - seat))
                .opponent
                .known_library
                .is_empty());
        }
    }
    s.step_action(0);
    s.step_action(0);
    for seat in 0..2 {
        assert_eq!(s.game().state.turn.turn_number, 0);
        assert_eq!(s.action_space().player, Some(PlayerId(seat)));
        assert_eq!(s.action_space().actions.len(), 6);
        assert_choice_contract(&s);
        let name = ["Leyline Axe", "Quicksilver, Brash Blur"][seat];
        let index = s.action_space().actions.iter().position(|a| matches!(a,Action::SelectCard {object,..} if s.game().state[*object].name == name)).unwrap();
        s.step_action(index);
        for viewer in 0..2 {
            let observation = Observation::for_player(s.game(), PlayerId(viewer));
            let permanents = if viewer == seat {
                observation.agent_permanents
            } else {
                observation.opponent_permanents
            };
            assert_eq!(permanents.len(), 1);
            assert!(!permanents[0].entered_this_turn);
        }
        let done = s.action_space().actions.len() - 1;
        s.step_action(done);
    }
    assert_eq!(s.game().state.turn.turn_number, 1);
    assert!(s.game().state.opening.is_none());
    let object = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Battlefield, PlayerId(1))[0];
    let permanent = s.game().state.permanent_of(object).unwrap();
    assert_eq!(
        s.game().characteristics(permanent).activated_abilities[0]
            .mana_cost
            .mana_value,
        5
    );
}

#[test]
fn cr_103_kept_player_waits_while_opponent_mulligans_to_zero() {
    use managym::state::{game_object::PlayerId, zone::ZoneType};
    let mut s = Scenario::opening(deck(&[("Forest", 40)]), deck(&[("Island", 40)]), 104);
    let keep = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Hand, PlayerId(0))
        .to_vec();
    s.step_action(0);
    for count in 1..=7 {
        assert_eq!(s.action_space().player, Some(PlayerId(1)));
        s.step_action(1);
        for _ in 0..count {
            s.step_action(0);
        }
    }
    s.assert_zone_size(1, ZoneType::Hand, 0);
    assert_eq!(
        s.game()
            .state
            .zones
            .zone_objects(ZoneType::Hand, PlayerId(0))
            .to_vec(),
        keep
    );
    assert_eq!(s.game().state.turn.turn_number, 1);
}
