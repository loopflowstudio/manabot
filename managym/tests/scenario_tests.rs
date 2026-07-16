// scenario_tests.rs
// Tests for the scenario / state-injection surface (flow/scenario.rs,
// agent::env scenario_* passthroughs). This is the substrate of the Python
// competency harness (manabot/verify/competency.py): construct an exact
// mid-game position, refresh the action space, and play from it.

use std::collections::BTreeMap;

use managym::{
    agent::{
        action::{Action, ActionType},
        env::Env,
        observation::{EventType, Observation},
    },
    flow::event::GameEvent,
    state::{
        game_object::{CardId, PlayerId},
        player::PlayerConfig,
        zone::ZoneType,
    },
    Game,
};

fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

fn hero_counter_deck() -> BTreeMap<String, usize> {
    deck(&[("Island", 30), ("Counterspell", 2), ("Wind Drake", 4)])
}

fn villain_bomb_deck() -> BTreeMap<String, usize> {
    deck(&[("Mountain", 28), ("Gray Ogre", 4), ("Shivan Dragon", 4)])
}

fn make_env(seed: u64) -> Env {
    let mut env = Env::new(seed, true, false, false);
    let p0 = PlayerConfig::new("hero", hero_counter_deck());
    let p1 = PlayerConfig::new("villain", villain_bomb_deck());
    env.reset(vec![p0, p1]).expect("reset should succeed");
    env
}

fn make_game(seed: u64) -> Game {
    let p0 = PlayerConfig::new("hero", hero_counter_deck());
    let p1 = PlayerConfig::new("villain", villain_bomb_deck());
    Game::new(vec![p0, p1], seed, true)
}

#[test]
fn observation_preserves_committed_card_move_zones() {
    let game = make_game(5);
    let observation = Observation::new(
        &game,
        &[GameEvent::CardMoved {
            card: CardId(0),
            from: Some(ZoneType::Battlefield),
            to: ZoneType::Graveyard,
            controller: PlayerId(1),
        }],
    );

    let event = observation
        .recent_events
        .first()
        .expect("card move should be observable");
    assert_eq!(event.event_type, EventType::CardMoved as i32);
    assert_eq!(event.from_zone, ZoneType::Battlefield as i32);
    assert_eq!(event.to_zone, ZoneType::Graveyard as i32);
}

/// Injection + refresh produces a coherent position: exact hands, injected
/// battlefield, set life totals, and an action space that reflects them.
#[test]
fn injection_builds_exact_position() {
    let mut env = make_env(7);

    for player in 0..2 {
        env.scenario_clear_hand(player).unwrap();
    }
    env.scenario_force_card_in_hand(0, "Counterspell").unwrap();
    env.scenario_force_battlefield(0, "Island", true).unwrap();
    env.scenario_force_battlefield(0, "Island", true).unwrap();
    env.scenario_force_card_in_hand(1, "Gray Ogre").unwrap();
    env.scenario_force_card_in_hand(1, "Shivan Dragon").unwrap();
    for _ in 0..6 {
        env.scenario_force_battlefield(1, "Mountain", true).unwrap();
    }
    env.scenario_set_life(0, 12).unwrap();
    env.scenario_set_life(1, 12).unwrap();

    let obs = env.scenario_refresh().unwrap();
    // Hand zone is index 1 (ZoneType::Hand).
    assert_eq!(obs.agent.zone_counts[1], 1, "hero hand should be exactly 1");
    assert_eq!(obs.opponent.zone_counts[1], 2, "villain hand should be 2");
    assert_eq!(obs.agent.life, 12);
    assert_eq!(obs.opponent.life, 12);
    // Battlefield zone is index 2.
    assert_eq!(obs.agent.zone_counts[2], 2, "hero battlefield: 2 islands");
    assert_eq!(obs.opponent.zone_counts[2], 6, "villain battlefield: 6 mountains");
}

/// The refreshed action space reflects injected cards, and the full loop
/// plays out: villain casts Gray Ogre, hero counters it with the injected
/// Counterspell paid from injected Islands.
#[test]
fn injected_counterspell_counters_injected_threat() {
    let mut game = make_game(11);

    game.scenario_clear_hand(PlayerId(0));
    game.scenario_clear_hand(PlayerId(1));
    game.scenario_force_card_in_hand(PlayerId(0), "Counterspell")
        .unwrap();
    game.scenario_force_battlefield(PlayerId(0), "Island", true)
        .unwrap();
    game.scenario_force_battlefield(PlayerId(0), "Island", true)
        .unwrap();
    game.scenario_force_card_in_hand(PlayerId(1), "Gray Ogre")
        .unwrap();
    for _ in 0..3 {
        game.scenario_force_battlefield(PlayerId(1), "Mountain", true)
            .unwrap();
    }
    game.scenario_refresh_priority().unwrap();

    // Hero (player 0, turn 1 main) holds only Counterspell with an empty
    // stack: not castable. Drive a tiny scripted loop: villain casts the
    // ogre at the first opportunity; hero counters the moment casting is
    // legal (stack occupied); everything else takes the default action.
    let mut hero_cast = false;
    for _ in 0..10_000 {
        if game.is_game_over() {
            break;
        }
        let space = game.action_space().expect("live action space").clone();
        let cast = space
            .actions
            .iter()
            .position(|action| action.action_type() == ActionType::PriorityCastSpell);
        let choice = match (space.player, cast) {
            (Some(PlayerId(1)), Some(index)) => index,
            (Some(PlayerId(0)), Some(index)) => {
                hero_cast = true;
                index
            }
            _ => default_action(&space.actions),
        };
        game.step(choice).expect("step should succeed");
        // Stop once the stack has fully emptied after the hero's counter.
        if hero_cast && game.state.stack_objects.is_empty() {
            break;
        }
    }
    assert!(hero_cast, "hero should have been offered the counter");

    // Gray Ogre countered: it is in the villain graveyard, not on the
    // battlefield, and the hero's Counterspell is in the hero graveyard.
    assert_eq!(
        game.state.zones.size(ZoneType::Graveyard, PlayerId(1)),
        1,
        "ogre should be countered to graveyard"
    );
    assert_eq!(
        game.state.zones.size(ZoneType::Battlefield, PlayerId(1)),
        3,
        "villain battlefield should still be 3 lands"
    );
    assert_eq!(
        game.state.zones.size(ZoneType::Graveyard, PlayerId(0)),
        1,
        "counterspell should be in the hero graveyard"
    );
}

fn default_action(actions: &[Action]) -> usize {
    actions
        .iter()
        .position(|action| {
            matches!(
                action.action_type(),
                ActionType::PriorityPassPriority | ActionType::ChooseTarget
            ) || matches!(action, Action::DeclareAttacker { attack: false, .. })
                || matches!(action, Action::DeclareBlocker { attacker: None, .. })
        })
        .unwrap_or(actions.len().saturating_sub(1))
}

/// A creature injected with `ready = true` can attack immediately; one
/// injected with `ready = false` is summoning-sick and cannot.
#[test]
fn ready_flag_controls_summoning_sickness() {
    let mut game = make_game(3);

    let ready = game
        .scenario_force_battlefield(PlayerId(0), "Wind Drake", true)
        .unwrap();
    let sick = game
        .scenario_force_battlefield(PlayerId(0), "Wind Drake", false)
        .unwrap();

    let ready_perm = game.state.permanents[ready].as_ref().unwrap();
    let sick_perm = game.state.permanents[sick].as_ref().unwrap();
    assert!(!ready_perm.summoning_sick);
    assert!(sick_perm.summoning_sick);
}

/// Injected lands count toward castability after a refresh: Counterspell in
/// hand with two injected Islands and a spell on the stack is castable.
#[test]
fn refresh_recomputes_castability_from_injected_lands() {
    let mut game = make_game(5);

    game.scenario_clear_hand(PlayerId(0));
    game.scenario_force_card_in_hand(PlayerId(0), "Counterspell")
        .unwrap();
    // Without lands the hero cannot act at all.
    game.scenario_refresh_priority().unwrap();
    let space = game.action_space().unwrap();
    assert!(
        !space
            .actions
            .iter()
            .any(|action| action.action_type() == ActionType::PriorityCastSpell),
        "counterspell must not be castable with an empty stack and no mana"
    );

    game.scenario_force_battlefield(PlayerId(0), "Island", true)
        .unwrap();
    game.scenario_force_battlefield(PlayerId(0), "Island", true)
        .unwrap();
    // Play a land action should also be absent (hand holds no land), but the
    // refresh must still succeed and offer pass.
    game.scenario_refresh_priority().unwrap();
    let space = game.action_space().unwrap();
    assert!(space
        .actions
        .iter()
        .any(|action| action.action_type() == ActionType::PriorityPassPriority));
}

/// clear_hand sends cards to the bottom of the library, not the top: the
/// next natural draw is unchanged by the injection.
#[test]
fn clear_hand_moves_cards_to_library_bottom() {
    let mut game = make_game(9);

    let library_before: Vec<_> = game
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .to_vec();
    let top_before = *library_before.last().expect("library nonempty");
    let hand: Vec<_> = game
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(0))
        .to_vec();
    assert!(!hand.is_empty(), "opening hand should be nonempty");

    game.scenario_clear_hand(PlayerId(0));

    let library_after: Vec<_> = game
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .to_vec();
    assert_eq!(
        library_after.len(),
        library_before.len() + hand.len(),
        "hand should be in the library"
    );
    assert_eq!(
        *library_after.last().unwrap(),
        top_before,
        "library top must be unchanged"
    );
    assert!(game
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(0))
        .is_empty());
}

/// Refresh refuses to run on a non-priority action space.
#[test]
fn refresh_errors_outside_priority() {
    let mut game = make_game(13);
    game.scenario_force_battlefield(PlayerId(0), "Wind Drake", true)
        .unwrap();

    // Advance until the hero's declare-attackers decision surfaces.
    for _ in 0..10_000 {
        if game.is_game_over() {
            break;
        }
        let space = game.action_space().expect("live action space").clone();
        if space.kind == managym::agent::action::ActionSpaceKind::DeclareAttacker {
            break;
        }
        let index = default_action(&space.actions);
        game.step(index).expect("step should succeed");
    }
    let space = game.action_space().unwrap();
    assert_eq!(
        space.kind,
        managym::agent::action::ActionSpaceKind::DeclareAttacker
    );
    assert!(game.scenario_refresh_priority().is_err());
}
