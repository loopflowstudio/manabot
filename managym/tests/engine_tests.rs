use std::collections::BTreeMap;

use managym::{
    agent::{
        action::{ActionSpaceKind, ActionType},
        env::Env,
        observation::Observation,
    },
    flow::turn::{PhaseKind, StepKind},
    state::{player::PlayerConfig, zone::ZoneType},
    Game,
};

fn mixed_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Mountain".to_string(), 12),
        ("Forest".to_string(), 12),
        ("Llanowar Elves".to_string(), 18),
        ("Grey Ogre".to_string(), 18),
    ])
}

fn aggro_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([("Mountain".to_string(), 20), ("Grey Ogre".to_string(), 20)])
}

fn make_game(seed: u64, skip_trivial: bool) -> Game {
    let p1 = PlayerConfig::new("gaea", mixed_deck());
    let p2 = PlayerConfig::new("urza", mixed_deck());
    Game::new(vec![p1, p2], seed, skip_trivial)
}

fn reset_env(seed: u64, skip_trivial: bool) -> (Env, Observation) {
    let mut env = Env::new(seed, skip_trivial, false, false);
    let p1 = PlayerConfig::new("gaea", mixed_deck());
    let p2 = PlayerConfig::new("urza", mixed_deck());
    let (obs, _) = env.reset(vec![p1, p2]).expect("reset should succeed");
    (env, obs)
}

#[test]
fn initial_state_setup() {
    let game = make_game(42, true);
    assert_eq!(game.state.players[0].life, 20);
    assert_eq!(game.state.players[1].life, 20);
    let p0_hand = game
        .state
        .zones
        .size(ZoneType::Hand, managym::state::game_object::PlayerId(0));
    let p1_hand = game
        .state
        .zones
        .size(ZoneType::Hand, managym::state::game_object::PlayerId(1));
    // Player 0 goes first and skips their first draw, so has 7 cards.
    // Player 1 draws on turn 1, so has 7 or 8 depending on game progression.
    assert_eq!(p0_hand + p1_hand, 14);
    assert_eq!(p0_hand, 7);
    assert_eq!(p1_hand, 7);
}

#[test]
fn full_game_loop_completes() {
    let mut game = make_game(7, true);
    let mut steps = 0;
    while !game.is_game_over() && steps < 4000 {
        game.step(0).expect("step should succeed");
        steps += 1;
    }

    assert!(game.is_game_over(), "game did not finish");
    assert!(game.winner_index().is_some(), "winner should exist");
}

#[test]
fn win_by_empty_library() {
    let p1 = PlayerConfig::new("empty", BTreeMap::new());
    let p2 = PlayerConfig::new("loaded", aggro_deck());
    let mut game = Game::new(vec![p1, p2], 11, true);

    let mut steps = 0;
    while !game.is_game_over() && steps < 100 {
        game.step(0).expect("step should succeed");
        steps += 1;
    }

    assert!(game.is_game_over());
    assert_eq!(game.winner_index(), Some(1));
}

#[test]
fn deterministic_with_same_seed_and_actions() {
    let p1 = PlayerConfig::new("gaea", mixed_deck());
    let p2 = PlayerConfig::new("urza", mixed_deck());

    let mut env1 = Env::new(99, true, false, false);
    let mut env2 = Env::new(99, true, false, false);

    let (obs1, _) = env1.reset(vec![p1.clone(), p2.clone()]).unwrap();
    let (obs2, _) = env2.reset(vec![p1, p2]).unwrap();
    assert_eq!(obs1.to_json(), obs2.to_json());

    let mut trace1 = Vec::new();
    let mut trace2 = Vec::new();

    for _ in 0..500 {
        let (obs_a, reward_a, term_a, trunc_a, _) = env1.step(0).unwrap();
        let (obs_b, reward_b, term_b, trunc_b, _) = env2.step(0).unwrap();

        trace1.push((obs_a.to_json(), reward_a, term_a, trunc_a));
        trace2.push((obs_b.to_json(), reward_b, term_b, trunc_b));

        if term_a || trunc_a {
            break;
        }
    }

    assert_eq!(trace1, trace2);
}

#[test]
fn observation_contract_enum_values_stable() {
    assert_eq!(PhaseKind::Beginning as i32, 0);
    assert_eq!(PhaseKind::PrecombatMain as i32, 1);
    assert_eq!(PhaseKind::Combat as i32, 2);
    assert_eq!(PhaseKind::PostcombatMain as i32, 3);
    assert_eq!(PhaseKind::Ending as i32, 4);

    assert_eq!(StepKind::Untap as i32, 0);
    assert_eq!(StepKind::Cleanup as i32, 11);
}

#[test]
fn combat_damage_reduces_life() {
    // Grey Ogre deck: creatures will eventually attack and deal damage.
    // skip_trivial=false so we control every decision.
    let p1 = PlayerConfig::new("attacker", aggro_deck());
    let p2 = PlayerConfig::new("defender", aggro_deck());
    let mut game = Game::new(vec![p1, p2], 42, false);

    let mut damage_dealt = false;

    for _ in 0..2000 {
        if game.is_game_over() {
            break;
        }

        let space = match game.action_space() {
            Some(s) => s.clone(),
            None => break,
        };

        let action_index = match space.kind {
            // During attack declaration, always attack
            ActionSpaceKind::DeclareAttacker => space
                .actions
                .iter()
                .position(|a| a.action_type() == ActionType::DeclareAttacker)
                .unwrap_or(0),
            // During block declaration, don't block (last action)
            ActionSpaceKind::DeclareBlocker => space.actions.len() - 1,
            ActionSpaceKind::ChooseTarget => 0,
            // Priority: play lands and cast spells when possible, else pass
            ActionSpaceKind::Priority => 0,
            ActionSpaceKind::GameOver => break,
            // Mid-resolution / cost decisions: take the first option.
            ActionSpaceKind::Scry
            | ActionSpaceKind::LookAndSelect
            | ActionSpaceKind::PayOrNot
            | ActionSpaceKind::Modal
            | ActionSpaceKind::DiscardThenDraw
            | ActionSpaceKind::Waterbend => 0,
        };

        game.step(action_index).expect("step should succeed");

        // Check if any player took combat damage
        if game.state.players[0].life < 20 || game.state.players[1].life < 20 {
            damage_dealt = true;
            break;
        }
    }

    assert!(
        damage_dealt,
        "combat damage should reduce a player's life below 20"
    );
}

#[test]
fn invalid_game_action_preserves_action_space() {
    let mut game = make_game(42, false);
    let action_count = game
        .action_space()
        .expect("initial action space should exist")
        .actions
        .len();

    let error = game
        .step(action_count)
        .expect_err("out-of-range action should fail");
    assert!(
        error.to_string().contains("Action index"),
        "unexpected error: {error}"
    );

    assert!(
        game.action_space().is_some(),
        "action space should still be available after invalid action"
    );
    game.step(0)
        .expect("valid action should still work after invalid action");
}

#[test]
fn env_reports_negative_action_index() {
    let (mut env, obs) = reset_env(7, true);
    let action_count = obs.action_space.actions.len();

    let error = env.step(-1).expect_err("negative action index should fail");
    assert_eq!(
        error.to_string(),
        format!("Action index -1 out of bounds: {action_count}")
    );

    env.step(0)
        .expect("valid action should still work after negative action");
}

#[test]
fn observation_stays_valid_through_game() {
    let (mut env, mut obs) = reset_env(7, true);
    assert!(obs.validate(), "initial observation should validate");

    for _ in 0..2000 {
        let (next_obs, _reward, done, truncated, _info) = env.step(0).expect("step should succeed");
        assert!(
            next_obs.validate(),
            "observation should validate at every step"
        );
        obs = next_obs;

        if done || truncated {
            assert!(obs.game_over, "terminal observation should mark game_over");
            break;
        }
    }

    assert!(obs.game_over, "game should complete within step limit");
}

#[test]
fn agent_player_index_alternates() {
    let (mut env, mut obs) = reset_env(7, true);

    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..1000 {
        seen.insert(obs.agent.player_index);
        let (next_obs, _reward, done, truncated, _info) = env.step(0).expect("step should succeed");
        obs = next_obs;
        if done || truncated {
            break;
        }
    }

    assert!(
        seen.len() >= 2,
        "expected agent player index to alternate, saw {seen:?}"
    );
}

/// Stage-1 smoke: 200 random-vs-random games on a deck containing every TLA
/// proof card must reach a terminal state with no panics and no stuck games.
#[test]
fn random_smoke_tla_proof_cards() {
    use rand::{Rng, SeedableRng};

    let tla_deck = BTreeMap::from([
        ("Plains".to_string(), 9),
        ("Island".to_string(), 8),
        ("Kyoshi Warriors".to_string(), 3),
        ("Avatar Enthusiasts".to_string(), 4),
        ("Invasion Reinforcements".to_string(), 3),
        ("Jeong Jeong's Deserters".to_string(), 3),
        ("South Pole Voyager".to_string(), 2),
        ("Tiger-Seal".to_string(), 3),
        ("Otter-Penguin".to_string(), 3),
        ("Forecasting Fortune Teller".to_string(), 2),
    ]);

    for seed in 0..200_u64 {
        let p0 = PlayerConfig::new("ally", tla_deck.clone());
        let p1 = PlayerConfig::new("penguin", tla_deck.clone());
        let mut game = Game::new(vec![p0, p1], seed, true);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);

        let mut steps = 0_usize;
        while !game.is_game_over() {
            let action_count = game
                .action_space()
                .map(|space| space.actions.len())
                .unwrap_or(1)
                .max(1);
            let action = rng.gen_range(0..action_count);
            game.step(action)
                .unwrap_or_else(|e| panic!("seed {seed}: step failed: {e:?}"));
            steps += 1;
            assert!(steps < 50_000, "seed {seed}: game appears stuck");
        }
        assert!(game.winner_index().is_some(), "seed {seed}: no winner");
    }
}

/// Stage-2 smoke: 200 random-vs-random games between the UR-decisions deck
/// and the GW-costs deck (every Stage-2 card included) must reach a
/// terminal state with no panics and no stuck games. Random action
/// selection exercises every new ActionSpace kind (scry, look-and-select,
/// pay-or-not, modal, discard-then-draw, waterbend, multi-targeting).
#[test]
fn random_smoke_stage2_cards() {
    use rand::{Rng, SeedableRng};

    let ur_deck = BTreeMap::from([
        ("Island".to_string(), 7),
        ("Mountain".to_string(), 6),
        ("Glider Kids".to_string(), 3),
        ("Firebending Lesson".to_string(), 3),
        ("It'll Quench Ya!".to_string(), 2),
        ("Accumulate Wisdom".to_string(), 3),
        ("Pop Quiz".to_string(), 2),
        ("Igneous Inspiration".to_string(), 2),
        ("Divide by Zero".to_string(), 2),
        ("Crossroads of Destiny".to_string(), 2),
        ("Waterfall Aerialist".to_string(), 3),
        ("Otter-Penguin".to_string(), 2),
    ]);
    let gw_deck = BTreeMap::from([
        ("Plains".to_string(), 7),
        ("Forest".to_string(), 6),
        ("Water Tribe Rallier".to_string(), 4),
        ("Allies at Last".to_string(), 3),
        ("Badgermole Cub".to_string(), 3),
        ("Kyoshi Warriors".to_string(), 4),
        ("Invasion Reinforcements".to_string(), 3),
        ("South Pole Voyager".to_string(), 3),
        ("Forecasting Fortune Teller".to_string(), 4),
    ]);

    for seed in 0..200_u64 {
        let p0 = PlayerConfig::new("decisions", ur_deck.clone());
        let p1 = PlayerConfig::new("costs", gw_deck.clone());
        let mut game = Game::new(vec![p0, p1], seed, true);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x57a6e2);

        let mut steps = 0_usize;
        while !game.is_game_over() {
            let action_count = game
                .action_space()
                .map(|space| space.actions.len())
                .unwrap_or(0);
            assert!(action_count > 0, "seed {seed}: empty action space");
            let action = rng.gen_range(0..action_count);
            game.step(action)
                .unwrap_or_else(|e| panic!("seed {seed}: step failed: {e:?}"));
            steps += 1;
            assert!(steps < 50_000, "seed {seed}: game appears stuck");
        }
        assert!(game.winner_index().is_some(), "seed {seed}: no winner");
    }
}

/// The Milestone-1 UR Lessons decklist (wave/rules/01-two-deck-slice.md).
fn ur_lessons_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Island".to_string(), 9),
        ("Mountain".to_string(), 8),
        ("Tiger-Seal".to_string(), 2),
        ("Otter-Penguin".to_string(), 2),
        ("Fire Nation Cadets".to_string(), 2),
        ("First-Time Flyer".to_string(), 2),
        ("Forecasting Fortune Teller".to_string(), 1),
        ("Dragonfly Swarm".to_string(), 1),
        ("Firebending Lesson".to_string(), 4),
        ("Igneous Inspiration".to_string(), 2),
        ("Pop Quiz".to_string(), 2),
        ("Divide by Zero".to_string(), 2),
        ("It'll Quench Ya!".to_string(), 2),
        ("Accumulate Wisdom".to_string(), 2),
    ])
}

/// The Milestone-1 GW Allies decklist (wave/rules/01-two-deck-slice.md).
fn gw_allies_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Plains".to_string(), 9),
        ("Forest".to_string(), 8),
        ("Water Tribe Rallier".to_string(), 2),
        ("Invasion Reinforcements".to_string(), 2),
        ("Compassionate Healer".to_string(), 2),
        ("Earth Kingdom Jailer".to_string(), 2),
        ("White Lotus Reinforcements".to_string(), 2),
        ("Earth King's Lieutenant".to_string(), 2),
        ("Kyoshi Warriors".to_string(), 2),
        ("Badgermole Cub".to_string(), 2),
        ("Suki, Kyoshi Warrior".to_string(), 1),
        ("South Pole Voyager".to_string(), 1),
        ("Allies at Last".to_string(), 2),
        ("Yip Yip!".to_string(), 1),
        ("Fancy Footwork".to_string(), 2),
    ])
}

/// Stage-3 smoke: 200 random-vs-random games between the ACTUAL Milestone-1
/// decklists (every card of both 40s registered) must reach a terminal
/// state with no panics, no empty action spaces, and no stuck games.
#[test]
fn random_smoke_milestone1_decklists() {
    use rand::{Rng, SeedableRng};

    for seed in 0..200_u64 {
        let p0 = PlayerConfig::new("lessons", ur_lessons_deck());
        let p1 = PlayerConfig::new("allies", gw_allies_deck());
        let mut game = Game::new(vec![p0, p1], seed, true);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x3a6e3);

        let mut steps = 0_usize;
        while !game.is_game_over() {
            let action_count = game
                .action_space()
                .map(|space| space.actions.len())
                .unwrap_or(0);
            assert!(action_count > 0, "seed {seed}: empty action space");
            let action = rng.gen_range(0..action_count);
            game.step(action)
                .unwrap_or_else(|e| panic!("seed {seed}: step failed: {e:?}"));
            steps += 1;
            assert!(steps < 50_000, "seed {seed}: game appears stuck");
        }
        assert!(game.winner_index().is_some(), "seed {seed}: no winner");
    }
}
