// search_tests.rs
// Tests for determinized-search support: clone independence, determinization
// invariants, and random playout termination.

use std::collections::{BTreeMap, BTreeSet};

use managym::{
    agent::env::Env,
    state::{
        game_object::{CardId, PlayerId},
        player::PlayerConfig,
        zone::ZoneType,
    },
    Game,
};

fn interactive_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Island".to_string(), 12),
        ("Mountain".to_string(), 12),
        ("Gray Ogre".to_string(), 6),
        ("Wind Drake".to_string(), 6),
        ("Man-o'-War".to_string(), 4),
        ("Raging Goblin".to_string(), 4),
        ("Lightning Bolt".to_string(), 6),
        ("Counterspell".to_string(), 4),
        ("Ancestral Recall".to_string(), 3),
        ("Pyroclasm".to_string(), 3),
    ])
}

fn make_game(seed: u64) -> Game {
    let p1 = PlayerConfig::new("hero", interactive_deck());
    let p2 = PlayerConfig::new("villain", interactive_deck());
    Game::new(vec![p1, p2], seed, true)
}

fn make_env(seed: u64) -> Env {
    let mut env = Env::new(seed, true, false, false);
    let p1 = PlayerConfig::new("hero", interactive_deck());
    let p2 = PlayerConfig::new("villain", interactive_deck());
    env.reset(vec![p1, p2]).expect("reset should succeed");
    env
}

/// Advance a game a fixed number of random steps to reach a mid-game state.
fn advance_random(game: &mut Game, steps: usize, seed: u64) {
    game.reseed(seed);
    for _ in 0..steps {
        if game.is_game_over() {
            break;
        }
        let count = game.action_space().map_or(0, |s| s.actions.len());
        if count == 0 {
            break;
        }
        let index = {
            use rand::Rng;
            game.state.rng.gen_range(0..count)
        };
        game.step(index).expect("step should succeed");
    }
}

fn zone_snapshot(game: &Game, zone: ZoneType, player: PlayerId) -> Vec<CardId> {
    game.state.zones.zone_cards(zone, player).clone()
}

fn hidden_pool(game: &Game, player: PlayerId) -> BTreeSet<usize> {
    let mut pool = BTreeSet::new();
    for card in game.state.zones.zone_cards(ZoneType::Hand, player) {
        pool.insert(card.0);
    }
    for card in game.state.zones.zone_cards(ZoneType::Library, player) {
        pool.insert(card.0);
    }
    pool
}

#[test]
fn clone_is_independent_of_original() {
    let mut game = make_game(3);
    advance_random(&mut game, 20, 99);
    assert!(!game.is_game_over());

    let original_life = [game.state.players[0].life, game.state.players[1].life];
    let original_turn = game.state.turn.turn_number;
    let original_hands = [
        zone_snapshot(&game, ZoneType::Hand, PlayerId(0)),
        zone_snapshot(&game, ZoneType::Hand, PlayerId(1)),
    ];
    let original_libraries = [
        zone_snapshot(&game, ZoneType::Library, PlayerId(0)),
        zone_snapshot(&game, ZoneType::Library, PlayerId(1)),
    ];

    // Play the clone to terminal; the original must be untouched.
    let mut clone = game.clone();
    clone.reseed(7);
    let winner = clone
        .random_playout(100_000, None)
        .expect("playout should succeed");
    assert!(clone.is_game_over());
    assert!(winner.is_some() || clone.winner_index().is_none());

    assert!(!game.is_game_over(), "original mutated by clone playout");
    assert_eq!(
        [game.state.players[0].life, game.state.players[1].life],
        original_life
    );
    assert_eq!(game.state.turn.turn_number, original_turn);
    for player in [PlayerId(0), PlayerId(1)] {
        assert_eq!(
            zone_snapshot(&game, ZoneType::Hand, player),
            original_hands[player.0]
        );
        assert_eq!(
            zone_snapshot(&game, ZoneType::Library, player),
            original_libraries[player.0]
        );
    }
}

#[test]
fn determinize_preserves_public_state_and_hand_sizes() {
    let mut game = make_game(11);
    advance_random(&mut game, 40, 5);
    assert!(!game.is_game_over());

    let hero = PlayerId(0);
    let villain = PlayerId(1);

    let hero_hand_before = zone_snapshot(&game, ZoneType::Hand, hero);
    let villain_hand_size = game.state.zones.size(ZoneType::Hand, villain);
    let villain_library_size = game.state.zones.size(ZoneType::Library, villain);
    let hero_library_before: BTreeSet<usize> = zone_snapshot(&game, ZoneType::Library, hero)
        .iter()
        .map(|c| c.0)
        .collect();
    let villain_pool_before = hidden_pool(&game, villain);
    let life_before = [game.state.players[0].life, game.state.players[1].life];
    let turn_before = game.state.turn.turn_number;
    let action_count_before = game.action_space().map(|s| s.actions.len());
    let public_zones = [
        ZoneType::Battlefield,
        ZoneType::Graveyard,
        ZoneType::Exile,
        ZoneType::Stack,
    ];
    let public_before: Vec<Vec<CardId>> = public_zones
        .iter()
        .flat_map(|zone| {
            [
                zone_snapshot(&game, *zone, hero),
                zone_snapshot(&game, *zone, villain),
            ]
        })
        .collect();

    game.determinize(hero, 12345);

    // Sizes preserved.
    assert_eq!(
        game.state.zones.size(ZoneType::Hand, villain),
        villain_hand_size
    );
    assert_eq!(
        game.state.zones.size(ZoneType::Library, villain),
        villain_library_size
    );
    // Hero's hand is known information: identical, including order.
    assert_eq!(zone_snapshot(&game, ZoneType::Hand, hero), hero_hand_before);
    // Hero's library is the same multiset (only order may change).
    let hero_library_after: BTreeSet<usize> = zone_snapshot(&game, ZoneType::Library, hero)
        .iter()
        .map(|c| c.0)
        .collect();
    assert_eq!(hero_library_after, hero_library_before);
    // Villain's hidden pool (hand + library) is conserved.
    assert_eq!(hidden_pool(&game, villain), villain_pool_before);
    // Public state untouched.
    assert_eq!(
        [game.state.players[0].life, game.state.players[1].life],
        life_before
    );
    assert_eq!(game.state.turn.turn_number, turn_before);
    assert_eq!(
        game.action_space().map(|s| s.actions.len()),
        action_count_before
    );
    let public_after: Vec<Vec<CardId>> = public_zones
        .iter()
        .flat_map(|zone| {
            [
                zone_snapshot(&game, *zone, hero),
                zone_snapshot(&game, *zone, villain),
            ]
        })
        .collect();
    assert_eq!(public_after, public_before);

    // Zone bookkeeping stays consistent: every hand/library card's tracked
    // zone matches its containing vector.
    for (zone, player) in [
        (ZoneType::Hand, villain),
        (ZoneType::Library, villain),
        (ZoneType::Hand, hero),
        (ZoneType::Library, hero),
    ] {
        for card in game.state.zones.zone_cards(zone, player) {
            assert_eq!(game.state.zones.zone_of(*card), Some(zone));
        }
    }
}

#[test]
fn determinize_resamples_opponent_hand() {
    let mut game = make_game(17);
    advance_random(&mut game, 40, 23);
    assert!(!game.is_game_over());

    let villain = PlayerId(1);
    let hand_before = zone_snapshot(&game, ZoneType::Hand, villain);
    assert!(!hand_before.is_empty(), "test needs a nonempty villain hand");

    // Across several seeds, at least one determinization must produce a
    // different villain hand (36-card unseen pool: astronomically likely).
    let mut changed = false;
    for seed in 0..8u64 {
        let mut world = game.clone();
        world.determinize(PlayerId(0), seed);
        if zone_snapshot(&world, ZoneType::Hand, villain) != hand_before {
            changed = true;
            break;
        }
    }
    assert!(changed, "determinize never resampled the villain hand");
}

#[test]
fn determinize_is_deterministic_in_seed() {
    let mut game = make_game(29);
    advance_random(&mut game, 30, 31);
    assert!(!game.is_game_over());

    let mut world_a = game.clone();
    let mut world_b = game.clone();
    world_a.determinize(PlayerId(0), 777);
    world_b.determinize(PlayerId(0), 777);
    for zone in [ZoneType::Hand, ZoneType::Library] {
        for player in [PlayerId(0), PlayerId(1)] {
            assert_eq!(
                zone_snapshot(&world_a, zone, player),
                zone_snapshot(&world_b, zone, player)
            );
        }
    }
}

#[test]
fn random_playout_terminates_with_winner() {
    for seed in 0..5u64 {
        let mut game = make_game(seed);
        game.reseed(1000 + seed);
        let winner = game
            .random_playout(100_000, None)
            .expect("playout should succeed");
        assert!(game.is_game_over(), "playout did not reach terminal");
        assert!(winner.is_some(), "playout ended without a winner");
    }
}

#[test]
fn random_playout_respects_step_cap() {
    let mut game = make_game(41);
    let mut hit_cap = false;
    let winner = game
        .random_playout(1, Some(&mut hit_cap))
        .expect("playout should succeed");
    assert!(hit_cap, "1-step playout should hit the cap");
    assert!(winner.is_none());
    assert!(!game.is_game_over());
}

#[test]
fn env_flat_mc_scores_shape_and_bounds() {
    let env = make_env(51);
    let action_count = env.action_count().expect("action count");
    let result = env
        .flat_mc_scores(2, 2, 9, 2000)
        .expect("flat_mc_scores should succeed");
    assert_eq!(result.scores.len(), action_count);
    assert_eq!(result.simulations, (action_count * 4) as u64);
    for score in &result.scores {
        assert!((0.0..=1.0).contains(score), "score {score} out of [0,1]");
    }
    assert!(result.cap_hits <= result.simulations);
}

#[test]
fn env_fork_is_independent() {
    let env = make_env(61);
    let mut fork = env.fork().expect("fork should succeed");
    let winner = fork
        .random_playout(77, 100_000)
        .expect("playout should succeed");
    assert!(winner.is_some());
    assert!(fork.is_game_over());
    assert!(!env.is_game_over(), "original env mutated by fork playout");
    // Original still usable for search after fork finished a game.
    let result = env.flat_mc_scores(1, 1, 3, 2000).expect("search works");
    assert!(!result.scores.is_empty());
}

/// Flat MC must survive Stage-2 decision points: play random games on the
/// Stage-2 decks and, whenever a mid-resolution decision surfaces, run a
/// small flat-MC evaluation from it (rollouts step through the same
/// decision machinery).
#[test]
fn flat_mc_survives_stage2_decision_points() {
    use managym::agent::action::ActionSpaceKind;

    let stage2_deck = BTreeMap::from([
        ("Island".to_string(), 6),
        ("Mountain".to_string(), 6),
        ("Plains".to_string(), 2), // Glider Kids is white (oracle {2}{W})
        ("Glider Kids".to_string(), 4),
        ("Firebending Lesson".to_string(), 4),
        ("Accumulate Wisdom".to_string(), 4),
        ("Pop Quiz".to_string(), 3),
        ("Igneous Inspiration".to_string(), 3),
        ("Crossroads of Destiny".to_string(), 3),
    ]);

    let is_decision_kind = |kind: ActionSpaceKind| {
        matches!(
            kind,
            ActionSpaceKind::Scry
                | ActionSpaceKind::LookAndSelect
                | ActionSpaceKind::PayOrNot
                | ActionSpaceKind::Modal
                | ActionSpaceKind::DiscardThenDraw
                | ActionSpaceKind::Waterbend
        )
    };

    let mut decisions_evaluated = 0_usize;
    for seed in 0..12_u64 {
        let mut env = Env::new(seed, true, false, false);
        let p0 = PlayerConfig::new("hero", stage2_deck.clone());
        let p1 = PlayerConfig::new("villain", stage2_deck.clone());
        env.reset(vec![p0, p1]).expect("reset");

        let mut steps = 0_usize;
        while !env.is_game_over() && steps < 800 {
            if decisions_evaluated < 6
                && env.action_space_kind().is_some_and(is_decision_kind)
            {
                let result = env
                    .flat_mc_scores(1, 1, seed ^ 0xf1a7, 400)
                    .expect("flat MC at a decision point");
                assert!(!result.scores.is_empty());
                decisions_evaluated += 1;
            }
            let action = env.random_action_index().expect("action index");
            env.step(action as i64).expect("step");
            steps += 1;
        }
    }
    assert!(
        decisions_evaluated > 0,
        "expected at least one Stage-2 decision point across seeds"
    );
}

/// Flat MC must survive decision points in the real Milestone-1 matchup:
/// play random games between the actual UR Lessons and GW Allies lists
/// and run a small flat-MC evaluation from surfaced decision points.
#[test]
fn flat_mc_survives_milestone1_matchup() {
    use managym::agent::action::ActionSpaceKind;

    let ur_deck = BTreeMap::from([
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
    ]);
    let gw_deck = BTreeMap::from([
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
    ]);

    let is_decision_kind = |kind: ActionSpaceKind| {
        matches!(
            kind,
            ActionSpaceKind::Scry
                | ActionSpaceKind::LookAndSelect
                | ActionSpaceKind::PayOrNot
                | ActionSpaceKind::Modal
                | ActionSpaceKind::DiscardThenDraw
                | ActionSpaceKind::Waterbend
                | ActionSpaceKind::ChooseTarget
        )
    };

    let mut decisions_evaluated = 0_usize;
    for seed in 0..12_u64 {
        let mut env = Env::new(seed, true, false, false);
        let p0 = PlayerConfig::new("lessons", ur_deck.clone());
        let p1 = PlayerConfig::new("allies", gw_deck.clone());
        env.reset(vec![p0, p1]).expect("reset");

        let mut steps = 0_usize;
        while !env.is_game_over() && steps < 800 {
            if decisions_evaluated < 8
                && env.action_space_kind().is_some_and(is_decision_kind)
            {
                let result = env
                    .flat_mc_scores(1, 1, seed ^ 0x57a3e3, 400)
                    .expect("flat MC at a decision point");
                assert!(!result.scores.is_empty());
                decisions_evaluated += 1;
            }
            let action = env.random_action_index().expect("action index");
            env.step(action as i64).expect("step");
            steps += 1;
        }
    }
    assert!(
        decisions_evaluated > 0,
        "expected at least one decision point across seeds"
    );
}
