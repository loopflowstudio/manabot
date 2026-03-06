// setup.rs
// Game construction and initialization.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::behavior_tracker::BehaviorTracker,
    cardsets::alpha::CardRegistry,
    flow::{
        game::{Game, GameState},
        priority::PriorityState,
        turn::TurnState,
    },
    state::{
        game_object::{CardId, CardVec, IdGenerator, PermanentVec, PlayerId},
        player::{Player, PlayerConfig},
        zone::{ZoneManager, ZoneType},
    },
};

impl Game {
    pub fn new(player_configs: Vec<PlayerConfig>, seed: u64, skip_trivial: bool) -> Self {
        assert_eq!(player_configs.len(), 2, "game supports exactly two players");

        let mut id_gen = IdGenerator::default();
        let registry = CardRegistry::default();

        let mut players = [
            Player::new(id_gen.next_id(), 0, player_configs[0].name.clone()),
            Player::new(id_gen.next_id(), 1, player_configs[1].name.clone()),
        ];

        let mut cards = CardVec::default();
        let mut card_to_permanent = CardVec::default();
        let mut zones = ZoneManager::default();

        for (player_index, config) in player_configs.iter().enumerate() {
            for (name, qty) in &config.decklist {
                for _ in 0..*qty {
                    let card = registry
                        .instantiate(name, PlayerId(player_index), id_gen.next_id())
                        .unwrap_or_else(|| panic!("unknown card in decklist: {name}"));
                    let card_id = CardId(cards.len());
                    cards.push(card);
                    card_to_permanent.push(None);
                    players[player_index].deck.push(card_id);
                }
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for player in [PlayerId(0), PlayerId(1)] {
            let deck = players[player.0].deck.clone();
            for card in deck {
                zones.move_card(card, player, ZoneType::Library);
            }
            zones.shuffle(ZoneType::Library, player, &mut rng);
            // CR 103.4, 103.5 — Each player shuffles then draws an opening hand of seven cards.
            for _ in 0..7 {
                if let Some(card) = zones.top(ZoneType::Library, player) {
                    zones.move_card(card, player, ZoneType::Hand);
                }
            }
        }

        let mut game = Self {
            state: GameState {
                cards,
                permanents: PermanentVec::default(),
                card_to_permanent,
                players,
                zones,
                turn: TurnState::new(PlayerId(0)),
                priority: PriorityState::default(),
                combat: None,
                mana_cache: [None, None],
                stack: Vec::new(),
                pending_events: Vec::new(),
                pending_triggers: Vec::new(),
                pending_trigger_choice: None,
                trigger_enqueue_counter: 0,
                rng,
                id_gen,
                card_registry: registry,
            },
            skip_trivial,
            current_action_space: None,
            skip_trivial_count: 0,
            trackers: [BehaviorTracker::new(false), BehaviorTracker::new(false)],
        };

        game.trackers[0].on_game_start();
        game.trackers[1].on_game_start();

        let _ = game.tick();
        game
    }
}
