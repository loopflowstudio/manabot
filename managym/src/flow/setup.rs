// setup.rs
// Game construction and initialization.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    cards::{content_pack::default_content_pack, deck_admission::admit_submitted},
    flow::{
        card_capture::{snapshots, CaptureMode, CardCapture},
        game::{Game, GameState, TrackedState},
        priority::PriorityState,
        statistics::Statistics,
        turn::TurnState,
    },
    state::{
        game_object::{Card, CardVec, IdGenerator, ObjectVec, PermanentVec, PlayerId},
        player::{GameSetup, Player},
        zone::{ZoneManager, ZoneType},
    },
};

#[cfg(test)]
thread_local! {
    pub(crate) static CONSTRUCTIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl Game {
    /// A match on the catalog. Panics when a deck submitted to the world fails
    /// admission; [`crate::agent::env::Env::reset`] checks that first.
    pub fn new(player_configs: impl Into<GameSetup>, seed: u64, skip_trivial: bool) -> Self {
        Self::construct(&player_configs.into(), seed, skip_trivial, None, false)
    }

    #[cfg(test)]
    pub(crate) fn new_capturing(
        player_configs: impl Into<GameSetup>,
        seed: u64,
        skip_trivial: bool,
    ) -> Self {
        Self::construct(
            &player_configs.into(),
            seed,
            skip_trivial,
            Some(CaptureMode::Cards),
            false,
        )
    }

    pub(crate) fn construct(
        player_configs: &GameSetup,
        seed: u64,
        skip_trivial: bool,
        capture: Option<CaptureMode>,
        statistics: bool,
    ) -> Self {
        #[cfg(test)]
        CONSTRUCTIONS.set(CONSTRUCTIONS.get() + 1);
        let disclosure = player_configs.decklists.resolve(seed);
        admit_submitted(player_configs)
            .unwrap_or_else(|error| panic!("submitted deck is not admitted: {error}"));
        let content = default_content_pack();
        assert_eq!(player_configs.len(), 2, "game supports exactly two players");

        let mut id_gen = IdGenerator::default();

        let mut players = [
            Player::new(id_gen.next_id(), 0, player_configs[0].name.clone()),
            Player::new(id_gen.next_id(), 1, player_configs[1].name.clone()),
        ];

        let mut cards = Vec::new();
        let mut objects = ObjectVec::default();
        let mut object_to_permanent = ObjectVec::default();
        let mut zones = ZoneManager::default();

        for (player_index, config) in player_configs.iter().enumerate() {
            for (name, qty) in &config.deck.main {
                for _ in 0..*qty {
                    let object = content
                        .instantiate(name, PlayerId(player_index), id_gen.next_id())
                        .unwrap_or_else(|| panic!("unknown card in decklist: {name}"));
                    let object_id = zones.allocate();
                    cards.push(Card::of(&object));
                    objects.push(object);
                    let card_id = objects
                        .card_id(object_id)
                        .expect("deck entries must be cards");
                    object_to_permanent.push(None);
                    players[player_index].deck.push(card_id);
                }
            }
        }

        // Allocate outside copies after both main decks so setup preserves deal identity.
        for (player_index, config) in player_configs.iter().enumerate() {
            for (name, qty) in &config.deck.sideboard {
                assert!(*qty > 0, "sideboard count must be positive: {name}");
                let definition_id = content
                    .definition_id(name)
                    .unwrap_or_else(|| panic!("unknown card in sideboard: {name}"));
                assert!(
                    !content
                        .definition(definition_id)
                        .expect("resolved definition")
                        .is_token,
                    "token cannot be a sideboard card: {name}"
                );
                for _ in 0..*qty {
                    let object = content
                        .instantiate(name, PlayerId(player_index), id_gen.next_id())
                        .expect("validated sideboard definition");
                    let object_id = zones.allocate();
                    cards.push(Card::of(&object));
                    objects.push(object);
                    let card_id = objects
                        .card_id(object_id)
                        .expect("deck entries must be cards");
                    object_to_permanent.push(None);
                    players[player_index].sideboard.push(card_id);
                }
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for player in [PlayerId(0), PlayerId(1)] {
            let deck = players[player.0].deck.clone();
            for card in deck {
                zones.place(&objects, zones.object_of(card), player, ZoneType::Library);
            }
            zones.shuffle(ZoneType::Library, player, &mut rng);
            // CR 103.3, 103.5 — Shuffle, then draw the ordinary seven-card hand.
            for _ in 0..7 {
                if let Some(object) = zones.top(ZoneType::Library, player) {
                    zones.place(&objects, object, player, ZoneType::Hand);
                }
            }
        }

        let mut game = Self {
            derived: Default::default(),
            payment_cache: Default::default(),
            payment_sources: Default::default(),
            state: TrackedState::new(GameState {
                procedure: crate::flow::game::GameProcedure::OpeningHands(Default::default()),
                cards: CardVec::new(cards),
                objects,
                permanents: PermanentVec::default(),
                object_to_permanent,
                object_lki: Default::default(),
                spell_lki: Default::default(),
                players,
                disclosure,
                zones,
                library_order_pins: Default::default(),
                turn: TurnState::new(PlayerId(0)),
                priority: PriorityState::default(),
                stack_objects: Vec::new(),
                combat: None,
                events: Default::default(),
                pending_events: Default::default(),
                observation_events: Default::default(),
                pending_triggers: Vec::new(),
                trigger_placement: None,
                delayed_triggers: Vec::new(),
                exile_links: Vec::new(),
                object_counters: Default::default(),
                effects: crate::state::effect::EffectRecords::default(),
                cast_permissions: Default::default(),
                suspended_decision: None,
                cast_caller: None,
                trigger_enqueue_counter: 0,
                rng,
                id_gen,
            }),
            policy_history: Default::default(),
            decision_memory: Default::default(),
            decision_memory_enabled: false,
            decision_memory_include_own: false,
            skip_trivial,
            mana_floats: player_configs.mana_floats,
            mana_payment: player_configs.mana_payment,
            current_action_space: None,
            decision_epoch: 0,
            semantic_object_candidates: Default::default(),
            pending_choice: None,
            skip_trivial_count: 0,
            skipped_priority_passes: 0,
            card_capture: None,
            statistics: None,
            capture_drawing: None,
            capture_originals: Default::default(),
        };

        game.state.turn.turn_number = 0;
        if capture.is_some() || statistics {
            let initial = snapshots(&game);
            if statistics {
                game.statistics = Some(Statistics::new(&initial, true));
            }
            game.card_capture = capture.map(|mode| CardCapture::new(&game, initial.clone(), mode));
            game.capture_originals = initial.into_iter().map(|card| (card.copy, card)).collect();
        }
        let _ = game.tick();
        game
    }
}
