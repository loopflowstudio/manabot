// game.rs
// Core game structs: GameState and Game.

use rand_chacha::ChaCha8Rng;

use crate::{
    agent::{action::ActionSpace, behavior_tracker::BehaviorTracker},
    cardsets::alpha::CardRegistry,
    flow::{
        combat::CombatState, event::GameEvent, priority::PriorityState, trigger::PendingTrigger,
        turn::TurnState,
    },
    state::{
        card::Card,
        game_object::{CardVec, IdGenerator, PermanentId, PermanentVec},
        mana::Mana,
        permanent::Permanent,
        player::Player,
        stack::StackObject,
        zone::ZoneManager,
    },
};

#[derive(Clone, Debug)]
pub struct GameState {
    pub cards: CardVec<Card>,
    pub permanents: PermanentVec<Option<Permanent>>,
    pub card_to_permanent: CardVec<Option<PermanentId>>,
    pub players: [Player; 2],
    pub zones: ZoneManager,
    pub turn: TurnState,
    pub priority: PriorityState,
    pub combat: Option<CombatState>,
    pub mana_cache: [Option<Mana>; 2],
    pub stack: Vec<StackObject>,
    pub pending_events: Vec<GameEvent>,
    pub pending_triggers: Vec<PendingTrigger>,
    pub pending_trigger_choice: Option<PendingTrigger>,
    pub trigger_enqueue_counter: u64,
    pub rng: ChaCha8Rng,
    pub id_gen: IdGenerator,
    pub card_registry: CardRegistry,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub state: GameState,
    pub skip_trivial: bool,
    pub current_action_space: Option<ActionSpace>,
    pub skip_trivial_count: usize,
    pub trackers: [BehaviorTracker; 2],
}
