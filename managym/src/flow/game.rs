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
        game_object::{CardId, CardVec, IdGenerator, PermanentId, PermanentVec, PlayerId, Target},
        mana::Mana,
        stack_object::StackObject,
        zone::ZoneManager,
    },
};

#[derive(Clone, Debug)]
pub struct GameState {
    pub cards: CardVec<crate::state::card::Card>,
    pub permanents: PermanentVec<Option<crate::state::permanent::Permanent>>,
    pub card_to_permanent: CardVec<Option<PermanentId>>,
    pub players: [crate::state::player::Player; 2],
    pub zones: ZoneManager,
    pub turn: TurnState,
    pub priority: PriorityState,
    pub stack_objects: Vec<StackObject>,
    pub combat: Option<CombatState>,
    pub mana_cache: [Option<Mana>; 2],
    pub events: Vec<GameEvent>,
    pub pending_events: Vec<GameEvent>,
    pub observation_events: Vec<GameEvent>,
    pub pending_triggers: Vec<PendingTrigger>,
    pub pending_trigger_choice: Option<PendingTrigger>,
    pub trigger_enqueue_counter: u64,
    pub rng: ChaCha8Rng,
    pub id_gen: IdGenerator,
    pub card_registry: CardRegistry,
}

#[derive(Clone, Debug)]
pub enum PendingChoice {
    ChooseTarget {
        player: PlayerId,
        card: CardId,
        legal_targets: Vec<Target>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CombatDamagePass {
    FirstStrike,
    NormalWithFirstStrike,
    Normal,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub state: GameState,
    pub skip_trivial: bool,
    pub current_action_space: Option<ActionSpace>,
    pub pending_choice: Option<PendingChoice>,
    pub skip_trivial_count: usize,
    pub trackers: [BehaviorTracker; 2],
}

impl Game {
    pub fn take_observation_events(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.state.observation_events)
    }
}
