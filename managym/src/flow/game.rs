// game.rs
// Core game structs: GameState and Game.

use rand_chacha::ChaCha8Rng;

use crate::{
    agent::{action::ActionSpace, behavior_tracker::BehaviorTracker},
    cardsets::alpha::CardRegistry,
    flow::{
        combat::CombatState,
        decision::SuspendedResolution,
        event::GameEvent,
        priority::PriorityState,
        trigger::{DelayedTrigger, ExileLink, PendingTrigger},
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
    /// One-shot delayed triggers (earthbend returns) watching specific
    /// cards' next departure from the battlefield.
    pub delayed_triggers: Vec<DelayedTrigger>,
    /// "Exiled until [source] leaves the battlefield" linkages (Jailer).
    pub exile_links: Vec<ExileLink>,
    /// A resolution paused on a mid-resolution player decision.
    pub suspended_decision: Option<SuspendedResolution>,
    pub trigger_enqueue_counter: u64,
    pub rng: ChaCha8Rng,
    pub id_gen: IdGenerator,
    pub card_registry: CardRegistry,
}

/// Cast-time / activation-time choice pipeline. Casting a spell walks
/// KickerChoice? -> ChooseTargets (per requirement) -> payment; activating a
/// waterbend ability walks Waterbend until the cost is paid.
#[derive(Clone, Debug)]
pub enum PendingChoice {
    /// "You may pay the kicker cost" (CR 601.2b) — asked before targeting.
    KickerChoice { player: PlayerId, card: CardId },
    /// Choosing targets for requirement `requirement_index` of the card's
    /// `target_requirements()` (CR 601.2c).
    ChooseTargets {
        player: PlayerId,
        card: CardId,
        kicked: bool,
        requirement_index: usize,
        chosen: Vec<Target>,
        chosen_req_indices: Vec<usize>,
        /// Legal, not-yet-chosen targets for the current requirement.
        legal_targets: Vec<Target>,
    },
    /// Paying a waterbend activation cost: tap artifacts/creatures for {1}
    /// each, then pay the remainder with mana.
    Waterbend {
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
        /// Generic mana still owed (colored components are paid with mana
        /// at the end).
        remaining_generic: u8,
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
