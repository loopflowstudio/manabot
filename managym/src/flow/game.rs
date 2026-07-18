// game.rs
// Core game structs: GameState and Game.

use std::{cell::RefCell, collections::BTreeMap, sync::Arc};

use rand_chacha::ChaCha8Rng;

use crate::{
    agent::{action::ActionSpace, behavior_tracker::BehaviorTracker},
    cardsets::alpha::ContentPack,
    decision::ObjectCandidateAddress,
    flow::{
        combat::CombatState,
        decision::SuspendedResolution,
        event::GameEvent,
        event_log::{CowStats, EventLog},
        priority::PriorityState,
        trigger::{DelayedTrigger, ExileLink, PendingTrigger},
        turn::TurnState,
        undo::UndoJournal,
    },
    state::{
        game_object::{
            CardId, CardVec, IdGenerator, Incarnation, ObjectLki, ObjectRef, PermanentId,
            PermanentVec, PlayerId, Target,
        },
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
    /// Current CR 400.7 generation for each stable physical card entity.
    pub object_incarnations: CardVec<Incarnation>,
    /// Last-known battlefield facts keyed by the departed exact object.
    pub object_lki: BTreeMap<ObjectRef, ObjectLki>,
    pub players: [crate::state::player::Player; 2],
    pub zones: ZoneManager,
    pub turn: TurnState,
    pub priority: PriorityState,
    pub stack_objects: Vec<StackObject>,
    pub combat: Option<CombatState>,
    pub mana_cache: [Option<Mana>; 2],
    pub events: EventLog,
    pub pending_events: EventLog,
    pub observation_events: EventLog,
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
    pub content: Arc<ContentPack>,
}

/// Cast-time / activation-time choice pipeline. Casting a spell walks
/// KickerChoice? -> ChooseTargets (per requirement) -> payment; activating a
/// waterbend ability walks Waterbend until the cost is paid.
#[derive(Clone, Debug, serde::Serialize)]
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

#[derive(Debug)]
pub struct Game {
    pub state: GameState,
    pub skip_trivial: bool,
    pub current_action_space: Option<ActionSpace>,
    /// Monotonic identity for the currently published external decision.
    ///
    /// This is deliberately narrower than the Game protocol's match revision
    /// and prompt authority. Clones retain it so a command can be evaluated on
    /// an exact search fork, while publishing any later decision invalidates
    /// commands decoded from an older structured offer set.
    pub(crate) decision_epoch: u64,
    /// Authority-private exact bindings for object candidates already
    /// published through semantic DecisionFrames. This derived index is not
    /// part of the rules-state witness or client wire representation.
    pub(crate) semantic_object_candidates: RefCell<BTreeMap<ObjectCandidateAddress, ObjectRef>>,
    pub pending_choice: Option<PendingChoice>,
    pub skip_trivial_count: usize,
    pub trackers: [BehaviorTracker; 2],
    pub(crate) undo: Option<UndoJournal>,
}

impl Clone for Game {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            skip_trivial: self.skip_trivial,
            current_action_space: self.current_action_space.clone(),
            decision_epoch: self.decision_epoch,
            semantic_object_candidates: self.semantic_object_candidates.clone(),
            pending_choice: self.pending_choice.clone(),
            skip_trivial_count: self.skip_trivial_count,
            trackers: self.trackers.clone(),
            undo: None,
        }
    }
}

impl Game {
    pub fn take_observation_events(&mut self) -> Vec<GameEvent> {
        self.journal_observation_events();
        self.state.observation_events.take_vec()
    }

    pub(crate) fn admit_page_cow_root(&mut self) {
        self.state.events.make_paged_root();
        self.state.pending_events.make_paged_root();
        self.state.observation_events.make_paged_root();
    }

    pub(crate) fn page_cow_fork(&self, stats: Arc<CowStats>) -> Self {
        let GameState {
            cards,
            permanents,
            card_to_permanent,
            object_incarnations,
            object_lki,
            players,
            zones,
            turn,
            priority,
            stack_objects,
            combat,
            mana_cache,
            events,
            pending_events,
            observation_events,
            pending_triggers,
            pending_trigger_choice,
            delayed_triggers,
            exile_links,
            suspended_decision,
            trigger_enqueue_counter,
            rng,
            id_gen,
            content,
        } = &self.state;
        Self {
            state: GameState {
                cards: cards.clone(),
                permanents: permanents.clone(),
                card_to_permanent: card_to_permanent.clone(),
                object_incarnations: object_incarnations.clone(),
                object_lki: object_lki.clone(),
                players: players.clone(),
                zones: zones.clone(),
                turn: turn.clone(),
                priority: priority.clone(),
                stack_objects: stack_objects.clone(),
                combat: combat.clone(),
                mana_cache: mana_cache.clone(),
                events: events.fork_shared(stats.clone()),
                pending_events: pending_events.fork_shared(stats.clone()),
                observation_events: observation_events.fork_shared(stats),
                pending_triggers: pending_triggers.clone(),
                pending_trigger_choice: pending_trigger_choice.clone(),
                delayed_triggers: delayed_triggers.clone(),
                exile_links: exile_links.clone(),
                suspended_decision: suspended_decision.clone(),
                trigger_enqueue_counter: *trigger_enqueue_counter,
                rng: rng.clone(),
                id_gen: id_gen.clone(),
                content: content.clone(),
            },
            skip_trivial: self.skip_trivial,
            current_action_space: self.current_action_space.clone(),
            decision_epoch: self.decision_epoch,
            semantic_object_candidates: self.semantic_object_candidates.clone(),
            pending_choice: self.pending_choice.clone(),
            skip_trivial_count: self.skip_trivial_count,
            trackers: self.trackers.clone(),
            undo: None,
        }
    }
}
