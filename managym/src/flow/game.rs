// game.rs
// Core game structs: GameState and Game.

use std::{cell::RefCell, collections::BTreeMap, sync::Arc};

use rand_chacha::ChaCha8Rng;

use crate::{
    agent::{action::ActionSpace, behavior_tracker::BehaviorTracker},
    cards::content_pack::ContentPack,
    decision::ObjectCandidateAddress,
    flow::{
        cast_permission::CastPermissions,
        combat::CombatState,
        decision::SuspendedResolution,
        event::GameEvent,
        priority::PriorityState,
        trigger::{DelayedTrigger, ExileLink, PendingTrigger, TriggerPlacement},
        turn::TurnState,
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
    pub events: Vec<GameEvent>,
    pub pending_events: Vec<GameEvent>,
    pub observation_events: Vec<GameEvent>,
    pub pending_triggers: Vec<PendingTrigger>,
    pub trigger_placement: Option<TriggerPlacement>,
    /// Delayed triggered abilities (CR 603.7) waiting for their events, in
    /// creation order; public, and observed as continuation frames.
    pub delayed_triggers: Vec<DelayedTrigger>,
    /// "Exiled until [source] leaves the battlefield" linkages (Jailer).
    pub exile_links: Vec<ExileLink>,
    /// Continuous effects from resolved spells and abilities, in timestamp
    /// order (`state::effect`).
    pub effects: Vec<crate::state::effect::ContinuousEffect>,
    /// Exact-object casting grants (airbend) and hand-only restrictions.
    pub cast_permissions: CastPermissions,
    /// A resolution paused on a mid-resolution player decision.
    pub suspended_decision: Option<SuspendedResolution>,
    pub trigger_enqueue_counter: u64,
    pub rng: ChaCha8Rng,
    pub id_gen: IdGenerator,
    pub content: Arc<ContentPack>,
}

/// The object whose activated ability is being activated: a battlefield
/// permanent, or a card in its owner's hand or graveyard (CR 602.1, 113.6).
/// Card abilities index the card's printed `activated_abilities`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ActivationSource {
    Permanent(PermanentId),
    Card(CardId),
}

/// What a caster announces while casting a spell (CR 601.2b) that changes
/// its total cost or resolution. Serialized flat; `x` and `free` are omitted
/// when unset so ordinary casts keep their shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CastChoices {
    /// The optional additional cost (kicker, or a waterbend additional cost)
    /// is paid (CR 702.33).
    pub kicked: bool,
    /// The announced value of X (CR 107.3, 601.2b).
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub x: u8,
    /// Cast without paying its mana cost (CR 118.9), as an effect directs;
    /// timing and zone permissions don't apply (CR 608.2g).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub free: bool,
    /// Cast, paying its costs, as a resolving effect directs ("you may
    /// cast"): timing and zone permissions don't apply (CR 608.2g).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub resolving: bool,
}

/// A fully announced spell or activation waiting for selected costs.
#[derive(Clone, Debug, serde::Serialize)]
pub enum Payment {
    Spell {
        card: CardId,
        #[serde(flatten)]
        cast: CastChoices,
        mode: Option<usize>,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
    },
    Activation {
        source: ActivationSource,
        ability_index: usize,
        targets: Vec<Target>,
        /// The announced value of X (CR 602.2b, 601.2b).
        #[serde(skip_serializing_if = "crate::state::is_zero")]
        x: u8,
    },
}

/// Cast-time / activation-time choice pipeline. Casting a spell walks
/// AnnounceX? -> KickerChoice? -> SpellMode? -> ChooseTargets (per requirement) -> payment; activating a
/// waterbend ability walks Waterbend until the cost is paid.
#[derive(Clone, Debug, serde::Serialize)]
pub enum PendingChoice {
    ManaPayment {
        player: PlayerId,
        payment: Box<Payment>,
        sacrifice: Option<PermanentId>,
        alternative: Option<crate::state::mana::ManaCost>,
    },
    SacrificePayment {
        player: PlayerId,
        payment: Box<Payment>,
        cost: Box<crate::state::card::SacrificeCost>,
    },
    /// Choosing the activation's targets one at a time (CR 602.2b, 601.2c);
    /// `chosen` holds those already selected for its single clause.
    ActivatedTarget {
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        chosen: Vec<Target>,
    },
    /// Choosing the permanent to tap for a "Tap an untapped [predicate] you
    /// control" activation cost.
    TapPayment {
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        targets: Vec<Target>,
    },
    /// Choosing the card to discard for a "Discard a card" activation cost.
    DiscardPayment {
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        targets: Vec<Target>,
    },
    /// Announcing X (CR 601.2b): the chosen value is offered as a mode index.
    AnnounceX {
        player: PlayerId,
        card: CardId,
        #[serde(flatten)]
        cast: CastChoices,
    },
    /// "You may pay the kicker cost" (CR 601.2b) — asked before targeting.
    KickerChoice {
        player: PlayerId,
        card: CardId,
        #[serde(flatten)]
        cast: CastChoices,
    },
    /// Announcing the value of X for an activated ability (CR 602.2b,
    /// 601.2b) before its waterbend cost is paid.
    ActivationX {
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    },
    SpellMode {
        player: PlayerId,
        card: CardId,
        #[serde(flatten)]
        cast: CastChoices,
    },
    /// Choosing targets for requirement `requirement_index` of the card's
    /// `target_requirements()` (CR 601.2c).
    ChooseTargets {
        player: PlayerId,
        card: CardId,
        #[serde(flatten)]
        cast: CastChoices,
        mode: Option<usize>,
        requirement_index: usize,
        chosen: Vec<Target>,
        chosen_req_indices: Vec<usize>,
        /// Legal, not-yet-chosen targets for the current requirement.
        legal_targets: Vec<Target>,
    },
    /// Paying a crew cost (CR 702.122a): tap other untapped creatures one
    /// at a time until their total power reaches the crew number.
    Crew {
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
        /// Power still owed; the cost is paid once it reaches zero.
        remaining_power: i32,
    },
    /// Paying a waterbend cost (an ability's, or a spell's additional or
    /// alternative cost): tap artifacts/creatures for {1} each, then pay the
    /// remainder with mana.
    Waterbend {
        player: PlayerId,
        payment: Box<Payment>,
        /// Generic mana further taps may still pay.
        remaining_generic: u8,
        /// Generic mana already paid by taps.
        tapped: u8,
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
    /// Evaluation-only facts. Never cloned, hashed, or exposed to policies.
    pub(crate) card_capture: Option<crate::flow::card_capture::CardCapture>,
    pub state: GameState,
    /// Derived public suffix, updated only by emit; excluded from rules-state
    /// hashing like other projections. Ordinary observation reads share it.
    pub(crate) policy_history: crate::agent::policy_history::HistoryWindow,
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
}

impl Clone for Game {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            policy_history: self.policy_history.clone(),
            skip_trivial: self.skip_trivial,
            current_action_space: self.current_action_space.clone(),
            decision_epoch: self.decision_epoch,
            semantic_object_candidates: self.semantic_object_candidates.clone(),
            pending_choice: self.pending_choice.clone(),
            skip_trivial_count: self.skip_trivial_count,
            trackers: self.trackers.clone(),
            card_capture: None,
        }
    }
}

impl Game {
    pub fn take_observation_events(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.state.observation_events)
    }
}
