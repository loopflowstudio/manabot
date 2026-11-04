// game.rs
// Core game structs: GameState and Game.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

use rand_chacha::ChaCha8Rng;

use crate::{
    agent::{action::ActionSpace, behavior_tracker::BehaviorTracker},
    decision::ObjectCandidateAddress,
    flow::{
        cast_permission::CastPermissions,
        combat::CombatState,
        decision::SuspendedResolution,
        event::GameEvent,
        priority::PriorityState,
        statics::DerivedCache,
        trigger::{DelayedTrigger, ExileLink, PendingTrigger, TriggerPlacement},
        turn::TurnState,
    },
    state::{
        card::ActivatedAbilityDefinition,
        game_object::{
            CardId, CardVec, IdGenerator, Incarnation, ObjectLkiLog, ObjectRef, PermanentId,
            PermanentVec, PlayerId, Target,
        },
        mana::Mana,
        permanent::{CounterKind, Permanent},
        stack_object::StackObject,
        zone::{ZoneManager, ZoneType},
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
    pub object_lki: ObjectLkiLog,
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
    pub effects: crate::state::effect::EffectRecords,
    /// Exact-object casting grants (airbend) and hand-only restrictions.
    pub cast_permissions: CastPermissions,
    /// A resolution paused on a mid-resolution player decision.
    pub suspended_decision: Option<SuspendedResolution>,
    pub trigger_enqueue_counter: u64,
    pub rng: ChaCha8Rng,
    pub id_gen: IdGenerator,
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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
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

/// Counters for the data `Game` derives from state and caches in
/// `DerivedCache`, each advanced by the writes that can change what its
/// derivation reads:
///
/// - `characteristics`: permanents' characteristics and the static-source
///   index, which read permanents, cards, the battlefield and graveyards,
///   combat, continuous-effect records, and the turn's active player and
///   history.
/// - `mana_pools`: both players' pools, which a static condition may read
///   ("as long as you have six or more unspent mana"). The cache compares it
///   only while a static source with such a condition is on the battlefield.
/// - `triggers`: the trigger-source index (`TriggerIndex`): which battlefield
///   permanents have triggered abilities, printed or granted.
///
/// `DerefMut` on `TrackedState` advances every counter. Its typed accessors
/// write fields no derivation reads, or advance only the counters their
/// field feeds. Debug builds recompute every cached value on read and assert
/// it matches, so an accessor that leaves a counter behind fails tests
/// rather than serving stale data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Revisions {
    pub characteristics: u64,
    pub mana_pools: u64,
    pub triggers: u64,
}

/// `GameState` with the `Revisions` its derived caches key on. State can
/// change only through `DerefMut`, which advances every revision (a mutable
/// borrow that writes nothing still advances them), and the accessors below,
/// which advance exactly the revisions their field feeds.
#[derive(Clone)]
pub struct TrackedState {
    state: GameState,
    revisions: Revisions,
}

impl TrackedState {
    pub(crate) fn new(state: GameState) -> Self {
        Self {
            state,
            revisions: Revisions::default(),
        }
    }

    pub(crate) fn revisions(&self) -> Revisions {
        self.revisions
    }

    // Fields no cached derivation reads: writing them keeps every revision.

    pub(crate) fn priority_mut(&mut self) -> &mut PriorityState {
        &mut self.state.priority
    }

    pub(crate) fn rng_mut(&mut self) -> &mut ChaCha8Rng {
        &mut self.state.rng
    }

    /// Ids are read from the objects and records that carry them, never
    /// from the generator.
    pub(crate) fn id_gen_mut(&mut self) -> &mut IdGenerator {
        &mut self.state.id_gen
    }

    pub(crate) fn events_mut(&mut self) -> &mut Vec<GameEvent> {
        &mut self.state.events
    }

    pub(crate) fn pending_events_mut(&mut self) -> &mut Vec<GameEvent> {
        &mut self.state.pending_events
    }

    pub(crate) fn observation_events_mut(&mut self) -> &mut Vec<GameEvent> {
        &mut self.state.observation_events
    }

    /// The producible-mana cache has its own invalidation (`invalidate_mana_cache`).
    pub(crate) fn mana_cache_mut(&mut self) -> &mut [Option<Mana>; 2] {
        &mut self.state.mana_cache
    }

    pub(crate) fn stack_objects_mut(&mut self) -> &mut Vec<StackObject> {
        &mut self.state.stack_objects
    }

    pub(crate) fn pending_triggers_mut(&mut self) -> &mut Vec<PendingTrigger> {
        &mut self.state.pending_triggers
    }

    pub(crate) fn trigger_placement_mut(&mut self) -> &mut Option<TriggerPlacement> {
        &mut self.state.trigger_placement
    }

    pub(crate) fn trigger_enqueue_counter_mut(&mut self) -> &mut u64 {
        &mut self.state.trigger_enqueue_counter
    }

    pub(crate) fn delayed_triggers_mut(&mut self) -> &mut Vec<DelayedTrigger> {
        &mut self.state.delayed_triggers
    }

    pub(crate) fn suspended_decision_mut(&mut self) -> &mut Option<SuspendedResolution> {
        &mut self.state.suspended_decision
    }

    pub(crate) fn object_lki_mut(&mut self) -> &mut ObjectLkiLog {
        &mut self.state.object_lki
    }

    pub(crate) fn life_mut(&mut self, player: PlayerId) -> Option<&mut i32> {
        self.state
            .players
            .get_mut(player.0)
            .map(|player| &mut player.life)
    }

    /// Step progress flags, which only the game loop reads.
    pub(crate) fn set_step_initialized(&mut self, value: bool) {
        self.state.turn.step_initialized = value;
    }

    pub(crate) fn set_turn_based_actions_complete(&mut self, value: bool) {
        self.state.turn.turn_based_actions_complete = value;
    }

    pub(crate) fn set_cleanup_priority(&mut self, value: bool) {
        self.state.turn.cleanup_priority = value;
    }

    /// Advance to the next step; a new turn changes the active player and
    /// resets the turn histories conditions read.
    pub(crate) fn advance_step(&mut self) {
        let turn_number = self.state.turn.turn_number;
        self.state.turn.advance_step();
        if self.state.turn.turn_number != turn_number {
            self.revisions.characteristics += 1;
            self.revisions.triggers += 1;
        }
    }

    // Permanent fields no cached derivation reads: tapped state, damage,
    // summoning sickness and activation limits. Counters, control,
    // attachment, animation and combat status go through `DerefMut`.

    fn permanent_untracked(&mut self, permanent_id: PermanentId) -> Option<&mut Permanent> {
        self.state
            .permanents
            .get_mut(permanent_id.0)
            .and_then(Option::as_mut)
    }

    /// Returns whether the permanent was untapped.
    pub(crate) fn tap(&mut self, permanent_id: PermanentId) -> bool {
        match self.permanent_untracked(permanent_id) {
            Some(permanent) if !permanent.tapped => {
                permanent.tap();
                true
            }
            _ => false,
        }
    }

    /// Untap (CR 122.1d): a removed stun counter is a characteristics change.
    pub(crate) fn untap(&mut self, permanent_id: PermanentId) {
        let Some(permanent) = self.permanent_untracked(permanent_id) else {
            return;
        };
        let removes_stun = permanent.tapped && permanent.counter_count(CounterKind::Stun) > 0;
        permanent.untap();
        if removes_stun {
            self.revisions.characteristics += 1;
        }
    }

    pub(crate) fn mark_damage(&mut self, permanent_id: PermanentId, amount: i32, deathtouch: bool) {
        if let Some(permanent) = self.permanent_untracked(permanent_id) {
            permanent.take_damage(amount);
            if deathtouch {
                permanent.deathtouch_damage = true;
            }
        }
    }

    pub(crate) fn clear_damage(&mut self, permanent_id: PermanentId) {
        if let Some(permanent) = self.permanent_untracked(permanent_id) {
            permanent.clear_damage();
        }
    }

    pub(crate) fn clear_summoning_sickness(&mut self, permanent_id: PermanentId) {
        if let Some(permanent) = self.permanent_untracked(permanent_id) {
            permanent.summoning_sick = false;
        }
    }

    pub(crate) fn clear_permanent_turn_history(&mut self, permanent_id: PermanentId) {
        if let Some(permanent) = self.permanent_untracked(permanent_id) {
            permanent.clear_turn_history();
        }
    }

    pub(crate) fn record_activation(
        &mut self,
        permanent_id: PermanentId,
        ability_index: usize,
        ability: &ActivatedAbilityDefinition,
    ) {
        if let Some(permanent) = self.permanent_untracked(permanent_id) {
            permanent.record_activation(ability_index, ability);
        }
    }

    // Mana pools feed `mana_pools` only.

    pub(crate) fn mana_pool_mut(&mut self, player: PlayerId) -> &mut Mana {
        self.revisions.mana_pools += 1;
        &mut self.state.players[player.0].mana_pool
    }

    /// The ordinary and until-end-of-combat pools together.
    pub(crate) fn mana_pools_mut(&mut self, player: PlayerId) -> (&mut Mana, &mut Mana) {
        self.revisions.mana_pools += 1;
        let player = &mut self.state.players[player.0];
        (&mut player.mana_pool, &mut player.combat_mana_pool)
    }

    pub(crate) fn combat_mana_pool_mut(&mut self, player: PlayerId) -> &mut Mana {
        self.revisions.mana_pools += 1;
        &mut self.state.players[player.0].combat_mana_pool
    }

    // Zone changes feed the derivations that read the zones they touch.

    /// Move `card` to `to_zone` under `holder`. Characteristics read the
    /// battlefield and graveyards; the trigger index reads the battlefield.
    pub(crate) fn move_card(&mut self, card: CardId, holder: PlayerId, to_zone: ZoneType) {
        let from_zone = self.state.zones.zone_of(card);
        let touches = |zone: ZoneType| matches!(zone, ZoneType::Battlefield | ZoneType::Graveyard);
        if touches(to_zone) || from_zone.is_some_and(touches) {
            self.revisions.characteristics += 1;
        }
        if to_zone == ZoneType::Battlefield || from_zone == Some(ZoneType::Battlefield) {
            self.revisions.triggers += 1;
        }
        self.state.zones.move_card(card, holder, to_zone);
    }

    /// Advance the card's CR 400.7 incarnation as it leaves `from_zone`.
    /// Only a battlefield object's exact identity reaches a cached derivation
    /// (through its effect records and static attachments).
    pub(crate) fn advance_incarnation(&mut self, card: CardId, from_zone: ZoneType) {
        if from_zone == ZoneType::Battlefield {
            self.revisions.characteristics += 1;
            self.revisions.triggers += 1;
        }
        let incarnation = self
            .state
            .object_incarnations
            .get_mut(card.0)
            .expect("card entity must have an incarnation");
        *incarnation = incarnation
            .checked_next()
            .expect("rules object incarnation overflow");
    }
}

/// Formats as the state alone: the revisions are bookkeeping, not game state.
impl std::fmt::Debug for TrackedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.state.fmt(f)
    }
}

impl Deref for TrackedState {
    type Target = GameState;

    fn deref(&self) -> &GameState {
        &self.state
    }
}

impl DerefMut for TrackedState {
    fn deref_mut(&mut self) -> &mut GameState {
        self.revisions.characteristics += 1;
        self.revisions.mana_pools += 1;
        self.revisions.triggers += 1;
        &mut self.state
    }
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
    pub state: TrackedState,
    /// Characteristics, static sources and trigger sources derived from
    /// `state` at its current revisions; cloned with it (its `Arc`s are
    /// immutable), never hashed.
    pub(crate) derived: RefCell<DerivedCache>,
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
            derived: RefCell::new(self.derived.borrow().clone()),
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
        std::mem::take(self.state.observation_events_mut())
    }
}
