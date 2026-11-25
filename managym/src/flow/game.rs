// game.rs
// Core game structs: GameState and Game.

use crate::state::card::SpellProfile;
use crate::state::player::{ManaFloats, ManaPayment};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
};

use rand_chacha::ChaCha8Rng;

use crate::{
    agent::action::ActionSpace,
    decision::ObjectCandidateAddress,
    flow::{
        cast_permission::CastPermissions,
        combat::CombatState,
        decision::SuspendedResolution,
        event::{EventLedger, GameEvent},
        payment::Payment,
        priority::PriorityState,
        statics::DerivedCache,
        trigger::{DelayedTrigger, ExileLink, PendingTrigger, TriggerPlacement},
        turn::{DamageHistory, TurnState},
    },
    state::{
        card::{ActivatedAbilityDefinition, CardDefId},
        game_object::{
            CardId, CardVec, IdGenerator, ObjectId, ObjectLkiLog, ObjectLookupError, ObjectVec,
            PermanentId, PermanentVec, PlayerId, Target,
        },
        knowledge::LibraryKnowledge,
        mana::ManaPool,
        permanent::{CounterKind, Permanent, SpellEntry},
        stack_object::{SpellLki, StackObject},
        zone::{ZoneManager, ZoneType},
    },
};

#[derive(Clone, Debug, serde::Serialize)]
pub enum GameProcedure {
    OpeningHands(crate::flow::opening::OpeningHands),
    Playing,
    Finished(GameOutcome),
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(untagged)]
pub enum GameOutcome {
    Losses {
        losses: [LossCauses; 2],
    },
    Victory {
        winner: PlayerId,
    },
    /// Neither player won. `mandatory_loop`: the game repeated mandatory
    /// actions with no way to stop (CR 104.4b).
    Draw {
        mandatory_loop: bool,
    },
}

/// Loss conditions committed for one seat.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct LossCauses {
    pub life_total: bool,
    pub deck_empty: bool,
    pub poison: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub instructed: bool,
}

impl LossCauses {
    pub fn any(self) -> bool {
        self.life_total || self.deck_empty || self.poison || self.instructed
    }

    /// Single-reason projection for Env clients, preserving empty-draw precedence.
    pub fn reason(self) -> Option<&'static str> {
        if self.deck_empty {
            Some("deck_empty")
        } else if self.life_total {
            Some("life_total")
        } else if self.poison {
            Some("poison")
        } else if self.instructed {
            Some("instructed_loss")
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub procedure: GameProcedure,
    pub disclosure: crate::state::player::DecklistDisclosure,
    /// Immutable physical cards, in deal order, including both sideboards.
    pub cards: CardVec,
    /// Every object a game has made: its cards, tokens and copies of spells,
    /// told apart by `Object::kind` (CR 109.1).
    pub objects: ObjectVec<crate::state::game_object::Object>,
    pub permanents: PermanentVec<Option<crate::state::permanent::Permanent>>,
    /// Each slot's battlefield permanent; read through [`GameState::permanent_of`].
    pub object_to_permanent: ObjectVec<Option<PermanentId>>,
    /// Last-known battlefield facts keyed by the departed exact object.
    pub object_lki: ObjectLkiLog,
    /// Each spell that has left the stack, as it last existed there.
    pub spell_lki: BTreeMap<ObjectId, SpellLki>,
    pub players: [crate::state::player::Player; 2],
    pub zones: ZoneManager,
    /// Physical appearances whose library rank was observed or chosen since the
    /// last rules shuffle. Private branch sampling reads these, rules play does not.
    pub(crate) library_order_pins: Box<[BTreeSet<ObjectId>; 2]>,
    pub turn: TurnState,
    pub priority: PriorityState,
    pub stack_objects: Vec<StackObject>,
    pub combat: Option<Box<CombatState>>,
    /// Committed rules events; forks share the sealed history.
    pub events: EventLedger,
    pub pending_events: Vec<GameEvent>,
    pub observation_events: Vec<GameEvent>,
    pub pending_triggers: Vec<PendingTrigger>,
    pub trigger_placement: Option<Box<TriggerPlacement>>,
    /// Delayed triggered abilities (CR 603.7) waiting for their events, in
    /// creation order; public, and observed as continuation frames.
    pub delayed_triggers: Vec<DelayedTrigger>,
    /// "Exiled until [source] leaves the battlefield" linkages (Jailer).
    pub exile_links: Vec<ExileLink>,
    /// Counters on objects off the battlefield (CR 122.1: "exile it with a
    /// stash counter on it"), by exact appearance; a permanent's are on its
    /// `Permanent`. A zone change ends the appearance and its counters (CR
    /// 400.7), so only live objects have entries.
    pub object_counters: BTreeMap<ObjectId, crate::state::permanent::Counters>,
    /// Continuous effects from resolved spells and abilities, in timestamp
    /// order (`state::effect`).
    pub effects: crate::state::effect::EffectRecords,
    /// Exact-object casting grants (airbend) and hand-only restrictions.
    pub cast_permissions: Box<CastPermissions>,
    /// A resolution paused on a mid-resolution player decision.
    pub suspended_decision: Option<SuspendedResolution>,
    /// Effect waiting for its announced spell to commit (CR 608.2g).
    pub cast_caller: Option<Box<SuspendedResolution>>,
    pub trigger_enqueue_counter: u64,
    pub rng: ChaCha8Rng,
    pub id_gen: IdGenerator,
}

impl GameState {
    pub fn opening(&self) -> Option<&crate::flow::opening::OpeningHands> {
        match &self.procedure {
            GameProcedure::OpeningHands(opening) => Some(opening),
            _ => None,
        }
    }

    pub fn loss(&self, seat: usize) -> Option<LossCauses> {
        match self.procedure {
            GameProcedure::Finished(GameOutcome::Losses { losses }) if losses[seat].any() => {
                Some(losses[seat])
            }
            _ => None,
        }
    }

    pub(crate) fn is_game_over(&self) -> bool {
        matches!(self.procedure, GameProcedure::Finished(_))
    }

    /// The game ended in a draw by a mandatory loop (CR 104.4b).
    pub fn mandatory_loop_draw(&self) -> bool {
        matches!(
            self.procedure,
            GameProcedure::Finished(GameOutcome::Draw {
                mandatory_loop: true
            })
        )
    }

    /// Put the same appearance in a zone without a zone change. Callers own
    /// rules events; ordinary play uses `Game::move_object`.
    pub fn place_object(&mut self, object: ObjectId, holder: PlayerId, zone: ZoneType) {
        if self.zones.zone_of(object) == Some(ZoneType::Library) {
            self.library_order_pins[self.objects[object.slot()].owner.0].remove(&object);
        }
        self.zones.place(&self.objects, object, holder, zone);
        if zone == ZoneType::Library {
            self.library_order_pins[holder.0].insert(object);
        }
    }

    /// Materialize hidden-zone contents, retaining transient occupants and order.
    pub fn reassign_hidden(
        &mut self,
        player: PlayerId,
        hand: Vec<ObjectId>,
        library: Vec<ObjectId>,
    ) {
        self.zones
            .reassign_hidden(&self.objects, player, hand, library);
    }

    /// Every stored object under the id its slot holds now, in creation
    /// order, including objects in no zone.
    pub fn latest_objects(
        &self,
    ) -> impl Iterator<Item = (ObjectId, &crate::state::game_object::Object)> {
        self.zones.latest().zip(self.objects.iter())
    }

    /// Cards still in the exile appearances produced by this source's moves.
    pub(crate) fn exiled_with(&self, source: ObjectId) -> impl Iterator<Item = ObjectId> + '_ {
        self.exile_links
            .iter()
            .filter(move |link| {
                link.source == source
                    && self.zones.zone_of(link.exiled_object) == Some(ZoneType::Exile)
            })
            .map(|link| link.exiled_object)
    }

    /// `object` while it still names an object in a zone (CR 400.7).
    pub fn live(&self, object: ObjectId) -> Option<ObjectId> {
        self.zones.zone_of(object).map(|_| object)
    }

    /// The current appearance of a physical card. A card outside the game
    /// (a sideboard) is in no zone: this is the id it would enter with.
    pub fn object_of(&self, card: CardId) -> ObjectId {
        self.zones.object_of(card)
    }

    /// The physical card behind an appearance; stale objects, tokens and
    /// spell copies have no card.
    pub fn card_of(&self, object: ObjectId) -> Option<CardId> {
        self.object(object).ok()?;
        self.objects.card_id(object)
    }

    /// Read one appearance, rejecting departed objects even when their
    /// physical card still occupies the same storage slot.
    pub fn object(
        &self,
        object: ObjectId,
    ) -> Result<&crate::state::game_object::Object, ObjectLookupError> {
        let value = self
            .objects
            .get(object.slot())
            .ok_or(ObjectLookupError::Missing)?;
        if self.zones.zone_of(object).is_none() {
            return Err(ObjectLookupError::Stale);
        }
        Ok(value)
    }

    /// Zone-entry authority may inspect an allocated row before it is in a
    /// zone (new tokens and outside-game cards). It still rejects a retired
    /// generation; ordinary rules reads use `object`.
    pub(crate) fn entry_object(&self, object: ObjectId) -> &crate::state::game_object::Object {
        assert!(self.zones.is_latest(object), "retired entry appearance");
        &self.objects[object.slot()]
    }

    pub(crate) fn entry_object_mut(
        &mut self,
        object: ObjectId,
    ) -> &mut crate::state::game_object::Object {
        assert!(self.zones.is_latest(object), "retired entry appearance");
        &mut self.objects[object.slot()]
    }

    /// Definition provenance for a source that may have departed. A live
    /// source supplies its current printing; departure restores its retained
    /// base, independently of any later appearance in the same storage slot.
    pub(crate) fn source_base(
        &self,
        source: ObjectId,
    ) -> (
        crate::state::card::CardDefId,
        &crate::state::card::PrintedCharacteristics,
    ) {
        self.object(source)
            .map(|row| (row.definition_id, &**row))
            .unwrap_or_else(|_| self.objects.retained_base(source).expect("source printing"))
    }

    /// The battlefield permanent this appearance is.
    pub fn permanent_of(&self, object: ObjectId) -> Option<PermanentId> {
        self.live(object)?;
        self.object_to_permanent
            .get(object.slot())
            .copied()
            .flatten()
    }

    /// Casting choices and payment belong to this exact battlefield appearance.
    pub(crate) fn spell_entry(&self, object: ObjectId) -> Option<&SpellEntry> {
        self.permanents[self.permanent_of(object)?]
            .as_ref()?
            .spell_entry
            .as_deref()
    }

    /// The object the card behind `object` is now: a test follows one card
    /// across zones with it. A token or copy stays the object it was.
    #[cfg(test)]
    pub(crate) fn now(&self, object: ObjectId) -> ObjectId {
        self.card_id(object)
            .map_or(object, |card| self.object_of(card))
    }

    /// The card this object is or was, without the staleness check of
    /// [`GameState::card_of`]: for a card outside the game (a sideboard is no
    /// zone), a viewer capture of a departed object, and stale cast grants
    /// whose callers check the zone themselves.
    pub fn card_id(&self, object: ObjectId) -> Option<CardId> {
        self.objects.card_id(object)
    }

    /// Card-only consumers narrow at the zone boundary; the ordered zone itself
    /// also retains transient tokens and copies until state-based actions.
    pub fn zone_cards(
        &self,
        zone: ZoneType,
        player: PlayerId,
    ) -> impl Iterator<Item = CardId> + '_ {
        self.zones.zone_objects(zone, player).cards(&self.objects)
    }
}

impl std::ops::Index<ObjectId> for GameState {
    type Output = crate::state::game_object::Object;
    fn index(&self, object: ObjectId) -> &Self::Output {
        self.object(object).expect("current object appearance")
    }
}
impl std::ops::Index<&ObjectId> for GameState {
    type Output = crate::state::game_object::Object;
    fn index(&self, object: &ObjectId) -> &Self::Output {
        &self[*object]
    }
}
impl std::ops::IndexMut<ObjectId> for GameState {
    fn index_mut(&mut self, object: ObjectId) -> &mut Self::Output {
        self.object(object).expect("current object appearance");
        &mut self.objects.0[object.slot()]
    }
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
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub exile_instead_of_graveyard: bool,
    #[serde(rename = "adventure", skip_serializing_if = "SpellProfile::is_front")]
    pub profile: SpellProfile,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<crate::flow::cast_permission::PlayRoute>,
    /// Casting via flashback replaces any departure from the stack with exile.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub flashback: bool,
    /// Cast from a prepared copy: a prepared spell (Codie). Read from the
    /// preparation grant while the copy is still in exile.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub from_prepared: bool,
    /// The caster chose or needed the printed permission to cast with flash.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub flash_surcharge: bool,
    /// The optional additional cost (kicker, or a waterbend additional cost)
    /// is paid (CR 702.33).
    pub kicked: bool,
    /// The announced value of X (CR 107.3, 601.2b).
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub x: u8,
    /// Cast without paying its mana cost (CR 118.9): a resolving effect's
    /// free offer or a permission such as Omniscience.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub free: bool,
    /// Cast as a resolving effect directs, paid or free: timing and zone
    /// permissions don't apply (CR 608.2g), so no flash surcharge is owed.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub resolving: bool,
}

/// An activation between announcement and payment. Its controller and X
/// stay fixed while either player chooses targets for successive clauses.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ActivatedTargeting {
    pub player: PlayerId,
    pub source: ActivationSource,
    pub ability_index: usize,
    pub chosen: Vec<Target>,
    /// The clause of each chosen target; empty while every target is clause
    /// 0. Clauses are chosen in order, but an optional clause can be left
    /// empty, so a target's position does not name its clause.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub target_req_indices: Vec<usize>,
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub requirement: usize,
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub x: i32,
}

impl ActivatedTargeting {
    /// Add `target` to clause `requirement`, the clause now being chosen.
    pub fn choose(&mut self, target: Target, requirement: usize) {
        self.chosen.push(target);
        if requirement > 0 || !self.target_req_indices.is_empty() {
            // Every earlier target without a recorded clause is clause 0.
            self.target_req_indices.resize(self.chosen.len() - 1, 0);
            self.target_req_indices.push(requirement);
        }
    }

    /// The targets already chosen for clause `requirement`.
    pub fn clause_targets(&self, requirement: usize) -> Vec<Target> {
        self.chosen
            .iter()
            .enumerate()
            .filter(|(slot, _)| {
                self.target_req_indices.get(*slot).copied().unwrap_or(0) == requirement
            })
            .map(|(_, target)| *target)
            .collect()
    }
}

/// Cast-time / activation-time choice pipeline. Casting a spell walks
/// AnnounceX? -> KickerChoice? -> SpellMode? -> ChooseTargets (per requirement) -> Pay; activating an
/// ability walks ActivationX? -> ActivatedTarget? -> Pay.
#[derive(Clone, Debug, serde::Serialize)]
pub enum PendingChoice {
    /// Paying the announced cost: the step is `payment.owed[0]`
    /// (`Game::paying`). Choosing a payment for it shrinks or removes that
    /// component and appends to `payment.receipt`.
    Pay {
        player: PlayerId,
        payment: Box<Payment>,
    },
    ActivatedTarget(ActivatedTargeting),
    /// Announcing X (CR 601.2b): the chosen value is offered as a mode index.
    AnnounceX {
        player: PlayerId,
        #[serde(rename = "card")]
        object: ObjectId,
        #[serde(flatten)]
        cast: CastChoices,
    },
    /// "You may pay the kicker cost" (CR 601.2b) — asked before targeting.
    KickerChoice {
        player: PlayerId,
        #[serde(rename = "card")]
        object: ObjectId,
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
        #[serde(rename = "card")]
        object: ObjectId,
        #[serde(flatten)]
        cast: CastChoices,
    },
    /// Choosing targets for requirement `requirement_index` of the card's
    /// `target_requirements()` (CR 601.2c).
    ChooseTargets {
        player: PlayerId,
        #[serde(rename = "card")]
        object: ObjectId,
        #[serde(flatten)]
        cast: CastChoices,
        mode: Option<usize>,
        requirement_index: usize,
        chosen: Vec<Target>,
        chosen_req_indices: Vec<usize>,
        /// Legal, not-yet-chosen targets for the current requirement.
        legal_targets: Vec<Target>,
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
/// - `tapped`: tap/untap state. Layers follow it only when a scope,
///   condition or value predicate reads it. Dynamic activation finalization
///   refreshes independently; trigger kinds follow the raw board. Removing a
///   stun counter also advances
///   `characteristics`.
/// - `triggers`: the trigger-source index (`TriggerIndex`): which battlefield
///   permanents have triggered abilities, printed or granted.
/// - `life`: both life totals, which a static condition may read ("as long
///   as you have 25 or more life"); compared like `mana_pools`.
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
    pub tapped: u64,
    pub triggers: u64,
    pub life: u64,
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

    // Viewer knowledge changes neither legality nor layered characteristics or
    // mana production. Keep rule caches when only these projections change.
    pub(crate) fn known_hand_mut(&mut self, player: PlayerId) -> &mut BTreeMap<CardDefId, u32> {
        &mut self.state.players[player.0].known_hand
    }

    pub(crate) fn known_library_mut(&mut self, player: PlayerId) -> &mut LibraryKnowledge {
        &mut self.state.players[player.0].known_library
    }

    pub(crate) fn library_order_pins_mut(&mut self, owner: PlayerId) -> &mut BTreeSet<ObjectId> {
        &mut self.state.library_order_pins[owner.0]
    }

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

    pub(crate) fn events_mut(&mut self) -> &mut EventLedger {
        &mut self.state.events
    }

    pub(crate) fn pending_events_mut(&mut self) -> &mut Vec<GameEvent> {
        &mut self.state.pending_events
    }

    pub(crate) fn observation_events_mut(&mut self) -> &mut Vec<GameEvent> {
        &mut self.state.observation_events
    }

    pub(crate) fn stack_objects_mut(&mut self) -> &mut Vec<StackObject> {
        &mut self.state.stack_objects
    }

    pub(crate) fn pending_triggers_mut(&mut self) -> &mut Vec<PendingTrigger> {
        &mut self.state.pending_triggers
    }

    pub(crate) fn trigger_placement_mut(&mut self) -> &mut Option<Box<TriggerPlacement>> {
        &mut self.state.trigger_placement
    }

    pub(crate) fn trigger_enqueue_counter_mut(&mut self) -> &mut u64 {
        &mut self.state.trigger_enqueue_counter
    }

    pub(crate) fn delayed_triggers_mut(&mut self) -> &mut Vec<DelayedTrigger> {
        &mut self.state.delayed_triggers
    }

    pub(crate) fn cast_caller_mut(&mut self) -> &mut Option<Box<SuspendedResolution>> {
        &mut self.state.cast_caller
    }

    pub(crate) fn suspended_decision_mut(&mut self) -> &mut Option<SuspendedResolution> {
        &mut self.state.suspended_decision
    }

    /// At stable priority with an empty stack only delayed instructions can
    /// still refer to a departed form. Pending decisions never reach this point.
    pub(crate) fn release_resolved_forms(&mut self) {
        if self.state.stack_objects.is_empty() {
            let delayed = &self.state.delayed_triggers;
            let still_watched = |object| {
                delayed.iter().any(|trigger| {
                    trigger.source_identity.object() == object
                        || trigger
                            .remembered
                            .iter()
                            .any(|binding| binding.object == object)
                })
            };
            self.state.object_lki.retain_forms(still_watched);
            let zones = &self.state.zones;
            self.state.exile_links.retain(|link| {
                zones.zone_of(link.exiled_object) == Some(ZoneType::Exile)
                    && (zones.zone_of(link.source) == Some(ZoneType::Battlefield)
                        || still_watched(link.source))
            });
        }
    }

    pub(crate) fn object_lki_mut(&mut self) -> &mut ObjectLkiLog {
        &mut self.state.object_lki
    }

    pub(crate) fn spell_lki_mut(&mut self) -> &mut BTreeMap<ObjectId, SpellLki> {
        &mut self.state.spell_lki
    }

    /// Life totals feed `life` only.
    pub(crate) fn life_mut(&mut self, player: PlayerId) -> Option<&mut i32> {
        self.revisions.life += 1;
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

    // Permanent fields no cached derivation reads: damage,
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
        self.revisions.tapped += 1;
        match self.permanent_untracked(permanent_id) {
            Some(permanent) if !permanent.tapped => {
                permanent.tap();
                true
            }
            _ => false,
        }
    }

    /// Untap (CR 122.1d): a removed stun counter is a characteristics change.
    /// Returns whether the permanent went from tapped to untapped.
    pub(crate) fn untap(&mut self, permanent_id: PermanentId) -> bool {
        self.revisions.tapped += 1;
        let Some(permanent) = self.permanent_untracked(permanent_id) else {
            return false;
        };
        let removes_stun = permanent.tapped && permanent.counter_count(CounterKind::Stun) > 0;
        let untapped = permanent.untap();
        if removes_stun {
            self.revisions.characteristics += 1;
        }
        untapped
    }

    /// Trigger matching and resolution read hits directly. Neither the layer
    /// board nor the index of potential trigger sources depends on them.
    pub(crate) fn damage_history_mut(&mut self) -> &mut DamageHistory {
        &mut self.state.turn.damage_pairs
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

    /// Record an activation against the ability's use limits.
    pub(crate) fn record_activation(
        &mut self,
        permanent_id: PermanentId,
        ability_index: usize,
        ability: &ActivatedAbilityDefinition,
    ) {
        if !ability.is_use_limited() {
            return;
        }
        if let Some(permanent) = self.permanent_untracked(permanent_id) {
            permanent.record_activation(ability_index, ability);
        }
    }

    // Mana pools feed `mana_pools` only.

    pub(crate) fn mana_pool_mut(&mut self, player: PlayerId) -> &mut ManaPool {
        self.revisions.mana_pools += 1;
        &mut self.state.players[player.0].mana_pool
    }

    /// The ordinary and until-end-of-combat pools together.
    pub(crate) fn mana_pools_mut(&mut self, player: PlayerId) -> (&mut ManaPool, &mut ManaPool) {
        self.revisions.mana_pools += 1;
        let player = &mut self.state.players[player.0];
        (&mut player.mana_pool, &mut player.combat_mana_pool)
    }

    pub(crate) fn combat_mana_pool_mut(&mut self, player: PlayerId) -> &mut ManaPool {
        self.revisions.mana_pools += 1;
        &mut self.state.players[player.0].combat_mana_pool
    }

    /// Name the stack appearance of the cast `player` recorded last. The
    /// cast was recorded (through `DerefMut`) before its spell moved, so
    /// derivations already saw it; only the entry's generation changes here,
    /// and no cached derivation reads `spells_cast`.
    pub(crate) fn name_last_cast(&mut self, player: PlayerId, spell: ObjectId) {
        let cast = self.state.turn.spells_cast[player.0]
            .last_mut()
            .expect("a recorded cast");
        debug_assert_eq!(cast.object.slot(), spell.slot(), "the cast's own spell");
        cast.object = spell;
    }

    // Zone changes feed the derivations that read the zones they touch.

    /// Mint the appearance `object` becomes in `holder`'s `to_zone` (CR
    /// 400.7) and return it: the slot's next generation, with the row's
    /// base and entry values consumed together. Characteristics read the
    /// battlefield and graveyards; the trigger index reads the battlefield;
    /// a hidden-zone move invalidates nothing. Only a battlefield object's
    /// identity reaches a cached derivation (through its effect records and
    /// static attachments).
    pub(crate) fn mint_appearance(
        &mut self,
        object: ObjectId,
        holder: PlayerId,
        to_zone: ZoneType,
        entry_copy: bool,
    ) -> ObjectId {
        let from_zone = self.state.zones.zone_of(object);
        let touches = |zone: ZoneType| matches!(zone, ZoneType::Battlefield | ZoneType::Graveyard);
        if touches(to_zone) || from_zone.is_some_and(touches) {
            self.revisions.characteristics += 1;
        }
        if to_zone == ZoneType::Battlefield || from_zone == Some(ZoneType::Battlefield) {
            self.revisions.triggers += 1;
        }
        assert!(
            self.state.zones.is_latest(object),
            "retired mint appearance"
        );
        self.state.object_counters.remove(&object);
        self.state.objects[object.slot()].mint_appearance(from_zone, to_zone, entry_copy);
        self.state
            .zones
            .move_object(&self.state.objects, object, holder, to_zone)
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
        self.revisions.tapped += 1;
        self.revisions.triggers += 1;
        &mut self.state
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum CombatDamagePass {
    FirstStrike,
    NormalWithFirstStrike,
    Normal,
}

#[derive(Debug)]
pub struct Game {
    /// Optional private counters and facts. Search clones omit them; rollback
    /// snapshots preserve them. Never hashed or exposed to policies.
    pub(crate) statistics: Option<crate::flow::statistics::Statistics>,
    pub(crate) capture_originals:
        std::collections::BTreeMap<usize, crate::flow::card_capture::CardSnapshot>,
    pub(crate) capture_drawing: Option<ObjectId>,
    pub(crate) card_capture: Option<crate::flow::card_capture::CardCapture>,
    pub state: TrackedState,
    /// Characteristics, static sources and trigger sources derived from
    /// `state` at its current revisions; cloned with it (its `Arc`s are
    /// immutable), never hashed.
    pub(crate) derived: RefCell<DerivedCache>,
    /// Bounded exact payment queries; copied by value on forks, never hashed.
    pub(crate) payment_cache: RefCell<crate::flow::mana::PaymentCache>,
    /// One player's mana sources for the questions of a live `PaymentScan`;
    /// empty outside one and in every clone.
    pub(crate) payment_sources: RefCell<crate::flow::mana::SharedSources>,
    /// Derived public suffix, updated only by emit; excluded from rules-state
    /// hashing like other projections. Ordinary observation reads share it.
    pub(crate) decision_memory_enabled: bool,
    pub(crate) decision_memory_include_own: bool,
    pub(crate) decision_memory: [crate::agent::decision_memory::DecisionLog; 2],
    pub(crate) policy_history: crate::agent::policy_history::HistoryWindow,
    pub skip_trivial: bool,
    pub mana_floats: [ManaFloats; 2],
    pub mana_payment: [ManaPayment; 2],
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
    pub(crate) semantic_object_candidates: RefCell<BTreeMap<ObjectCandidateAddress, ObjectId>>,
    pub pending_choice: Option<PendingChoice>,
    /// Collapsed action spaces. Historical replay digests include this count.
    pub skip_trivial_count: usize,
    /// Forced passes consumed directly in tick_priority, before a space exists.
    pub(crate) skipped_priority_passes: usize,
}

// Inline footprint on 64-bit hosts. Remeasure heap traffic and live banks
// with bench_footprint before raising a ceiling. Object and Permanent repeat
// for each occupied row. TrackedState allows 40 bytes for its five
// revision counters in addition to the GameState ceiling.
#[cfg(target_pointer_width = "64")]
const _: () = {
    use crate::state::game_object::Object;
    use std::mem::size_of;

    assert!(size_of::<Object>() <= 40);
    assert!(size_of::<Permanent>() <= 168);
    assert!(size_of::<GameState>() <= 4096);
    assert!(size_of::<TrackedState>() <= 4136);
    assert!(size_of::<Game>() <= 5248);
    let cache_limit = if cfg!(debug_assertions) { 112 } else { 104 };
    assert!(size_of::<DerivedCache>() <= cache_limit);
};

impl Clone for Game {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            derived: RefCell::new(self.derived.borrow().clone()),
            payment_cache: self.payment_cache.clone(),
            payment_sources: Default::default(),
            policy_history: self.policy_history.clone(),
            decision_memory: self.decision_memory.clone(),
            // Search branches retain past facts but never record hypothetical choices.
            decision_memory_enabled: false,
            decision_memory_include_own: self.decision_memory_include_own,
            skip_trivial: self.skip_trivial,
            mana_floats: self.mana_floats,
            mana_payment: self.mana_payment,
            current_action_space: self.current_action_space.clone(),
            decision_epoch: self.decision_epoch,
            semantic_object_candidates: self.semantic_object_candidates.clone(),
            pending_choice: self.pending_choice.clone(),
            skip_trivial_count: self.skip_trivial_count,
            skipped_priority_passes: self.skipped_priority_passes,
            card_capture: None,
            statistics: None,
            capture_drawing: None,
            capture_originals: Default::default(),
        }
    }
}

impl Game {
    pub fn take_observation_events(&mut self) -> Vec<GameEvent> {
        std::mem::take(self.state.observation_events_mut())
    }
}
