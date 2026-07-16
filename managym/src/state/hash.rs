//! Versioned deterministic hashing for mutable match state.

use std::fmt;

use rand::RngCore;
use serde::Serialize;

use crate::{
    flow::{combat::CombatState, game::GameState, turn::TurnState},
    state::{
        card::CardDefId,
        game_object::{
            CardVec, Incarnation, ObjectId, ObjectLki, ObjectRef, PermanentId, PermanentVec,
            PlayerId,
        },
    },
};

/// Canonical encoding and identity contract used by [`GameState::deterministic_hash`].
///
/// Increment this before changing field inclusion, field order, ordering rules,
/// serialization, digest algorithm, or the meaning of an encoded identity.
pub const MATCH_STATE_HASH_VERSION: u32 = 1;

/// A versioned BLAKE3 digest of canonical mutable match state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatchStateHash {
    version: u32,
    digest: [u8; 32],
}

impl MatchStateHash {
    pub const fn version(self) -> u32 {
        self.version
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn to_hex(self) -> String {
        blake3::Hash::from(self.digest).to_hex().to_string()
    }
}

impl fmt::Display for MatchStateHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "v{}:{}", self.version, self.to_hex())
    }
}

#[derive(Serialize)]
struct CanonicalCard {
    object_id: ObjectId,
    definition_id: CardDefId,
    owner: PlayerId,
}

#[derive(Serialize)]
struct CanonicalTurn {
    active_player: PlayerId,
    turn_number: u32,
    lands_played: u32,
    cards_drawn_this_turn: [u32; 2],
    ability_resolutions_this_turn: Vec<(usize, usize, u32)>,
    current_phase: usize,
    current_step: usize,
    step_initialized: bool,
    turn_based_actions_complete: bool,
}

impl From<&TurnState> for CanonicalTurn {
    fn from(turn: &TurnState) -> Self {
        Self {
            active_player: turn.active_player,
            turn_number: turn.turn_number,
            lands_played: turn.lands_played,
            cards_drawn_this_turn: turn.cards_drawn_this_turn,
            ability_resolutions_this_turn: turn
                .ability_resolutions_this_turn
                .iter()
                .map(|(&(card, ability), &count)| (card, ability, count))
                .collect(),
            current_phase: turn.current_phase,
            current_step: turn.current_step,
            step_initialized: turn.step_initialized,
            turn_based_actions_complete: turn.turn_based_actions_complete,
        }
    }
}

#[derive(Serialize)]
struct CanonicalCombat {
    attackers: Vec<PermanentId>,
    attacker_to_blockers: Vec<(PermanentId, Vec<PermanentId>)>,
    attackers_to_declare: Vec<PermanentId>,
    blockers_to_declare: Vec<PermanentId>,
}

impl From<&CombatState> for CanonicalCombat {
    fn from(combat: &CombatState) -> Self {
        Self {
            attackers: combat.attackers.clone(),
            attacker_to_blockers: combat
                .attacker_to_blockers
                .iter()
                .map(|(&attacker, blockers)| (attacker, blockers.clone()))
                .collect(),
            attackers_to_declare: combat.attackers_to_declare.clone(),
            blockers_to_declare: combat.blockers_to_declare.clone(),
        }
    }
}

#[derive(Serialize)]
struct CanonicalMatchState<'a> {
    hash_contract_version: u32,
    content_schema_version: u32,
    content_digest: String,
    cards: Vec<CanonicalCard>,
    permanents: &'a PermanentVec<Option<crate::state::permanent::Permanent>>,
    card_to_permanent: &'a CardVec<Option<PermanentId>>,
    object_incarnations: &'a CardVec<Incarnation>,
    object_lki: Vec<(ObjectRef, ObjectLki)>,
    players: &'a [crate::state::player::Player; 2],
    zones: &'a crate::state::zone::ZoneManager,
    turn: CanonicalTurn,
    priority: &'a crate::flow::priority::PriorityState,
    stack_objects: &'a [crate::state::stack_object::StackObject],
    combat: Option<CanonicalCombat>,
    mana_cache: &'a [Option<crate::state::mana::Mana>; 2],
    events: &'a crate::flow::event_log::EventLog,
    pending_events: &'a crate::flow::event_log::EventLog,
    observation_events: &'a crate::flow::event_log::EventLog,
    pending_triggers: &'a [crate::flow::trigger::PendingTrigger],
    pending_trigger_choice: &'a Option<crate::flow::trigger::PendingTrigger>,
    delayed_triggers: &'a [crate::flow::trigger::DelayedTrigger],
    exile_links: &'a [crate::flow::trigger::ExileLink],
    suspended_decision: &'a Option<crate::flow::decision::SuspendedResolution>,
    trigger_enqueue_counter: u64,
    rng_probe: [u64; 8],
    allocation_watermark: u32,
}

impl<'a> From<&'a GameState> for CanonicalMatchState<'a> {
    fn from(state: &'a GameState) -> Self {
        // Keep this exhaustive: adding a mutable GameState field must break the
        // build until its canonical representation is deliberately chosen.
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
        } = state;

        let cards = cards
            .iter()
            .map(|card| CanonicalCard {
                object_id: card.id,
                definition_id: card.definition_id,
                owner: card.owner,
            })
            .collect();
        let object_lki = object_lki
            .iter()
            .map(|(&object_ref, &lki)| (object_ref, lki))
            .collect();
        let mut rng = rng.clone();
        let mut rng_probe = [0_u64; 8];
        for value in &mut rng_probe {
            *value = rng.next_u64();
        }

        Self {
            hash_contract_version: MATCH_STATE_HASH_VERSION,
            content_schema_version: content.schema_version,
            content_digest: content.content_digest(),
            cards,
            permanents,
            card_to_permanent,
            object_incarnations,
            object_lki,
            players,
            zones,
            turn: CanonicalTurn::from(turn),
            priority,
            stack_objects,
            combat: combat.as_ref().map(CanonicalCombat::from),
            mana_cache,
            events,
            pending_events,
            observation_events,
            pending_triggers,
            pending_trigger_choice,
            delayed_triggers,
            exile_links,
            suspended_decision,
            trigger_enqueue_counter: *trigger_enqueue_counter,
            rng_probe,
            allocation_watermark: id_gen.watermark(),
        }
    }
}

impl GameState {
    /// Canonical logical JSON used by higher-level state contracts that add
    /// authority outside `GameState` (for example, the current legal offer).
    pub(crate) fn deterministic_hash_value(&self) -> serde_json::Value {
        serde_json::to_value(CanonicalMatchState::from(self))
            .expect("canonical match state contains only serializable values")
    }

    /// Hash every mutable match fact plus the shared content schema and digest.
    ///
    /// Canonicalization uses fixed struct field order, semantic vector order,
    /// and sorted entries from ordered maps. It never observes allocations,
    /// copied card definitions, compatibility card names, or formatting output.
    pub fn deterministic_hash(&self) -> MatchStateHash {
        let canonical = CanonicalMatchState::from(self);
        let bytes = serde_json::to_vec(&canonical)
            .expect("canonical match state contains only serializable values");
        MatchStateHash {
            version: MATCH_STATE_HASH_VERSION,
            digest: *blake3::hash(&bytes).as_bytes(),
        }
    }
}
