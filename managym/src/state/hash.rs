//! Versioned deterministic hashing for mutable match state.
//!
//! V16 projects identities that were bare rows to their slot alone, including
//! CardMoved's successor. Its captured departure is also omitted from that old
//! event shape. Two states differing only in those appearances' generations
//! can therefore hash alike, as before the identity split; a future hash
//! version must encode the complete identities rather than these projections.

use std::fmt;

use rand::RngCore;
use serde::Serialize;

use crate::{
    cards::content_pack::{default_content_pack, ContentPack},
    flow::{combat::CombatState, game::GameState, turn::TurnState},
    state::{
        card::CardDefId,
        game_object::{
            ObjectId, ObjectLki, ObjectVec, PermanentId, PermanentVec, PlayerId, PublicId,
        },
    },
};

/// Canonical encoding and identity contract used by [`GameState::deterministic_hash`].
///
/// Increment this before changing field inclusion, field order, ordering rules,
/// serialization, digest algorithm, or the meaning of an encoded identity.
pub const MATCH_STATE_HASH_VERSION: u32 = 32;

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
    /// A token's or spell copy's characteristics, which can differ from its
    /// registered source definition. They carry `is_token: true`, which
    /// encodes the object being no card rather than a row fact.
    copied_characteristics: Option<super::card::ObjectDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    copy: Option<super::effect::CopyEffect>,
    object_id: PublicId,
    definition_id: CardDefId,
    /// The printing behind a face-down, melded or entry-copy form, which the
    /// shown `definition_id` does not name. Omitted on ordinary objects.
    #[serde(skip_serializing_if = "Option::is_none")]
    backing_definition_id: Option<CardDefId>,
    /// A transformed object's front face (CR 712.8). Omitted on its front.
    #[serde(skip_serializing_if = "Option::is_none")]
    front_definition_id: Option<CardDefId>,
    owner: PlayerId,
}

#[derive(Serialize)]
struct CanonicalTurn<'a> {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    player_controls: Vec<crate::flow::turn::PlayerControl>,
    active_player: PlayerId,
    turn_number: u32,
    lands_played: u32,
    spells_cast: [Vec<crate::flow::turn::RecordedCast>; 2],
    #[serde(skip_serializing_if = "crate::flow::turn::DamageHistory::is_empty")]
    damage_pairs: &'a crate::flow::turn::DamageHistory,
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "crate::state::game_object::exact"
    )]
    attacks_this_turn: Vec<(ObjectId, u32)>,

    history: [crate::flow::turn::PlayerTurnHistory; 2],
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    ability_triggers_this_turn: Vec<(ObjectId, (CardDefId, usize), u32)>,
    #[serde(skip_serializing_if = "crate::flow::turn::UsedTriggerGrants::is_empty")]
    used_trigger_grants: &'a crate::flow::turn::UsedTriggerGrants,
    #[serde(serialize_with = "crate::state::game_object::exact")]
    ability_resolutions_this_turn: Vec<(ObjectId, (CardDefId, usize), u32)>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    extra_turns: Vec<PlayerId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resume_with: Option<PlayerId>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    additional_combats: Vec<crate::flow::turn::AdditionalCombat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    combat_attackers: Option<crate::state::predicate::CardPredicate>,
    current_phase: usize,
    current_step: usize,
    step_initialized: bool,
    turn_based_actions_complete: bool,
    cleanup_priority: bool,
}

impl<'a> From<&'a TurnState> for CanonicalTurn<'a> {
    fn from(turn: &'a TurnState) -> Self {
        Self {
            player_controls: turn.player_controls.clone(),
            active_player: turn.active_player,
            turn_number: turn.turn_number,
            lands_played: turn.lands_played,
            spells_cast: turn.spells_cast.clone(),
            damage_pairs: &turn.damage_pairs,
            attacks_this_turn: turn.attacks_this_turn.clone(),

            history: turn.history,
            used_trigger_grants: &turn.used_trigger_grants,
            ability_triggers_this_turn: turn
                .ability_triggers_this_turn
                .iter()
                .map(|(&(source, face, index), &count)| (source, (face, index), count))
                .collect(),
            ability_resolutions_this_turn: turn
                .ability_resolutions_this_turn
                .iter()
                .map(|(&(object, face, ability), &count)| (object, (face, ability), count))
                .collect(),
            extra_turns: turn.extra_turns.clone(),
            resume_with: turn.resume_with,
            additional_combats: turn.additional_combats.clone(),
            combat_attackers: turn.combat_attackers.clone(),
            current_phase: turn.current_phase,
            current_step: turn.current_step,
            step_initialized: turn.step_initialized,
            turn_based_actions_complete: turn.turn_based_actions_complete,
            cleanup_priority: turn.cleanup_priority,
        }
    }
}

#[derive(Serialize)]
struct CanonicalCombat {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    first_strikers: Vec<PermanentId>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    regular_damage_step: bool,
    attackers: Vec<PermanentId>,
    attacker_to_blockers: Vec<(PermanentId, Vec<PermanentId>)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    already_attacking: Option<usize>,
    attackers_to_declare: Vec<PermanentId>,
    blockers_to_declare: Vec<PermanentId>,
}

impl From<&CombatState> for CanonicalCombat {
    fn from(combat: &CombatState) -> Self {
        Self {
            first_strikers: combat.first_strikers.clone(),
            regular_damage_step: combat.regular_damage_step,
            attackers: combat.attackers.clone(),
            attacker_to_blockers: combat
                .attacker_to_blockers
                .iter()
                .map(|(&attacker, blockers)| (attacker, blockers.clone()))
                .collect(),
            already_attacking: (combat.already_attacking != 0).then_some(combat.already_attacking),
            attackers_to_declare: combat.attackers_to_declare.clone(),
            blockers_to_declare: combat.blockers_to_declare.clone(),
        }
    }
}

#[derive(Serialize)]
struct CanonicalMatchState<'a> {
    procedure: &'a crate::flow::game::GameProcedure,
    #[serde(skip_serializing_if = "crate::state::player::DecklistDisclosure::is_open")]
    disclosure: crate::state::player::DecklistDisclosure,
    hash_contract_version: u32,
    content_schema_version: u32,
    content_digest: String,
    #[serde(rename = "cards")]
    objects: Vec<CanonicalCard>,
    permanents: &'a PermanentVec<Option<crate::state::permanent::Permanent>>,
    #[serde(rename = "card_to_permanent")]
    object_to_permanent: &'a ObjectVec<Option<PermanentId>>,
    object_incarnations: Vec<u32>,
    #[serde(serialize_with = "crate::state::game_object::exact")]
    object_lki: Vec<(ObjectId, &'a ObjectLki)>,
    /// Omitted when empty: no spell has left the stack.
    #[serde(
        serialize_with = "crate::state::game_object::exact",
        skip_serializing_if = "Vec::is_empty"
    )]
    spell_lki: Vec<(ObjectId, &'a crate::state::stack_object::SpellLki)>,
    players: &'a [crate::state::player::Player; 2],
    zones: &'a crate::state::zone::ZoneManager,
    turn: CanonicalTurn<'a>,
    priority: &'a crate::flow::priority::PriorityState,
    stack_objects: &'a [crate::state::stack_object::StackObject],
    combat: Option<CanonicalCombat>,
    events: &'a crate::flow::event::EventLedger,
    pending_events: &'a [crate::flow::event::GameEvent],
    observation_events: &'a [crate::flow::event::GameEvent],
    pending_triggers: &'a [crate::flow::trigger::PendingTrigger],
    trigger_placement: Option<&'a crate::flow::trigger::TriggerPlacement>,
    delayed_triggers: &'a [crate::flow::trigger::DelayedTrigger],
    exile_links: &'a [crate::flow::trigger::ExileLink],
    /// Omitted when empty, keeping hashes of states without them unchanged.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    object_counters: Vec<(ObjectId, &'a crate::state::permanent::Counters)>,
    effects: &'a [crate::state::effect::ContinuousEffect],
    /// Omitted when empty, keeping hashes of states without grants unchanged.
    #[serde(skip_serializing_if = "crate::flow::cast_permission::CastPermissions::is_empty")]
    cast_permissions: &'a crate::flow::cast_permission::CastPermissions,
    suspended_decision: &'a Option<crate::flow::decision::SuspendedResolution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cast_caller: Option<&'a crate::flow::decision::SuspendedResolution>,
    trigger_enqueue_counter: u64,
    rng_probe: [u64; 8],
    allocation_watermark: u32,
}

impl<'a> CanonicalMatchState<'a> {
    fn new(state: &'a GameState, content: &ContentPack) -> Self {
        // Keep this exhaustive: adding a mutable GameState field must break the
        // build until its canonical representation is deliberately chosen.
        let GameState {
            procedure,
            disclosure,
            // Immutable roster facts already occur in the v16 object rows.
            cards: _,
            // Sampling history has no effect on ordinary rules execution.
            library_order_pins: _,
            objects,
            permanents,
            object_to_permanent,
            object_lki,
            spell_lki,
            players,
            zones,
            turn,
            priority,
            stack_objects,
            combat,
            events,
            pending_events,
            observation_events,
            pending_triggers,
            trigger_placement,
            delayed_triggers,
            exile_links,
            object_counters,
            effects,
            cast_permissions,
            suspended_decision,
            cast_caller,
            trigger_enqueue_counter,
            rng,
            id_gen,
        } = state;

        let objects = objects
            .iter()
            .map(|object| CanonicalCard {
                copied_characteristics: (!object.is_card()).then(|| content.object_row(object)),
                copy: object.copy().cloned(),
                object_id: object.id,
                definition_id: object.definition_id,
                backing_definition_id: object.backing().map(|(id, _)| *id),
                front_definition_id: object
                    .form_faces()
                    .filter(|faces| faces.back_up)
                    .map(|faces| faces.front.0),
                owner: object.owner,
            })
            .collect();
        let spell_lki = spell_lki
            .iter()
            .map(|(&object, lki)| (object, lki))
            .collect();
        let object_lki = object_lki
            .iter()
            .map(|(&object_ref, lki)| (object_ref, lki))
            .collect();
        let mut rng = rng.clone();
        let mut rng_probe = [0_u64; 8];
        for value in &mut rng_probe {
            *value = rng.next_u64();
        }

        Self {
            procedure,
            disclosure: *disclosure,
            hash_contract_version: MATCH_STATE_HASH_VERSION,
            content_schema_version: content.schema_version,
            content_digest: content.content_digest(),
            objects,
            permanents,
            object_to_permanent,
            object_incarnations: zones.generations().collect(),
            object_lki,
            spell_lki,
            players,
            zones,
            turn: CanonicalTurn::from(turn),
            priority,
            stack_objects,
            combat: combat.as_deref().map(CanonicalCombat::from),
            events,
            pending_events,
            observation_events,
            pending_triggers,
            trigger_placement: trigger_placement.as_deref(),
            delayed_triggers,
            exile_links,
            object_counters: object_counters
                .iter()
                .map(|(&id, counters)| (id, counters))
                .collect(),
            effects: effects.as_slice(),
            cast_permissions,
            suspended_decision,
            cast_caller: cast_caller.as_deref(),
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
        serde_json::to_value(CanonicalMatchState::new(self, default_content_pack()))
            .expect("canonical match state contains only serializable values")
    }

    /// Hash every mutable match fact plus the shared content schema and digest.
    ///
    /// Canonicalization uses fixed struct field order, semantic vector order,
    /// and sorted entries from ordered maps. It never observes allocations,
    /// definition allocation identity or formatting output.
    pub fn deterministic_hash(&self) -> MatchStateHash {
        self.deterministic_hash_over(default_content_pack())
    }

    /// The hash this state would have in a world with `content`: the proof that
    /// the hash binds the catalog's schema and digest, not its allocation.
    pub fn deterministic_hash_over(&self, content: &ContentPack) -> MatchStateHash {
        let canonical = CanonicalMatchState::new(self, content);
        let bytes = serde_json::to_vec(&canonical)
            .expect("canonical match state contains only serializable values");
        MatchStateHash {
            version: MATCH_STATE_HASH_VERSION,
            digest: *blake3::hash(&bytes).as_bytes(),
        }
    }
}

/// Turn-history counts above this compare alike in a loop fingerprint: no
/// card asks for an occurrence or amount this high, and a mandatory loop
/// passes every lower threshold before it is watched.
const LOOP_COUNT_SATURATION: u64 = 16;

impl GameState {
    /// CR 104.4b: the rules-relevant state a mandatory loop returns to,
    /// sampled after stabilization. The canonical match state, without what
    /// every repetition advances on its own: life and poison totals (the loop
    /// watch tracks their progress itself), the event logs, priority
    /// bookkeeping, identity and timestamp allocation, stack objects' ids,
    /// and turn-history counts past [`LOOP_COUNT_SATURATION`]. Anything a
    /// finite chain uses up (other counters, objects, cards, mana) stays in.
    pub(crate) fn loop_fingerprint(&self) -> u64 {
        use serde_json::Value;

        fn saturate(value: &mut Value) {
            match value {
                Value::Number(number) => {
                    if number.as_u64().is_some_and(|n| n > LOOP_COUNT_SATURATION) {
                        *value = Value::from(LOOP_COUNT_SATURATION);
                    }
                }
                Value::Array(items) => items.iter_mut().for_each(saturate),
                Value::Object(fields) => fields.values_mut().for_each(saturate),
                _ => {}
            }
        }

        let mut canonical = self.deterministic_hash_value();
        let state = canonical
            .as_object_mut()
            .expect("canonical match state is an object");
        for key in [
            "events",
            "pending_events",
            "observation_events",
            "priority",
            "trigger_enqueue_counter",
            "allocation_watermark",
        ] {
            state.remove(key);
        }
        if let Some(Value::Array(players)) = state.get_mut("players") {
            for player in players {
                let Some(player) = player.as_object_mut() else {
                    continue;
                };
                player.remove("life");
                // Poison is a total the watch tracks itself, like life; any
                // other player counter is used up and stays.
                if let Some(Value::Object(counters)) = player.get_mut("counters") {
                    counters.remove("poison");
                    if counters.is_empty() {
                        player.remove("counters");
                    }
                }
            }
        }
        // A stack object's id is its allocation order, renumbered each cycle.
        // Each entry is `{variant: body}`; the id lives in the body.
        // Stabilization already emptied `pending_triggers` and `trigger_placement`.
        if let Some(Value::Array(stack)) = state.get_mut("stack_objects") {
            for entry in stack {
                if let Some(variant) = entry.as_object_mut() {
                    for body in variant.values_mut() {
                        if let Some(fields) = body.as_object_mut() {
                            fields.remove("id");
                        }
                    }
                }
            }
        }
        if let Some(Value::Object(turn)) = state.get_mut("turn") {
            if let Some(history) = turn.get_mut("history") {
                saturate(history);
            }
            // Entries are `[object, [definition, ability], count]`: only the
            // count saturates, so distinct sources never compare alike.
            for key in [
                "ability_triggers_this_turn",
                "ability_resolutions_this_turn",
            ] {
                if let Some(Value::Array(entries)) = turn.get_mut(key) {
                    for entry in entries {
                        if let Some(count) = entry.get_mut(2) {
                            saturate(count);
                        }
                    }
                }
            }
        }
        let bytes = serde_json::to_vec(&canonical).expect("canonical match state serializes");
        let digest = blake3::hash(&bytes);
        u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("eight bytes"))
    }
}
