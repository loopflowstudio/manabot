//! Canonical schema-8 viewer JSON: one borrowed writer and the typed decoder.
//! Handwritten maps emit lexical keys directly; derived nested records retain
//! sorted Value boundaries. Exact bytes bind semantic and search-witness hashes.
//! Canonical round-trip validation rejects unknown fields and missing facts.
use super::observation::{CardTypeData, ContinuousEffectData, CopyData, KeywordData};
use crate::{
    agent::{
        action::{ActionSpaceKind, ActionType},
        observation::{
            ActionOption, ActionSpaceData, CardData, CastPermissionData, CharacteristicsData,
            EventData, Observation, PermanentData, SpellData, StackObjectData, StackObjectKindData,
            StackTargetData, StackTargetKindData, TurnData,
        },
    },
    flow::turn::{PhaseKind, StepKind},
    state::{
        mana::{Color, ManaCost},
        zone::ZoneType,
    },
};
use serde::{ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

/// Emit the schema's canonical bytes without constructing its handwritten trees.
/// Every map below writes keys lexically; these bytes bind SHA-256 semantic
/// identities and BLAKE3 viewer witnesses, including omissions and nulls.
pub(super) fn to_json(observation: &Observation) -> String {
    serde_json::to_string(&ObservationJson(observation, false))
        .expect("observation fields are JSON serializable")
}

/// The canonical bytes plus the public policy history, for a standalone
/// observation that must restore its `events` input. Semantic viewer state,
/// decision captures and search witnesses keep the canonical bytes: captures
/// declare their own bounded history, so the viewer state never carries it.
pub(super) fn to_json_with_history(observation: &Observation) -> String {
    serde_json::to_string(&ObservationJson(observation, true))
        .expect("observation fields are JSON serializable")
}

/// Derived records retain their own serde omissions and enum representations.
/// Going through Value sorts all nested keys (including numeric map keys after
/// conversion to strings). Direct derived serialization uses declaration order.
/// Keep this boundary at these leaves, never around the handwritten views.
fn sorted_record(value: &impl Serialize) -> serde_json::Value {
    serde_json::to_value(value).expect("observation records are JSON serializable")
}

/// The observation and whether to write its nonempty public policy history.
struct ObservationJson<'a>(&'a Observation, bool);

impl Serialize for ObservationJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("action_space", &ActionSpaceJson(&value.action_space))?;
        map.serialize_entry("agent", &sorted_record(&value.agent))?;
        map.serialize_entry("agent_cards", &CardsJson(&value.agent_cards))?;
        map.serialize_entry("agent_permanents", &sorted_record(&value.agent_permanents))?;
        map.serialize_entry("agent_sideboard", &sorted_record(&value.agent_sideboard))?;
        if value.decklists_closed {
            map.serialize_entry("decklists_closed", &true)?;
        }
        map.serialize_entry("game_over", &value.game_over)?;
        map.serialize_entry("opponent", &sorted_record(&value.opponent))?;
        map.serialize_entry("opponent_cards", &CardsJson(&value.opponent_cards))?;
        if !value.decision_memory.is_empty() {
            map.serialize_entry("decision_memory", &sorted_record(&value.decision_memory))?;
        }
        map.serialize_entry(
            "opponent_permanents",
            &sorted_record(&value.opponent_permanents),
        )?;
        if self.1 && !value.policy_history.is_empty() {
            map.serialize_entry("policy_history", &sorted_record(&value.policy_history))?;
        }
        map.serialize_entry("recent_events", &EventsJson(&value.recent_events))?;
        if let Some(effects) = &value.shared_effects {
            map.serialize_entry("shared_effects", &sorted_record(effects))?;
        }
        map.serialize_entry("stack_objects", &StackObjectsJson(&value.stack_objects))?;
        map.serialize_entry("turn", &TurnJson(&value.turn))?;
        map.serialize_entry("won", &value.won)?;
        map.end()
    }
}

struct TurnJson<'a>(&'a TurnData);

impl Serialize for TurnJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("active_player_id", &value.active_player_id)?;
        map.serialize_entry("agent_player_id", &value.agent_player_id)?;
        map.serialize_entry("phase", &(value.phase as i32))?;
        map.serialize_entry("step", &(value.step as i32))?;
        map.serialize_entry("turn_number", &value.turn_number)?;
        map.end()
    }
}

struct ActionSpaceJson<'a>(&'a ActionSpaceData);

impl Serialize for ActionSpaceJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("actions", &ActionsJson(&value.actions))?;
        map.serialize_entry("type", &(value.action_space_type as i32))?;
        map.end()
    }
}

struct ActionsJson<'a>(&'a [ActionOption]);

impl Serialize for ActionsJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(ActionJson))
    }
}

struct ActionJson<'a>(&'a ActionOption);

impl Serialize for ActionJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("ability_index", &value.ability_index)?;
        map.serialize_entry("continuation_index", &value.continuation_index)?;
        map.serialize_entry("declared", &value.declared)?;
        if let Some(description) = &value.description {
            map.serialize_entry("description", description)?;
        }
        map.serialize_entry("focus", &value.focus)?;
        map.serialize_entry("mode_index", &value.mode_index)?;
        if let Some(name_definition) = &value.name_definition {
            map.serialize_entry("name_definition", name_definition)?;
        }
        if let Some(name_group) = &value.name_group {
            map.serialize_entry("name_group", name_group)?;
        }
        if let Some(name_index) = &value.name_index {
            map.serialize_entry("name_index", name_index)?;
        }
        if value.pays_directly {
            map.serialize_entry("pays_directly", &true)?;
        }
        map.serialize_entry("type", &(value.action_type as i32))?;
        map.end()
    }
}

struct CardsJson<'a>(&'a [CardData]);

impl Serialize for CardsJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(CardJson))
    }
}

struct CardJson<'a>(&'a CardData);

/// Compatibility card records write the card list's frozen schema-8 bytes.
impl Serialize for CardData {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CardJson(self).serialize(serializer)
    }
}

impl Serialize for CardJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("card_types", &sorted_record(&value.card_types))?;
        if let Some(permission) = &value.cast_permission {
            map.serialize_entry("cast_permission", &CastPermissionJson(permission))?;
        }
        if value.face_down {
            map.serialize_entry("face_down", &true)?;
        }
        if value.from_battlefield_this_turn {
            map.serialize_entry("from_battlefield_this_turn", &true)?;
        }
        map.serialize_entry("id", &value.id)?;
        map.serialize_entry("is_ally", &value.is_ally)?;
        map.serialize_entry("is_legendary", &value.is_legendary)?;
        map.serialize_entry("is_lesson", &value.is_lesson)?;
        map.serialize_entry("is_token", &value.is_token)?;
        map.serialize_entry("keywords", &sorted_record(&value.keywords))?;
        map.serialize_entry("kicker_cost", &value.kicker_cost)?;
        map.serialize_entry("mana_cost", &ManaCostJson(&value.mana_cost))?;
        map.serialize_entry("name", &value.name)?;
        map.serialize_entry("owner_id", &value.owner_id)?;
        map.serialize_entry("power", &value.power)?;
        map.serialize_entry("registry_key", &value.registry_key)?;
        map.serialize_entry("text_box", &value.text_box)?;
        map.serialize_entry("toughness", &value.toughness)?;
        map.serialize_entry("ward_cost", &value.ward_cost)?;
        if value.ward_life > 0 {
            map.serialize_entry("ward_life", &value.ward_life)?;
            map.serialize_entry("ward_program", &value.ward_program)?;
        }
        if let Some(x) = &value.x {
            map.serialize_entry("x", x)?;
        }
        map.serialize_entry("zone", &(value.zone as i32))?;
        map.end()
    }
}

struct ManaCostJson<'a>(&'a ManaCost);

impl Serialize for ManaCostJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        let cost: [i32; 6] = std::array::from_fn(|index| i32::from(value.cost[index]));
        map.serialize_entry("cost", &cost)?;
        if !value.hybrid.is_empty() {
            map.serialize_entry("hybrid", &HybridJson(&value.hybrid))?;
        }
        map.serialize_entry("mana_value", &i32::from(value.mana_value))?;
        map.end()
    }
}

struct HybridJson<'a>(&'a [[Color; 2]]);

impl Serialize for HybridJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(|pip| [pip[0] as u8, pip[1] as u8]))
    }
}

struct CastPermissionJson<'a>(&'a CastPermissionData);

impl Serialize for CastPermissionJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        if let Some(after_turn) = &value.after_turn {
            map.serialize_entry("after_turn", after_turn)?;
        }
        if value.any_mana {
            map.serialize_entry("any_mana", &true)?;
        }
        if let Some(cost) = &value.cost {
            map.serialize_entry("cost", &ManaCostJson(cost))?;
        }
        if value.flash {
            map.serialize_entry("flash", &true)?;
        }
        if value.foretold {
            map.serialize_entry("foretold", &true)?;
        }
        if value.own_turn_only {
            map.serialize_entry("own_turn_only", &true)?;
        }
        map.serialize_entry("player_id", &value.player_id)?;
        if let Some(source_permanent_id) = &value.source_permanent_id {
            map.serialize_entry("source_permanent_id", source_permanent_id)?;
        }
        if value.this_turn {
            map.serialize_entry("this_turn", &true)?;
        }
        if value.through_next_turn {
            map.serialize_entry("through_next_turn", &true)?;
        }
        if value.waterbend {
            map.serialize_entry("waterbend", &true)?;
        }
        map.end()
    }
}

struct StackObjectsJson<'a>(&'a [StackObjectData]);

impl Serialize for StackObjectsJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(StackObjectJson))
    }
}

struct StackObjectJson<'a>(&'a StackObjectData);

impl Serialize for StackObjectJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("ability_index", &value.ability_index)?;
        if let Some(spell) = &value.spell {
            let card = spell.legacy_card(value.x);
            map.serialize_entry("card", &CardJson(&card))?;
        }
        map.serialize_entry("chosen_mode", &value.chosen_mode)?;
        map.serialize_entry("controller_id", &value.controller_id)?;
        map.serialize_entry("description", &value.description)?;
        if let Some(keywords) = &value.granted_keywords {
            map.serialize_entry("granted_keywords", &sorted_record(keywords))?;
        }
        if value.kicked {
            map.serialize_entry("kicked", &true)?;
        }
        map.serialize_entry("kind", &(value.kind as i32))?;
        map.serialize_entry("source_card_registry_key", &value.source_card_registry_key)?;
        map.serialize_entry("source_permanent_id", &value.source_permanent_id)?;
        map.serialize_entry("stack_object_id", &value.stack_object_id)?;
        map.serialize_entry("targets", &StackTargetsJson(&value.targets))?;
        if value.x != 0 {
            map.serialize_entry("x", &value.x)?;
        }
        map.end()
    }
}

struct StackTargetsJson<'a>(&'a [StackTargetData]);

impl Serialize for StackTargetsJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(StackTargetJson))
    }
}

struct StackTargetJson<'a>(&'a StackTargetData);

impl Serialize for StackTargetJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("card_id", &value.card_id)?;
        map.serialize_entry("kind", &(value.kind as i32))?;
        map.serialize_entry("permanent_id", &value.permanent_id)?;
        map.serialize_entry("player_id", &value.player_id)?;
        map.serialize_entry("stack_object_id", &value.stack_object_id)?;
        map.end()
    }
}

struct EventsJson<'a>(&'a [EventData]);

impl Serialize for EventsJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(EventJson))
    }
}

struct EventJson<'a>(&'a EventData);

impl Serialize for EventJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("amount", &value.amount)?;
        map.serialize_entry("controller_id", &value.controller_id)?;
        map.serialize_entry("event_type", &value.event_type)?;
        map.serialize_entry("from_zone", &value.from_zone)?;
        map.serialize_entry("source_id", &value.source_id)?;
        map.serialize_entry("source_incarnation", &value.source_incarnation)?;
        map.serialize_entry("source_kind", &value.source_kind)?;
        map.serialize_entry("target_id", &value.target_id)?;
        map.serialize_entry("target_incarnation", &value.target_incarnation)?;
        map.serialize_entry("target_kind", &value.target_kind)?;
        map.serialize_entry("to_zone", &value.to_zone)?;
        map.end()
    }
}

// Typed decoding and canonical round-trip validation.

macro_rules! integer_enum {
    ($name:ident, $ty:ty, {$($value:literal => $variant:path),+ $(,)?}) => {
        pub fn $name<'de, D: Deserializer<'de>>(deserializer: D) -> Result<$ty, D::Error> {
            match i32::deserialize(deserializer)? {
                $($value => Ok($variant),)+
                value => Err(serde::de::Error::custom(format!("unsupported enum value {value}"))),
            }
        }
    };
}
integer_enum!(phase, PhaseKind, {0 => PhaseKind::Beginning, 1 => PhaseKind::PrecombatMain, 2 => PhaseKind::Combat, 3 => PhaseKind::PostcombatMain, 4 => PhaseKind::Ending});
integer_enum!(step, StepKind, {0 => StepKind::Untap, 1 => StepKind::Upkeep, 2 => StepKind::Draw, 3 => StepKind::Main, 4 => StepKind::BeginningOfCombat, 5 => StepKind::DeclareAttackers, 6 => StepKind::DeclareBlockers, 7 => StepKind::CombatDamage, 8 => StepKind::EndOfCombat, 9 => StepKind::PostcombatMain, 10 => StepKind::End, 11 => StepKind::Cleanup});
integer_enum!(zone, ZoneType, {0 => ZoneType::Library, 1 => ZoneType::Hand, 2 => ZoneType::Battlefield, 3 => ZoneType::Graveyard, 4 => ZoneType::Stack, 5 => ZoneType::Exile, 6 => ZoneType::Command});
integer_enum!(action, ActionType, {0 => ActionType::PriorityPlayLand, 1 => ActionType::PriorityCastSpell, 2 => ActionType::PriorityPassPriority, 3 => ActionType::DeclareAttacker, 4 => ActionType::DeclareBlocker, 5 => ActionType::ChooseTarget, 6 => ActionType::PriorityActivateAbility, 7 => ActionType::ScryKeep, 8 => ActionType::ScryBottom, 9 => ActionType::SelectCard, 10 => ActionType::DeclineChoice, 11 => ActionType::PayCost, 12 => ActionType::ChooseMode, 13 => ActionType::TapForCost, 14 => ActionType::LearnTakeLesson, 15 => ActionType::LearnDiscard, 16 => ActionType::ChooseTrigger});
integer_enum!(space, ActionSpaceKind, {0 => ActionSpaceKind::GameOver, 1 => ActionSpaceKind::Priority, 2 => ActionSpaceKind::DeclareAttacker, 3 => ActionSpaceKind::DeclareBlocker, 4 => ActionSpaceKind::ChooseTarget, 5 => ActionSpaceKind::Scry, 6 => ActionSpaceKind::LookAndSelect, 7 => ActionSpaceKind::PayOrNot, 8 => ActionSpaceKind::Modal, 9 => ActionSpaceKind::Learn, 10 => ActionSpaceKind::Waterbend, 11 => ActionSpaceKind::Discard, 12 => ActionSpaceKind::LegendRule, 13 => ActionSpaceKind::OrderTriggers});
integer_enum!(stack_kind, StackObjectKindData, {0 => StackObjectKindData::Spell, 1 => StackObjectKindData::ActivatedAbility, 2 => StackObjectKindData::TriggeredAbility});
integer_enum!(stack_target, StackTargetKindData, {0 => StackTargetKindData::Player, 1 => StackTargetKindData::Permanent, 2 => StackTargetKindData::StackObject, 3 => StackTargetKindData::Card});

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CardMana {
    cost: [u8; 6],
    mana_value: u8,
    #[serde(default)]
    hybrid: Vec<[u8; 2]>,
}
pub fn mana_cost<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ManaCost, D::Error> {
    let raw = CardMana::deserialize(deserializer)?;
    let mut cost = [0; 7];
    cost[..6].copy_from_slice(&raw.cost);
    let colored = raw.cost.iter().map(|v| *v as usize).sum::<usize>() + raw.hybrid.len();
    cost[6] = usize::from(raw.mana_value)
        .checked_sub(colored)
        .ok_or_else(|| serde::de::Error::custom("invalid mana value"))? as u8;
    let colors = [
        Color::White,
        Color::Blue,
        Color::Black,
        Color::Red,
        Color::Green,
    ];
    let hybrid = raw
        .hybrid
        .into_iter()
        .map(|pair| {
            Ok([
                *colors
                    .get(pair[0] as usize)
                    .ok_or_else(|| serde::de::Error::custom("invalid hybrid color"))?,
                *colors
                    .get(pair[1] as usize)
                    .ok_or_else(|| serde::de::Error::custom("invalid hybrid color"))?,
            ])
        })
        .collect::<Result<_, D::Error>>()?;
    Ok(ManaCost {
        cost,
        mana_value: raw.mana_value,
        hybrid,
        purpose: Default::default(),
        x: 0,
    })
}
pub fn optional_mana_cost<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ManaCost>, D::Error> {
    mana_cost(deserializer).map(Some)
}
pub fn parse(text: &str) -> Result<Observation, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let observation: Observation =
        serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    let roundtrip: serde_json::Value =
        serde_json::from_str(&to_json_with_history(&observation)).map_err(|e| e.to_string())?;
    if value != roundtrip || !observation.validate() {
        return Err("noncanonical viewer observation".into());
    }
    upgrade_historical(observation)
}

/// A capture written before permanents and spells carried their card
/// records lists them as cards: each side's battlefield card rows in
/// permanent order, and stack spells' rows in stack order. Move them onto
/// their permanents and spells, so every reader sees the current shape. A
/// capture whose exact identities cannot be reconstructed is refused.
fn upgrade_historical(mut observation: Observation) -> Result<Observation, String> {
    if !observation.historical() {
        return Ok(observation);
    }
    let original = observation.clone();
    for (cards, permanents) in [
        (
            &mut observation.agent_cards,
            &mut observation.agent_permanents,
        ),
        (
            &mut observation.opponent_cards,
            &mut observation.opponent_permanents,
        ),
    ] {
        let battlefield: Vec<CardData> = cards
            .iter()
            .filter(|card| card.zone == ZoneType::Battlefield)
            .cloned()
            .collect();
        if battlefield.len() != permanents.len()
            || battlefield
                .iter()
                .zip(permanents.iter())
                .any(|(card, permanent)| card.registry_key != permanent.registry_key)
        {
            return Err("historical permanent identity cannot be reconstructed".into());
        }
        cards.retain(|card| card.zone != ZoneType::Battlefield);
        for (permanent, card) in permanents.iter_mut().zip(battlefield) {
            if permanent.power != card.power
                || permanent.toughness != card.toughness
                || permanent.keywords != card.keywords
            {
                // Older captures can record different card and permanent
                // projections. Keep both for legacy encoding; v11 refuses
                // historical observations whose identity is not normalized.
                return Ok(original);
            }
            permanent.object_id = Some(card.id);
            permanent.owner_id = card.owner_id;
            permanent.is_token = card.is_token;
            permanent.characteristics = card.characteristics;
        }
    }
    for object in observation
        .stack_objects
        .iter_mut()
        .filter(|object| object.kind == StackObjectKindData::Spell && object.spell.is_none())
    {
        let candidates: Vec<_> = observation
            .agent_cards
            .iter()
            .chain(&observation.opponent_cards)
            .filter(|card| {
                card.zone == ZoneType::Stack && card.registry_key == object.source_card_registry_key
            })
            .collect();
        if candidates.len() != 1 {
            return Err("historical spell identity is missing or ambiguous".into());
        }
        let id = candidates[0].id;
        let cards = if observation.agent_cards.iter().any(|c| c.id == id) {
            &mut observation.agent_cards
        } else {
            &mut observation.opponent_cards
        };
        let slot = cards.iter().position(|c| c.id == id).unwrap();
        let card = cards.remove(slot);
        object.spell = Some(SpellData::from_legacy_card(card));
    }
    Ok(observation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_captures_move_battlefield_and_spell_rows_onto_their_records() {
        use crate::{
            flow::game::Game,
            state::{game_object::PlayerId, player::PlayerConfig},
        };
        let mut game = Game::new(
            vec![
                PlayerConfig::new(
                    "a",
                    [("Mountain".into(), 20), ("Lightning Bolt".into(), 20)].into(),
                ),
                PlayerConfig::new(
                    "b",
                    [("Forest".into(), 20), ("Llanowar Elves".into(), 20)].into(),
                ),
            ],
            111,
            false,
        );
        let objects: Vec<_> = game.state.zones.latest().collect();
        game.move_object(objects[0], ZoneType::Battlefield);
        game.move_object(objects[40], ZoneType::Battlefield);
        game.push_spell_to_stack(objects[1], PlayerId(0), vec![], vec![], Default::default());
        let current = Observation::for_player(&game, PlayerId(0));
        assert!(!current.historical());
        // The same observation as a capture written before card records.
        let mut historical = crate::agent::legacy_input::view(&current);
        for permanent in historical
            .agent_permanents
            .iter_mut()
            .chain(&mut historical.opponent_permanents)
        {
            permanent.object_id = None;
        }
        for object in &mut historical.stack_objects {
            object.spell = None;
        }
        assert!(historical.historical());
        let upgraded = parse(&historical.to_json()).unwrap();
        assert_eq!(upgraded.to_json(), current.to_json());
        // The v9/v10 rows read the capture's own card lists.
        let legacy = crate::agent::legacy_input::view(&upgraded);
        assert_eq!(legacy.agent_cards, historical.agent_cards);
        assert_eq!(legacy.opponent_cards, historical.opponent_cards);
        assert!(!historical.agent_permanents.is_empty() && !historical.stack_objects.is_empty());
    }

    #[test]
    fn every_action_family_in_the_tensor_contract_decodes() {
        for index in 0..crate::agent::observation_encoder::ACTION_TYPE_DIM {
            assert_eq!(action(serde_json::json!(index)).unwrap() as usize, index);
        }
        assert!(action(serde_json::json!(17)).is_err());
    }

    #[test]
    fn canonical_json_preserves_baseline_bytes_and_hashes() {
        use sha2::{Digest, Sha256};
        use std::{fs, path::Path};

        #[derive(Deserialize)]
        struct Case {
            name: String,
            sha256: String,
            blake3: String,
        }
        #[derive(Deserialize)]
        struct Provenance {
            cases: Vec<Case>,
        }
        // These bytes were captured on the baseline writer. Never derive
        // the expectation by parsing/re-serializing it with the current writer.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/observation_json");
        let provenance: Provenance =
            serde_json::from_str(&fs::read_to_string(root.join("provenance.json")).unwrap())
                .unwrap();
        assert_eq!(provenance.cases.len(), 11);
        for case in provenance.cases {
            let expected = fs::read_to_string(root.join(format!("{}.json", case.name))).unwrap();
            let observation: Observation = serde_json::from_str(&expected).unwrap();
            let actual = observation.to_json();
            assert_eq!(actual.as_bytes(), expected.as_bytes(), "{}", case.name);
            assert_eq!(
                format!("{:x}", Sha256::digest(actual.as_bytes())),
                case.sha256,
                "{}",
                case.name
            );
            assert_eq!(
                blake3::hash(actual.as_bytes()).to_hex().as_str(),
                case.blake3,
                "{}",
                case.name
            );
        }
    }
}

// Frozen schema-8 boundary records. Live observations never store these.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireCardData {
    #[serde(deserialize_with = "super::observation_json::zone")]
    pub zone: ZoneType,
    pub owner_id: i32,
    pub id: i32,
    pub registry_key: i32,
    pub name: String,
    pub text_box: String,
    pub power: i32,
    pub toughness: i32,
    /// Set for an object that is no card: a token or a copy of a spell.
    pub is_token: bool,
    pub is_legendary: bool,
    /// Subtype tags the current pool cares about.
    pub is_ally: bool,
    pub is_lesson: bool,
    /// Ward cost (mana value; 0 = no ward).
    pub ward_cost: i32,
    /// The life ward costs and that price in learning tokens; omitted
    /// unless the ward is paid in life.
    #[serde(default)]
    pub ward_life: i32,
    #[serde(default)]
    pub ward_program: Vec<(u16, i32)>,
    /// Kicker cost (mana value; 0 = no kicker).
    pub kicker_cost: i32,
    pub card_types: CardTypeData,
    pub keywords: KeywordData,
    #[serde(deserialize_with = "super::observation_json::mana_cost")]
    pub mana_cost: ManaCost,
    /// Foretold and face down in exile; listed only for its owner.
    #[serde(default)]
    pub face_down: bool,
    /// The permission to play this card from exile, when it has one.
    #[serde(default)]
    pub cast_permission: Option<CastPermissionData>,
    /// The value of X announced for this spell on the stack (CR 107.3),
    /// when its mana cost has X.
    #[serde(default)]
    pub x: Option<i32>,
    /// A graveyard card that was put there from the battlefield this turn
    /// (`TurnState::graveyard_from_battlefield`); policy input v10.
    #[serde(default)]
    pub from_battlefield_this_turn: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WirePermanentData {
    /// Effective colors when a layer changes the copiable colors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colors: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub keyword_counter_timestamps: BTreeMap<String, i32>,
    /// Transformations of this exact permanent; repeated flips still count.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub transforms: u32,
    pub id: i32,
    /// Definition of the backing card, in the match's bound content pack.
    pub registry_key: i32,
    pub controller_id: i32,
    pub tapped: bool,
    pub damage: i32,
    pub is_summoning_sick: bool,
    pub plus1_counters: i32,
    pub cant_be_blocked_this_turn: bool,
    /// Effective power/toughness — CDA base, static continuous effects,
    /// until-EOT deltas, and +1/+1 counters all applied.
    pub power: i32,
    pub toughness: i32,
    /// A creature only through an effect: earthbend (a land that's also a
    /// 0/0 creature), crew, or another type-changing effect.
    pub is_animated: bool,
    /// This permanent holds cards in exile until it leaves (Jailer).
    pub has_exile_link: bool,
    pub exile_if_dies_this_turn: bool,
    /// Declared as an attacker in the current combat. Omitted when false, so
    /// observations retained before this field stay canonical.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub attacking: bool,
    /// Exhaust abilities this object has activated; each can never be
    /// activated again by this object (CR 702.177a).
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub exhausted_abilities: i32,
    /// "Once each turn" abilities this object has used this turn: activated
    /// abilities, once-each-turn triggers that fired and "do this only once
    /// each turn" options taken.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub once_each_turn_used: i32,
    /// Firebending granted until end of turn, summed over grants.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub granted_firebending: i32,
    pub temp_subtypes: Vec<String>,
    /// Effective keywords — printed keywords unioned with until-EOT grants
    /// (Yip Yip! flying, Enter the Avatar State, ...): what the permanent can
    /// actually do now. The battlefield card entry reads the same layered view.
    pub keywords: KeywordData,
    /// Counters other than +1/+1 by kind (CR 122.1; `CounterKind` names in
    /// snake case), only the kinds present.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub counters: BTreeMap<String, i32>,
    /// The permanent this Aura or Equipment is attached to (CR 301.5, 303.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<i32>,
    /// Name and creature types set by an effect (Honest Work); absent while
    /// the printed ones apply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub creature_types: Option<Vec<String>>,
    /// A land only through an effect (a static land-type grant).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_granted_land: bool,
    /// The attacking creature this one blocks (CR 509.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocking: Option<i32>,
    /// It has lost all abilities (CR 613.1f), until end of turn or while an
    /// Aura sets its characteristics.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub abilities_lost: bool,
    /// Activated abilities (indices in its ability list) whose limit is
    /// spent: exhaust (CR 702.177a) or once each turn (CR 602.5b).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub used_abilities: Vec<usize>,
    /// It was dealt damage this turn, whether or not that damage is still
    /// marked.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dealt_damage_this_turn: bool,
    /// Continuous effects on this permanent, in timestamp order; the
    /// effective values above already apply them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<ContinuousEffectData>,
    /// Its timestamp (CR 613.7d/e), the order its static abilities apply in
    /// among the records; absent while it is `id`, the timestamp it entered
    /// with (an attachment renews it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i32>,
    /// The copy effect that made it a copy; absent for an original.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copy: Option<CopyData>,
    /// Its object's card record on the battlefield: name, rules text, types,
    /// token status, costs and ward, read from the same layered view as the
    /// fields above (its power, toughness and keywords equal them). `id` is
    /// the object's id, which actions on the object (sacrifice, payment and
    /// selection choices) name. Absent only in historical captures, whose
    /// card lists held battlefield objects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card: Option<CardData>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireStackObjectData {
    /// Derived ability text; absent on spells and historical observations.
    #[serde(default)]
    pub description: Option<String>,
    pub stack_object_id: i32,
    #[serde(deserialize_with = "super::observation_json::stack_kind")]
    pub kind: StackObjectKindData,
    pub controller_id: i32,
    pub source_card_registry_key: i32,
    pub source_permanent_id: Option<i32>,
    pub chosen_mode: Option<usize>,
    pub ability_index: Option<i32>,
    pub targets: Vec<StackTargetData>,
    /// The announced value of X (CR 107.3); omitted when zero.
    #[serde(default)]
    pub x: i32,
    /// Keywords a static grants this spell while on the stack (a Lesson's
    /// lifelink); absent when none.
    #[serde(default)]
    pub granted_keywords: Option<KeywordData>,
    /// Whether a spell's kicker cost was paid (CR 702.33).
    #[serde(default)]
    pub kicked: bool,
    /// A spell's card record (zone stack, with its announced X); absent for
    /// abilities and in historical captures, whose card lists held spells.
    #[serde(default)]
    pub card: Option<CardData>,
}

impl From<WireCardData> for CardData {
    fn from(w: WireCardData) -> Self {
        Self {
            zone: w.zone,
            owner_id: w.owner_id,
            id: w.id,
            registry_key: w.registry_key,
            is_token: w.is_token,
            face_down: w.face_down,
            cast_permission: w.cast_permission,
            x: w.x,
            from_battlefield_this_turn: w.from_battlefield_this_turn,
            characteristics: CharacteristicsData {
                name: w.name,
                text_box: w.text_box,
                power: w.power,
                toughness: w.toughness,
                is_legendary: w.is_legendary,
                is_ally: w.is_ally,
                is_lesson: w.is_lesson,
                ward_cost: w.ward_cost,
                ward_life: w.ward_life,
                ward_program: w.ward_program,
                kicker_cost: w.kicker_cost,
                card_types: w.card_types,
                keywords: w.keywords,
                mana_cost: w.mana_cost,
            },
        }
    }
}

impl From<WirePermanentData> for PermanentData {
    fn from(w: WirePermanentData) -> Self {
        let card = w.card;
        Self {
            colors: w.colors,
            keyword_counter_timestamps: w.keyword_counter_timestamps,
            transforms: w.transforms,
            id: w.id,
            registry_key: w.registry_key,
            controller_id: w.controller_id,
            tapped: w.tapped,
            damage: w.damage,
            is_summoning_sick: w.is_summoning_sick,
            plus1_counters: w.plus1_counters,
            cant_be_blocked_this_turn: w.cant_be_blocked_this_turn,
            is_animated: w.is_animated,
            has_exile_link: w.has_exile_link,
            exile_if_dies_this_turn: w.exile_if_dies_this_turn,
            attacking: w.attacking,
            exhausted_abilities: w.exhausted_abilities,
            once_each_turn_used: w.once_each_turn_used,
            granted_firebending: w.granted_firebending,
            temp_subtypes: w.temp_subtypes,
            counters: w.counters,
            attached_to: w.attached_to,
            name: w.name,
            creature_types: w.creature_types,
            is_granted_land: w.is_granted_land,
            blocking: w.blocking,
            abilities_lost: w.abilities_lost,
            used_abilities: w.used_abilities,
            dealt_damage_this_turn: w.dealt_damage_this_turn,
            effects: w.effects,
            timestamp: w.timestamp,
            copy: w.copy,
            object_id: card.as_ref().map(|c| c.id),
            owner_id: card.as_ref().map_or(0, |c| c.owner_id),
            is_token: card.as_ref().is_some_and(|c| c.is_token),
            characteristics: card.map_or_else(
                || CharacteristicsData {
                    power: w.power,
                    toughness: w.toughness,
                    keywords: w.keywords,
                    ..Default::default()
                },
                |c| c.characteristics,
            ),
        }
    }
}

impl PermanentData {
    pub(crate) fn legacy_card(&self) -> Option<CardData> {
        Some(CardData {
            zone: ZoneType::Battlefield,
            owner_id: self.owner_id,
            id: self.object_id?,
            registry_key: self.registry_key,
            is_token: self.is_token,
            face_down: false,
            cast_permission: None,
            x: None,
            from_battlefield_this_turn: false,
            characteristics: self.characteristics.clone(),
        })
    }
}

impl Serialize for PermanentData {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        WirePermanentData {
            colors: self.colors.clone(),
            keyword_counter_timestamps: self.keyword_counter_timestamps.clone(),
            transforms: self.transforms,
            id: self.id,
            registry_key: self.registry_key,
            controller_id: self.controller_id,
            tapped: self.tapped,
            damage: self.damage,
            is_summoning_sick: self.is_summoning_sick,
            plus1_counters: self.plus1_counters,
            cant_be_blocked_this_turn: self.cant_be_blocked_this_turn,
            is_animated: self.is_animated,
            has_exile_link: self.has_exile_link,
            exile_if_dies_this_turn: self.exile_if_dies_this_turn,
            attacking: self.attacking,
            exhausted_abilities: self.exhausted_abilities,
            once_each_turn_used: self.once_each_turn_used,
            granted_firebending: self.granted_firebending,
            temp_subtypes: self.temp_subtypes.clone(),
            counters: self.counters.clone(),
            attached_to: self.attached_to,
            name: self.name.clone(),
            creature_types: self.creature_types.clone(),
            is_granted_land: self.is_granted_land,
            blocking: self.blocking,
            abilities_lost: self.abilities_lost,
            used_abilities: self.used_abilities.clone(),
            dealt_damage_this_turn: self.dealt_damage_this_turn,
            effects: self.effects.clone(),
            timestamp: self.timestamp,
            copy: self.copy.clone(),
            power: self.characteristics.power,
            toughness: self.characteristics.toughness,
            keywords: self.characteristics.keywords.clone(),
            card: self.legacy_card(),
        }
        .serialize(serializer)
    }
}

impl From<WireStackObjectData> for StackObjectData {
    fn from(w: WireStackObjectData) -> Self {
        Self {
            description: w.description,
            stack_object_id: w.stack_object_id,
            kind: w.kind,
            controller_id: w.controller_id,
            source_card_registry_key: w.source_card_registry_key,
            source_permanent_id: w.source_permanent_id,
            chosen_mode: w.chosen_mode,
            ability_index: w.ability_index,
            targets: w.targets,
            x: w.x,
            granted_keywords: w.granted_keywords,
            kicked: w.kicked,
            spell: w.card.map(SpellData::from_legacy_card),
        }
    }
}

impl SpellData {
    /// Schema 8 represents spell copies with the card record's token flag.
    pub(crate) fn from_legacy_card(card: CardData) -> Self {
        Self {
            object_id: card.id,
            owner_id: card.owner_id,
            registry_key: card.registry_key,
            is_copy: card.is_token,
            has_announced_x: card.x.is_some(),
            characteristics: card.characteristics,
        }
    }

    pub(crate) fn legacy_card(&self, x: i32) -> CardData {
        CardData {
            zone: ZoneType::Stack,
            owner_id: self.owner_id,
            id: self.object_id,
            registry_key: self.registry_key,
            is_token: self.is_copy,
            face_down: false,
            cast_permission: None,
            x: self.has_announced_x.then_some(x),
            from_battlefield_this_turn: false,
            characteristics: self.characteristics.clone(),
        }
    }
}
