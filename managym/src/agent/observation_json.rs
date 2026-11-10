//! Canonical schema-8 viewer JSON: one borrowed writer and the typed decoder.
//! Handwritten maps emit lexical keys directly; derived nested records retain
//! sorted Value boundaries. Exact bytes bind semantic and search-witness hashes.
//! Canonical round-trip validation rejects unknown fields and missing facts.
use crate::{
    agent::{
        action::{ActionSpaceKind, ActionType},
        observation::{
            ActionOption, ActionSpaceData, CardData, CastPermissionData, EventData, Observation,
            StackObjectData, StackObjectKindData, StackTargetData, StackTargetKindData, TurnData,
        },
    },
    flow::turn::{PhaseKind, StepKind},
    state::{
        mana::{Color, ManaCost},
        zone::ZoneType,
    },
};
use serde::{ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};

/// Emit the schema's canonical bytes without constructing its handwritten trees.
/// Every map below writes keys lexically; these bytes bind SHA-256 semantic
/// identities and BLAKE3 viewer witnesses, including omissions and nulls.
pub(super) fn to_json(observation: &Observation) -> String {
    serde_json::to_string(&ObservationJson(observation))
        .expect("observation fields are JSON serializable")
}

/// Derived records retain their own serde omissions and enum representations.
/// Going through Value sorts all nested keys (including numeric map keys after
/// conversion to strings). Direct derived serialization uses declaration order.
/// Keep this boundary at these leaves, never around the handwritten views.
fn sorted_record(value: &impl Serialize) -> serde_json::Value {
    serde_json::to_value(value).expect("observation records are JSON serializable")
}

struct ObservationJson<'a>(&'a Observation);

impl Serialize for ObservationJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("action_space", &ActionSpaceJson(&value.action_space))?;
        map.serialize_entry("agent", &sorted_record(&value.agent))?;
        map.serialize_entry("agent_cards", &CardsJson(&value.agent_cards))?;
        map.serialize_entry("agent_permanents", &sorted_record(&value.agent_permanents))?;
        map.serialize_entry("agent_sideboard", &sorted_record(&value.agent_sideboard))?;
        map.serialize_entry("game_over", &value.game_over)?;
        map.serialize_entry("opponent", &sorted_record(&value.opponent))?;
        map.serialize_entry("opponent_cards", &CardsJson(&value.opponent_cards))?;
        map.serialize_entry(
            "opponent_permanents",
            &sorted_record(&value.opponent_permanents),
        )?;
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
        map.serialize_entry("chosen_mode", &value.chosen_mode)?;
        map.serialize_entry("controller_id", &value.controller_id)?;
        map.serialize_entry("description", &value.description)?;
        if let Some(keywords) = &value.granted_keywords {
            map.serialize_entry("granted_keywords", &sorted_record(keywords))?;
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
        serde_json::from_str(&observation.to_json()).map_err(|e| e.to_string())?;
    if value != roundtrip || !observation.validate() {
        return Err("noncanonical viewer observation".into());
    }
    Ok(observation)
}

#[cfg(test)]
mod tests {
    use super::*;

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
