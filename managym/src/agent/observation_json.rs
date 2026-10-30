//! Decode schema-7 viewer JSON into the authoritative native types.
//! Canonical round-trip validation rejects unknown fields and missing facts.
use crate::{
    agent::{
        action::{ActionSpaceKind, ActionType},
        observation::{Observation, StackObjectKindData, StackTargetKindData},
    },
    flow::turn::{PhaseKind, StepKind},
    state::{
        mana::{Color, ManaCost},
        zone::ZoneType,
    },
};
use serde::{Deserialize, Deserializer};

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
}
