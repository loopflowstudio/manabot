//! Complete continuous-effect payloads and exact memberships, before tensor admission.
//!
//! This is a viewer projection of public resolved effects, not a second rules
//! store. Equal complete payloads share storage; every original record retains
//! an ordered membership, including duplicates. No timestamps are coalesced.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use crate::{
    agent::observation::ContinuousEffectData,
    flow::game::Game,
    state::{
        effect::Affected,
        game_object::{ObjectRef, PlayerId},
    },
};

/// A current public join plus its exact rules identity. Players have no incarnation.
/// An unavailable join never follows a physical card into a newer incarnation.
/// Private physical-card storage indexes never enter this public projection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct EffectObject {
    pub object_id: Option<i32>,
    pub incarnation: Option<u32>,
}

impl EffectObject {
    fn permanent(game: &Game, object: ObjectRef) -> Self {
        Self {
            object_id: game.lookup_current_permanent(object).ok().map(|id| {
                game.state.permanents[id]
                    .as_ref()
                    .expect("current permanent")
                    .id
                    .0 as i32
            }),
            incarnation: Some(object.incarnation.0),
        }
    }

    fn player(game: &Game, player: PlayerId) -> Self {
        Self {
            object_id: Some(game.state.players[player.0].id.0 as i32),
            incarnation: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct EffectMembership {
    pub target: EffectObject,
    /// Original position in EffectRecords, allowing exact ordered expansion.
    pub order: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedEffectData {
    pub effect: ContinuousEffectData,
    pub controller_id: i32,
    pub source: Option<EffectObject>,
    pub members: Vec<EffectMembership>,
}

/// Both viewers may see every resolved effect on a public permanent/player.
/// Source-dependent expiry is derived from the native ObjectRef, never a name.
pub fn project(game: &Game) -> Vec<SharedEffectData> {
    let mut shared: Vec<SharedEffectData> = Vec::new();
    for (order, record) in game.state.effects.iter().enumerate() {
        let effect = ContinuousEffectData::new(game, record);
        let controller_id = game.state.players[record.controller.0].id.0 as i32;
        let source = record
            .source_object
            .map(|object| EffectObject::permanent(game, object));
        let target = match record.affected {
            Affected::Player(player) => EffectObject::player(game, player),
            Affected::Object(object) => EffectObject::permanent(game, object),
        };
        let membership = EffectMembership { target, order };
        if let Some(payload) = shared.iter_mut().find(|payload| {
            payload.effect == effect
                && payload.controller_id == controller_id
                && payload.source == source
        }) {
            payload.members.push(membership);
        } else {
            shared.push(SharedEffectData {
                effect,
                controller_id,
                source,
                members: vec![membership],
            });
        }
    }
    shared
}

// V7 CSR: headers own half-open slices of typed details and object memberships.
// Header: modification halves, three values, three presence flags, duration,
// timestamp rank, controller/source joins, detail offset/count, member offset/count,
// validity. A membership retains its original one-based record order.
pub const EFFECT_ROWS: usize = 128;
pub const EFFECT_DIM: usize = 18;
pub const DETAIL_ROWS: usize = 512;
pub const DETAIL_DIM: usize = 3;
pub const MEMBER_ROWS: usize = 2048;
pub const MEMBER_DIM: usize = 2;

pub fn encode(
    observation: &super::observation::Observation,
    objects: &HashMap<i32, i32>,
    ranks: &BTreeMap<i32, f32>,
    headers: &mut [f32],
    details: &mut [f32],
    members: &mut [f32],
) -> Result<(), super::observation_encoder::ObservationEncodeError> {
    use super::observation_encoder::{split_id, ObservationEncodeError as Error};
    use crate::cards::tokens::name_value;

    let records = observation
        .shared_effects
        .as_ref()
        .ok_or(Error::InvalidShared(
            "v7 requires native shared-effect provenance; historical absence is unknown",
        ))?;
    let detail_count: usize = records
        .iter()
        .map(|record| {
            record.effect.keywords.len()
                + record.effect.subtypes.len()
                + usize::from(record.effect.color.is_some())
        })
        .sum();
    let member_count: usize = records.iter().map(|record| record.members.len()).sum();
    for (field, actual, capacity) in [
        ("effects", records.len(), EFFECT_ROWS),
        ("effect_details", detail_count, DETAIL_ROWS),
        ("effect_members", member_count, MEMBER_ROWS),
    ] {
        if actual > capacity {
            return Err(Error::Capacity {
                field,
                actual,
                capacity,
            });
        }
    }
    let join = |id: Option<i32>| -> Result<f32, Error> {
        id.and_then(|id| objects.get(&id))
            .map(|index| (index + 1) as f32)
            .ok_or(Error::InvalidShared(
                "shared effect has an unavailable object join",
            ))
    };
    headers.fill(0.0);
    details.fill(0.0);
    members.fill(0.0);
    let (mut d, mut m) = (0, 0);
    let mut orders = std::collections::BTreeSet::new();
    for (row, record) in headers
        .as_chunks_mut::<EFFECT_DIM>()
        .0
        .iter_mut()
        .zip(records)
    {
        let effect = &record.effect;
        row[..2].copy_from_slice(&split_id(name_value(&format!(
            "Modification::{}",
            effect.modification
        ))));
        for (i, value) in [effect.power, effect.toughness, effect.amount]
            .into_iter()
            .enumerate()
        {
            row[2 + i] = value.unwrap_or(0) as f32 / 10.0;
            row[5 + i] = u8::from(value.is_some()) as f32;
        }
        row[8] = match effect.duration.as_str() {
            "EndOfTurn" => 1.0,
            "Indefinite" => 2.0,
            "WhileYouControlSource" if record.source.is_some() => 3.0,
            _ => {
                return Err(Error::InvalidShared(
                    "invalid shared effect duration/source",
                ))
            }
        };
        row[9] = *ranks
            .get(&effect.timestamp)
            .ok_or(Error::InvalidShared("missing effect timestamp"))?;
        row[10] = join(Some(record.controller_id))?;
        row[11] = effect.source_registry_key.map_or(0.0, |id| (id + 1) as f32);
        row[12] = record
            .source
            .as_ref()
            .map(|source| join(source.object_id))
            .transpose()?
            .unwrap_or(0.0);
        row[13] = d as f32;
        for (kind, token) in effect
            .keywords
            .iter()
            .map(|s| (1.0, name_value(s)))
            .chain(effect.subtypes.iter().map(|s| (2.0, name_value(s))))
            .chain(
                effect
                    .color
                    .iter()
                    .map(|s| (3.0, name_value(&format!("Color::{s}")))),
            )
        {
            details[d * DETAIL_DIM] = kind;
            details[d * DETAIL_DIM + 1..(d + 1) * DETAIL_DIM].copy_from_slice(&split_id(token));
            d += 1;
        }
        row[14] = d as f32 - row[13];
        row[15] = m as f32;
        if record.members.is_empty() {
            return Err(Error::InvalidShared("shared effect has no memberships"));
        }
        for member in &record.members {
            if member.order >= MEMBER_ROWS || !orders.insert(member.order) {
                return Err(Error::InvalidShared(
                    "invalid shared effect membership order",
                ));
            }
            members[m * MEMBER_DIM] = join(member.target.object_id)?;
            members[m * MEMBER_DIM + 1] = (member.order + 1) as f32;
            m += 1;
        }
        row[16] = m as f32 - row[15];
        row[17] = 1.0;
    }
    Ok(())
}
