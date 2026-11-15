//! Complete continuous-effect payloads and exact memberships, before tensor admission.
//!
//! This is a viewer projection of public resolved effects, not a second rules
//! store. Equal complete payloads share storage; every original record retains
//! an ordered membership, including duplicates. No timestamps are coalesced.

use super::observation_layout::{detail, effect, member};

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use crate::{
    agent::observation::ContinuousEffectData,
    flow::game::Game,
    state::{
        effect::Affected,
        game_object::{ObjectId, PlayerId},
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
    fn permanent(game: &Game, object: ObjectId) -> Self {
        Self {
            object_id: game.lookup_current_permanent(object).ok().map(|id| {
                game.state.permanents[id]
                    .as_ref()
                    .expect("current permanent")
                    .id
                    .0 as i32
            }),
            incarnation: Some(object.generation()),
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
/// Source-dependent expiry is derived from the native ObjectId, never a name.
pub fn project(game: &Game) -> Vec<SharedEffectData> {
    let mut shared: Vec<SharedEffectData> = Vec::new();
    for (order, record) in game.state.effects.iter().enumerate() {
        let effect = ContinuousEffectData::new(record);
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

// CSR: headers own half-open slices of typed details and object memberships.
// Header: modification halves, three values, three presence flags, duration,
// timestamp rank, controller/source joins, detail offset/count, member offset/count,
// validity. A membership retains its original one-based record order.
pub const EFFECT_ROWS: usize = 128;
pub const EFFECT_DIM: usize = effect::WIDTH;
pub const DETAIL_ROWS: usize = 4096;
pub const DETAIL_DIM: usize = detail::WIDTH;
pub const MEMBER_ROWS: usize = 2048;
pub const MEMBER_DIM: usize = member::WIDTH;

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
            "policy input requires native shared-effect provenance; historical absence is unknown",
        ))?;
    let detail_count: usize = records
        .iter()
        .map(|record| {
            record.effect.program.len()
                + record.effect.keywords.len()
                + record.effect.subtypes.len()
                + usize::from(record.effect.color.is_some())
        })
        .sum();
    let member_count: usize = records.iter().map(|record| record.members.len()).sum();
    for (field, actual, capacity) in [
        ("effects", records.len(), EFFECT_ROWS),
        ("effect_details", detail_count, details.len() / DETAIL_DIM),
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
        row[effect::MODIFICATION_HIGH..effect::POWER].copy_from_slice(&split_id(name_value(
            &format!("Modification::{}", effect.modification),
        )));
        for (i, value) in [effect.power, effect.toughness, effect.amount]
            .into_iter()
            .enumerate()
        {
            row[effect::POWER + i] = value.unwrap_or(0) as f32 / 10.0;
            row[effect::HAS_POWER + i] = u8::from(value.is_some()) as f32;
        }
        row[effect::DURATION] = duration_code(&effect.duration)
            .filter(|_| effect.duration != "WhileYouControlSource" || record.source.is_some())
            .ok_or(Error::InvalidShared(
                "invalid shared effect duration/source",
            ))?;
        row[effect::TIMESTAMP] = *ranks
            .get(&effect.timestamp)
            .ok_or(Error::InvalidShared("missing effect timestamp"))?;
        row[effect::CONTROLLER] = join(Some(record.controller_id))?;
        row[effect::SOURCE_DEFINITION] =
            effect.source_registry_key.map_or(0.0, |id| (id + 1) as f32);
        row[effect::SOURCE] = record
            .source
            .as_ref()
            .map(|source| join(source.object_id))
            .transpose()?
            .unwrap_or(0.0);
        row[effect::DETAIL_OFFSET] = d as f32;
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
            details[d * DETAIL_DIM + detail::KIND] = kind;
            details[d * DETAIL_DIM + detail::VALUE_HIGH..(d + 1) * DETAIL_DIM]
                .copy_from_slice(&split_id(token));
            d += 1;
        }
        for &(kind, value) in &effect.program {
            details[d * DETAIL_DIM + detail::KIND] = 10.0 + f32::from(kind);
            details[d * DETAIL_DIM + detail::VALUE_HIGH..(d + 1) * DETAIL_DIM]
                .copy_from_slice(&split_id(value));
            d += 1;
        }
        row[effect::DETAIL_COUNT] = d as f32 - row[effect::DETAIL_OFFSET];
        row[effect::MEMBER_OFFSET] = m as f32;
        if record.members.is_empty() {
            return Err(Error::InvalidShared("shared effect has no memberships"));
        }
        for member in &record.members {
            if member.order >= MEMBER_ROWS || !orders.insert(member.order) {
                return Err(Error::InvalidShared(
                    "invalid shared effect membership order",
                ));
            }
            members[m * MEMBER_DIM + member::OBJECT] = join(member.target.object_id)?;
            members[m * MEMBER_DIM + member::ORDER] = (member.order + 1) as f32;
            m += 1;
        }
        row[effect::MEMBER_COUNT] = m as f32 - row[effect::MEMBER_OFFSET];
        row[effect::VALID] = 1.0;
    }
    Ok(())
}

/// The header duration column and the copy-exception fact's duration: the record's
/// `EffectDuration` code, or `None` for a name no record carries.
pub(super) fn duration_code(duration: &str) -> Option<f32> {
    match duration {
        "EndOfTurn" => Some(1.0),
        "Indefinite" => Some(2.0),
        "WhileYouControlSource" => Some(3.0),
        "UntilYourNextTurn" => Some(4.0),
        _ => None,
    }
}
