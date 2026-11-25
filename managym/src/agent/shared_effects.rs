//! Complete continuous-effect payloads and exact memberships, before tensor admission.
//!
//! This is a viewer projection of public resolved effects, not a second rules
//! store. Equal complete payloads share storage; every original record retains
//! an ordered membership, including duplicates. No timestamps are coalesced.

use super::observation_layout::{detail, effect, member};
use super::observation_rows::{
    Cell::{self, Integer, Scalar, Word},
    RowWriter, Table,
};

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use crate::{
    agent::observation::ContinuousEffectData,
    cards::tokens::TokenKind,
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
            incarnation: Some(
                game.object_event_ref(object)
                    .map_or(object.generation(), |event| event.incarnation),
            ),
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

pub const PROGRAM_DETAIL_KINDS: [(&str, TokenKind); 9] = [
    ("program_structure", TokenKind::Structure),
    ("program_field", TokenKind::Field),
    ("program_variant", TokenKind::Variant),
    ("program_integer", TokenKind::Integer),
    ("program_boolean", TokenKind::Boolean),
    ("program_text", TokenKind::Text),
    ("program_definition_ref", TokenKind::DefinitionRef),
    ("program_null", TokenKind::Null),
    ("program_program_kind", TokenKind::ProgramKind),
];
pub fn program_detail_kind(kind: u16) -> u16 {
    10 + kind
}

pub const DETAIL_ROWS: usize = 4096;
pub const DETAIL_DIM: usize = detail::WIDTH;
pub const MEMBER_ROWS: usize = 2048;
pub const MEMBER_DIM: usize = member::WIDTH;

pub fn encode(
    observation: &super::observation::Observation,
    objects: &HashMap<i32, i32>,
    ranks: &BTreeMap<i32, i64>,
    detail_capacity: usize,
    out: &mut impl RowWriter,
) -> Result<[usize; 3], super::observation_encoder::ObservationEncodeError> {
    use super::observation_encoder::ObservationEncodeError as Error;
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
        ("effect_details", detail_count, detail_capacity),
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
    let join = |id: Option<i32>| -> Result<Cell, Error> {
        id.and_then(|id| objects.get(&id))
            .map(|index| Integer(i64::from(*index) + 1))
            .ok_or(Error::InvalidShared(
                "shared effect has an unavailable object join",
            ))
    };
    let (mut d, mut m) = (0, 0);
    let mut orders = std::collections::BTreeSet::new();
    for (slot, record) in records.iter().enumerate() {
        let mut row = [Cell::ZERO; EFFECT_DIM];
        let effect = &record.effect;
        row[effect::MODIFICATION_HIGH] = Word(name_value(&format!(
            "Modification::{}",
            effect.modification
        )));
        for (i, value) in [effect.power, effect.toughness, effect.amount]
            .into_iter()
            .enumerate()
        {
            row[effect::POWER + i] = Scalar(value.unwrap_or(0) as f32 / 10.0);
            row[effect::HAS_POWER + i] = Integer(i64::from(value.is_some()));
        }
        row[effect::DURATION] = Integer(
            duration_code(&effect.duration)
                .filter(|_| effect.duration != "WhileYouControlSource" || record.source.is_some())
                .ok_or(Error::InvalidShared(
                    "invalid shared effect duration/source",
                ))?,
        );
        row[effect::TIMESTAMP] = Integer(
            *ranks
                .get(&effect.timestamp)
                .ok_or(Error::InvalidShared("missing effect timestamp"))?,
        );
        row[effect::CONTROLLER] = join(Some(record.controller_id))?;
        row[effect::SOURCE_DEFINITION] =
            Integer(effect.source_registry_key.map_or(0, |id| i64::from(id) + 1));
        row[effect::SOURCE] = record
            .source
            .as_ref()
            .map(|source| join(source.object_id))
            .transpose()?
            .unwrap_or(Cell::ZERO);
        let detail_start = d;
        row[effect::DETAIL_OFFSET] = Integer(d as i64);
        for (kind, token) in effect
            .keywords
            .iter()
            .map(|s| (1, name_value(s)))
            .chain(effect.subtypes.iter().map(|s| (2, name_value(s))))
            .chain(
                effect
                    .color
                    .iter()
                    .map(|s| (3, name_value(&format!("Color::{s}")))),
            )
        {
            out.write(
                Table::EffectDetails,
                0,
                d,
                &[Integer(kind), Word(token), Cell::ZERO],
            )?;
            d += 1;
        }
        for &(kind, value) in &effect.program {
            out.write(
                Table::EffectDetails,
                0,
                d,
                &[
                    Integer(i64::from(program_detail_kind(kind))),
                    Word(value),
                    Cell::ZERO,
                ],
            )?;
            d += 1;
        }
        row[effect::DETAIL_COUNT] = Integer((d - detail_start) as i64);
        let member_start = m;
        row[effect::MEMBER_OFFSET] = Integer(m as i64);
        if record.members.is_empty() {
            return Err(Error::InvalidShared("shared effect has no memberships"));
        }
        for member in &record.members {
            if member.order >= MEMBER_ROWS || !orders.insert(member.order) {
                return Err(Error::InvalidShared(
                    "invalid shared effect membership order",
                ));
            }
            out.write(
                Table::EffectMembers,
                0,
                m,
                &[
                    join(member.target.object_id)?,
                    Integer(member.order as i64 + 1),
                ],
            )?;
            m += 1;
        }
        row[effect::MEMBER_COUNT] = Integer((m - member_start) as i64);
        row[effect::VALID] = Integer(1);
        out.write(Table::Effects, 0, slot, &row)?;
    }
    Ok([records.len(), d, m])
}

/// The header duration column and the copy-exception fact's duration: the record's
/// `EffectDuration` code, or `None` for a name no record carries.
pub(super) fn duration_code(duration: &str) -> Option<i64> {
    match duration {
        "EndOfTurn" => Some(1),
        "Indefinite" => Some(2),
        "WhileYouControlSource" => Some(3),
        "UntilYourNextTurn" => Some(4),
        _ => None,
    }
}
