//! Destinations of the shared observation encoder. Integer cells retain their
//! source value until the destination chooses its physical representation.

use super::{
    observation::Observation,
    observation_encoder::{
        encode_rows, EncodedObservationMut, ObservationEncodeError, ObservationEncoderConfig,
    },
    observation_layout::{ArrayLayout, PackedField, Storage},
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cell {
    Integer(i64),
    Scalar(f32),
    /// One signed word occupying two columns in the dense checkpoint layout.
    Word(i32),
    /// Raw fact payload and the divisor for its logical numeric reading.
    Payload(i64, u8),
}

impl Cell {
    pub const ZERO: Self = Self::Integer(0);

    pub fn scalar(self) -> f32 {
        match self {
            Self::Integer(n) => n as f32,
            Self::Scalar(n) => n,
            Self::Payload(n, divisor) => n as f32 / f32::from(divisor),
            Self::Word(_) => unreachable!("words have a dedicated dense projection"),
        }
    }

    pub fn integer(self) -> Result<i32, ObservationEncodeError> {
        match self {
            Self::Integer(n) | Self::Payload(n, _) => i32::try_from(n).map_err(|_| {
                ObservationEncodeError::Unrepresentable(format!("integer {n} exceeds int32"))
            }),
            Self::Word(n) => Ok(n),
            Self::Scalar(_) => unreachable!("scalar in an integer column"),
        }
    }
}

pub trait RowWriter {
    fn write(
        &mut self,
        table: Table,
        seat: usize,
        slot: usize,
        row: &[Cell],
    ) -> Result<(), ObservationEncodeError>;
    fn reset_memory(&mut self) {}
}

macro_rules! tables {
    ($($table:ident => $field:ident),*) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum Table { $($table,)* ActionFocus }

        impl Table {
            pub fn name(self) -> &'static str {
                match self { $(Self::$table => stringify!($field),)* Self::ActionFocus => "action_focus" }
            }
        }
        impl EncodedObservationMut<'_> {
            fn values(&mut self, table: Table) -> &mut [f32] {
                match table { $(Table::$table => self.$field,)* Table::ActionFocus => unreachable!() }
            }
        }
    }
}
tables! {
    Relations => relations,
    Effects => effects,
    EffectDetails => effect_details,
    EffectMembers => effect_members,
    NameCandidates => name_candidates,
    Facts => facts,
    SemanticCards => semantic_cards,
    KnownHand => known_hand,
    KnownLibrary => known_library,
    Decklists => decklists,
    PlayerTurnState => player_turn_state,
    CardTurnState => card_turn_state,
    AgentPlayer => agent_player,
    OpponentPlayer => opponent_player,
    AgentCards => agent_cards,
    OpponentCards => opponent_cards,
    AgentPermanents => agent_permanents,
    OpponentPermanents => opponent_permanents,
    Stack => stack,
    Actions => actions,
    Events => events,
    DecisionRecords => decision_records,
    DecisionReferenceDetails => decision_reference_details,
    DecisionReferences => decision_references,
    DecisionMembers => decision_members
}

pub(super) fn write_dense_row(row: &[Cell], target: &mut [f32]) {
    let mut i = 0;
    while i < row.len() {
        match row[i] {
            Cell::Word(n) => {
                target[i] = (n as u32 >> 16) as f32;
                target[i + 1] = (n as u32 & 65535) as f32;
                i += 2;
            }
            cell => {
                target[i] = cell.scalar();
                i += 1;
            }
        }
    }
}

impl RowWriter for EncodedObservationMut<'_> {
    fn write(
        &mut self,
        table: Table,
        seat: usize,
        slot: usize,
        row: &[Cell],
    ) -> Result<(), ObservationEncodeError> {
        if table == Table::ActionFocus {
            let start = slot * row.len();
            for (dst, value) in self.action_focus[start..].iter_mut().zip(row) {
                *dst = value.integer()?;
            }
            return Ok(());
        }
        let buffer = self.values(table);
        let start = seat * (buffer.len() / 2) + slot * row.len();
        let target = &mut buffer[start..start + row.len()];
        write_dense_row(row, target);
        match table {
            Table::AgentPlayer => self.agent_player_valid[slot] = 1.0,
            Table::OpponentPlayer => self.opponent_player_valid[slot] = 1.0,
            Table::AgentCards => self.agent_cards_valid[slot] = 1.0,
            Table::OpponentCards => self.opponent_cards_valid[slot] = 1.0,
            Table::AgentPermanents => self.agent_permanents_valid[slot] = 1.0,
            Table::OpponentPermanents => self.opponent_permanents_valid[slot] = 1.0,
            Table::Stack => self.stack_valid[slot] = 1.0,
            Table::Actions => self.actions_valid[slot] = 1.0,
            Table::Events => self.events_valid[slot] = 1.0,
            _ => {}
        }
        Ok(())
    }

    fn reset_memory(&mut self) {
        self.decision_records.fill(0.0);
        self.decision_reference_details.fill(0.0);
        self.decision_references.fill(0.0);
        self.decision_members.fill(0.0);
    }
}

pub struct PackedTable {
    pub offsets: Vec<i32>,
    pub values: Vec<u8>,
    fields: Vec<PackedField>,
    element: bool,
    pub seat_axis: bool,
    counts: [i32; 2],
}

pub struct PackedRows {
    pub tables: BTreeMap<String, PackedTable>,
}

impl RowWriter for PackedRows {
    fn write(
        &mut self,
        table: Table,
        seat: usize,
        slot: usize,
        row: &[Cell],
    ) -> Result<(), ObservationEncodeError> {
        let packed = self
            .tables
            .get_mut(table.name())
            .expect("configured observation table");
        debug_assert_eq!(slot, packed.counts[seat] as usize);
        packed.counts[seat] = packed.counts[seat]
            .checked_add(1)
            .ok_or_else(offset_overflow)?;
        let mut append = |cell: Cell, storage| -> Result<(), ObservationEncodeError> {
            match storage {
                Storage::Int32 => packed
                    .values
                    .extend_from_slice(&cell.integer()?.to_ne_bytes()),
                Storage::Uint8 => {
                    let n = cell.integer()?;
                    debug_assert!(matches!(n, 0 | 1));
                    packed.values.push(n as u8);
                }
                Storage::Float32 => packed
                    .values
                    .extend_from_slice(&cell.scalar().to_ne_bytes()),
            }
            Ok(())
        };
        if packed.element {
            for cell in row {
                append(*cell, packed.fields[0].storage)?;
            }
        } else {
            for field in &packed.fields {
                append(row[field.index], field.storage)?;
            }
        }
        Ok(())
    }
}

fn offset_overflow() -> ObservationEncodeError {
    ObservationEncodeError::Unrepresentable("observation batch exceeds int32 row offsets".into())
}

impl PackedRows {
    pub fn new(arrays: &BTreeMap<String, ArrayLayout>) -> Self {
        Self {
            tables: arrays
                .iter()
                .filter(|(key, _)| !key.ends_with("_valid"))
                .map(|(key, array)| {
                    let seat_axis =
                        array.axes[0] == "viewer_relative_seat" && key != "player_turn_state";
                    (
                        key.clone(),
                        PackedTable {
                            fields: array.packed_fields(key),
                            element: array.element.is_some(),
                            seat_axis,
                            counts: [0; 2],
                            offsets: vec![0],
                            values: Vec::new(),
                        },
                    )
                })
                .collect(),
        }
    }
    /// Append a complete game in seat order, including offsets for empty seats.
    /// On an encoding error the caller discards this batch.
    pub fn append(
        &mut self,
        obs: &Observation,
        config: &ObservationEncoderConfig,
    ) -> Result<(), ObservationEncodeError> {
        encode_rows(obs, config, self, None)?;
        for table in self.tables.values_mut() {
            for count in &table.counts[..if table.seat_axis { 2 } else { 1 }] {
                table.offsets.push(
                    table
                        .offsets
                        .last()
                        .unwrap()
                        .checked_add(*count)
                        .ok_or_else(offset_overflow)?,
                );
            }
            table.counts = [0; 2];
        }
        Ok(())
    }
}
