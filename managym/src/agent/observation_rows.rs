//! Destinations of the shared observation encoder. Integer cells retain their
//! source value until the destination chooses its physical representation.
//!
//! `PackedRows` is the transport: CSR batches for policies. `InputRows` reads
//! one observation's rows in their dense column reading, for engine tests and
//! benchmarks that inspect a column.

use super::{
    observation::Observation,
    observation_encoder::{encode_rows, ObservationEncodeError, ObservationEncoderConfig},
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

pub(crate) fn write_dense_row(row: &[Cell], target: &mut [f32]) {
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

/// One observation's rows by table and seat. Each row is in its dense column
/// reading: a signed word fills its high and low columns, a payload reads as
/// its scaled value. Only live rows are present; there is no padding.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputRows {
    tables: BTreeMap<&'static str, [Vec<Vec<f32>>; 2]>,
}

impl InputRows {
    pub fn read(
        obs: &Observation,
        config: &ObservationEncoderConfig,
    ) -> Result<Self, ObservationEncodeError> {
        let mut rows = Self::default();
        encode_rows(obs, config, &mut rows)?;
        Ok(rows)
    }

    /// The rows of `table` for `seat` (0 for tables without a seat axis).
    pub fn seat(&self, table: &str, seat: usize) -> &[Vec<f32>] {
        self.tables.get(table).map_or(&[], |seats| &seats[seat])
    }

    /// Every row of `table`, seat 0 first.
    pub fn rows(&self, table: &str) -> Vec<&[f32]> {
        (0..2)
            .flat_map(|seat| self.seat(table, seat))
            .map(Vec::as_slice)
            .collect()
    }

    /// The `slot`th row of `table` across seats.
    pub fn row(&self, table: &str, slot: usize) -> &[f32] {
        self.rows(table)[slot]
    }

    /// Every cell of `table` in row order.
    pub fn cells(&self, table: &str) -> Vec<f32> {
        self.rows(table).concat()
    }

    pub fn count(&self, table: &str) -> usize {
        self.rows(table).len()
    }
}

impl RowWriter for InputRows {
    fn write(
        &mut self,
        table: Table,
        seat: usize,
        slot: usize,
        row: &[Cell],
    ) -> Result<(), ObservationEncodeError> {
        let rows = &mut self.tables.entry(table.name()).or_default()[seat];
        debug_assert_eq!(slot, rows.len());
        let mut dense = vec![0.0; row.len()];
        write_dense_row(row, &mut dense);
        rows.push(dense);
        Ok(())
    }
}

pub struct PackedTable {
    pub offsets: Vec<i32>,
    pub values: Vec<u8>,
    fields: Vec<PackedField>,
    element: bool,
    pub seat_axis: bool,
    counts: [i32; 2],
    committed_bytes: usize,
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
                    let seat_axis = array.packed_seat_axis(key);
                    (
                        key.clone(),
                        PackedTable {
                            fields: array.packed_fields(key),
                            element: array.element.is_some(),
                            seat_axis,
                            counts: [0; 2],
                            committed_bytes: 0,
                            offsets: vec![0],
                            values: Vec::new(),
                        },
                    )
                })
                .collect(),
        }
    }
    /// Append a complete game in seat order, including offsets for empty seats.
    /// Roll back partial records before an isolated writer continues on a sibling.
    pub fn append(
        &mut self,
        obs: &Observation,
        config: &ObservationEncoderConfig,
    ) -> Result<(), ObservationEncodeError> {
        let result = encode_rows(obs, config, self).and_then(|()| {
            for table in self.tables.values() {
                table
                    .counts
                    .iter()
                    .try_fold(*table.offsets.last().unwrap(), |n, count| {
                        n.checked_add(*count).ok_or_else(offset_overflow)
                    })?;
            }
            Ok(())
        });
        for table in self.tables.values_mut() {
            if result.is_ok() {
                for count in &table.counts[..if table.seat_axis { 2 } else { 1 }] {
                    table.offsets.push(table.offsets.last().unwrap() + count);
                }
                table.committed_bytes = table.values.len();
            } else {
                table.values.truncate(table.committed_bytes);
            }
            table.counts = [0; 2];
        }
        result
    }
}
