//! A JSON header binds the configuration and packed schema. In table-name order,
//! offsets determine each values block's length. A final SHA-256 covers both.
//! All indexing below follows layout-derived widths and bounds checked on borrowed bytes.
use super::{FrozenTable, ObservationBatch, ObservationLayout, PackedTableLayout};
use crate::agent::{
    continuation::RELATION_ROWS,
    observation_encoder::{FactKind, ObservationEncoderConfig},
    observation_extents::{joins, Join},
    observation_layout::{
        action, decision_member, decision_record, decision_reference, decision_reference_detail,
        detail, effect, fact, history, member, name, relation, ArrayLayout, Storage,
    },
    shared_effects::MEMBER_ROWS,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    format: String,
    batch_size: usize,
    config: SavedConfig,
    schema: String,
}

// These wire fields identify v11 in existing format-1 archives.
#[derive(Serialize, Deserialize)]
struct SavedConfig {
    layout: String,
    turn_state: bool,
    #[serde(flatten)]
    encoder: ObservationEncoderConfig,
}

const FORMAT: &str = if cfg!(target_endian = "little") {
    "managym.ObservationBatch/1-le"
} else {
    "managym.ObservationBatch/1-be"
};

fn schema(layout: &ObservationLayout) -> String {
    let mut digest = Sha256::new();
    for (key, array) in &layout.arrays {
        let fields: Vec<_> = array
            .packed_fields(key)
            .into_iter()
            .map(|f| (f.name, f.storage.dtype(), f.index))
            .collect();
        serde_json::to_writer(
            &mut digest,
            &(
                key,
                &array.axes,
                &array.shape,
                fields,
                &array.row_kind_names,
            ),
        )
        .expect("native schema serializes");
    }
    format!("{:x}", digest.finalize())
}

pub(super) fn encode<'py>(
    py: Python<'py>,
    batch: &ObservationBatch,
) -> PyResult<Bound<'py, PyBytes>> {
    let layout = batch.layout.borrow(py);
    let mut bytes = serde_json::to_vec(&Header {
        format: FORMAT.into(),
        config: SavedConfig {
            layout: "OneToken".into(),
            turn_state: true,
            encoder: layout.config,
        },
        batch_size: batch.batch_size,
        schema: schema(&layout),
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    bytes.push(b'\n');
    for table in batch.tables.values() {
        bytes.extend_from_slice(table.offsets.as_bytes(py));
        bytes.extend_from_slice(table.values.as_bytes(py));
    }
    bytes.extend_from_slice(&Sha256::digest(&bytes));
    Ok(PyBytes::new_bound(py, &bytes))
}

fn require(valid: bool, what: &'static str) -> PyResult<()> {
    if valid {
        Ok(())
    } else {
        Err(PyValueError::new_err(what))
    }
}

fn take<'a>(input: &mut &'a [u8], len: usize) -> PyResult<&'a [u8]> {
    let (prefix, rest) = input
        .split_at_checked(len)
        .ok_or_else(|| PyValueError::new_err("truncated observation batch"))?;
    *input = rest;
    Ok(prefix)
}

fn word(bytes: &[u8]) -> i32 {
    i32::from_ne_bytes(bytes[..4].try_into().unwrap())
}

struct Table<'a> {
    offsets: &'a [u8],
    values: &'a [u8],
    stride: usize,
    capacity: usize,
    seat_axis: bool,
    // Logical column index to packed byte offset, for native joins.
    columns: Vec<usize>,
}

impl<'a> Table<'a> {
    fn read(
        input: &mut &'a [u8],
        key: &str,
        array: &ArrayLayout,
        spec: PackedTableLayout,
        batch_size: usize,
    ) -> PyResult<Self> {
        let PackedTableLayout {
            seat_axis,
            capacity,
            stride,
        } = spec;
        let groups = if seat_axis { 2 } else { 1 };
        let fields = array.packed_fields(key);
        let mut columns = vec![0; array.columns.len().max(1)];
        let mut width = 0;
        for field in &fields {
            columns[field.index] = width;
            width += field.storage.bytes();
        }
        let overflow = || PyValueError::new_err("observation batch dimensions overflow");
        let size = batch_size
            .checked_mul(groups)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(overflow)?;
        let offsets = take(input, size)?;
        require(word(offsets) == 0, "invalid first row offset")?;
        let mut end = 0;
        for bytes in offsets[4..].chunks_exact(4) {
            let next = word(bytes);
            require(
                next >= end && (next - end) as usize <= capacity,
                "invalid row offsets or capacity",
            )?;
            end = next;
        }
        let values = take(
            input,
            stride.checked_mul(end as usize).ok_or_else(overflow)?,
        )?;
        for row in values.chunks_exact(stride) {
            for f in &fields {
                let bytes = &row[columns[f.index]..];
                match f.storage {
                    Storage::Uint8 => require(
                        bytes[0] <= 1 && (f.name != "valid" || bytes[0] == 1),
                        "invalid packed flag",
                    )?,
                    Storage::Float32 => require(
                        f32::from_bits(word(bytes) as u32).is_finite(),
                        "nonfinite observation value",
                    )?,
                    Storage::Int32 => (),
                }
            }
        }
        Ok(Self {
            offsets,
            values,
            capacity,
            stride,
            seat_axis,
            columns,
        })
    }

    fn range(&self, group: usize) -> Range<usize> {
        word(&self.offsets[group * 4..]) as usize..word(&self.offsets[(group + 1) * 4..]) as usize
    }

    fn rows(&self, group: usize) -> impl ExactSizeIterator<Item = &[u8]> {
        let range = self.range(group);
        self.values[range.start * self.stride..range.end * self.stride].chunks_exact(self.stride)
    }

    fn integer(&self, row: &[u8], col: usize) -> i32 {
        word(&row[self.columns[col]..])
    }
}

pub(super) fn decode(py: Python<'_>, bytes: &[u8]) -> PyResult<ObservationBatch> {
    require(bytes.len() >= 32, "truncated observation batch")?;
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    require(
        Sha256::digest(body).as_slice() == digest,
        "corrupt or truncated observation batch",
    )?;
    let boundary = body
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| PyValueError::new_err("missing observation batch header"))?;
    let header: Header = serde_json::from_slice(&body[..boundary])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    require(
        header.format == FORMAT,
        "incompatible observation batch format or byte order",
    )?;
    require(
        header.config.layout == "OneToken" && header.config.turn_state,
        "observation batch requires policy input v11; v0.2.5 is the last release supporting v9-v10",
    )?;
    let layout = ObservationLayout::from_config(header.config.encoder)?;
    require(
        header.schema == schema(&layout),
        "incompatible packed observation layout",
    )?;
    let mut input = &body[boundary + 1..];
    let mut tables = BTreeMap::new();
    for (key, &spec) in &layout.packed {
        tables.insert(
            key.as_str(),
            Table::read(
                &mut input,
                key,
                &layout.arrays[key],
                spec,
                header.batch_size,
            )?,
        );
    }
    require(input.is_empty(), "trailing observation batch data")?;
    for game in 0..header.batch_size {
        validate_game(&tables, game, layout.config.policy_history)?;
    }
    let tables = tables
        .into_iter()
        .map(|(key, table)| {
            (
                key.to_owned(),
                FrozenTable {
                    offsets: PyBytes::new_bound(py, table.offsets).unbind(),
                    values: PyBytes::new_bound(py, table.values).unbind(),
                    stride: table.stride,
                    seat_axis: table.seat_axis,
                },
            )
        })
        .collect();
    Ok(ObservationBatch {
        layout: Py::new(py, layout)?,
        batch_size: header.batch_size,
        tables,
    })
}

type Tables<'a> = BTreeMap<&'a str, Table<'a>>;

fn validate_game(tables: &Tables, game: usize, policy_history: bool) -> PyResult<()> {
    let count = |key: &str| tables[key].range(game).len();
    require(
        count("agent_player") == 1
            && count("opponent_player") == 1
            && count("player_turn_state") == 2,
        "missing player rows",
    )?;
    for (seat, key) in ["agent_cards", "opponent_cards"].into_iter().enumerate() {
        for companion in ["semantic_cards", "card_turn_state"] {
            require(
                tables[companion].range(game * 2 + seat).len() == count(key),
                "card companion offsets differ",
            )?;
        }
    }
    require(
        count("actions") == count("action_focus") && count("actions") == count("name_candidates"),
        "action companion offsets differ",
    )?;
    let mut start = 0usize;
    let mut objects = Vec::new();
    for key in [
        "agent_player",
        "opponent_player",
        "agent_cards",
        "opponent_cards",
        "agent_permanents",
        "opponent_permanents",
        "stack",
    ] {
        let next = start.saturating_add(tables[key].capacity);
        require(next <= i32::MAX as usize, "object coordinates exceed int32")?;
        objects.push(start..start + count(key));
        start = next;
    }
    let object = |value: i32, base: usize| {
        let slot = i64::from(value) - base as i64;
        require(
            slot == -1 || (slot >= 0 && objects.iter().any(|r| r.contains(&(slot as usize)))),
            "object reference does not name a live row",
        )
    };
    for (key, table) in tables {
        for join in joins(key, policy_history) {
            match *join {
                Join::Objects => {
                    for row in table.rows(game) {
                        for bytes in row.chunks_exact(4) {
                            object(word(bytes), 0)?;
                        }
                    }
                }
                Join::Object(col, base) => {
                    for row in table.rows(game) {
                        object(table.integer(row, col), base)?;
                    }
                }
            }
        }
    }

    let relations = &tables["relations"];
    let mut frames = BTreeSet::new();
    for row in relations.rows(game) {
        let frame = relations.integer(row, relation::FRAME);
        let kind = relations.integer(row, relation::KIND);
        if matches!(kind, 1 | 11) {
            require(
                frame == frames.len() as i32 + 1,
                "invalid continuation frame header",
            )?;
            frames.insert(frame);
        }
        require(
            frames.contains(&frame) && frame <= RELATION_ROWS as i32,
            "invalid continuation frame reference",
        )?;
    }
    for (key, col) in [
        ("relations", relation::PARENT),
        ("actions", action::CONTINUATION),
    ] {
        let table = &tables[key];
        for row in table.rows(game) {
            let id = table.integer(row, col);
            require(
                id == 0 || frames.contains(&id),
                "missing continuation frame",
            )?;
        }
    }
    let names = &tables["name_candidates"];
    for row in names.rows(game) {
        let first = names.integer(row, name::GROUP_FIRST);
        let last = names.integer(row, name::GROUP_LAST);
        require(
            first >= 0 && last >= first && (first == 0) == (last == 0),
            "invalid name group span",
        )?;
    }
    let mut program = None;
    let facts = &tables["facts"];
    for row in facts.rows(game) {
        let code = facts.integer(row, fact::KIND);
        let kind = FactKind::ALL
            .iter()
            .find(|k| **k as i32 == code)
            .ok_or_else(|| PyValueError::new_err("invalid fact kind"))?;
        let id = facts.integer(row, fact::ID_HIGH);
        if matches!(
            kind,
            FactKind::EffectProgram | FactKind::ContinuationProgram
        ) {
            let ordinal = facts.integer(row, fact::VALUE_2);
            let owner = [
                code,
                id,
                facts.integer(row, fact::OBJECT),
                facts.integer(row, fact::DURATION),
                facts.integer(row, fact::TIMESTAMP),
                facts.integer(row, fact::SOURCE_DEFINITION),
            ];
            require(
                (1..=9).contains(&facts.integer(row, fact::VALUE))
                    && (ordinal == 0 || program == Some((owner, i64::from(ordinal)))),
                "invalid fact program run",
            )?;
            program = Some((owner, i64::from(ordinal) + 1));
        }
        if matches!(
            kind,
            FactKind::CombatRestriction
                | FactKind::ContinuationProgram
                | FactKind::TriggeringAmount
        ) || (*kind == FactKind::Transforms && id != 0)
        {
            require(frames.contains(&id), "invalid fact frame reference")?;
        }
    }
    if policy_history {
        let events = &tables["events"];
        for row in events.rows(game) {
            for (col, max) in [
                (history::KIND, 5),
                (history::SOURCE_OWNER, 2),
                (history::TARGET_OWNER, 2),
                (history::SOURCE_ZONE, 7),
                (history::TARGET_ZONE, 7),
                (history::SOURCE_STATUS, 2),
                (history::TARGET_STATUS, 2),
            ] {
                require(
                    (0..=max).contains(&events.integer(row, col)),
                    "invalid public history code",
                )?;
            }
        }
    }
    validate_effects(tables, game)?;
    if tables.contains_key("decision_records") {
        validate_memory(tables, game)?;
    }
    Ok(())
}

fn span(start: i32, size: i32, cursor: &mut usize, limit: usize) -> PyResult<Range<usize>> {
    require(
        start >= 0 && start as usize == *cursor && size >= 0 && size as usize <= limit - *cursor,
        "invalid companion row partition",
    )?;
    *cursor += size as usize;
    Ok(start as usize..*cursor)
}

fn validate_effects(tables: &Tables, game: usize) -> PyResult<()> {
    let effects = &tables["effects"];
    let details = &tables["effect_details"];
    let members = &tables["effect_members"];
    let mut d = 0;
    let mut m = 0;
    for row in effects.rows(game) {
        let get = |col| effects.integer(row, col);
        require(
            get(effect::CONTROLLER) > 0 && get(effect::MEMBER_COUNT) > 0,
            "missing effect controller or member",
        )?;
        span(
            get(effect::DETAIL_OFFSET),
            get(effect::DETAIL_COUNT),
            &mut d,
            details.range(game).len(),
        )?;
        span(
            get(effect::MEMBER_OFFSET),
            get(effect::MEMBER_COUNT),
            &mut m,
            members.range(game).len(),
        )?;
    }
    require(
        d == details.range(game).len() && m == members.range(game).len(),
        "unowned effect rows",
    )?;
    for row in details.rows(game) {
        require(
            matches!(details.integer(row, detail::KIND), 1..=3 | 11..=19),
            "invalid effect detail kind",
        )?;
    }
    let mut orders = BTreeSet::new();
    for row in members.rows(game) {
        let order = members.integer(row, member::ORDER);
        require(
            members.integer(row, member::OBJECT) > 0
                && (1..=MEMBER_ROWS as i32).contains(&order)
                && orders.insert(order),
            "invalid effect membership",
        )?;
    }
    Ok(())
}

fn validate_memory(tables: &Tables, game: usize) -> PyResult<()> {
    let records = &tables["decision_records"];
    let members = &tables["decision_members"];
    let refs = &tables["decision_references"];
    let details = &tables["decision_reference_details"];
    let mut joins = members.rows(game);
    let mut seen_refs = 0usize;
    let mut m = 0;
    let mut position = 0u64;
    for (index, row) in records.rows(game).enumerate() {
        let get = |col| records.integer(row, col);
        let unsigned = |col| u64::from(get(col) as u32);
        let count = unsigned(decision_record::COUNT_HIGH);
        require(
            unsigned(decision_record::ORDINAL_HIGH) == index as u64
                && count > 0
                && unsigned(decision_record::FIRST_HIGH) == position
                && unsigned(decision_record::LAST_HIGH) + 1 == position + count,
            "invalid decision record ordinal or span",
        )?;
        position += count;
        require(
            (1..=2).contains(&get(decision_record::CHOOSER_ROLE))
                && (0..=4).contains(&get(decision_record::PHASE))
                && (0..=11).contains(&get(decision_record::STEP)),
            "invalid decision context",
        )?;
        require(
            get(decision_record::MEMBER_OFFSET) == m as i32,
            "invalid decision member offset",
        )?;
        for col in [
            decision_record::MANA_COUNT,
            decision_record::RESPONSE_COUNT,
            decision_record::CHOICE_COUNT,
        ] {
            let n = get(col);
            require(
                n >= 0 && n as usize <= joins.len(),
                "invalid decision member count",
            )?;
            for ordinal in 0..n {
                let join = joins.next().unwrap();
                let id = members.integer(join, decision_member::REFERENCE);
                require(
                    id >= 1
                        && id as usize <= refs.range(game).len()
                        && id as usize <= seen_refs + 1
                        && members.integer(join, decision_member::ORDINAL) == ordinal,
                    "invalid decision member reference or ordinal",
                )?;
                seen_refs = seen_refs.max(id as usize);
                m += 1;
            }
        }
    }
    require(
        joins.len() == 0 && seen_refs == refs.range(game).len(),
        "unowned decision members or references",
    )?;
    let mut d = 0;
    let mut children = details.rows(game);
    for row in refs.rows(game) {
        let range = span(
            refs.integer(row, decision_reference::DETAIL_OFFSET),
            refs.integer(row, decision_reference::DETAIL_COUNT),
            &mut d,
            details.range(game).len(),
        )?;
        let mut program = 0;
        let mut mana = 0;
        for _ in range {
            let child = children.next().unwrap();
            let ordinal = details.integer(child, decision_reference_detail::ORDINAL);
            let token = details.integer(child, decision_reference_detail::TOKEN_KIND);
            match details.integer(child, decision_reference_detail::FIELD) {
                0 if mana == 0 && ordinal == program && (1..=9).contains(&token) => program += 1,
                1 if ordinal == mana && token == -1 => mana += 1,
                _ => return Err(PyValueError::new_err("invalid decision detail coordinate")),
            }
        }
        require(mana % 6 == 0, "incomplete decision mana production")?;
    }
    require(children.len() == 0, "unowned decision details")
}
