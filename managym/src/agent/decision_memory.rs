//! Viewer-owned decision prefixes. Capture precedes execution; only a successful
//! external commit appends. Fixed-size immutable chunks share fork prefixes and
//! keep append/coalescing work bounded independently of the game's length.

use std::{collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind},
    flow::game::Game,
    state::{
        game_object::{ObjectId, PlayerId},
        zone::ZoneType,
    },
};

const CHUNK: usize = 32;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecisionLog {
    // Transport lineage only: Arc equality compares (), not allocation identity.
    // Serialization remains the record sequence; decoding starts a fresh lineage.
    origin: Arc<()>,
    tail: Option<Arc<Chunk>>,
    len: usize,
    snapshots: Option<Arc<SnapshotChunk>>,
    snapshot_count: usize,
    detail_count: usize,
    member_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Chunk {
    previous: Option<Arc<Chunk>>,
    rows: Vec<DecisionRecord>,
}

/// Snapshot chunks are immutable prefix ownership, shared by forked logs.
/// Their hash buckets are lookup hints; complete equality decides reuse.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SnapshotChunk {
    previous: Option<Arc<SnapshotChunk>>,
    rows: Vec<Arc<DecisionReference>>,
    buckets: BTreeMap<[u8; 32], Vec<usize>>,
}

fn reference_hash(reference: &DecisionReference) -> [u8; 32] {
    // Canonical serialization excludes private cache data and retains every
    // viewer-entitled field, ordered mana alternative and program coordinate.
    *blake3::hash(&serde_json::to_vec(reference).expect("reference serializes")).as_bytes()
}

impl DecisionLog {
    /// A new append-only timeline with the same immutable record and reference chunks.
    pub(crate) fn fork_lineage(&self) -> Self {
        Self {
            origin: Arc::new(()),
            ..self.clone()
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    /// Transport revision: every committed decision, including a coalesced pass.
    pub fn tail_version(&self) -> u64 {
        self.tail.as_ref().map_or(0, |tail| {
            u64::from(tail.rows.last().unwrap().last_position) + 1
        })
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Projected table sizes maintained at commit, without prefix traversal.
    pub fn detail_count(&self) -> usize {
        self.detail_count
    }
    pub fn reference_count(&self) -> usize {
        self.snapshot_count
    }
    pub fn member_count(&self) -> usize {
        self.member_count
    }

    pub fn iter(&self) -> impl Iterator<Item = &DecisionRecord> {
        self.records_since(0)
    }

    fn records_since(&self, start: usize) -> impl Iterator<Item = &DecisionRecord> {
        let mut chunks = Vec::new();
        let mut next = self.tail.as_deref();
        let mut end = self.len;
        while let Some(chunk) = next {
            let begin = end - chunk.rows.len();
            if end <= start {
                break;
            }
            chunks.push(&chunk.rows[start.saturating_sub(begin)..]);
            end = begin;
            next = chunk.previous.as_deref();
        }
        chunks.into_iter().rev().flatten()
    }

    fn snapshots(&self) -> impl Iterator<Item = &Arc<DecisionReference>> {
        self.snapshots_since(0)
    }

    fn snapshots_since(&self, start: usize) -> impl Iterator<Item = &Arc<DecisionReference>> {
        let mut chunks = Vec::new();
        let mut next = self.snapshots.as_deref();
        let mut end = self.snapshot_count;
        while let Some(chunk) = next {
            let begin = end - chunk.rows.len();
            if end <= start {
                break;
            }
            chunks.push(&chunk.rows[start.saturating_sub(begin)..]);
            end = begin;
            next = chunk.previous.as_deref();
        }
        chunks.into_iter().rev().flatten()
    }

    fn intern(&mut self, reference: Arc<DecisionReference>) -> Arc<DecisionReference> {
        if self
            .snapshots()
            .any(|existing| Arc::ptr_eq(existing, &reference))
        {
            return reference;
        }
        let key = reference_hash(&reference);
        let mut next = self.snapshots.as_deref();
        while let Some(chunk) = next {
            if let Some(indexes) = chunk.buckets.get(&key) {
                for &index in indexes {
                    if chunk.rows[index] == reference {
                        return chunk.rows[index].clone();
                    }
                }
            }
            next = chunk.previous.as_deref();
        }
        if self
            .snapshots
            .as_ref()
            .is_none_or(|chunk| chunk.rows.len() == CHUNK)
        {
            self.snapshots = Some(Arc::new(SnapshotChunk {
                previous: self.snapshots.take(),
                rows: Vec::new(),
                buckets: BTreeMap::new(),
            }));
        }
        let chunk = Arc::make_mut(self.snapshots.as_mut().unwrap());
        chunk.buckets.entry(key).or_default().push(chunk.rows.len());
        chunk.rows.push(reference.clone());
        self.snapshot_count += 1;
        self.detail_count += (if definition_only(&reference) {
            0
        } else {
            reference.program.len()
        }) + reference.mana.len() * 6;
        reference
    }

    fn intern_record(&mut self, row: &mut DecisionRecord) {
        for reference in row
            .context
            .mana_sources
            .iter_mut()
            .chain(&mut row.response)
            .chain(&mut row.choice.references)
        {
            *reference = self.intern(reference.clone());
        }
    }

    fn push(&mut self, mut row: DecisionRecord) {
        self.intern_record(&mut row);
        if let Some(tail) = self.tail.as_mut() {
            let last = tail.rows.last().expect("nonempty chunk");
            let position = last
                .last_position
                .checked_add(1)
                .expect("decision position exhausted");
            row.first_position = position;
            row.last_position = position;
            row.ordinal = self.len as u32;
            if row.actor == last.actor
                && row.choice.kind.as_deref() == Some("PassPriority")
                && last.choice == row.choice
                && last.context == row.context
                && last.response == row.response
                && last.kind == row.kind
            {
                let last = Arc::make_mut(tail).rows.last_mut().unwrap();
                last.count = last.count.checked_add(1).expect("pass count exhausted");
                last.last_position = position;
                return;
            }
        }
        if self
            .tail
            .as_ref()
            .is_none_or(|tail| tail.rows.len() == CHUNK)
        {
            self.tail = Some(Arc::new(Chunk {
                previous: self.tail.take(),
                rows: Vec::new(),
            }));
        }
        self.member_count +=
            row.context.mana_sources.len() + row.response.len() + row.choice.references.len();
        Arc::make_mut(self.tail.as_mut().unwrap()).rows.push(row);
        self.len += 1;
    }
}

impl Serialize for DecisionLog {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de> Deserialize<'de> for DecisionLog {
    fn deserialize<D: Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        let rows = Vec::<DecisionRecord>::deserialize(decoder)?;
        let mut log = Self::default();
        // Retained projections preserve their exact spans and multiplicities.
        for mut row in rows {
            log.intern_record(&mut row);
            if log
                .tail
                .as_ref()
                .is_none_or(|tail| tail.rows.len() == CHUNK)
            {
                log.tail = Some(Arc::new(Chunk {
                    previous: log.tail.take(),
                    rows: Vec::new(),
                }));
            }
            log.member_count +=
                row.context.mana_sources.len() + row.response.len() + row.choice.references.len();
            Arc::make_mut(log.tail.as_mut().unwrap()).rows.push(row);
            log.len += 1;
        }
        Ok(log)
    }
}

/// A reference owns the characteristics visible at emission, never a future
/// lookup. Definition and appearance are provenance, never numeric features.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionReference {
    pub role: u8,
    pub definition: Option<u32>,
    pub appearance: Option<u32>,
    pub name: Option<String>,
    /// Human-readable native rules snapshot, derived alongside the learning program.
    pub description: Arc<str>,
    pub power: i32,
    pub toughness: i32,
    pub tapped: bool,
    pub ready: bool,
    pub attacking: bool,
    pub blocking: bool,
    /// Complete effective rules characteristics as existing learning tokens.
    pub program: Arc<Vec<(u16, i32)>>,
    /// Public production alternatives, W/U/B/R/G/C; no hidden-spell tests.
    pub mana: Vec<[u32; 6]>,
    /// Exact emission characteristics for reusing immutable rule payloads.
    /// Parsed evidence needs no cache and never resumes native execution.
    #[serde(skip)]
    source: Option<Arc<crate::state::card::PrintedCharacteristics>>,
}

impl PartialEq for DecisionReference {
    fn eq(&self, other: &Self) -> bool {
        self.role == other.role
            && self.definition == other.definition
            && self.appearance == other.appearance
            && self.name == other.name
            && self.description == other.description
            && self.power == other.power
            && self.toughness == other.toughness
            && self.tapped == other.tapped
            && self.ready == other.ready
            && self.attacking == other.attacking
            && self.blocking == other.blocking
            && self.program == other.program
            && self.mana == other.mana
    }
}
impl Eq for DecisionReference {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionContext {
    pub turn: u32,
    pub phase: u8,
    pub step: u8,
    pub life: [i32; 2],
    pub hand: [u32; 2],
    /// Total, tapped, land, creature, artifact, enchantment, planeswalker, battle.
    pub board_counts: [[u32; 8]; 2],
    pub mana_sources: Vec<Arc<DecisionReference>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicChoice {
    /// None means unavailable, permanently, even if later revealed.
    pub kind: Option<String>,
    pub mode: Option<u32>,
    pub ability: Option<u32>,
    pub named: Option<String>,
    pub name_row: Option<u32>,
    pub declared: Option<bool>,
    pub references: Vec<Arc<DecisionReference>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRecord {
    pub actor: u8,
    pub ordinal: u32,
    pub first_position: u32,
    pub last_position: u32,
    pub count: u32,
    pub kind: String,
    pub context: DecisionContext,
    pub response: Vec<Arc<DecisionReference>>,
    pub choice: PublicChoice,
}

fn snapshot(
    game: &Game,
    object: ObjectId,
    viewer: PlayerId,
    committed_public: bool,
) -> DecisionReference {
    let visible = game.state.live(object).is_some()
        && (committed_public || game.visible_reference(object, viewer).object.is_some());
    if !visible {
        return DecisionReference {
            role: 0,
            definition: None,
            appearance: None,
            name: None,
            description: Arc::from(""),
            power: 0,
            toughness: 0,
            tapped: false,
            ready: false,
            attacking: false,
            blocking: false,
            program: Arc::default(),
            mana: Vec::new(),
            source: None,
        };
    }
    let data = &game.state[object];
    let mut printed = game.state.source_base(object).1.clone();
    let mut mana = Vec::new();
    let mut role = if data.owner == viewer { 1 } else { 2 };
    let mut tapped = false;
    let mut ready = false;
    let mut attacking = false;
    let mut blocking = false;
    if let Some(id) = game.state.permanent_of(object) {
        let permanent = game.state.permanents[id].as_ref().unwrap();
        let view = game.characteristics(id);
        role = if permanent.controller == viewer { 1 } else { 2 };
        tapped = permanent.tapped;
        ready = game.permanent_can_tap(id);
        attacking = permanent.attacking;
        blocking = game.state.combat.as_ref().is_some_and(|combat| {
            combat
                .attacker_to_blockers
                .values()
                .any(|blockers| blockers.contains(&id))
        });
        printed.name = view.name.clone();
        printed.power = Some(view.power);
        printed.toughness = Some(view.toughness);
        printed.types = view.types.clone();
        printed.subtypes = view.subtypes.clone();
        printed.color_override = Some(view.colors.clone());
        printed.keywords = view.keywords;
        printed.mana_abilities = view.mana_abilities.clone();
        printed.triggered_mana_abilities = view.triggered_mana_abilities.clone();
        printed.statics = view.static_abilities.clone();
        printed.abilities = view
            .triggered_abilities
            .iter()
            .map(|trigger| trigger.ability.clone())
            .collect();
        printed.activated_abilities = view.activated_abilities.clone();
        if !tapped {
            for ability in &view.mana_abilities {
                mana.push(ability.mana.mana.map(u32::from));
            }
            for ability in &view.activated_abilities {
                if let Some(choices) = ability.mana_choices() {
                    mana.extend(choices.iter().map(|choice| choice.mana.map(u32::from)));
                }
            }
        }
    }
    // A source definition, object id or later reveal cannot key this cache.
    // Complete effective characteristics decide reuse; issued prefixes own the
    // immutable payload, and only their reachable snapshots are searched.
    let existing = game.decision_memory[viewer.0]
        .snapshots()
        .find(|reference| reference.source.as_deref() == Some(&printed));
    let (program, description) = if let Some(reference) = existing {
        (reference.program.clone(), reference.description.clone())
    } else {
        let snapshot = crate::state::card::ObjectDefinition::face(printed.clone());
        let program = crate::cards::tokens::definition_tokens(
            &snapshot,
            crate::cards::content_pack::default_content_pack(),
        )
        .expect("registered public characteristics")
        .into_iter()
        .map(|(kind, value)| (kind as u16, value))
        .collect();
        (
            Arc::new(program),
            Arc::<str>::from(serde_json::to_string(&snapshot).expect("snapshot serializes")),
        )
    };
    let (power, toughness) = game
        .state
        .permanent_of(object)
        .map(|id| {
            let view = game.characteristics(id);
            (view.power, view.toughness)
        })
        .unwrap_or((0, 0));
    DecisionReference {
        role,
        definition: data.definition_id.catalog_id().map(|id| id.0),
        appearance: Some(game.observed_object_id(object).0),
        name: Some(printed.name.clone()),
        description,
        power,
        toughness,
        tapped,
        ready,
        attacking,
        blocking,
        program,
        mana,
        source: Some(Arc::new(printed)),
    }
}

fn reference(
    game: &Game,
    object: ObjectId,
    viewer: PlayerId,
    committed_public: bool,
) -> Arc<DecisionReference> {
    let value = snapshot(game, object, viewer, committed_public);
    game.decision_memory[viewer.0]
        .snapshots()
        .find(|existing| existing.as_ref() == &value)
        .cloned()
        .unwrap_or_else(|| Arc::new(value))
}

fn target_reference(
    game: &Game,
    target: crate::state::game_object::Target,
    viewer: PlayerId,
) -> Arc<DecisionReference> {
    use crate::state::game_object::Target;
    let object = match target {
        Target::Player(player) => {
            return Arc::new(DecisionReference {
                role: if player == viewer { 1 } else { 2 },
                ..Default::default()
            })
        }
        Target::Permanent(id) => game.permanent_object(id),
        Target::StackSpell(object) | Target::GraveyardCard(object) => Some(object),
        Target::StackAbility(id) => game.stack_ability_source(id),
    };
    object
        .map(|object| reference(game, object, viewer, false))
        .unwrap_or_default()
}

pub const RECORD_DIM: usize = super::observation_layout::decision_record::WIDTH;
pub const DETAIL_DIM: usize = super::observation_layout::decision_reference_detail::WIDTH;
pub const REFERENCE_DIM: usize = super::observation_layout::decision_reference::WIDTH;
pub const MEMBER_DIM: usize = super::observation_layout::decision_member::WIDTH;
/// Names introduced by this projection, registered in the shared catalog.
pub const RECORD_NAMES: &[&str] = &[
    "Priority",
    "DeclareAttacker",
    "DeclareBlocker",
    "ChooseTarget",
    "Scry",
    "LookAndSelect",
    "PayOrNot",
    "Modal",
    "Learn",
    "Waterbend",
    "Discard",
    "LegendRule",
    "OrderTriggers",
    "PassPriority",
    "PriorityPlayLand",
    "PriorityCastSpell",
    "PriorityActivateAbility",
    "PayCost",
    "DeclineChoice",
    "ChooseMode",
    "TapForCost",
    "LearnTakeLesson",
    "LearnDiscard",
    "SelectCard",
    "Life",
    "Hand",
    "Role",
    "Power",
    "Toughness",
    "Tapped",
    "Attacking",
    "Blocking",
    "Program",
    "Mana",
    "Choice",
    "Mode",
    "Declared",
    "Definition",
    "Ability",
    "Named",
    "BoardCounts",
    "Ready",
    "Foretell",
    "TurnFaceUp",
];

use super::{
    observation_encoder::ObservationEncodeError,
    observation_rows::{
        write_dense_row,
        Cell::{self, Integer, Word},
        RowWriter, Table,
    },
};

/// The catalog is immutable shared rules data, not a viewer or object cache.
/// Both the complete canonical description and complete token program must
/// agree, so token/back-face metadata and effective modifications cannot alias.
fn definition_only(reference: &DecisionReference) -> bool {
    use std::sync::OnceLock;
    type Programs = std::collections::HashMap<u32, (String, Vec<(u16, i32)>)>;
    static PROGRAMS: OnceLock<Programs> = OnceLock::new();
    let programs = PROGRAMS.get_or_init(|| {
        let pack = crate::cards::content_pack::default_content_pack();
        pack.definition_entries()
            .map(|(id, definition)| {
                let tokens = crate::cards::tokens::definition_tokens(definition, pack)
                    .expect("registered definition")
                    .into_iter()
                    .map(|(kind, value)| (kind as u16, value))
                    .collect();
                (
                    id.0,
                    (
                        serde_json::to_string(definition).expect("definition serializes"),
                        tokens,
                    ),
                )
            })
            .collect()
    });
    reference
        .definition
        .and_then(|id| programs.get(&id))
        .is_some_and(|(description, program)| {
            description.as_str() == reference.description.as_ref()
                && program.as_slice() == reference.program.as_slice()
        })
}

/// One viewer prefix, with disjoint record members and shared exact snapshots.
#[derive(Default)]
pub struct EncodedMemory {
    pub reset: bool,
    pub records: Vec<[f32; RECORD_DIM]>,
    pub references: Vec<[f32; REFERENCE_DIM]>,
    pub details: Vec<[f32; DETAIL_DIM]>,
    pub members: Vec<[f32; MEMBER_DIM]>,
}

impl RowWriter for EncodedMemory {
    fn write(
        &mut self,
        table: Table,
        _seat: usize,
        _slot: usize,
        row: &[Cell],
    ) -> Result<(), ObservationEncodeError> {
        fn append<const N: usize>(target: &mut Vec<[f32; N]>, row: &[Cell]) {
            let mut dense = [0.0; N];
            write_dense_row(row, &mut dense);
            target.push(dense);
        }
        match table {
            Table::DecisionRecords => append(&mut self.records, row),
            Table::DecisionReferences => append(&mut self.references, row),
            Table::DecisionReferenceDetails => append(&mut self.details, row),
            Table::DecisionMembers => append(&mut self.members, row),
            _ => unreachable!("not a decision-memory table"),
        }
        Ok(())
    }

    fn reset_memory(&mut self) {
        self.reset = true;
    }
}

/// Caller-owned native transport state, never serialized with game facts.
#[derive(Default)]
pub struct MemoryEncoding {
    origin: Option<Arc<()>>,
    counts: [usize; 4],
    version: u64,
    indexes: std::collections::HashMap<usize, usize>,
    // Keep pointer-indexed snapshots alive until this buffer changes games.
    log: Option<DecisionLog>,
}

pub fn encode(log: &DecisionLog) -> EncodedMemory {
    encode_suffix(log, &mut MemoryEncoding::default()).1
}

pub fn encode_suffix(
    log: &DecisionLog,
    cursor: &mut MemoryEncoding,
) -> ([usize; 4], EncodedMemory) {
    let mut out = EncodedMemory::default();
    let starts =
        write_suffix(log, cursor, &mut out).expect("dense memory projection is infallible");
    (starts, out)
}

pub(super) fn write_suffix(
    log: &DecisionLog,
    cursor: &mut MemoryEncoding,
    out: &mut impl RowWriter,
) -> Result<[usize; 4], ObservationEncodeError> {
    let reset = cursor
        .origin
        .as_ref()
        .is_none_or(|origin| !Arc::ptr_eq(origin, &log.origin));
    if reset {
        *cursor = MemoryEncoding::default();
        cursor.origin = Some(log.origin.clone());
        out.reset_memory();
    }
    let mut starts = cursor.counts;
    if starts[0] > 0 && cursor.version != log.tail_version() {
        starts[0] -= 1;
        let last = cursor
            .log
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .rows
            .last()
            .unwrap();
        starts[3] -=
            last.context.mana_sources.len() + last.response.len() + last.choice.references.len();
    }
    write_from(log, starts, &mut cursor.indexes, out)?;
    cursor.counts = [
        log.len(),
        log.detail_count(),
        log.reference_count(),
        log.member_count(),
    ];
    cursor.version = log.tail_version();
    cursor.log = Some(log.clone());
    Ok(starts)
}

fn write_from(
    log: &DecisionLog,
    starts: [usize; 4],
    indexes: &mut std::collections::HashMap<usize, usize>,
    out: &mut impl RowWriter,
) -> Result<(), ObservationEncodeError> {
    use crate::cards::tokens::name_value;
    let [record_start, mut detail_slot, reference_start, mut member_slot] = starts;
    for (offset, reference) in log.snapshots_since(reference_start).enumerate() {
        let index = reference_start + offset;
        indexes.insert(Arc::as_ptr(reference) as usize, index + 1);
        let mut header = [Cell::ZERO; REFERENCE_DIM];
        header[0] = Integer(reference.role as i64);
        header[1] = Integer(reference.definition.map_or(0, |id| i64::from(id) + 1));
        header[2] = Word(reference.power);
        header[4] = Word(reference.toughness);
        header[6] = Integer(i64::from(reference.tapped));
        header[7] = Integer(i64::from(reference.ready));
        header[8] = Integer(i64::from(reference.attacking));
        header[9] = Integer(i64::from(reference.blocking));
        let detail_start = detail_slot;
        header[10] = Integer(detail_start as i64);
        header[13] = Integer(i64::from(definition_only(reference)));
        for (ordinal, &(kind, value)) in reference
            .program
            .iter()
            .enumerate()
            .filter(|_| header[13] == Cell::ZERO)
        {
            out.write(
                Table::DecisionReferenceDetails,
                0,
                detail_slot,
                &[
                    Cell::ZERO,
                    Integer(kind as i64),
                    Word(value),
                    Cell::ZERO,
                    Integer(ordinal as i64),
                    Integer(1),
                ],
            )?;
            detail_slot += 1;
        }
        for (ordinal, production) in reference.mana.iter().enumerate() {
            for (color, &amount) in production.iter().enumerate() {
                out.write(
                    Table::DecisionReferenceDetails,
                    0,
                    detail_slot,
                    &[
                        Integer(1),
                        Integer(-1),
                        Word(amount as i32),
                        Cell::ZERO,
                        Integer((ordinal * 6 + color) as i64),
                        Integer(1),
                    ],
                )?;
                detail_slot += 1;
            }
        }
        header[11] = Integer((detail_slot - detail_start) as i64);
        header[12] = Integer(1);
        out.write(Table::DecisionReferences, 0, index, &header)?;
    }
    for (index, record) in log.records_since(record_start).enumerate() {
        let mut header = [Cell::ZERO; RECORD_DIM];
        for (offset, value) in [
            (0, record.ordinal),
            (2, record.count),
            (4, record.context.turn),
            (6, record.first_position),
            (8, record.last_position),
        ] {
            header[offset] = Word(value as i32);
        }
        header[10] = Integer(record.context.phase as i64);
        header[11] = Integer(record.context.step as i64);
        header[12] = Word(name_value(&format!("DecisionMemory::{}", record.kind)));
        header[14] = Integer(member_slot as i64);
        for (group, references) in [
            &record.context.mana_sources,
            &record.response,
            &record.choice.references,
        ]
        .into_iter()
        .enumerate()
        {
            header[15 + group] = Integer(references.len() as i64);
            for (ordinal, reference) in references.iter().enumerate() {
                out.write(
                    Table::DecisionMembers,
                    0,
                    member_slot,
                    &[
                        Integer(indexes[&(Arc::as_ptr(reference) as usize)] as i64),
                        Integer(ordinal as i64),
                    ],
                )?;
                member_slot += 1;
            }
        }
        header[18] = Integer(record.actor as i64);
        header[19] = Integer(1);
        let context = record
            .context
            .life
            .into_iter()
            .chain(record.context.hand.map(|n| n as i32))
            .chain(
                record
                    .context
                    .board_counts
                    .into_iter()
                    .flatten()
                    .map(|n| n as i32),
            );
        for (index, value) in context.enumerate() {
            header[20 + 2 * index] = Word(value);
        }
        for (index, value) in [
            record
                .choice
                .kind
                .as_ref()
                .map(|kind| name_value(&format!("DecisionMemory::{kind}")) as u32),
            record.choice.mode,
            record.choice.ability,
            record.choice.name_row.map(|n| n + 1),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(value) = value {
                let offset = 60 + index * 3;
                header[offset] = Integer(1);
                header[offset + 1] = Word(value as i32);
            }
        }
        if let Some(value) = record.choice.declared {
            header[72] = Integer(1);
            header[73] = Integer(i64::from(value));
        }
        out.write(Table::DecisionRecords, 0, record_start + index, &header)?;
    }
    debug_assert_eq!(detail_slot, log.detail_count());
    debug_assert_eq!(member_slot, log.member_count());
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PendingDecision {
    viewer: usize,
    record: DecisionRecord,
    own: Option<DecisionRecord>,
}

/// Rebase only the opposing viewer's projection, never recapture private actor
/// data. Their private knowledge also cannot be declassified to the actor:
/// non-announced references must be entitled to both viewers at emission.
fn own_projection(
    game: &Game,
    record: &DecisionRecord,
    actor: PlayerId,
    announces: bool,
) -> DecisionRecord {
    let mut own = record.clone();
    own.actor = 1;
    own.context.life.reverse();
    own.context.hand.reverse();
    own.context.board_counts.reverse();
    for (group, references) in [
        &mut own.context.mana_sources,
        &mut own.response,
        &mut own.choice.references,
    ]
    .into_iter()
    .enumerate()
    {
        for reference in references {
            let entitled = reference.definition.is_none()
                || (group == 2 && announces)
                || reference.appearance.is_some_and(|appearance| {
                    game.state
                        .latest_objects()
                        .find(|(object, _)| game.observed_object_id(*object).0 == appearance)
                        .is_some_and(|(object, _)| {
                            game.visible_reference(object, actor).definition.is_some()
                        })
                });
            if !entitled {
                *reference = Arc::new(DecisionReference::default());
            } else {
                let reference = Arc::make_mut(reference);
                reference.role = match reference.role {
                    1 => 2,
                    2 => 1,
                    _ => 0,
                };
            }
        }
    }
    own
}

/// Pending records contain no post-choice/private data. Callers discard them
/// on rejection; publication appends only after successful execution.
pub(crate) fn capture(
    game: &Game,
    space: &ActionSpace,
    action: &Action,
) -> Option<PendingDecision> {
    if !game.decision_memory_enabled || space.actions.len() <= 1 {
        return None;
    }
    let chooser = space.player?;
    let viewer = PlayerId(1 - chooser.0);
    let players = [viewer, chooser];
    let mut board_counts = [[0; 8]; 2];
    let mut board = Vec::new();
    let mut attacking = Vec::new();
    for (seat, &player) in players.iter().enumerate() {
        for id in game.battlefield_permanents(player) {
            let permanent = game.state.permanents[id].as_ref().unwrap();
            let view = game.characteristics(id);
            let flags = [
                true,
                permanent.tapped,
                view.is_land(),
                view.is_creature(),
                view.types.is_artifact(),
                view.types.is_enchantment(),
                view.types.is_planeswalker(),
                view.types.is_battle(),
            ];
            for (count, flag) in board_counts[seat].iter_mut().zip(flags) {
                *count += u32::from(flag);
            }
            let object = game.permanent_object(id).expect("battlefield object");
            if !permanent.tapped
                && (!view.mana_abilities.is_empty()
                    || view
                        .activated_abilities
                        .iter()
                        .any(|ability| ability.mana_choices().is_some()))
            {
                board.push(reference(game, object, viewer, false));
            }
            if permanent.attacking {
                attacking.push(reference(game, object, viewer, false));
            }
        }
    }
    let hand = players.map(|player| game.state.zones.size(ZoneType::Hand, player) as u32);
    let resource_seat = usize::from(action.player() != viewer);
    if matches!(action, Action::PassPriority { .. })
        && (space.kind != ActionSpaceKind::Priority
            || hand[resource_seat] == 0
            || !board.iter().any(|r| {
                r.role as usize == resource_seat + 1
                    && r.ready
                    && r.mana
                        .iter()
                        .any(|mana| mana.iter().any(|&amount| amount > 0))
            }))
    {
        return None;
    }
    let mut choice = PublicChoice {
        kind: None,
        mode: None,
        ability: None,
        named: None,
        name_row: None,
        declared: None,
        references: Vec::new(),
    };
    // Exhaustive privacy classification. Private choices share one unavailable
    // projection; action indexes and private alternatives are never retained.
    let public = match action {
        Action::PlayLand { object, .. }
        | Action::CastSpell { object, .. }
        | Action::ActivateCardAbility { object, .. } => {
            choice
                .references
                .push(reference(game, *object, viewer, true));
            true
        }
        Action::PassPriority { .. } => true,
        Action::DeclareAttacker {
            permanent, attack, ..
        } => {
            choice.declared = Some(attack.is_some());
            if let Some(crate::flow::combat::AttackTarget::Permanent(defender)) = attack {
                let attacker = game
                    .permanent_object(*permanent)
                    .expect("declared attacker");
                choice.references.extend(
                    [attacker, *defender].map(|object| reference(game, object, viewer, true)),
                );
            }
            true
        }
        Action::DeclareBlocker { attacker, .. } => {
            choice.declared = Some(attacker.is_some());
            true
        }
        Action::ChooseTarget { target, .. } => {
            choice
                .references
                .push(target_reference(game, *target, viewer));
            true
        }
        Action::ActivateAbility { ability_index, .. } => {
            choice.ability = Some(*ability_index as u32);
            true
        }
        Action::WaterbendTap { .. } | Action::TapForMana { .. } => true,
        Action::ChooseMode { mode, .. } => {
            use crate::flow::decision::Decision;
            match game
                .state
                .suspended_decision
                .as_ref()
                .map(|pending| &pending.decision)
            {
                Some(Decision::NameCard { names, scope, .. }) => {
                    if let Some(crate::cards::names::NameOption::Name(row)) = scope.option(*mode) {
                        choice.name_row = Some(row as u32);
                        choice.named = names.get(row).map(|candidate| candidate.name.clone());
                        true
                    } else {
                        false
                    }
                }
                Some(
                    Decision::Modal { .. }
                    | Decision::ManaChoice { .. }
                    | Decision::ManaSpend { .. }
                    | Decision::ChoosePaymentX { .. },
                ) => {
                    choice.mode = Some(*mode as u32);
                    true
                }
                _ => false,
            }
        }
        Action::PayCost { .. } => true,
        Action::Decline { .. } => matches!(
            space.kind,
            ActionSpaceKind::PayOrNot | ActionSpaceKind::Learn | ActionSpaceKind::Waterbend
        ),
        Action::LearnTakeLesson { object, .. } | Action::LearnDiscard { object, .. } => {
            choice
                .references
                .push(reference(game, *object, viewer, true));
            true
        }
        Action::SelectCard { object, .. }
            if matches!(
                space.kind,
                ActionSpaceKind::Discard | ActionSpaceKind::LegendRule
            ) =>
        {
            choice
                .references
                .push(reference(game, *object, viewer, true));
            true
        }
        Action::TurnFaceUp { object, .. } => {
            choice
                .references
                .push(reference(game, *object, viewer, false));
            true
        }
        Action::Foretell { .. } => {
            choice
                .references
                .push(Arc::new(DecisionReference::default()));
            true
        }
        Action::ChooseTrigger { .. } | Action::ScryCard { .. } | Action::SelectCard { .. } => false,
    };
    if let Action::ActivateCardAbility { ability_index, .. } = action {
        choice.ability = Some(*ability_index as u32);
    }
    if public {
        choice.kind = Some(match action {
            Action::PassPriority { .. } => "PassPriority".into(),
            Action::TurnFaceUp { .. } => "TurnFaceUp".into(),
            Action::Foretell { .. } => "Foretell".into(),
            _ => format!("{:?}", action.action_type()),
        });
        if choice.references.is_empty() {
            choice.references = crate::agent::observation::Observation::action_focus(game, action)
                .iter()
                .filter_map(|id| {
                    game.state
                        .latest_objects()
                        .find(|(_, data)| data.id == *id)
                        .map(|(object, _)| reference(game, object, viewer, false))
                })
                .collect();
        }
    }
    let mut response = Vec::new();
    for stack in game.state.stack_objects.iter().rev() {
        let source = stack
            .source_ref()
            .map(|object| reference(game, object, viewer, false))
            .unwrap_or_default();
        response.push(source);
        response.extend(
            stack
                .targets()
                .iter()
                .map(|&target| target_reference(game, target, viewer)),
        );
    }
    response.extend(attacking);
    let record = DecisionRecord {
        actor: 2,
        ordinal: 0,
        first_position: 0,
        last_position: 0,
        count: 1,
        kind: format!("{:?}", space.kind),
        context: DecisionContext {
            turn: game.state.turn.turn_number,
            phase: game.state.turn.current_phase_kind() as u8,
            step: game.state.turn.current_step_kind() as u8,
            life: players.map(|p| game.state.players[p.0].life),
            hand,
            board_counts,
            mana_sources: board,
        },
        response,
        choice,
    };
    let announces = matches!(
        action,
        Action::PlayLand { .. }
            | Action::CastSpell { .. }
            | Action::ActivateCardAbility { .. }
            | Action::LearnTakeLesson { .. }
            | Action::LearnDiscard { .. }
    ) || (matches!(action, Action::SelectCard { .. })
        && matches!(
            space.kind,
            ActionSpaceKind::Discard | ActionSpaceKind::LegendRule
        ));
    let own = game
        .decision_memory_include_own
        .then(|| own_projection(game, &record, chooser, announces));
    Some(PendingDecision {
        viewer: viewer.0,
        record,
        own,
    })
}

pub(crate) fn commit(game: &mut Game, pending: impl IntoIterator<Item = PendingDecision>) {
    for pending in pending {
        if let Some(own) = pending.own {
            game.decision_memory[1 - pending.viewer].push(own);
        }
        game.decision_memory[pending.viewer].push(pending.record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::{
            observation::Observation,
            observation_encoder::{self, ObservationEncoderConfig},
        },
        state::player::PlayerConfig,
    };

    fn game() -> Game {
        let players: Vec<PlayerConfig> = (0..2)
            .map(|seat| {
                PlayerConfig::new(
                    format!("seat{seat}"),
                    [("Mountain".into(), 20), ("Lightning Bolt".into(), 20)].into(),
                )
            })
            .collect();
        let mut game = Game::new(players, 111, true);
        game.keep_opening_hands();
        game.decision_memory_enabled = true;
        let land = game
            .state
            .latest_objects()
            .find(|(_, data)| data.owner == PlayerId(0) && data.name == "Mountain")
            .unwrap()
            .0;
        game.move_object(land, ZoneType::Battlefield);
        let actions = game.compute_player_actions(PlayerId(0));
        game.publish_action_space(ActionSpace {
            player: Some(PlayerId(0)),
            kind: ActionSpaceKind::Priority,
            actions,
            focus: Vec::new(),
        });
        game
    }

    fn pending_pass(game: &Game) -> Option<PendingDecision> {
        let space = game.current_action_space.as_ref().unwrap();
        capture(
            game,
            space,
            &Action::PassPriority {
                player: PlayerId(0),
            },
        )
    }

    #[test]
    fn decision_memory_disabled_and_search_clones_do_not_capture() {
        let mut game = game();
        assert!(pending_pass(&game).is_some());
        let search = game.clone();
        assert!(pending_pass(&search).is_none());
        assert!(pending_pass(&game.rollback_snapshot()).is_some());
        game.decision_memory_enabled = false;
        let pass = game
            .current_action_space
            .as_ref()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. }))
            .unwrap();
        game.step(pass).unwrap();
        assert!(game.decision_memory.iter().all(DecisionLog::is_empty));
    }

    #[test]
    fn decision_memory_commit_replay_rejection_and_viewer_identity() {
        let mut first = game();
        assert!(first.step(usize::MAX).is_err());
        assert!(first.decision_memory.iter().all(DecisionLog::is_empty));
        let pass = first
            .current_action_space
            .as_ref()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. }))
            .unwrap();
        first.step(pass).unwrap();
        assert!(first.decision_memory[0].is_empty());
        assert_eq!(first.decision_memory[1].len(), 1);
        let mut replay = game();
        replay.step(pass).unwrap();
        assert_eq!(first.decision_memory, replay.decision_memory);
        let observation = Observation::for_player(&first, PlayerId(1));
        let decoded = crate::agent::observation_json::parse(&observation.to_json()).unwrap();
        assert_eq!(decoded.decision_memory, observation.decision_memory);
    }

    #[test]
    fn suffix_encoding_matches_dense_through_chunks_coalescing_and_reset() {
        let mut game = game();
        let pending = pending_pass(&game).unwrap();
        let mut cursor = MemoryEncoding::default();
        let mut retained = EncodedMemory::default();
        for step in 0..100 {
            let mut row = pending.clone();
            row.record.context.turn = step / 2; // each second pass coalesces
            commit(&mut game, Some(row));
            let log = &game.decision_memory[1];
            for _ in 0..2 {
                // unchanged reads write no rows
                let (starts, suffix) = encode_suffix(log, &mut cursor);
                if suffix.reset {
                    retained = EncodedMemory::default();
                }
                retained.records.truncate(starts[0]);
                retained.details.truncate(starts[1]);
                retained.references.truncate(starts[2]);
                retained.members.truncate(starts[3]);
                retained.records.extend(suffix.records);
                retained.details.extend(suffix.details);
                retained.references.extend(suffix.references);
                retained.members.extend(suffix.members);
                let dense = encode(log);
                assert_eq!(retained.records, dense.records);
                assert_eq!(retained.details, dense.details);
                assert_eq!(retained.references, dense.references);
                assert_eq!(retained.members, dense.members);
            }
        }
        let restored: DecisionLog =
            serde_json::from_str(&serde_json::to_string(&game.decision_memory[1]).unwrap())
                .unwrap();
        assert!(encode_suffix(&restored, &mut cursor).1.reset);
        assert!(encode_suffix(&DecisionLog::default(), &mut cursor).1.reset);
    }

    #[test]
    fn decision_memory_prefixes_coalesce_without_mutating_observations_or_forks() {
        let mut game = game();
        let pending = pending_pass(&game);
        commit(&mut game, pending.clone());
        let old = Observation::for_player(&game, PlayerId(1));
        let mut fork = game.clone();
        fork.decision_memory_enabled = true;
        assert!(Arc::ptr_eq(
            game.decision_memory[1].tail.as_ref().unwrap(),
            fork.decision_memory[1].tail.as_ref().unwrap()
        ));
        commit(&mut fork, pending.clone());
        assert_eq!(old.decision_memory.iter().next().unwrap().count, 1);
        assert_eq!(game.decision_memory[1].iter().next().unwrap().count, 1);
        assert_eq!(fork.decision_memory[1].iter().next().unwrap().count, 2);
        let encoded = encode(&fork.decision_memory[1]);
        assert_eq!(
            fork.decision_memory[1].detail_count(),
            encoded.details.len()
        );
        assert_eq!(
            fork.decision_memory[1].member_count(),
            encoded.members.len()
        );
        fork.state.players[0].life -= 1;
        let changed = pending_pass(&fork);
        commit(&mut fork, changed);
        assert_eq!(fork.decision_memory[1].len(), 2);
        for turn in 2..100 {
            let mut row = pending.clone().unwrap();
            row.record.context.turn = turn;
            commit(&mut fork, Some(row));
        }
        assert!(fork.decision_memory[1].len() > CHUNK);
        assert_eq!(old.decision_memory.len(), 1);
        let rows: Vec<_> = fork.decision_memory[1].iter().collect();
        assert!(rows
            .windows(2)
            .all(|pair| pair[0].last_position < pair[1].first_position));
    }

    #[test]
    fn decision_memory_pass_filter_is_public_and_forced_choices_are_omitted() {
        let mut game = game();
        assert!(pending_pass(&game).is_some());
        let mut space = game.current_action_space.as_ref().unwrap().clone();
        space
            .actions
            .retain(|action| matches!(action, Action::PassPriority { .. }));
        assert!(capture(&game, &space, &space.actions[0]).is_none());
        let id = game.battlefield_permanents(PlayerId(0))[0];
        game.state.permanents[id].as_mut().unwrap().tapped = true;
        assert!(pending_pass(&game).is_none());
        game.state.permanents[id].as_mut().unwrap().tapped = false;
        game.scenario_clear_hand(PlayerId(0));
        assert!(pending_pass(&game).is_none());
    }

    #[test]
    fn decision_memory_private_choices_remain_unavailable_after_reveal() {
        let mut game = game();
        game.decision_memory_include_own = true;
        let object = game
            .state
            .zones
            .latest()
            .find(|&id| game.state.zones.zone_of(id) == Some(ZoneType::Library))
            .unwrap();
        let actions: Vec<_> = [false, true]
            .map(|to_bottom| Action::ScryCard {
                player: PlayerId(0),
                object,
                to_bottom,
            })
            .into();
        let space = ActionSpace {
            player: Some(PlayerId(0)),
            kind: ActionSpaceKind::Scry,
            actions,
            focus: Vec::new(),
        };
        let first = capture(&game, &space, &space.actions[0]);
        let second = capture(&game, &space, &space.actions[1]);
        assert_eq!(first, second);
        commit(&mut game, first);
        let before = game.decision_memory[1].clone();
        let own_before = game.decision_memory[0].clone();
        assert!(own_before
            .iter()
            .all(|row| row.actor == 1 && row.choice.kind.is_none()));
        game.move_object(object, ZoneType::Graveyard);
        assert_eq!(game.decision_memory[1], before);
        assert_eq!(game.decision_memory[0], own_before);
        assert!(before.iter().next().unwrap().choice.kind.is_none());
    }

    #[test]
    fn decision_memory_capacity_refuses_each_complete_companion_before_writes() {
        let mut game = game();
        let mut pending = pending_pass(&game).unwrap();
        let mut changed = (*pending.record.context.mana_sources[0]).clone();
        changed.tapped = true;
        pending.record.response.push(Arc::new(changed));
        commit(&mut game, Some(pending.clone()));
        pending.record.context.life[0] -= 1;
        commit(&mut game, Some(pending));
        let obs = Observation::for_player(&game, PlayerId(1));
        let memory = encode(&obs.decision_memory);
        let config = ObservationEncoderConfig {
            max_decision_records: memory.records.len(),
            max_decision_references: memory.references.len(),
            max_decision_reference_details: memory.details.len(),
            max_decision_members: memory.members.len(),
            ..Default::default()
        };
        assert!(observation_encoder::encode(&obs, &config).is_ok());
        for field in [
            "decision_records",
            "decision_references",
            "decision_reference_details",
            "decision_members",
        ] {
            let mut small = config;
            match field {
                "decision_records" => small.max_decision_records -= 1,
                "decision_references" => small.max_decision_references -= 1,
                "decision_reference_details" => small.max_decision_reference_details -= 1,
                "decision_members" => small.max_decision_members -= 1,
                _ => unreachable!(),
            }
            match observation_encoder::encode(&obs, &small) {
                Err(observation_encoder::ObservationEncodeError::Capacity {
                    field: actual,
                    ..
                }) => assert_eq!(actual, field),
                _ => panic!("missing complete-prefix capacity error for {field}"),
            }
        }
    }

    #[test]
    fn decision_memory_chooser_is_resolved_once_but_pass_uses_rules_resources() {
        use crate::{flow::turn::PlayerControl, state::ability::PlayerControlDuration};
        let mut game = game();
        game.state.turn.player_controls.push(PlayerControl {
            player: PlayerId(0),
            controller: PlayerId(1),
            duration: PlayerControlDuration::NextTurn,
            active: true,
        });
        let space = game.current_action_space.as_ref().unwrap().clone();
        game.publish_action_space(space);
        let pending = pending_pass(&game).expect("seat zero still has the resources");
        assert_eq!(
            pending.viewer, 0,
            "the other chooser is the opponent of viewer zero"
        );
        commit(&mut game, Some(pending));
        assert_eq!(game.decision_memory[0].len(), 1);
        assert!(game.decision_memory[1].is_empty());
    }
    /// A manifest turned face up is named by its permanent alias (CR 708.2),
    /// and its actor's own record keeps the definition it now shows.
    #[test]
    fn decision_memory_own_record_keeps_a_turned_up_manifest() {
        use crate::flow::zones::FormTransition;
        let mut game = game();
        game.decision_memory_include_own = true;
        let card = game
            .state
            .zones
            .zone_objects(ZoneType::Library, PlayerId(0))
            .iter()
            .copied()
            .find(|&object| game.state[object].name == "Mountain")
            .unwrap();
        let manifest = game.manifest(card, PlayerId(0));
        assert!(game.turn_over(manifest, FormTransition::TurnFaceUp));
        let alias = game.observed_object_id(manifest).0;
        assert_ne!(
            alias, game.state[manifest].id.0,
            "the alias is not the card"
        );
        let pending = pending_pass(&game).unwrap();
        let shown = |record: &DecisionRecord| {
            record
                .context
                .mana_sources
                .iter()
                .find(|reference| reference.appearance == Some(alias))
                .map(|reference| reference.definition)
        };
        let mountain = Some(game.state[manifest].definition_id.0);
        assert_eq!(shown(&pending.record), Some(mountain));
        assert_eq!(shown(pending.own.as_ref().unwrap()), Some(mountain));
    }

    #[test]
    fn decision_memory_own_option_rebases_only_the_opposing_public_projection() {
        let mut game = game();
        game.decision_memory_include_own = true;
        let pending = pending_pass(&game).unwrap();
        commit(&mut game, Some(pending.clone()));
        let other = game.decision_memory[1].iter().next().unwrap();
        let own = game.decision_memory[0].iter().next().unwrap();
        assert_eq!((other.actor, own.actor), (2, 1));
        assert_eq!(
            own.context.life,
            [other.context.life[1], other.context.life[0]]
        );
        assert_eq!(
            own.context.hand,
            [other.context.hand[1], other.context.hand[0]]
        );
        for (own, other) in own
            .context
            .mana_sources
            .iter()
            .zip(&other.context.mana_sources)
        {
            assert_eq!(own.definition, other.definition);
            assert_eq!(own.program, other.program);
            assert_eq!(own.role, if other.role == 0 { 0 } else { 3 - other.role });
        }
        // Even identical public passes from opposite actors cannot coalesce.
        let mut other_actor = pending.record;
        other_actor.actor = 1;
        game.decision_memory[1].push(other_actor);
        assert_eq!(game.decision_memory[1].len(), 2);
        let issued = Observation::for_player(&game, PlayerId(0));
        let mut fork = game.clone();
        fork.decision_memory_enabled = true;
        commit(&mut fork, pending_pass(&game));
        assert_eq!(issued.decision_memory.len(), 1);
        assert_eq!(game.decision_memory[0].iter().next().unwrap().count, 1);
        assert_eq!(fork.decision_memory[0].iter().next().unwrap().count, 2);
    }

    #[test]
    fn decision_memory_shared_snapshots_preserve_repeated_members_and_roundtrip() {
        let mut game = game();
        let mut pending = pending_pass(&game).unwrap();
        let reference = pending.record.context.mana_sources[0].clone();
        pending.record.response = vec![reference.clone(), reference];
        commit(&mut game, Some(pending.clone()));
        pending.record.context.life[0] -= 1;
        commit(&mut game, Some(pending));
        let log = &game.decision_memory[1];
        assert_eq!(log.reference_count(), 1);
        let encoded = encode(log);
        assert_eq!(encoded.members.len(), 6);
        assert_eq!(log.detail_count(), encoded.details.len());
        assert_eq!(log.member_count(), encoded.members.len());
        assert_eq!(log.reference_count(), encoded.references.len());
        assert!(encoded.members.iter().all(|row| row[0] == 1.));
        assert_eq!(encoded.members[2][1], 1.);
        let decoded: DecisionLog =
            serde_json::from_str(&serde_json::to_string(log).unwrap()).unwrap();
        assert_eq!(log, &decoded);
        let restored = encode(&decoded);
        assert_eq!(encoded.records, restored.records);
        assert_eq!(encoded.references, restored.references);
        assert_eq!(encoded.details, restored.details);
        assert_eq!(encoded.members, restored.members);
    }

    #[test]
    fn decision_memory_definition_shortcut_requires_exact_description_and_program() {
        let game = game();
        let object = game
            .state
            .latest_objects()
            .find(|(_, row)| row.name == "Lightning Bolt")
            .unwrap()
            .0;
        let exact = reference(&game, object, PlayerId(1), true);
        assert!(definition_only(&exact));
        let mut changed = (*exact).clone();
        changed.description = Arc::from("different effective rules");
        assert!(!definition_only(&changed));
        changed = (*exact).clone();
        Arc::make_mut(&mut changed.program)[0].1 ^= 1;
        assert!(!definition_only(&changed));
        assert!(!definition_only(&DecisionReference::default()));
    }
    #[test]
    fn decision_memory_own_projection_cannot_declassify_opponents_private_hand() {
        let mut game = game();
        let hidden = game
            .state
            .latest_objects()
            .find(|(object, row)| {
                row.owner == PlayerId(1)
                    && game.state.zones.zone_of(*object) == Some(ZoneType::Hand)
            })
            .unwrap()
            .0;
        assert!(game
            .visible_reference(hidden, PlayerId(0))
            .definition
            .is_none());
        let mut pending = pending_pass(&game).unwrap();
        pending.record.response = vec![reference(&game, hidden, PlayerId(1), false)];
        assert!(pending.record.response[0].definition.is_some());
        let own = own_projection(&game, &pending.record, PlayerId(0), false);
        assert!(own.response[0].definition.is_none());
        assert!(own.response[0].program.is_empty());
        assert!(own.response[0].description.is_empty());
        game.move_object(hidden, ZoneType::Graveyard);
        assert!(own.response[0].definition.is_none());
    }
}
