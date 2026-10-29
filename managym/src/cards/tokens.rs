//! Learning tokens derived from the engine's own card types.
//!
//! A card's rules live once, in the [`CardDefinition`] its set file writes.
//! The policy reads each definition as a token sequence produced by walking
//! that same value with a `serde` serializer, so every field, `Effect`,
//! `TriggerCondition` and keyword reaches the tokens without a second
//! vocabulary: adding a variant adds its tokens.
//!
//! Each definition contributes one *characteristics* span and one span per
//! program (each triggered, mana, triggered-mana and activated ability, plus
//! the spell). Identity and printed text (`key`, `scryfall_id`, `name`,
//! `text_box`) are never tokens. A `token_name` becomes a definition
//! reference to the created token's row, so the policy reads the token's own
//! definition in place.
//!
//! Token values are stable hashes of the Rust names (`"Effect::DealDamage"`,
//! `"amount"`, a subtype such as `"Ally"`); integers and booleans carry their
//! value. Rows are the pack's [`CardDefId`]s.
//!
//! Entry point: [`learning_catalog`].

use serde::ser::{self, Serialize};

use super::ContentError;
use crate::cards::content_pack::ContentPack;
use crate::state::{
    ability::{Effect, TargetRequirement},
    card::{CardDefId, CardDefinition},
};

/// What a token's value means. Numbering is part of the learning schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum TokenKind {
    /// Opening or closing a struct, variant body or list.
    Structure = 1,
    Field = 2,
    Variant = 3,
    Integer = 4,
    Boolean = 5,
    /// A string value: a subtype, supertype or other printed category.
    Text = 6,
    /// A reference to another definition row (a created token).
    DefinitionRef = 7,
    Null = 8,
    ProgramKind = 9,
}

/// Which part of a definition a program span encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ProgramKind {
    Triggered = 1,
    Mana = 2,
    TriggeredMana = 3,
    Activated = 4,
    Spell = 5,
}

/// One pack's token catalog in the flat layout the Python projection binds:
/// `definition_offsets[row]..[row + 1]` is a characteristics span;
/// `definition_program_rows[definition_program_offsets[row]..]` lists its
/// program spans in `program_offsets`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LearningCatalog {
    pub token_kind: Vec<u16>,
    pub token_value: Vec<i32>,
    pub definition_offsets: Vec<i32>,
    pub program_offsets: Vec<i32>,
    pub definition_program_offsets: Vec<i32>,
    pub definition_program_rows: Vec<i32>,
    pub definition_ref_source_tokens: Vec<i32>,
    pub definition_ref_target_rows: Vec<i32>,
}

/// Fields that name or print a card rather than describe its rules. (`key`
/// and `scryfall_id` are not serialized at all.)
const IDENTITY_FIELDS: [&str; 2] = ["name", "text_box"];

/// Fields encoded as their own program spans instead of characteristics.
const PROGRAM_FIELDS: [&str; 6] = [
    "abilities",
    "mana_abilities",
    "triggered_mana_abilities",
    "activated_abilities",
    "spell_effects",
    "targeting",
];

/// Stable token value for a Rust name or printed category.
pub fn name_value(name: &str) -> i32 {
    let hash = blake3::hash(name.as_bytes());
    let bytes = hash.as_bytes();
    (u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) & 0x7fff_ffff) as i32
}

/// One definition's tokens, characteristics then programs, as `(kind, value)`.
/// Token names in its effects must name definitions in `pack`.
pub fn definition_tokens(
    definition: &CardDefinition,
    pack: &ContentPack,
) -> Result<Vec<(TokenKind, i32)>, ContentError> {
    let mut writer = TokenWriter::new(pack);
    definition.serialize(&mut writer)?;
    let mut out = writer.tokens;
    for (kind, tokens) in definition_programs_of(definition, pack)? {
        out.push((TokenKind::ProgramKind, kind as i32, None));
        out.extend(tokens);
    }
    Ok(out
        .into_iter()
        .map(|(kind, value, _)| (kind, value))
        .collect())
}

/// The token catalog of every definition in `pack`, rows in `CardDefId` order.
pub fn learning_catalog(pack: &ContentPack) -> Result<LearningCatalog, ContentError> {
    let mut out = LearningCatalog::default();
    let mut programs: Vec<Vec<(TokenKind, i32, Option<i32>)>> = Vec::new();
    let mut definition_programs: Vec<Vec<i32>> = Vec::new();

    out.definition_offsets.push(0);
    for (_, definition) in pack.definition_entries() {
        let mut writer = TokenWriter::new(pack);
        definition.serialize(&mut writer)?;
        append(&mut out, writer.tokens);
        out.definition_offsets.push(out.token_kind.len() as i32);

        let mut rows = Vec::new();
        for (kind, tokens) in definition_programs_of(definition, pack)? {
            rows.push(programs.len() as i32);
            let mut span = vec![(TokenKind::ProgramKind, kind as i32, None)];
            span.extend(tokens);
            programs.push(span);
        }
        definition_programs.push(rows);
    }

    out.program_offsets.push(out.token_kind.len() as i32);
    for span in programs {
        append(&mut out, span);
        out.program_offsets.push(out.token_kind.len() as i32);
    }
    out.definition_program_offsets.push(0);
    for rows in definition_programs {
        out.definition_program_rows.extend(rows);
        out.definition_program_offsets
            .push(out.definition_program_rows.len() as i32);
    }
    Ok(out)
}

type Span = Vec<(TokenKind, i32, Option<i32>)>;

fn append(out: &mut LearningCatalog, tokens: Span) {
    for (kind, value, reference) in tokens {
        if let Some(row) = reference {
            out.definition_ref_source_tokens
                .push(out.token_kind.len() as i32);
            out.definition_ref_target_rows.push(row);
        }
        out.token_kind.push(kind as u16);
        out.token_value.push(value);
    }
}

/// The spell's resolution and its explicit targeting clauses form one program.
#[derive(serde::Serialize)]
struct SpellProgram<'a> {
    effects: &'a [Effect],
    targeting: &'a [TargetRequirement],
}

fn definition_programs_of(
    definition: &CardDefinition,
    pack: &ContentPack,
) -> Result<Vec<(ProgramKind, Span)>, ContentError> {
    fn tokens<T: Serialize>(value: &T, pack: &ContentPack) -> Result<Span, ContentError> {
        let mut writer = TokenWriter::new(pack);
        value.serialize(&mut writer)?;
        Ok(writer.tokens)
    }
    let mut out = Vec::new();
    for ability in &definition.abilities {
        out.push((ProgramKind::Triggered, tokens(ability, pack)?));
    }
    for ability in &definition.mana_abilities {
        out.push((ProgramKind::Mana, tokens(ability, pack)?));
    }
    for ability in &definition.triggered_mana_abilities {
        out.push((ProgramKind::TriggeredMana, tokens(ability, pack)?));
    }
    for ability in &definition.activated_abilities {
        out.push((ProgramKind::Activated, tokens(ability, pack)?));
    }
    if !definition.spell_effects.is_empty() || !definition.targeting.is_empty() {
        let spell = SpellProgram {
            effects: &definition.spell_effects,
            targeting: &definition.targeting,
        };
        out.push((ProgramKind::Spell, tokens(&spell, pack)?));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The serializer.

struct TokenWriter<'p> {
    pack: &'p ContentPack,
    tokens: Span,
    /// Name of the struct field being written, for `token_name` references.
    field: Option<&'static str>,
    /// Depth of nested structs; the top-level `CardDefinition` is depth 1.
    depth: usize,
}

impl<'p> TokenWriter<'p> {
    fn new(pack: &'p ContentPack) -> Self {
        Self {
            pack,
            tokens: Vec::new(),
            field: None,
            depth: 0,
        }
    }

    fn push(&mut self, kind: TokenKind, value: i32) {
        self.tokens.push((kind, value, None));
    }

    fn open(&mut self, name: &str) {
        self.push(TokenKind::Structure, name_value(name));
    }

    fn close(&mut self, name: &str) {
        self.push(TokenKind::Structure, name_value(&format!("/{name}")));
    }

    fn integer(&mut self, value: i64) -> Result<(), ContentError> {
        let value = i32::try_from(value)
            .map_err(|_| ContentError::Malformed(format!("integer {value} exceeds i32")))?;
        self.push(TokenKind::Integer, value);
        Ok(())
    }
}

impl ser::Error for ContentError {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        ContentError::Malformed(message.to_string())
    }
}

fn unsupported<T>(what: &str) -> Result<T, ContentError> {
    Err(ContentError::Malformed(format!(
        "learning tokens do not encode {what}"
    )))
}

impl<'a, 'p> ser::Serializer for &'a mut TokenWriter<'p> {
    type Ok = ();
    type Error = ContentError;
    type SerializeSeq = Compound<'a, 'p>;
    type SerializeTuple = Compound<'a, 'p>;
    type SerializeTupleStruct = Compound<'a, 'p>;
    type SerializeTupleVariant = Compound<'a, 'p>;
    type SerializeMap = Compound<'a, 'p>;
    type SerializeStruct = Compound<'a, 'p>;
    type SerializeStructVariant = Compound<'a, 'p>;

    fn serialize_bool(self, value: bool) -> Result<(), ContentError> {
        self.push(TokenKind::Boolean, i32::from(value));
        Ok(())
    }
    fn serialize_i8(self, value: i8) -> Result<(), ContentError> {
        self.integer(value.into())
    }
    fn serialize_i16(self, value: i16) -> Result<(), ContentError> {
        self.integer(value.into())
    }
    fn serialize_i32(self, value: i32) -> Result<(), ContentError> {
        self.integer(value.into())
    }
    fn serialize_i64(self, value: i64) -> Result<(), ContentError> {
        self.integer(value)
    }
    fn serialize_u8(self, value: u8) -> Result<(), ContentError> {
        self.integer(value.into())
    }
    fn serialize_u16(self, value: u16) -> Result<(), ContentError> {
        self.integer(value.into())
    }
    fn serialize_u32(self, value: u32) -> Result<(), ContentError> {
        self.integer(value.into())
    }
    fn serialize_u64(self, value: u64) -> Result<(), ContentError> {
        self.integer(i64::try_from(value).unwrap_or(i64::MAX))
    }
    fn serialize_f32(self, _: f32) -> Result<(), ContentError> {
        unsupported("floating point")
    }
    fn serialize_f64(self, _: f64) -> Result<(), ContentError> {
        unsupported("floating point")
    }
    fn serialize_char(self, _: char) -> Result<(), ContentError> {
        unsupported("characters")
    }
    fn serialize_str(self, value: &str) -> Result<(), ContentError> {
        if self.field == Some("token_name") {
            let row = self.pack.definition_id(value).ok_or_else(|| {
                ContentError::Malformed(format!("token {value:?} is not in the pack"))
            })?;
            let CardDefId(row) = row;
            self.tokens
                .push((TokenKind::DefinitionRef, 0, Some(row as i32)));
        } else {
            self.push(TokenKind::Text, name_value(value));
        }
        Ok(())
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<(), ContentError> {
        unsupported("bytes")
    }
    fn serialize_none(self) -> Result<(), ContentError> {
        self.push(TokenKind::Null, 0);
        Ok(())
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), ContentError> {
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), ContentError> {
        self.push(TokenKind::Null, 0);
        Ok(())
    }
    fn serialize_unit_struct(self, name: &'static str) -> Result<(), ContentError> {
        self.open(name);
        self.close(name);
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        name: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), ContentError> {
        self.push(
            TokenKind::Variant,
            name_value(&format!("{name}::{variant}")),
        );
        Ok(())
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<(), ContentError> {
        self.open(name);
        value.serialize(&mut *self)?;
        self.close(name);
        Ok(())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), ContentError> {
        let path = format!("{name}::{variant}");
        self.push(TokenKind::Variant, name_value(&path));
        value.serialize(&mut *self)?;
        self.close(&path);
        Ok(())
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Compound<'a, 'p>, ContentError> {
        self.open("[");
        Ok(Compound::new(self, "[".into(), false))
    }
    fn serialize_tuple(self, _: usize) -> Result<Compound<'a, 'p>, ContentError> {
        self.open("[");
        Ok(Compound::new(self, "[".into(), false))
    }
    fn serialize_tuple_struct(
        self,
        name: &'static str,
        _: usize,
    ) -> Result<Compound<'a, 'p>, ContentError> {
        self.open(name);
        Ok(Compound::new(self, name.into(), false))
    }
    fn serialize_tuple_variant(
        self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Compound<'a, 'p>, ContentError> {
        let path = format!("{name}::{variant}");
        self.push(TokenKind::Variant, name_value(&path));
        Ok(Compound::new(self, path, false))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Compound<'a, 'p>, ContentError> {
        unsupported("maps")
    }
    fn serialize_struct(
        self,
        name: &'static str,
        _: usize,
    ) -> Result<Compound<'a, 'p>, ContentError> {
        self.open(name);
        self.depth += 1;
        let top = self.depth == 1 && name == "CardDefinition";
        Ok(Compound::new(self, name.into(), top))
    }
    fn serialize_struct_variant(
        self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Compound<'a, 'p>, ContentError> {
        let path = format!("{name}::{variant}");
        self.push(TokenKind::Variant, name_value(&path));
        self.depth += 1;
        Ok(Compound::new(self, path, false))
    }
}

struct Compound<'a, 'p> {
    writer: &'a mut TokenWriter<'p>,
    name: String,
    /// The definition's own struct: identity and program fields are skipped.
    definition: bool,
}

impl<'a, 'p> Compound<'a, 'p> {
    fn new(writer: &'a mut TokenWriter<'p>, name: String, definition: bool) -> Self {
        Self {
            writer,
            name,
            definition,
        }
    }

    fn element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ContentError> {
        let field = self.writer.field.take();
        value.serialize(&mut *self.writer)?;
        self.writer.field = field;
        Ok(())
    }

    fn field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), ContentError> {
        if self.definition && (IDENTITY_FIELDS.contains(&key) || PROGRAM_FIELDS.contains(&key)) {
            return Ok(());
        }
        self.writer.push(TokenKind::Field, name_value(key));
        let outer = self.writer.field.replace(key);
        value.serialize(&mut *self.writer)?;
        self.writer.field = outer;
        Ok(())
    }

    fn end_struct(self) -> Result<(), ContentError> {
        self.writer.depth -= 1;
        self.writer.close(&self.name);
        Ok(())
    }

    fn end(self) -> Result<(), ContentError> {
        self.writer.close(&self.name);
        Ok(())
    }
}

impl ser::SerializeSeq for Compound<'_, '_> {
    type Ok = ();
    type Error = ContentError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ContentError> {
        // List elements keep the enclosing field, so a `Vec<String>` of
        // token names would still resolve; no current field nests that way.
        value.serialize(&mut *self.writer)
    }
    fn end(self) -> Result<(), ContentError> {
        Compound::end(self)
    }
}

impl ser::SerializeTuple for Compound<'_, '_> {
    type Ok = ();
    type Error = ContentError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ContentError> {
        self.element(value)
    }
    fn end(self) -> Result<(), ContentError> {
        Compound::end(self)
    }
}

impl ser::SerializeTupleStruct for Compound<'_, '_> {
    type Ok = ();
    type Error = ContentError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ContentError> {
        self.element(value)
    }
    fn end(self) -> Result<(), ContentError> {
        Compound::end(self)
    }
}

impl ser::SerializeTupleVariant for Compound<'_, '_> {
    type Ok = ();
    type Error = ContentError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ContentError> {
        self.element(value)
    }
    fn end(self) -> Result<(), ContentError> {
        Compound::end(self)
    }
}

impl ser::SerializeMap for Compound<'_, '_> {
    type Ok = ();
    type Error = ContentError;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, _: &T) -> Result<(), ContentError> {
        unsupported("maps")
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, _: &T) -> Result<(), ContentError> {
        unsupported("maps")
    }
    fn end(self) -> Result<(), ContentError> {
        unsupported("maps")
    }
}

impl ser::SerializeStruct for Compound<'_, '_> {
    type Ok = ();
    type Error = ContentError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), ContentError> {
        self.field(key, value)
    }
    fn end(self) -> Result<(), ContentError> {
        self.end_struct()
    }
}

impl ser::SerializeStructVariant for Compound<'_, '_> {
    type Ok = ();
    type Error = ContentError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), ContentError> {
        self.field(key, value)
    }
    fn end(self) -> Result<(), ContentError> {
        self.end_struct()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::content_pack::default_content_pack;

    #[test]
    fn learning_schema_numbers_the_engine_kinds() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../content/semantic/current/learning_schema.json"
        ))
        .unwrap();
        let kinds = [
            ("structure", TokenKind::Structure),
            ("field", TokenKind::Field),
            ("variant", TokenKind::Variant),
            ("integer", TokenKind::Integer),
            ("boolean", TokenKind::Boolean),
            ("text", TokenKind::Text),
            ("definition_ref", TokenKind::DefinitionRef),
            ("null", TokenKind::Null),
            ("program_kind", TokenKind::ProgramKind),
        ];
        for (name, kind) in kinds {
            assert_eq!(schema["token_kinds"][name], kind as u16, "{name}");
        }
        let programs = [
            ("triggered", ProgramKind::Triggered),
            ("mana", ProgramKind::Mana),
            ("triggered_mana", ProgramKind::TriggeredMana),
            ("activated", ProgramKind::Activated),
            ("spell", ProgramKind::Spell),
        ];
        for (name, kind) in programs {
            assert_eq!(schema["program_kinds"][name], kind as i32, "{name}");
        }
    }

    #[test]
    fn default_catalog_spans_every_definition_and_resolves_token_references() {
        let pack = default_content_pack();
        let catalog = learning_catalog(&pack).unwrap();
        assert_eq!(catalog.definition_offsets.len(), pack.len() + 1);
        assert_eq!(
            catalog.definition_ref_source_tokens.len(),
            catalog.definition_ref_target_rows.len()
        );
        for (&source, &row) in catalog
            .definition_ref_source_tokens
            .iter()
            .zip(&catalog.definition_ref_target_rows)
        {
            assert_eq!(
                catalog.token_kind[source as usize],
                TokenKind::DefinitionRef as u16
            );
            assert!(pack.definition(CardDefId(row as u32)).unwrap().is_token);
        }
    }
}
