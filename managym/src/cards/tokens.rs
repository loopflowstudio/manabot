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
//! reference to the created token's row, and a double-faced card's
//! `back_face` one to its back face's row, so the policy reads that
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

impl TokenKind {
    /// The kind a written `u16` names; writers only ever store these values.
    fn of(kind: u16) -> Self {
        match kind {
            1 => Self::Structure,
            2 => Self::Field,
            3 => Self::Variant,
            4 => Self::Integer,
            5 => Self::Boolean,
            6 => Self::Text,
            7 => Self::DefinitionRef,
            8 => Self::Null,
            9 => Self::ProgramKind,
            other => unreachable!("unknown token kind {other}"),
        }
    }
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
    writer.append_definition(definition)?;
    Ok(writer
        .into_rules_tokens()
        .into_iter()
        .map(|(kind, value)| (TokenKind::of(kind), value))
        .collect())
}

/// Serialize a resolved rules value with the same vocabulary as card programs.
pub fn rules_tokens<T: Serialize>(
    value: &T,
    pack: &ContentPack,
) -> Result<Vec<(u16, i32)>, ContentError> {
    let mut writer = TokenWriter::new(pack);
    writer.append_rules(value)?;
    Ok(writer.into_rules_tokens())
}

/// The token catalog of every definition in `pack`, rows in `CardDefId` order.
pub fn learning_catalog(pack: &ContentPack) -> Result<LearningCatalog, ContentError> {
    let mut out = LearningCatalog::default();
    let mut programs: Vec<Span> = Vec::new();
    let mut definition_programs: Vec<Vec<i32>> = Vec::new();

    out.definition_offsets.push(0);
    for (_, definition) in pack.definition_entries() {
        let mut writer = TokenWriter::new(pack);
        definition.serialize(&mut writer)?;
        append(&mut out, writer.span());
        out.definition_offsets.push(out.token_kind.len() as i32);

        let mut rows = Vec::new();
        for (kind, tokens) in definition_programs_of(definition, pack)? {
            rows.push(programs.len() as i32);
            let mut span = TokenWriter::new(pack);
            span.push(TokenKind::ProgramKind, kind as i32);
            span.extend(tokens);
            programs.push(span.span());
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

/// A written token stream: `(kind, value)` exactly as the policy reads them,
/// plus the positions of its definition references and their target rows, so
/// the catalog can join created tokens and back faces.
struct Span {
    tokens: Vec<(u16, i32)>,
    references: Vec<(usize, i32)>,
}

fn append(out: &mut LearningCatalog, span: Span) {
    let offset = out.token_kind.len();
    for (index, row) in span.references {
        out.definition_ref_source_tokens
            .push((offset + index) as i32);
        out.definition_ref_target_rows.push(row);
    }
    for (kind, value) in span.tokens {
        out.token_kind.push(kind);
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
        Ok(writer.span())
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

/// Writes tokens in their final `(kind, value)` form; no second pass converts
/// them, and names are hashed in place without temporary strings.
pub(crate) struct TokenWriter<'p> {
    pack: &'p ContentPack,
    tokens: Vec<(u16, i32)>,
    /// Positions of definition reference tokens and their target rows.
    references: Vec<(usize, i32)>,
    /// Name of the struct field being written, for `token_name` references.
    field: Option<&'static str>,
    /// Depth of nested structs; the top-level `CardDefinition` is depth 1.
    depth: usize,
}

impl<'p> TokenWriter<'p> {
    pub(crate) fn new(pack: &'p ContentPack) -> Self {
        Self {
            pack,
            tokens: Vec::new(),
            references: Vec::new(),
            field: None,
            depth: 0,
        }
    }

    /// Append to one retained-program stream without allocating a writer per field.
    pub(crate) fn append_rules<T: Serialize>(&mut self, value: &T) -> Result<(), ContentError> {
        value.serialize(self)
    }

    pub(crate) fn append_definition(
        &mut self,
        definition: &CardDefinition,
    ) -> Result<(), ContentError> {
        definition.serialize(&mut *self)?;
        for (kind, span) in definition_programs_of(definition, self.pack)? {
            self.push(TokenKind::ProgramKind, kind as i32);
            self.extend(span);
        }
        Ok(())
    }

    pub(crate) fn into_rules_tokens(self) -> Vec<(u16, i32)> {
        self.tokens
    }

    fn span(self) -> Span {
        Span {
            tokens: self.tokens,
            references: self.references,
        }
    }

    fn extend(&mut self, span: Span) {
        let offset = self.tokens.len();
        self.references.extend(
            span.references
                .into_iter()
                .map(|(index, row)| (offset + index, row)),
        );
        self.tokens.extend(span.tokens);
    }

    fn push(&mut self, kind: TokenKind, value: i32) {
        self.tokens.push((kind as u16, value));
    }

    /// `Name`, `Name::Variant` or their `/`-prefixed closing form, hashed as
    /// one path: the same value `name_value` gives the joined string.
    fn path_value(name: &str, variant: Option<&str>, closing: bool) -> i32 {
        let mut hash = blake3::Hasher::new();
        if closing {
            hash.update(b"/");
        }
        hash.update(name.as_bytes());
        if let Some(variant) = variant {
            hash.update(b"::");
            hash.update(variant.as_bytes());
        }
        let bytes = hash.finalize();
        (u32::from_le_bytes(bytes.as_bytes()[..4].try_into().unwrap()) & 0x7fff_ffff) as i32
    }

    fn open(&mut self, name: &'static str) {
        self.push(TokenKind::Structure, Self::path_value(name, None, false));
    }

    fn variant(&mut self, name: &'static str, variant: &'static str) {
        self.push(
            TokenKind::Variant,
            Self::path_value(name, Some(variant), false),
        );
    }

    fn close(&mut self, name: &'static str, variant: Option<&'static str>) {
        self.push(TokenKind::Structure, Self::path_value(name, variant, true));
    }

    fn integer(&mut self, value: i64) -> Result<(), ContentError> {
        let value = i32::try_from(value)
            .map_err(|_| ContentError(format!("integer {value} exceeds i32")))?;
        self.push(TokenKind::Integer, value);
        Ok(())
    }
}

impl ser::Error for ContentError {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        ContentError(message.to_string())
    }
}

fn unsupported<T>(what: &str) -> Result<T, ContentError> {
    Err(ContentError(format!(
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
        if matches!(self.field, Some("token_name" | "back_face")) {
            let row = self
                .pack
                .definition_id(value)
                .ok_or_else(|| ContentError(format!("{value:?} is not in the pack")))?;
            let CardDefId(row) = row;
            self.references.push((self.tokens.len(), row as i32));
            self.push(TokenKind::DefinitionRef, row as i32);
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
        self.close(name, None);
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        name: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), ContentError> {
        self.variant(name, variant);
        Ok(())
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<(), ContentError> {
        self.open(name);
        value.serialize(&mut *self)?;
        self.close(name, None);
        Ok(())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), ContentError> {
        self.variant(name, variant);
        value.serialize(&mut *self)?;
        self.close(name, Some(variant));
        Ok(())
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Compound<'a, 'p>, ContentError> {
        self.open("[");
        Ok(Compound::new(self, "[", None, false))
    }
    fn serialize_tuple(self, _: usize) -> Result<Compound<'a, 'p>, ContentError> {
        self.open("[");
        Ok(Compound::new(self, "[", None, false))
    }
    fn serialize_tuple_struct(
        self,
        name: &'static str,
        _: usize,
    ) -> Result<Compound<'a, 'p>, ContentError> {
        self.open(name);
        Ok(Compound::new(self, name, None, false))
    }
    fn serialize_tuple_variant(
        self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Compound<'a, 'p>, ContentError> {
        self.variant(name, variant);
        Ok(Compound::new(self, name, Some(variant), false))
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
        Ok(Compound::new(self, name, None, top))
    }
    fn serialize_struct_variant(
        self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Compound<'a, 'p>, ContentError> {
        self.variant(name, variant);
        self.depth += 1;
        Ok(Compound::new(self, name, Some(variant), false))
    }
}

pub(crate) struct Compound<'a, 'p> {
    writer: &'a mut TokenWriter<'p>,
    name: &'static str,
    variant: Option<&'static str>,
    /// The definition's own struct: identity and program fields are skipped.
    definition: bool,
}

impl<'a, 'p> Compound<'a, 'p> {
    fn new(
        writer: &'a mut TokenWriter<'p>,
        name: &'static str,
        variant: Option<&'static str>,
        definition: bool,
    ) -> Self {
        Self {
            writer,
            name,
            variant,
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
        self.writer.close(self.name, self.variant);
        Ok(())
    }

    fn end(self) -> Result<(), ContentError> {
        self.writer.close(self.name, self.variant);
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
    fn retained_rules_tokens_preserve_created_definition_identity() {
        let pack = default_content_pack();
        let food = rules_tokens(&Effect::create_tokens("Food", 1), pack).unwrap();
        let ally = rules_tokens(&Effect::create_tokens("Ally", 1), pack).unwrap();
        assert_ne!(food, ally);
        for (name, tokens) in [("Food", food), ("Ally", ally)] {
            assert!(tokens.contains(&(
                TokenKind::DefinitionRef as u16,
                pack.definition_id(name).unwrap().0 as i32
            )));
        }
    }

    #[test]
    fn default_catalog_spans_every_definition_and_resolves_token_references() {
        let pack = default_content_pack();
        let catalog = learning_catalog(pack).unwrap();
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
            // A created token, or a double-faced card's back face.
            let row = CardDefId(row as u32);
            assert!(pack.definition(row).unwrap().is_token || pack.front_face(row).is_some());
        }
    }
}
