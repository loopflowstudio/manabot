//! Compile the typed Rust set definitions into versioned semantic IR, in process.
//!
//! `cardsets/` is the only card authority. This module serializes those typed
//! definitions, resolves their symbolic references (definition refs and target
//! roles), assigns stable numeric opcodes and program indexes, and hashes the
//! result. The runtime, the shared pool descriptor and the learning projection
//! (through the native extension) all read these in-memory documents; nothing
//! is generated on disk.
//!
//! Principal entry points:
//! - [`document`]: the compiled IR for one current preset (a setup in
//!   `setups.json`, or `pool`: every card), shared by every consumer in the process.
//! - [`incomplete`]: defined cards whose behavior is known to be incomplete,
//!   with the reason; every other defined card is supported.
//!
//! The IR keeps the shape of the retired offline compiler
//! (`manabot/semantic/compiler.py`), so historical frozen IR stays readable by
//! the same parser and the learning projection binds without a separate schema.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::IrError;
use crate::cardsets;

const SETUPS: &str = include_str!("../../../content/semantic/current/setups.json");

pub const IR_SCHEMA_VERSION: u64 = 1;
const SOURCE_SCHEMA_VERSION: u64 = 1;

/// Opcode names in numeric order: `OPCODE_NAMES[n - 1]` names opcode `n`.
/// These numbers are learning token identities; append, never reorder.
pub(crate) const OPCODE_NAMES: [&str; 71] = [
    "add_mana",
    "draw_cards",
    "put_top_cards_in_hand",
    "create_token",
    "put_counters",
    "gain_life",
    "scry",
    "tap",
    "untap",
    "modify_pt",
    "grant_keywords",
    "restrict_blocking",
    "deal_damage",
    "return_to_hand",
    "learn",
    "counter_unless_pays",
    "look_and_select",
    "branch",
    "for_each_target",
    "earthbend",
    "exile_until_source_leaves",
    "deal_power_damage",
    "set_power_from_count",
    "search_basic_land",
    "optional_discard_draw",
    "discard_target_player",
    "destroy",
    "exile",
    "buff_target_value",
    "return_to_hand_draw_if_controlled",
    "modify_triggering_permanent",
    "deal_damage_value",
    "put_counters_value",
    "create_tokens_value",
    "exile_with_controller_token",
    "damage_opponent_creatures",
    "mill_cards",
    "mill_and_select",
    "drain_life_value",
    "counter_spell",
    "cant_be_blocked_target",
    "return_graveyard_target",
    "modal",
    "target_clause",
    "prevent_life_gain",
    "combat_requires_other",
    "blocking_restriction",
    "power_from_land_counters",
    "spell_cost_reduction",
    "static_pt",
    "search_basic_lands",
    "sacrifice",
    "deal_damage_to_controller",
    "drain_life",
    "unblockable_if",
    "add_lesson_mana",
    "grant_flashback",
    "grant_mana",
    "create_nonlegendary_copy",
    "modify_controlled",
    "waterbend_or_discard",
    "discard",
    "exile_if_dies",
    "add_subtype",
    "optional_payment",
    "airbend",
    "airbend_all",
    "opponents_cast_only_from_hand",
    "exile_top_card_play_this_turn",
    "sacrifice_source",
    "put_into_library_second_or_bottom",
];

/// Program kinds in numeric order, as for [`OPCODE_NAMES`].
const PROGRAM_KINDS: [&str; 6] = [
    "mana",
    "spell",
    "triggered",
    "activated",
    "static",
    "triggered_mana",
];

/// Instruction fields holding nested instruction lists.
const NESTED: [&str; 5] = ["then", "otherwise", "body", "if_paid", "if_declined"];

// ---------------------------------------------------------------------------
// Catalog: every typed definition plus the setup data.

#[derive(Clone)]
struct Catalog {
    /// Serialized typed definitions by semantic key.
    definitions: BTreeMap<String, Value>,
    setups: BTreeMap<String, Value>,
    token_templates: BTreeSet<String>,
    /// Reasons by semantic key for definitions marked incomplete.
    incomplete: BTreeMap<String, String>,
}

fn catalog() -> Result<&'static Catalog, IrError> {
    static CATALOG: OnceLock<Result<Catalog, IrError>> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            let typed = cardsets::catalog();
            let mut definitions = BTreeMap::new();
            let mut names = BTreeSet::new();
            for card in &typed.definitions {
                if !names.insert(card.registry_name) {
                    return malformed("catalog", format!("duplicate card {}", card.registry_name));
                }
                let value = serde_json::to_value(card).map_err(serialization)?;
                if definitions.insert(card.key.to_owned(), value).is_some() {
                    return malformed("catalog", format!("duplicate definition {}", card.key));
                }
            }
            Ok(Catalog {
                definitions,
                setups: serde_json::from_str(SETUPS).map_err(serialization)?,
                token_templates: typed
                    .token_templates
                    .iter()
                    .map(|k| k.to_string())
                    .collect(),
                incomplete: typed
                    .incomplete
                    .iter()
                    .map(|(key, reason)| (key.to_string(), reason.to_string()))
                    .collect(),
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

impl Catalog {
    /// Resolve every token/definition obligation of `roots`, sorted by key.
    fn closure(&self, roots: &BTreeSet<String>) -> Result<Vec<Value>, IrError> {
        let mut pending = roots.iter().cloned().collect::<Vec<_>>();
        let mut resolved = BTreeSet::new();
        while let Some(key) = pending.pop() {
            if resolved.contains(&key) {
                continue;
            }
            let definition = self
                .definitions
                .get(&key)
                .ok_or_else(|| IrError::Malformed(format!("missing content definition {key:?}")))?;
            references(definition, &mut pending);
            resolved.insert(key);
        }
        Ok(resolved
            .iter()
            .map(|key| self.definitions[key].clone())
            .collect())
    }

    fn source(&self, preset: &str) -> Result<Value, IrError> {
        let (mut source, roots) = match preset {
            "pool" => {
                let roots = self
                    .definitions
                    .iter()
                    .filter(|(_, row)| !is_token(row))
                    .map(|(key, _)| key.clone())
                    .collect::<BTreeSet<_>>();
                let source = json!({
                    "schema_version": SOURCE_SCHEMA_VERSION,
                    "pack_key": "tla-pool-v1",
                    "decks": [],
                    "pool": roots,
                    "token_templates": self.token_templates,
                });
                let roots = roots.union(&self.token_templates).cloned().collect();
                (source, roots)
            }
            _ => {
                let setup = self
                    .setups
                    .get(preset)
                    .ok_or_else(|| IrError::Malformed(format!("unknown preset {preset:?}")))?;
                let mut roots = BTreeSet::new();
                for deck in array(&setup["decks"], "setup.decks")? {
                    for field in ["cards", "sideboard"] {
                        for entry in deck
                            .get(field)
                            .and_then(Value::as_array)
                            .into_iter()
                            .flatten()
                        {
                            roots.insert(
                                string(&entry["definition"], "setup.definition")?.to_owned(),
                            );
                        }
                    }
                }
                let mut source = setup.clone();
                source["schema_version"] = json!(SOURCE_SCHEMA_VERSION);
                (source, roots)
            }
        };
        source["definitions"] = Value::Array(self.closure(&roots)?);
        Ok(source)
    }
}

fn references(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                match (key.as_str(), child) {
                    ("definition_ref", Value::String(reference)) => out.push(reference.clone()),
                    _ => references(child, out),
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|child| references(child, out)),
        _ => {}
    }
}

fn is_token(definition: &Value) -> bool {
    definition["characteristics"]["token"] == Value::Bool(true)
}

// ---------------------------------------------------------------------------
// Public entry points.

/// Every current preset: authored setups plus the whole-world `pool`.
pub fn presets() -> Result<Vec<String>, IrError> {
    let mut presets = catalog()?.setups.keys().cloned().collect::<Vec<_>>();
    presets.push("pool".to_owned());
    Ok(presets)
}

/// The compiled IR document for one current preset.
pub fn document(preset: &str) -> Result<&'static Value, IrError> {
    static DOCUMENTS: OnceLock<Result<BTreeMap<String, Value>, IrError>> = OnceLock::new();
    DOCUMENTS
        .get_or_init(|| {
            let catalog = catalog()?;
            presets()?
                .into_iter()
                .map(|preset| Ok((preset.clone(), compile_source(catalog.source(&preset)?)?)))
                .collect()
        })
        .as_ref()
        .map_err(Clone::clone)?
        .get(preset)
        .ok_or_else(|| IrError::Malformed(format!("unknown preset {preset:?}")))
}

/// The compiled IR document whose `pack_key` is `pack_key`.
pub fn document_for_pack(pack_key: &str) -> Result<&'static Value, IrError> {
    for preset in presets()? {
        let document = document(&preset)?;
        if document["pack_key"] == pack_key {
            return Ok(document);
        }
    }
    Err(IrError::Malformed(format!(
        "unknown content pack {pack_key:?}"
    )))
}

/// Reasons by semantic key for defined cards whose behavior is incomplete.
/// They run in the fixed setups that name them but are not supported for deck
/// entry; every other defined card is supported.
pub fn incomplete() -> Result<&'static BTreeMap<String, String>, IrError> {
    Ok(&catalog()?.incomplete)
}

/// Canonical compact JSON digest shared with the learning projection: sorted
/// keys, no whitespace, UTF-8 without ASCII escaping.
pub fn sha256(value: &Value) -> String {
    // serde_json's default map is ordered, so `to_string` is canonical.
    let encoded = serde_json::to_string(value).expect("JSON values serialize");
    Sha256::digest(encoded.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// Compilation of one source document.

/// Validate and lower one source document (definitions already resolved to
/// their closure) into deterministic IR.
fn compile_source(source: Value) -> Result<Value, IrError> {
    let pack_key = string(&source["pack_key"], "source.pack_key")?.to_owned();
    let mut source_by_key = BTreeMap::new();
    for definition in array(&source["definitions"], "source.definitions")? {
        let key = string(&definition["key"], "definition.key")?;
        if source_by_key.insert(key.to_owned(), definition).is_some() {
            return malformed("source.definitions", format!("duplicate definition {key}"));
        }
    }
    let indexes = source_by_key
        .keys()
        .enumerate()
        .map(|(index, key)| (key.clone(), index))
        .collect::<BTreeMap<_, _>>();

    let mut definitions = Vec::new();
    let mut programs: Vec<Value> = Vec::new();
    let mut registry_names = BTreeSet::new();
    for (definition_key, raw) in &source_by_key {
        let context = format!("definition[{definition_key}]");
        let registry_name = string(&raw["registry_name"], &context)?;
        if !registry_names.insert(registry_name) {
            return malformed(&context, format!("duplicate registry name {registry_name}"));
        }
        let mut raw_programs = array(&raw["programs"], &context)?
            .iter()
            .collect::<Vec<_>>();
        raw_programs.sort_by_key(|program| program["key"].as_str().unwrap_or_default());
        let mut program_indexes = Vec::new();
        let mut local_keys = BTreeSet::new();
        for raw_program in raw_programs {
            let local_key = string(&raw_program["key"], &context)?;
            if !local_keys.insert(local_key) {
                return malformed(&context, format!("duplicate program {local_key}"));
            }
            let program_key = format!("{definition_key}.{local_key}");
            let mut program = compile_program(raw_program, &program_key, &indexes)?;
            program["definition_index"] = json!(indexes[definition_key]);
            program["program_index"] = json!(programs.len());
            program_indexes.push(programs.len());
            programs.push(program);
        }
        let mut definition = json!({
            "characteristics": raw["characteristics"],
            "content_pack_binding": {"kind": "legacy_registry_name", "value": registry_name},
            "program_indexes": program_indexes,
            "semantic_index": indexes[definition_key],
            "semantic_key": definition_key,
        });
        if let Some(text_box) = raw.get("text_box") {
            definition["text_box"] = text_box.clone();
        }
        definitions.push(definition);
    }

    let (decks, deck_keys) = compile_decks(&source, &indexes, &source_by_key)?;
    let pool = string_set(&source, "pool")?;
    let tokens = string_set(&source, "token_templates")?;
    for key in &pool {
        match source_by_key.get(key) {
            Some(definition) if !is_token(definition) => {}
            _ => return malformed("source.pool", format!("not a deck definition: {key:?}")),
        }
    }
    for key in &tokens {
        match source_by_key.get(key) {
            Some(definition) if is_token(definition) => {}
            _ => return malformed("source.token_templates", format!("not a token: {key:?}")),
        }
    }

    // Every definition must be reachable from a deck, the pool, a token
    // template or an instruction; anything else is unreviewed dead content.
    let mut closure = deck_keys
        .iter()
        .chain(&pool)
        .chain(&tokens)
        .map(|key| indexes[key])
        .collect::<BTreeSet<_>>();
    for program in &programs {
        walk(&program["instructions"], &mut |instruction| {
            if let Some(index) = instruction.get("definition_index").and_then(Value::as_u64) {
                closure.insert(index as usize);
            }
        });
    }
    if closure.len() != definitions.len() {
        let absent = source_by_key
            .keys()
            .filter(|key| !closure.contains(&indexes[*key]))
            .cloned()
            .collect::<Vec<_>>();
        return malformed(
            "coverage",
            format!(
                "definitions outside admission closure: {}",
                absent.join(", ")
            ),
        );
    }

    let mut ir = json!({
        "definitions": definitions,
        "decks": decks,
        "pack_key": pack_key,
        "programs": programs,
        "schema_version": IR_SCHEMA_VERSION,
        "source_hash": sha256(&source),
        "setup_schema_version": 1,
    });
    ir["ir_hash"] = json!(sha256(&ir));
    Ok(ir)
}

fn compile_program(
    raw: &Value,
    program_key: &str,
    indexes: &BTreeMap<String, usize>,
) -> Result<Value, IrError> {
    let context = format!("program[{program_key}]");
    let kind = string(&raw["kind"], &context)?;
    let kind_number = PROGRAM_KINDS
        .iter()
        .position(|name| *name == kind)
        .ok_or_else(|| IrError::Malformed(format!("{context}: unknown program kind {kind:?}")))?
        + 1;
    let triggered = matches!(kind, "triggered" | "triggered_mana");
    if triggered != raw.get("trigger").is_some() {
        return malformed(&context, "exactly the triggered programs carry a trigger");
    }
    if matches!(kind, "activated" | "spell") != raw.get("cost").is_some() {
        return malformed(
            &context,
            "exactly the activated and spell programs carry a cost",
        );
    }
    if let Some(trigger) = raw.get("trigger") {
        check_trigger(trigger, &context)?;
    }

    let targets = raw
        .get("targets")
        .map(|targets| array(targets, &context).cloned())
        .transpose()?
        .unwrap_or_default();
    let mut roles = BTreeSet::new();
    for target in &targets {
        let role = string(&target["role"], &context)?;
        let (min, max) = (
            integer(&target["min"], &context)?,
            integer(&target["max"], &context)?,
        );
        if min < 0 || max < 1 || min > max {
            return malformed(&context, format!("invalid target bounds for {role}"));
        }
        if !roles.insert(role.to_owned()) {
            return malformed(&context, format!("duplicate target role {role}"));
        }
    }
    let instructions = compile_instructions(&raw["ops"], &format!("{context}.ops"), indexes)?;
    check_references(&instructions, &roles, &context, false)?;

    let mut program = json!({
        "instructions": instructions,
        "kind": kind_number,
        "kind_name": kind,
        "semantic_key": program_key,
        "targets": targets,
    });
    for field in ["target_binding", "cost", "trigger"] {
        if let Some(value) = raw.get(field) {
            program[field] = value.clone();
        }
    }
    Ok(program)
}

fn check_trigger(trigger: &Value, context: &str) -> Result<(), IrError> {
    let event = string(&trigger["event"], context)?;
    if matches!(event, "cast_spell" | "sacrificed") != trigger.get("predicate").is_some() {
        return malformed(
            context,
            "exactly cast_spell and sacrificed triggers carry a predicate",
        );
    }
    match (event, trigger.get("n")) {
        ("draw_nth_card", Some(n)) if integer(n, context)? >= 1 => Ok(()),
        ("draw_nth_card", _) => malformed(context, "draw_nth_card requires n >= 1"),
        (_, Some(_)) => malformed(context, "only draw_nth_card accepts n"),
        _ => Ok(()),
    }
}

fn compile_instructions(
    values: &Value,
    context: &str,
    indexes: &BTreeMap<String, usize>,
) -> Result<Vec<Value>, IrError> {
    let mut compiled = Vec::new();
    for (index, raw) in array(values, context)?.iter().enumerate() {
        let context = format!("{context}[{index}]");
        let Value::Object(fields) = raw else {
            return malformed(&context, "instruction must be an object");
        };
        let op = string(&raw["op"], &context)?;
        let opcode = OPCODE_NAMES
            .iter()
            .position(|name| *name == op)
            .ok_or_else(|| IrError::Malformed(format!("{context}: unknown opcode {op:?}")))?
            + 1;
        let mut lowered = Map::new();
        for (key, value) in fields {
            match key.as_str() {
                "op" => {}
                "definition_ref" => {
                    let reference = string(value, &context)?;
                    let index = indexes.get(reference).ok_or_else(|| {
                        IrError::Malformed(format!(
                            "{context}: unknown definition reference {reference:?}"
                        ))
                    })?;
                    lowered.insert("definition_index".into(), json!(index));
                }
                // Every nested instruction list crosses the same boundary.
                nested if NESTED.contains(&nested) => {
                    let body =
                        compile_instructions(value, &format!("{context}.{nested}"), indexes)?;
                    lowered.insert(key.clone(), Value::Array(body));
                }
                "modes" => {
                    let modes = array(value, &context)?;
                    if modes.is_empty() {
                        return malformed(&context, "modal requires modes");
                    }
                    let modes = modes
                        .iter()
                        .enumerate()
                        .map(|(i, mode)| {
                            compile_instructions(mode, &format!("{context}.modes[{i}]"), indexes)
                                .map(Value::Array)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    lowered.insert(key.clone(), Value::Array(modes));
                }
                _ => {
                    lowered.insert(key.clone(), value.clone());
                }
            }
        }
        lowered.insert("opcode".into(), json!(opcode));
        lowered.insert("op_name".into(), json!(op));
        compiled.push(Value::Object(lowered));
    }
    Ok(compiled)
}

/// Target references must name a declared role or a builtin; `current_target`
/// exists only inside a `for_each_target` / `target_clause` body.
fn check_references(
    instructions: &[Value],
    roles: &BTreeSet<String>,
    context: &str,
    current_target: bool,
) -> Result<(), IrError> {
    for (index, instruction) in instructions.iter().enumerate() {
        let context = format!("{context}[{index}]");
        for field in ["target", "sources"] {
            let Some(reference) = instruction.get(field).and_then(Value::as_str) else {
                continue;
            };
            let builtin = matches!(
                reference,
                "source" | "controller" | "triggering_spell" | "triggering_permanent"
            ) || (current_target && reference == "current_target")
                || reference.starts_with("each:");
            if !builtin && !roles.contains(reference) {
                return malformed(
                    &context,
                    format!("{field} references unknown target role {reference:?}"),
                );
            }
        }
        for mode in instruction
            .get("modes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            check_references(array(mode, &context)?, roles, &context, false)?;
        }
        if let Some(role) = instruction.get("role").and_then(Value::as_str) {
            if !roles.contains(role) {
                return malformed(
                    &context,
                    format!("role references unknown target role {role:?}"),
                );
            }
            check_references(
                array(&instruction["body"], &context)?,
                roles,
                &context,
                true,
            )?;
        }
        for field in ["then", "otherwise", "if_paid", "if_declined"] {
            if let Some(nested) = instruction.get(field).and_then(Value::as_array) {
                check_references(nested, roles, &format!("{context}.{field}"), current_target)?;
            }
        }
    }
    Ok(())
}

type Decks = (Vec<Value>, BTreeSet<String>);

fn compile_decks(
    source: &Value,
    indexes: &BTreeMap<String, usize>,
    source_by_key: &BTreeMap<String, &Value>,
) -> Result<Decks, IrError> {
    let mut decks = Vec::new();
    let mut deck_keys = BTreeSet::new();
    let mut seen = BTreeSet::new();
    for (deck_index, deck) in array(&source["decks"], "source.decks")?.iter().enumerate() {
        let context = format!("source.decks[{deck_index}]");
        let key = string(&deck["key"], &context)?;
        if !seen.insert(key) {
            return malformed(&context, format!("duplicate deck {key}"));
        }
        let card_count = integer(&deck["card_count"], &context)?;
        let mut compiled = json!({"card_count": card_count, "key": key});
        for field in ["cards", "sideboard"] {
            let mut cards = Vec::new();
            for entry in deck
                .get(field)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let definition = string(&entry["definition"], &context)?;
                let index = *indexes.get(definition).ok_or_else(|| {
                    IrError::Malformed(format!("{context}: unknown definition {definition:?}"))
                })?;
                if field == "sideboard" && is_token(source_by_key[definition]) {
                    return malformed(&context, "token cannot be a sideboard card");
                }
                let count = integer(&entry["count"], &context)?;
                if count < 1 {
                    return malformed(&context, "card counts must be positive");
                }
                deck_keys.insert(definition.to_owned());
                cards.push((index, count));
            }
            cards.sort();
            if field == "cards" && cards.iter().map(|(_, count)| count).sum::<i64>() != card_count {
                return malformed(
                    &context,
                    format!("entries do not sum to {card_count} cards"),
                );
            }
            if field == "sideboard" && cards.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return malformed(&context, "duplicate sideboard entry");
            }
            compiled[field] = cards
                .into_iter()
                .map(|(index, count)| json!({"count": count, "definition_index": index}))
                .collect();
        }
        decks.push(compiled);
    }
    decks.sort_by(|a, b| a["key"].as_str().cmp(&b["key"].as_str()));
    Ok((decks, deck_keys))
}

fn walk(instructions: &Value, visit: &mut dyn FnMut(&Value)) {
    for instruction in instructions.as_array().into_iter().flatten() {
        visit(instruction);
        for mode in instruction
            .get("modes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            walk(mode, visit);
        }
        for field in NESTED {
            if let Some(nested) = instruction.get(field) {
                walk(nested, visit);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Narrowing helpers.

fn malformed<T>(context: &str, message: impl std::fmt::Display) -> Result<T, IrError> {
    Err(IrError::Malformed(format!("{context}: {message}")))
}

fn serialization(error: serde_json::Error) -> IrError {
    IrError::Malformed(error.to_string())
}

fn array<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>, IrError> {
    value
        .as_array()
        .ok_or_else(|| IrError::Malformed(format!("{context}: must be an array")))
}

fn string<'a>(value: &'a Value, context: &str) -> Result<&'a str, IrError> {
    value
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| IrError::Malformed(format!("{context}: must be a non-empty string")))
}

fn integer(value: &Value, context: &str) -> Result<i64, IrError> {
    value
        .as_i64()
        .ok_or_else(|| IrError::Malformed(format!("{context}: must be an integer")))
}

fn string_set(source: &Value, field: &str) -> Result<BTreeSet<String>, IrError> {
    let mut keys = BTreeSet::new();
    for key in source
        .get(field)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let key = string(key, field)?;
        if !keys.insert(key.to_owned()) {
            return malformed(field, format!("duplicate entry {key}"));
        }
    }
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_compiles_and_hashes_its_own_document() {
        for preset in presets().unwrap() {
            let document = document(&preset).unwrap();
            let mut unhashed = document.clone();
            let declared = unhashed.as_object_mut().unwrap().remove("ir_hash").unwrap();
            assert_eq!(declared, json!(sha256(&unhashed)), "{preset}");
        }
    }

    #[test]
    fn unresolved_references_fail_closed() {
        let catalog = catalog().unwrap();
        let mut source = catalog.source("two_deck").unwrap();
        let definitions = source["definitions"].as_array_mut().unwrap();
        let with_target = definitions
            .iter_mut()
            .find(|d| d.to_string().contains("\"target\":\"source\""))
            .unwrap();
        let rewritten =
            with_target
                .to_string()
                .replacen("\"target\":\"source\"", "\"target\":\"nobody\"", 1);
        *with_target = serde_json::from_str(&rewritten).unwrap();
        let error = compile_source(source).unwrap_err().to_string();
        assert!(error.contains("unknown target role \"nobody\""), "{error}");
    }

    fn definition_named<'a>(catalog: &'a mut Catalog, name: &str) -> (String, &'a mut Value) {
        let (key, definition) = catalog
            .definitions
            .iter_mut()
            .find(|(_, definition)| definition["registry_name"] == name)
            .unwrap();
        (key.clone(), definition)
    }

    #[test]
    fn missing_token_fails_closed() {
        let mut catalog = catalog().unwrap().clone();
        let (ally, _) = definition_named(&mut catalog, "Ally");
        catalog.definitions.remove(&ally);
        let error = catalog.source("two_deck").unwrap_err().to_string();
        assert!(error.contains("missing content definition"), "{error}");
    }
}
