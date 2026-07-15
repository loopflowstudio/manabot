use std::collections::{BTreeMap, BTreeSet};

use managym::{
    cardsets::alpha::ContentPack,
    state::{
        card::{CardDefinition, CardType},
        mana::{Color, ManaCost},
    },
};
use serde_json::Value;

const TWO_DECK_IR: &str = include_str!("../../content/semantic/v1/generated/two_deck.ir.json");

fn walk_instructions<'a>(instructions: &'a [Value], out: &mut Vec<&'a Value>) {
    for instruction in instructions {
        out.push(instruction);
        for field in ["then", "otherwise", "body"] {
            if let Some(nested) = instruction.get(field).and_then(Value::as_array) {
                walk_instructions(nested, out);
            }
        }
    }
}

fn json_strings(value: Option<&Value>) -> BTreeSet<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|item| item.as_str().expect("string array item").to_owned())
        .collect()
}

fn type_names(definition: &CardDefinition) -> BTreeSet<String> {
    definition
        .types
        .types
        .iter()
        .map(|card_type| match card_type {
            CardType::Artifact => "artifact",
            CardType::Battle => "battle",
            CardType::Creature => "creature",
            CardType::Enchantment => "enchantment",
            CardType::Instant => "instant",
            CardType::Kindred => "kindred",
            CardType::Land => "land",
            CardType::Planeswalker => "planeswalker",
            CardType::Sorcery => "sorcery",
        })
        .map(str::to_owned)
        .collect()
}

fn keyword_names(definition: &CardDefinition) -> BTreeSet<String> {
    [
        ("deathtouch", definition.keywords.deathtouch),
        ("defender", definition.keywords.defender),
        ("double_strike", definition.keywords.double_strike),
        ("first_strike", definition.keywords.first_strike),
        ("flash", definition.keywords.flash),
        ("flying", definition.keywords.flying),
        ("haste", definition.keywords.haste),
        ("hexproof", definition.keywords.hexproof),
        ("lifelink", definition.keywords.lifelink),
        ("menace", definition.keywords.menace),
        ("reach", definition.keywords.reach),
        ("trample", definition.keywords.trample),
        ("vigilance", definition.keywords.vigilance),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then(|| name.to_owned()))
    .collect()
}

fn color_names(definition: &CardDefinition) -> BTreeSet<String> {
    definition
        .colors()
        .into_iter()
        .map(|color| match color {
            Color::White => "W",
            Color::Blue => "U",
            Color::Black => "B",
            Color::Red => "R",
            Color::Green => "G",
            Color::Colorless => "C",
            Color::Generic => panic!("generic is not a card color"),
        })
        .map(str::to_owned)
        .collect()
}

#[test]
fn semantic_definitions_bind_once_to_real_content_pack_ids() {
    let ir: Value = serde_json::from_str(TWO_DECK_IR).expect("valid checked-in semantic IR");
    assert_eq!(ir["schema_version"], 1);
    let definitions = ir["definitions"].as_array().expect("definitions array");
    let pack = ContentPack::default();
    let mut bindings = BTreeMap::new();
    let mut bound_ids = BTreeSet::new();

    for (expected_index, definition) in definitions.iter().enumerate() {
        assert_eq!(
            definition["semantic_index"].as_u64(),
            Some(expected_index as u64)
        );
        assert_eq!(
            definition["content_pack_binding"]["kind"].as_str(),
            Some("legacy_registry_name")
        );
        let semantic_key = definition["semantic_key"].as_str().expect("semantic key");
        let registry_name = definition["content_pack_binding"]["value"]
            .as_str()
            .expect("legacy registry name");
        let card_def_id = pack
            .definition_id(registry_name)
            .unwrap_or_else(|| panic!("{semantic_key} does not bind to ContentPack"));
        assert_eq!(
            pack.definition(card_def_id).expect("bound definition").name,
            registry_name
        );
        assert!(bindings.insert(expected_index, card_def_id).is_none());
        assert!(
            bound_ids.insert(card_def_id),
            "duplicate binding for {registry_name}"
        );
    }

    assert_eq!(bindings.len(), definitions.len());
}

#[test]
fn semantic_characteristics_match_the_bound_content_pack_definitions() {
    let ir: Value = serde_json::from_str(TWO_DECK_IR).expect("valid checked-in semantic IR");
    let pack = ContentPack::default();

    for semantic_definition in ir["definitions"].as_array().expect("definitions array") {
        let registry_name = semantic_definition["content_pack_binding"]["value"]
            .as_str()
            .expect("legacy registry name");
        let card_def_id = pack
            .definition_id(registry_name)
            .unwrap_or_else(|| panic!("{registry_name} does not bind to ContentPack"));
        let definition = pack.definition(card_def_id).expect("bound definition");
        let characteristics = &semantic_definition["characteristics"];

        assert_eq!(
            type_names(definition),
            json_strings(characteristics.get("types")),
            "{registry_name} types"
        );
        assert_eq!(
            definition
                .supertypes
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>(),
            json_strings(characteristics.get("supertypes")),
            "{registry_name} supertypes"
        );
        assert_eq!(
            definition.subtypes.iter().cloned().collect::<BTreeSet<_>>(),
            json_strings(characteristics.get("subtypes")),
            "{registry_name} subtypes"
        );
        assert_eq!(
            keyword_names(definition),
            json_strings(characteristics.get("keywords")),
            "{registry_name} keywords"
        );
        assert_eq!(
            definition.power.map(i64::from),
            characteristics["power"].as_i64(),
            "{registry_name} power"
        );
        assert_eq!(
            definition.toughness.map(i64::from),
            characteristics["toughness"].as_i64(),
            "{registry_name} toughness"
        );
        assert_eq!(
            definition.is_token,
            characteristics["token"].as_bool().unwrap_or(false),
            "{registry_name} token status"
        );

        let expected_mana_cost = characteristics["mana_cost"].as_str().map(ManaCost::parse);
        assert_eq!(
            definition.mana_cost, expected_mana_cost,
            "{registry_name} mana cost"
        );
        if characteristics.get("colors").is_some() {
            assert_eq!(
                color_names(definition),
                json_strings(characteristics.get("colors")),
                "{registry_name} colors"
            );
        }
    }
}

#[test]
fn executable_ir_has_no_card_name_dispatch_or_unresolved_definition_refs() {
    let ir: Value = serde_json::from_str(TWO_DECK_IR).expect("valid checked-in semantic IR");
    let definition_count = ir["definitions"].as_array().unwrap().len() as u64;
    let programs = ir["programs"].as_array().expect("programs array");
    let mut instructions = Vec::new();
    for program in programs {
        assert!(program["definition_index"].as_u64().unwrap() < definition_count);
        walk_instructions(
            program["instructions"].as_array().unwrap(),
            &mut instructions,
        );
    }

    assert!(!instructions.is_empty());
    for instruction in instructions {
        let object = instruction.as_object().expect("instruction object");
        assert!(instruction["opcode"].is_u64());
        assert!(!object.contains_key("card_name"));
        assert!(!object.contains_key("registry_name"));
        assert!(!object.contains_key("definition_ref"));
        if let Some(definition_index) = instruction.get("definition_index") {
            assert!(definition_index.as_u64().unwrap() < definition_count);
        }
    }
}
