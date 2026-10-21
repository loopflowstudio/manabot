use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use crate::state::{
    ability::{Ability, Effect, TargetSpec, TriggerCondition, TriggerSubject},
    card::{
        basic_land, ActivatedAbilityDefinition, Card, CardDefId, CardDefinition, CardType,
        CardTypes, Keywords, ManaAbility,
    },
    game_object::{ObjectId, PlayerId},
    mana::{Color, Mana, ManaCost},
};

pub const CONTENT_PACK_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ContentPackDefinitionManifest {
    pub card_def_id: CardDefId,
    pub registry_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CompiledSemanticManifest {
    pub pack_key: String,
    pub ir_hash: String,
    pub source_hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ContentPackManifest {
    pub world_version: String,
    pub schema_version: u32,
    pub content_digest: String,
    pub compiled_semantics: Option<CompiledSemanticManifest>,
    pub definitions: Vec<ContentPackDefinitionManifest>,
}

#[derive(Clone, Debug)]
pub struct ContentPack {
    pub schema_version: u32,
    compiled_semantics: Option<CompiledSemanticManifest>,
    cards_by_name: BTreeMap<String, CardDefId>,
    definitions: Vec<Arc<CardDefinition>>,
}

/// Legacy Rust name retained for conformance callers. Match state owns a
/// shared `Arc<ContentPack>` rather than cloning this value.
pub type CardRegistry = ContentPack;

static DEFAULT_CONTENT_PACK: OnceLock<Arc<ContentPack>> = OnceLock::new();

/// Process-wide immutable pack used by ordinary matches and every search fork.
pub fn default_content_pack() -> Arc<ContentPack> {
    Arc::clone(DEFAULT_CONTENT_PACK.get_or_init(|| Arc::new(ContentPack::default())))
}

impl ContentPack {
    /// Representation-neutral fingerprint of the immutable card definitions.
    ///
    /// Sorting by card name keeps benchmark snapshots stable across storage
    /// changes and preserves the contract-v1 digest across W2-179's move to
    /// shared definitions.
    pub fn content_digest(&self) -> String {
        let mut definitions: Vec<_> = self.definitions().collect();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        let bytes = serde_json::to_vec(&definitions).expect("card definitions are serializable");
        blake3::hash(&bytes).to_hex().to_string()
    }

    /// Read-only descriptor for binding derived consumers to this exact pack.
    ///
    /// Registry names are used only at the adapter boundary. Runtime match
    /// facts and learning projections continue to carry typed definition IDs.
    pub fn manifest(&self) -> ContentPackManifest {
        ContentPackManifest {
            world_version: crate::WORLD_VERSION.into(),
            schema_version: self.schema_version,
            content_digest: self.content_digest(),
            compiled_semantics: self.compiled_semantics.clone(),
            definitions: self
                .definition_entries()
                .map(|(card_def_id, definition)| ContentPackDefinitionManifest {
                    card_def_id,
                    registry_name: definition.name.clone(),
                })
                .collect(),
        }
    }
}

impl Default for ContentPack {
    fn default() -> Self {
        let mut out = Self {
            schema_version: CONTENT_PACK_SCHEMA_VERSION,
            compiled_semantics: None,
            cards_by_name: BTreeMap::new(),
            definitions: Vec::new(),
        };
        out.register_all_cards();
        out
    }
}

impl ContentPack {
    pub(crate) fn from_compiled_semantics(
        definitions: Vec<CardDefinition>,
        compiled_semantics: CompiledSemanticManifest,
    ) -> Self {
        let mut out = Self {
            schema_version: CONTENT_PACK_SCHEMA_VERSION,
            compiled_semantics: Some(compiled_semantics),
            cards_by_name: BTreeMap::new(),
            definitions: Vec::with_capacity(definitions.len()),
        };
        for definition in definitions {
            out.register_card(definition);
        }
        out
    }

    pub fn compiled_semantics(&self) -> Option<&CompiledSemanticManifest> {
        self.compiled_semantics.as_ref()
    }

    pub fn register_all_cards(&mut self) {
        self.register_basic_lands();
        self.register_alpha();
        self.register_ice_age();
        self.register_visions();
        self.register_strixhaven();
        self.register_tla();
        self.register_tla_lessons_increment();
        self.register_mkm();
        self.register_elemental();
        self.register_sacrifice();
        self.register_temur();
        // Append research vanilla definitions without renumbering existing cards.
        self.register_creature(
            "Savannah Lions",
            "W",
            &["Cat"],
            2,
            1,
            Keywords::default(),
            "",
        );
        self.register_creature(
            "Glory Seeker",
            "1W",
            &["Human", "Soldier"],
            2,
            2,
            Keywords::default(),
            "",
        );
        self.register_creature(
            "Hill Giant",
            "3R",
            &["Giant"],
            3,
            3,
            Keywords::default(),
            "",
        );
