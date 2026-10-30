//! The immutable card catalog a match runs against.
//!
//! A [`ContentPack`] maps registry names to stable [`CardDefId`]s and shared
//! [`CardDefinition`]s. The default pack holds every current card; authored
//! setups and pools build keyed packs from chosen definitions
//! ([`ContentPack::from_definitions`]). Packs carry a digest and manifest so
//! derived consumers bind to exactly one content identity.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use super::fixtures;
use crate::state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{Card, CardDefId, CardDefinition},
    game_object::{ObjectId, PlayerId},
};

pub const CONTENT_PACK_SCHEMA_VERSION: u32 = 1;

/// The default pack's historical prefix, by printed name.
const LEGACY_ORDER: [&str; 29] = [
    "Plains",
    "Island",
    "Swamp",
    "Mountain",
    "Forest",
    "Llanowar Elves",
    "Gray Ogre",
    "Lightning Bolt",
    "Ancestral Recall",
    "Counterspell",
    "Wind Drake",
    "Giant Spider",
    "Raging Goblin",
    "Serra Angel",
    "Typhoid Rats",
    "War Mammoth",
    "Wall of Stone",
    "Boggart Brute",
    "Youthful Knight",
    "Fencing Ace",
    "Healer's Hawk",
    "Craw Wurm",
    "Shivan Dragon",
    "Pyroclasm",
    "Man-o'-War",
    "Pop Quiz",
    "Igneous Inspiration",
    "Divide by Zero",
    "Waterfall Aerialist",
];

/// Registered after every other typed definition, by printed name.
const TRAILING_ORDER: [&str; 2] = ["Crossroads of Destiny", "Proft's Eidetic Memory"];

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ContentPackDefinitionManifest {
    pub card_def_id: CardDefId,
    pub registry_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ContentPackManifest {
    pub world_version: String,
    pub schema_version: u32,
    pub content_digest: String,
    /// The authored setup or pool this pack was built for; `None` for the
    /// default pack.
    pub pack_key: Option<String>,
    pub definitions: Vec<ContentPackDefinitionManifest>,
}

#[derive(Clone, Debug)]
pub struct ContentPack {
    pub schema_version: u32,
    pack_key: Option<String>,
    cards_by_name: BTreeMap<String, CardDefId>,
    definitions: Vec<Arc<CardDefinition>>,
    digest: OnceLock<String>,
    public_definitions: OnceLock<Vec<serde_json::Value>>,
}

/// Legacy Rust name retained for conformance callers. Match state owns a
/// shared `Arc<ContentPack>` rather than cloning this value.
pub type CardRegistry = ContentPack;

/// Every current card from `cardsets/` and the rules fixtures, in default-pack
/// order. [`LEGACY_ORDER`] keeps the historical prefix (and so its `CardDefId`s)
/// that older evidence records; the remaining cards follow by name, then
/// [`TRAILING_ORDER`].
fn default_order() -> Vec<CardDefinition> {
    let mut cards = crate::cardsets::catalog()
        .iter()
        .cloned()
        .chain([fixtures::crossroads_of_destiny()])
        .map(|definition| (definition.name.clone(), definition))
        .collect::<BTreeMap<_, _>>();
    let mut take = |name: &str| {
        cards
            .remove(name)
            .unwrap_or_else(|| panic!("ordered card {name} has no definition"))
    };
    let legacy = LEGACY_ORDER.map(&mut take);
    let trailing = TRAILING_ORDER.map(&mut take);
    legacy
        .into_iter()
        .chain(cards.into_values())
        .chain(trailing)
        .collect()
}

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
        self.digest
            .get_or_init(|| {
                let mut definitions: Vec<_> = self.definitions().collect();
                definitions.sort_by(|left, right| left.name.cmp(&right.name));
                let bytes =
                    serde_json::to_vec(&definitions).expect("card definitions are serializable");
                blake3::hash(&bytes).to_hex().to_string()
            })
            .clone()
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
            pack_key: self.pack_key.clone(),
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
        Self::from_definitions(None, default_order())
    }
}

impl ContentPack {
    /// A pack of exactly `definitions`, in order.
    pub(crate) fn from_definitions(
        pack_key: Option<String>,
        definitions: Vec<CardDefinition>,
    ) -> Self {
        let mut out = Self {
            schema_version: CONTENT_PACK_SCHEMA_VERSION,
            pack_key,
            cards_by_name: BTreeMap::new(),
            digest: OnceLock::new(),
            public_definitions: OnceLock::new(),
            definitions: Vec::with_capacity(definitions.len()),
        };
        for definition in definitions {
            out.register_card(definition);
        }
        out
    }

    /// The authored setup or pool this pack was built for.
    pub fn pack_key(&self) -> Option<&str> {
        self.pack_key.as_deref()
    }

    pub fn register_card(&mut self, mut definition: CardDefinition) {
        self.digest.take();
        self.public_definitions.take();
        // Ward (CR 702.21) is a triggered ability: "Whenever this permanent
        // becomes the target of a spell or ability an opponent controls,
        // counter it unless that player pays [cost]."
        if let Some(ward_cost) = definition.ward.clone() {
            definition.abilities.push(Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::BecomesTargeted {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CounterUnlessPays { cost: ward_cost }],
            });
        }
        let definition_id = CardDefId(
            self.definitions
                .len()
                .try_into()
                .expect("content pack exceeds CardDefId capacity"),
        );
        let name = definition.name.clone();
        assert!(
            self.cards_by_name.insert(name, definition_id).is_none(),
            "duplicate card definition"
        );
        self.definitions.push(Arc::new(definition));
    }

    /// Iterate every registered card definition (conformance tests audit
    /// these against the Scryfall fixture).
    pub fn definitions(&self) -> impl Iterator<Item = &CardDefinition> {
        self.definitions.iter().map(Arc::as_ref)
    }

    /// Stable `(CardDefId, definition)` pairs for replay, measurement, and
    /// content fingerprinting without inspecting the pack's storage layout.
    pub fn definition_entries(&self) -> impl Iterator<Item = (CardDefId, &CardDefinition)> {
        self.definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| (CardDefId(index as u32), definition.as_ref()))
    }

    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    pub fn definition_id(&self, name: &str) -> Option<CardDefId> {
        self.cards_by_name.get(name).copied()
    }

    pub fn definition(&self, id: CardDefId) -> Option<&CardDefinition> {
        self.definitions.get(id.0 as usize).map(Arc::as_ref)
    }

    /// Immutable public characteristics reused by every capture of this content
    /// identity. Rules programs stay native and are not reconstructed from text.
    pub fn public_definition(&self, id: CardDefId) -> Option<&serde_json::Value> {
        self.public_definitions
            .get_or_init(|| {
                self.definitions()
                    .map(|card| {
                        serde_json::json!({
                            "name": card.name, "text_box": card.text_box,
                            "mana_cost": card.mana_cost, "types": card.types,
                            "supertypes": card.supertypes, "subtypes": card.subtypes,
                            "keywords": card.keywords, "power": card.power,
                            "toughness": card.toughness, "kicker": card.kicker,
                            "ward": card.ward, "is_token": card.is_token,
                        })
                    })
                    .collect()
            })
            .get(id.0 as usize)
    }

    pub fn instantiate(&self, name: &str, owner: PlayerId, object_id: ObjectId) -> Option<Card> {
        let definition_id = self.definition_id(name)?;
        let definition = Arc::clone(self.definitions.get(definition_id.0 as usize)?);
        Some(Card::from_definition(
            object_id,
            owner,
            definition_id,
            definition,
        ))
    }
}
