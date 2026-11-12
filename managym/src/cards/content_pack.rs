//! The world's one immutable card catalog.
//!
//! A [`ContentPack`] maps card names and token keys to stable [`CardDefId`]s
//! and their [`ObjectDefinition`] rows. A process holds one ([`default_content_pack`]):
//! every card and token definition in `managym-cards` plus the rules fixtures, in
//! name order, with each double-faced card's back face registered right after its
//! front. Cards resolve by name; tokens, which may share a name, by `token.*` key. Every
//! match, authored setup and search fork runs on it. Its digest and manifest
//! name the world's cards for derived consumers (the world identity, learning
//! catalogs, checkpoints).

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use super::fixtures;
use super::names::{NameCandidate, NameVocabulary};
use crate::agent::observation::KeywordData;
use crate::state::{
    ability::{Ability, PaymentCost},
    card::{
        Card, CardDefId, CardDefinition, ObjectDefinition, PrintedCharacteristics, TokenDefinition,
    },
    game_object::{ObjectId, PlayerId},
};

pub const CONTENT_PACK_SCHEMA_VERSION: u32 = 1;

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
    pub definitions: Vec<ContentPackDefinitionManifest>,
}

#[derive(Clone)]
pub struct ContentPack {
    pub schema_version: u32,
    cards_by_name: BTreeMap<String, CardDefId>,
    tokens_by_key: BTreeMap<&'static str, CardDefId>,
    definitions: Vec<ObjectDefinition>,
    names: OnceLock<Arc<NameVocabulary>>,
    digest: OnceLock<String>,
    public_definitions: OnceLock<Vec<serde_json::Value>>,
}

static DEFAULT_CONTENT_PACK: OnceLock<ContentPack> = OnceLock::new();

/// The process-wide catalog every match, authored setup and search fork reads.
pub fn default_content_pack() -> &'static ContentPack {
    DEFAULT_CONTENT_PACK.get_or_init(ContentPack::default)
}

impl ContentPack {
    /// Sorted real names, including registered back faces.
    /// Built once with the catalog; decisions and search forks share the allocation.
    pub fn name_candidates(&self) -> Arc<NameVocabulary> {
        Arc::clone(self.names.get_or_init(|| {
            NameVocabulary::new(
                self.definition_entries()
                    .filter(|(id, card)| {
                        !card.is_token && (!card.key.is_empty() || self.front_face(*id).is_some())
                    })
                    .map(|(definition, card)| NameCandidate {
                        name: card.name.clone(),
                        definition: Some(definition),
                    })
                    .collect(),
            )
        }))
    }

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

    /// Read-only descriptor for binding derived consumers to the catalog.
    ///
    /// Registry names are used only at the adapter boundary. Runtime match
    /// facts and learning projections continue to carry typed definition IDs.
    pub fn manifest(&self) -> ContentPackManifest {
        ContentPackManifest {
            world_version: crate::WORLD_VERSION.into(),
            schema_version: self.schema_version,
            content_digest: self.content_digest(),
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

/// A freshly allocated catalog: every current card and token from
/// `managym-cards` and the rules fixtures, in name order (tokens sharing a
/// name follow key order). Matches share [`default_content_pack`]; a
/// separate allocation exists to show that match state identity ignores it.
impl Default for ContentPack {
    fn default() -> Self {
        let fixture = fixtures::crossroads_of_destiny();
        let mut rows = managym_cards::catalog()
            .iter()
            .chain([&fixture])
            .map(Row::Card)
            .chain(managym_cards::token_catalog().iter().map(Row::Token))
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| (left.name(), left.key()).cmp(&(right.name(), right.key())));
        let mut out = Self {
            schema_version: CONTENT_PACK_SCHEMA_VERSION,
            cards_by_name: BTreeMap::new(),
            tokens_by_key: BTreeMap::new(),
            names: OnceLock::new(),
            digest: OnceLock::new(),
            public_definitions: OnceLock::new(),
            definitions: Vec::with_capacity(rows.len()),
        };
        for row in rows {
            match row {
                Row::Card(card) => out.register_card(card.clone()),
                Row::Token(token) => out.register_token(token),
            }
        }
        out
    }
}

/// A definition the default catalog registers, before it is a row.
enum Row<'a> {
    Card(&'a CardDefinition),
    Token(&'a TokenDefinition),
}

impl Row<'_> {
    fn name(&self) -> &str {
        match self {
            Self::Card(card) => &card.name,
            Self::Token(token) => &token.name,
        }
    }

    fn key(&self) -> &'static str {
        match self {
            Self::Card(card) => card.key,
            Self::Token(token) => token.key,
        }
    }
}

impl ContentPack {
    /// Append `definition`, and then its back face, as the next rows.
    ///
    /// Registration is the catalog's one normalization of a definition: ward
    /// (CR 702.21) is a triggered ability, "Whenever this permanent becomes
    /// the target of a spell or ability an opponent controls, counter it
    /// unless that player pays [cost]", so a set file states ward once as a
    /// keyword and the catalog adds the trigger. Besides the catalog's own
    /// construction, only content-sensitivity tests extend a copy.
    pub fn register_card(&mut self, card: CardDefinition) {
        self.register(ObjectDefinition::card(&card));
        // A double-faced card's back face is its own row, right after its
        // front, so a transformed permanent observes and tokenizes as it.
        if let Some(back_face) = card.back_face {
            self.register(ObjectDefinition::face(*back_face));
        }
    }

    /// Append `token`'s object definition as the next row, found by its key.
    pub fn register_token(&mut self, token: &TokenDefinition) {
        self.register(ObjectDefinition::token(token));
    }

    /// Register a card face by name, or a token row (`is_token`) by key.
    fn register(&mut self, mut definition: ObjectDefinition) {
        self.names.take();
        self.digest.take();
        self.public_definitions.take();
        if let Some(ward_cost) = definition.ward().cloned() {
            Arc::make_mut(&mut definition.printed)
                .abilities
                .push(Ability::ward(ward_cost));
        }
        let definition_id = CardDefId(
            self.definitions
                .len()
                .try_into()
                .expect("content pack exceeds CardDefId capacity"),
        );
        if definition.is_token {
            assert!(
                self.tokens_by_key
                    .insert(definition.key, definition_id)
                    .is_none(),
                "duplicate token definition {}",
                definition.key
            );
        } else {
            assert!(
                self.cards_by_name
                    .insert(definition.name.clone(), definition_id)
                    .is_none(),
                "duplicate card definition {}",
                definition.name
            );
        }
        self.definitions.push(definition);
    }

    /// The back face's row of the double-faced card whose front is `front`.
    pub fn back_face(&self, front: CardDefId) -> Option<(CardDefId, Arc<PrintedCharacteristics>)> {
        self.definition(front)?.back_face.as_ref()?;
        let back = CardDefId(front.0 + 1);
        Some((back, Arc::clone(&self.definitions[back.0 as usize].printed)))
    }

    /// The front face whose back face is `back`; `None` for any other row.
    pub fn front_face(&self, back: CardDefId) -> Option<CardDefId> {
        let front = CardDefId(back.0.checked_sub(1)?);
        self.definition(front)?.back_face.as_ref()?;
        Some(front)
    }

    /// Iterate every registered card definition (conformance tests audit
    /// these against the Scryfall fixture).
    pub fn definitions(&self) -> impl Iterator<Item = &ObjectDefinition> {
        self.definitions.iter()
    }

    /// Stable `(CardDefId, definition)` pairs for replay, measurement, and
    /// content fingerprinting without inspecting the pack's storage layout.
    pub fn definition_entries(&self) -> impl Iterator<Item = (CardDefId, &ObjectDefinition)> {
        self.definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| (CardDefId(index as u32), definition))
    }

    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    /// The row of the card (or back face, or rules fixture) named `name`.
    /// Tokens are not found by name; see [`Self::token_id`].
    pub fn definition_id(&self, name: &str) -> Option<CardDefId> {
        self.cards_by_name.get(name).copied()
    }

    /// The row of the token definition keyed `key` (`token.<name>`).
    pub fn token_id(&self, key: &str) -> Option<CardDefId> {
        self.tokens_by_key.get(key).copied()
    }

    pub fn definition(&self, id: CardDefId) -> Option<&ObjectDefinition> {
        self.definitions.get(id.0 as usize)
    }

    /// Immutable public characteristics reused by every capture of this content
    /// identity. Rules programs stay native and are not reconstructed from text.
    /// Keywords take the observation's object shape (as card and permanent
    /// rows do), not the definition's own list, so the capture contract
    /// (`DECISION_CAPTURE_VERSION`) does not follow the content encoding.
    pub fn public_definition(&self, id: CardDefId) -> Option<&serde_json::Value> {
        self.public_definitions
            .get_or_init(|| {
                self.definitions()
                    .map(|card| {
                        serde_json::json!({
                            "name": card.name, "text_box": card.text_box,
                            "mana_cost": card.mana_cost, "types": card.types,
                            "supertypes": card.supertypes, "subtypes": card.subtypes,
                            "keywords": KeywordData::from_keywords(&card.keywords),
                            "power": card.power,
                            "toughness": card.toughness, "kicker": card.optional_cost(),
                            "ward": card.ward().map(PaymentCost::mana_cost),
                            "is_token": card.is_token,
                        })
                    })
                    .collect()
            })
            .get(id.0 as usize)
    }

    /// The printed name of the token keyed `key`, for descriptions; the key
    /// itself when it names no token.
    pub fn token_name<'a>(&'a self, key: &'a str) -> &'a str {
        self.token_id(key)
            .and_then(|id| self.definition(id))
            .map_or(key, |token| token.name.as_str())
    }

    /// `card`'s current copiable values in the row shape, with its token
    /// status and its shown face's back face. Records of a copy (pending
    /// creations, canonical state) serialize this; the token a copy creates
    /// takes only `printed`.
    pub fn object_row(&self, card: &Card) -> ObjectDefinition {
        ObjectDefinition {
            key: "",
            scryfall_id: None,
            printed: card.copiable_definition(),
            is_token: card.is_token,
            back_face: self
                .definition(card.definition_id)
                .and_then(|row| row.back_face.clone()),
        }
    }

    /// A new card object of the card named `name`.
    pub fn instantiate(&self, name: &str, owner: PlayerId, object_id: ObjectId) -> Option<Card> {
        self.instantiate_row(self.definition_id(name)?, owner, object_id)
    }

    /// A new token object of the token definition keyed `key`.
    pub fn instantiate_token(
        &self,
        key: &str,
        owner: PlayerId,
        object_id: ObjectId,
    ) -> Option<Card> {
        self.instantiate_row(self.token_id(key)?, owner, object_id)
    }

    fn instantiate_row(
        &self,
        definition_id: CardDefId,
        owner: PlayerId,
        object_id: ObjectId,
    ) -> Option<Card> {
        let row = self.definitions.get(definition_id.0 as usize)?;
        Some(Card::from_definition(
            object_id,
            owner,
            definition_id,
            Arc::clone(&row.printed),
            row.is_token,
        ))
    }
}
