//! The world's one immutable card catalog.
//!
//! A [`ContentPack`] maps card names and token and emblem keys to stable [`CardDefId`]s
//! and their [`ObjectDefinition`] rows. A process holds one ([`default_content_pack`]):
//! every card, token and emblem definition in `managym-cards` plus the rules fixtures, in
//! name order, with each double-faced card's back face registered right after its
//! front. Cards resolve by name; tokens and emblems resolve by their definition keys. Every
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
        CardDefId, CardDefinition, CastOption, EmblemDefinition, ObjectDefinition,
        PrintedCharacteristics, SpellProfile, TokenDefinition,
    },
    game_object::{
        CopiableValues, Object, ObjectKind, ObjectLki, PlayerId, PublicId, TransformingFaces,
    },
};

pub const CONTENT_PACK_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ContentPackDefinitionManifest {
    pub card_def_id: CardDefId,
    /// The card's or token's key; `None` for a back face or rules fixture.
    /// With the name it identifies the row: two tokens may share a name.
    pub key: Option<String>,
    pub registry_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ContentPackManifest {
    pub schema_version: u32,
    pub content_digest: String,
    pub definitions: Vec<ContentPackDefinitionManifest>,
}

#[derive(Clone)]
pub struct ContentPack {
    pub schema_version: u32,
    cards_by_name: BTreeMap<String, CardDefId>,
    emblems_by_key: BTreeMap<&'static str, CardDefId>,
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

/// The printed face used to announce, pay for or stack `object` under `profile`.
/// Reads the front or inset face directly without touching the global catalog
/// unless `profile` selects a modal back face.
#[inline]
pub(crate) fn spell_face(
    object: &Object,
    profile: SpellProfile,
) -> Option<&Arc<PrintedCharacteristics>> {
    if profile == SpellProfile::BackFace && object.modal_back {
        default_content_pack().spell_definition(object, profile)
    } else {
        object.front_or_inset(profile)
    }
}

impl ContentPack {
    /// Sorted real names, including registered back faces.
    /// Built once with the catalog; decisions and search forks share the allocation.
    pub fn name_candidates(&self) -> Arc<NameVocabulary> {
        Arc::clone(self.names.get_or_init(|| {
            let mut names = BTreeMap::new();
            for (id, card) in self.definition_entries() {
                if card.kind != ObjectKind::Card
                    || (card.scryfall_id.is_none()
                        && card.key.is_empty()
                        && self.front_face(id).is_none())
                {
                    continue;
                }
                names.insert(card.name.clone(), Some(id));
                if let Some(inset) = &card.inset {
                    names
                        .entry(inset.characteristics().name.clone())
                        .or_insert(None);
                }
            }
            NameVocabulary::new(
                names
                    .into_iter()
                    .map(|(name, definition)| NameCandidate { name, definition })
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
    /// Keys and registry names are used only at the adapter boundary. Runtime
    /// match facts and learning projections continue to carry typed
    /// definition IDs.
    pub fn manifest(&self) -> ContentPackManifest {
        ContentPackManifest {
            schema_version: self.schema_version,
            content_digest: self.content_digest(),
            definitions: self
                .definition_entries()
                .map(|(card_def_id, definition)| ContentPackDefinitionManifest {
                    card_def_id,
                    key: (!definition.key.is_empty()).then(|| definition.key.to_owned()),
                    registry_name: definition.registry_name().into(),
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
        let fixtures = [
            fixtures::crossroads_of_destiny(),
            fixtures::tidal_errand(),
            fixtures::kelp_hollow(),
            fixtures::thornwood_arbor(),
        ];
        let mut rows = managym_cards::catalog()
            .iter()
            .chain(&fixtures)
            .map(Row::Card)
            .chain(managym_cards::token_catalog().iter().map(Row::Token))
            .chain(managym_cards::emblem_catalog().iter().map(Row::Emblem))
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| (left.name(), left.key()).cmp(&(right.name(), right.key())));
        let mut out = Self {
            schema_version: CONTENT_PACK_SCHEMA_VERSION,
            cards_by_name: BTreeMap::new(),
            tokens_by_key: BTreeMap::new(),
            emblems_by_key: BTreeMap::new(),
            names: OnceLock::new(),
            digest: OnceLock::new(),
            public_definitions: OnceLock::new(),
            definitions: Vec::with_capacity(rows.len()),
        };
        for row in rows {
            match row {
                Row::Card(card) => out.register_card(card.clone()),
                Row::Token(token) => out.register_token(token),
                Row::Emblem(emblem) => out.register(ObjectDefinition {
                    key: emblem.key,
                    label: Some(emblem.label),
                    scryfall_id: None,
                    printed: Arc::new(emblem.printed.clone()),
                    kind: ObjectKind::Emblem,
                    back_face: None,
                }),
            }
        }
        out.register(managym_cards::ragnarok());
        out
    }
}

/// A definition the default catalog registers, before it is a row.
enum Row<'a> {
    Card(&'a CardDefinition),
    Token(&'a TokenDefinition),
    Emblem(&'a EmblemDefinition),
}

impl Row<'_> {
    fn name(&self) -> &str {
        match self {
            Self::Card(card) => &card.name,
            Self::Token(token) => &token.name,
            Self::Emblem(emblem) => emblem.label,
        }
    }

    fn key(&self) -> &'static str {
        match self {
            Self::Card(card) => card.key,
            Self::Token(token) => token.key,
            Self::Emblem(emblem) => emblem.key,
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
        if definition.kind != ObjectKind::Card {
            let keys = if definition.kind == ObjectKind::Emblem {
                &mut self.emblems_by_key
            } else {
                &mut self.tokens_by_key
            };
            assert!(
                keys.insert(definition.key, definition_id).is_none(),
                "duplicate object definition {}",
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

    pub(crate) fn back_face_row(&self, front: CardDefId) -> Option<(CardDefId, &ObjectDefinition)> {
        self.definition(front)?.back_face.as_ref()?;
        let back = CardDefId(front.0 + 1);
        Some((back, &self.definitions[back.0 as usize]))
    }

    /// Both modes of a double-faced card use their registered face rows.
    pub fn spell_definition<'a>(
        &'a self,
        object: &'a Object,
        profile: SpellProfile,
    ) -> Option<&'a Arc<PrintedCharacteristics>> {
        if profile == SpellProfile::BackFace && object.modal_back {
            return self
                .back_face_row(object.definition_id)
                .map(|(_, row)| &row.printed);
        }
        object.front_or_inset(profile)
    }

    /// The printed spell a departed card was cast as, from the base it
    /// retains (`definition`, `printing`): its inset Adventure or Omen, or
    /// its back face. The base already is that spell when it is the back
    /// face itself or a spell copy's own printing.
    pub fn recorded_spell<'a>(
        &'a self,
        definition: CardDefId,
        printing: &'a PrintedCharacteristics,
        profile: SpellProfile,
    ) -> &'a PrintedCharacteristics {
        match profile {
            SpellProfile::Front => printing,
            SpellProfile::Adventure | SpellProfile::Omen => printing
                .inset_definition(profile)
                .map_or(printing, |spell| &**spell),
            SpellProfile::BackFace => self
                .back_face_row(definition)
                .map_or(printing, |(_, back)| &back.printed),
        }
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
                        let mut public = serde_json::json!({
                            "name": card.registry_name(), "text_box": card.text_box,
                            "mana_cost": card.mana_cost, "types": card.types,
                            "supertypes": card.supertypes, "subtypes": card.subtypes,
                            "keywords": KeywordData::from_keywords(&card.keywords),
                            "power": card.power,
                            "toughness": card.toughness, "kicker": card.optional_cost(),
                            "ward": card.ward().and_then(PaymentCost::mana),
                            "is_token": card.kind == ObjectKind::Token,
                        });
                        if let Some(loyalty) = card.loyalty {
                            public["printed_loyalty"] = loyalty.into();
                        }
                        // A ward paid in life has no mana cost to state; the
                        // key is absent on every other definition.
                        if let Some(PaymentCost::Life(life)) = card.ward() {
                            public["ward_life"] = (*life).into();
                        }
                        // A cost choice other than kicker names itself, so
                        // a reader of `kicked` says "bargained"; the key is
                        // absent on every other definition.
                        if let Some(option) = card
                            .cast_option()
                            .filter(|option| *option != CastOption::Kicker)
                        {
                            let (taken, cost) = option.words();
                            public["cast_option"] =
                                serde_json::json!({ "taken": taken, "cost": cost });
                        }
                        public
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

    /// The raw faces `object` transforms between (CR 712.9, 712.10): a
    /// double-faced card's (including modal) or a transforming token's (CR 707.8a).
    /// A copy of one face on any other backing, a melded permanent and a
    /// face-down permanent have none. Physical backing decides, never the
    /// catalog row of the face it shows.
    pub fn transforming_faces(&self, object: &Object) -> Option<TransformingFaces> {
        match object.form {
            Some(_) => object.form_faces().cloned(),
            None => self.catalog_faces(object.definition_id, object.copiable_definition()),
        }
    }

    /// The front-face-up faces of the catalog double-faced card `front`
    /// showing `printed`: what an object with no battlefield form, live or
    /// departed, transforms between.
    fn catalog_faces(
        &self,
        front: CardDefId,
        printed: Arc<PrintedCharacteristics>,
    ) -> Option<TransformingFaces> {
        let (back, row) = self.back_face_row(front)?;
        Some(TransformingFaces {
            front: (front, printed),
            back: (back, Arc::clone(&row.printed)),
            back_up: false,
        })
    }

    /// `card`'s current copiable values in the row shape, with its token
    /// status and, on a transforming object's front face, its back face.
    /// Records of a copy (pending creations, canonical state) serialize this;
    /// the token a copy creates takes only `printed`.
    pub fn object_row(&self, card: &Object) -> ObjectDefinition {
        ObjectDefinition {
            key: "",
            label: None,
            scryfall_id: None,
            printed: card.copiable_definition(),
            kind: card.kind(),
            back_face: self
                .transforming_faces(card)
                .filter(|faces| !faces.back_up)
                .map(|faces| faces.back.1.name.clone()),
        }
    }

    pub(crate) fn copiable_values(&self, object: &Object) -> CopiableValues {
        CopiableValues {
            definition_id: object.definition_id.catalog_id(),
            definition: self.object_row(object),
            faces: self.transforming_faces(object),
            copy: object.copy().cloned(),
        }
    }

    /// A departed ordinary card needs only its catalog id. Retain a snapshot
    /// when the catalog cannot reconstruct its kind, face, or copy exceptions.
    pub(crate) fn retain_copiable_values(&self, object: &Object) -> Option<Arc<CopiableValues>> {
        if object.copy().is_none()
            && object.form_faces().is_none()
            && self.definition(object.definition_id).is_some_and(|row| {
                row.kind == object.kind() && &row.printed == object.copiable_face()
            })
        {
            return None;
        }
        Some(Arc::new(self.copiable_values(object)))
    }

    /// The copiable values `lki`'s object had as it last existed (CR 608.2h).
    /// Without a retained snapshot it was an ordinary catalog appearance, so
    /// a front-face-up transforming card keeps its faces (CR 707.8a).
    pub(crate) fn departed_copiable_values(&self, lki: &ObjectLki) -> CopiableValues {
        if let Some(values) = lki.copiable.as_deref() {
            return values.clone();
        }
        let definition = self
            .definition(lki.definition_id)
            .expect("catalog copy source")
            .clone();
        CopiableValues {
            definition_id: Some(lki.definition_id),
            faces: self.catalog_faces(lki.definition_id, Arc::clone(&definition.printed)),
            definition,
            copy: None,
        }
    }

    /// A new card object of the card named `name`.
    pub fn instantiate(&self, name: &str, owner: PlayerId, object_id: PublicId) -> Option<Object> {
        self.instantiate_row(self.definition_id(name)?, owner, object_id)
    }

    /// A new token object of the token definition keyed `key`.
    pub fn instantiate_token(
        &self,
        key: &str,
        owner: PlayerId,
        object_id: PublicId,
    ) -> Option<Object> {
        self.instantiate_row(self.token_id(key)?, owner, object_id)
    }

    pub fn emblem_id(&self, key: &str) -> Option<CardDefId> {
        self.emblems_by_key.get(key).copied()
    }

    pub fn instantiate_emblem(&self, key: &str, owner: PlayerId, id: PublicId) -> Option<Object> {
        self.instantiate_row(self.emblem_id(key)?, owner, id)
    }

    fn instantiate_row(
        &self,
        definition_id: CardDefId,
        owner: PlayerId,
        object_id: PublicId,
    ) -> Option<Object> {
        let row = self.definitions.get(definition_id.0 as usize)?;
        Some(Object::from_definition(
            object_id,
            owner,
            definition_id,
            Arc::clone(&row.printed),
            row.kind,
        ))
    }
}
