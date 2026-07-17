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
    }

    pub fn register_card(&mut self, mut definition: CardDefinition) {
        // Ward (CR 702.21) is a triggered ability: "Whenever this permanent
        // becomes the target of a spell an opponent controls, counter it
        // unless that player pays [cost]."
        if let Some(ward_cost) = definition.ward.clone() {
            definition.abilities.push(Ability::Triggered {
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

    fn register_basic_lands(&mut self) {
        self.register_card(basic_land("Plains", Color::White));
        self.register_card(basic_land("Island", Color::Blue));
        self.register_card(basic_land("Swamp", Color::Black));
        self.register_card(basic_land("Mountain", Color::Red));
        self.register_card(basic_land("Forest", Color::Green));
    }

    #[allow(clippy::too_many_arguments)]
    fn register_creature(
        &mut self,
        name: &str,
        mana_cost: &str,
        subtypes: &[&str],
        power: i32,
        toughness: i32,
        keywords: Keywords,
        text_box: &str,
    ) {
        self.register_card(CardDefinition {
            name: name.to_string(),
            mana_cost: Some(ManaCost::parse(mana_cost)),
            types: CardTypes::new([CardType::Creature]),
            subtypes: subtypes
                .iter()
                .map(|subtype| (*subtype).to_string())
                .collect(),
            keywords,
            text_box: text_box.to_string(),
            power: Some(power),
            toughness: Some(toughness),
            ..Default::default()
        });
    }

    fn register_alpha(&mut self) {
        self.register_card(CardDefinition {
            name: "Llanowar Elves".to_string(),
            mana_cost: Some(ManaCost::parse("G")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".to_string(), "Druid".to_string()],
            mana_abilities: vec![ManaAbility {
                mana: Mana::single(Color::Green),
            }],
            text_box: "{T}: Add {G}.".to_string(),
            power: Some(1),
            toughness: Some(1),
            ..Default::default()
        });

        self.register_creature("Gray Ogre", "2R", &["Ogre"], 2, 2, Keywords::default(), "");

        self.register_card(CardDefinition {
            name: "Lightning Bolt".to_string(),
            mana_cost: Some(ManaCost::parse("R")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::DealDamage {
                amount: 3,
                target: TargetSpec::CreatureOrPlayer,
            }],
            text_box: "Lightning Bolt deals 3 damage to any target.".to_string(),
            ..Default::default()
        });

        // DEVIATION (documented in wave/rules/01-two-deck-slice.md): oracle
        // is "Target player draws three cards." — the engine has no
        // target-player draw, so the caster always draws (the self-target
        // case; the opponent-draw line is not available).
        self.register_card(CardDefinition {
            name: "Ancestral Recall".to_string(),
            mana_cost: Some(ManaCost::parse("U")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::DrawCards { count: 3 }],
            text_box: "Target player draws three cards.".to_string(),
            ..Default::default()
        });

        self.register_card(CardDefinition {
            name: "Counterspell".to_string(),
            mana_cost: Some(ManaCost::parse("UU")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::CounterSpell {
                target: TargetSpec::Spell,
            }],
            text_box: "Counter target spell.".to_string(),
            ..Default::default()
        });

        self.register_creature(
            "Wind Drake",
            "2U",
            &["Drake"],
            2,
            2,
            Keywords {
                flying: true,
                ..Default::default()
            },
            "Flying",
        );
        self.register_creature(
            "Giant Spider",
            "3G",
            &["Spider"],
            2,
            4,
            Keywords {
                reach: true,
                ..Default::default()
            },
            "Reach (This creature can block creatures with flying.)",
        );
        self.register_creature(
            "Raging Goblin",
            "R",
            &["Goblin", "Berserker"],
            1,
            1,
            Keywords {
                haste: true,
                ..Default::default()
            },
            "Haste (This creature can attack and {T} as soon as it comes under your control.)",
        );
        self.register_creature(
            "Serra Angel",
            "3WW",
            &["Angel"],
            4,
            4,
            Keywords {
                flying: true,
                vigilance: true,
                ..Default::default()
            },
            "Flying\nVigilance (Attacking doesn't cause this creature to tap.)",
        );
        self.register_creature(
            "Typhoid Rats",
            "B",
            &["Rat"],
            1,
            1,
            Keywords {
                deathtouch: true,
                ..Default::default()
            },
            "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)",
        );
        self.register_creature(
            "War Mammoth",
            "3G",
            &["Elephant"],
            3,
            3,
            Keywords {
                trample: true,
                ..Default::default()
            },
            "Trample",
        );
        self.register_creature(
            "Wall of Stone",
            "1RR",
            &["Wall"],
            0,
            8,
            Keywords {
                defender: true,
                ..Default::default()
            },
            "Defender (This creature can't attack.)",
        );
        self.register_creature(
            "Boggart Brute",
            "2R",
            &["Goblin", "Warrior"],
            3,
            2,
            Keywords {
                menace: true,
                ..Default::default()
            },
            "Menace (This creature can't be blocked except by two or more creatures.)",
        );
        self.register_creature(
            "Youthful Knight",
            "1W",
            &["Human", "Knight"],
            2,
            1,
            Keywords {
                first_strike: true,
                ..Default::default()
            },
            "First strike",
        );
        self.register_creature(
            "Fencing Ace",
            "1W",
            &["Human", "Soldier"],
            1,
            1,
            Keywords {
                double_strike: true,
                ..Default::default()
            },
            "Double strike (This creature deals both first-strike and regular combat damage.)",
        );
        self.register_creature(
            "Healer's Hawk",
            "W",
            &["Bird"],
            1,
            1,
            Keywords {
                flying: true,
                lifelink: true,
                ..Default::default()
            },
            "Flying\nLifelink (Damage dealt by this creature also causes you to gain that much life.)",
        );
        self.register_creature("Craw Wurm", "4GG", &["Wurm"], 6, 4, Keywords::default(), "");
        self.register_card(CardDefinition {
            name: "Shivan Dragon".to_string(),
            mana_cost: Some(ManaCost::parse("4RR")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".to_string()],
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("R"),
                sacrifice_source: false,
                waterbend: false,
                effect: Effect::ModifyUntilEot {
                    power_delta: 1,
                    toughness_delta: 0,
                },
            }],
            text_box: "Flying\n{R}: This creature gets +1/+0 until end of turn.".to_string(),
            power: Some(5),
            toughness: Some(5),
            ..Default::default()
        });
    }
}
