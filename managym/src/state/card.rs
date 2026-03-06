use std::collections::BTreeSet;

use super::{
    ability::Ability,
    game_object::{ObjectId, PlayerId},
    mana::{Color, Colors, Mana, ManaCost},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CardType {
    Creature,
    Instant,
    Sorcery,
    Planeswalker,
    Land,
    Enchantment,
    Artifact,
    Kindred,
    Battle,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CardTypes {
    pub types: BTreeSet<CardType>,
}

impl CardTypes {
    pub fn new(types: impl IntoIterator<Item = CardType>) -> Self {
        Self {
            types: types.into_iter().collect(),
        }
    }

    pub fn is_castable(&self) -> bool {
        !self.is_land() && !self.types.is_empty()
    }

    pub fn is_permanent(&self) -> bool {
        self.is_creature()
            || self.is_land()
            || self.is_artifact()
            || self.is_enchantment()
            || self.is_planeswalker()
            || self.is_battle()
    }

    pub fn is_non_land_permanent(&self) -> bool {
        self.is_permanent() && !self.is_land()
    }

    pub fn is_non_creature_permanent(&self) -> bool {
        self.is_permanent() && !self.is_creature()
    }

    pub fn is_spell(&self) -> bool {
        self.types.contains(&CardType::Instant) || self.types.contains(&CardType::Sorcery)
    }

    pub fn is_instant(&self) -> bool {
        self.types.contains(&CardType::Instant)
    }

    pub fn is_instant_speed(&self) -> bool {
        self.is_instant()
    }

    pub fn is_creature(&self) -> bool {
        self.types.contains(&CardType::Creature)
    }

    pub fn is_land(&self) -> bool {
        self.types.contains(&CardType::Land)
    }

    pub fn is_planeswalker(&self) -> bool {
        self.types.contains(&CardType::Planeswalker)
    }

    pub fn is_enchantment(&self) -> bool {
        self.types.contains(&CardType::Enchantment)
    }

    pub fn is_artifact(&self) -> bool {
        self.types.contains(&CardType::Artifact)
    }

    pub fn is_kindred(&self) -> bool {
        self.types.contains(&CardType::Kindred)
    }

    pub fn is_battle(&self) -> bool {
        self.types.contains(&CardType::Battle)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManaAbility {
    pub mana: Mana,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Keywords {
    pub flying: bool,
    pub reach: bool,
    pub haste: bool,
    pub vigilance: bool,
    pub trample: bool,
    pub first_strike: bool,
    pub double_strike: bool,
    pub deathtouch: bool,
    pub lifelink: bool,
    pub defender: bool,
    pub menace: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActivatedAbilityEffect {
    SelfGetsUntilEot {
        power_delta: i32,
        toughness_delta: i32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivatedAbilityDefinition {
    pub mana_cost: ManaCost,
    pub effect: ActivatedAbilityEffect,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CardDefinition {
    pub name: String,
    pub mana_cost: Option<ManaCost>,
    pub types: CardTypes,
    pub supertypes: Vec<String>,
    pub subtypes: Vec<String>,
    pub abilities: Vec<Ability>,
    pub mana_abilities: Vec<ManaAbility>,
    pub activated_abilities: Vec<ActivatedAbilityDefinition>,
    pub keywords: Keywords,
    pub text_box: String,
    pub power: Option<i32>,
    pub toughness: Option<i32>,
}

impl CardDefinition {
    pub fn colors(&self) -> Colors {
        self.mana_cost
            .as_ref()
            .map(|m| m.colors())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    pub id: ObjectId,
    pub registry_key: ObjectId,
    pub name: String,
    pub mana_cost: Option<ManaCost>,
    pub colors: Colors,
    pub types: CardTypes,
    pub supertypes: Vec<String>,
    pub subtypes: Vec<String>,
    pub abilities: Vec<Ability>,
    pub mana_abilities: Vec<ManaAbility>,
    pub activated_abilities: Vec<ActivatedAbilityDefinition>,
    pub keywords: Keywords,
    pub text_box: String,
    pub power: Option<i32>,
    pub toughness: Option<i32>,
    pub owner: PlayerId,
}

impl Card {
    pub fn from_definition(
        id: ObjectId,
        owner: PlayerId,
        registry_key: ObjectId,
        definition: &CardDefinition,
    ) -> Self {
        Self {
            id,
            registry_key,
            name: definition.name.clone(),
            mana_cost: definition.mana_cost.clone(),
            colors: definition.colors(),
            types: definition.types.clone(),
            supertypes: definition.supertypes.clone(),
            subtypes: definition.subtypes.clone(),
            abilities: definition.abilities.clone(),
            mana_abilities: definition.mana_abilities.clone(),
            activated_abilities: definition.activated_abilities.clone(),
            keywords: definition.keywords.clone(),
            text_box: definition.text_box.clone(),
            power: definition.power,
            toughness: definition.toughness,
            owner,
        }
    }
}

impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{name: {}}}", self.name)
    }
}

pub fn basic_land(name: &str, color: Color) -> CardDefinition {
    CardDefinition {
        name: name.to_string(),
        types: CardTypes::new([CardType::Land]),
        supertypes: vec!["basic".to_string()],
        subtypes: vec![name.to_string()],
        abilities: vec![],
        mana_abilities: vec![ManaAbility {
            mana: Mana::single(color),
        }],
        text_box: format!("{{T}}: Add {{{}}}.", color.symbol()),
        ..Default::default()
    }
}
