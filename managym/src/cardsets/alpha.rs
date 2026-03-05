use std::collections::BTreeMap;

use crate::state::{
    card::{basic_land, Card, CardDefinition, CardType, CardTypes, ManaAbility},
    game_object::{IdGenerator, ObjectId, PlayerId},
    mana::{Color, Mana, ManaCost},
};

#[derive(Clone, Debug)]
struct RegisteredCard {
    registry_key: ObjectId,
    definition: CardDefinition,
}

#[derive(Clone, Debug)]
pub struct CardRegistry {
    cards: BTreeMap<String, RegisteredCard>,
    registry_key_gen: IdGenerator,
}

impl Default for CardRegistry {
    fn default() -> Self {
        let mut out = Self {
            cards: BTreeMap::new(),
            registry_key_gen: IdGenerator::default(),
        };
        out.register_all_cards();
        out
    }
}

impl CardRegistry {
    pub fn register_all_cards(&mut self) {
        self.register_basic_lands();
        self.register_alpha();
    }

    pub fn register_card(&mut self, definition: CardDefinition) {
        let registry_key = self.registry_key_gen.next_id();
        let name = definition.name.clone();
        self.cards.insert(
            name,
            RegisteredCard {
                registry_key,
                definition,
            },
        );
    }

    pub fn instantiate(&self, name: &str, owner: PlayerId, object_id: ObjectId) -> Option<Card> {
        let registered = self.cards.get(name)?;
        Some(Card::from_definition(
            object_id,
            owner,
            registered.registry_key,
            &registered.definition,
        ))
    }

    fn register_basic_lands(&mut self) {
        self.register_card(basic_land("Plains", Color::White));
        self.register_card(basic_land("Island", Color::Blue));
        self.register_card(basic_land("Swamp", Color::Black));
        self.register_card(basic_land("Mountain", Color::Red));
        self.register_card(basic_land("Forest", Color::Green));
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

        self.register_card(CardDefinition {
            name: "Grey Ogre".to_string(),
            mana_cost: Some(ManaCost::parse("2R")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Ogre".to_string()],
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        });

        self.register_card(CardDefinition {
            name: "Lightning Bolt".to_string(),
            mana_cost: Some(ManaCost::parse("R")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Lightning Bolt deals 3 damage to any target.".to_string(),
            ..Default::default()
        });

        self.register_card(CardDefinition {
            name: "Counterspell".to_string(),
            mana_cost: Some(ManaCost::parse("UU")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Counter target spell.".to_string(),
            ..Default::default()
        });
    }
}
