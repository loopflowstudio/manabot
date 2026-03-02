use std::collections::BTreeMap;

use crate::state::{
    card::{basic_land, Card, CardDefinition, CardType, CardTypes, ManaAbility},
    game_object::{IdGenerator, ObjectId, PlayerId},
    mana::{Color, Mana, ManaCost},
};

#[derive(Clone, Debug)]
pub struct CardRegistry {
    cards: BTreeMap<String, CardDefinition>,
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

    pub fn register_card(&mut self, mut card: CardDefinition) {
        card.registry_key = self.registry_key_gen.next_id();
        self.cards.insert(card.name.clone(), card);
    }

    pub fn instantiate(&self, name: &str, owner: PlayerId, object_id: ObjectId) -> Option<Card> {
        let definition = self.cards.get(name)?;
        Some(Card::from_definition(object_id, owner, definition))
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
            registry_key: ObjectId(0),
            name: "Llanowar Elves".to_string(),
            mana_cost: Some(ManaCost::parse("G")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec![],
            subtypes: vec!["Elf".to_string(), "Druid".to_string()],
            mana_abilities: vec![ManaAbility {
                mana: Mana::single(Color::Green),
            }],
            text_box: "{T}: Add {G}.".to_string(),
            power: Some(1),
            toughness: Some(1),
        });

        self.register_card(CardDefinition {
            registry_key: ObjectId(0),
            name: "Grey Ogre".to_string(),
            mana_cost: Some(ManaCost::parse("2R")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec![],
            subtypes: vec!["Ogre".to_string()],
            mana_abilities: vec![],
            text_box: "".to_string(),
            power: Some(2),
            toughness: Some(2),
        });
    }
}
