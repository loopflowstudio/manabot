use std::collections::BTreeMap;

use crate::state::{
    ability::{Effect, TargetSpec},
    card::{
        basic_land, ActivatedAbilityDefinition, Card, CardDefinition, CardType, CardTypes,
        Keywords, ManaAbility,
    },
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
        self.register_visions();
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

    fn register_creature(
        &mut self,
        name: &str,
        mana_cost: &str,
        subtypes: &[&str],
        power: i32,
        toughness: i32,
        keywords: Keywords,
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

        self.register_creature("Grey Ogre", "2R", &["Ogre"], 2, 2, Keywords::default());

        self.register_card(CardDefinition {
            name: "Lightning Bolt".to_string(),
            mana_cost: Some(ManaCost::parse("R")),
            types: CardTypes::new([CardType::Instant]),
            spell_effect: Some(Effect::DealDamage {
                amount: 3,
                target: TargetSpec::CreatureOrPlayer,
            }),
            text_box: "Lightning Bolt deals 3 damage to any target.".to_string(),
            ..Default::default()
        });

        self.register_card(CardDefinition {
            name: "Counterspell".to_string(),
            mana_cost: Some(ManaCost::parse("UU")),
            types: CardTypes::new([CardType::Instant]),
            spell_effect: Some(Effect::CounterSpell {
                target: TargetSpec::Spell,
            }),
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
        );
        self.register_creature(
            "Raging Goblin",
            "R",
            &["Goblin"],
            1,
            1,
            Keywords {
                haste: true,
                ..Default::default()
            },
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
        );
        self.register_creature("Craw Wurm", "4GG", &["Wurm"], 6, 4, Keywords::default());
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
                effect: Effect::ModifyUntilEot {
                    power_delta: 1,
                    toughness_delta: 0,
                },
            }],
            power: Some(5),
            toughness: Some(5),
            ..Default::default()
        });
    }
}
