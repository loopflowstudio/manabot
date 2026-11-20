//! Magic Origins.
use std::sync::Arc;

use managym_state::{
    ability::{Ability, Effect, EffectValue, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    mana::{Color, Mana, ManaCost, SpendEffect},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![boggart_brute(), elemental_bond(), pyromancers_goggles()]
}

fn boggart_brute() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)";

    CardDefinition {
        key: "ori.boggart_brute",
        scryfall_id: "9d735ebf-61a4-4507-9399-6d32c8903ded",
        printed: PrintedCharacteristics {
            name: "Boggart Brute".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(3),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Menace]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn elemental_bond() -> CardDefinition {
    let text_box = "Whenever a creature you control with power 3 or greater enters, draw a card.";
    CardDefinition {
        key: "ori.elemental_bond",
        scryfall_id: "554a8769-c840-4c9d-9959-b075c174457b",
        printed: PrintedCharacteristics {
            name: "Elemental Bond".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another_you_control(CardPredicate {
                        min_power: Some(3),
                        ..CardPredicate::creature()
                    }),
                },
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pyromancers_goggles() -> CardDefinition {
    let text_box = "{T}: Add {R}. When that mana is spent to cast a red instant or sorcery spell, copy that spell and you may choose new targets for the copy.";
    let effect = SpendEffect::CopySpell {
        spell: Arc::new(CardPredicate {
            colors_any: vec![Color::Red],
            card_types_any: vec![CardType::Instant, CardType::Sorcery],
            ..Default::default()
        }),
    };
    CardDefinition {
        key: "ori.pyromancers_goggles",
        scryfall_id: "1163ce9f-cf22-422e-a4b5-0240b88e2816",
        printed: PrintedCharacteristics {
            name: "Pyromancer's Goggles".into(),
            mana_cost: Some(ManaCost::parse("{5}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec!["Legendary".into()],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::of_class(
                    effect.into(),
                    [0, 0, 0, 1, 0, 0],
                ))],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
