//! Kaldheim Commander.
use managym_state::{
    ability::{Ability, Actor, Effect, EffectValue, Per, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, TapCost,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![lathril_blade_of_the_elves()]
}

fn lathril_blade_of_the_elves() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nWhenever Lathril deals combat damage to a player, create that many 1/1 green Elf Warrior creature tokens.\n{T}, Tap ten untapped Elves you control: Each opponent loses 10 life and you gain 10 life.";
    CardDefinition {
        key: "khc.lathril_blade_of_the_elves",
        scryfall_id: "547888c3-a9a6-4413-b29a-6bcd8a9279bf",
        printed: PrintedCharacteristics {
            name: "Lathril, Blade of the Elves".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Elf".into(), "Noble".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Menace]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::This,
                    to: Some(Actor::AnyPlayer),
                    per: Per::Object,
                },
                intervening_if: None,
                effects: vec![Effect::create_tokens(
                    "token.elf_warrior_1_1_g",
                    EffectValue::TriggeringAmount,
                )],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                tap_cost: Some(TapCost {
                    predicate: CardPredicate::subtype("Elf"),
                    count: 10,
                }),
                effects: vec![Effect::DrainLife {
                    amount: EffectValue::Fixed(10),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
