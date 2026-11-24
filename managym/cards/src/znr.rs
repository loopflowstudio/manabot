//! Zendikar Rising.
use managym_state::{
    ability::{
        Ability, Condition, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        AdditionalCost, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![nullpriest_of_oblivion()]
}

fn nullpriest_of_oblivion() -> CardDefinition {
    let text_box = "Kicker {3}{B} (You may pay an additional {3}{B} as you cast this spell.)\nLifelink\nMenace (This creature can't be blocked except by two or more creatures.)\nWhen this creature enters, if it was kicked, return target creature card from your graveyard to the battlefield.";
    CardDefinition {
        key: "znr.nullpriest_of_oblivion",
        scryfall_id: "086fc7fb-efcf-4676-8455-39b63edaec6a",
        printed: PrintedCharacteristics {
            name: "Nullpriest of Oblivion".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Cleric".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Lifelink, Keyword::Menace]),
            additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{3}{B}"))],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::Kicked),
                effects: vec![Effect::return_to_battlefield(
                    ObjectSelector::Target(TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate::creature(),
                    }),
                    false,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
