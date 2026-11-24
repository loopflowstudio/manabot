//! Zendikar Rising.
use managym_state::{
    ability::{
        Ability, Condition, Effect, ObjectSelector, PaymentCost, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, ManaAbility, PrintedCharacteristics,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![nullpriest_of_oblivion(), crawling_barrens()]
}

fn crawling_barrens() -> CardDefinition {
    let text_box = "{T}: Add {C}.\n{4}: Put two +1/+1 counters on this land. Then you may have it become a 0/0 Elemental creature until end of turn. It's still a land.";
    CardDefinition {
        key: "znr.crawling_barrens",
        scryfall_id: "7bd0e025-7a75-4641-a51a-27df9dcde05f",
        printed: PrintedCharacteristics {
            name: "Crawling Barrens".into(),
            types: CardTypes::new([CardType::Land]),
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("C"),
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{4}"),
                effects: vec![
                    Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne, 2),
                    Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::default()),
                        if_paid: [
                            Modification::BecomeCreature,
                            Modification::base_pt(Some(0), Some(0)),
                            Modification::AddSubtypes(vec!["Elemental".into()]),
                        ]
                        .into_iter()
                        .map(|modification| {
                            Effect::until_end_of_turn(ObjectSelector::This, modification)
                        })
                        .collect(),
                        if_declined: vec![],
                    },
                ],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
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
