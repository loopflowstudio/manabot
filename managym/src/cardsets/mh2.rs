use crate::state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![tireless_provisioner()]
}

fn tireless_provisioner() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, create a Food token or a Treasure token. (Food is an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\" Treasure is an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";

    CardDefinition {
        key: "mh2.tireless_provisioner",
        scryfall_id: "5c5be54d-660e-42ab-b5ea-5e1cf3bad0bc",
        name: "Tireless Provisioner".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Elf".into(), "Scout".into()],
        power: Some(3),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::CreateToken {
                        token_name: "Food".into(),
                        count: 1,
                        tapped_and_attacking: false,
                        attach_source: false,
                    }],
                    vec![Effect::CreateToken {
                        token_name: "Treasure".into(),
                        count: 1,
                        tapped_and_attacking: false,
                        attach_source: false,
                    }],
                ],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
