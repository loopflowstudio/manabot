use crate::state::{
    ability::{Ability, Effect, TargetSpec, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![corrupt_court_official()]
}

fn corrupt_court_official() -> CardDefinition {
    let text_box = "When this creature enters, target opponent discards a card.";

    CardDefinition {
        key: "ptk.corrupt_court_official",
        scryfall_id: "9d3ba2e3-e680-47cd-81c5-555deea7d00f",
        name: "Corrupt Court Official".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Advisor".into()],
        power: Some(1),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DiscardTargetPlayer {
                count: 1,
                target: TargetSpec::Opponent,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
