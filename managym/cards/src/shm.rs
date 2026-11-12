//! Shadowmoor.
use managym_state::{
    ability::{Effect, ObjectSelector},
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![rhys_the_redeemed()]
}

fn rhys_the_redeemed() -> CardDefinition {
    let text_box = "{2}{G/W}, {T}: Create a 1/1 green and white Elf Warrior creature token.\n{4}{G/W}{G/W}, {T}: For each creature token you control, create a token that's a copy of that creature.";
    let creature_token = CardPredicate {
        token: Some(true),
        ..CardPredicate::creature()
    };
    CardDefinition {
        key: "shm.rhys_the_redeemed",
        scryfall_id: "59327a58-cd41-479c-81c7-31c9d3b29508",
        name: "Rhys the Redeemed".into(),
        mana_cost: Some(ManaCost::parse("{G/W}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Elf".into(), "Warrior".into()],
        power: Some(1),
        toughness: Some(1),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}{G/W}"),
                tap_source: true,
                effects: vec![Effect::create_tokens("Elf Warrior", 1)],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{4}{G/W}{G/W}"),
                tap_source: true,
                effects: vec![Effect::CreateTokenCopy {
                    of: ObjectSelector::each_you_control(creature_token),
                    exceptions: vec![],
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
