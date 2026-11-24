//! Magic 2011.
use managym_state::{
    ability::{Effect, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, ActivationZone, CardDefinition, CardType, CardTypes,
        PrintedCharacteristics,
    },
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![reassembling_skeleton()]
}

fn reassembling_skeleton() -> CardDefinition {
    let rules_text = "{1}{B}: Return this card from your graveyard to the battlefield tapped.";
    CardDefinition {
        key: "m11.reassembling_skeleton",
        scryfall_id: "a2ec6120-2de0-4166-be85-2d773fa7ebbb",
        printed: PrintedCharacteristics {
            name: "Reassembling Skeleton".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Skeleton".into(), "Warrior".into()],
            power: Some(1),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                zone: ActivationZone::Graveyard,
                mana_cost: ManaCost::parse("{1}{B}"),
                effects: vec![Effect::return_to_battlefield(ObjectSelector::This, true)],
                ..Default::default()
            }],
            text_box: rules_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
