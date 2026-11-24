//! Magic 2011.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, ActivationZone, CardDefinition, CardType, CardTypes,
        PrintedCharacteristics,
    },
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        CardDefinition {
            key: "m11.ajanis_pridemate",
            scryfall_id: "f70d1452-6b61-4c63-841f-4256ac498e9f",
            printed: PrintedCharacteristics {
                name: "Ajani's Pridemate".into(),
                mana_cost: Some(ManaCost::parse("{1}{W}")),
                types: CardTypes::new([CardType::Creature]),
                subtypes: vec!["Cat".into(), "Soldier".into()],
                power: Some(2),
                toughness: Some(2),
                abilities: vec![Ability::when_you_gain_life(Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                ))],
                text_box: "Whenever you gain life, put a +1/+1 counter on this creature.".into(),
                ..Default::default()
            },
            ..Default::default()
        },
        reassembling_skeleton(),
    ]
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
