//! Return to Ravnica.
use crate::state::{
    card::{CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![fencing_ace()]
}

fn fencing_ace() -> CardDefinition {
    let text_box =
        "Double strike (This creature deals both first-strike and regular combat damage.)";

    CardDefinition {
        key: "rtr.fencing_ace",
        scryfall_id: "a42d3066-f4ec-4d28-83ab-e48141206c72",
        name: "Fencing Ace".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            double_strike: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}
