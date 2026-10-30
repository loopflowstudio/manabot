//! Khans of Tarkir.
use crate::state::ability::AffectedSpec;
use crate::state::effect::Modification;
use crate::state::{
    ability::{Ability, Effect, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![monastery_swiftspear()]
}

fn monastery_swiftspear() -> CardDefinition {
    let text_box = "Haste\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";

    CardDefinition {
        key: "ktk.monastery_swiftspear",
        scryfall_id: "b81c6c8b-a9cf-4866-89ba-7f8ad077b836",
        name: "Monastery Swiftspear".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Monk".into()],
        power: Some(1),
        toughness: Some(2),
        keywords: Keywords {
            haste: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::YouCastSpell {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::pump(1, 1),
            )],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
