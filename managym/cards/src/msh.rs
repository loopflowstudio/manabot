//! Marvel Super Heroes.
use managym_state::{
    card::{
        AdditionalCost, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![ironheart_clever_champion()]
}

fn ironheart_clever_champion() -> CardDefinition {
    let text_box = "Improvise (Your artifacts can help cast this spell. Each artifact you tap after you're done activating mana abilities pays for {1}.)\nFlying\nNoncreature spells you cast have improvise.";
    CardDefinition {
        key: "msh.ironheart_clever_champion",
        scryfall_id: "395e477a-861f-4661-b329-6c1ad5343ed5",
        printed: PrintedCharacteristics {
            name: "Ironheart, Clever Champion".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Hero".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            additional_costs: vec![AdditionalCost::Improvise],
            statics: vec![StaticAbility::SpellsHave {
                predicate: CardPredicate::noncreature(),
                cost: AdditionalCost::Improvise,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
