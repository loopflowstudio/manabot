//! Amonkhet.
use managym_state::{
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![vizier_of_the_menagerie()]
}

fn vizier_of_the_menagerie() -> CardDefinition {
    CardDefinition {
        key: "akh.vizier_of_the_menagerie",
        scryfall_id: "ca204351-7a7e-4e4b-8c2b-f90fa0f9d724",
        printed: PrintedCharacteristics {
            name: "Vizier of the Menagerie".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Snake".into(), "Cleric".into()],
            power: Some(3),
            toughness: Some(4),
            statics: vec![
                StaticAbility::LookAtLibraryTop,
                StaticAbility::CastFromLibraryTop(CardPredicate::creature()),
                StaticAbility::SpendAnyManaToCast(CardPredicate::creature()),
            ],
            text_box: "You may look at the top card of your library any time.\nYou may cast creature spells from the top of your library.\nYou can spend mana of any type to cast creature spells.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
