//! Amonkhet.
use managym_state::{
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PowerCda, PrintedCharacteristics,
        StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![enigma_drake()]
}

fn enigma_drake() -> CardDefinition {
    let text_box = "Flying\nEnigma Drake's power is equal to the number of instant and sorcery cards in your graveyard.";
    CardDefinition {
        key: "akh.enigma_drake",
        scryfall_id: "66286631-c16e-410c-b963-25cfe8005d8f",
        printed: PrintedCharacteristics {
            name: "Enigma Drake".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Drake".into()],
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::CharacteristicPower(
                PowerCda::GraveyardMatching(Box::new(CardPredicate {
                    card_types_any: vec![CardType::Instant, CardType::Sorcery],
                    ..Default::default()
                })),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
