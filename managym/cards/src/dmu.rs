//! Dominaria United.
use managym_state::{
    ability::PaymentCost,
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![tolarian_terror()]
}

fn tolarian_terror() -> CardDefinition {
    let text_box = "This spell costs {1} less to cast for each instant and sorcery card in your graveyard.\nWard {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)";
    CardDefinition {
        key: "dmu.tolarian_terror",
        scryfall_id: "42f01cba-43d4-46ad-b7a5-d7631b0e1347",
        printed: PrintedCharacteristics {
            name: "Tolarian Terror".into(),
            mana_cost: Some(ManaCost::parse("{6}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Serpent".into()],
            power: Some(5),
            toughness: Some(5),
            statics: vec![
                StaticAbility::CostsLessPerGraveyardCard(CardPredicate {
                    card_types_any: vec![CardType::Instant, CardType::Sorcery],
                    ..Default::default()
                }),
                StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse("{2}"))),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
