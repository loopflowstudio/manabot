//! Dominaria United.
use managym_state::{
    ability::PaymentCost,
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
    predicate::CardPredicate,
};
use managym_state::{
    ability::{Ability, Effect, TargetSpec},
    card::{Keyword, Keywords},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![prayer_of_binding(), tolarian_terror()]
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

fn prayer_of_binding() -> CardDefinition {
    let text_box = "Flash\nWhen this enchantment enters, exile up to one target nonland permanent an opponent controls until this enchantment leaves the battlefield. You gain 2 life.";
    CardDefinition {
        key: "dmu.prayer_of_binding",
        scryfall_id: "322f90b6-6b49-458d-9d5b-b601bfdd0af8",
        printed: PrintedCharacteristics {
            name: "Prayer of Binding".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::enters(vec![
                Effect::exile_until_source_leaves(
                    TargetSpec::PermanentOpponentControls {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    },
                    0,
                ),
                Effect::gain_life(2),
            ])],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
