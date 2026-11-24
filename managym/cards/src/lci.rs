//! The Lost Caverns of Ixalan.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetRequirement, TargetSpec, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![mischievous_pup()]
}

fn mischievous_pup() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nWhen this creature enters, return up to one other target permanent you control to its owner's hand.";
    CardDefinition {
        key: "lci.mischievous_pup",
        scryfall_id: "7c219861-f808-456f-b551-37512764062d",
        printed: PrintedCharacteristics {
            name: "Mischievous Pup".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dog".into()],
            power: Some(3),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        min: 0,
                        another: true,
                        ..TargetRequirement::one(TargetSpec::PermanentYouControl {
                            predicate: CardPredicate::default(),
                        })
                    },
                    effects: vec![Effect::return_to_hand(ObjectSelector::Target(
                        TargetSpec::PermanentYouControl {
                            predicate: CardPredicate::default(),
                        },
                    ))],
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
