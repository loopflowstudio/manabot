//! Magic 2011.
use managym_state::{
    ability::{Ability, Effect, LinkedExile, ObjectSelector, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![hoarding_dragon()]
}

fn hoarding_dragon() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, you may search your library for an artifact card, exile it, then shuffle.\nWhen this creature dies, you may put the exiled card into its owner's hand.";
    CardDefinition {
        key: "m11.hoarding_dragon",
        scryfall_id: "1f8b6932-e62d-4d38-bd0e-9ab8d4a56762",
        printed: PrintedCharacteristics {
            name: "Hoarding Dragon".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![
                Ability::enters(vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::search_library(
                            CardPredicate {
                                card_type: Some(CardType::Artifact),
                                ..Default::default()
                            },
                            1.into(),
                            vec![Effect::exile_linked(
                                ObjectSelector::Selected,
                                LinkedExile::WithSource,
                            )],
                        )],
                        vec![],
                    ],
                }]),
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::Modal {
                        modes: vec![
                            vec![Effect::return_to_hand(ObjectSelector::ExiledWithSource)],
                            vec![],
                        ],
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
