//! Core Set 2021.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![shipwreck_dowser()]
}

fn shipwreck_dowser() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\nWhen this creature enters, return target instant or sorcery card from your graveyard to your hand.";
    CardDefinition {
        key: "m21.shipwreck_dowser",
        scryfall_id: "59d38ef7-5017-4ea3-b97f-a8fe12d03e98",
        printed: PrintedCharacteristics {
            name: "Shipwreck Dowser".into(),
            subtypes: vec!["Merfolk".into(), "Wizard".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            power: Some(3),
            toughness: Some(3),
            abilities: vec![
                Ability::prowess(),
                Ability::when_enters(
                    TriggerSubject::This,
                    vec![Effect::return_to_hand(ObjectSelector::Target(
                        TargetSpec::GraveyardCardYouOwn {
                            predicate: CardPredicate::instant_or_sorcery(),
                        },
                    ))],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
