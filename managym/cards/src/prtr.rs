//! Return to Ravnica promos.
use managym_state::{
    card::{
        CardDefinition, CardType, CardTypes, GraveyardMover, PrintedCharacteristics, Redirect,
        ReplacementEffect,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dryad_militant()]
}

fn dryad_militant() -> CardDefinition {
    let text_box = "({G/W} can be paid with either {G} or {W}.)\nIf an instant or sorcery card would be put into a graveyard from anywhere, exile it instead.";
    CardDefinition {
        key: "prtr.dryad_militant",
        scryfall_id: "d88f8449-748f-4843-b4a2-480f6ebd384d",
        printed: PrintedCharacteristics {
            name: "Dryad Militant".into(),
            mana_cost: Some(ManaCost::parse("{G/W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dryad".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(1),
            replacement_effects: vec![ReplacementEffect::GraveyardInstead {
                what: GraveyardMover::Matching(CardPredicate::instant_or_sorcery()),
                to: Redirect::Exile,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
