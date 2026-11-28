//! Darksteel.
use managym_state::{
    card::{
        CardDefinition, CardType, CardTypes, GraveyardMover, Keyword, Keywords,
        PrintedCharacteristics, Redirect, ReplacementEffect,
    },
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![darksteel_colossus()]
}

fn darksteel_colossus() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nIndestructible (Damage and effects that say \"destroy\" don't destroy this creature.)\nIf Darksteel Colossus would be put into a graveyard from anywhere, reveal Darksteel Colossus and shuffle it into its owner's library instead.";
    CardDefinition {
        key: "dst.darksteel_colossus",
        scryfall_id: "cbc27b24-f085-48b0-8757-cd11fbf25b91",
        printed: PrintedCharacteristics {
            name: "Darksteel Colossus".into(),
            mana_cost: Some(ManaCost::parse("{11}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Golem".into()],
            power: Some(11),
            toughness: Some(11),
            keywords: Keywords::of(&[Keyword::Trample, Keyword::Indestructible]),
            replacement_effects: vec![ReplacementEffect::GraveyardInstead {
                what: GraveyardMover::This,
                to: Redirect::OwnerLibraryShuffled { reveal: true },
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
