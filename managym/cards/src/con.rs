//! Conflux.
use managym_state::{
    card::{
        CardDefinition, CardType, CardTypes, GraveyardMover, PrintedCharacteristics, Redirect,
        ReplacementEffect, SourceQuality, StaticAbility,
    },
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![progenitus()]
}

fn progenitus() -> CardDefinition {
    let text_box = "Protection from everything\nIf Progenitus would be put into a graveyard from anywhere, reveal Progenitus and shuffle it into its owner's library instead.";
    CardDefinition {
        key: "con.progenitus",
        scryfall_id: "bcc764b0-3046-4bde-b424-c0f4e1a6169b",
        printed: PrintedCharacteristics {
            name: "Progenitus".into(),
            mana_cost: Some(ManaCost::parse("{W}{W}{U}{U}{B}{B}{R}{R}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Hydra".into(), "Avatar".into()],
            power: Some(10),
            toughness: Some(10),
            replacement_effects: vec![ReplacementEffect::GraveyardInstead {
                what: GraveyardMover::This,
                to: Redirect::OwnerLibraryShuffled { reveal: true },
            }],
            statics: vec![StaticAbility::Protection(SourceQuality::Everything)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
