//! Urza's Saga.
use managym_state::{
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        ReplacementEffect,
    },
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![CardDefinition {
        key: "usg.fog_bank",
        scryfall_id: "6ade0d30-5a57-439e-95e8-5f865880031f",
        printed: PrintedCharacteristics {
            name: "Fog Bank".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Wall".into()],
            power: Some(0),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Defender, Keyword::Flying]),
            replacement_effects: vec![ReplacementEffect::PreventCombatDamageToOrFromSelf],
            text_box: "Defender (This creature can't attack.)\nFlying\nPrevent all combat damage that would be dealt to and dealt by this creature.".into(),
            ..Default::default()
        },
        ..Default::default()
    }]
}
