//! Urza's Saga.
use managym_state::{
    ability::{Effect, TargetSpec},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        ReplacementEffect,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![duress(), fog_bank()]
}

fn fog_bank() -> CardDefinition {
    CardDefinition {
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
    }
}

fn duress() -> CardDefinition {
    let text_box = "Target opponent reveals their hand. You choose a noncreature, nonland card from it. That player discards that card.";
    CardDefinition {
        key: "usg.duress",
        scryfall_id: "ca367f49-0f4a-4b7f-8104-851893fbcd8a",
        printed: PrintedCharacteristics {
            spell_effects: vec![Effect::RevealHandDiscardChosen {
                target: TargetSpec::Opponent,
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Land, CardType::Creature],
                    ..Default::default()
                },
            }],
            name: "Duress".into(),
            types: CardTypes::new([CardType::Sorcery]),
            mana_cost: Some(ManaCost::parse("{B}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
