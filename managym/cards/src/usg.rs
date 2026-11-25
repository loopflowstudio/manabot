//! Urza's Saga.
use managym_state::{
    ability::{Effect, TargetRequirement, TargetSpec},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        ReplacementEffect, StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![confiscate(), duress(), fog_bank()]
}

fn confiscate() -> CardDefinition {
    let permanent = TargetSpec::Permanent {
        predicate: CardPredicate::default(),
    };
    CardDefinition {
        key: "usg.confiscate",
        scryfall_id: "7cba6d4a-58d0-42d6-b49b-65c72b86007f",
        printed: PrintedCharacteristics {
            name: "Confiscate".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![Modification::Control],
            )],
            spell_effects: vec![Effect::enchant(permanent.clone())],
            targeting: vec![TargetRequirement::one(permanent)],
            text_box: "Enchant permanent\nYou control enchanted permanent.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
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
