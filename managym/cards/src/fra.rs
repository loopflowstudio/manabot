//! Reality Fracture.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{
        CardDefinition, CardType, CardTypes, InsetSpell, Keyword, Keywords, PrintedCharacteristics,
        ReplacementEffect,
    },
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    let seed_suture = InsetSpell::Prepared(
        PrintedCharacteristics {
            name: "Seed Suture".into(),
            mana_cost: Some(ManaCost::parse("{G/W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::put_counters(
                    ObjectSelector::Target(TargetSpec::Creature),
                    CounterKind::PlusOnePlusOne,
                    1,
                ),
                Effect::gain_life(1),
            ],
            text_box: "Put a +1/+1 counter on target creature. You gain 1 life.".into(),
            ..Default::default()
        }
        .into(),
    );
    let prepared_creature = PrintedCharacteristics {
        types: CardTypes::new([CardType::Creature]),
        replacement_effects: vec![ReplacementEffect::EntersPrepared],
        inset: Some(seed_suture),
        ..Default::default()
    };
    vec![
        CardDefinition {
            key: "fra.blossom_blessed_angel",
            scryfall_id: "5e77fbf0-9d1f-4e7d-a02e-43065d11b0d9",
            printed: PrintedCharacteristics {
                name: "Blossom-Blessed Angel".into(),
                mana_cost: Some(ManaCost::parse("{3}{W}")),
                subtypes: vec!["Angel".into(), "Cleric".into()],
                power: Some(2),
                toughness: Some(4),
                keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance]),
                text_box: "Flying, vigilance\nThis creature enters prepared.".into(),
                ..prepared_creature.clone()
            },
            ..Default::default()
        },
        CardDefinition {
            key: "fra.emergency_phytomedic",
            scryfall_id: "de94d388-919d-44ff-baef-8c90a417ac6d",
            printed: PrintedCharacteristics {
                name: "Emergency Phytomedic".into(),
                mana_cost: Some(ManaCost::parse("{G/W}")),
                subtypes: vec!["Dryad".into(), "Cleric".into()],
                power: Some(1),
                toughness: Some(1),
                text_box: "This creature enters prepared.".into(),
                ..prepared_creature
            },
            ..Default::default()
        },
    ]
}
