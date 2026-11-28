//! Journey into Nyx promos.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, EntryScope,
        PrintedCharacteristics, ReplacementEffect,
    },
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![heroes_bane()]
}

fn heroes_bane() -> CardDefinition {
    let text_box = "This creature enters with four +1/+1 counters on it.\n{2}{G}{G}: Put X +1/+1 counters on this creature, where X is its power.";
    CardDefinition {
        key: "pjou.heroes_bane",
        scryfall_id: "f4fed10c-74f8-49af-8daa-b9c081f549f8",
        printed: PrintedCharacteristics {
            name: "Heroes' Bane".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Hydra".into()],
            power: Some(0),
            toughness: Some(0),
            replacement_effects: vec![ReplacementEffect::EntersWithCounters {
                on: EntryScope::This,
                kind: CounterKind::PlusOnePlusOne,
                count: 4.into(),
                condition: None,
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}{G}{G}"),
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    EffectValue::SourcePower,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
