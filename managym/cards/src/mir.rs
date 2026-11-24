//! Mirage.

use managym_state::{
    ability::{Effect, TargetSpec},
    card::{
        CardDefinition, CardType, CardTypes, PrintedCharacteristics, Restriction, StaticAbility,
        StaticRestriction, StaticScope,
    },
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![pacifism()]
}

fn pacifism() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature can't attack or block.";
    CardDefinition {
        key: "mir.pacifism",
        scryfall_id: "c891df1b-bae6-4d6d-85ee-42901c149f98",
        printed: PrintedCharacteristics {
            name: "Pacifism".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            spell_effects: vec![Effect::enchant(TargetSpec::Creature)],
            statics: vec![StaticAbility::Restriction(StaticRestriction {
                scope: StaticScope::Attached,
                kind: Restriction::CantAttackOrBlock,
                unless: None,
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
