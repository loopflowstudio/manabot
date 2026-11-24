//! Worldwake.
use managym_state::{
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![basilisk_collar()]
}

fn basilisk_collar() -> CardDefinition {
    let text_box = "Equipped creature has deathtouch and lifelink. (Any amount of damage it deals to a creature is enough to destroy it. Damage dealt by this creature also causes you to gain that much life.)\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "wwk.basilisk_collar",
        scryfall_id: "55cdba1b-7a80-435f-9cff-b9365f62e311",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![Modification::AddKeywords(Keywords::of(&[
                    Keyword::Deathtouch,
                    Keyword::Lifelink,
                ]))],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{2}"))],
            name: "Basilisk Collar".into(),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            mana_cost: Some(ManaCost::parse("{1}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
