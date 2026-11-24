//! Lorwyn.
use managym_state::{
    ability::Effect,
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![imperious_perfect()]
}

fn imperious_perfect() -> CardDefinition {
    let text_box = "Other Elves you control get +1/+1.\n{G}, {T}: Create a 1/1 green Elf Warrior creature token.";
    CardDefinition {
        key: "lrw.imperious_perfect",
        scryfall_id: "706fce74-fed9-4bf7-949d-7df6bef29238",
        printed: PrintedCharacteristics {
            name: "Imperious Perfect".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(2),
            statics: vec![StaticAbility::continuous(
                StaticScope::OtherYouControl(CardPredicate::subtype("Elf")),
                None,
                vec![Modification::pump(1, 1)],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{G}"),
                tap_source: true,
                effects: vec![Effect::create_tokens("token.elf_warrior_1_1_g", 1)],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
