//! Urza's Legacy.
use managym_state::{
    ability::{Effect, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        ManaAbility, PrintedCharacteristics, ReplacementEffect,
    },
    effect::Modification,
    mana::{Color, Mana, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![treetop_village()]
}

fn treetop_village() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {G}.\n{1}{G}: This land becomes a 3/3 green Ape creature with trample until end of turn. It's still a land. (It can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "ulg.treetop_village",
        scryfall_id: "02212bd8-0c0f-4e8e-99f1-a8477476c03a",
        printed: PrintedCharacteristics {
            name: "Treetop Village".into(),
            types: CardTypes::new([CardType::Land]),
            replacement_effects: vec![ReplacementEffect::EntersTapped],
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("G"),
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}{G}"),
                effects: [
                    Modification::BecomeCreature,
                    Modification::base_pt(Some(3), Some(3)),
                    Modification::AddSubtypes(vec!["Ape".into()]),
                    Modification::SetColors([Color::Green].into_iter().collect()),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                ]
                .into_iter()
                .map(|modification| Effect::until_end_of_turn(ObjectSelector::This, modification))
                .collect(),
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
