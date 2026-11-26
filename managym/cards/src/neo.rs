//! Kamigawa: Neon Dynasty.
use managym_state::{
    ability::Effect,
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, EntryChoiceKind,
        ManaAbility, PrintedCharacteristics, ReplacementEffect,
    },
    mana::{CreatureManaPurpose, Mana},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![secluded_courtyard(), uncharted_haven()]
}

fn uncharted_haven() -> CardDefinition {
    let text_box = "This land enters tapped. As it enters, choose a color.\n{T}: Add one mana of the chosen color.";
    CardDefinition {
        key: "neo.uncharted_haven",
        scryfall_id: "1d4ad89a-3a00-4bf4-a357-4a8a089d4a82",
        printed: PrintedCharacteristics {
            name: "Uncharted Haven".into(),
            types: CardTypes::new([CardType::Land]),
            replacement_effects: vec![
                ReplacementEffect::EntersTapped,
                ReplacementEffect::AsEntersChoose {
                    kind: EntryChoiceKind::Color,
                    look_at_opponent_hand: false,
                },
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::AddOneColorManaValue {
                    amount: 1.into(),
                    chosen_color: true,
                    creature_purposes: vec![],
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn secluded_courtyard() -> CardDefinition {
    let text_box = "As this land enters, choose a creature type.\n{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast a creature spell of the chosen type or activate an ability of a creature source of the chosen type.";
    CardDefinition {
        key: "neo.secluded_courtyard",
        scryfall_id: "0539b1a5-8704-476f-ba1f-2fe01190e157",
        printed: PrintedCharacteristics {
            name: "Secluded Courtyard".into(),
            types: CardTypes::new([CardType::Land]),
            replacement_effects: vec![ReplacementEffect::AsEntersChoose {
                kind: EntryChoiceKind::CreatureType,
                look_at_opponent_hand: false,
            }],
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("C"),
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::AddOneColorManaValue {
                    amount: 1.into(),
                    chosen_color: false,
                    creature_purposes: vec![
                        CreatureManaPurpose::Cast,
                        CreatureManaPurpose::Activate,
                    ],
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
