//! Shadowmoor.
use managym_state::{
    ability::{Actor, Effect, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        ReplacementEffect, StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::{Color, ManaCost},
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![wilt_leaf_liege(), rhys_the_redeemed()]
}

fn rhys_the_redeemed() -> CardDefinition {
    let text_box = "{2}{G/W}, {T}: Create a 1/1 green and white Elf Warrior creature token.\n{4}{G/W}{G/W}, {T}: For each creature token you control, create a token that's a copy of that creature.";
    let creature_token = CardPredicate {
        token: Some(true),
        ..CardPredicate::creature()
    };
    CardDefinition {
        key: "shm.rhys_the_redeemed",
        scryfall_id: "59327a58-cd41-479c-81c7-31c9d3b29508",
        printed: PrintedCharacteristics {
            name: "Rhys the Redeemed".into(),
            mana_cost: Some(ManaCost::parse("{G/W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Elf".into(), "Warrior".into()],
            power: Some(1),
            toughness: Some(1),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{2}{G/W}"),
                    tap_source: true,
                    effects: vec![Effect::create_tokens("token.elf_warrior", 1)],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{4}{G/W}{G/W}"),
                    tap_source: true,
                    effects: vec![Effect::create_token_copies(
                        ObjectSelector::each_you_control(creature_token),
                        1,
                    )],
                    ..Default::default()
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wilt_leaf_liege() -> CardDefinition {
    let text_box = "({G/W} can be paid with either {G} or {W}.)\nOther green creatures you control get +1/+1.\nOther white creatures you control get +1/+1.\nIf a spell or ability an opponent controls causes you to discard this card, put it onto the battlefield instead of putting it into your graveyard.";
    CardDefinition {
        key: "shm.wilt_leaf_liege",
        scryfall_id: "e6a2881f-e771-47d7-a39e-692054ee727f",
        printed: PrintedCharacteristics {
            name: "Wilt-Leaf Liege".into(),
            mana_cost: Some(ManaCost::parse("{1}{G/W}{G/W}{G/W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Knight".into()],
            power: Some(4),
            toughness: Some(4),
            statics: [Color::Green, Color::White]
                .into_iter()
                .map(|color| {
                    StaticAbility::continuous(
                        StaticScope::OtherYouControl(CardPredicate {
                            colors_any: vec![color],
                            ..CardPredicate::creature()
                        }),
                        None,
                        vec![Modification::pump(1, 1)],
                    )
                })
                .collect(),
            replacement_effects: vec![ReplacementEffect::DiscardedInstead {
                by: Actor::Opponent,
                to: ZoneType::Battlefield,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
