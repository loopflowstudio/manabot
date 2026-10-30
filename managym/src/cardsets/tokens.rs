//! Tokens created by cards outside TLA. TLA defines its own fixed tokens.
use crate::state::{
    ability::Effect,
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes},
    mana::{Color, Colors, Mana},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![human_soldier(), treasure()]
}

fn human_soldier() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "token.human_soldier",
        scryfall_id: "90de6d1e-e654-4f4a-8014-061ce69e540f",
        name: "Human Soldier".into(),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(1),
        toughness: Some(1),
        is_token: true,
        color_override: Some(Colors::from([Color::White])),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn treasure() -> CardDefinition {
    let text_box = "{T}, Sacrifice this token: Add one mana of any color.";

    CardDefinition {
        key: "token.treasure",
        scryfall_id: "a3a684b7-27e0-4d9e-a064-9e03c6e50c89",
        name: "Treasure".into(),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Treasure".into()],
        is_token: true,
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            sacrifice_source: true,
            effect: Effect::Modal {
                modes: vec![
                    vec![Effect::AddMana {
                        mana: Mana::parse("W"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("U"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("B"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("R"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("G"),
                        until_end_of_combat: false,
                    }],
                ],
            },
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
