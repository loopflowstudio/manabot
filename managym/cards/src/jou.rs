//! Journey into Nyx.
use managym_state::card::CardDefinition;
use managym_state::{
    ability::{Ability, Effect, TargetSpec},
    card::{CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

use super::ths::temple;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![banishing_light(), temple_of_epiphany(), temple_of_malady()]
}

fn temple_of_epiphany() -> CardDefinition {
    temple(
        "jou.temple_of_epiphany",
        "b882c6bf-b795-49fe-8242-a928aadb6f13",
        "Temple of Epiphany",
        ["U", "R"],
    )
}

fn temple_of_malady() -> CardDefinition {
    temple(
        "jou.temple_of_malady",
        "f30220f1-1992-4b5c-9e13-1762fb673155",
        "Temple of Malady",
        ["B", "G"],
    )
}

fn banishing_light() -> CardDefinition {
    let text_box = "When this enchantment enters, exile target nonland permanent an opponent controls until this enchantment leaves the battlefield.";
    CardDefinition {
        key: "jou.banishing_light",
        scryfall_id: "fbaa4800-30cc-4a80-a6cc-9a24ada9eb40",
        printed: PrintedCharacteristics {
            name: "Banishing Light".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::enters(vec![Effect::exile_until_source_leaves(
                TargetSpec::PermanentOpponentControls {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Land],
                        ..Default::default()
                    },
                },
                1,
            )])],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
