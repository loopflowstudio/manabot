//! Alara Reborn.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![maelstrom_pulse()]
}

fn maelstrom_pulse() -> CardDefinition {
    let text_box = "Destroy target nonland permanent and all other permanents with the same name as that permanent.";
    CardDefinition {
        key: "arb.maelstrom_pulse",
        scryfall_id: "eb651c3a-cb27-4b73-8eb6-b87d65211097",
        printed: PrintedCharacteristics {
            name: "Maelstrom Pulse".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::destroy(ObjectSelector::TargetAndSameName(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Land],
                        ..Default::default()
                    },
                },
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
