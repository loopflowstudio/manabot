//! Guildpact.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{AdditionalCost, CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![shattering_spree()]
}

fn shattering_spree() -> CardDefinition {
    let text_box = "Replicate {R} (When you cast this spell, copy it for each time you paid its replicate cost. You may choose new targets for the copies.)\nDestroy target artifact.";
    CardDefinition {
        key: "gpt.shattering_spree",
        scryfall_id: "d6dcff21-5900-43c4-a38b-cdc19c704ce4",
        printed: PrintedCharacteristics {
            name: "Shattering Spree".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            additional_costs: vec![AdditionalCost::Replicate(ManaCost::parse("{R}"))],
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Artifact],
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
