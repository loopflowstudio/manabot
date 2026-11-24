//! Magic 2015 Promos.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetRequirement, TargetSpec, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![reclamation_sage()]
}

fn reclamation_sage() -> CardDefinition {
    let target = TargetSpec::Permanent {
        predicate: CardPredicate {
            card_types_any: vec![CardType::Artifact, CardType::Enchantment],
            ..Default::default()
        },
    };
    let text_box = "When this creature enters, you may destroy target artifact or enchantment.";
    CardDefinition {
        key: "pm15.reclamation_sage",
        scryfall_id: "cac912bb-f18a-4501-a5c0-aed8cb0a9e43",
        printed: PrintedCharacteristics {
            name: "Reclamation Sage".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Shaman".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(target.clone()),
                    effects: vec![Effect::Modal {
                        modes: vec![
                            vec![Effect::destroy(ObjectSelector::Target(target))],
                            vec![],
                        ],
                    }],
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
