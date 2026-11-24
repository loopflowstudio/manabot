//! Hour of Devastation.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![abrade()]
}

fn abrade() -> CardDefinition {
    let text_box =
        "Choose one —\n• Abrade deals 3 damage to target creature.\n• Destroy target artifact.";
    CardDefinition {
        key: "hou.abrade",
        scryfall_id: "84319dfb-eaf7-4b98-8c4f-30f5e779591b",
        printed: PrintedCharacteristics {
            name: "Abrade".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::deal_damage(
                        3,
                        ObjectSelector::Target(TargetSpec::Creature),
                    )],
                    vec![Effect::destroy(ObjectSelector::Target(
                        TargetSpec::Permanent {
                            predicate: CardPredicate {
                                card_types_any: vec![CardType::Artifact],
                                ..Default::default()
                            },
                        },
                    ))],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
