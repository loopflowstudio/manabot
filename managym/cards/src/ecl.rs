//! Lorwyn Eclipsed.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetRequirement, TargetSpec},
    card::{AdditionalCost, CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![appeal_to_eirdu()]
}

fn appeal_to_eirdu() -> CardDefinition {
    let text_box = "Convoke (Your creatures can help cast this spell. Each creature you tap while casting this spell pays for {1} or one mana of that creature's color.)\nOne or two target creatures each get +2/+1 until end of turn.";
    CardDefinition {
        key: "ecl.appeal_to_eirdu",
        scryfall_id: "68ce9752-21f9-48e9-bf48-7f76f1cecbc5",
        printed: PrintedCharacteristics {
            name: "Appeal to Eirdu".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            types: CardTypes::new([CardType::Instant]),
            additional_costs: vec![AdditionalCost::Convoke],
            spell_effects: vec![Effect::ForEachTarget {
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(2, 1),
                )],
            }],
            targeting: vec![TargetRequirement {
                max: 2,
                ..TargetRequirement::one(TargetSpec::Creature)
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
