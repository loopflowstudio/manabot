//! Time Spiral.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![cancel()]
}

fn cancel() -> CardDefinition {
    CardDefinition {
        key: "tsp.cancel",
        scryfall_id: "b4e175f7-f649-451b-9ee5-ad1140b2e8a7",
        printed: PrintedCharacteristics {
            name: "Cancel".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::CounterSpell {
                what: ObjectSelector::Target(TargetSpec::Spell),
                exile: false,
            }],
            text_box: "Counter target spell.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
