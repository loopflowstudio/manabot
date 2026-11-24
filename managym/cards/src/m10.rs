//! Magic 2010.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![essence_scatter()]
}

fn essence_scatter() -> CardDefinition {
    CardDefinition {
        key: "m10.essence_scatter",
        scryfall_id: "c231101e-6620-46fc-a0ad-a53291d12dc2",
        printed: PrintedCharacteristics {
            name: "Essence Scatter".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::CounterSpell {
                what: ObjectSelector::Target(TargetSpec::SpellMatching {
                    predicate: CardPredicate::creature(),
                }),
                exile: false,
            }],
            text_box: "Counter target creature spell.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
