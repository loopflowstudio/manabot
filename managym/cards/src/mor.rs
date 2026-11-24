//! Morningtide.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![negate()]
}

fn negate() -> CardDefinition {
    CardDefinition {
        key: "mor.negate",
        scryfall_id: "5a501252-e722-4ebf-bcf7-f53a42745fa7",
        printed: PrintedCharacteristics {
            name: "Negate".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::CounterSpell {
                what: ObjectSelector::Target(TargetSpec::SpellMatching {
                    predicate: CardPredicate::noncreature(),
                }),
                exile: false,
            }],
            text_box: "Counter target noncreature spell.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
