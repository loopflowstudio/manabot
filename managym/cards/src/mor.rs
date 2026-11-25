//! Morningtide.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, Keyword, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![kindled_fury(), negate()]
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

fn kindled_fury() -> CardDefinition {
    let text_box = "Target creature gets +1/+0 and gains first strike until end of turn. (It deals combat damage before creatures without first strike.)";
    CardDefinition {
        key: "mor.kindled_fury",
        scryfall_id: "993956c9-30d8-41ee-84c2-c06d0512aea4",
        printed: PrintedCharacteristics {
            name: "Kindled Fury".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                1,
                0,
                Keyword::FirstStrike,
            ),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
