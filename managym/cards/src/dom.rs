//! Dominaria.
use managym_state::effect::Modification;
use managym_state::{
    ability::{Effect, ObjectSelector, TargetRequirement, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![run_amok()]
}

fn run_amok() -> CardDefinition {
    let text_box = "Target attacking creature gets +3/+3 and gains trample until end of turn.";
    let attacking = || TargetSpec::Permanent {
        predicate: CardPredicate {
            attacking: true,
            ..CardPredicate::creature()
        },
    };
    CardDefinition {
        key: "dom.run_amok",
        scryfall_id: "d3f2aa2c-deec-4927-a2d6-f5dce5967e26",
        name: "Run Amok".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::until_end_of_turn(
                ObjectSelector::Target(attacking()),
                Modification::pump(3, 3),
            ),
            Effect::until_end_of_turn(
                ObjectSelector::Target(attacking()),
                Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
            ),
        ],
        targeting: vec![TargetRequirement::one(attacking())],
        text_box: text_box.into(),
        ..Default::default()
    }
}
