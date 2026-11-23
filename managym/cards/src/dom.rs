//! Dominaria.
use managym_state::effect::Modification;
use managym_state::{
    ability::{Effect, ObjectSelector, TargetRequirement, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![run_amok(), muldrotha()]
}

fn muldrotha() -> CardDefinition {
    let text_box = "During each of your turns, you may play a land and cast a permanent spell of each permanent type from your graveyard. (If a card has multiple permanent types, choose one as you play it.)";
    CardDefinition {
        key: "dom.muldrotha_the_gravetide",
        scryfall_id: "c654737d-34ac-42ff-ae27-3a3bbb930fc1",
        printed: PrintedCharacteristics {
            name: "Muldrotha, the Gravetide".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Elemental".into(), "Avatar".into()],
            power: Some(6),
            toughness: Some(6),
            statics: vec![managym_state::card::StaticAbility::PlayGraveyardByPermanentType],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
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
        printed: PrintedCharacteristics {
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
        },
        ..Default::default()
    }
}
