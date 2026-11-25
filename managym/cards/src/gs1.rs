//! Global Series Jiang Yanggu & Mu Yanling.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, Per, Scope, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![ancestor_dragon()]
}

fn ancestor_dragon() -> CardDefinition {
    let text_box = "Flying\nWhenever one or more creatures you control attack, you gain 1 life for each attacking creature.";
    CardDefinition {
        key: "gs1.ancestor_dragon",
        scryfall_id: "9ba9d1f0-a864-490c-b258-6bf9de251c4b",
        printed: PrintedCharacteristics {
            name: "Ancestor Dragon".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(5),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate::creature()),
                    Per::Batch,
                ),
                intervening_if: None,
                effects: vec![Effect::gain_life(
                    ObjectSelector::each(
                        Scope::Battlefield,
                        CardPredicate {
                            attacking: true,
                            ..CardPredicate::creature()
                        },
                    )
                    .count(),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
