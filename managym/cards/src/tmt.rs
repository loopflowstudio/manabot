//! Teenage Mutant Ninja Turtles.
use managym_state::{
    ability::{Ability, Actor, Condition, Effect, ObjectSelector, TriggerCondition, TurnStep},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![michelangelo_game_master()]
}

fn michelangelo_game_master() -> CardDefinition {
    let text_box = "Disappear — At the beginning of your end step, if a permanent left the battlefield under your control this turn, put a +1/+1 counter on Michelangelo.";
    CardDefinition {
        key: "tmt.michelangelo_game_master",
        scryfall_id: "2e914c3d-2eed-48bf-af9a-a8998fd5111d",
        printed: PrintedCharacteristics {
            name: "Michelangelo, Game Master".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Mutant".into(), "Ninja".into(), "Turtle".into()],
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::PermanentLeftThisTurn {
                    who: Actor::You,
                    nonland: false,
                }),
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
