//! Dragon's Maze.
use managym_state::{
    ability::{Condition, Effect, EffectValue, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        ReplacementEffect,
    },
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![CardDefinition {
        key: "dgm.mazes_end",
        scryfall_id: "401f7042-24fd-42a0-ae7c-e6b7de1aa446",
        printed: PrintedCharacteristics {
            name: "Maze's End".into(),
            types: CardTypes::new([CardType::Land]),
            replacement_effects: vec![ReplacementEffect::EntersTapped],
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    tap_source: true,
                    effects: vec![Effect::add_mana(Mana::parse("C"))],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    tap_source: true,
                    return_source: true,
                    mana_cost: ManaCost::parse("{3}"),
                    effects: vec![
                        Effect::search_library(
                            CardPredicate::subtype("Gate"),
                            EffectValue::Fixed(1),
                            vec![Effect::return_to_battlefield(ObjectSelector::Selected, false)],
                        ),
                        Effect::If {
                            condition: Condition::ControlsAtLeast {
                                count: 10,
                                predicate: CardPredicate::subtype("Gate"),
                                distinct_names: true,
                            },
                            then: vec![Effect::WinGame],
                            otherwise: vec![],
                        },
                    ],
                    ..Default::default()
                },
            ],
            text_box: "This land enters tapped.\n{T}: Add {C}.\n{3}, {T}, Return this land to its owner's hand: Search your library for a Gate card, put it onto the battlefield, then shuffle. If you control ten or more Gates with different names, you win the game.".into(),
            ..Default::default()
        },
        ..Default::default()
    }]
}
