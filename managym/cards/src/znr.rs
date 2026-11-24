//! Zendikar Rising.
use managym_state::ability::{
    Ability, Aggregate, Effect, EffectValue, ObjectProperty, ObjectSelector, PlayerSelector,
    TargetSpec, TriggerSubject,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, ManaAbility,
    PrintedCharacteristics,
};
use managym_state::effect::Modification;
use managym_state::mana::{Mana, ManaCost};
use managym_state::permanent::CounterKind;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        feed_the_swarm(),
        spitfire_lagac(),
        crawling_barrens(),
        broken_wings(),
        highborn_vampire(),
        CardDefinition {
            key: "znr.marauding_blight_priest",
            scryfall_id: "730ddbcd-0814-4e22-85e9-78b0878324b6",
            printed: PrintedCharacteristics {
                name: "Marauding Blight-Priest".into(),
                mana_cost: Some(ManaCost::parse("{2}{B}")),
                types: CardTypes::new([CardType::Creature]),
                subtypes: vec!["Vampire".into(), "Cleric".into()],
                power: Some(3),
                toughness: Some(2),
                abilities: vec![Ability::when_you_gain_life(Effect::LoseLife {
                    player: PlayerSelector::EachOpponent,
                    amount: EffectValue::Fixed(1),
                })],
                text_box: "Whenever you gain life, each opponent loses 1 life.".into(),
                ..Default::default()
            },
            ..Default::default()
        },
    ]
}

fn feed_the_swarm() -> CardDefinition {
    let text_box = "Destroy target creature or enchantment an opponent controls. You lose life equal to that permanent's mana value.";
    let target = || {
        ObjectSelector::Target(TargetSpec::PermanentOpponentControls {
            predicate: CardPredicate {
                card_types_any: vec![CardType::Creature, CardType::Enchantment],
                ..Default::default()
            },
        })
    };
    CardDefinition {
        key: "znr.feed_the_swarm",
        scryfall_id: "f6b2eba7-862a-4efd-9f65-065fb2070855",
        printed: PrintedCharacteristics {
            name: "Feed the Swarm".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::destroy(target()),
                // The permanent as it is, or as it last existed once destroyed.
                Effect::LoseLife {
                    player: PlayerSelector::You,
                    amount: target().property(ObjectProperty::ManaValue, Aggregate::Sum),
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn highborn_vampire() -> CardDefinition {
    CardDefinition {
        key: "znr.highborn_vampire",
        scryfall_id: "24c40082-516e-4381-a4cc-e61c5a9a6cac",
        printed: PrintedCharacteristics {
            name: "Highborn Vampire".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Warrior".into()],
            power: Some(4),
            toughness: Some(3),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn broken_wings() -> CardDefinition {
    CardDefinition {
        key: "znr.broken_wings",
        scryfall_id: "c0fc2dfd-85b0-4add-be18-b39549235921",
        printed: PrintedCharacteristics {
            name: "Broken Wings".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        any_of: vec![
                            CardPredicate {
                                card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                                ..Default::default()
                            },
                            CardPredicate {
                                flying: true,
                                ..CardPredicate::creature()
                            },
                        ],
                        ..Default::default()
                    },
                },
            ))],
            text_box: "Destroy target artifact, enchantment, or creature with flying.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn crawling_barrens() -> CardDefinition {
    let text_box = "{T}: Add {C}.\n{4}: Put two +1/+1 counters on this land. Then you may have it become a 0/0 Elemental creature until end of turn. It's still a land.";
    CardDefinition {
        key: "znr.crawling_barrens",
        scryfall_id: "7bd0e025-7a75-4641-a51a-27df9dcde05f",
        printed: PrintedCharacteristics {
            name: "Crawling Barrens".into(),
            types: CardTypes::new([CardType::Land]),
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("C"),
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{4}"),
                effects: vec![
                    Effect::put_counters(
                        ObjectSelector::ExactSource,
                        CounterKind::PlusOnePlusOne,
                        2,
                    ),
                    Effect::Modal {
                        modes: vec![
                            vec![
                                Effect::until_end_of_turn(
                                    ObjectSelector::ExactSource,
                                    Modification::BecomeCreature,
                                ),
                                Effect::until_end_of_turn(
                                    ObjectSelector::ExactSource,
                                    Modification::AddSubtypes(vec!["Elemental".into()]),
                                ),
                                Effect::until_end_of_turn(
                                    ObjectSelector::ExactSource,
                                    Modification::base_pt(Some(0), Some(0)),
                                ),
                            ],
                            vec![],
                        ],
                    },
                ],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn spitfire_lagac() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, this creature deals 1 damage to each opponent.";
    CardDefinition {
        key: "znr.spitfire_lagac",
        scryfall_id: "47f26493-812f-4c14-91c4-d2ab549a7b8a",
        printed: PrintedCharacteristics {
            name: "Spitfire Lagac".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Lizard".into()],
            power: Some(3),
            toughness: Some(4),
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Effect::DealDamageToPlayer {
                    player: PlayerSelector::EachOpponent,
                    amount: 1.into(),
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
