//! Dragon's Maze.
use managym_state::{
    ability::{Condition, Effect, EffectValue, ObjectSelector, TargetRequirement, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, ReplacementEffect, StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![unflinching_courage(), mazes_end()]
}

fn mazes_end() -> CardDefinition {
    CardDefinition {
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
    }
}

fn unflinching_courage() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature gets +2/+2 and has trample and lifelink. (Damage dealt by the creature also causes its controller to gain that much life.)";
    CardDefinition {
        key: "dgm.unflinching_courage",
        scryfall_id: "35952c24-d728-4ec6-b0d1-b8183a18554a",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![
                    Modification::AddKeywords(Keywords::of(&[Keyword::Trample, Keyword::Lifelink])),
                    Modification::pump(2, 2),
                ],
            )],
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::AttachSourceTo {
                what: ObjectSelector::Target(TargetSpec::Creature),
            }],
            name: "Unflinching Courage".into(),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            mana_cost: Some(ManaCost::parse("{1}{G}{W}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
