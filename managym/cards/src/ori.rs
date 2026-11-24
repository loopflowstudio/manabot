//! Magic Origins.
use managym_state::ability::{
    Ability, Actor, CompareOp, Condition, Effect, EffectValue, ObjectSelector, TargetSpec,
    TriggerCondition, TriggerSubject, TurnStep,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
    PrintedCharacteristics,
};
use managym_state::mana::{Color, Mana, ManaCost, SpendEffect};
use managym_state::predicate::CardPredicate;
use std::sync::Arc;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        dwynens_elite(),
        boggart_brute(),
        demonic_pact(),
        elemental_bond(),
        pyromancers_goggles(),
        skyraker_giant(),
    ]
}

fn boggart_brute() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)";

    CardDefinition {
        key: "ori.boggart_brute",
        scryfall_id: "9d735ebf-61a4-4507-9399-6d32c8903ded",
        printed: PrintedCharacteristics {
            name: "Boggart Brute".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(3),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Menace]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn elemental_bond() -> CardDefinition {
    let text_box = "Whenever a creature you control with power 3 or greater enters, draw a card.";
    CardDefinition {
        key: "ori.elemental_bond",
        scryfall_id: "554a8769-c840-4c9d-9959-b075c174457b",
        printed: PrintedCharacteristics {
            name: "Elemental Bond".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another_you_control(CardPredicate {
                        min_power: Some(3),
                        ..CardPredicate::creature()
                    }),
                },
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pyromancers_goggles() -> CardDefinition {
    let text_box = "{T}: Add {R}. When that mana is spent to cast a red instant or sorcery spell, copy that spell and you may choose new targets for the copy.";
    let effect = SpendEffect::CopySpell {
        spell: Arc::new(CardPredicate {
            colors_any: vec![Color::Red],
            ..CardPredicate::instant_or_sorcery()
        }),
    };
    CardDefinition {
        key: "ori.pyromancers_goggles",
        scryfall_id: "1163ce9f-cf22-422e-a4b5-0240b88e2816",
        printed: PrintedCharacteristics {
            name: "Pyromancer's Goggles".into(),
            mana_cost: Some(ManaCost::parse("{5}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec!["Legendary".into()],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::of_class(
                    effect.into(),
                    [0, 0, 0, 1, 0, 0],
                ))],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn demonic_pact() -> CardDefinition {
    CardDefinition {
        key: "ori.demonic_pact",
        scryfall_id: "82c04014-91f9-4197-b4b4-f62c4739a5c2",
        printed: PrintedCharacteristics {
            name: "Demonic Pact".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            text_box: "At the beginning of your upkeep, choose one that hasn't been chosen —\n• This enchantment deals 4 damage to any target and you gain 4 life.\n• Target opponent discards two cards.\n• Draw two cards.\n• You lose the game.".into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
                intervening_if: None,
                effects: vec![Effect::TriggerModes {
                    up_to: None,
                    unchosen: true,
                    modes: vec![
                        vec![Effect::deal_damage(4, ObjectSelector::Target(TargetSpec::AnyTarget)), Effect::gain_life(4)],
                        vec![Effect::Discard { what: ObjectSelector::Target(TargetSpec::Opponent), count: 2 }],
                        vec![Effect::DrawCards { count: EffectValue::Fixed(2) }],
                        vec![Effect::LoseGame],
                    ],
                }],
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dwynens_elite() -> CardDefinition {
    let text_box = "When this creature enters, if you control another Elf, create a 1/1 green Elf Warrior creature token.";
    CardDefinition {
        key: "ori.dwynens_elite",
        scryfall_id: "c203722b-3f16-4b6c-9b2e-18169d3f80c9",
        printed: PrintedCharacteristics {
            name: "Dwynen's Elite".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::compare(
                    ObjectSelector::each_other_you_control(CardPredicate::subtype("Elf")).count(),
                    CompareOp::Ge,
                    1,
                )),
                effects: vec![Effect::create_tokens("token.elf_warrior_1_1_g", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn skyraker_giant() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)";
    CardDefinition {
        key: "ori.skyraker_giant",
        scryfall_id: "c5f0d87a-8f37-4598-9106-c3545dadf6fd",
        printed: PrintedCharacteristics {
            name: "Skyraker Giant".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Giant".into()],
            power: Some(4),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Reach]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
