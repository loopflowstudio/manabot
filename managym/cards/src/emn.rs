//! Eldritch Moon.
use managym_state::{
    ability::{
        Ability, Effect, ObjectSelector, PlayerSelector, TargetRequirement, TargetSpec,
        TriggerCondition,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::{Colors, Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![imprisoned_in_the_moon(), mirrorwing_dragon()]
}

fn mirrorwing_dragon() -> CardDefinition {
    let text_box = "Flying\nWhenever a player casts an instant or sorcery spell that targets only this creature, that player copies that spell for each other creature they control that the spell could target. Each copy targets a different one of those creatures.";
    CardDefinition {
        key: "emn.mirrorwing_dragon",
        scryfall_id: "4b7e33a8-765b-4909-a5c6-5fe8f8774a51",
        printed: PrintedCharacteristics {
            name: "Mirrorwing Dragon".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(4),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::SpellTargetsOnlyThis {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Instant, CardType::Sorcery],
                        ..Default::default()
                    },
                },
                intervening_if: None,
                effects: vec![Effect::ForPlayer {
                    player: PlayerSelector::TriggeringPlayer,
                    effects: vec![Effect::CopyTriggeringSpellForEach {
                        each: ObjectSelector::each_other_you_control(CardPredicate::creature()),
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn imprisoned_in_the_moon() -> CardDefinition {
    let text_box = "Enchant creature, land, or planeswalker\nEnchanted permanent is a colorless land with \"{T}: Add {C}\" and loses all other card types and abilities.";
    let enchantable = TargetSpec::Permanent {
        predicate: CardPredicate {
            card_types_any: vec![CardType::Creature, CardType::Land, CardType::Planeswalker],
            ..Default::default()
        },
    };
    CardDefinition {
        key: "emn.imprisoned_in_the_moon",
        scryfall_id: "7990ebba-e9f2-4ba4-a352-e26ec81d4bed",
        printed: PrintedCharacteristics {
            name: "Imprisoned in the Moon".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            // One effect across layers 4, 5 and 6; the loss precedes the grant
            // it keeps.
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::Attached,
                condition: None,
                characteristic_defining: false,
                modifications: vec![
                    Modification::SetTypes {
                        types: vec![CardType::Land],
                        subtypes: vec![],
                    },
                    Modification::SetColors(Colors::new()),
                    Modification::LoseAllAbilities,
                    Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
                        tap_source: true,
                        effects: vec![Effect::add_mana(Mana::parse("C"))],
                        ..Default::default()
                    })),
                ],
            })],
            spell_effects: vec![Effect::AttachSourceTo {
                what: ObjectSelector::Target(enchantable.clone()),
            }],
            targeting: vec![TargetRequirement::one(enchantable)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
