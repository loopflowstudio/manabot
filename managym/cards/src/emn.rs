//! Eldritch Moon.
use managym_state::{
    ability::{
        Ability, CompareOp, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector,
        TargetRequirement, TargetSpec, TriggerCondition,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticScope,
    },
    effect::{EffectDuration, Modification},
    mana::{Colors, Mana, ManaCost},
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        cemetery_recruitment(),
        harmless_offering(),
        imprisoned_in_the_moon(),
        mirrorwing_dragon(),
    ]
}

fn harmless_offering() -> CardDefinition {
    let permanent = TargetSpec::PermanentYouControl {
        predicate: CardPredicate::default(),
    };
    CardDefinition {
        key: "emn.harmless_offering",
        scryfall_id: "f8f3cc4f-7943-4025-b332-b40653b13014",
        printed: PrintedCharacteristics {
            name: "Harmless Offering".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::Opponent),
                    effects: vec![],
                },
                Effect::TargetClause {
                    requirement: TargetRequirement::one(permanent.clone()),
                    effects: vec![Effect::GainControl {
                        player: PlayerSelector::TargetClause(0),
                        what: ObjectSelector::Target(permanent),
                        duration: EffectDuration::Indefinite,
                    }],
                },
            ],
            text_box: "Target opponent gains control of target permanent you control.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
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
                    predicate: CardPredicate::instant_or_sorcery(),
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
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![
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
            )],
            spell_effects: vec![Effect::enchant(enchantable.clone())],
            targeting: vec![TargetRequirement::one(enchantable)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn cemetery_recruitment() -> CardDefinition {
    let text_box = "Return target creature card from your graveyard to your hand. If it's a Zombie card, draw a card.";
    CardDefinition {
        key: "emn.cemetery_recruitment",
        scryfall_id: "3a23adea-9f4a-409c-a37d-323eee781273",
        printed: PrintedCharacteristics {
            name: "Cemetery Recruitment".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::return_to_hand(ObjectSelector::Target(TargetSpec::GraveyardCardYouOwn {
                    predicate: CardPredicate::creature(),
                })),
                // "it": the card this returned, as it is in the hand.
                Effect::If {
                    condition: Condition::compare(
                        EffectValue::Objects {
                            of: Box::new(ObjectSelector::Moved),
                            predicate: Some(CardPredicate {
                                subtype: Some("Zombie".into()),
                                ..Default::default()
                            }),
                            in_zone: Some(ZoneType::Hand),
                        },
                        CompareOp::Ge,
                        1,
                    ),
                    then: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                    otherwise: vec![],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
