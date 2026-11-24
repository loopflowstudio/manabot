//! Rivals of Ixalan.
use managym_state::ability::{
    Ability, Aggregate, Condition, Effect, ObjectProperty, ObjectSelector, TargetSpec,
    TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
    PrintedCharacteristics, StaticAbility, StaticScope,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        brasss_bounty(),
        ghalta_primal_hunger(),
        fanatical_firebrand(),
        thrashing_brontodon(),
        kitesail_corsair(),
        moment_of_craving(),
        moment_of_triumph(),
        zetalpa_primal_dawn(),
        swab_goblin(),
        gleaming_barrier(),
    ]
}

fn fanatical_firebrand() -> CardDefinition {
    let text_box = "Haste (This creature can attack and {T} as soon as it comes under your control.)\n{T}, Sacrifice this creature: It deals 1 damage to any target.";
    CardDefinition {
        key: "rix.fanatical_firebrand",
        scryfall_id: "5e5565de-028c-4799-a9f6-4dcd685639eb",
        printed: PrintedCharacteristics {
            name: "Fanatical Firebrand".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Pirate".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Haste]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::deal_damage(
                    1,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn thrashing_brontodon() -> CardDefinition {
    let text_box = "{1}, Sacrifice this creature: Destroy target artifact or enchantment.";
    let artifact_or_enchantment = CardPredicate {
        card_types_any: vec![CardType::Artifact, CardType::Enchantment],
        ..Default::default()
    };
    CardDefinition {
        key: "rix.thrashing_brontodon",
        scryfall_id: "0d9264ff-9f7c-46f3-862a-fee7ad213250",
        printed: PrintedCharacteristics {
            name: "Thrashing Brontodon".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(3),
            toughness: Some(4),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                sacrifice_source: true,
                effects: vec![Effect::destroy(ObjectSelector::Target(
                    TargetSpec::Permanent {
                        predicate: artifact_or_enchantment,
                    },
                ))],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn brasss_bounty() -> CardDefinition {
    let text_box = "For each land you control, create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "rix.brasss_bounty",
        scryfall_id: "13505866-6ce9-4d11-b3f3-fc2f09839b71",
        printed: PrintedCharacteristics {
            name: "Brass's Bounty".into(),
            mana_cost: Some(ManaCost::parse("{6}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_tokens(
                "token.treasure",
                ObjectSelector::each_you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                })
                .count(),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ghalta_primal_hunger() -> CardDefinition {
    let text_box = "This spell costs {X} less to cast, where X is the total power of creatures you control.\nTrample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "rix.ghalta_primal_hunger",
        scryfall_id: "0104b5b3-9376-4ad7-9a77-3e564e9c42e6",
        printed: PrintedCharacteristics {
            name: "Ghalta, Primal Hunger".into(),
            mana_cost: Some(ManaCost::parse("{10}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Elder".into(), "Dinosaur".into()],
            power: Some(12),
            toughness: Some(12),
            keywords: Keywords::of(&[Keyword::Trample]),
            statics: vec![StaticAbility::CostsLess {
                amount: ObjectSelector::each_you_control(CardPredicate::creature())
                    .property(ObjectProperty::Power, Aggregate::Sum),
                condition: None,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gleaming_barrier() -> CardDefinition {
    let text_box = "Defender\nWhen this creature dies, create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "rix.gleaming_barrier",
        scryfall_id: "62447a76-4aa7-4823-941e-84bc18eb672a",
        printed: PrintedCharacteristics {
            name: "Gleaming Barrier".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Wall".into()],
            power: Some(0),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Defender]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::create_tokens("token.treasure", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn swab_goblin() -> CardDefinition {
    CardDefinition {
        key: "rix.swab_goblin",
        scryfall_id: "0403b8e5-29c5-4a4a-b9e7-1e79a7452f14",
        printed: PrintedCharacteristics {
            name: "Swab Goblin".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Pirate".into()],
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn zetalpa_primal_dawn() -> CardDefinition {
    let text_box = "Flying, double strike, vigilance, trample, indestructible";
    CardDefinition {
        key: "rix.zetalpa_primal_dawn",
        scryfall_id: "3d10560f-199d-4d04-b573-90024f8aecc4",
        printed: PrintedCharacteristics {
            name: "Zetalpa, Primal Dawn".into(),
            mana_cost: Some(ManaCost::parse("{6}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elder".into(), "Dinosaur".into()],
            power: Some(4),
            toughness: Some(8),
            supertypes: vec!["Legendary".into()],
            keywords: Keywords::of(&[
                Keyword::Flying,
                Keyword::Indestructible,
                Keyword::Vigilance,
                Keyword::Trample,
                Keyword::DoubleStrike,
            ]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn moment_of_craving() -> CardDefinition {
    let text_box = "Target creature gets -2/-2 until end of turn. You gain 2 life.";
    CardDefinition {
        key: "rix.moment_of_craving",
        scryfall_id: "c0e3ea55-162d-4466-ba4e-b938a0845fb5",
        printed: PrintedCharacteristics {
            name: "Moment of Craving".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(-2, -2),
                ),
                Effect::gain_life(2),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn moment_of_triumph() -> CardDefinition {
    let text_box = "Target creature gets +2/+2 until end of turn. You gain 2 life.";
    CardDefinition {
        key: "rix.moment_of_triumph",
        scryfall_id: "08be8c18-eca3-4960-b174-e4a78579ed63",
        printed: PrintedCharacteristics {
            name: "Moment of Triumph".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(2, 2),
                ),
                Effect::gain_life(2),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kitesail_corsair() -> CardDefinition {
    let text_box = "This creature has flying as long as it's attacking.";
    CardDefinition {
        key: "rix.kitesail_corsair",
        scryfall_id: "4b8d0e8d-c2d4-4682-8095-827ffd79539b",
        printed: PrintedCharacteristics {
            name: "Kitesail Corsair".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Pirate".into()],
            power: Some(2),
            toughness: Some(1),
            statics: vec![StaticAbility::continuous(
                StaticScope::This,
                Some(Condition::SourceMatches(CardPredicate {
                    attacking: true,
                    ..Default::default()
                })),
                vec![Modification::AddKeywords(Keywords::of(&[Keyword::Flying]))],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
