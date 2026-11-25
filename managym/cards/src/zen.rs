//! Zendikar.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PaymentCost, Per,
        PlayerSelector, Scope, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
        TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, ManaAbility, PrintedCharacteristics, ReplacementEffect, StaticAbility,
        StaticPtBuff, StaticScope,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        gatekeeper_of_malakir(),
        adventuring_gear(),
        day_of_judgment(),
        vampire_nighthawk(),
        rampaging_baloths(),
        beastmaster_ascension(),
        bloodchief_ascension(),
        burst_lightning(),
        expedition_map(),
        into_the_roil(),
        lotus_cobra(),
        valakut_the_molten_pinnacle(),
        rite_of_replication(),
    ]
}

fn lotus_cobra() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, add one mana of any color.";

    CardDefinition {
        key: "zen.lotus_cobra",
        scryfall_id: "19adde22-e5eb-4815-beb6-c520b3274cc9",
        printed: PrintedCharacteristics {
            name: "Lotus Cobra".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Snake".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::you_control(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::mana_choice(
                    ["W", "U", "B", "R", "G"].map(Mana::parse),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn beastmaster_ascension() -> CardDefinition {
    let text_box = "Whenever a creature you control attacks, you may put a quest counter on this enchantment.\nAs long as this enchantment has seven or more quest counters on it, creatures you control get +5/+5.";
    CardDefinition {
        key: "zen.beastmaster_ascension",
        scryfall_id: "02d7ea78-d539-4c34-8bb3-941353e3da46",
        printed: PrintedCharacteristics {
            name: "Beastmaster Ascension".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate::creature()),
                    Per::Object,
                ),
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Mana(ManaCost::default()),
                    if_paid: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::Quest,
                        1,
                    )],
                    if_declined: vec![],
                }],
            }],
            statics: vec![StaticAbility::PtBuff(StaticPtBuff {
                scope: StaticScope::YouControl(CardPredicate::creature()),
                condition: Some(Condition::SourceCountersAtLeast {
                    kind: CounterKind::Quest,
                    count: 7,
                }),
                power: 5,
                toughness: 5,
                per_creature_type_max: None,
                per_color_among: None,
                per_graveyard_matching: None,
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bloodchief_ascension() -> CardDefinition {
    let text_box = "At the beginning of each end step, if an opponent lost 2 or more life this turn, you may put a quest counter on this enchantment. (Damage causes loss of life.)\nWhenever a card is put into an opponent's graveyard from anywhere, if this enchantment has three or more quest counters on it, you may have that player lose 2 life. If you do, you gain 2 life.";
    // "You may" with nothing to pay is an optional payment of no mana.
    let may = |if_paid: Vec<Effect>| Effect::OptionalPayment {
        cost: PaymentCost::Mana(ManaCost::default()),
        if_paid,
        if_declined: vec![],
    };
    CardDefinition {
        key: "zen.bloodchief_ascension",
        scryfall_id: "aa213dbb-c52a-4084-92aa-d0b5d97a97d9",
        printed: PrintedCharacteristics {
            name: "Bloodchief Ascension".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![
                Ability::Triggered {
                    intervening_if: Some(Condition::OpponentLostLifeThisTurnAtLeast { amount: 2 }),
                    condition: TriggerCondition::beginning_of(TurnStep::End, Actor::AnyPlayer),
                    effects: vec![may(vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::Quest,
                        1,
                    )])],
                },
                Ability::Triggered {
                    intervening_if: Some(Condition::SourceCountersAtLeast {
                        kind: CounterKind::Quest,
                        count: 3,
                    }),
                    condition: TriggerCondition::PutIntoGraveyard {
                        whose: Actor::Opponent,
                        predicate: CardPredicate::default(),
                    },
                    effects: vec![may(vec![
                        Effect::LoseLife {
                            player: PlayerSelector::TriggeringPlayer,
                            amount: EffectValue::Fixed(2),
                        },
                        Effect::gain_life(2),
                    ])],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn valakut_the_molten_pinnacle() -> CardDefinition {
    let text_box = "This land enters tapped.\nWhenever a Mountain you control enters, if you control at least five other Mountains, you may have this land deal 3 damage to any target.\n{T}: Add {R}.";
    CardDefinition {
        key: "zen.valakut_the_molten_pinnacle",
        scryfall_id: "37bce60d-2cb0-4772-9f5c-122a7ed426a0",
        printed: PrintedCharacteristics {
            name: "Valakut, the Molten Pinnacle".into(),
            types: CardTypes::new([CardType::Land]),
            replacement_effects: vec![ReplacementEffect::EntersTapped],
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("R"),
            }],
            abilities: vec![Ability::Triggered {
                // The entering Mountain and at least five others.
                intervening_if: Some(Condition::ControlsAtLeast {
                    distinct_names: false,
                    count: 6,
                    predicate: CardPredicate::subtype("Mountain"),
                }),
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another_you_control(CardPredicate::subtype(
                        "Mountain",
                    )),
                },
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::AnyTarget),
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::default()),
                        if_paid: vec![Effect::deal_damage(
                            3,
                            ObjectSelector::Target(TargetSpec::AnyTarget),
                        )],
                        if_declined: vec![],
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn burst_lightning() -> CardDefinition {
    let text_box = "Kicker {4} (You may pay an additional {4} as you cast this spell.)\nBurst Lightning deals 2 damage to any target. If this spell was kicked, it deals 4 damage instead.";
    CardDefinition {
        key: "zen.burst_lightning",
        scryfall_id: "2dc16614-5cf8-444d-a5ae-cac25018af68",
        printed: PrintedCharacteristics {
            name: "Burst Lightning".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Instant]),
            additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{4}"))],
            spell_effects: vec![Effect::If {
                condition: Condition::Kicked,
                then: vec![Effect::deal_damage(
                    4,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
                otherwise: vec![Effect::deal_damage(
                    2,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn expedition_map() -> CardDefinition {
    let text_box = "{2}, {T}, Sacrifice this artifact: Search your library for a land card, reveal it, put it into your hand, then shuffle.";
    CardDefinition {
        key: "zen.expedition_map",
        scryfall_id: "c55bee97-593f-441f-b96c-a998d5212a55",
        printed: PrintedCharacteristics {
            name: "Expedition Map".into(),
            mana_cost: Some(ManaCost::parse("{1}")),
            types: CardTypes::new([CardType::Artifact]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}"),
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::search_library(
                    CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    },
                    EffectValue::Fixed(1),
                    vec![Effect::return_to_hand(ObjectSelector::Selected)],
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn into_the_roil() -> CardDefinition {
    let text_box = "Kicker {1}{U} (You may pay an additional {1}{U} as you cast this spell.)\nReturn target nonland permanent to its owner's hand. If this spell was kicked, draw a card.";
    CardDefinition {
        key: "zen.into_the_roil",
        scryfall_id: "5dba9972-dd8b-407b-9374-a8f0ed1a96db",
        printed: PrintedCharacteristics {
            name: "Into the Roil".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Instant]),
            additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{1}{U}"))],
            spell_effects: vec![
                Effect::return_to_hand(ObjectSelector::Target(TargetSpec::Permanent {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Land],
                        ..Default::default()
                    },
                })),
                Effect::If {
                    condition: Condition::Kicked,
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

fn rampaging_baloths() -> CardDefinition {
    let text_box = "Trample\nLandfall — Whenever a land you control enters, create a 4/4 green Beast creature token.";
    CardDefinition {
        key: "zen.rampaging_baloths",
        scryfall_id: "66ae703d-b133-4749-9d38-216abe6c6647",
        printed: PrintedCharacteristics {
            name: "Rampaging Baloths".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Beast".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Effect::create_tokens("token.beast_4_4_g", 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vampire_nighthawk() -> CardDefinition {
    let text_box = "Flying\nDeathtouch (Any amount of damage this deals to a creature is enough to destroy it.)\nLifelink (Damage dealt by this creature also causes you to gain that much life.)";
    CardDefinition {
        key: "zen.vampire_nighthawk",
        scryfall_id: "44f19fe3-7a17-4c45-adfa-590f73dfebfa",
        printed: PrintedCharacteristics {
            name: "Vampire Nighthawk".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Shaman".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Deathtouch, Keyword::Flying, Keyword::Lifelink]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn day_of_judgment() -> CardDefinition {
    CardDefinition {
        key: "zen.day_of_judgment",
        scryfall_id: "2aa98fca-972b-46c2-bdec-6ace35c988d5",
        printed: PrintedCharacteristics {
            name: "Day of Judgment".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::destroy(ObjectSelector::each(
                Scope::Battlefield,
                CardPredicate::creature(),
            ))],
            text_box: "Destroy all creatures.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn adventuring_gear() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, equipped creature gets +2/+2 until end of turn.\nEquip {1} ({1}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "zen.adventuring_gear",
        scryfall_id: "3aa395f2-656e-4bf3-bd9b-6240bd3e2774",
        printed: PrintedCharacteristics {
            name: "Adventuring Gear".into(),
            mana_cost: Some(ManaCost::parse("{1}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{1}"))],
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Effect::until_end_of_turn(
                    ObjectSelector::Attached,
                    Modification::pump(2, 2),
                )],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rite_of_replication() -> CardDefinition {
    let text_box = "Kicker {5} (You may pay an additional {5} as you cast this spell.)\nCreate a token that's a copy of target creature. If this spell was kicked, create five of those tokens instead.";
    let copy =
        |count| Effect::create_token_copies(ObjectSelector::Target(TargetSpec::Creature), count);
    CardDefinition {
        key: "zen.rite_of_replication",
        scryfall_id: "4530fe45-8a3d-48e9-a7a5-abf8fb1485e3",
        printed: PrintedCharacteristics {
            name: "Rite of Replication".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{5}"))],
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::If {
                condition: Condition::Kicked,
                then: vec![copy(5)],
                otherwise: vec![copy(1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gatekeeper_of_malakir() -> CardDefinition {
    CardDefinition {
key: "zen.gatekeeper_of_malakir",
scryfall_id: "71db3698-a45c-4eaf-87e6-30502c0c10f4",
printed: PrintedCharacteristics {
name: "Gatekeeper of Malakir".into(),
mana_cost: Some(ManaCost::parse("{B}{B}")),
types: CardTypes::new([CardType::Creature]),
subtypes: vec!["Vampire".into(), "Warrior".into()],
power: Some(2),
toughness: Some(2),
additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{B}"))],
abilities: vec![Ability::Triggered {
condition: TriggerCondition::EntersTheBattlefield { subject: TriggerSubject::This },
intervening_if: Some(Condition::Kicked),
effects: vec![Effect::ForPlayer { player: PlayerSelector::Target(Box::new(TargetSpec::Player)), effects: vec![Effect::sacrifice(CardPredicate::creature())] }]
}],
text_box: "Kicker {B} (You may pay an additional {B} as you cast this spell.)\nWhen this creature enters, if it was kicked, target player sacrifices a creature of their choice.".into(),
..Default::default()
},
..Default::default()
}
}
