// sacrifice.rs
// Elemental sacrifice and landfall cards using shared typed costs and events.

use super::alpha::CardRegistry;
use crate::state::{
    ability::{
        Ability, Effect, StaticCondition, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords, SacrificeCost,
    },
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

impl CardRegistry {
    pub(super) fn register_sacrifice(&mut self) {
        self.register_card(CardDefinition {
            name: "Bastion of Remembrance".into(),
            text_box: "When this enchantment enters, create a 1/1 white Human Soldier creature token.\nWhenever a creature you control dies, each opponent loses 1 life and you gain 1 life.".into(),
            types: CardTypes::new([CardType::Enchantment]),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::CreateToken {
                        token_name: "Human Soldier".into(),
                        count: 1,
                        tapped_and_attacking: false,
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::AnyYouControl(CardPredicate::creature()),
                    },
                    effects: vec![Effect::DrainLife { amount: 1 }],
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Callous Inspector".into(),
            text_box: "Menace (This creature can't be blocked except by two or more creatures.)\nWhen this creature dies, it deals 1 damage to you. Create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            mana_cost: Some(ManaCost::parse("{B}")),
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords {
                menace: true,
                ..Default::default()
            },
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![
                    Effect::DealDamageToController { amount: 1 },
                    Effect::CreateToken {
                        token_name: "Clue".into(),
                        count: 1,
                        tapped_and_attacking: false,
                    },
                ],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Curious Farm Animals".into(),
            text_box: "When this creature dies, you gain 3 life.\n{2}, Sacrifice this creature: Destroy up to one target artifact or enchantment.".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Boar".into(), "Elk".into(), "Bird".into(), "Ox".into()],
            mana_cost: Some(ManaCost::parse("{W}")),
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::GainLife { amount: 3 }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("2"),
                effect: Effect::TargetClause {
                    requirement: TargetRequirement::up_to(
                        1,
                        TargetSpec::Permanent {
                            predicate: CardPredicate {
                                card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                                ..Default::default()
                            },
                        },
                    ),
                    effects: vec![Effect::Destroy {
                        target: TargetSpec::Permanent {
                            predicate: CardPredicate {
                                card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                                ..Default::default()
                            },
                        },
                    }],
                },
                sacrifice_source: true,
                ..Default::default()
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Cycle of Renewal".into(),
            text_box: "Sacrifice a land. Search your library for up to two basic land cards, put them onto the battlefield tapped, then shuffle.".into(),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            spell_effects: vec![
                Effect::Sacrifice {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    },
                },
                Effect::SearchBasicLand {
                    subtype: None,
                    count: 2,
                    tapped: true,
                    untap_if_lands: None,
                    bottom_source: false,
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Deadly Dispute".into(),
            text_box: "As an additional cost to cast this spell, sacrifice an artifact or creature.\nDraw two cards and create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")".into(),
            types: CardTypes::new([CardType::Instant]),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            sacrifice_cost: Some(SacrificeCost {
                predicate: CardPredicate {
                    card_types_any: vec![CardType::Artifact, CardType::Creature],
                    ..Default::default()
                },
                another: false,
                or_mana: None,
            }),
            spell_effects: vec![
                Effect::DrawCards { count: 2 },
                Effect::CreateToken {
                    token_name: "Treasure".into(),
                    count: 1,
                    tapped_and_attacking: false,
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Deadly Precision".into(),
            text_box: "As an additional cost to cast this spell, pay {4} or sacrifice an artifact or creature.\nDestroy target creature.".into(),
            types: CardTypes::new([CardType::Sorcery]),
            mana_cost: Some(ManaCost::parse("{B}")),
            sacrifice_cost: Some(SacrificeCost {
                predicate: CardPredicate {
                    card_types_any: vec![CardType::Artifact, CardType::Creature],
                    ..Default::default()
                },
                another: false,
                or_mana: Some(ManaCost::parse("4")),
            }),
            spell_effects: vec![Effect::Destroy {
                target: TargetSpec::Creature,
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Fabled Passage".into(),
            text_box: "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Then if you control four or more lands, untap that land.".into(),
            types: CardTypes::new([CardType::Land]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                effect: Effect::SearchBasicLand {
                    subtype: None,
                    count: 1,
                    tapped: true,
                    untap_if_lands: Some(4),
                    bottom_source: false,
                },
                tap_source: true,
                sacrifice_source: true,
                ..Default::default()
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "June, Bounty Hunter".into(),
            text_box: "June can't be blocked as long as you've drawn two or more cards this turn.\n{1}, Sacrifice another creature: Create a Clue token. Activate only during your turn. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Mercenary".into()],
            supertypes: vec!["Legendary".into()],
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            power: Some(2),
            toughness: Some(2),
            unblockable_if: Some(StaticCondition::CardsDrawnAtLeast { count: 2 }),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("1"),
                effect: Effect::CreateToken {
                    token_name: "Clue".into(),
                    count: 1,
                    tapped_and_attacking: false,
                },
                own_turn_only: true,
                sacrifice: Some(SacrificeCost {
                    predicate: CardPredicate::creature(),
                    another: true,
                    or_mana: None,
                }),
                ..Default::default()
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Lotus Cobra".into(),
            text_box: "Landfall — Whenever a land you control enters, add one mana of any color."
                .into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Snake".into()],
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnyYouControl(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::Modal {
                    modes: ["W", "U", "B", "R", "G"]
                        .into_iter()
                        .map(|color| {
                            vec![Effect::AddMana {
                                mana: Mana::parse(color),
                                until_end_of_combat: false,
                            }]
                        })
                        .collect(),
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Mayhem Devil".into(),
            text_box: "Whenever a player sacrifices a permanent, this creature deals 1 damage to any target.".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Devil".into()],
            mana_cost: Some(ManaCost::parse("{1}{B}{R}")),
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Sacrificed {
                    predicate: CardPredicate::default(),
                    controller_only: false,
                },
                effects: vec![Effect::DealDamage {
                    amount: 1,
                    target: TargetSpec::CreatureOrPlayer,
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Prismatic Vista".into(),
            text_box: "{T}, Pay 1 life, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield, then shuffle.".into(),
            types: CardTypes::new([CardType::Land]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                effect: Effect::SearchBasicLand {
                    subtype: None,
                    count: 1,
                    tapped: false,
                    untap_if_lands: None,
                    bottom_source: false,
                },
                tap_source: true,
                sacrifice_source: true,
                life_cost: 1,
                ..Default::default()
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Sazh's Chocobo".into(),
            text_box: "Landfall — Whenever a land you control enters, put a +1/+1 counter on this creature.".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into()],
            mana_cost: Some(ManaCost::parse("{G}")),
            power: Some(0),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnyYouControl(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::PutCountersOnSource { count: 1 }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Tireless Provisioner".into(),
            text_box: "Landfall — Whenever a land you control enters, create a Food token or a Treasure token. (Food is an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\" Treasure is an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Scout".into()],
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            power: Some(3),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnyYouControl(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::CreateToken {
                            token_name: "Food".into(),
                            count: 1,
                            tapped_and_attacking: false,
                        }],
                        vec![Effect::CreateToken {
                            token_name: "Treasure".into(),
                            count: 1,
                            tapped_and_attacking: false,
                        }],
                    ],
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Tireless Tracker".into(),
            text_box: "Landfall — Whenever a land you control enters, investigate. (Create a Clue token. It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you sacrifice a Clue, put a +1/+1 counter on this creature.".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Scout".into()],
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::AnyYouControl(CardPredicate {
                            card_type: Some(CardType::Land),
                            ..Default::default()
                        }),
                    },
                    effects: vec![Effect::CreateToken {
                        token_name: "Clue".into(),
                        count: 1,
                        tapped_and_attacking: false,
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Sacrificed {
                        predicate: CardPredicate::subtype("Clue"),
                        controller_only: true,
                    },
                    effects: vec![Effect::PutCountersOnSource { count: 1 }],
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Tolls of War".into(),
            text_box: "When this enchantment enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you sacrifice a permanent during your turn, create a 1/1 white Ally creature token. This ability triggers only once each turn.".into(),
            types: CardTypes::new([CardType::Enchantment]),
            mana_cost: Some(ManaCost::parse("{W}{B}")),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::CreateToken {
                        token_name: "Clue".into(),
                        count: 1,
                        tapped_and_attacking: false,
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::OnceEachTurn {
                        condition: Box::new(TriggerCondition::ActiveIf {
                            active_if: StaticCondition::YourTurn,
                            condition: Box::new(TriggerCondition::Sacrificed {
                                predicate: CardPredicate::default(),
                                controller_only: true,
                            }),
                        }),
                    },
                    effects: vec![Effect::CreateToken {
                        token_name: "Ally".into(),
                        count: 1,
                        tapped_and_attacking: false,
                    }],
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Unlucky Cabbage Merchant".into(),
            text_box: "When this creature enters, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nWhenever you sacrifice a Food, you may search your library for a basic land card and put it onto the battlefield tapped. If you search your library this way, put this creature on the bottom of its owner's library, then shuffle.".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Citizen".into()],
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            power: Some(2),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::CreateToken {
                        token_name: "Food".into(),
                        count: 1,
                        tapped_and_attacking: false,
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Sacrificed {
                        predicate: CardPredicate::subtype("Food"),
                        controller_only: true,
                    },
                    effects: vec![Effect::Modal {
                        modes: vec![
                            vec![Effect::SearchBasicLand {
                                subtype: None,
                                count: 1,
                                tapped: true,
                                untap_if_lands: None,
                                bottom_source: true,
                            }],
                            vec![],
                        ],
                    }],
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Weaponize the Monsters".into(),
            text_box: "{2}, Sacrifice a creature: This enchantment deals 2 damage to any target."
                .into(),
            types: CardTypes::new([CardType::Enchantment]),
            mana_cost: Some(ManaCost::parse("{R}")),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("2"),
                effect: Effect::DealDamage {
                    amount: 2,
                    target: TargetSpec::CreatureOrPlayer,
                },
                sacrifice: Some(SacrificeCost {
                    predicate: CardPredicate::creature(),
                    another: false,
                    or_mana: None,
                }),
                ..Default::default()
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Treasure".into(),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Treasure".into()],
            is_token: true,
            text_box: "{T}, Sacrifice this token: Add one mana of any color.".into(),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                sacrifice_source: true,
                effect: Effect::Modal {
                    modes: ["W", "U", "B", "R", "G"]
                        .into_iter()
                        .map(|color| {
                            vec![Effect::AddMana {
                                mana: Mana::parse(color),
                                until_end_of_combat: false,
                            }]
                        })
                        .collect(),
                },
                ..Default::default()
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Human Soldier".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            is_token: true,
            color_override: Some([crate::state::mana::Color::White].into_iter().collect()),
            power: Some(1),
            toughness: Some(1),
            ..Default::default()
        });
    }
}
