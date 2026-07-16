// tla.rs
// Avatar: The Last Airbender — Milestone-1 two-deck slice, Stage 1 cards.
// Every card here is registered with its full oracle behavior; cards whose
// text needs Stage 2+ machinery (choices, costs, statics) are not registered.

use crate::state::{
    ability::{
        Ability, Effect, StaticCondition, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords, PowerCda,
        StaticPtBuff, StaticScope, TriggeredManaAbility,
    },
    mana::{Color, Colors, Mana, ManaCost},
    predicate::CardPredicate,
};

use super::alpha::CardRegistry;

fn ally() -> CardPredicate {
    CardPredicate::subtype("Ally")
}

fn lesson() -> CardPredicate {
    CardPredicate::subtype("Lesson")
}

fn lesson_in_graveyard() -> StaticCondition {
    StaticCondition::GraveyardAtLeast {
        count: 1,
        predicate: lesson(),
    }
}

impl CardRegistry {
    pub(super) fn register_tla(&mut self) {
        self.register_tla_tokens();

        // Kyoshi Warriors {3}{W} 3/3 — Human Warrior Ally
        // When this creature enters, create a 1/1 white Ally creature token.
        self.register_card(CardDefinition {
            name: "Kyoshi Warriors".to_string(),
            mana_cost: Some(ManaCost::parse("3W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Ally".to_string(),
                    count: 1,
                    tapped_and_attacking: false,
                }],
            }],
            text_box: "When this creature enters, create a 1/1 white Ally creature token."
                .to_string(),
            power: Some(3),
            toughness: Some(3),
            ..Default::default()
        });

        // Avatar Enthusiasts {2}{W} 2/2 — Human Peasant Ally
        // Whenever another Ally you control enters, put a +1/+1 counter on
        // this creature.
        self.register_card(CardDefinition {
            name: "Avatar Enthusiasts".to_string(),
            mana_cost: Some(ManaCost::parse("2W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Peasant".into(), "Ally".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(ally()),
                },
                effects: vec![Effect::PutCountersOnSource { count: 1 }],
            }],
            text_box:
                "Whenever another Ally you control enters, put a +1/+1 counter on this creature."
                    .to_string(),
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        });

        // Invasion Reinforcements {1}{W} 1/1 — Human Warrior Ally
        // Flash. When this creature enters, create a 1/1 white Ally creature token.
        self.register_card(CardDefinition {
            name: "Invasion Reinforcements".to_string(),
            mana_cost: Some(ManaCost::parse("1W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
            keywords: Keywords {
                flash: true,
                ..Default::default()
            },
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Ally".to_string(),
                    count: 1,
                    tapped_and_attacking: false,
                }],
            }],
            text_box: "Flash\nWhen this creature enters, create a 1/1 white Ally creature token."
                .to_string(),
            power: Some(1),
            toughness: Some(1),
            ..Default::default()
        });

        // Jeong Jeong's Deserters {1}{W} 1/2 — Human Rebel Ally
        // When this creature enters, put a +1/+1 counter on target creature.
        self.register_card(CardDefinition {
            name: "Jeong Jeong's Deserters".to_string(),
            mana_cost: Some(ManaCost::parse("1W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::PutCounters {
                    count: 1,
                    target: TargetSpec::Creature,
                }],
            }],
            text_box: "When this creature enters, put a +1/+1 counter on target creature."
                .to_string(),
            power: Some(1),
            toughness: Some(2),
            ..Default::default()
        });

        // South Pole Voyager {1}{W} 2/2 — Human Scout Ally
        // Whenever this creature or another Ally you control enters, you gain
        // 1 life. If this is the second time this ability has resolved this
        // turn, draw a card.
        self.register_card(CardDefinition {
            name: "South Pole Voyager".to_string(),
            mana_cost: Some(ManaCost::parse("1W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Scout".into(), "Ally".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnyYouControl(ally()),
                },
                effects: vec![
                    Effect::GainLife { amount: 1 },
                    Effect::OnNthResolutionThisTurn {
                        n: 2,
                        effect: Box::new(Effect::DrawCards { count: 1 }),
                    },
                ],
            }],
            text_box: "Whenever this creature or another Ally you control enters, you gain 1 life. If this is the second time this ability has resolved this turn, draw a card.".to_string(),
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        });

        // Tiger-Seal {U} 3/3 — Cat Seal
        // Vigilance. At the beginning of your upkeep, tap this creature.
        // Whenever you draw your second card each turn, untap this creature.
        self.register_card(CardDefinition {
            name: "Tiger-Seal".to_string(),
            mana_cost: Some(ManaCost::parse("U")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Seal".into()],
            keywords: Keywords {
                vigilance: true,
                ..Default::default()
            },
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::BeginningOfYourUpkeep,
                    effects: vec![Effect::TapSource],
                },
                Ability::Triggered {
                    condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
                    effects: vec![Effect::UntapSource],
                },
            ],
            text_box: "Vigilance\nAt the beginning of your upkeep, tap this creature.\nWhenever you draw your second card each turn, untap this creature.".to_string(),
            power: Some(3),
            toughness: Some(3),
            ..Default::default()
        });

        // Otter-Penguin {1}{U} 2/1 — Otter Bird
        // Whenever you draw your second card each turn, this creature gets
        // +1/+2 until end of turn and can't be blocked this turn.
        self.register_card(CardDefinition {
            name: "Otter-Penguin".to_string(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Otter".into(), "Bird".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
                effects: vec![
                    Effect::ModifyUntilEot {
                        power_delta: 1,
                        toughness_delta: 2,
                    },
                    Effect::CantBeBlockedThisTurnSource,
                ],
            }],
            text_box: "Whenever you draw your second card each turn, this creature gets +1/+2 until end of turn and can't be blocked this turn.".to_string(),
            power: Some(2),
            toughness: Some(1),
            ..Default::default()
        });

        // Forecasting Fortune Teller {1}{U} 1/3 — Human Advisor Ally
        // When this creature enters, create a Clue token.
        self.register_card(CardDefinition {
            name: "Forecasting Fortune Teller".to_string(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Advisor".into(), "Ally".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Clue".to_string(),
                    count: 1,
                    tapped_and_attacking: false,
                }],
            }],
            text_box: "When this creature enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")".to_string(),
            power: Some(1),
            toughness: Some(3),
            ..Default::default()
        });

        // ---- Stage 2: decisions & costs ----

        // Glider Kids {2}{W} 2/3 — Human Pilot Ally
        // Flying. When this creature enters, scry 1.
        self.register_card(CardDefinition {
            name: "Glider Kids".to_string(),
            mana_cost: Some(ManaCost::parse("2W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Pilot".into(), "Ally".into()],
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Scry { count: 1 }],
            }],
            text_box: "Flying\nWhen this creature enters, scry 1. (Look at the top card of your library. You may put it on the bottom.)".to_string(),
            power: Some(2),
            toughness: Some(3),
            ..Default::default()
        });

        // Firebending Lesson {R} Instant — Lesson
        // Kicker {4}. Firebending Lesson deals 2 damage to target creature.
        // If this spell was kicked, it deals 5 damage to that creature
        // instead.
        self.register_card(CardDefinition {
            name: "Firebending Lesson".to_string(),
            mana_cost: Some(ManaCost::parse("R")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            kicker: Some(ManaCost::parse("4")),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::IfKicked {
                then: vec![Effect::DealDamage {
                    amount: 5,
                    target: TargetSpec::Creature,
                }],
                otherwise: vec![Effect::DealDamage {
                    amount: 2,
                    target: TargetSpec::Creature,
                }],
            }],
            text_box: "Kicker {4} (You may pay an additional {4} as you cast this spell.)\nFirebending Lesson deals 2 damage to target creature. If this spell was kicked, it deals 5 damage to that creature instead.".to_string(),
            ..Default::default()
        });

        // It'll Quench Ya! {1}{U} Instant — Lesson
        // Counter target spell unless its controller pays {2}.
        self.register_card(CardDefinition {
            name: "It'll Quench Ya!".to_string(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            targeting: vec![TargetRequirement::one(TargetSpec::Spell)],
            spell_effects: vec![Effect::CounterUnlessPays {
                cost: ManaCost::parse("2"),
            }],
            text_box: "Counter target spell unless its controller pays {2}.".to_string(),
            ..Default::default()
        });

        // Accumulate Wisdom {1}{U} Instant — Lesson
        // Look at the top three cards of your library. Put one of those
        // cards into your hand and the rest on the bottom of your library
        // in any order. Put each of those cards into your hand instead if
        // there are three or more Lesson cards in your graveyard.
        // (Engine bottoms the rest in a random order — no reorder
        // sub-decision, see wave doc Stage-2 notes.)
        self.register_card(CardDefinition {
            name: "Accumulate Wisdom".to_string(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            spell_effects: vec![Effect::IfGraveyardAtLeast {
                count: 3,
                predicate: CardPredicate::subtype("Lesson"),
                then: vec![Effect::PutTopCardsInHand { count: 3 }],
                otherwise: vec![Effect::LookAndSelect {
                    look: 3,
                    min_select: 1,
                    max_select: 1,
                    predicate: CardPredicate::default(),
                }],
            }],
            text_box: "Look at the top three cards of your library. Put one of those cards into your hand and the rest on the bottom of your library in any order. Put each of those cards into your hand instead if there are three or more Lesson cards in your graveyard.".to_string(),
            ..Default::default()
        });

        // Water Tribe Rallier {1}{W} 2/2 — Human Soldier Ally
        // Waterbend {5}: Look at the top four cards of your library. You may
        // reveal a creature card with power 3 or less from among them and
        // put it into your hand. Put the rest on the bottom of your library
        // in a random order.
        self.register_card(CardDefinition {
            name: "Water Tribe Rallier".to_string(),
            mana_cost: Some(ManaCost::parse("1W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("5"),
                sacrifice_source: false,
                waterbend: true,
                effect: Effect::LookAndSelect {
                    look: 4,
                    min_select: 0,
                    max_select: 1,
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        max_power: Some(3),
                        ..CardPredicate::default()
                    },
                },
            }],
            text_box: "Waterbend {5}: Look at the top four cards of your library. You may reveal a creature card with power 3 or less from among them and put it into your hand. Put the rest on the bottom of your library in a random order. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)".to_string(),
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        });

        // Allies at Last {2}{G} Instant
        // Affinity for Allies. Up to two target creatures you control each
        // deal damage equal to their power to target creature an opponent
        // controls.
        self.register_card(CardDefinition {
            name: "Allies at Last".to_string(),
            mana_cost: Some(ManaCost::parse("2G")),
            types: CardTypes::new([CardType::Instant]),
            cost_reduction_per: Some(ally()),
            targeting: vec![
                TargetRequirement::up_to(2, TargetSpec::CreatureYouControl),
                TargetRequirement::one(TargetSpec::CreatureOpponentControls),
            ],
            spell_effects: vec![Effect::TargetCreaturesDealPowerDamageToLastTarget],
            text_box: "Affinity for Allies (This spell costs {1} less to cast for each Ally you control.)\nUp to two target creatures you control each deal damage equal to their power to target creature an opponent controls.".to_string(),
            ..Default::default()
        });

        // Badgermole Cub {1}{G} 2/2 — Badger Mole
        // When this creature enters, earthbend 1.
        // Whenever you tap a creature for mana, add an additional {G}.
        // (Triggered mana ability, CR 605.1b — no stack; composes with
        // waterbend payments.)
        self.register_card(CardDefinition {
            name: "Badgermole Cub".to_string(),
            mana_cost: Some(ManaCost::parse("1G")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Badger".into(), "Mole".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Earthbend {
                    count: 1,
                    target: TargetSpec::LandYouControl,
                }],
            }],
            triggered_mana_abilities: vec![TriggeredManaAbility {
                predicate: CardPredicate::creature(),
                mana: Mana::single(Color::Green),
            }],
            text_box: "When this creature enters, earthbend 1. (Target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)\nWhenever you tap a creature for mana, add an additional {G}.".to_string(),
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        });

        // ---- Stage 3: specials ----

        // Fire Nation Cadets {R} 1/2 — Human Soldier
        // This creature has firebending 2 as long as there's a Lesson card
        // in your graveyard. (Whenever this creature attacks, add {R}{R}.
        // This mana lasts until end of combat.)
        // {2}: This creature gets +1/+0 until end of turn.
        self.register_card(CardDefinition {
            name: "Fire Nation Cadets".to_string(),
            mana_cost: Some(ManaCost::parse("R")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            abilities: vec![Ability::Triggered {
                // Conditionally-granted firebending: the attack trigger
                // only exists while a Lesson is in the controller's
                // graveyard (checked when the attack happens).
                condition: TriggerCondition::ActiveIf {
                    active_if: lesson_in_graveyard(),
                    condition: Box::new(TriggerCondition::Attacks {
                        subject: TriggerSubject::This,
                    }),
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RR"),
                    until_end_of_combat: true,
                }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("2"),
                sacrifice_source: false,
                waterbend: false,
                effect: Effect::ModifyUntilEot {
                    power_delta: 1,
                    toughness_delta: 0,
                },
            }],
            text_box: "This creature has firebending 2 as long as there's a Lesson card in your graveyard. (Whenever this creature attacks, add {R}{R}. This mana lasts until end of combat.)\n{2}: This creature gets +1/+0 until end of turn.".to_string(),
            power: Some(1),
            toughness: Some(2),
            ..Default::default()
        });

        // First-Time Flyer {1}{U} 1/2 — Human Pilot Ally
        // Flying. This creature gets +1/+1 as long as there's a Lesson card
        // in your graveyard.
        self.register_card(CardDefinition {
            name: "First-Time Flyer".to_string(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Pilot".into(), "Ally".into()],
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            static_pt_buffs: vec![StaticPtBuff {
                scope: StaticScope::This,
                condition: Some(lesson_in_graveyard()),
                power: 1,
                toughness: 1,
            }],
            text_box: "Flying\nThis creature gets +1/+1 as long as there's a Lesson card in your graveyard.".to_string(),
            power: Some(1),
            toughness: Some(2),
            ..Default::default()
        });

        // Dragonfly Swarm {1}{U}{R} */3 — Dragon Insect
        // Flying, ward {1}. This creature's power is equal to the number of
        // noncreature, nonland cards in your graveyard. When this creature
        // dies, if there's a Lesson card in your graveyard, draw a card.
        // (Engine ward triggers on spells only — see wave doc.)
        self.register_card(CardDefinition {
            name: "Dragonfly Swarm".to_string(),
            mana_cost: Some(ManaCost::parse("1UR")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into(), "Insect".into()],
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            ward: Some(ManaCost::parse("1")),
            power_cda: Some(PowerCda::GraveyardMatching(CardPredicate {
                not_card_types: vec![CardType::Creature, CardType::Land],
                ..CardPredicate::default()
            })),
            abilities: vec![Ability::Triggered {
                // Intervening if (CR 603.4): checked when it dies (ActiveIf)
                // and again on resolution (IfGraveyardAtLeast).
                condition: TriggerCondition::ActiveIf {
                    active_if: lesson_in_graveyard(),
                    condition: Box::new(TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    }),
                },
                effects: vec![Effect::IfGraveyardAtLeast {
                    count: 1,
                    predicate: lesson(),
                    then: vec![Effect::DrawCards { count: 1 }],
                    otherwise: vec![],
                }],
            }],
            text_box: "Flying, ward {1}\nThis creature's power is equal to the number of noncreature, nonland cards in your graveyard.\nWhen this creature dies, if there's a Lesson card in your graveyard, draw a card.".to_string(),
            power: None,
            toughness: Some(3),
            ..Default::default()
        });

        // Compassionate Healer {1}{W} 2/2 — Human Cleric Ally
        // Whenever this creature becomes tapped, you gain 1 life and scry 1.
        self.register_card(CardDefinition {
            name: "Compassionate Healer".to_string(),
            mana_cost: Some(ManaCost::parse("1W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Cleric".into(), "Ally".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::BecomesTapped {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::GainLife { amount: 1 }, Effect::Scry { count: 1 }],
            }],
            text_box: "Whenever this creature becomes tapped, you gain 1 life and scry 1. (Look at the top card of your library. You may put it on the bottom.)".to_string(),
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        });

        // Earth Kingdom Jailer {2}{W} 3/3 — Human Soldier Ally
        // When this creature enters, exile up to one target artifact,
        // creature, or enchantment an opponent controls with mana value 3
        // or greater until this creature leaves the battlefield.
        self.register_card(CardDefinition {
            name: "Earth Kingdom Jailer".to_string(),
            mana_cost: Some(ManaCost::parse("2W")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::ExileUntilSourceLeaves {
                    target: TargetSpec::PermanentOpponentControls {
                        predicate: CardPredicate {
                            card_types_any: vec![
                                CardType::Artifact,
                                CardType::Creature,
                                CardType::Enchantment,
                            ],
                            min_mana_value: Some(3),
                            ..CardPredicate::default()
                        },
                    },
                }],
            }],
            text_box: "When this creature enters, exile up to one target artifact, creature, or enchantment an opponent controls with mana value 3 or greater until this creature leaves the battlefield.".to_string(),
            power: Some(3),
            toughness: Some(3),
            ..Default::default()
        });

        // White Lotus Reinforcements {1}{G}{W} 2/3 — Human Soldier Ally
        // Vigilance. Other Allies you control get +1/+1.
        self.register_card(CardDefinition {
            name: "White Lotus Reinforcements".to_string(),
            mana_cost: Some(ManaCost::parse("1GW")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
            keywords: Keywords {
                vigilance: true,
                ..Default::default()
            },
            static_pt_buffs: vec![StaticPtBuff {
                scope: StaticScope::OtherYouControl(ally()),
                condition: None,
                power: 1,
                toughness: 1,
            }],
            text_box: "Vigilance\nOther Allies you control get +1/+1.".to_string(),
            power: Some(2),
            toughness: Some(3),
            ..Default::default()
        });

        // Earth King's Lieutenant {G}{W} 1/1 — Human Soldier Ally
        // Trample. When this creature enters, put a +1/+1 counter on each
        // other Ally creature you control. Whenever another Ally you
        // control enters, put a +1/+1 counter on this creature.
        self.register_card(CardDefinition {
            name: "Earth King's Lieutenant".to_string(),
            mana_cost: Some(ManaCost::parse("GW")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
            keywords: Keywords {
                trample: true,
                ..Default::default()
            },
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::PutCountersOnEachMatching {
                        count: 1,
                        predicate: CardPredicate {
                            card_type: Some(CardType::Creature),
                            subtype: Some("Ally".to_string()),
                            ..CardPredicate::default()
                        },
                        other: true,
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::AnotherYouControl(ally()),
                    },
                    effects: vec![Effect::PutCountersOnSource { count: 1 }],
                },
            ],
            text_box: "Trample\nWhen this creature enters, put a +1/+1 counter on each other Ally creature you control.\nWhenever another Ally you control enters, put a +1/+1 counter on this creature.".to_string(),
            power: Some(1),
            toughness: Some(1),
            ..Default::default()
        });

        // Suki, Kyoshi Warrior {2}{G/W}{G/W} */4 — Legendary Human Warrior
        // Ally. Suki's power is equal to the number of creatures you
        // control. Whenever Suki attacks, create a 1/1 white Ally creature
        // token that's tapped and attacking.
        // (Hybrid {G/W} registered as {G}{W} — engine deviation, see wave
        // doc; legend rule not enforced, single copy in the deck.)
        self.register_card(CardDefinition {
            name: "Suki, Kyoshi Warrior".to_string(),
            mana_cost: Some(ManaCost::parse("2GW")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["legendary".into()],
            subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
            power_cda: Some(PowerCda::CreaturesYouControl),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Ally".to_string(),
                    count: 1,
                    tapped_and_attacking: true,
                }],
            }],
            text_box: "Suki's power is equal to the number of creatures you control.\nWhenever Suki attacks, create a 1/1 white Ally creature token that's tapped and attacking.".to_string(),
            power: None,
            toughness: Some(4),
            ..Default::default()
        });

        // Yip Yip! {W} Instant — Lesson
        // Target creature you control gets +2/+2 until end of turn. If that
        // creature is an Ally, it also gains flying until end of turn.
        self.register_card(CardDefinition {
            name: "Yip Yip!".to_string(),
            mana_cost: Some(ManaCost::parse("W")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            targeting: vec![TargetRequirement::one(TargetSpec::CreatureYouControl)],
            spell_effects: vec![
                Effect::BuffTarget {
                    power: 2,
                    toughness: 2,
                    target: TargetSpec::CreatureYouControl,
                },
                Effect::IfTargetMatches {
                    predicate: ally(),
                    then: vec![Effect::GrantKeywordsToTarget {
                        keywords: Keywords {
                            flying: true,
                            ..Default::default()
                        },
                        target: TargetSpec::CreatureYouControl,
                    }],
                },
            ],
            text_box: "Target creature you control gets +2/+2 until end of turn. If that creature is an Ally, it also gains flying until end of turn.".to_string(),
            ..Default::default()
        });

        // Fancy Footwork {2}{W} Instant — Lesson
        // Untap one or two target creatures. They each get +2/+2 until end
        // of turn.
        self.register_card(CardDefinition {
            name: "Fancy Footwork".to_string(),
            mana_cost: Some(ManaCost::parse("2W")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            targeting: vec![TargetRequirement {
                spec: TargetSpec::Creature,
                min: 1,
                max: 2,
            }],
            spell_effects: vec![Effect::ForEachTarget {
                effects: vec![
                    Effect::UntapTarget {
                        target: TargetSpec::Creature,
                    },
                    Effect::BuffTarget {
                        power: 2,
                        toughness: 2,
                        target: TargetSpec::Creature,
                    },
                ],
            }],
            text_box: "Untap one or two target creatures. They each get +2/+2 until end of turn."
                .to_string(),
            ..Default::default()
        });

        // Enter the Avatar State {W} Instant — Lesson
        // Until end of turn, target creature you control becomes an Avatar
        // in addition to its other types and gains flying, first strike,
        // lifelink, and hexproof.
        // (Type addition not modeled — Avatar has no mechanical relevance
        // in Milestone 1; see wave doc.)
        self.register_card(CardDefinition {
            name: "Enter the Avatar State".to_string(),
            mana_cost: Some(ManaCost::parse("W")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            targeting: vec![TargetRequirement::one(TargetSpec::CreatureYouControl)],
            spell_effects: vec![Effect::GrantKeywordsToTarget {
                keywords: Keywords {
                    flying: true,
                    first_strike: true,
                    lifelink: true,
                    hexproof: true,
                    ..Default::default()
                },
                target: TargetSpec::CreatureYouControl,
            }],
            text_box: "Until end of turn, target creature you control becomes an Avatar in addition to its other types and gains flying, first strike, lifelink, and hexproof.".to_string(),
            ..Default::default()
        });

        // Crossroads of Destiny {1} Sorcery — modal-machinery proof card
        // (no Milestone-1 deck card is modal; the framework is exercised
        // here). Choose one — You gain 3 life; or draw a card.
        // not a real card — invented for the modal framework; not on
        // Scryfall, excluded from the conformance fixture.
        self.register_card(CardDefinition {
            name: "Crossroads of Destiny".to_string(),
            mana_cost: Some(ManaCost::parse("1")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::GainLife { amount: 3 }],
                    vec![Effect::DrawCards { count: 1 }],
                ],
            }],
            text_box: "Choose one —\n• You gain 3 life.\n• Draw a card.".to_string(),
            ..Default::default()
        });
    }

    fn register_tla_tokens(&mut self) {
        // 1/1 white Ally creature token.
        self.register_card(CardDefinition {
            name: "Ally".to_string(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Ally".into()],
            color_override: Some(Colors::from([Color::White])),
            is_token: true,
            text_box: String::new(),
            power: Some(1),
            toughness: Some(1),
            ..Default::default()
        });

        // Clue — artifact token: "{2}, Sacrifice this token: Draw a card."
        self.register_card(CardDefinition {
            name: "Clue".to_string(),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Clue".into()],
            is_token: true,
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("2"),
                sacrifice_source: true,
                waterbend: false,
                effect: Effect::DrawCards { count: 1 },
            }],
            text_box: "{2}, Sacrifice this token: Draw a card.".to_string(),
            ..Default::default()
        });
    }
}
