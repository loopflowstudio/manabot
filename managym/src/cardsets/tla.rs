// tla.rs
// Avatar: The Last Airbender — Milestone-1 two-deck slice, Stage 1 cards.
// Every card here is registered with its full oracle behavior; cards whose
// text needs Stage 2+ machinery (choices, costs, statics) are not registered.

use crate::state::{
    ability::{Ability, Effect, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords,
        TriggeredManaAbility,
    },
    mana::{Color, Colors, Mana, ManaCost},
    predicate::CardPredicate,
};

use super::alpha::CardRegistry;

fn ally() -> CardPredicate {
    CardPredicate::subtype("Ally")
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
            text_box:
                "Flash\nWhen this creature enters, create a 1/1 white Ally creature token."
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

        // Glider Kids {1}{U} 2/1 — Human Ally
        // When this creature enters, scry 1.
        self.register_card(CardDefinition {
            name: "Glider Kids".to_string(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Ally".into()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Scry { count: 1 }],
            }],
            text_box: "When this creature enters, scry 1.".to_string(),
            power: Some(2),
            toughness: Some(1),
            ..Default::default()
        });

        // Firebending Lesson {1}{R} Sorcery — Lesson
        // Kicker {3}. Firebending Lesson deals 2 damage to any target. If
        // this spell was kicked, it deals 5 damage instead.
        self.register_card(CardDefinition {
            name: "Firebending Lesson".to_string(),
            mana_cost: Some(ManaCost::parse("1R")),
            types: CardTypes::new([CardType::Sorcery]),
            subtypes: vec!["Lesson".into()],
            kicker: Some(ManaCost::parse("3")),
            spell_effects: vec![Effect::IfKicked {
                then: vec![Effect::DealDamage {
                    amount: 5,
                    target: TargetSpec::CreatureOrPlayer,
                }],
                otherwise: vec![Effect::DealDamage {
                    amount: 2,
                    target: TargetSpec::CreatureOrPlayer,
                }],
            }],
            text_box: "Kicker {3}\nFirebending Lesson deals 2 damage to any target. If this spell was kicked, it deals 5 damage instead.".to_string(),
            ..Default::default()
        });

        // It'll Quench Ya! {1}{U} Instant
        // Counter target spell unless its controller pays {2}.
        self.register_card(CardDefinition {
            name: "It'll Quench Ya!".to_string(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![TargetRequirement::one(TargetSpec::Spell)],
            spell_effects: vec![Effect::CounterUnlessPays {
                cost: ManaCost::parse("2"),
            }],
            text_box: "Counter target spell unless its controller pays {2}.".to_string(),
            ..Default::default()
        });

        // Accumulate Wisdom {1}{U} Instant — Lesson
        // Look at the top three cards of your library. Put one of them into
        // your hand and the rest on the bottom of your library in a random
        // order. If there are three or more Lesson cards in your graveyard,
        // put all three into your hand instead.
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
            text_box: "Look at the top three cards of your library. Put one of them into your hand and the rest on the bottom of your library in a random order. If there are three or more Lesson cards in your graveyard, put all three into your hand instead.".to_string(),
            ..Default::default()
        });

        // Water Tribe Rallier {1}{G} 2/1 — Human Warrior Ally
        // Waterbend {5}: Look at the top four cards of your library. You may
        // reveal a creature card with power 3 or less from among them and
        // put it into your hand. Put the rest on the bottom of your library
        // in a random order.
        self.register_card(CardDefinition {
            name: "Water Tribe Rallier".to_string(),
            mana_cost: Some(ManaCost::parse("1G")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
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
            text_box: "Waterbend {5}: Look at the top four cards of your library. You may reveal a creature card with power 3 or less from among them and put it into your hand. Put the rest on the bottom of your library in a random order. (You may tap untapped artifacts and creatures you control to help pay waterbend costs; each pays {1}.)".to_string(),
            power: Some(2),
            toughness: Some(1),
            ..Default::default()
        });

        // Allies at Last {4}{W} Sorcery
        // Affinity for Allies. Up to two target creatures you control each
        // deal damage equal to their power to target creature an opponent
        // controls.
        self.register_card(CardDefinition {
            name: "Allies at Last".to_string(),
            mana_cost: Some(ManaCost::parse("4W")),
            types: CardTypes::new([CardType::Sorcery]),
            cost_reduction_per: Some(ally()),
            targeting: vec![
                TargetRequirement::up_to(2, TargetSpec::CreatureYouControl),
                TargetRequirement::one(TargetSpec::CreatureOpponentControls),
            ],
            spell_effects: vec![Effect::TargetCreaturesDealPowerDamageToLastTarget],
            text_box: "Affinity for Allies (This spell costs {1} less to cast for each Ally you control.)\nUp to two target creatures you control each deal damage equal to their power to target creature an opponent controls.".to_string(),
            ..Default::default()
        });

        // Badgermole Cub {1}{G} 1/2 — Badger
        // Whenever you tap a creature for mana, add {G}. (Triggered mana
        // ability, CR 605.1b — no stack; composes with waterbend payments.
        // Earthbend lands in Stage 3.)
        self.register_card(CardDefinition {
            name: "Badgermole Cub".to_string(),
            mana_cost: Some(ManaCost::parse("1G")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Badger".into(), "Mole".into()],
            triggered_mana_abilities: vec![TriggeredManaAbility {
                predicate: CardPredicate::creature(),
                mana: Mana::single(Color::Green),
            }],
            text_box: "Whenever you tap a creature for mana, add {G}.".to_string(),
            power: Some(1),
            toughness: Some(2),
            ..Default::default()
        });

        // Crossroads of Destiny {1} Sorcery — modal-machinery proof card
        // (no Milestone-1 deck card is modal; the framework is exercised
        // here). Choose one — You gain 3 life; or draw a card.
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
