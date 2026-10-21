// elemental.rs
// Elemental combat and Learn registrations. Rules execute as typed engine
// operations; text is independently checked against the Scryfall snapshot.

use super::alpha::CardRegistry;
use crate::state::{
    ability::{
        Ability, Effect, EffectValue, StaticCondition, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords, PowerCda,
        SpellCostReduction, StaticPtBuff, StaticScope,
    },
    mana::{Colors, Mana, ManaCost},
    predicate::CardPredicate,
};

impl CardRegistry {
    pub(super) fn register_elemental(&mut self) {
        self.register_card(CardDefinition {
            name: "Professor of Symbology".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Kor".into(), "Cleric".into()],
            text_box: "When this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Gnarled Professor".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Treefolk".into(), "Druid".into()],
            text_box: "Trample\nWhen this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            keywords: Keywords {
                trample: true,
                ..Default::default()
            },
            power: Some(5),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Eyetwitch".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Eye".into(), "Bat".into()],
            text_box: "Flying\nWhen this creature dies, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Pretending Poxbearers".into(),
            mana_cost: Some(ManaCost::parse("{1}{W/B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Citizen".into(), "Ally".into()],
            text_box: "When this creature dies, create a 1/1 white Ally creature token.".into(),
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Ally".into(),
                    count: 1,
                    tapped_and_attacking: false,
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Messenger Hawk".into(),
            mana_cost: Some(ManaCost::parse("{2}{U/B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into(), "Scout".into()],
            text_box: "Flying\nWhen this creature enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nThis creature gets +2/+0 as long as you've drawn two or more cards this turn.".into(),
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            power: Some(1),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Clue".into(),
                    count: 1,
                    tapped_and_attacking: false,
                }],
            }],
            static_pt_buffs: vec![StaticPtBuff {
                scope: StaticScope::This,
                condition: Some(StaticCondition::CardsDrawnAtLeast { count: 2 }),
                power: 2,
                toughness: 0,
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Knowledge Seeker".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Fox".into(), "Spirit".into()],
            text_box: "Vigilance\nWhenever you draw your second card each turn, put a +1/+1 counter on this creature.\nWhen this creature dies, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")".into(),
            keywords: Keywords {
                vigilance: true,
                ..Default::default()
            },
            power: Some(2),
            toughness: Some(1),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
                    effects: vec![Effect::PutCountersOnSource { count: 1 }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::CreateToken {
                        token_name: "Clue".into(),
                        count: 1,
                        tapped_and_attacking: false,
                    }],
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Enthusiastic Study".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Target creature gets +3/+1 and gains trample until end of turn.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            spell_effects: vec![
                Effect::BuffTarget {
                    power: 3,
                    toughness: 1,
                    target: TargetSpec::Creature,
                },
                Effect::GrantKeywordsToTarget {
                    keywords: Keywords {
                        trample: true,
                        ..Default::default()
                    },
                    target: TargetSpec::Creature,
                },
                Effect::Learn,
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Octopus Form".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            text_box: "Target creature you control gets +1/+1 and gains hexproof until end of turn. Untap it. (It can't be the target of spells or abilities your opponents control.)".into(),
            spell_effects: vec![
                Effect::BuffTarget {
                    power: 1,
                    toughness: 1,
                    target: TargetSpec::CreatureYouControl,
                },
                Effect::GrantKeywordsToTarget {
                    keywords: Keywords {
                        hexproof: true,
                        ..Default::default()
                    },
                    target: TargetSpec::CreatureYouControl,
                },
                Effect::UntapTarget {
                    target: TargetSpec::CreatureYouControl,
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Lightning Strike".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Lightning Strike deals 3 damage to any target.".into(),
            spell_effects: vec![Effect::DealDamage {
                amount: 3,
                target: TargetSpec::CreatureOrPlayer,
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Monastery Swiftspear".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Monk".into()],
            text_box: "Haste\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)".into(),
            keywords: Keywords {
                haste: true,
                ..Default::default()
            },
            power: Some(1),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::YouCastSpell {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                },
                effects: vec![Effect::ModifyUntilEot {
                    power_delta: 1,
                    toughness_delta: 1,
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Haru, Hidden Talent".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Peasant".into(), "Ally".into()],
            text_box: "Whenever another Ally you control enters, earthbend 1. (Target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)".into(),
            supertypes: vec!["legendary".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate::subtype("Ally")),
                },
                effects: vec![Effect::Earthbend {
                    count: 1,
                    target: TargetSpec::LandYouControl,
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Toph, the Blind Bandit".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
            text_box: "When Toph enters, earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)\nToph's power is equal to the number of +1/+1 counters on lands you control.".into(),
            supertypes: vec!["legendary".into()],
            power: Some(0),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Earthbend {
                    count: 2,
                    target: TargetSpec::LandYouControl,
                }],
            }],
            power_cda: Some(PowerCda::CountersOnLandsYouControl),
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Cat-Gator".into(),
            mana_cost: Some(ManaCost::parse("{6}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Fish".into(), "Crocodile".into()],
            text_box: "Lifelink\nWhen this creature enters, it deals damage equal to the number of Swamps you control to any target.".into(),
            keywords: Keywords {
                lifelink: true,
                ..Default::default()
            },
            power: Some(3),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DealDamageValue {
                    amount: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate::subtype("Swamp"),
                    },
                    target: TargetSpec::CreatureOrPlayer,
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Uncle Iroh".into(),
            mana_cost: Some(ManaCost::parse("{1}{R/G}{R/G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Noble".into(), "Ally".into()],
            text_box: "Firebending 1 (Whenever this creature attacks, add {R}. This mana lasts until end of combat.)\nLesson spells you cast cost {1} less to cast.".into(),
            supertypes: vec!["legendary".into()],
            power: Some(4),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("R"),
                    until_end_of_combat: true,
                }],
            }],
            spell_cost_reductions: vec![SpellCostReduction {
                first_on_own_turn: false,
                predicate: CardPredicate::subtype("Lesson"),
                condition: StaticCondition::Always,
                generic: 1,
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Spectral Sailor".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into(), "Pirate".into()],
            text_box: "Flash (You may cast this spell any time you could cast an instant.)\nFlying\n{3}{U}: Draw a card.".into(),
            keywords: Keywords {
                flying: true,
                flash: true,
                ..Default::default()
            },
            power: Some(1),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: false,
                mana_cost: ManaCost::parse("3U"),
                sacrifice_source: false,
                waterbend: false,
                effect: Effect::DrawCards { count: 1 },
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Giant Cindermaw".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into(), "Beast".into()],
            text_box: "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nPlayers can't gain life.".into(),
            keywords: Keywords {
                trample: true,
                ..Default::default()
            },
            power: Some(4),
            toughness: Some(3),
            prevents_life_gain: true,
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Tiger-Dillo".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Armadillo".into()],
            text_box: "This creature can't attack or block unless you control another creature with power 4 or greater.".into(),
            power: Some(4),
            toughness: Some(3),
            combat_requires_other: Some(CardPredicate {
                card_type: Some(CardType::Creature),
                min_power: Some(4),
                ..Default::default()
            }),
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Savor".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Target creature gets -2/-2 until end of turn. Create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")".into(),
            spell_effects: vec![
                Effect::BuffTarget {
                    power: -2,
                    toughness: -2,
                    target: TargetSpec::Creature,
                },
                Effect::CreateToken {
                    token_name: "Food".into(),
                    count: 1,
                    tapped_and_attacking: false,
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Fists of Flame".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Draw a card. Until end of turn, target creature gains trample and gets +1/+0 for each card you've drawn this turn. (A creature with trample can deal excess combat damage to the player or planeswalker it's attacking.)".into(),
            spell_effects: vec![
                Effect::DrawCards { count: 1 },
                Effect::BuffTargetValue {
                    power: EffectValue::CardsDrawnMinus { subtract: 0 },
                    toughness: 0,
                    target: TargetSpec::Creature,
                },
                Effect::GrantKeywordsToTarget {
                    keywords: Keywords {
                        trample: true,
                        ..Default::default()
                    },
                    target: TargetSpec::Creature,
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Power Word Kill".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Destroy target non-Angel, non-Demon, non-Devil, non-Dragon creature.".into(),
            spell_effects: vec![Effect::Destroy {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        not_subtypes: vec![
                            "Angel".into(),
                            "Demon".into(),
                            "Devil".into(),
                            "Dragon".into(),
                        ],
                        ..Default::default()
                    },
                },
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Epic Downfall".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: "Exile target creature with mana value 3 or greater.".into(),
            spell_effects: vec![Effect::Exile {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        min_mana_value: Some(3),
                        ..Default::default()
                    },
                },
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Corrupt Court Official".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Advisor".into()],
            text_box: "When this creature enters, target opponent discards a card.".into(),
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DiscardTargetPlayer {
                    count: 1,
                    target: TargetSpec::Opponent,
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Ozai's Cruelty".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            subtypes: vec!["Lesson".into()],
            text_box:
                "Ozai's Cruelty deals 2 damage to target player. That player discards two cards."
                    .into(),
            spell_effects: vec![
                Effect::DealDamage {
                    amount: 2,
                    target: TargetSpec::Player,
                },
                Effect::DiscardTargetPlayer {
                    count: 2,
                    target: TargetSpec::Player,
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Shared Roots".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            subtypes: vec!["Lesson".into()],
            text_box: "Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.".into(),
            spell_effects: vec![Effect::SearchBasicLand { subtype: None }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Field Trip".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: "Search your library for a basic Forest card, put that card onto the battlefield tapped, then shuffle.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            spell_effects: vec![
                Effect::SearchBasicLand {
                    subtype: Some("Forest".into()),
                },
                Effect::Learn,
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Abandon Attachments".into(),
            mana_cost: Some(ManaCost::parse("{1}{U/R}")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            text_box: "You may discard a card. If you do, draw two cards.".into(),
            spell_effects: vec![Effect::OptionalDiscardDraw { count: 2 }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Boomerang Basics".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            subtypes: vec!["Lesson".into()],
            text_box: "Return target nonland permanent to its owner's hand. If you controlled that permanent, draw a card.".into(),
            spell_effects: vec![Effect::ReturnToHandDrawIfControlled {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Land],
                        ..Default::default()
                    },
                },
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Master Piandao".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
            text_box: "First strike\nWhenever Master Piandao attacks, look at the top four cards of your library. You may reveal an Ally, Equipment, or Lesson card from among them and put it into your hand. Put the rest on the bottom of your library in a random order.".into(),
            supertypes: vec!["legendary".into()],
            keywords: Keywords {
                first_strike: true,
                ..Default::default()
            },
            power: Some(4),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::LookAndSelect {
                    reveal: true,
                    look: 4,
                    min_select: 0,
                    max_select: 1,
                    predicate: CardPredicate {
                        subtypes_any: vec!["Ally".into(), "Equipment".into(), "Lesson".into()],
                        ..Default::default()
                    },
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Foggy Swamp Spirit Keeper".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Druid".into(), "Ally".into()],
            text_box: "Lifelink\nWhenever you draw your second card each turn, create a 1/1 colorless Spirit creature token with \"This token can't block or be blocked by non-Spirit creatures.\"".into(),
            keywords: Keywords {
                lifelink: true,
                ..Default::default()
            },
            power: Some(2),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
                effects: vec![Effect::CreateToken {
                    token_name: "Spirit".into(),
                    count: 1,
                    tapped_and_attacking: false,
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Wartime Protestors".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
            text_box: "Haste\nWhenever another Ally you control enters, put a +1/+1 counter on that creature and it gains haste until end of turn.".into(),
            keywords: Keywords {
                haste: true,
                ..Default::default()
            },
            power: Some(4),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate::subtype("Ally")),
                },
                effects: vec![Effect::ModifyTriggeringPermanent {
                    counters: 1,
                    keywords: Keywords {
                        haste: true,
                        ..Default::default()
                    },
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Food".into(),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Food".into()],
            is_token: true,
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("2"),
                sacrifice_source: true,
                waterbend: false,
                effect: Effect::GainLife { amount: 3 },
            }],
            text_box: "{2}, {T}, Sacrifice this token: You gain 3 life.".into(),
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Spirit".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into()],
            is_token: true,
            power: Some(1),
            toughness: Some(1),
            color_override: Some(Colors::new()),
            block_restriction: Some(CardPredicate {
                not_subtypes: vec!["Spirit".into()],
                ..Default::default()
            }),
            blocking_restriction: Some(CardPredicate {
                not_subtypes: vec!["Spirit".into()],
                ..Default::default()
            }),
            text_box: "This token can't block or be blocked by non-Spirit creatures.".into(),
            ..Default::default()
        });
        let weaken = Effect::TargetClause {
            requirement: TargetRequirement::one(TargetSpec::Creature),
            effects: vec![Effect::BuffTarget {
                power: -1,
                toughness: -1,
                target: TargetSpec::Creature,
            }],
        };
        let counter = Effect::TargetClause {
            requirement: TargetRequirement::one(TargetSpec::Creature),
            effects: vec![Effect::PutCounters {
                count: 1,
                target: TargetSpec::Creature,
            }],
        };
        self.register_card(CardDefinition {
            name: "Azula Always Lies".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            text_box: "Choose one or both —\n• Target creature gets -1/-1 until end of turn.\n• Put a +1/+1 counter on target creature.".into(),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![weaken.clone()],
                    vec![counter.clone()],
                    vec![weaken, counter],
                ],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Iroh's Demonstration".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            subtypes: vec!["Lesson".into()],
            text_box: "Choose one —\n• Iroh's Demonstration deals 1 damage to each creature your opponents control.\n• Iroh's Demonstration deals 4 damage to target creature.".into(),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::DamageOpponentCreatures { amount: 1 }],
                    vec![Effect::DealDamage {
                        amount: 4,
                        target: TargetSpec::Creature,
                    }],
                ],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Origin of Metalbending".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            text_box: "Choose one —\n• Destroy target artifact or enchantment.\n• Put a +1/+1 counter on target creature you control. It gains indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)".into(),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::Destroy {
                        target: TargetSpec::Permanent {
                            predicate: CardPredicate {
                                card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                                ..Default::default()
                            },
                        },
                    }],
                    vec![
                        Effect::PutCounters {
                            count: 1,
                            target: TargetSpec::CreatureYouControl,
                        },
                        Effect::GrantKeywordsToTarget {
                            keywords: Keywords {
                                indestructible: true,
                                ..Default::default()
                            },
                            target: TargetSpec::CreatureYouControl,
                        },
                    ],
                ],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Gather the White Lotus".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: "Create a 1/1 white Ally creature token for each Plains you control. Scry 2. (Look at the top two cards of your library, then put any number of them on the bottom and the rest on top in any order.)".into(),
            spell_effects: vec![
                Effect::CreateTokensValue {
                    token_name: "Ally".into(),
                    count: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate::subtype("Plains"),
                    },
                },
                Effect::Scry { count: 2 },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Zuko's Exile".into(),
            mana_cost: Some(ManaCost::parse("{5}")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            text_box: "Exile target artifact, creature, or enchantment. Its controller creates a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")".into(),
            spell_effects: vec![Effect::ExileWithControllerToken {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Artifact, CardType::Creature, CardType::Enchantment],
                        ..Default::default()
                    },
                },
                token_name: "Clue".into(),
            }],
            ..Default::default()
        });
        let earthbend = Effect::TargetClause {
            requirement: TargetRequirement::one(TargetSpec::LandYouControl),
            effects: vec![Effect::Earthbend {
                count: 1,
                target: TargetSpec::LandYouControl,
            }],
        };
        self.register_card(CardDefinition {
            name: "Dai Li Agents".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(3),
            toughness: Some(4),
            text_box: "When this creature enters, earthbend 1, then earthbend 1. (To earthbend 1, target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)\nWhenever this creature attacks, each opponent loses X life and you gain X life, where X is the number of creatures you control with +1/+1 counters on them.".into(),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield { subject: TriggerSubject::This },
                    effects: vec![earthbend.clone(), earthbend],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Attacks { subject: TriggerSubject::This },
                    effects: vec![Effect::DrainLifeValue { amount: EffectValue::CreaturesWithCounters }],
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Jet, Freedom Fighter".into(),
            mana_cost: Some(ManaCost::parse("{2}{R/W}{R/W}{R/W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
            supertypes: vec!["legendary".into()],
            power: Some(3),
            toughness: Some(1),
            text_box: "When Jet enters, he deals damage equal to the number of creatures you control to target creature an opponent controls.\nWhen Jet dies, put a +1/+1 counter on each of up to two target creatures.".into(),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield { subject: TriggerSubject::This },
                    effects: vec![Effect::DealDamageValue {
                        amount: EffectValue::BattlefieldMatching { predicate: CardPredicate::creature() },
                        target: TargetSpec::CreatureOpponentControls,
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Dies { subject: TriggerSubject::This },
                    effects: vec![Effect::TargetClause {
                        requirement: TargetRequirement::up_to(2, TargetSpec::Creature),
                        effects: vec![Effect::ForEachTarget {
                            effects: vec![Effect::PutCounters { count: 1, target: TargetSpec::Creature }],
                        }],
                    }],
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Ostrich-Horse".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into(), "Horse".into()],
            power: Some(3),
            toughness: Some(1),
            text_box: "When this creature enters, mill three cards. You may put a land card from among them into your hand. If you don't, put a +1/+1 counter on this creature. (To mill three cards, put the top three cards of your library into your graveyard.)".into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield { subject: TriggerSubject::This },
                effects: vec![Effect::MillAndSelect {
                    count: 3,
                    predicate: CardPredicate { card_type: Some(CardType::Land), ..Default::default() },
                    if_declined: vec![Effect::PutCountersOnSource { count: 1 }],
                }],
            }],
            ..Default::default()
        });
        let land = TargetSpec::Permanent {
            predicate: CardPredicate {
                card_type: Some(CardType::Land),
                ..Default::default()
            },
        };
        self.register_card(CardDefinition {
            name: "Sokka's Haiku".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            types: CardTypes::new([CardType::Instant]),
            subtypes: vec!["Lesson".into()],
            text_box:
                "Counter target spell.\nDraw a card, then mill three cards.\nUntap target land."
                    .into(),
            spell_effects: vec![
                Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::Spell),
                    effects: vec![Effect::CounterSpell {
                        target: TargetSpec::Spell,
                    }],
                },
                Effect::DrawCards { count: 1 },
                Effect::MillCards { count: 3 },
                Effect::TargetClause {
                    requirement: TargetRequirement::one(land.clone()),
                    effects: vec![Effect::UntapTarget { target: land }],
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Momo, Friendly Flier".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Lemur".into(), "Bat".into(), "Ally".into()],
            supertypes: vec!["legendary".into()],
            power: Some(1),
            toughness: Some(1),
            text_box: "Flying\nThe first non-Lemur creature spell with flying you cast during each of your turns costs {1} less to cast.\nWhenever another creature you control with flying enters, Momo gets +1/+1 until end of turn.".into(),
            keywords: Keywords { flying: true, ..Default::default() },
            spell_cost_reductions: vec![SpellCostReduction {
                first_on_own_turn: true,
                generic: 1,
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    flying: true,
                    not_subtypes: vec!["Lemur".into()],
                    ..Default::default()
                },
                condition: StaticCondition::Always,
            }],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate {
                        card_type: Some(CardType::Creature),
                        flying: true,
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::ModifyUntilEot { power_delta: 1, toughness_delta: 1 }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Deserter's Disciple".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
            power: Some(2),
            toughness: Some(2),
            text_box: "{T}: Another target creature you control with power 2 or less can't be blocked this turn.".into(),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("0"),
                sacrifice_source: false,
                waterbend: false,
                effect: Effect::CantBeBlockedTarget {
                    target: TargetSpec::PermanentYouControl {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Creature),
                            max_power: Some(2),
                            ..Default::default()
                        },
                    },
                },
            }],
            ..Default::default()
        });
        let permanent_card = TargetSpec::GraveyardCardYouOwn {
            predicate: CardPredicate {
                card_types_any: vec![
                    CardType::Artifact,
                    CardType::Creature,
                    CardType::Enchantment,
                    CardType::Land,
                    CardType::Planeswalker,
                    CardType::Battle,
                ],
                ..Default::default()
            },
        };
        self.register_card(CardDefinition {
            name: "True Ancestry".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            subtypes: vec!["Lesson".into()],
            text_box: "Return up to one target permanent card from your graveyard to your hand.\nCreate a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")".into(),
            spell_effects: vec![
                Effect::TargetClause {
                    requirement: TargetRequirement::up_to(1, permanent_card.clone()),
                    effects: vec![Effect::ReturnGraveyardTarget { target: permanent_card }],
                },
                Effect::CreateToken { token_name: "Clue".into(), count: 1, tapped_and_attacking: false },
            ],
            ..Default::default()
        });
    }
}
