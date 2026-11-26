//! Core Set 2019.
use managym_state::ability::{
    Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, Per, TargetSpec,
    TriggerCondition, TriggerSubject, TurnStep,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, EntryChoiceKind, Keyword,
    Keywords, PrintedCharacteristics, ReplacementEffect, StaticAbility, StaticScope,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        diamond_mare(),
        leonin_vanguard(),
        volley_veteran(),
        mystic_archaeologist(),
        vampire_neonate(),
        kargan_dragonrider(),
        exclusion_mage(),
        herald_of_faith(),
        skeleton_archer(),
        viashino_pyromancer(),
        aggressive_mammoth(),
        meteor_golem(),
        gigantosaurus(),
        lathliss_dragon_queen(),
        heroic_reinforcements(),
    ]
}

fn leonin_vanguard() -> CardDefinition {
    let text_box = "At the beginning of combat on your turn, if you control three or more creatures, this creature gets +1/+1 until end of turn and you gain 1 life.";
    CardDefinition {
        key: "m19.leonin_vanguard",
        scryfall_id: "724738ad-6a9b-4ef6-b637-558645cd8151",
        printed: PrintedCharacteristics {
            name: "Leonin Vanguard".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Soldier".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: Some(Condition::ControlsAtLeast {
                    count: 3,
                    predicate: CardPredicate::creature(),
                    distinct_names: false,
                }),
                effects: vec![
                    Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::ModifyPt {
                            power: 1.into(),
                            toughness: 1.into(),
                        },
                    ),
                    Effect::gain_life(1),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn volley_veteran() -> CardDefinition {
    let text_box = "When this creature enters, it deals damage to target creature an opponent controls equal to the number of Goblins you control.";
    CardDefinition {
        key: "m19.volley_veteran",
        scryfall_id: "164960fc-6e80-4a53-90d7-5a18c0a28083",
        printed: PrintedCharacteristics {
            name: "Volley Veteran".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(4),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::deal_damage(
                    ObjectSelector::each_you_control(CardPredicate::subtype("Goblin")).count(),
                    ObjectSelector::Target(TargetSpec::CreatureOpponentControls),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn heroic_reinforcements() -> CardDefinition {
    let text_box = "Create two 1/1 white Soldier creature tokens. Until end of turn, creatures you control get +1/+1 and gain haste. (They can attack and {T} this turn.)";
    CardDefinition {
        key: "m19.heroic_reinforcements",
        scryfall_id: "33c35bf8-ae43-41aa-aae9-4d7513f9058c",
        printed: PrintedCharacteristics {
            name: "Heroic Reinforcements".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::create_tokens("token.soldier_1_1_w", 2),
                Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::creature()),
                    Modification::pump(1, 1),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::creature()),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn lathliss_dragon_queen() -> CardDefinition {
    let text_box = "Flying\nWhenever another nontoken Dragon you control enters, create a 5/5 red Dragon creature token with flying.\n{1}{R}: Dragons you control get +1/+0 until end of turn.";
    CardDefinition {
        key: "m19.lathliss_dragon_queen",
        scryfall_id: "54a4c37d-5eeb-42c9-9688-c2ed0d5044cd",
        printed: PrintedCharacteristics {
            name: "Lathliss, Dragon Queen".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dragon".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::another_you_control(CardPredicate {
                    token: Some(false),
                    ..CardPredicate::subtype("Dragon")
                }),
                vec![Effect::create_tokens("token.dragon_5_5_r_flying", 1)],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}{R}"),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::subtype("Dragon")),
                    Modification::pump(1, 0),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gigantosaurus() -> CardDefinition {
    CardDefinition {
        key: "m19.gigantosaurus",
        scryfall_id: "c1db84d8-d426-4c0d-b44e-5be7b0f5f5bf",
        printed: PrintedCharacteristics {
            name: "Gigantosaurus".into(),
            mana_cost: Some(ManaCost::parse("{G}{G}{G}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(10),
            toughness: Some(10),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn meteor_golem() -> CardDefinition {
    CardDefinition {
        key: "m19.meteor_golem",
        scryfall_id: "1bdb0b15-d651-4730-8be9-d0e01145311b",
        printed: PrintedCharacteristics {
            name: "Meteor Golem".into(),
            subtypes: vec!["Golem".into()],
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{7}")),
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::destroy(ObjectSelector::Target(
                    TargetSpec::PermanentOpponentControls {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    },
                ))],
            )],
            text_box:
                "When this creature enters, destroy target nonland permanent an opponent controls."
                    .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn aggressive_mammoth() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nOther creatures you control have trample.";
    CardDefinition {
        key: "m19.aggressive_mammoth",
        scryfall_id: "323f3c76-5e79-43e6-ae78-f555810edbc3",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::OtherYouControl(CardPredicate::creature()),
                None,
                vec![Modification::AddKeywords(Keywords::of(&[Keyword::Trample]))],
            )],
            name: "Aggressive Mammoth".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elephant".into()],
            mana_cost: Some(ManaCost::parse("{3}{G}{G}{G}")),
            power: Some(8),
            toughness: Some(8),
            keywords: Keywords::of(&[Keyword::Trample]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn exclusion_mage() -> CardDefinition {
    let text_box = "When this creature enters, return target creature an opponent controls to its owner's hand.";
    CardDefinition {
        key: "m19.exclusion_mage",
        scryfall_id: "ccad82f5-5c5c-42ad-b66e-942f0d9631ca",
        printed: PrintedCharacteristics {
            name: "Exclusion Mage".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::return_to_hand(ObjectSelector::Target(
                    TargetSpec::CreatureOpponentControls,
                ))],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn herald_of_faith() -> CardDefinition {
    let text_box = "Flying\nWhenever this creature attacks, you gain 2 life.";
    CardDefinition {
        key: "m19.herald_of_faith",
        scryfall_id: "452591ca-7273-4e47-820b-3ff89697a036",
        printed: PrintedCharacteristics {
            name: "Herald of Faith".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into()],
            power: Some(4),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![Effect::gain_life(2)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn skeleton_archer() -> CardDefinition {
    let text_box = "When this creature enters, it deals 1 damage to any target.";
    CardDefinition {
        key: "m19.skeleton_archer",
        scryfall_id: "8fee5cc4-a686-4ce6-aa6b-1b8a88e6dea3",
        printed: PrintedCharacteristics {
            name: "Skeleton Archer".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Skeleton".into(), "Archer".into()],
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::deal_damage(
                    1,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn viashino_pyromancer() -> CardDefinition {
    let text_box = "When this creature enters, it deals 2 damage to target player or planeswalker.";
    CardDefinition {
        key: "m19.viashino_pyromancer",
        scryfall_id: "a82fdb2f-b199-44ff-9615-90fc074cb8b0",
        printed: PrintedCharacteristics {
            name: "Viashino Pyromancer".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Lizard".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::deal_damage(
                    2,
                    ObjectSelector::Target(TargetSpec::PlayerOrPlaneswalker),
                )],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mystic_archaeologist() -> CardDefinition {
    let text_box = "{3}{U}{U}: Draw two cards.";
    CardDefinition {
        key: "m19.mystic_archaeologist",
        scryfall_id: "1b19cad0-5754-4625-8303-c8310bc7cbd5",
        printed: PrintedCharacteristics {
            name: "Mystic Archaeologist".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{U}{U}"),
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vampire_neonate() -> CardDefinition {
    let text_box = "{2}, {T}: Each opponent loses 1 life and you gain 1 life.";
    CardDefinition {
        key: "m19.vampire_neonate",
        scryfall_id: "167822a5-2ab5-42f5-afa4-562fe2d7501b",
        printed: PrintedCharacteristics {
            name: "Vampire Neonate".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into()],
            power: Some(0),
            toughness: Some(3),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}"),
                tap_source: true,
                effects: vec![Effect::DrainLife {
                    amount: EffectValue::Fixed(1),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kargan_dragonrider() -> CardDefinition {
    let text_box = "As long as you control a Dragon, this creature has flying.";
    CardDefinition {
        key: "m19.kargan_dragonrider",
        scryfall_id: "34750304-c536-47d4-922d-a3654c37ffbc",
        printed: PrintedCharacteristics {
            name: "Kargan Dragonrider".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(2),
            statics: vec![StaticAbility::continuous(
                StaticScope::This,
                Some(Condition::ControlsAtLeast {
                    distinct_names: false,
                    count: 1,
                    predicate: CardPredicate::subtype("Dragon"),
                }),
                vec![Modification::AddKeywords(Keywords::of(&[Keyword::Flying]))],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn diamond_mare() -> CardDefinition {
    let text_box = "As this creature enters, choose a color.\nWhenever you cast a spell of the chosen color, you gain 1 life.";
    CardDefinition {
        key: "m19.diamond_mare",
        scryfall_id: "ca600b3f-2c70-489b-b218-6e3245b90114",
        printed: PrintedCharacteristics {
            name: "Diamond Mare".into(),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{2}")),
            subtypes: vec!["Horse".into()],
            power: Some(1),
            toughness: Some(3),
            replacement_effects: vec![ReplacementEffect::AsEntersChoose {
                kind: EntryChoiceKind::Color,
                look_at_opponent_hand: false,
            }],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        chosen_color: true,
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![Effect::gain_life(1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
