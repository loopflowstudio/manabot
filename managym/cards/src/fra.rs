//! Reality Fracture.
use super::enters_prepared;
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, KeywordAction, ObjectSelector, Per, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        LayerValue, PrintedCharacteristics, StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        prudent_fateseer(),
        diviner_of_victory(),
        CardDefinition {
            key: "fra.blossom_blessed_angel",
            scryfall_id: "5e77fbf0-9d1f-4e7d-a02e-43065d11b0d9",
            printed: PrintedCharacteristics {
                name: "Blossom-Blessed Angel".into(),
                mana_cost: Some(ManaCost::parse("{3}{W}")),
                subtypes: vec!["Angel".into(), "Cleric".into()],
                power: Some(2),
                toughness: Some(4),
                keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance]),
                text_box: "Flying, vigilance\nThis creature enters prepared.".into(),
                ..enters_prepared(seed_suture())
            },
            ..Default::default()
        },
        CardDefinition {
            key: "fra.emergency_phytomedic",
            scryfall_id: "de94d388-919d-44ff-baef-8c90a417ac6d",
            printed: PrintedCharacteristics {
                name: "Emergency Phytomedic".into(),
                mana_cost: Some(ManaCost::parse("{G/W}")),
                subtypes: vec!["Dryad".into(), "Cleric".into()],
                power: Some(1),
                toughness: Some(1),
                text_box: "This creature enters prepared.".into(),
                ..enters_prepared(seed_suture())
            },
            ..Default::default()
        },
        carnivorous_cultivator(),
        fatehold_chronologist(),
        hallway_heckler(),
        konstrari_improviser(),
        pompous_battlemage(),
        semester_foreseer(),
        theorix_metamage(),
        vigorbloom_vanguard(),
        void_extrapolator(),
    ]
}

fn carnivorous_cultivator() -> CardDefinition {
    let text = "Deathtouch\nThis creature enters prepared.\nWhenever this creature deals combat damage to a player, return target land card from your graveyard to your hand.";
    CardDefinition {
        key: "fra.carnivorous_cultivator",
        scryfall_id: "79dd5c54-5ea5-47b5-8f9b-50ed57a5ea45",
        printed: PrintedCharacteristics {
            name: "Carnivorous Cultivator".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            text_box: text.into(),
            subtypes: vec!["Elf".into(), "Warlock".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::This,
                    to: Actor::AnyPlayer,
                    per: Per::Object,
                },
                intervening_if: None,
                effects: vec![Effect::return_to_hand(ObjectSelector::Target(
                    TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Land),
                            ..Default::default()
                        },
                    },
                ))],
            }],
            ..enters_prepared(enroot())
        },
        ..Default::default()
    }
}

fn fatehold_chronologist() -> CardDefinition {
    let text = "Flying\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "fra.fatehold_chronologist",
        scryfall_id: "29e7ec16-0c16-48aa-8e09-ce6e0d5bd40b",
        printed: PrintedCharacteristics {
            name: "Fatehold Chronologist".into(),
            mana_cost: Some(ManaCost::parse("{1}{W/U}")),
            text_box: text.into(),
            subtypes: vec!["Bird".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying]),
            ..enters_prepared(peer_review())
        },
        ..Default::default()
    }
}

fn hallway_heckler() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)\n{T}, Discard a card: Draw a card.";
    CardDefinition {
        key: "fra.hallway_heckler",
        scryfall_id: "7e324816-552f-455d-97c4-5ea6b26d2e6e",
        printed: PrintedCharacteristics {
            name: "Hallway Heckler".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            text_box: text.into(),
            subtypes: vec!["Elemental".into(), "Sorcerer".into()],
            power: Some(2),
            toughness: Some(3),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                discard_card: true,
                effects: vec![Effect::DrawCards { count: 1.into() }],
                ..Default::default()
            }],
            ..enters_prepared(vicious_verse())
        },
        ..Default::default()
    }
}

fn konstrari_improviser() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "fra.konstrari_improviser",
        scryfall_id: "42e28bd2-486b-45d4-8840-6e33c19c2d57",
        printed: PrintedCharacteristics {
            name: "Konstrari Improviser".into(),
            mana_cost: Some(ManaCost::parse("{1}{R/G}")),
            text_box: text.into(),
            subtypes: vec!["Human".into(), "Artificer".into()],
            power: Some(2),
            toughness: Some(2),
            ..enters_prepared(soul_tether())
        },
        ..Default::default()
    }
}

fn pompous_battlemage() -> CardDefinition {
    let text = "Prowess\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "fra.pompous_battlemage",
        scryfall_id: "aa0f77ac-741a-444a-8bf0-a42c644726bf",
        printed: PrintedCharacteristics {
            name: "Pompous Battlemage".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            text_box: text.into(),
            subtypes: vec!["Goblin".into(), "Sorcerer".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(1, 1),
                )],
            }],
            ..enters_prepared(improvised_act())
        },
        ..Default::default()
    }
}

fn semester_foreseer() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)\nWhen this creature enters, surveil 1.";
    CardDefinition {
        key: "fra.semester_foreseer",
        scryfall_id: "7e5a640b-fd88-4b5a-9dfc-1d8f5a5cef41",
        printed: PrintedCharacteristics {
            name: "Semester Foreseer".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            text_box: text.into(),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(3),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::Surveil { count: 1 }],
            }],
            ..enters_prepared(peer_review())
        },
        ..Default::default()
    }
}

fn theorix_metamage() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)\nThreshold — This creature gets +1/+0 and has flying as long as there are seven or more cards in your graveyard.";
    CardDefinition {
        key: "fra.theorix_metamage",
        scryfall_id: "fb6bad96-841d-4738-8e62-92f346f914fd",
        printed: PrintedCharacteristics {
            name: "Theorix Metamage".into(),
            mana_cost: Some(ManaCost::parse("{2}{U/B}")),
            text_box: text.into(),
            subtypes: vec!["Shade".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(3),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::This,
                condition: Some(Condition::GraveyardAtLeast {
                    count: 7,
                    predicate: CardPredicate::default(),
                }),
                characteristic_defining: false,
                modifications: vec![
                    Modification::ModifyPt {
                        power: LayerValue::Fixed(1),
                        toughness: LayerValue::Fixed(0),
                    },
                    Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
                ],
            })],
            ..enters_prepared(omit_variables())
        },
        ..Default::default()
    }
}

fn vigorbloom_vanguard() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)\nEach creature you control with a +1/+1 counter on it has vigilance.";
    CardDefinition {
        key: "fra.vigorbloom_vanguard",
        scryfall_id: "acefc515-bf97-4dc0-b0f7-ae8ae5a61671",
        printed: PrintedCharacteristics {
            name: "Vigorbloom Vanguard".into(),
            mana_cost: Some(ManaCost::parse("{1}{G/W}")),
            text_box: text.into(),
            subtypes: vec!["Troll".into(), "Druid".into()],
            power: Some(2),
            toughness: Some(2),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::YouControl(CardPredicate {
                    with_counter: true,
                    ..CardPredicate::creature()
                }),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::AddKeywords(Keywords::of(&[
                    Keyword::Vigilance,
                ]))],
            })],
            ..enters_prepared(seed_suture())
        },
        ..Default::default()
    }
}

fn void_extrapolator() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)\nThreshold — This creature gets +1/+1 as long as there are seven or more cards in your graveyard.";
    CardDefinition {
        key: "fra.void_extrapolator",
        scryfall_id: "0eae2efb-bf25-48ee-9c07-9098008110ad",
        printed: PrintedCharacteristics {
            name: "Void Extrapolator".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            text_box: text.into(),
            subtypes: vec!["Aetherborn".into(), "Warlock".into()],
            power: Some(2),
            toughness: Some(2),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::This,
                condition: Some(Condition::GraveyardAtLeast {
                    count: 7,
                    predicate: CardPredicate::default(),
                }),
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(1),
                    toughness: LayerValue::Fixed(1),
                }],
            })],
            ..enters_prepared(omit_variables())
        },
        ..Default::default()
    }
}

fn enroot() -> PrintedCharacteristics {
    PrintedCharacteristics {
        name: "Enroot".into(),
        mana_cost: Some(ManaCost::parse("{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: "Search your library for a land card, put it into your graveyard, then shuffle."
            .into(),
        spell_effects: vec![Effect::search_library(
            CardPredicate {
                card_type: Some(CardType::Land),
                ..Default::default()
            },
            1.into(),
            vec![Effect::move_to(
                ObjectSelector::Selected,
                ZoneType::Graveyard,
            )],
        )],
        ..Default::default()
    }
}

fn peer_review() -> PrintedCharacteristics {
    PrintedCharacteristics {
        name: "Peer Review".into(),
        mana_cost: Some(ManaCost::parse("{2}{W/U}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: "Create a 2/2 colorless Wizard Soldier creature token named Cadet. Surveil 1."
            .into(),
        spell_effects: vec![
            Effect::create_tokens("token.cadet", 1),
            Effect::Surveil { count: 1 },
        ],
        ..Default::default()
    }
}

fn vicious_verse() -> PrintedCharacteristics {
    let text = "Vicious Verse deals 1 damage to target opponent.";
    PrintedCharacteristics {
        name: "Vicious Verse".into(),
        mana_cost: Some(ManaCost::parse("{B/R}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::deal_damage(
            1,
            ObjectSelector::Target(TargetSpec::Opponent),
        )],
        ..Default::default()
    }
}

fn soul_tether() -> PrintedCharacteristics {
    let text =
        "Create a Heartwood token. (It's a red and green artifact with \"{T}: Add {R} or {G}.\")";
    PrintedCharacteristics {
        name: "Soul Tether".into(),
        mana_cost: Some(ManaCost::parse("{2}{R/G}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::create_tokens("token.heartwood", 1)],
        ..Default::default()
    }
}

fn improvised_act() -> PrintedCharacteristics {
    let text = "You may discard a card. If you do, draw a card.";
    PrintedCharacteristics {
        name: "Improvised Act".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::discard_draw(1, 1)],
        ..Default::default()
    }
}

fn omit_variables() -> PrintedCharacteristics {
    PrintedCharacteristics {
        name: "Omit Variables".into(),
        mana_cost: Some(ManaCost::parse("{U/B}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box:
            "Mill three cards. (Put the top three cards of your library into your graveyard.)"
                .into(),
        spell_effects: vec![Effect::MillCards {
            count: 3.into(),
            target: None,
        }],
        ..Default::default()
    }
}

fn seed_suture() -> PrintedCharacteristics {
    let text = "Put a +1/+1 counter on target creature. You gain 1 life.";
    PrintedCharacteristics {
        name: "Seed Suture".into(),
        mana_cost: Some(ManaCost::parse("{G/W}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![
            Effect::put_counters(
                ObjectSelector::Target(TargetSpec::Creature),
                CounterKind::PlusOnePlusOne,
                1,
            ),
            Effect::gain_life(1),
        ],
        ..Default::default()
    }
}

fn diviner_of_victory() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)\nWhenever you scry or surveil, this creature gets +1/+1 until end of turn.";
    let spell_text = "Return target creature an opponent controls with mana value 3 or less to its owner's hand. Surveil 1.";
    CardDefinition {
        key: "fra.diviner_of_victory",
        scryfall_id: "0853bb80-8664-432a-8457-600139fd96d5",
        printed: PrintedCharacteristics {
            name: "Diviner of Victory".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            subtypes: vec!["Dwarf".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Performs {
                    who: Actor::You,
                    any_of: vec![KeywordAction::Scry, KeywordAction::Surveil],
                },
                intervening_if: None,
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(1, 1),
                )],
            }],
            text_box: text.into(),
            ..enters_prepared(PrintedCharacteristics {
                name: "Unwind History".into(),
                mana_cost: Some(ManaCost::parse("{1}{U}")),
                types: CardTypes::new([CardType::Sorcery]),
                text_box: spell_text.into(),
                spell_effects: vec![
                    Effect::return_to_hand(ObjectSelector::Target(
                        TargetSpec::PermanentOpponentControls {
                            predicate: CardPredicate {
                                max_mana_value: Some(3),
                                ..CardPredicate::creature()
                            },
                        },
                    )),
                    Effect::Surveil { count: 1 },
                ],
                ..Default::default()
            })
        },
        ..Default::default()
    }
}

fn prudent_fateseer() -> CardDefinition {
    let text = "This creature enters prepared.\nWhenever you scry or surveil, creatures you control get +1/+0 until end of turn. This ability triggers only once each turn.";
    CardDefinition {
        key: "fra.prudent_fateseer",
        scryfall_id: "6c1c790b-9e0e-4964-9ea3-554843907f06",
        printed: PrintedCharacteristics {
            name: "Prudent Fateseer".into(),
            mana_cost: Some(ManaCost::parse("{1}{W/U}{W/U}")),
            subtypes: vec!["Dwarf".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::OnceEachTurn {
                    condition: Box::new(TriggerCondition::Performs {
                        who: Actor::You,
                        any_of: vec![KeywordAction::Scry, KeywordAction::Surveil],
                    }),
                },
                intervening_if: None,
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::creature()),
                    Modification::pump(1, 0),
                )],
            }],
            text_box: text.into(),
            ..enters_prepared(peer_review())
        },
        ..Default::default()
    }
}
