//! The Hobbit cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PaymentCost, PlayWindow,
        PlayerSelector, Scope, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
        TurnStep,
    },
    card::{
        CardDefinition, CardType, CardTypes, InsetSpell, Keyword, Keywords, LayerValue,
        PrintedCharacteristics, StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        most_decrepit_old_bird(),
        my_precious(),
        thranduil_sindarin_liege(),
        laketown_mariners(),
        gollum_silent_slinker(),
        gloin_the_mighty(),
        the_arkenstone(),
        bilbo_baggins_burglar(),
        velvetwing_butterflies(),
        bofur_reliable_guardian(),
        beorn_the_fierce(),
        beorn_reluctant_host(),
        smaug_the_great_calamity(),
        bilbo_thief_in_the_night(),
        glamdring_foe_hammer(),
        gandalf_goblins_bane(),
    ]
}

fn beorn_the_fierce() -> CardDefinition {
    let text_box = "Trample\nOther Bears you control get +2/+2.\nAt the beginning of combat on your turn, put a trample counter on up to one target creature you control. It becomes a Bear in addition to its other types. Then if you control three or more Bears, draw two cards.";
    CardDefinition {
        key: "hob.beorn_the_fierce",
        scryfall_id: "367d5f8b-77ee-47f7-bc71-972d62c280a9",
        printed: PrintedCharacteristics {
            name: "Beorn the Fierce".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Bear".into(), "Shapeshifter".into(), "Warrior".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Trample]),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::OtherYouControl(CardPredicate::subtype("Bear")),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(2),
                    toughness: LayerValue::Fixed(2),
                }],
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::up_to(1, TargetSpec::CreatureYouControl),
                    effects: vec![
                        Effect::put_counters(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            CounterKind::Trample,
                            1,
                        ),
                        Effect::indefinitely(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::AddSubtypes(vec!["Bear".into()]),
                        ),
                        Effect::If {
                            condition: Condition::ControlsAtLeast {
                                distinct_names: false,
                                count: 3,
                                predicate: CardPredicate::subtype("Bear"),
                            },
                            then: vec![Effect::DrawCards {
                                count: EffectValue::Fixed(2),
                            }],
                            otherwise: vec![],
                        },
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn smaug_the_great_calamity() -> CardDefinition {
    CardDefinition {
        key: "hob.smaug_the_great_calamity",
        scryfall_id: "419ca9e5-8413-4378-a4ef-eda5a1024218",
        printed: PrintedCharacteristics {
            name: "Smaug, the Great Calamity".into(),
            mana_cost: Some(ManaCost::parse("{5}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dragon".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: "Flying".into(),
            inset: Some(InsetSpell::Adventure(
                PrintedCharacteristics {
                    name: "Spew Flame".into(),
                    mana_cost: Some(ManaCost::parse("{4}{R}")),
                    types: CardTypes::new([CardType::Sorcery]),
                    subtypes: vec!["Adventure".into()],
                    spell_effects: vec![Effect::deal_damage(
                        5,
                        ObjectSelector::Target(TargetSpec::Creature),
                    )],
                    text_box: "Spew Flame deals 5 damage to target creature.".into(),
                    ..Default::default()
                }
                .into(),
            )),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bilbo_thief_in_the_night() -> CardDefinition {
    use managym_state::ability::{Per, TriggerSubject};
    CardDefinition {
        key: "hob.bilbo_thief_in_the_night",
        scryfall_id: "484c7f83-8339-4ae1-8350-68ce1f7d05a3",
        printed: PrintedCharacteristics {
            name: "Bilbo, Thief in the Night".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Halfling".into(), "Rogue".into()],
            power: Some(2), toughness: Some(2),
            statics: vec![StaticAbility::SpellsFromOutsideHandCostLess(1)],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: None,
                effects: vec![Effect::CastFromGraveyard {
                    predicate: CardPredicate {card_types_any: vec![CardType::Artifact, CardType::Instant, CardType::Sorcery], ..Default::default()},
                    exile_instants_and_sorceries: true
                }]
            }],
            text_box: "Spells you cast from anywhere other than your hand cost {1} less to cast.\nWhenever Bilbo attacks, you may cast an artifact, instant, or sorcery spell from your graveyard. If an instant or sorcery spell cast this way would be put into your graveyard, exile it instead.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bofur_reliable_guardian() -> CardDefinition {
    let rules_text = "Target artifact or creature you control gains hexproof and indestructible until end of turn. (Then exile this card. You may cast the creature later from exile.)";
    let inset = PrintedCharacteristics {
        name: "Concerted Care".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Adventure".into()],
        spell_effects: vec![Effect::until_end_of_turn(
            ObjectSelector::Target(TargetSpec::PermanentYouControl {
                predicate: CardPredicate {
                    card_types_any: vec![CardType::Artifact, CardType::Creature],
                    ..Default::default()
                },
            }),
            Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible, Keyword::Hexproof])),
        )],
        text_box: rules_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "hob.bofur_reliable_guardian",
        scryfall_id: "6b8e6435-7de4-41d5-bc7d-8e24c11897d0",
        printed: PrintedCharacteristics {
            name: "Bofur, Reliable Guardian".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dwarf".into(), "Scout".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            inset: Some(InsetSpell::Adventure(inset.into())),
            text_box: "Lifelink".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn velvetwing_butterflies() -> CardDefinition {
    let rules_text = "Tap one or two target creatures. (Then exile this card. You may cast the creature later from exile.)";
    let inset = PrintedCharacteristics {
        name: "Gaze in Wonder".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Adventure".into()],
        spell_effects: vec![Effect::TargetClause {
            requirement: TargetRequirement {
                min: 1,
                ..TargetRequirement::up_to(2, TargetSpec::Creature)
            },
            effects: vec![Effect::tap(ObjectSelector::Targets(TargetSpec::Creature))],
        }],
        text_box: rules_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "hob.velvetwing_butterflies",
        scryfall_id: "5cc0f994-5048-4898-926e-b56cbc97e0ca",
        printed: PrintedCharacteristics {
            name: "Velvetwing Butterflies".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Insect".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying]),
            inset: Some(InsetSpell::Adventure(inset.into())),
            text_box: "Flying".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bilbo_baggins_burglar() -> CardDefinition {
    let rules_text = "Scry 2. (Then exile this card. You may cast the creature later from exile.)";
    let inset = PrintedCharacteristics {
        name: "Take a Glance".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        spell_effects: vec![Effect::Scry { count: 2 }],
        text_box: rules_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "hob.bilbo_baggins_burglar",
        scryfall_id: "6a109b3e-9f5b-4625-abb7-6b992c10530b",
        printed: PrintedCharacteristics {
            name: "Bilbo Baggins, Burglar".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Halfling".into(), "Rogue".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            inset: Some(InsetSpell::Adventure(inset.into())),
            text_box: "When Bilbo Baggins enters, draw a card.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn laketown_mariners() -> CardDefinition {
    let front_text="Vigilance\nWard {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)";
    let spell_text="Exile two target creatures and/or lands you control, then return them to the battlefield under their owner's control.";
    let controlled = TargetSpec::PermanentYouControl {
        predicate: CardPredicate {
            card_types_any: vec![CardType::Creature, CardType::Land],
            ..Default::default()
        },
    };
    let inset = PrintedCharacteristics {
        name: "Gone Fishing".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Adventure".into()],
        text_box: spell_text.into(),
        spell_effects: vec![Effect::TargetClause {
            requirement: TargetRequirement {
                min: 2,
                ..TargetRequirement::up_to(2, controlled.clone())
            },
            effects: vec![
                Effect::exile(ObjectSelector::Targets(controlled)),
                Effect::Move {
                    unrevealed: false,
                    what: ObjectSelector::Moved,
                    to: ZoneType::Battlefield,
                    library_end: None,
                    tapped: false,
                    transformed: false,
                    under: Some(PlayerSelector::Owner),
                    with_counter: None,
                    linked: None,
                },
            ],
        }],
        ..Default::default()
    };
    CardDefinition {
        key: "hob.laketown_mariners",
        scryfall_id: "4202a678-a5f4-47f9-9c18-e88ab9ad20a4",
        printed: PrintedCharacteristics {
            name: "Lake-town Mariners".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Citizen".into()],
            power: Some(6),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            text_box: front_text.into(),
            statics: vec![StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse(
                "{2}",
            )))],
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gollum_silent_slinker() -> CardDefinition {
    let front_text = "Menace (This creature can't be blocked except by two or more creatures.)";
    let spell_text="Put a +1/+1 counter on up to one target creature. Target player gains 2 life. (Then exile this card. You may cast the creature later from exile.)";

    let inset = PrintedCharacteristics {
        name: "Meager Meal".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        text_box: spell_text.into(),
        spell_effects: vec![
            Effect::TargetClause {
                requirement: TargetRequirement::up_to(1, TargetSpec::Creature),
                effects: vec![Effect::put_counters(
                    ObjectSelector::Target(TargetSpec::Creature),
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            },
            Effect::TargetClause {
                requirement: TargetRequirement::one(TargetSpec::Player),
                effects: vec![Effect::ForPlayer {
                    player: PlayerSelector::Target(Box::new(TargetSpec::Player)),
                    effects: vec![Effect::gain_life(2)],
                }],
            },
        ],
        ..Default::default()
    };
    CardDefinition {
        key: "hob.gollum_silent_slinker",
        scryfall_id: "6cfaa182-3fec-4907-8814-b4d29c33cec3",
        printed: PrintedCharacteristics {
            name: "Gollum, Silent Slinker".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Halfling".into(), "Horror".into()],
            power: Some(4),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Menace]),
            text_box: front_text.into(),
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gloin_the_mighty() -> CardDefinition {
    let front_text = "At the beginning of your first main phase, add {R}{R}.";
    let spell_text="Easy Pickings deals 1 damage to each creature your opponents control. (Then exile this card. You may cast the creature later from exile.)";

    let inset = PrintedCharacteristics {
        name: "Easy Pickings".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        text_box: spell_text.into(),
        spell_effects: vec![Effect::deal_damage(
            1,
            ObjectSelector::each(Scope::OpponentsControl, CardPredicate::creature()),
        )],
        ..Default::default()
    };
    CardDefinition {
        key: "hob.gloin_the_mighty",
        scryfall_id: "5793b8eb-2fc5-454d-8fa2-20346fef167a",
        printed: PrintedCharacteristics {
            name: "Glóin the Mighty".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dwarf".into(), "Warrior".into()],
            power: Some(4),
            toughness: Some(3),
            text_box: front_text.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::FirstMain, Actor::You),
                intervening_if: None,
                effects: vec![Effect::add_mana(Mana::parse("RR"))],
            }],
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn the_arkenstone() -> CardDefinition {
    let front_text =
        "Creatures you control get +1/+1.\nAt the beginning of your end step, draw a card.";
    let spell_text="Search your library for a legendary creature card, reveal it, put it into your hand, then shuffle. (Then exile this card. You may cast the artifact later from exile.)";

    let inset = PrintedCharacteristics {
        name: "Seek the Heart".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        text_box: spell_text.into(),
        spell_effects: vec![Effect::search_library(
            CardPredicate {
                supertype: Some("Legendary".into()),
                ..CardPredicate::creature()
            },
            EffectValue::Fixed(1),
            vec![Effect::return_to_hand(ObjectSelector::Selected)],
        )],
        ..Default::default()
    };
    CardDefinition {
        key: "hob.the_arkenstone",
        scryfall_id: "a56a88ba-fcfa-4b56-bdae-a080b297b871",
        printed: PrintedCharacteristics {
            name: "The Arkenstone".into(),
            mana_cost: Some(ManaCost::parse("{5}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec!["Legendary".into()],
            text_box: front_text.into(),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::YouControl(CardPredicate::creature()),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(1),
                    toughness: LayerValue::Fixed(1),
                }],
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: None,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn beorn_reluctant_host() -> CardDefinition {
    let till_and_tend = PrintedCharacteristics {
        name: "Till and Tend".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        text_box: "You may play an additional land this turn. (Then exile this card. You may cast the creature later from exile.)".into(),
        spell_effects: vec![Effect::until_end_of_turn(ObjectSelector::You, Modification::AdditionalLandPlay)],
        ..Default::default()
    };
    CardDefinition {
        key: "hob.beorn_reluctant_host",
        scryfall_id: "804589b7-3ef9-473d-97cc-c61a2d41f70d",
        printed: PrintedCharacteristics {
            name: "Beorn, Reluctant Host".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            supertypes: vec!["Legendary".into()],
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Bear".into(), "Shapeshifter".into()],
            keywords: Keywords::of(&[Keyword::Trample]),
            power: Some(5),
            toughness: Some(5),
            text_box: "Trample".into(),
            inset: Some(InsetSpell::Adventure(till_and_tend.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Keep the mill result bound while the caster chooses from those cards.
fn mill_then_recover(count: i32, predicate: CardPredicate, min: usize, max: i32) -> Vec<Effect> {
    vec![
        Effect::MillCards {
            count: EffectValue::Fixed(count),
            target: None,
        },
        Effect::SelectCards {
            from: ObjectSelector::CardsThisWay,
            min,
            max: EffectValue::Fixed(max),
            predicate,
            effects: vec![Effect::return_to_hand(ObjectSelector::Selected)],
            if_declined: vec![],
        },
    ]
}

fn most_decrepit_old_bird() -> CardDefinition {
    let text = "Flying\nThreshold — This creature gets +1/+1 as long as there are seven or more cards in your graveyard.";
    let adventure_text =
        "Mill four cards, then put an instant or sorcery card from among them into your hand.";
    let inset = PrintedCharacteristics {
        name: "Speak Secrets".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        spell_effects: mill_then_recover(
            4,
            CardPredicate {
                card_types_any: vec![CardType::Instant, CardType::Sorcery],
                ..Default::default()
            },
            1,
            1,
        ),
        text_box: adventure_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "hob.most_decrepit_old_bird",
        scryfall_id: "2d838feb-89f2-4cdb-a5ab-ec880f28d873",
        printed: PrintedCharacteristics {
            name: "Most Decrepit Old Bird".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into()],
            power: Some(1),
            toughness: Some(1),
            text_box: text.into(),
            keywords: Keywords::of(&[Keyword::Flying]),
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
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn thranduil_sindarin_liege() -> CardDefinition {
    let text = "Other Elves you control get +1/+1.\nLandfall — Whenever a land you control enters, create a 1/1 green Elf creature token.";
    let adventure_text = "Mill four cards, then put up to two land cards from among them into your hand. (Then exile this card. You may cast the creature later from exile.)";
    let inset = PrintedCharacteristics {
        name: "Silvan Rally".into(),
        mana_cost: Some(ManaCost::parse("{1}{G/U}{G/U}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        spell_effects: mill_then_recover(
            4,
            CardPredicate {
                card_type: Some(CardType::Land),
                ..Default::default()
            },
            0,
            2,
        ),
        text_box: adventure_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "hob.thranduil_sindarin_liege",
        scryfall_id: "481870ee-d1f7-421b-86e1-570ea933bbbc",
        printed: PrintedCharacteristics {
            name: "Thranduil, Sindarin Liege".into(),
            mana_cost: Some(ManaCost::parse("{2}{G/U}{G/U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Noble".into()],
            supertypes: vec!["Legendary".into()],
            power: Some(2),
            toughness: Some(3),
            text_box: text.into(),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::OtherYouControl(CardPredicate::subtype("Elf")),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(1),
                    toughness: LayerValue::Fixed(1),
                }],
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::you_control(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                intervening_if: None,
                effects: vec![Effect::create_tokens("token.elf", 1)],
            }],
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn glamdring_foe_hammer() -> CardDefinition {
    let text = "Instant and sorcery spells you cast cost {X} less to cast, where X is equipped creature's power.\nEquip {2}";
    let adventure_text = "Mill six cards, then put all instant and sorcery cards from among them into your hand. (Then exile this card. You may cast the artifact later from exile.)";
    let spells = || CardPredicate {
        card_types_any: vec![CardType::Instant, CardType::Sorcery],
        ..Default::default()
    };
    let inset = PrintedCharacteristics {
        name: "Gleam of Death".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        spell_effects: vec![
            Effect::MillCards {
                count: EffectValue::Fixed(6),
                target: None,
            },
            Effect::return_to_hand(ObjectSelector::Matching {
                of: Box::new(ObjectSelector::CardsThisWay),
                predicate: spells(),
            }),
        ],
        text_box: adventure_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "hob.glamdring_foe_hammer",
        scryfall_id: "a5cfbfde-783e-46ca-b3cf-11f16209d6cb",
        printed: PrintedCharacteristics {
            name: "Glamdring, Foe-hammer".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            supertypes: vec!["Legendary".into()],
            text_box: text.into(),
            statics: vec![StaticAbility::SpellsCostLess(
                managym_state::card::SpellCostReduction {
                    first_on_own_turn: false,
                    predicate: spells(),
                    condition: Condition::Always,
                    generic: EffectValue::AttachedCreaturePower,
                },
            )],
            activated_abilities: vec![managym_state::card::ActivatedAbilityDefinition {
                sorcery_timing: true,
                mana_cost: ManaCost::parse("{2}"),
                effects: vec![Effect::AttachSourceTo {
                    what: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                }],
                ..Default::default()
            }],
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn my_precious() -> CardDefinition {
    let text = "Equipped creature has hexproof and can't be blocked.\nEquip—{2}, Pay 2 life.";
    let adventure_text =
        "As an additional cost to cast this spell, sacrifice a creature.\nDraw two cards.";
    let inset = PrintedCharacteristics {
        name: "Allure of Power".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Adventure".into()],
        spell_effects: vec![Effect::DrawCards { count: 2.into() }],
        text_box: adventure_text.into(),
        additional_costs: vec![managym_state::card::AdditionalCost::Sacrifice(Box::new(
            managym_state::card::SacrificeCost {
                predicate: CardPredicate::creature(),
                another: false,
                or_mana: None,
            },
        ))],
        ..Default::default()
    };
    CardDefinition {
        key: "hob.my_precious",
        scryfall_id: "15ae4d50-be2f-412c-bb6b-b0a06b60474a",
        printed: PrintedCharacteristics {
            name: "My Precious".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            supertypes: vec!["Legendary".into()],
            text_box: text.into(),
            statics: vec![
                StaticAbility::KeywordGrant(managym_state::card::StaticKeywordGrant {
                    scope: StaticScope::Attached,
                    condition: None,
                    keywords: Keywords::of(&[Keyword::Hexproof]),
                }),
                StaticAbility::Restriction(managym_state::card::StaticRestriction {
                    scope: StaticScope::Attached,
                    kind: managym_state::card::Restriction::CantBeBlocked,
                    unless: None,
                }),
            ],
            activated_abilities: vec![managym_state::card::ActivatedAbilityDefinition {
                sorcery_timing: true,
                mana_cost: ManaCost::parse("{2}"),
                life_cost: 2,
                effects: vec![Effect::AttachSourceTo {
                    what: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                }],
                ..Default::default()
            }],
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gandalf_goblins_bane() -> CardDefinition {
    let text = "Whenever you cast a noncreature spell, Gandalf gets +1/+1 until end of turn and deals 1 damage to each opponent.";
    let adventure_text = "Look at the top two cards of your library and exile them face down. For as long as they remain exiled, you may play them if you control a Wizard. (Then exile this card. You may cast the creature later from exile.)";
    let inset = PrintedCharacteristics {
        name: "Flameshape".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Adventure".into()],
        spell_effects: vec![Effect::ExileTopCardsToPlay {
            count: 2,
            until: PlayWindow::WhileExiled,
            face_down: true,
            condition: Some(Condition::ControlsAtLeast {
                distinct_names: false,
                count: 1,
                predicate: CardPredicate::subtype("Wizard"),
            }),
        }],
        text_box: adventure_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "hob.gandalf_goblins_bane",
        scryfall_id: "9b0d29a1-7da9-4fb3-8536-8ff8d8acae0b",
        printed: PrintedCharacteristics {
            name: "Gandalf, Goblins' Bane".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Avatar".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(3),
            text_box: text.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![
                    Effect::until_end_of_turn(ObjectSelector::This, Modification::pump(1, 1)),
                    Effect::DealDamageToPlayer {
                        player: PlayerSelector::EachOpponent,
                        amount: EffectValue::Fixed(1),
                    },
                ],
            }],
            inset: Some(InsetSpell::Adventure(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}
