//! Non-TLA definitions required by current authored setups.
//! Edit these typed definitions; `semantic::compile` lowers them in process.
use super::authoring::{
    Card, CardType, Characteristics, Color, Cost, Duration, Event, Keyword, Op, Predicate,
    PrintedPower, Program, ProgramKind, Quantity, Sacrifice, Selector, Subject, Target, Trigger,
};
pub(super) fn cards() -> Vec<Card> {
    vec![
        enthusiastic_study(),
        eyetwitch(),
        field_trip(),
        fists_of_flame(),
        giant_cindermaw(),
        gnarled_professor(),
        monastery_swiftspear(),
        power_word_kill(),
        professor_of_symbology(),
        savor(),
        spectral_sailor(),
        glory_seeker(),
        hill_giant(),
        lightning_bolt(),
        savannah_lions(),
        bastion_of_remembrance(),
        deadly_dispute(),
        fabled_passage(),
        human_soldier(),
        lotus_cobra(),
        mayhem_devil(),
        plains(),
        prismatic_vista(),
        sazh_s_chocobo(),
        swamp(),
        tireless_provisioner(),
        tireless_tracker(),
        treasure(),
        weaponize_the_monsters(),
        divide_by_zero(),
        igneous_inspiration(),
        pop_quiz(),
        forest(),
        island(),
        mountain(),
        prophetic_prism(),
    ]
}
fn enthusiastic_study() -> Card {
    let text_box = "Target creature gets +3/+1 and gains trample until end of turn.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{R}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "elemental.enthusiastic_study",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 3,
                    target: "target_0",
                    toughness: 1,
                },
                Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![Keyword::Trample],
                    target: "target_0",
                },
                Op::Learn,
            ],
        )
        .cost(Cost::mana("{2}{R}"))
        .targets(vec![Target::one("target_0", Selector::Creature)])
        .inferred()],
        registry_name: "Enthusiastic Study",
        text_box,
    }
}

fn eyetwitch() -> Card {
    let text_box = "Flying\nWhen this creature dies, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{B}"),
            subtypes: Some(vec!["Eye", "Bat"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "elemental.eyetwitch",
        programs: vec![
            Program::new("trigger_0", ProgramKind::Triggered, vec![Op::Learn])
                .trigger(Trigger::new(Event::Dies, Subject::This)),
        ],
        registry_name: "Eyetwitch",
        text_box,
    }
}

fn field_trip() -> Card {
    let text_box = "Search your library for a basic Forest card, put that card onto the battlefield tapped, then shuffle.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{G}"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "elemental.field_trip",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::SearchBasicLand {
                    subtype: Some("Forest"),
                },
                Op::Learn,
            ],
        )
        .cost(Cost::mana("{2}{G}"))
        .inferred()],
        registry_name: "Field Trip",
        text_box,
    }
}

fn fists_of_flame() -> Card {
    let text_box = "Draw a card. Until end of turn, target creature gains trample and gets +1/+0 for each card you've drawn this turn. (A creature with trample can deal excess combat damage to the player or planeswalker it's attacking.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "elemental.fists_of_flame",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::DrawCards { count: 1 },
                Op::BuffTargetValue {
                    power: Quantity::CardsDrawnMinus { subtract: 0 },
                    target: "target_0",
                    toughness: 0,
                },
                Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![Keyword::Trample],
                    target: "target_0",
                },
            ],
        )
        .cost(Cost::mana("{1}{R}"))
        .targets(vec![Target::one("target_0", Selector::Creature)])
        .inferred()],
        registry_name: "Fists of Flame",
        text_box,
    }
}

fn giant_cindermaw() -> Card {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nPlayers can't gain life.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{R}"),
            subtypes: Some(vec!["Dinosaur", "Beast"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Trample]),
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "elemental.giant_cindermaw",
        programs: vec![Program::new(
            "prevent_life_gain",
            ProgramKind::Static,
            vec![Op::PreventLifeGain],
        )],
        registry_name: "Giant Cindermaw",
        text_box,
    }
}

fn gnarled_professor() -> Card {
    let text_box = "Trample\nWhen this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{G}{G}"),
            subtypes: Some(vec!["Treefolk", "Druid"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Trample]),
            power: Some(PrintedPower::Fixed(5)),
            toughness: Some(4),
            ..Default::default()
        },
        key: "elemental.gnarled_professor",
        programs: vec![
            Program::new("trigger_0", ProgramKind::Triggered, vec![Op::Learn])
                .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
        ],
        registry_name: "Gnarled Professor",
        text_box,
    }
}

fn monastery_swiftspear() -> Card {
    let text_box = "Haste\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{R}"),
            subtypes: Some(vec!["Human", "Monk"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Haste]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "elemental.monastery_swiftspear",
        programs: vec![Program::prowess("trigger_0")],
        registry_name: "Monastery Swiftspear",
        text_box,
    }
}

fn power_word_kill() -> Card {
    let text_box = "Destroy target non-Angel, non-Demon, non-Devil, non-Dragon creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "elemental.power_word_kill",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Destroy { target: "target_0" }],
        )
        .cost(Cost::mana("{1}{B}"))
        .targets(vec![Target::one(
            "target_0",
            Selector::Permanent {
                predicate: Predicate::All {
                    predicates: vec![
                        Predicate::NotSubtypes {
                            values: vec!["Angel", "Demon", "Devil", "Dragon"],
                        },
                        Predicate::CardType {
                            value: CardType::Creature,
                        },
                    ],
                },
            },
        )])
        .inferred()],
        registry_name: "Power Word Kill",
        text_box,
    }
}

fn professor_of_symbology() -> Card {
    let text_box = "When this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{W}"),
            subtypes: Some(vec!["Kor", "Cleric"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "elemental.professor_of_symbology",
        programs: vec![
            Program::new("trigger_0", ProgramKind::Triggered, vec![Op::Learn])
                .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
        ],
        registry_name: "Professor of Symbology",
        text_box,
    }
}

fn savor() -> Card {
    let text_box = "Target creature gets -2/-2 until end of turn. Create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "elemental.savor",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: -2,
                    target: "target_0",
                    toughness: -2,
                },
                Op::CreateToken {
                    count: 1,
                    definition_ref: "sacrifice.food",
                    tapped_and_attacking: Some(false),
                },
            ],
        )
        .cost(Cost::mana("{1}{B}"))
        .targets(vec![Target::one("target_0", Selector::Creature)])
        .inferred()],
        registry_name: "Savor",
        text_box,
    }
}

fn spectral_sailor() -> Card {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nFlying\n{3}{U}: Draw a card.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{U}"),
            subtypes: Some(vec!["Spirit", "Pirate"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying, Keyword::Flash]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "elemental.spectral_sailor",
        programs: vec![Program::new(
            "activate_0",
            ProgramKind::Activated,
            vec![Op::DrawCards { count: 1 }],
        )
        .cost(Cost {
            mana: Some("{3}{U}"),
            sacrifice_source: Some(false),
            tap_source: Some(false),
            waterbend: Some(false),
            ..Default::default()
        })],
        registry_name: "Spectral Sailor",
        text_box,
    }
}

fn glory_seeker() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1W"),
            subtypes: Some(vec!["Human", "Soldier"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "research.glory_seeker",
        programs: vec![],
        registry_name: "Glory Seeker",
        text_box,
    }
}

fn hill_giant() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("3R"),
            subtypes: Some(vec!["Giant"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "research.hill_giant",
        programs: vec![],
        registry_name: "Hill Giant",
        text_box,
    }
}

fn lightning_bolt() -> Card {
    let text_box = "Lightning Bolt deals 3 damage to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("R"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "research.lightning_bolt",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::DealDamage {
                amount: 3,
                target: "victim",
            }],
        )
        .cost(Cost::mana("R"))
        .targets(vec![Target::one("victim", Selector::CreatureOrPlayer)])],
        registry_name: "Lightning Bolt",
        text_box,
    }
}

fn savannah_lions() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("W"),
            subtypes: Some(vec!["Cat"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "research.savannah_lions",
        programs: vec![],
        registry_name: "Savannah Lions",
        text_box,
    }
}

fn bastion_of_remembrance() -> Card {
    let text_box = "When this enchantment enters, create a 1/1 white Human Soldier creature token.\nWhenever a creature you control dies, each opponent loses 1 life and you gain 1 life.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2B"),
            subtypes: Some(vec![]),
            types: vec![CardType::Enchantment],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.bastion_of_remembrance",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "sacrifice.human_soldier",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "p1",
                ProgramKind::Triggered,
                vec![Op::DrainLife { amount: 1 }],
            )
            .trigger(Trigger::new(
                Event::Dies,
                Subject::AnyYouControl {
                    predicate: Predicate::CardType {
                        value: CardType::Creature,
                    },
                },
            )),
        ],
        registry_name: "Bastion of Remembrance",
        text_box,
    }
}

fn deadly_dispute() -> Card {
    let text_box = "As an additional cost to cast this spell, sacrifice an artifact or creature.\nDraw two cards and create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1B"),
            subtypes: Some(vec![]),
            types: vec![CardType::Instant],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.deadly_dispute",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Spell,
            vec![
                Op::DrawCards { count: 2 },
                Op::CreateToken {
                    count: 1,
                    definition_ref: "sacrifice.treasure",
                    tapped_and_attacking: Some(false),
                },
            ],
        )
        .cost(Cost {
            mana: Some("1B"),
            sacrifice: Some(Sacrifice {
                another: false,
                or_mana: None,
                predicate: Predicate::TypesAny {
                    values: vec![CardType::Artifact, CardType::Creature],
                },
            }),
            ..Default::default()
        })],
        registry_name: "Deadly Dispute",
        text_box,
    }
}

fn fabled_passage() -> Card {
    let text_box = "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Then if you control four or more lands, untap that land.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec![]),
            types: vec![CardType::Land],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.fabled_passage",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::SearchBasicLands {
                bottom_source: false,
                count: 1,
                tapped: true,
                untap_if_lands: Some(4),
            }],
        )
        .cost(Cost {
            mana: Some("0"),
            sacrifice_source: Some(true),
            tap_source: Some(true),
            ..Default::default()
        })],
        registry_name: "Fabled Passage",
        text_box,
    }
}

fn human_soldier() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Human", "Soldier"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            token: Some(true),
            supertypes: Some(vec![]),
            colors: Some(vec![Color::W]),
            ..Default::default()
        },
        key: "sacrifice.human_soldier",
        programs: vec![],
        registry_name: "Human Soldier",
        text_box,
    }
}

fn lotus_cobra() -> Card {
    let text_box = "Landfall — Whenever a land you control enters, add one mana of any color.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1G"),
            subtypes: Some(vec!["Snake"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.lotus_cobra",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
        )
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnyYouControl {
                predicate: Predicate::CardType {
                    value: CardType::Land,
                },
            },
        ))],
        registry_name: "Lotus Cobra",
        text_box,
    }
}

fn mayhem_devil() -> Card {
    let text_box =
        "Whenever a player sacrifices a permanent, this creature deals 1 damage to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1BR"),
            subtypes: Some(vec!["Devil"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.mayhem_devil",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![Op::DealDamage {
                amount: 1,
                target: "target_0",
            }],
        )
        .targets(vec![Target::one("target_0", Selector::CreatureOrPlayer)])
        .trigger(Trigger {
            predicate: Some(Predicate::Any),
            controller_only: Some(false),
            ..Trigger::new(Event::Sacrificed, Subject::This)
        })
        .inferred()],
        registry_name: "Mayhem Devil",
        text_box,
    }
}

fn plains() -> Card {
    let text_box = "({T}: Add {W}.)";

    basic_land("sacrifice.plains", "Plains", text_box, "W")
}

fn prismatic_vista() -> Card {
    let text_box = "{T}, Pay 1 life, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield, then shuffle.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec![]),
            types: vec![CardType::Land],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.prismatic_vista",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::SearchBasicLands {
                bottom_source: false,
                count: 1,
                tapped: false,
                untap_if_lands: None,
            }],
        )
        .cost(Cost {
            mana: Some("0"),
            sacrifice_source: Some(true),
            tap_source: Some(true),
            life: Some(1),
            ..Default::default()
        })],
        registry_name: "Prismatic Vista",
        text_box,
    }
}

fn sazh_s_chocobo() -> Card {
    let text_box =
        "Landfall — Whenever a land you control enters, put a +1/+1 counter on this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("G"),
            subtypes: Some(vec!["Bird"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(0)),
            toughness: Some(1),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.sazh_s_chocobo",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![Op::PutCounters {
                count: 1,
                target: "source",
            }],
        )
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnyYouControl {
                predicate: Predicate::CardType {
                    value: CardType::Land,
                },
            },
        ))],
        registry_name: "Sazh's Chocobo",
        text_box,
    }
}

fn swamp() -> Card {
    let text_box = "({T}: Add {B}.)";

    basic_land("sacrifice.swamp", "Swamp", text_box, "B")
}

fn tireless_provisioner() -> Card {
    let text_box = "Landfall — Whenever a land you control enters, create a Food token or a Treasure token. (Food is an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\" Treasure is an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2G"),
            subtypes: Some(vec!["Elf", "Scout"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(2),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.tireless_provisioner",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![Op::Modal {
                modes: vec![
                    vec![Op::CreateToken {
                        count: 1,
                        definition_ref: "sacrifice.food",
                        tapped_and_attacking: Some(false),
                    }],
                    vec![Op::CreateToken {
                        count: 1,
                        definition_ref: "sacrifice.treasure",
                        tapped_and_attacking: Some(false),
                    }],
                ],
            }],
        )
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnyYouControl {
                predicate: Predicate::CardType {
                    value: CardType::Land,
                },
            },
        ))],
        registry_name: "Tireless Provisioner",
        text_box,
    }
}

fn tireless_tracker() -> Card {
    let text_box = "Landfall — Whenever a land you control enters, investigate. (Create a Clue token. It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you sacrifice a Clue, put a +1/+1 counter on this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2G"),
            subtypes: Some(vec!["Human", "Scout"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(2),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.tireless_tracker",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "sacrifice.clue",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(
                Event::EntersBattlefield,
                Subject::AnyYouControl {
                    predicate: Predicate::CardType {
                        value: CardType::Land,
                    },
                },
            )),
            Program::new(
                "p1",
                ProgramKind::Triggered,
                vec![Op::PutCounters {
                    count: 1,
                    target: "source",
                }],
            )
            .trigger(Trigger {
                predicate: Some(Predicate::Subtype { value: "Clue" }),
                controller_only: Some(true),
                ..Trigger::new(Event::Sacrificed, Subject::This)
            }),
        ],
        registry_name: "Tireless Tracker",
        text_box,
    }
}

fn treasure() -> Card {
    let text_box = "{T}, Sacrifice this token: Add one mana of any color.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Treasure"]),
            types: vec![CardType::Artifact],
            keywords: Some(vec![]),
            token: Some(true),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.treasure",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
        )
        .cost(Cost {
            mana: Some("0"),
            sacrifice_source: Some(true),
            tap_source: Some(true),
            ..Default::default()
        })],
        registry_name: "Treasure",
        text_box,
    }
}

fn weaponize_the_monsters() -> Card {
    let text_box = "{2}, Sacrifice a creature: This enchantment deals 2 damage to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("R"),
            subtypes: Some(vec![]),
            types: vec![CardType::Enchantment],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "sacrifice.weaponize_the_monsters",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::DealDamage {
                amount: 2,
                target: "target_0",
            }],
        )
        .cost(Cost {
            mana: Some("2"),
            sacrifice: Some(Sacrifice {
                another: false,
                or_mana: None,
                predicate: Predicate::CardType {
                    value: CardType::Creature,
                },
            }),
            ..Default::default()
        })
        .targets(vec![Target::one("target_0", Selector::CreatureOrPlayer)])
        .inferred()],
        registry_name: "Weaponize the Monsters",
        text_box,
    }
}

fn divide_by_zero() -> Card {
    let text_box = "Return target spell or permanent with mana value 1 or greater to its owner's hand.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2U"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "stx.divide_by_zero",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::ReturnToHand { target: "target" }, Op::Learn],
        )
        .cost(Cost::mana("2U"))
        .targets(vec![Target::one(
            "target",
            Selector::SpellOrPermanent { min_mana_value: 1 },
        )])],
        registry_name: "Divide by Zero",
        text_box,
    }
}

fn igneous_inspiration() -> Card {
    let text_box = "Igneous Inspiration deals 3 damage to any target.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2R"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "stx.igneous_inspiration",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::DealDamage {
                    amount: 3,
                    target: "victim",
                },
                Op::Learn,
            ],
        )
        .cost(Cost::mana("2R"))
        .targets(vec![Target::one("victim", Selector::CreatureOrPlayer)])],
        registry_name: "Igneous Inspiration",
        text_box,
    }
}

fn pop_quiz() -> Card {
    let text_box = "Draw a card.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2U"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "stx.pop_quiz",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::DrawCards { count: 1 }, Op::Learn],
        )
        .cost(Cost::mana("2U"))],
        registry_name: "Pop Quiz",
        text_box,
    }
}

fn forest() -> Card {
    let text_box = "({T}: Add {G}.)";

    basic_land("temur.forest", "Forest", text_box, "G")
}

fn island() -> Card {
    let text_box = "({T}: Add {U}.)";

    basic_land("temur.island", "Island", text_box, "U")
}

fn mountain() -> Card {
    let text_box = "({T}: Add {R}.)";

    basic_land("temur.mountain", "Mountain", text_box, "R")
}

fn prophetic_prism() -> Card {
    let text_box = "When this artifact enters, draw a card.\n{1}, {T}: Add one mana of any color.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}"),
            subtypes: Some(vec![]),
            types: vec![CardType::Artifact],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "temur.prophetic_prism",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::DrawCards { count: 1 }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "p1",
                ProgramKind::Activated,
                vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
            )
            .cost(Cost {
                mana: Some("1"),
                tap_source: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Prophetic Prism",
        text_box,
    }
}

fn basic_land(
    key: &'static str,
    registry_name: &'static str,
    text_box: &'static str,
    mana: &'static str,
) -> Card {
    Card {
        characteristics: Characteristics {
            subtypes: Some(vec![registry_name]),
            types: vec![CardType::Land],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec!["basic"]),
            ..Default::default()
        },
        key,
        programs: vec![Program::new(
            "p0",
            ProgramKind::Mana,
            vec![Op::AddMana { mana, until: None }],
        )],
        registry_name,
        text_box,
    }
}
