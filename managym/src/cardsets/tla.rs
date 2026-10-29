//! Current TLA cards and their fixed tokens.
//! Edit these typed definitions; `semantic::compile` lowers them in process.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Color, Condition, Controller, Cost, Count, Destination,
    Direction, Duration, Event, Incomplete, Keyword, Op, Payment, Predicate, PrintedPower, Program,
    ProgramKind, Quantity, Remainder, RemainderOrder, Sacrifice, Selector, SpendOnly, Subject,
    Target, Trigger, Until, Zone,
};
/// Defined cards whose behavior is incomplete. The fixed setups that name them
/// still run them; the world does not support them for deck entry until their
/// batch completes the missing behavior.
pub(super) const INCOMPLETE: &[Incomplete] = &[
    ("tla.accumulate_wisdom", "Controller must order the library remainder (CR 401.4); it is randomized"),
    ("tla.cat_gator", "Any target must include planeswalkers and battles (CR 115.4)"),
    ("tla.dragonfly_swarm", "Unreviewed until T4: Flying; spell-and-ability ward; graveyard CDA power; conditional death draw."),
    ("tla.great_divide_guide", "Unreviewed until T3: Continuous any-color mana grant to lands and Allies; source removal and creature sickness."),
    ("tla.lightning_strike", "Any target must include planeswalkers and battles (CR 115.4)"),
    ("tla.ran_and_shaw", "Unreviewed until T5: Flying/firebending; cast-only conditional ETB nonlegendary self copy; Dragon anthem activation."),
    ("tla.suki_kyoshi_warrior", "CDA power must apply outside the battlefield (CR 604.3)"),
    ("tla.water_tribe_rallier", "Library power filter must evaluate CDAs, such as Suki's (CR 604.3)"),
];

pub(super) fn cards() -> Vec<Card> {
    vec![
        abandon_attachments(),
        azula_always_lies(),
        boomerang_basics(),
        cat_gator(),
        combustion_technique(),
        corrupt_court_official(),
        dai_li_agents(),
        deserter_s_disciple(),
        enter_the_avatar_state(),
        epic_downfall(),
        foggy_swamp_spirit_keeper(),
        gather_the_white_lotus(),
        gran_gran(),
        haru_hidden_talent(),
        iroh_s_demonstration(),
        jet_freedom_fighter(),
        knowledge_seeker(),
        lightning_strike(),
        master_piandao(),
        messenger_hawk(),
        momo_friendly_flier(),
        octopus_form(),
        origin_of_metalbending(),
        ostrich_horse(),
        ozai_s_cruelty(),
        pretending_poxbearers(),
        shared_roots(),
        sokka_s_haiku(),
        tiger_dillo(),
        toph_the_blind_bandit(),
        true_ancestry(),
        uncle_iroh(),
        wartime_protestors(),
        zuko_s_exile(),
        ally(),
        callous_inspector(),
        clue(),
        curious_farm_animals(),
        cycle_of_renewal(),
        deadly_precision(),
        food(),
        june_bounty_hunter(),
        tolls_of_war(),
        unlucky_cabbage_merchant(),
        great_divide_guide(),
        hermitic_herbalist(),
        iroh_grand_lotus(),
        ran_and_shaw(),
        waterbending_lesson(),
        accumulate_wisdom(),
        airship_engine_room(),
        allies_at_last(),
        avatar_enthusiasts(),
        badgermole_cub(),
        boar_q_pine(),
        boiling_rock_prison(),
        cat_owl(),
        compassionate_healer(),
        cunning_maneuver(),
        dragonfly_swarm(),
        earth_kingdom_jailer(),
        earth_kingdom_protectors(),
        earth_kingdom_soldier(),
        earth_kings_lieutenant(),
        fancy_footwork(),
        fire_nation_cadets(),
        fire_sages(),
        firebending_lesson(),
        first_time_flyer(),
        foggy_bottom_swamp(),
        forecasting_fortune_teller(),
        glider_kids(),
        iguana_parrot(),
        invasion_reinforcements(),
        itll_quench_ya(),
        jeong_jeongs_deserters(),
        kyoshi_village(),
        kyoshi_warriors(),
        meditation_pools(),
        misty_palms_oasis(),
        north_pole_gates(),
        omashu_city(),
        otter_penguin(),
        pillar_launch(),
        rumble_arena(),
        serpents_pass(),
        south_pole_voyager(),
        suki_kyoshi_warrior(),
        sun_blessed_peak(),
        tiger_seal(),
        treetop_freedom_fighters(),
        walltop_sentries(),
        wandering_musicians(),
        water_tribe_captain(),
        water_tribe_rallier(),
        white_lotus_reinforcements(),
        yip_yip(),
        yuyan_archers(),
        ballistic_boulder(),
        bear(),
        dragon(),
        monk(),
        soldier(),
        spirit(),
        earthbending_lesson(),
        sandbenders_storm(),
        earth_village_ruffians(),
        geyser_leaper(),
        foggy_swamp_vinebender(),
        fire_nation_attacks(),
        rockalanche(),
        canyon_crawler(),
        mongoose_lizard(),
        saber_tooth_moose_lion(),
        giant_koi(),
        rabaroo_troop(),
        team_avatar(),
        merchant_of_many_hats(),
        hog_monkey(),
        rough_rhino_cavalry(),
        rebellious_captives(),
        barrels_of_blasting_jelly(),
        jasmine_dragon_tea_shop(),
        white_lotus_hideout(),
        waterbending_scroll(),
        aang_the_last_airbender(),
        airbending_lesson(),
        airbenders_reversal(),
        avatars_wrath(),
        zuko_exiled_prince(),
        aang_s_iceberg(),
        north_pole_patrol(),
        zuko_s_conviction(),
        sokka_bold_boomeranger(),
        seismic_sense(),
        aang_s_journey(),
        sparring_dummy(),
        lost_days(),
        serpent_of_the_pass(),
        elemental_teachings(),
        dai_li_indoctrination(),
        professor_zei_anthropologist(),
        united_front(),
        spirit_water_revival(),
        yue_the_moon_spirit(),
        solstice_revelations(),
    ]
}
fn abandon_attachments() -> Card {
    let text_box = "You may discard a card. If you do, draw two cards.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{U/R}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.abandon_attachments",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::OptionalDiscardDraw {
                count: 2,
                up_to: None,
            }],
        )
        .cost(Cost::mana("{1}{U/R}"))
        .inferred()],
        registry_name: "Abandon Attachments",
        text_box,
    }
}

fn azula_always_lies() -> Card {
    let text_box = "Choose one or both —\n• Target creature gets -1/-1 until end of turn.\n• Put a +1/+1 counter on target creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.azula_always_lies",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Modal {
                modes: vec![
                    vec![Op::TargetClause {
                        body: vec![Op::ModifyPt {
                            duration: Duration::EndOfTurn,
                            power: -1,
                            target: "target_0",
                            toughness: -1,
                        }],
                        role: "target_0",
                    }],
                    vec![Op::TargetClause {
                        body: vec![Op::PutCounters {
                            count: 1,
                            target: "target_1",
                        }],
                        role: "target_1",
                    }],
                    vec![
                        Op::TargetClause {
                            body: vec![Op::ModifyPt {
                                duration: Duration::EndOfTurn,
                                power: -1,
                                target: "target_2",
                                toughness: -1,
                            }],
                            role: "target_2",
                        },
                        Op::TargetClause {
                            body: vec![Op::PutCounters {
                                count: 1,
                                target: "target_3",
                            }],
                            role: "target_3",
                        },
                    ],
                ],
            }],
        )
        .cost(Cost::mana("{1}{B}"))
        .targets(vec![
            Target::one("target_0", Selector::Creature),
            Target::one("target_1", Selector::Creature),
            Target::one("target_2", Selector::Creature),
            Target::one("target_3", Selector::Creature),
        ])
        .inferred()],
        registry_name: "Azula Always Lies",
        text_box,
    }
}

fn boomerang_basics() -> Card {
    let text_box = "Return target nonland permanent to its owner's hand. If you controlled that permanent, draw a card.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{U}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.boomerang_basics",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::ReturnToHandDrawIfControlled { target: "target_0" }],
        )
        .cost(Cost::mana("{U}"))
        .targets(vec![Target::one(
            "target_0",
            Selector::Permanent {
                predicate: Predicate::NotCardTypes {
                    values: vec![CardType::Land],
                },
            },
        )])
        .inferred()],
        registry_name: "Boomerang Basics",
        text_box,
    }
}

fn cat_gator() -> Card {
    let text_box = "Lifelink\nWhen this creature enters, it deals damage equal to the number of Swamps you control to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{6}{B}"),
            subtypes: Some(vec!["Fish", "Crocodile"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Lifelink]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.cat_gator",
        programs: vec![Program::new(
            "trigger_0",
            ProgramKind::Triggered,
            vec![Op::DealDamageValue {
                amount: Quantity::BattlefieldMatching {
                    predicate: Predicate::Subtype { value: "Swamp" },
                },
                target: "target_0",
            }],
        )
        .targets(vec![Target::one("target_0", Selector::CreatureOrPlayer)])
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Cat-Gator",
        text_box,
    }
}

fn combustion_technique() -> Card {
    let text_box = "Combustion Technique deals damage equal to 2 plus the number of Lesson cards in your graveyard to target creature. If that creature would die this turn, exile it instead.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1R"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            keywords: Some(vec![]),
            token: Some(false),
            ..Default::default()
        },
        key: "tla.combustion_technique",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Spell,
            vec![
                Op::DealDamageValue {
                    amount: Quantity::GraveyardMatching {
                        base: 2,
                        predicate: Predicate::Subtype { value: "Lesson" },
                    },
                    target: "target",
                },
                Op::ExileIfDies { target: "target" },
            ],
        )
        .cost(Cost::mana("1R"))
        .targets(vec![Target::one("target", Selector::Creature)])],
        registry_name: "Combustion Technique",
        text_box,
    }
}

fn corrupt_court_official() -> Card {
    let text_box = "When this creature enters, target opponent discards a card.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            subtypes: Some(vec!["Human", "Advisor"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.corrupt_court_official",
        programs: vec![Program::new(
            "trigger_0",
            ProgramKind::Triggered,
            vec![Op::DiscardTargetPlayer {
                count: 1,
                target: "target_0",
            }],
        )
        .targets(vec![Target::one("target_0", Selector::Opponent)])
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Corrupt Court Official",
        text_box,
    }
}

fn dai_li_agents() -> Card {
    let text_box = "When this creature enters, earthbend 1, then earthbend 1. (To earthbend 1, target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)\nWhenever this creature attacks, each opponent loses X life and you gain X life, where X is the number of creatures you control with +1/+1 counters on them.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{B}{G}"),
            subtypes: Some(vec!["Human", "Soldier"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(4),
            ..Default::default()
        },
        key: "tla.dai_li_agents",
        programs: vec![
            Program::new(
                "trigger_0",
                ProgramKind::Triggered,
                vec![
                    Op::TargetClause {
                        body: vec![Op::Earthbend {
                            count: Count::Fixed(1),
                            target: "target_0",
                        }],
                        role: "target_0",
                    },
                    Op::TargetClause {
                        body: vec![Op::Earthbend {
                            count: Count::Fixed(1),
                            target: "target_1",
                        }],
                        role: "target_1",
                    },
                ],
            )
            .targets(vec![
                Target::one("target_0", Selector::LandYouControl),
                Target::one("target_1", Selector::LandYouControl),
            ])
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "trigger_1",
                ProgramKind::Triggered,
                vec![Op::DrainLifeValue {
                    amount: Quantity::CreaturesWithCounters,
                }],
            )
            .trigger(Trigger::new(Event::Attacks, Subject::This)),
        ],
        registry_name: "Dai Li Agents",
        text_box,
    }
}

fn deserter_s_disciple() -> Card {
    let text_box =
        "{T}: Another target creature you control with power 2 or less can't be blocked this turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            subtypes: Some(vec!["Human", "Rebel", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.deserter_s_disciple",
        programs: vec![Program::new(
            "activate_0",
            ProgramKind::Activated,
            vec![Op::CantBeBlockedTarget { target: "target_0" }],
        )
        .cost(Cost {
            mana: Some("0"),
            sacrifice_source: Some(false),
            tap_source: Some(true),
            waterbend: Some(false),
            ..Default::default()
        })
        .targets(vec![Target::one(
            "target_0",
            Selector::PermanentYouControl {
                predicate: Predicate::All {
                    predicates: vec![
                        Predicate::CardType {
                            value: CardType::Creature,
                        },
                        Predicate::PowerAtMost { value: 2 },
                    ],
                },
            },
        )])],
        registry_name: "Deserter's Disciple",
        text_box,
    }
}

fn enter_the_avatar_state() -> Card {
    let text_box = "Until end of turn, target creature you control becomes an Avatar in addition to its other types and gains flying, first strike, lifelink, and hexproof. (A creature with hexproof can't be the target of spells or abilities your opponents control.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("W"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            keywords: Some(vec![]),
            token: Some(false),
            ..Default::default()
        },
        key: "tla.enter_the_avatar_state",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Spell,
            vec![
                Op::AddSubtype {
                    subtype: "Avatar",
                    target: "target",
                },
                Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![
                        Keyword::Flying,
                        Keyword::FirstStrike,
                        Keyword::Lifelink,
                        Keyword::Hexproof,
                    ],
                    target: "target",
                },
            ],
        )
        .cost(Cost::mana("W"))
        .targets(vec![Target::one("target", Selector::CreatureYouControl)])],
        registry_name: "Enter the Avatar State",
        text_box,
    }
}

fn epic_downfall() -> Card {
    let text_box = "Exile target creature with mana value 3 or greater.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.epic_downfall",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Exile { target: "target_0" }],
        )
        .cost(Cost::mana("{1}{B}"))
        .targets(vec![Target::one(
            "target_0",
            Selector::Permanent {
                predicate: Predicate::All {
                    predicates: vec![
                        Predicate::CardType {
                            value: CardType::Creature,
                        },
                        Predicate::ManaValueAtLeast { value: 3 },
                    ],
                },
            },
        )])
        .inferred()],
        registry_name: "Epic Downfall",
        text_box,
    }
}

fn foggy_swamp_spirit_keeper() -> Card {
    let text_box = "Lifelink\nWhenever you draw your second card each turn, create a 1/1 colorless Spirit creature token with \"This token can't block or be blocked by non-Spirit creatures.\"";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{U}{B}"),
            subtypes: Some(vec!["Human", "Druid", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Lifelink]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(4),
            ..Default::default()
        },
        key: "tla.foggy_swamp_spirit_keeper",
        programs: vec![Program::new(
            "trigger_0",
            ProgramKind::Triggered,
            vec![Op::CreateToken {
                count: 1,
                definition_ref: "token.spirit",
                tapped_and_attacking: Some(false),
            }],
        )
        .trigger(Trigger {
            n: Some(2),
            ..Trigger::new(Event::DrawNthCard, Subject::This)
        })],
        registry_name: "Foggy Swamp Spirit Keeper",
        text_box,
    }
}

fn gather_the_white_lotus() -> Card {
    let text_box = "Create a 1/1 white Ally creature token for each Plains you control. Scry 2. (Look at the top two cards of your library, then put any number of them on the bottom and the rest on top in any order.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{W}"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.gather_the_white_lotus",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::CreateTokensValue {
                    count: Quantity::BattlefieldMatching {
                        predicate: Predicate::Subtype { value: "Plains" },
                    },
                    definition_ref: "token.ally",
                },
                Op::Scry { count: 2 },
            ],
        )
        .cost(Cost::mana("{4}{W}"))
        .inferred()],
        registry_name: "Gather the White Lotus",
        text_box,
    }
}

fn gran_gran() -> Card {
    let text_box = "Whenever Gran-Gran becomes tapped, draw a card, then discard a card.\nNoncreature spells you cast cost {1} less to cast as long as there are three or more Lesson cards in your graveyard.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("U"),
            subtypes: Some(vec!["Human", "Peasant", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(2),
            token: Some(false),
            supertypes: Some(vec!["Legendary"]),
            ..Default::default()
        },
        key: "tla.gran_gran",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::DrawCards { count: 1 }, Op::Discard],
            )
            .trigger(Trigger::new(Event::BecomesTapped, Subject::This)),
            Program::new(
                "p1",
                ProgramKind::Static,
                vec![Op::SpellCostReduction {
                    condition: Condition::GraveyardAtLeast {
                        count: 3,
                        predicate: Predicate::Subtype { value: "Lesson" },
                    },
                    first_on_own_turn: false,
                    generic: 1,
                    predicate: Predicate::NotCardTypes {
                        values: vec![CardType::Creature],
                    },
                }],
            ),
        ],
        registry_name: "Gran-Gran",
        text_box,
    }
}

fn haru_hidden_talent() -> Card {
    let text_box = "Whenever another Ally you control enters, earthbend 1. (Target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{G}"),
            subtypes: Some(vec!["Human", "Peasant", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.haru_hidden_talent",
        programs: vec![Program::new(
            "trigger_0",
            ProgramKind::Triggered,
            vec![Op::Earthbend {
                count: Count::Fixed(1),
                target: "target_0",
            }],
        )
        .targets(vec![Target::one("target_0", Selector::LandYouControl)])
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnotherYouControl {
                predicate: Predicate::Subtype { value: "Ally" },
            },
        ))],
        registry_name: "Haru, Hidden Talent",
        text_box,
    }
}

fn iroh_s_demonstration() -> Card {
    let text_box = "Choose one —\n• Iroh's Demonstration deals 1 damage to each creature your opponents control.\n• Iroh's Demonstration deals 4 damage to target creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.iroh_s_demonstration",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Modal {
                modes: vec![
                    vec![Op::DamageOpponentCreatures { amount: 1 }],
                    vec![Op::DealDamage {
                        amount: 4,
                        target: "target_0",
                    }],
                ],
            }],
        )
        .cost(Cost::mana("{1}{R}"))
        .targets(vec![Target::one("target_0", Selector::Creature)])
        .inferred()],
        registry_name: "Iroh's Demonstration",
        text_box,
    }
}

fn jet_freedom_fighter() -> Card {
    let text_box = "When Jet enters, he deals damage equal to the number of creatures you control to target creature an opponent controls.\nWhen Jet dies, put a +1/+1 counter on each of up to two target creatures.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{R/W}{R/W}{R/W}"),
            subtypes: Some(vec!["Human", "Rebel", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(1),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.jet_freedom_fighter",
        programs: vec![
            Program::new(
                "trigger_0",
                ProgramKind::Triggered,
                vec![Op::DealDamageValue {
                    amount: Quantity::BattlefieldMatching {
                        predicate: Predicate::CardType {
                            value: CardType::Creature,
                        },
                    },
                    target: "target_0",
                }],
            )
            .targets(vec![Target::one(
                "target_0",
                Selector::CreatureOpponentControls,
            )])
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "trigger_1",
                ProgramKind::Triggered,
                vec![Op::TargetClause {
                    body: vec![Op::ForEachTarget {
                        body: vec![Op::PutCounters {
                            count: 1,
                            target: "target_0",
                        }],
                        role: "target_0",
                    }],
                    role: "target_0",
                }],
            )
            .targets(vec![Target {
                another: false,
                max: 2,
                min: 0,
                role: "target_0",
                selector: Selector::Creature,
            }])
            .trigger(Trigger::new(Event::Dies, Subject::This)),
        ],
        registry_name: "Jet, Freedom Fighter",
        text_box,
    }
}

fn knowledge_seeker() -> Card {
    let text_box = "Vigilance\nWhenever you draw your second card each turn, put a +1/+1 counter on this creature.\nWhen this creature dies, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{U}"),
            subtypes: Some(vec!["Fox", "Spirit"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Vigilance]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.knowledge_seeker",
        programs: vec![
            Program::new(
                "trigger_0",
                ProgramKind::Triggered,
                vec![Op::PutCounters {
                    count: 1,
                    target: "source",
                }],
            )
            .trigger(Trigger {
                n: Some(2),
                ..Trigger::new(Event::DrawNthCard, Subject::This)
            }),
            Program::new(
                "trigger_1",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(Event::Dies, Subject::This)),
        ],
        registry_name: "Knowledge Seeker",
        text_box,
    }
}

fn lightning_strike() -> Card {
    let text_box = "Lightning Strike deals 3 damage to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.lightning_strike",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::DealDamage {
                amount: 3,
                target: "target_0",
            }],
        )
        .cost(Cost::mana("{1}{R}"))
        .targets(vec![Target::one("target_0", Selector::CreatureOrPlayer)])
        .inferred()],
        registry_name: "Lightning Strike",
        text_box,
    }
}

fn master_piandao() -> Card {
    let text_box = "First strike\nWhenever Master Piandao attacks, look at the top four cards of your library. You may reveal an Ally, Equipment, or Lesson card from among them and put it into your hand. Put the rest on the bottom of your library in a random order.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{W}"),
            subtypes: Some(vec!["Human", "Warrior", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::FirstStrike]),
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(4),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.master_piandao",
        programs: vec![Program::new(
            "trigger_0",
            ProgramKind::Triggered,
            vec![Op::LookAndSelect {
                destination: Destination::Hand,
                look: Count::Fixed(4),
                max_select: 1,
                min_select: 0,
                predicate: Predicate::SubtypesAny {
                    values: vec!["Ally", "Equipment", "Lesson"],
                },
                reveal: Some(true),
                remainder: None,
                remainder_order: None,
            }],
        )
        .trigger(Trigger::new(Event::Attacks, Subject::This))],
        registry_name: "Master Piandao",
        text_box,
    }
}

fn messenger_hawk() -> Card {
    let text_box = "Flying\nWhen this creature enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nThis creature gets +2/+0 as long as you've drawn two or more cards this turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{U/B}"),
            subtypes: Some(vec!["Bird", "Scout"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.messenger_hawk",
        programs: vec![
            Program::new(
                "trigger_0",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "pt_0",
                ProgramKind::Static,
                vec![Op::StaticPt {
                    condition: Condition::CardsDrawnAtLeast { count: 2 },
                    power: 2,
                    toughness: 0,
                }],
            ),
        ],
        registry_name: "Messenger Hawk",
        text_box,
    }
}

fn momo_friendly_flier() -> Card {
    let text_box = "Flying\nThe first non-Lemur creature spell with flying you cast during each of your turns costs {1} less to cast.\nWhenever another creature you control with flying enters, Momo gets +1/+1 until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{W}"),
            subtypes: Some(vec!["Lemur", "Bat", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.momo_friendly_flier",
        programs: vec![
            Program::new(
                "trigger_0",
                ProgramKind::Triggered,
                vec![Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 1,
                    target: "source",
                    toughness: 1,
                }],
            )
            .trigger(Trigger::new(
                Event::EntersBattlefield,
                Subject::AnotherYouControl {
                    predicate: Predicate::All {
                        predicates: vec![
                            Predicate::Flying,
                            Predicate::CardType {
                                value: CardType::Creature,
                            },
                        ],
                    },
                },
            )),
            Program::new(
                "reduce_cost_0",
                ProgramKind::Static,
                vec![Op::SpellCostReduction {
                    condition: Condition::Always,
                    first_on_own_turn: true,
                    generic: 1,
                    predicate: Predicate::All {
                        predicates: vec![
                            Predicate::Flying,
                            Predicate::NotSubtypes {
                                values: vec!["Lemur"],
                            },
                            Predicate::CardType {
                                value: CardType::Creature,
                            },
                        ],
                    },
                }],
            ),
        ],
        registry_name: "Momo, Friendly Flier",
        text_box,
    }
}

fn octopus_form() -> Card {
    let text_box = "Target creature you control gets +1/+1 and gains hexproof until end of turn. Untap it. (It can't be the target of spells or abilities your opponents control.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{U}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.octopus_form",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 1,
                    target: "target_0",
                    toughness: 1,
                },
                Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![Keyword::Hexproof],
                    target: "target_0",
                },
                Op::Untap { target: "target_0" },
            ],
        )
        .cost(Cost::mana("{U}"))
        .targets(vec![Target::one("target_0", Selector::CreatureYouControl)])
        .inferred()],
        registry_name: "Octopus Form",
        text_box,
    }
}

fn origin_of_metalbending() -> Card {
    let text_box = "Choose one —\n• Destroy target artifact or enchantment.\n• Put a +1/+1 counter on target creature you control. It gains indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{G}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.origin_of_metalbending",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Modal {
                modes: vec![
                    vec![Op::Destroy { target: "target_0" }],
                    vec![
                        Op::PutCounters {
                            count: 1,
                            target: "target_1",
                        },
                        Op::GrantKeywords {
                            duration: Duration::EndOfTurn,
                            keywords: vec![Keyword::Indestructible],
                            target: "target_1",
                        },
                    ],
                ],
            }],
        )
        .cost(Cost::mana("{1}{G}"))
        .targets(vec![
            Target::one(
                "target_0",
                Selector::Permanent {
                    predicate: Predicate::TypesAny {
                        values: vec![CardType::Artifact, CardType::Enchantment],
                    },
                },
            ),
            Target::one("target_1", Selector::CreatureYouControl),
        ])
        .inferred()],
        registry_name: "Origin of Metalbending",
        text_box,
    }
}

fn ostrich_horse() -> Card {
    let text_box = "When this creature enters, mill three cards. You may put a land card from among them into your hand. If you don't, put a +1/+1 counter on this creature. (To mill three cards, put the top three cards of your library into your graveyard.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{G}"),
            subtypes: Some(vec!["Bird", "Horse"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.ostrich_horse",
        programs: vec![Program::new(
            "trigger_0",
            ProgramKind::Triggered,
            vec![Op::MillAndSelect {
                count: 3,
                if_declined: vec![Op::PutCounters {
                    count: 1,
                    target: "source",
                }],
                predicate: Predicate::CardType {
                    value: CardType::Land,
                },
                milled_any: None,
                then: Vec::new(),
            }],
        )
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Ostrich-Horse",
        text_box,
    }
}

fn ozai_s_cruelty() -> Card {
    let text_box =
        "Ozai's Cruelty deals 2 damage to target player. That player discards two cards.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{B}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.ozai_s_cruelty",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::DealDamage {
                    amount: 2,
                    target: "target_0",
                },
                Op::DiscardTargetPlayer {
                    count: 2,
                    target: "target_0",
                },
            ],
        )
        .cost(Cost::mana("{2}{B}"))
        .targets(vec![Target::one("target_0", Selector::Player)])
        .inferred()],
        registry_name: "Ozai's Cruelty",
        text_box,
    }
}

fn pretending_poxbearers() -> Card {
    let text_box = "When this creature dies, create a 1/1 white Ally creature token.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{W/B}"),
            subtypes: Some(vec!["Human", "Citizen", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.pretending_poxbearers",
        programs: vec![Program::new(
            "trigger_0",
            ProgramKind::Triggered,
            vec![Op::CreateToken {
                count: 1,
                definition_ref: "token.ally",
                tapped_and_attacking: Some(false),
            }],
        )
        .trigger(Trigger::new(Event::Dies, Subject::This))],
        registry_name: "Pretending Poxbearers",
        text_box,
    }
}

fn shared_roots() -> Card {
    let text_box = "Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{G}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.shared_roots",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::SearchBasicLand {
                destination: None,
                subtype: None,
            }],
        )
        .cost(Cost::mana("{1}{G}"))
        .inferred()],
        registry_name: "Shared Roots",
        text_box,
    }
}

fn sokka_s_haiku() -> Card {
    let text_box = "Counter target spell.\nDraw a card, then mill three cards.\nUntap target land.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{U}{U}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.sokka_s_haiku",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::TargetClause {
                    body: vec![Op::CounterSpell { target: "target_0" }],
                    role: "target_0",
                },
                Op::DrawCards { count: 1 },
                Op::MillCards { count: 3 },
                Op::TargetClause {
                    body: vec![Op::Untap { target: "target_1" }],
                    role: "target_1",
                },
            ],
        )
        .cost(Cost::mana("{3}{U}{U}"))
        .targets(vec![
            Target::one("target_0", Selector::Spell),
            Target::one(
                "target_1",
                Selector::Permanent {
                    predicate: Predicate::CardType {
                        value: CardType::Land,
                    },
                },
            ),
        ])
        .inferred()],
        registry_name: "Sokka's Haiku",
        text_box,
    }
}

fn tiger_dillo() -> Card {
    let text_box = "This creature can't attack or block unless you control another creature with power 4 or greater.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            subtypes: Some(vec!["Cat", "Armadillo"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.tiger_dillo",
        programs: vec![Program::new(
            "combat_eligibility",
            ProgramKind::Static,
            vec![Op::CombatRequiresOther {
                predicate: Predicate::All {
                    predicates: vec![
                        Predicate::PowerAtLeast { value: 4 },
                        Predicate::CardType {
                            value: CardType::Creature,
                        },
                    ],
                },
            }],
        )],
        registry_name: "Tiger-Dillo",
        text_box,
    }
}

fn toph_the_blind_bandit() -> Card {
    let text_box = "When Toph enters, earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)\nToph's power is equal to the number of +1/+1 counters on lands you control.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{G}"),
            subtypes: Some(vec!["Human", "Warrior", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(0)),
            toughness: Some(3),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.toph_the_blind_bandit",
        programs: vec![
            Program::new(
                "trigger_0",
                ProgramKind::Triggered,
                vec![Op::Earthbend {
                    count: Count::Fixed(2),
                    target: "target_0",
                }],
            )
            .targets(vec![Target::one("target_0", Selector::LandYouControl)])
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "power",
                ProgramKind::Static,
                vec![Op::PowerFromLandCounters],
            ),
        ],
        registry_name: "Toph, the Blind Bandit",
        text_box,
    }
}

fn true_ancestry() -> Card {
    let text_box = "Return up to one target permanent card from your graveyard to your hand.\nCreate a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{G}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.true_ancestry",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::TargetClause {
                    body: vec![Op::ReturnGraveyardTarget {
                        target: "target_0",
                        destination: None,
                    }],
                    role: "target_0",
                },
                Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: Some(false),
                },
            ],
        )
        .cost(Cost::mana("{1}{G}"))
        .targets(vec![Target {
            another: false,
            max: 1,
            min: 0,
            role: "target_0",
            selector: Selector::GraveyardCardYouOwn {
                predicate: Predicate::TypesAny {
                    values: vec![
                        CardType::Artifact,
                        CardType::Creature,
                        CardType::Enchantment,
                        CardType::Land,
                        CardType::Planeswalker,
                        CardType::Battle,
                    ],
                },
            },
        }])
        .inferred()],
        registry_name: "True Ancestry",
        text_box,
    }
}

fn uncle_iroh() -> Card {
    let text_box = "Firebending 1 (Whenever this creature attacks, add {R}. This mana lasts until end of combat.)\nLesson spells you cast cost {1} less to cast.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R/G}{R/G}"),
            subtypes: Some(vec!["Human", "Noble", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(2),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.uncle_iroh",
        programs: vec![
            Program::firebending("trigger_0", "R"),
            Program::new(
                "reduce_cost_0",
                ProgramKind::Static,
                vec![Op::SpellCostReduction {
                    condition: Condition::Always,
                    first_on_own_turn: false,
                    generic: 1,
                    predicate: Predicate::Subtype { value: "Lesson" },
                }],
            ),
        ],
        registry_name: "Uncle Iroh",
        text_box,
    }
}

fn wartime_protestors() -> Card {
    let text_box = "Haste\nWhenever another Ally you control enters, put a +1/+1 counter on that creature and it gains haste until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{R}"),
            subtypes: Some(vec!["Human", "Rebel", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Haste]),
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(4),
            ..Default::default()
        },
        key: "tla.wartime_protestors",
        programs: vec![Program::new(
            "trigger_0",
            ProgramKind::Triggered,
            vec![Op::ModifyTriggeringPermanent {
                counters: 1,
                keywords: vec![Keyword::Haste],
            }],
        )
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnotherYouControl {
                predicate: Predicate::Subtype { value: "Ally" },
            },
        ))],
        registry_name: "Wartime Protestors",
        text_box,
    }
}

fn zuko_s_exile() -> Card {
    let text_box = "Exile target artifact, creature, or enchantment. Its controller creates a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{5}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.zuko_s_exile",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::ExileWithControllerToken {
                definition_ref: "token.clue",
                target: "target_0",
            }],
        )
        .cost(Cost::mana("{5}"))
        .targets(vec![Target::one(
            "target_0",
            Selector::Permanent {
                predicate: Predicate::TypesAny {
                    values: vec![
                        CardType::Artifact,
                        CardType::Creature,
                        CardType::Enchantment,
                    ],
                },
            },
        )])
        .inferred()],
        registry_name: "Zuko's Exile",
        text_box,
    }
}

fn ally() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            token: Some(true),
            supertypes: Some(vec![]),
            colors: Some(vec![Color::W]),
            ..Default::default()
        },
        key: "token.ally",
        programs: vec![],
        registry_name: "Ally",
        text_box,
    }
}

fn callous_inspector() -> Card {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nWhen this creature dies, it deals 1 damage to you. Create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("B"),
            subtypes: Some(vec!["Human", "Soldier"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Menace]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.callous_inspector",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![
                Op::DealDamageToController { amount: 1 },
                Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: Some(false),
                },
            ],
        )
        .trigger(Trigger::new(Event::Dies, Subject::This))],
        registry_name: "Callous Inspector",
        text_box,
    }
}

fn clue() -> Card {
    let text_box = "{2}, Sacrifice this token: Draw a card.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Clue"]),
            types: vec![CardType::Artifact],
            keywords: Some(vec![]),
            token: Some(true),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "token.clue",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::DrawCards { count: 1 }],
        )
        .cost(Cost {
            mana: Some("2"),
            sacrifice_source: Some(true),
            ..Default::default()
        })],
        registry_name: "Clue",
        text_box,
    }
}

fn curious_farm_animals() -> Card {
    let text_box = "When this creature dies, you gain 3 life.\n{2}, Sacrifice this creature: Destroy up to one target artifact or enchantment.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("W"),
            subtypes: Some(vec!["Boar", "Elk", "Bird", "Ox"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.curious_farm_animals",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::GainLife { amount: 3 }],
            )
            .trigger(Trigger::new(Event::Dies, Subject::This)),
            Program::new(
                "p1",
                ProgramKind::Activated,
                vec![Op::TargetClause {
                    body: vec![Op::Destroy { target: "target_0" }],
                    role: "target_0",
                }],
            )
            .cost(Cost {
                mana: Some("2"),
                sacrifice_source: Some(true),
                ..Default::default()
            })
            .targets(vec![Target {
                another: false,
                max: 1,
                min: 0,
                role: "target_0",
                selector: Selector::Permanent {
                    predicate: Predicate::TypesAny {
                        values: vec![CardType::Artifact, CardType::Enchantment],
                    },
                },
            }])
            .inferred(),
        ],
        registry_name: "Curious Farm Animals",
        text_box,
    }
}

fn cycle_of_renewal() -> Card {
    let text_box = "Sacrifice a land. Search your library for up to two basic land cards, put them onto the battlefield tapped, then shuffle.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2G"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.cycle_of_renewal",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Spell,
            vec![
                Op::Sacrifice {
                    predicate: Predicate::CardType {
                        value: CardType::Land,
                    },
                },
                Op::SearchBasicLands {
                    bottom_source: false,
                    count: 2,
                    tapped: true,
                    untap_if_lands: None,
                },
            ],
        )
        .cost(Cost::mana("2G"))],
        registry_name: "Cycle of Renewal",
        text_box,
    }
}

fn deadly_precision() -> Card {
    let text_box = "As an additional cost to cast this spell, pay {4} or sacrifice an artifact or creature.\nDestroy target creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("B"),
            subtypes: Some(vec![]),
            types: vec![CardType::Sorcery],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.deadly_precision",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Spell,
            vec![Op::Destroy { target: "target_0" }],
        )
        .cost(Cost {
            mana: Some("B"),
            sacrifice: Some(Sacrifice {
                another: false,
                or_mana: Some("4"),
                predicate: Predicate::TypesAny {
                    values: vec![CardType::Artifact, CardType::Creature],
                },
            }),
            ..Default::default()
        })
        .targets(vec![Target::one("target_0", Selector::Creature)])
        .inferred()],
        registry_name: "Deadly Precision",
        text_box,
    }
}

fn food() -> Card {
    let text_box = "{2}, {T}, Sacrifice this token: You gain 3 life.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Food"]),
            types: vec![CardType::Artifact],
            keywords: Some(vec![]),
            token: Some(true),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "token.food",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::GainLife { amount: 3 }],
        )
        .cost(Cost {
            mana: Some("2"),
            sacrifice_source: Some(true),
            tap_source: Some(true),
            ..Default::default()
        })],
        registry_name: "Food",
        text_box,
    }
}

fn june_bounty_hunter() -> Card {
    let text_box = "June can't be blocked as long as you've drawn two or more cards this turn.\n{1}, Sacrifice another creature: Create a Clue token. Activate only during your turn. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1B"),
            subtypes: Some(vec!["Human", "Mercenary"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            token: Some(false),
            supertypes: Some(vec!["Legendary"]),
            ..Default::default()
        },
        key: "tla.june_bounty_hunter",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Activated,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: Some(false),
                }],
            )
            .cost(Cost {
                mana: Some("1"),
                sacrifice: Some(Sacrifice {
                    another: true,
                    or_mana: None,
                    predicate: Predicate::CardType {
                        value: CardType::Creature,
                    },
                }),
                own_turn_only: Some(true),
                ..Default::default()
            }),
            Program::new(
                "unblockable",
                ProgramKind::Static,
                vec![Op::UnblockableIf {
                    condition: Condition::CardsDrawnAtLeast { count: 2 },
                }],
            ),
        ],
        registry_name: "June, Bounty Hunter",
        text_box,
    }
}

fn tolls_of_war() -> Card {
    let text_box = "When this enchantment enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you sacrifice a permanent during your turn, create a 1/1 white Ally creature token. This ability triggers only once each turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("WB"),
            subtypes: Some(vec![]),
            types: vec![CardType::Enchantment],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.tolls_of_war",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "p1",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.ally",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger {
                predicate: Some(Predicate::Any),
                controller_only: Some(true),
                once_each_turn: Some(true),
                own_turn_only: Some(true),
                ..Trigger::new(Event::Sacrificed, Subject::This)
            }),
        ],
        registry_name: "Tolls of War",
        text_box,
    }
}

fn unlucky_cabbage_merchant() -> Card {
    let text_box = "When this creature enters, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nWhenever you sacrifice a Food, you may search your library for a basic land card and put it onto the battlefield tapped. If you search your library this way, put this creature on the bottom of its owner's library, then shuffle.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1G"),
            subtypes: Some(vec!["Human", "Citizen"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.unlucky_cabbage_merchant",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.food",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "p1",
                ProgramKind::Triggered,
                vec![Op::Modal {
                    modes: vec![
                        vec![Op::SearchBasicLands {
                            bottom_source: true,
                            count: 1,
                            tapped: true,
                            untap_if_lands: None,
                        }],
                        vec![],
                    ],
                }],
            )
            .trigger(Trigger {
                predicate: Some(Predicate::Subtype { value: "Food" }),
                controller_only: Some(true),
                ..Trigger::new(Event::Sacrificed, Subject::This)
            }),
        ],
        registry_name: "Unlucky Cabbage Merchant",
        text_box,
    }
}

fn great_divide_guide() -> Card {
    let text_box = "Each land and Ally you control has \"{T}: Add one mana of any color.\"";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{G}"),
            subtypes: Some(vec!["Human", "Scout", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(3),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.great_divide_guide",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Static,
            vec![Op::GrantMana {
                predicates: vec![
                    Predicate::CardType {
                        value: CardType::Land,
                    },
                    Predicate::Subtype { value: "Ally" },
                ],
            }],
        )],
        registry_name: "Great Divide Guide",
        text_box,
    }
}

fn hermitic_herbalist() -> Card {
    let text_box = "{T}: Add one mana of any color.\n{T}: Add two mana in any combination of colors. Spend this mana only to cast Lesson spells.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{G}{U}"),
            subtypes: Some(vec!["Human", "Druid", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(3),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.hermitic_herbalist",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Activated,
                vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
            )
            .cost(Cost {
                mana: Some("0"),
                tap_source: Some(true),
                ..Default::default()
            }),
            Program::new(
                "p1",
                ProgramKind::Activated,
                vec![Op::Modal {
                    modes: vec![
                        vec![Op::lesson_mana("WW")],
                        vec![Op::lesson_mana("WU")],
                        vec![Op::lesson_mana("WB")],
                        vec![Op::lesson_mana("WR")],
                        vec![Op::lesson_mana("WG")],
                        vec![Op::lesson_mana("UU")],
                        vec![Op::lesson_mana("UB")],
                        vec![Op::lesson_mana("UR")],
                        vec![Op::lesson_mana("UG")],
                        vec![Op::lesson_mana("BB")],
                        vec![Op::lesson_mana("BR")],
                        vec![Op::lesson_mana("BG")],
                        vec![Op::lesson_mana("RR")],
                        vec![Op::lesson_mana("RG")],
                        vec![Op::lesson_mana("GG")],
                    ],
                }],
            )
            .cost(Cost {
                mana: Some("0"),
                tap_source: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Hermitic Herbalist",
        text_box,
    }
}

fn iroh_grand_lotus() -> Card {
    let text_box = "Firebending 2\nDuring your turn, each non-Lesson instant and sorcery card in your graveyard has flashback. The flashback cost is equal to that card's mana cost. (You may cast a card from your graveyard for its flashback cost. Then exile it.)\nDuring your turn, each Lesson card in your graveyard has flashback {1}.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{G}{U}{R}"),
            subtypes: Some(vec!["Human", "Noble", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(5)),
            toughness: Some(5),
            token: Some(false),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.iroh_grand_lotus",
        programs: vec![
            Program::firebending("p0", "RR"),
            Program::new("p1", ProgramKind::Static, vec![Op::GrantFlashback]),
        ],
        registry_name: "Iroh, Grand Lotus",
        text_box,
    }
}

fn ran_and_shaw() -> Card {
    let text_box = "Flying, firebending 2\nWhen Ran and Shaw enter, if you cast them and there are three or more Dragon and/or Lesson cards in your graveyard, create a token that's a copy of Ran and Shaw, except it's not legendary.\n{3}{R}: Dragons you control get +2/+0 until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{R}{R}"),
            subtypes: Some(vec!["Dragon"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(4),
            token: Some(false),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.ran_and_shaw",
        programs: vec![
            Program::firebending("p0", "RR"),
            Program::new(
                "p1",
                ProgramKind::Triggered,
                vec![Op::Branch {
                    condition: Condition::GraveyardAtLeast {
                        count: 3,
                        predicate: Predicate::SubtypesAny {
                            values: vec!["Dragon", "Lesson"],
                        },
                    },
                    otherwise: vec![],
                    then: vec![Op::CreateNonlegendaryCopy],
                }],
            )
            .trigger(Trigger {
                condition: Some(Condition::GraveyardAtLeast {
                    count: 3,
                    predicate: Predicate::SubtypesAny {
                        values: vec!["Dragon", "Lesson"],
                    },
                }),
                ..Trigger::new(Event::CastEnters, Subject::This)
            }),
            Program::new(
                "p2",
                ProgramKind::Activated,
                vec![Op::ModifyControlled {
                    power: 2,
                    predicate: Predicate::Subtype { value: "Dragon" },
                    toughness: 0,
                }],
            )
            .cost(Cost::mana("3R")),
        ],
        registry_name: "Ran and Shaw",
        text_box,
    }
}

fn waterbending_lesson() -> Card {
    let text_box = "Draw three cards. Then discard a card unless you waterbend {2}. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{U}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "tla.waterbending_lesson",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Spell,
            vec![
                Op::DrawCards { count: 3 },
                Op::OptionalPayment {
                    cost: Payment::Waterbend(2),
                    if_declined: vec![Op::Discard],
                    if_paid: vec![],
                },
            ],
        )
        .cost(Cost::mana("3U"))],
        registry_name: "Waterbending Lesson",
        text_box,
    }
}

fn accumulate_wisdom() -> Card {
    let text_box = "Look at the top three cards of your library. Put one of those cards into your hand and the rest on the bottom of your library in any order. Put each of those cards into your hand instead if there are three or more Lesson cards in your graveyard.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1U"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.accumulate_wisdom",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Branch {
                condition: Condition::GraveyardAtLeast {
                    count: 3,
                    predicate: Predicate::Subtype { value: "Lesson" },
                },
                otherwise: vec![Op::LookAndSelect {
                    destination: Destination::Hand,
                    look: Count::Fixed(3),
                    max_select: 1,
                    min_select: 1,
                    predicate: Predicate::Any,
                    reveal: None,
                    remainder: Some(Remainder::LibraryBottom),
                    remainder_order: Some(RemainderOrder::Random),
                }],
                then: vec![Op::PutTopCardsInHand { count: 3 }],
            }],
        )
        .cost(Cost::mana("1U"))],
        registry_name: "Accumulate Wisdom",
        text_box,
    }
}

fn airship_engine_room() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {U} or {R}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land(
        "tla.airship_engine_room",
        "Airship Engine Room",
        text_box,
        ["U", "R"],
    )
}

fn allies_at_last() -> Card {
    let text_box = "Affinity for Allies (This spell costs {1} less to cast for each Ally you control.)\nUp to two target creatures you control each deal damage equal to their power to target creature an opponent controls.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2G"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.allies_at_last",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::DealPowerDamage {
                sources: "allies",
                target: "victim",
            }],
        )
        .cost(Cost {
            mana: Some("2G"),
            affinity: Some(Predicate::Subtype { value: "Ally" }),
            ..Default::default()
        })
        .targets(vec![
            Target {
                another: false,
                max: 2,
                min: 0,
                role: "allies",
                selector: Selector::CreatureYouControl,
            },
            Target::one("victim", Selector::CreatureOpponentControls),
        ])],
        registry_name: "Allies at Last",
        text_box,
    }
}

fn avatar_enthusiasts() -> Card {
    let text_box =
        "Whenever another Ally you control enters, put a +1/+1 counter on this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{W}"),
            subtypes: Some(vec!["Human", "Peasant", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.avatar_enthusiasts",
        programs: vec![Program::new(
            "ally_enters",
            ProgramKind::Triggered,
            vec![Op::PutCounters {
                count: 1,
                target: "source",
            }],
        )
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnotherYouControl {
                predicate: Predicate::Subtype { value: "Ally" },
            },
        ))],
        registry_name: "Avatar Enthusiasts",
        text_box,
    }
}

fn badgermole_cub() -> Card {
    let text_box = "When this creature enters, earthbend 1. (Target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)\nWhenever you tap a creature for mana, add an additional {G}.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1G"),
            subtypes: Some(vec!["Badger", "Mole"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.badgermole_cub",
        programs: vec![
            Program::new(
                "earthbend",
                ProgramKind::Triggered,
                vec![Op::Earthbend {
                    count: Count::Fixed(1),
                    target: "land",
                }],
            )
            .targets(vec![Target::one("land", Selector::LandYouControl)])
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "waterbend_bonus",
                ProgramKind::TriggeredMana,
                vec![Op::AddMana {
                    mana: "G",
                    until: None,
                }],
            )
            .trigger(Trigger::new(
                Event::TappedForMana,
                Subject::AnyYouControl {
                    predicate: Predicate::CardType {
                        value: CardType::Creature,
                    },
                },
            )),
        ],
        registry_name: "Badgermole Cub",
        text_box,
    }
}

fn boar_q_pine() -> Card {
    let text_box = "Whenever you cast a noncreature spell, put a +1/+1 counter on this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{R}"),
            subtypes: Some(vec!["Boar", "Porcupine"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.boar_q_pine",
        programs: vec![Program::new(
            "grow",
            ProgramKind::Triggered,
            vec![Op::PutCounters {
                count: 1,
                target: "source",
            }],
        )
        .trigger(Trigger {
            predicate: Some(Predicate::NotCardTypes {
                values: vec![CardType::Creature],
            }),
            ..Trigger::new(Event::CastSpell, Subject::This)
        })],
        registry_name: "Boar-q-pine",
        text_box,
    }
}

fn boiling_rock_prison() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {B} or {R}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land(
        "tla.boiling_rock_prison",
        "Boiling Rock Prison",
        text_box,
        ["B", "R"],
    )
}

fn cat_owl() -> Card {
    let text_box = "Flying\nWhenever this creature attacks, untap target artifact or creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{W/U}"),
            subtypes: Some(vec!["Cat", "Bird"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.cat_owl",
        programs: vec![Program::new(
            "ability",
            ProgramKind::Triggered,
            vec![Op::Untap {
                target: "permanent",
            }],
        )
        .targets(vec![Target::one(
            "permanent",
            Selector::Permanent {
                predicate: Predicate::TypesAny {
                    values: vec![CardType::Artifact, CardType::Creature],
                },
            },
        )])
        .trigger(Trigger::new(Event::Attacks, Subject::This))],
        registry_name: "Cat-Owl",
        text_box,
    }
}

fn compassionate_healer() -> Card {
    let text_box = "Whenever this creature becomes tapped, you gain 1 life and scry 1. (Look at the top card of your library. You may put it on the bottom.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1W"),
            subtypes: Some(vec!["Human", "Cleric", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.compassionate_healer",
        programs: vec![Program::new(
            "tap_value",
            ProgramKind::Triggered,
            vec![Op::GainLife { amount: 1 }, Op::Scry { count: 1 }],
        )
        .trigger(Trigger::new(Event::BecomesTapped, Subject::This))],
        registry_name: "Compassionate Healer",
        text_box,
    }
}

fn cunning_maneuver() -> Card {
    let text_box = "Target creature gets +3/+1 until end of turn.\nCreate a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            subtypes: Some(vec![]),
            types: vec![CardType::Instant],
            keywords: Some(vec![]),
            ..Default::default()
        },
        key: "tla.cunning_maneuver",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 3,
                    target: "creature",
                    toughness: 1,
                },
                Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: None,
                },
            ],
        )
        .cost(Cost::mana("{1}{R}"))
        .targets(vec![Target::one("creature", Selector::Creature)])],
        registry_name: "Cunning Maneuver",
        text_box,
    }
}

fn dragonfly_swarm() -> Card {
    let text_box = "Flying, ward {1} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {1}.)\nThis creature's power is equal to the number of noncreature, nonland cards in your graveyard.\nWhen this creature dies, if there's a Lesson card in your graveyard, draw a card.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1UR"),
            subtypes: Some(vec!["Dragon", "Insect"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::CharacteristicDefined),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.dragonfly_swarm",
        programs: vec![
            Program::new(
                "graveyard_power",
                ProgramKind::Static,
                vec![Op::SetPowerFromCount {
                    controller: Controller::You,
                    predicate: Predicate::NotCardTypes {
                        values: vec![CardType::Creature, CardType::Land],
                    },
                    zone: Zone::YourGraveyard,
                }],
            ),
            Program::new(
                "ward",
                ProgramKind::Triggered,
                vec![Op::CounterUnlessPays {
                    cost: "1",
                    target: "triggering_spell",
                }],
            )
            .trigger(Trigger::new(Event::BecomesTargeted, Subject::This)),
            Program::new(
                "lesson_death_draw",
                ProgramKind::Triggered,
                vec![Op::Branch {
                    condition: Condition::GraveyardAtLeast {
                        count: 1,
                        predicate: Predicate::Subtype { value: "Lesson" },
                    },
                    otherwise: vec![],
                    then: vec![Op::DrawCards { count: 1 }],
                }],
            )
            .trigger(Trigger {
                condition: Some(Condition::GraveyardAtLeast {
                    count: 1,
                    predicate: Predicate::Subtype { value: "Lesson" },
                }),
                ..Trigger::new(Event::Dies, Subject::This)
            }),
        ],
        registry_name: "Dragonfly Swarm",
        text_box,
    }
}

fn earth_kingdom_jailer() -> Card {
    let text_box = "When this creature enters, exile up to one target artifact, creature, or enchantment an opponent controls with mana value 3 or greater until this creature leaves the battlefield.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2W"),
            subtypes: Some(vec!["Human", "Soldier", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.earth_kingdom_jailer",
        programs: vec![Program::new(
            "temporary_exile",
            ProgramKind::Triggered,
            vec![Op::ExileUntilSourceLeaves { target: "prisoner" }],
        )
        .targets(vec![Target {
            another: false,
            max: 1,
            min: 0,
            role: "prisoner",
            selector: Selector::PermanentOpponentControls {
                min_mana_value: 3,
                types_any: vec![
                    CardType::Artifact,
                    CardType::Creature,
                    CardType::Enchantment,
                ],
            },
        }])
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Earth Kingdom Jailer",
        text_box,
    }
}

fn earth_kingdom_protectors() -> Card {
    let text_box = "Vigilance\nSacrifice this creature: Another target Ally you control gains indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{W}"),
            subtypes: Some(vec!["Human", "Soldier", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Vigilance]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.earth_kingdom_protectors",
        programs: vec![Program::new(
            "protect",
            ProgramKind::Activated,
            vec![Op::GrantKeywords {
                duration: Duration::EndOfTurn,
                keywords: vec![Keyword::Indestructible],
                target: "ally",
            }],
        )
        .cost(Cost {
            mana: Some("0"),
            sacrifice_source: Some(true),
            ..Default::default()
        })
        .targets(vec![Target::one(
            "ally",
            Selector::PermanentYouControl {
                predicate: Predicate::Subtype { value: "Ally" },
            },
        )])],
        registry_name: "Earth Kingdom Protectors",
        text_box,
    }
}

fn earth_kingdom_soldier() -> Card {
    let text_box = "Vigilance\nWhen this creature enters, put a +1/+1 counter on each of up to two target creatures you control.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{G/W}"),
            subtypes: Some(vec!["Human", "Soldier"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Vigilance]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(4),
            ..Default::default()
        },
        key: "tla.earth_kingdom_soldier",
        programs: vec![Program::new(
            "ability",
            ProgramKind::Triggered,
            vec![Op::TargetClause {
                body: vec![Op::ForEachTarget {
                    body: vec![Op::PutCounters {
                        count: 1,
                        target: "creatures",
                    }],
                    role: "creatures",
                }],
                role: "creatures",
            }],
        )
        .targets(vec![Target {
            another: false,
            max: 2,
            min: 0,
            role: "creatures",
            selector: Selector::CreatureYouControl,
        }])
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Earth Kingdom Soldier",
        text_box,
    }
}

fn earth_kings_lieutenant() -> Card {
    let text_box = "Trample\nWhen this creature enters, put a +1/+1 counter on each other Ally creature you control.\nWhenever another Ally you control enters, put a +1/+1 counter on this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("GW"),
            subtypes: Some(vec!["Human", "Soldier", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Trample]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.earth_kings_lieutenant",
        programs: vec![
            Program::new(
                "arrival_counters",
                ProgramKind::Triggered,
                vec![Op::PutCounters {
                    count: 1,
                    target: "each:other_ally_creature_you_control",
                }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "ally_growth",
                ProgramKind::Triggered,
                vec![Op::PutCounters {
                    count: 1,
                    target: "source",
                }],
            )
            .trigger(Trigger::new(
                Event::EntersBattlefield,
                Subject::AnotherYouControl {
                    predicate: Predicate::Subtype { value: "Ally" },
                },
            )),
        ],
        registry_name: "Earth King's Lieutenant",
        text_box,
    }
}

fn fancy_footwork() -> Card {
    let text_box = "Untap one or two target creatures. They each get +2/+2 until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2W"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.fancy_footwork",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::ForEachTarget {
                body: vec![
                    Op::Untap {
                        target: "current_target",
                    },
                    Op::ModifyPt {
                        duration: Duration::EndOfTurn,
                        power: 2,
                        target: "current_target",
                        toughness: 2,
                    },
                ],
                role: "creatures",
            }],
        )
        .cost(Cost::mana("2W"))
        .targets(vec![Target {
            another: false,
            max: 2,
            min: 1,
            role: "creatures",
            selector: Selector::Creature,
        }])],
        registry_name: "Fancy Footwork",
        text_box,
    }
}

fn fire_nation_cadets() -> Card {
    let text_box = "This creature has firebending 2 as long as there's a Lesson card in your graveyard. (Whenever this creature attacks, add {R}{R}. This mana lasts until end of combat.)\n{2}: This creature gets +1/+0 until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("R"),
            subtypes: Some(vec!["Human", "Soldier"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.fire_nation_cadets",
        programs: vec![
            Program::new(
                "conditional_firebending",
                ProgramKind::Triggered,
                vec![Op::AddMana {
                    mana: "RR",
                    until: Some(Until::EndOfCombat),
                }],
            )
            .trigger(Trigger {
                condition: Some(Condition::GraveyardAtLeast {
                    count: 1,
                    predicate: Predicate::Subtype { value: "Lesson" },
                }),
                ..Trigger::new(Event::Attacks, Subject::This)
            }),
            Program::new(
                "pump",
                ProgramKind::Activated,
                vec![Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 1,
                    target: "source",
                    toughness: 0,
                }],
            )
            .cost(Cost::mana("2")),
        ],
        registry_name: "Fire Nation Cadets",
        text_box,
    }
}

fn fire_sages() -> Card {
    let text_box = "Firebending 1 (Whenever this creature attacks, add {R}. This mana lasts until end of combat.)\n{1}{R}{R}: Put a +1/+1 counter on this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            subtypes: Some(vec!["Human", "Cleric"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.fire_sages",
        programs: vec![
            Program::firebending("firebend", "R"),
            Program::new(
                "grow",
                ProgramKind::Activated,
                vec![Op::PutCounters {
                    count: 1,
                    target: "source",
                }],
            )
            .cost(Cost::mana("1RR")),
        ],
        registry_name: "Fire Sages",
        text_box,
    }
}

fn firebending_lesson() -> Card {
    let text_box = "Kicker {4} (You may pay an additional {4} as you cast this spell.)\nFirebending Lesson deals 2 damage to target creature. If this spell was kicked, it deals 5 damage to that creature instead.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("R"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.firebending_lesson",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Branch {
                condition: Condition::Kicked,
                otherwise: vec![Op::DealDamage {
                    amount: 2,
                    target: "victim",
                }],
                then: vec![Op::DealDamage {
                    amount: 5,
                    target: "victim",
                }],
            }],
        )
        .cost(Cost {
            mana: Some("R"),
            kicker: Some("4"),
            ..Default::default()
        })
        .targets(vec![Target::one("victim", Selector::Creature)])],
        registry_name: "Firebending Lesson",
        text_box,
    }
}

fn first_time_flyer() -> Card {
    let text_box =
        "Flying\nThis creature gets +1/+1 as long as there's a Lesson card in your graveyard.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1U"),
            subtypes: Some(vec!["Human", "Pilot", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.first_time_flyer",
        programs: vec![Program::new(
            "lesson_boost",
            ProgramKind::Static,
            vec![Op::Branch {
                condition: Condition::GraveyardAtLeast {
                    count: 1,
                    predicate: Predicate::Subtype { value: "Lesson" },
                },
                otherwise: vec![],
                then: vec![Op::ModifyPt {
                    duration: Duration::Static,
                    power: 1,
                    target: "source",
                    toughness: 1,
                }],
            }],
        )],
        registry_name: "First-Time Flyer",
        text_box,
    }
}

fn foggy_bottom_swamp() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {B} or {G}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land(
        "tla.foggy_bottom_swamp",
        "Foggy Bottom Swamp",
        text_box,
        ["B", "G"],
    )
}

fn forecasting_fortune_teller() -> Card {
    let text_box = "When this creature enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1U"),
            subtypes: Some(vec!["Human", "Advisor", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.forecasting_fortune_teller",
        programs: vec![Program::new(
            "create_clue",
            ProgramKind::Triggered,
            vec![Op::CreateToken {
                count: 1,
                definition_ref: "token.clue",
                tapped_and_attacking: None,
            }],
        )
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Forecasting Fortune Teller",
        text_box,
    }
}

fn glider_kids() -> Card {
    let text_box = "Flying\nWhen this creature enters, scry 1. (Look at the top card of your library. You may put it on the bottom.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{W}"),
            subtypes: Some(vec!["Human", "Pilot", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.glider_kids",
        programs: vec![
            Program::new("scry", ProgramKind::Triggered, vec![Op::Scry { count: 1 }])
                .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
        ],
        registry_name: "Glider Kids",
        text_box,
    }
}

fn iguana_parrot() -> Card {
    let text_box = "Flying, vigilance\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{U}"),
            subtypes: Some(vec!["Lizard", "Bird", "Pirate"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying, Keyword::Vigilance]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.iguana_parrot",
        programs: vec![Program::prowess("prowess")],
        registry_name: "Iguana Parrot",
        text_box,
    }
}

fn invasion_reinforcements() -> Card {
    let text_box = "Flash\nWhen this creature enters, create a 1/1 white Ally creature token.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1W"),
            subtypes: Some(vec!["Human", "Warrior", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flash]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.invasion_reinforcements",
        programs: vec![Program::new(
            "create_ally",
            ProgramKind::Triggered,
            vec![Op::CreateToken {
                count: 1,
                definition_ref: "token.ally",
                tapped_and_attacking: None,
            }],
        )
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Invasion Reinforcements",
        text_box,
    }
}

fn itll_quench_ya() -> Card {
    let text_box = "Counter target spell unless its controller pays {2}.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1U"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.itll_quench_ya",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::CounterUnlessPays {
                cost: "2",
                target: "target",
            }],
        )
        .cost(Cost::mana("1U"))
        .targets(vec![Target::one("target", Selector::Spell)])],
        registry_name: "It'll Quench Ya!",
        text_box,
    }
}

fn jeong_jeongs_deserters() -> Card {
    let text_box = "When this creature enters, put a +1/+1 counter on target creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1W"),
            subtypes: Some(vec!["Human", "Rebel", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.jeong_jeongs_deserters",
        programs: vec![Program::new(
            "rally_counter",
            ProgramKind::Triggered,
            vec![Op::PutCounters {
                count: 1,
                target: "beneficiary",
            }],
        )
        .targets(vec![Target::one("beneficiary", Selector::Creature)])
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Jeong Jeong's Deserters",
        text_box,
    }
}

fn kyoshi_village() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {G} or {W}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land("tla.kyoshi_village", "Kyoshi Village", text_box, ["G", "W"])
}

fn kyoshi_warriors() -> Card {
    let text_box = "When this creature enters, create a 1/1 white Ally creature token.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("3W"),
            subtypes: Some(vec!["Human", "Warrior", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.kyoshi_warriors",
        programs: vec![Program::new(
            "create_ally",
            ProgramKind::Triggered,
            vec![Op::CreateToken {
                count: 1,
                definition_ref: "token.ally",
                tapped_and_attacking: None,
            }],
        )
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Kyoshi Warriors",
        text_box,
    }
}

fn meditation_pools() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {G} or {U}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land(
        "tla.meditation_pools",
        "Meditation Pools",
        text_box,
        ["G", "U"],
    )
}

fn misty_palms_oasis() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {W} or {B}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land(
        "tla.misty_palms_oasis",
        "Misty Palms Oasis",
        text_box,
        ["W", "B"],
    )
}

fn north_pole_gates() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {W} or {U}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land(
        "tla.north_pole_gates",
        "North Pole Gates",
        text_box,
        ["W", "U"],
    )
}

fn omashu_city() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {R} or {G}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land("tla.omashu_city", "Omashu City", text_box, ["R", "G"])
}

fn otter_penguin() -> Card {
    let text_box = "Whenever you draw your second card each turn, this creature gets +1/+2 until end of turn and can't be blocked this turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1U"),
            subtypes: Some(vec!["Otter", "Bird"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.otter_penguin",
        programs: vec![Program::new(
            "second_draw_boost",
            ProgramKind::Triggered,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 1,
                    target: "source",
                    toughness: 2,
                },
                Op::RestrictBlocking {
                    duration: Duration::EndOfTurn,
                    predicate: Predicate::Any,
                    target: "source",
                },
            ],
        )
        .trigger(Trigger {
            n: Some(2),
            ..Trigger::new(Event::DrawNthCard, Subject::This)
        })],
        registry_name: "Otter-Penguin",
        text_box,
    }
}

fn pillar_launch() -> Card {
    let text_box = "Target creature gets +2/+2 and gains reach until end of turn. Untap it.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{G}"),
            subtypes: Some(vec![]),
            types: vec![CardType::Instant],
            keywords: Some(vec![]),
            ..Default::default()
        },
        key: "tla.pillar_launch",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 2,
                    target: "creature",
                    toughness: 2,
                },
                Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![Keyword::Reach],
                    target: "creature",
                },
                Op::Untap { target: "creature" },
            ],
        )
        .cost(Cost::mana("{G}"))
        .targets(vec![Target::one("creature", Selector::Creature)])],
        registry_name: "Pillar Launch",
        text_box,
    }
}

fn rumble_arena() -> Card {
    let text_box = "Vigilance\nWhen this land enters, scry 1. (Look at the top card of your library. You may put it on the bottom.)\n{T}: Add {C}.\n{1}, {T}: Add one mana of any color.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec![]),
            types: vec![CardType::Land],
            keywords: Some(vec![Keyword::Vigilance]),
            ..Default::default()
        },
        key: "tla.rumble_arena",
        programs: vec![
            Program::new("scry", ProgramKind::Triggered, vec![Op::Scry { count: 1 }])
                .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "colorless",
                ProgramKind::Mana,
                vec![Op::AddMana {
                    mana: "C",
                    until: None,
                }],
            ),
            Program::new(
                "filter",
                ProgramKind::Activated,
                vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
            )
            .cost(Cost {
                mana: Some("1"),
                tap_source: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Rumble Arena",
        text_box,
    }
}

fn serpents_pass() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {U} or {B}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land("tla.serpents_pass", "Serpent's Pass", text_box, ["U", "B"])
}

fn south_pole_voyager() -> Card {
    let text_box = "Whenever this creature or another Ally you control enters, you gain 1 life. If this is the second time this ability has resolved this turn, draw a card.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1W"),
            subtypes: Some(vec!["Human", "Scout", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.south_pole_voyager",
        programs: vec![Program::new(
            "ally_arrival",
            ProgramKind::Triggered,
            vec![
                Op::GainLife { amount: 1 },
                Op::Branch {
                    condition: Condition::NthResolution { n: 2 },
                    otherwise: vec![],
                    then: vec![Op::DrawCards { count: 1 }],
                },
            ],
        )
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnyYouControl {
                predicate: Predicate::Subtype { value: "Ally" },
            },
        ))],
        registry_name: "South Pole Voyager",
        text_box,
    }
}

fn suki_kyoshi_warrior() -> Card {
    let text_box = "Suki's power is equal to the number of creatures you control.\nWhenever Suki attacks, create a 1/1 white Ally creature token that's tapped and attacking.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2GW"),
            subtypes: Some(vec!["Human", "Warrior", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::CharacteristicDefined),
            toughness: Some(4),
            supertypes: Some(vec!["legendary"]),
            ..Default::default()
        },
        key: "tla.suki_kyoshi_warrior",
        programs: vec![
            Program::new(
                "creature_power",
                ProgramKind::Static,
                vec![Op::SetPowerFromCount {
                    controller: Controller::You,
                    predicate: Predicate::CardType {
                        value: CardType::Creature,
                    },
                    zone: Zone::Battlefield,
                }],
            ),
            Program::new(
                "attack_ally",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.ally",
                    tapped_and_attacking: Some(true),
                }],
            )
            .trigger(Trigger::new(Event::Attacks, Subject::This)),
        ],
        registry_name: "Suki, Kyoshi Warrior",
        text_box,
    }
}

fn sun_blessed_peak() -> Card {
    let text_box = "This land enters tapped.\n{T}: Add {R} or {W}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    draw_land(
        "tla.sun_blessed_peak",
        "Sun-Blessed Peak",
        text_box,
        ["R", "W"],
    )
}

fn tiger_seal() -> Card {
    let text_box = "Vigilance\nAt the beginning of your upkeep, tap this creature.\nWhenever you draw your second card each turn, untap this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("U"),
            subtypes: Some(vec!["Cat", "Seal"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Vigilance]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.tiger_seal",
        programs: vec![
            Program::new(
                "upkeep_tap",
                ProgramKind::Triggered,
                vec![Op::Tap { target: "source" }],
            )
            .trigger(Trigger::new(Event::UpkeepBegins, Subject::This)),
            Program::new(
                "second_draw_untap",
                ProgramKind::Triggered,
                vec![Op::Untap { target: "source" }],
            )
            .trigger(Trigger {
                n: Some(2),
                ..Trigger::new(Event::DrawNthCard, Subject::This)
            }),
        ],
        registry_name: "Tiger-Seal",
        text_box,
    }
}

fn treetop_freedom_fighters() -> Card {
    let text_box = "Haste\nWhen this creature enters, create a 1/1 white Ally creature token.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{R}"),
            subtypes: Some(vec!["Human", "Rebel", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Haste]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.treetop_freedom_fighters",
        programs: vec![Program::new(
            "reinforcements",
            ProgramKind::Triggered,
            vec![Op::CreateToken {
                count: 1,
                definition_ref: "token.ally",
                tapped_and_attacking: None,
            }],
        )
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Treetop Freedom Fighters",
        text_box,
    }
}

fn walltop_sentries() -> Card {
    let text_box = "Reach, deathtouch\nWhen this creature dies, if there's a Lesson card in your graveyard, you gain 2 life.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{G}"),
            subtypes: Some(vec!["Human", "Soldier", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Reach, Keyword::Deathtouch]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.walltop_sentries",
        programs: vec![Program::new(
            "ability",
            ProgramKind::Triggered,
            vec![Op::Branch {
                condition: Condition::GraveyardAtLeast {
                    count: 1,
                    predicate: Predicate::Subtype { value: "Lesson" },
                },
                otherwise: vec![],
                then: vec![Op::GainLife { amount: 2 }],
            }],
        )
        .trigger(Trigger {
            condition: Some(Condition::GraveyardAtLeast {
                count: 1,
                predicate: Predicate::Subtype { value: "Lesson" },
            }),
            ..Trigger::new(Event::Dies, Subject::This)
        })],
        registry_name: "Walltop Sentries",
        text_box,
    }
}

fn wandering_musicians() -> Card {
    let text_box =
        "Whenever this creature attacks, creatures you control get +1/+0 until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{R/W}"),
            subtypes: Some(vec!["Human", "Bard", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(5),
            ..Default::default()
        },
        key: "tla.wandering_musicians",
        programs: vec![Program::new(
            "ability",
            ProgramKind::Triggered,
            vec![Op::ModifyControlled {
                power: 1,
                predicate: Predicate::CardType {
                    value: CardType::Creature,
                },
                toughness: 0,
            }],
        )
        .trigger(Trigger::new(Event::Attacks, Subject::This))],
        registry_name: "Wandering Musicians",
        text_box,
    }
}

fn water_tribe_captain() -> Card {
    let text_box = "{5}: Creatures you control get +1/+1 until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{W}"),
            subtypes: Some(vec!["Human", "Soldier", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.water_tribe_captain",
        programs: vec![Program::new(
            "rally",
            ProgramKind::Activated,
            vec![Op::ModifyControlled {
                power: 1,
                predicate: Predicate::CardType {
                    value: CardType::Creature,
                },
                toughness: 1,
            }],
        )
        .cost(Cost::mana("5"))],
        registry_name: "Water Tribe Captain",
        text_box,
    }
}

fn water_tribe_rallier() -> Card {
    let text_box = "Waterbend {5}: Look at the top four cards of your library. You may reveal a creature card with power 3 or less from among them and put it into your hand. Put the rest on the bottom of your library in a random order. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1W"),
            subtypes: Some(vec!["Human", "Soldier", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.water_tribe_rallier",
        programs: vec![Program::new(
            "waterbend",
            ProgramKind::Activated,
            vec![Op::LookAndSelect {
                destination: Destination::Hand,
                look: Count::Fixed(4),
                max_select: 1,
                min_select: 0,
                predicate: Predicate::All {
                    predicates: vec![
                        Predicate::CardType {
                            value: CardType::Creature,
                        },
                        Predicate::PowerAtMost { value: 3 },
                    ],
                },
                reveal: Some(true),
                remainder: Some(Remainder::LibraryBottom),
                remainder_order: Some(RemainderOrder::Random),
            }],
        )
        .cost(Cost {
            mana: Some("5"),
            waterbend: Some(true),
            ..Default::default()
        })],
        registry_name: "Water Tribe Rallier",
        text_box,
    }
}

fn white_lotus_reinforcements() -> Card {
    let text_box = "Vigilance\nOther Allies you control get +1/+1.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1GW"),
            subtypes: Some(vec!["Human", "Soldier", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Vigilance]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.white_lotus_reinforcements",
        programs: vec![Program::new(
            "ally_anthem",
            ProgramKind::Static,
            vec![Op::ModifyPt {
                duration: Duration::Static,
                power: 1,
                target: "each:other_allies_you_control",
                toughness: 1,
            }],
        )],
        registry_name: "White Lotus Reinforcements",
        text_box,
    }
}

fn yip_yip() -> Card {
    let text_box = "Target creature you control gets +2/+2 until end of turn. If that creature is an Ally, it also gains flying until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("W"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.yip_yip",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 2,
                    target: "ally",
                    toughness: 2,
                },
                Op::Branch {
                    condition: Condition::TargetMatches {
                        predicate: Predicate::Subtype { value: "Ally" },
                        role: "ally",
                    },
                    otherwise: vec![],
                    then: vec![Op::GrantKeywords {
                        duration: Duration::EndOfTurn,
                        keywords: vec![Keyword::Flying],
                        target: "ally",
                    }],
                },
            ],
        )
        .cost(Cost::mana("W"))
        .targets(vec![Target::one("ally", Selector::CreatureYouControl)])],
        registry_name: "Yip Yip!",
        text_box,
    }
}

fn yuyan_archers() -> Card {
    let text_box =
        "Reach\nWhen this creature enters, you may discard a card. If you do, draw a card.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            subtypes: Some(vec!["Human", "Archer"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Reach]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.yuyan_archers",
        programs: vec![Program::new(
            "rummage",
            ProgramKind::Triggered,
            vec![Op::OptionalDiscardDraw {
                count: 1,
                up_to: None,
            }],
        )
        .trigger(Trigger::new(Event::EntersBattlefield, Subject::This))],
        registry_name: "Yuyan Archers",
        text_box,
    }
}

fn ballistic_boulder() -> Card {
    let text_box = "Flying";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Construct"]),
            types: vec![CardType::Artifact, CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            token: Some(true),
            colors: Some(vec![]),
            ..Default::default()
        },
        key: "token.ballistic_boulder",
        programs: vec![],
        registry_name: "Ballistic Boulder",
        text_box,
    }
}

fn bear() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Bear"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(4),
            token: Some(true),
            colors: Some(vec![Color::G]),
            ..Default::default()
        },
        key: "token.bear",
        programs: vec![],
        registry_name: "Bear",
        text_box,
    }
}

fn dragon() -> Card {
    let text_box = "Flying, firebending 4 (Whenever this token attacks, add {R}{R}{R}{R} to your mana pool. This mana lasts until end of combat.)";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Dragon"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(4),
            token: Some(true),
            colors: Some(vec![Color::R]),
            ..Default::default()
        },
        key: "token.dragon",
        programs: vec![Program::firebending("ability", "RRRR")],
        registry_name: "Dragon",
        text_box,
    }
}

fn monk() -> Card {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Monk"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            token: Some(true),
            colors: Some(vec![Color::R]),
            ..Default::default()
        },
        key: "token.monk",
        programs: vec![Program::prowess("ability")],
        registry_name: "Monk",
        text_box,
    }
}

fn soldier() -> Card {
    let text_box = "Firebending 1 (Whenever this token attacks, add {R}. This mana lasts until end of combat.)";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Soldier"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            token: Some(true),
            colors: Some(vec![Color::R]),
            ..Default::default()
        },
        key: "token.soldier",
        programs: vec![Program::firebending("ability", "R")],
        registry_name: "Soldier",
        text_box,
    }
}

fn spirit() -> Card {
    let text_box = "This token can't block or be blocked by non-Spirit creatures.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Spirit"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            token: Some(true),
            colors: Some(vec![]),
            ..Default::default()
        },
        key: "token.spirit",
        programs: vec![
            Program::new(
                "block_restriction",
                ProgramKind::Static,
                vec![Op::BlockingRestriction {
                    direction: Direction::BlockedBy,
                    predicate: Predicate::NotSubtypes {
                        values: vec!["Spirit"],
                    },
                }],
            ),
            Program::new(
                "blocking_restriction",
                ProgramKind::Static,
                vec![Op::BlockingRestriction {
                    direction: Direction::Blocking,
                    predicate: Predicate::NotSubtypes {
                        values: vec!["Spirit"],
                    },
                }],
            ),
        ],
        registry_name: "Spirit",
        text_box,
    }
}

/// The ten tapped dual lands share both abilities; only identity and colors vary.
fn draw_land(
    key: &'static str,
    registry_name: &'static str,
    text_box: &'static str,
    colors: [&'static str; 2],
) -> Card {
    Card {
        characteristics: Characteristics {
            subtypes: Some(vec![]),
            types: vec![CardType::Land],
            keywords: Some(vec![]),
            enters_tapped: Some(true),
            ..Default::default()
        },
        key,
        programs: vec![
            Program::new(
                "draw",
                ProgramKind::Activated,
                vec![Op::DrawCards { count: 1 }],
            )
            .cost(Cost {
                mana: Some("4"),
                sacrifice_source: Some(true),
                tap_source: Some(true),
                ..Default::default()
            }),
            Program::new(
                "mana",
                ProgramKind::Activated,
                vec![Op::choose_mana(&colors)],
            )
            .cost(Cost {
                mana: Some("0"),
                tap_source: Some(true),
                ..Default::default()
            }),
        ],
        registry_name,
        text_box,
    }
}

fn earthbending_lesson() -> Card {
    let text_box = "Earthbend 4. (Target land you control becomes a 0/0 creature with haste that's still a land. Put four +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{G}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.earthbending_lesson",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Earthbend {
                count: Count::Fixed(4),
                target: "land",
            }],
        )
        .cost(Cost::mana("{3}{G}"))
        .targets(vec![Target::one("land", Selector::LandYouControl)])
        .inferred()],
        registry_name: "Earthbending Lesson",
        text_box,
    }
}

fn sandbenders_storm() -> Card {
    let text_box = "Choose one —\n• Destroy target creature with power 4 or greater.\n• Earthbend 3. (Target land you control becomes a 0/0 creature with haste that's still a land. Put three +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{W}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.sandbenders_storm",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Modal {
                modes: vec![
                    vec![Op::Destroy { target: "creature" }],
                    vec![Op::Earthbend {
                        count: Count::Fixed(3),
                        target: "land",
                    }],
                ],
            }],
        )
        .cost(Cost::mana("{3}{W}"))
        .targets(vec![
            Target::one(
                "creature",
                Selector::Permanent {
                    predicate: Predicate::All {
                        predicates: vec![
                            Predicate::CardType {
                                value: CardType::Creature,
                            },
                            Predicate::PowerAtLeast { value: 4 },
                        ],
                    },
                },
            ),
            Target::one("land", Selector::LandYouControl),
        ])
        .inferred()],
        registry_name: "Sandbenders' Storm",
        text_box,
    }
}

fn earth_village_ruffians() -> Card {
    let text_box = "When this creature dies, earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{B/G}"),
            subtypes: Some(vec!["Human", "Soldier", "Rogue"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.earth_village_ruffians",
        programs: vec![Program::new(
            "dies",
            ProgramKind::Triggered,
            vec![Op::Earthbend {
                count: Count::Fixed(2),
                target: "land",
            }],
        )
        .targets(vec![Target::one("land", Selector::LandYouControl)])
        .trigger(Trigger::new(Event::Dies, Subject::This))],
        registry_name: "Earth Village Ruffians",
        text_box,
    }
}

fn geyser_leaper() -> Card {
    let text_box = "Flying\nWaterbend {4}: Draw a card, then discard a card. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{U}"),
            subtypes: Some(vec!["Human", "Warrior", "Ally"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.geyser_leaper",
        programs: vec![Program::new(
            "loot",
            ProgramKind::Activated,
            vec![Op::DrawCards { count: 1 }, Op::Discard],
        )
        .cost(Cost {
            mana: Some("{4}"),
            waterbend: Some(true),
            ..Default::default()
        })],
        registry_name: "Geyser Leaper",
        text_box,
    }
}

fn foggy_swamp_vinebender() -> Card {
    let text_box = "This creature can't be blocked by creatures with power 2 or less.\nWaterbend {5}: Put a +1/+1 counter on this creature. Activate only during your turn. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{G}"),
            subtypes: Some(vec!["Human", "Plant", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.foggy_swamp_vinebender",
        programs: vec![
            Program::new(
                "evasion",
                ProgramKind::Static,
                vec![Op::BlockingRestriction {
                    direction: Direction::BlockedBy,
                    predicate: Predicate::PowerAtMost { value: 2 },
                }],
            ),
            Program::new(
                "grow",
                ProgramKind::Activated,
                vec![Op::PutCounters {
                    count: 1,
                    target: "source",
                }],
            )
            .cost(Cost {
                mana: Some("{5}"),
                waterbend: Some(true),
                own_turn_only: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Foggy Swamp Vinebender",
        text_box,
    }
}

fn fire_nation_attacks() -> Card {
    let text_box = "Create two 2/2 red Soldier creature tokens with firebending 1. (Whenever a creature with firebending 1 attacks, add {R}. This mana lasts until end of combat.)\nFlashback {8}{R} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{R}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.fire_nation_attacks",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::CreateToken {
                count: 2,
                definition_ref: "token.soldier",
                tapped_and_attacking: None,
            }],
        )
        .cost(Cost {
            mana: Some("{4}{R}"),
            flashback: Some("{8}{R}"),
            ..Default::default()
        })],
        registry_name: "Fire Nation Attacks",
        text_box,
    }
}

fn rockalanche() -> Card {
    let text_box = "Earthbend X, where X is the number of Forests you control. (Target land you control becomes a 0/0 creature with haste that's still a land. Put X +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)\nFlashback {5}{G} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{G}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.rockalanche",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Earthbend {
                count: Count::Of(Quantity::BattlefieldMatching {
                    predicate: Predicate::Subtype { value: "Forest" },
                }),
                target: "land",
            }],
        )
        .cost(Cost {
            mana: Some("{2}{G}"),
            flashback: Some("{5}{G}"),
            ..Default::default()
        })
        .targets(vec![Target::one("land", Selector::LandYouControl)])
        .inferred()],
        registry_name: "Rockalanche",
        text_box,
    }
}

fn canyon_crawler() -> Card {
    let text_box = "Deathtouch\nWhen this creature enters, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nSwampcycling {2} ({2}, Discard this card: Search your library for a Swamp card, reveal it, put it into your hand, then shuffle.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{B}{B}"),
            subtypes: Some(vec!["Spider", "Beast"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Deathtouch]),
            power: Some(PrintedPower::Fixed(6)),
            toughness: Some(6),
            ..Default::default()
        },
        key: "tla.canyon_crawler",
        programs: vec![
            Program::new(
                "food",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.food",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::landcycling("swampcycling", "{2}", "Swamp"),
        ],
        registry_name: "Canyon Crawler",
        text_box,
    }
}

fn mongoose_lizard() -> Card {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nWhen this creature enters, it deals 1 damage to any target.\nMountaincycling {2} ({2}, Discard this card: Search your library for a Mountain card, reveal it, put it into your hand, then shuffle.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{R}{R}"),
            subtypes: Some(vec!["Mongoose", "Lizard"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Menace]),
            power: Some(PrintedPower::Fixed(5)),
            toughness: Some(6),
            ..Default::default()
        },
        key: "tla.mongoose_lizard",
        programs: vec![
            Program::new(
                "ping",
                ProgramKind::Triggered,
                vec![Op::DealDamage {
                    amount: 1,
                    target: "target_0",
                }],
            )
            .targets(vec![Target::one("target_0", Selector::CreatureOrPlayer)])
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::landcycling("mountaincycling", "{2}", "Mountain"),
        ],
        registry_name: "Mongoose Lizard",
        text_box,
    }
}

fn saber_tooth_moose_lion() -> Card {
    let text_box = "Reach\nForestcycling {2} ({2}, Discard this card: Search your library for a Forest card, reveal it, put it into your hand, then shuffle.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{G}{G}"),
            subtypes: Some(vec!["Elk", "Cat"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Reach]),
            power: Some(PrintedPower::Fixed(7)),
            toughness: Some(7),
            ..Default::default()
        },
        key: "tla.saber_tooth_moose_lion",
        programs: vec![Program::landcycling("forestcycling", "{2}", "Forest")],
        registry_name: "Saber-Tooth Moose-Lion",
        text_box,
    }
}

fn giant_koi() -> Card {
    let text_box = "Waterbend {3}: This creature can't be blocked this turn. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)\nIslandcycling {2} ({2}, Discard this card: Search your library for an Island card, reveal it, put it into your hand, then shuffle.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{U}{U}"),
            subtypes: Some(vec!["Fish"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(5)),
            toughness: Some(7),
            ..Default::default()
        },
        key: "tla.giant_koi",
        programs: vec![
            Program::new(
                "unblockable",
                ProgramKind::Activated,
                vec![Op::RestrictBlocking {
                    duration: Duration::EndOfTurn,
                    predicate: Predicate::Any,
                    target: "source",
                }],
            )
            .cost(Cost {
                mana: Some("{3}"),
                waterbend: Some(true),
                ..Default::default()
            }),
            Program::landcycling("islandcycling", "{2}", "Island"),
        ],
        registry_name: "Giant Koi",
        text_box,
    }
}

fn rabaroo_troop() -> Card {
    let text_box = "Landfall — Whenever a land you control enters, this creature gains flying until end of turn and you gain 1 life.\nPlainscycling {2} ({2}, Discard this card: Search your library for a Plains card, reveal it, put it into your hand, then shuffle.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{W}{W}"),
            subtypes: Some(vec!["Rabbit", "Kangaroo"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(5),
            ..Default::default()
        },
        key: "tla.rabaroo_troop",
        programs: vec![
            Program::new(
                "landfall",
                ProgramKind::Triggered,
                vec![
                    Op::GrantKeywords {
                        duration: Duration::EndOfTurn,
                        keywords: vec![Keyword::Flying],
                        target: "source",
                    },
                    Op::GainLife { amount: 1 },
                ],
            )
            .trigger(Trigger::new(
                Event::EntersBattlefield,
                Subject::AnyYouControl {
                    predicate: Predicate::CardType {
                        value: CardType::Land,
                    },
                },
            )),
            Program::landcycling("plainscycling", "{2}", "Plains"),
        ],
        registry_name: "Rabaroo Troop",
        text_box,
    }
}

fn team_avatar() -> Card {
    let text_box = "Whenever a creature you control attacks alone, it gets +X/+X until end of turn, where X is the number of creatures you control.\n{2}{W}, Discard this card: It deals damage equal to the number of creatures you control to target creature.";
    let creatures_you_control = || Quantity::BattlefieldMatching {
        predicate: Predicate::CardType {
            value: CardType::Creature,
        },
    };

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{W}"),
            types: vec![CardType::Enchantment],
            ..Default::default()
        },
        key: "tla.team_avatar",
        programs: vec![
            Program::new(
                "attacks_alone",
                ProgramKind::Triggered,
                vec![Op::BuffTargetValue {
                    power: creatures_you_control(),
                    target: "triggering_permanent",
                    toughness: Count::Of(creatures_you_control()),
                }],
            )
            .trigger(Trigger::new(
                Event::AttacksAlone,
                Subject::AnyYouControl {
                    predicate: Predicate::CardType {
                        value: CardType::Creature,
                    },
                },
            )),
            Program::new(
                "discard_damage",
                ProgramKind::Activated,
                vec![Op::DealDamageValue {
                    amount: creatures_you_control(),
                    target: "target_0",
                }],
            )
            .cost(Cost {
                mana: Some("{2}{W}"),
                zone: Some(Zone::YourHand),
                discard_source: Some(true),
                ..Default::default()
            })
            .targets(vec![Target::one("target_0", Selector::Creature)]),
        ],
        registry_name: "Team Avatar",
        text_box,
    }
}

fn merchant_of_many_hats() -> Card {
    let text_box = "{2}{B}: Return this card from your graveyard to your hand.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            subtypes: Some(vec!["Human", "Peasant", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.merchant_of_many_hats",
        programs: vec![Program::new(
            "return",
            ProgramKind::Activated,
            vec![Op::ReturnToHand { target: "source" }],
        )
        .cost(Cost {
            mana: Some("{2}{B}"),
            zone: Some(Zone::YourGraveyard),
            ..Default::default()
        })],
        registry_name: "Merchant of Many Hats",
        text_box,
    }
}

fn hog_monkey() -> Card {
    let text_box = "At the beginning of combat on your turn, target creature you control with a +1/+1 counter on it gains menace until end of turn. (It can't be blocked except by two or more creatures.)\nExhaust — {5}: Put two +1/+1 counters on this creature. (Activate each exhaust ability only once.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{B}"),
            subtypes: Some(vec!["Boar", "Monkey"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.hog_monkey",
        programs: vec![
            Program::new(
                "combat_menace",
                ProgramKind::Triggered,
                vec![Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![Keyword::Menace],
                    target: "target_0",
                }],
            )
            .targets(vec![Target::one(
                "target_0",
                Selector::PermanentYouControl {
                    predicate: Predicate::All {
                        predicates: vec![
                            Predicate::CardType {
                                value: CardType::Creature,
                            },
                            Predicate::WithCounter,
                        ],
                    },
                },
            )])
            .trigger(Trigger::new(Event::CombatBegins, Subject::This)),
            Program::new(
                "exhaust",
                ProgramKind::Activated,
                vec![Op::PutCounters {
                    count: 2,
                    target: "source",
                }],
            )
            .cost(Cost {
                mana: Some("{5}"),
                exhaust: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Hog-Monkey",
        text_box,
    }
}

fn rough_rhino_cavalry() -> Card {
    let text_box = "Firebending 2 (Whenever this creature attacks, add {R}{R}. This mana lasts until end of combat.)\nExhaust — {8}: Put two +1/+1 counters on this creature. It gains trample until end of turn. (Activate each exhaust ability only once.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{R}"),
            subtypes: Some(vec!["Human", "Mercenary"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(5)),
            toughness: Some(5),
            ..Default::default()
        },
        key: "tla.rough_rhino_cavalry",
        programs: vec![
            Program::firebending("firebending", "RR"),
            Program::new(
                "exhaust",
                ProgramKind::Activated,
                vec![
                    Op::PutCounters {
                        count: 2,
                        target: "source",
                    },
                    Op::GrantKeywords {
                        duration: Duration::EndOfTurn,
                        keywords: vec![Keyword::Trample],
                        target: "source",
                    },
                ],
            )
            .cost(Cost {
                mana: Some("{8}"),
                exhaust: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Rough Rhino Cavalry",
        text_box,
    }
}

fn rebellious_captives() -> Card {
    let text_box = "Exhaust — {6}: Put two +1/+1 counters on this creature, then earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped. Activate each exhaust ability only once.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{G}"),
            subtypes: Some(vec!["Human", "Peasant", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.rebellious_captives",
        programs: vec![Program::new(
            "exhaust",
            ProgramKind::Activated,
            vec![
                Op::PutCounters {
                    count: 2,
                    target: "source",
                },
                Op::Earthbend {
                    count: Count::Fixed(2),
                    target: "land",
                },
            ],
        )
        .cost(Cost {
            mana: Some("{6}"),
            exhaust: Some(true),
            ..Default::default()
        })
        .targets(vec![Target::one("land", Selector::LandYouControl)])],
        registry_name: "Rebellious Captives",
        text_box,
    }
}

fn barrels_of_blasting_jelly() -> Card {
    let text_box = "{1}: Add one mana of any color. Activate only once each turn.\n{5}, {T}, Sacrifice this artifact: It deals 5 damage to target creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}"),
            types: vec![CardType::Artifact],
            ..Default::default()
        },
        key: "tla.barrels_of_blasting_jelly",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Activated,
                vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
            )
            .cost(Cost {
                mana: Some("{1}"),
                once_each_turn: Some(true),
                ..Default::default()
            }),
            Program::new(
                "p1",
                ProgramKind::Activated,
                vec![Op::DealDamage {
                    amount: 5,
                    target: "target_0",
                }],
            )
            .cost(Cost {
                mana: Some("{5}"),
                tap_source: Some(true),
                sacrifice_source: Some(true),
                ..Default::default()
            })
            .targets(vec![Target::one("target_0", Selector::Creature)]),
        ],
        registry_name: "Barrels of Blasting Jelly",
        text_box,
    }
}

fn jasmine_dragon_tea_shop() -> Card {
    let text_box = "{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast an Ally spell or activate an ability of an Ally source.\n{5}, {T}: Create a 1/1 white Ally creature token.";

    Card {
        characteristics: Characteristics {
            types: vec![CardType::Land],
            ..Default::default()
        },
        key: "tla.jasmine_dragon_tea_shop",
        programs: vec![
            Program::new(
                "colorless",
                ProgramKind::Mana,
                vec![Op::AddMana {
                    mana: "C",
                    until: None,
                }],
            ),
            Program::new(
                "p0",
                ProgramKind::Activated,
                vec![Op::choose_restricted_mana(SpendOnly {
                    spell_subtypes: vec!["Ally"],
                    ability_source_subtypes: vec!["Ally"],
                })],
            )
            .cost(Cost {
                mana: Some("0"),
                tap_source: Some(true),
                ..Default::default()
            }),
            Program::new(
                "p1",
                ProgramKind::Activated,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.ally",
                    tapped_and_attacking: Some(false),
                }],
            )
            .cost(Cost {
                mana: Some("{5}"),
                tap_source: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Jasmine Dragon Tea Shop",
        text_box,
    }
}

fn white_lotus_hideout() -> Card {
    let text_box = "{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast a Lesson or Shrine spell.\n{1}, {T}: Add one mana of any color.";

    Card {
        characteristics: Characteristics {
            types: vec![CardType::Land],
            ..Default::default()
        },
        key: "tla.white_lotus_hideout",
        programs: vec![
            Program::new(
                "colorless",
                ProgramKind::Mana,
                vec![Op::AddMana {
                    mana: "C",
                    until: None,
                }],
            ),
            Program::new(
                "p0",
                ProgramKind::Activated,
                vec![Op::choose_restricted_mana(SpendOnly {
                    spell_subtypes: vec!["Lesson", "Shrine"],
                    ..Default::default()
                })],
            )
            .cost(Cost {
                mana: Some("0"),
                tap_source: Some(true),
                ..Default::default()
            }),
            Program::new(
                "p1",
                ProgramKind::Activated,
                vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
            )
            .cost(Cost {
                mana: Some("{1}"),
                tap_source: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "White Lotus Hideout",
        text_box,
    }
}

fn waterbending_scroll() -> Card {
    let text_box = "{6}, {T}: Draw a card. This ability costs {1} less to activate for each Island you control.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{U}"),
            types: vec![CardType::Artifact],
            ..Default::default()
        },
        key: "tla.waterbending_scroll",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::DrawCards { count: 1 }],
        )
        .cost(Cost {
            mana: Some("{6}"),
            tap_source: Some(true),
            affinity: Some(Predicate::Subtype { value: "Island" }),
            ..Default::default()
        })],
        registry_name: "Waterbending Scroll",
        text_box,
    }
}

fn nonland_permanent() -> Selector {
    Selector::Permanent {
        predicate: Predicate::NotCardTypes {
            values: vec![CardType::Land],
        },
    }
}

fn aang_the_last_airbender() -> Card {
    let text_box = "Flying\nWhen Aang enters, airbend up to one other target nonland permanent. (Exile it. While it's exiled, its owner may cast it for {2} rather than its mana cost.)\nWhenever you cast a Lesson spell, Aang gains lifelink until end of turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{W}"),
            subtypes: Some(vec!["Human", "Avatar", "Ally"]),
            supertypes: Some(vec!["Legendary"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "tla.aang_the_last_airbender",
        programs: vec![
            Program::new(
                "enters",
                ProgramKind::Triggered,
                vec![Op::TargetClause {
                    role: "permanent",
                    body: vec![Op::Airbend {
                        target: "current_target",
                    }],
                }],
            )
            .targets(vec![Target {
                min: 0,
                another: true,
                ..Target::one("permanent", nonland_permanent())
            }])
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "lesson_cast",
                ProgramKind::Triggered,
                vec![Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![Keyword::Lifelink],
                    target: "source",
                }],
            )
            .trigger(Trigger {
                predicate: Some(Predicate::Subtype { value: "Lesson" }),
                ..Trigger::new(Event::CastSpell, Subject::This)
            }),
        ],
        registry_name: "Aang, the Last Airbender",
        text_box,
    }
}

fn airbending_lesson() -> Card {
    let text_box = "Airbend target nonland permanent. (Exile it. While it's exiled, its owner may cast it for {2} rather than its mana cost.)\nDraw a card.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{W}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.airbending_lesson",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::Airbend {
                    target: "permanent",
                },
                Op::DrawCards { count: 1 },
            ],
        )
        .cost(Cost::mana("{2}{W}"))
        .targets(vec![Target::one("permanent", nonland_permanent())])
        .inferred()],
        registry_name: "Airbending Lesson",
        text_box,
    }
}

fn airbenders_reversal() -> Card {
    let text_box = "Choose one —\n• Destroy target attacking creature.\n• Airbend target creature you control. (Exile it. While it's exiled, its owner may cast it for {2} rather than its mana cost.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{W}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.airbender_s_reversal",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Modal {
                modes: vec![
                    vec![Op::Destroy { target: "attacker" }],
                    vec![Op::Airbend { target: "own" }],
                ],
            }],
        )
        .cost(Cost::mana("{1}{W}"))
        .targets(vec![
            Target::one(
                "attacker",
                Selector::Permanent {
                    predicate: Predicate::All {
                        predicates: vec![
                            Predicate::CardType {
                                value: CardType::Creature,
                            },
                            Predicate::Attacking,
                        ],
                    },
                },
            ),
            Target::one("own", Selector::CreatureYouControl),
        ])
        .inferred()],
        registry_name: "Airbender's Reversal",
        text_box,
    }
}

fn avatars_wrath() -> Card {
    let text_box = "Choose up to one target creature, then airbend all other creatures. (Exile them. While each one is exiled, its owner may cast it for {2} rather than its mana cost.)\nUntil your next turn, your opponents can't cast spells from anywhere other than their hands.\nExile Avatar's Wrath.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{W}{W}"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.avatar_s_wrath",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::TargetClause {
                    role: "spared",
                    body: vec![Op::AirbendAll {
                        except: "current_target",
                        predicate: Predicate::CardType {
                            value: CardType::Creature,
                        },
                    }],
                },
                Op::OpponentsCastOnlyFromHand,
                Op::Exile { target: "source" },
            ],
        )
        .cost(Cost::mana("{2}{W}{W}"))
        .targets(vec![Target {
            min: 0,
            ..Target::one("spared", Selector::Creature)
        }])],
        registry_name: "Avatar's Wrath",
        text_box,
    }
}

fn zuko_exiled_prince() -> Card {
    let text_box = "Firebending 3 (Whenever this creature attacks, add {R}{R}{R}. This mana lasts until end of combat.)\n{3}: Exile the top card of your library. You may play that card this turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{R}"),
            subtypes: Some(vec!["Human", "Noble"]),
            supertypes: Some(vec!["Legendary"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.zuko_exiled_prince",
        programs: vec![
            Program::firebending("firebending", "RRR"),
            Program::new(
                "impulse",
                ProgramKind::Activated,
                vec![Op::ExileTopCardPlayThisTurn],
            )
            .cost(Cost::mana("{3}")),
        ],
        registry_name: "Zuko, Exiled Prince",
        text_box,
    }
}

fn aang_s_iceberg() -> Card {
    let text_box = "Flash\nWhen this enchantment enters, exile up to one other target nonland permanent until this enchantment leaves the battlefield.\nWaterbend {3}: Sacrifice this enchantment. If you do, scry 2. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{W}"),
            types: vec![CardType::Enchantment],
            keywords: Some(vec![Keyword::Flash]),
            ..Default::default()
        },
        key: "tla.aang_s_iceberg",
        programs: vec![
            Program::new(
                "freeze",
                ProgramKind::Triggered,
                vec![Op::TargetClause {
                    role: "frozen",
                    body: vec![Op::ExileUntilSourceLeaves {
                        target: "current_target",
                    }],
                }],
            )
            .targets(vec![Target {
                min: 0,
                another: true,
                ..Target::one("frozen", nonland_permanent())
            }])
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "thaw",
                ProgramKind::Activated,
                vec![Op::SacrificeSource {
                    then: vec![Op::Scry { count: 2 }],
                }],
            )
            .cost(Cost {
                mana: Some("{3}"),
                waterbend: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Aang's Iceberg",
        text_box,
    }
}

fn north_pole_patrol() -> Card {
    let text_box = "{T}: Untap another target permanent you control.\nWaterbend {3}, {T}: Tap target creature an opponent controls. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{U}"),
            subtypes: Some(vec!["Human", "Soldier", "Ally"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.north_pole_patrol",
        programs: vec![
            Program::new(
                "untap",
                ProgramKind::Activated,
                vec![Op::Untap {
                    target: "permanent",
                }],
            )
            .cost(Cost {
                mana: Some("0"),
                tap_source: Some(true),
                ..Default::default()
            })
            .targets(vec![Target {
                another: true,
                ..Target::one(
                    "permanent",
                    Selector::PermanentYouControl {
                        predicate: Predicate::Any,
                    },
                )
            }]),
            Program::new(
                "tap",
                ProgramKind::Activated,
                vec![Op::Tap { target: "creature" }],
            )
            .cost(Cost {
                mana: Some("{3}"),
                tap_source: Some(true),
                waterbend: Some(true),
                ..Default::default()
            })
            .targets(vec![Target::one(
                "creature",
                Selector::CreatureOpponentControls,
            )]),
        ],
        registry_name: "North Pole Patrol",
        text_box,
    }
}

fn zuko_s_conviction() -> Card {
    let text_box = "Kicker {4} (You may pay an additional {4} as you cast this spell.)\nReturn target creature card from your graveyard to your hand. If this spell was kicked, instead put that card onto the battlefield tapped.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{B}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.zuko_s_conviction",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Branch {
                condition: Condition::Kicked,
                otherwise: vec![Op::ReturnGraveyardTarget {
                    target: "creature",
                    destination: None,
                }],
                then: vec![Op::ReturnGraveyardTarget {
                    target: "creature",
                    destination: Some(Destination::BattlefieldTapped),
                }],
            }],
        )
        .cost(Cost {
            mana: Some("{B}"),
            kicker: Some("{4}"),
            ..Default::default()
        })
        .targets(vec![Target::one(
            "creature",
            Selector::GraveyardCardYouOwn {
                predicate: Predicate::CardType {
                    value: CardType::Creature,
                },
            },
        )])],
        registry_name: "Zuko's Conviction",
        text_box,
    }
}

fn sokka_bold_boomeranger() -> Card {
    let text_box = "When Sokka enters, discard up to two cards, then draw that many cards.\nWhenever you cast an artifact or Lesson spell, put a +1/+1 counter on Sokka.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{U}{R}"),
            subtypes: Some(vec!["Human", "Warrior", "Ally"]),
            supertypes: Some(vec!["Legendary"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "tla.sokka_bold_boomeranger",
        programs: vec![
            Program::new(
                "enters",
                ProgramKind::Triggered,
                vec![Op::OptionalDiscardDraw {
                    count: 1,
                    up_to: Some(2),
                }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "grows",
                ProgramKind::Triggered,
                vec![Op::PutCounters {
                    count: 1,
                    target: "source",
                }],
            )
            .trigger(Trigger {
                predicate: Some(Predicate::AnyOf {
                    predicates: vec![
                        Predicate::CardType {
                            value: CardType::Artifact,
                        },
                        Predicate::Subtype { value: "Lesson" },
                    ],
                }),
                ..Trigger::new(Event::CastSpell, Subject::This)
            }),
        ],
        registry_name: "Sokka, Bold Boomeranger",
        text_box,
    }
}

fn seismic_sense() -> Card {
    let text_box = "Look at the top X cards of your library, where X is the number of lands you control. You may reveal a creature or land card from among them and put it into your hand. Put the rest on the bottom of your library in a random order.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{G}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.seismic_sense",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::LookAndSelect {
                destination: Destination::Hand,
                look: Count::Of(Quantity::BattlefieldMatching {
                    predicate: Predicate::CardType {
                        value: CardType::Land,
                    },
                }),
                max_select: 1,
                min_select: 0,
                predicate: Predicate::TypesAny {
                    values: vec![CardType::Creature, CardType::Land],
                },
                reveal: Some(true),
                remainder: Some(Remainder::LibraryBottom),
                remainder_order: Some(RemainderOrder::Random),
            }],
        )
        .cost(Cost::mana("{G}"))],
        registry_name: "Seismic Sense",
        text_box,
    }
}

fn aang_s_journey() -> Card {
    let text_box = "Kicker {2} (You may pay an additional {2} as you cast this spell.)\nSearch your library for a basic land card. If this spell was kicked, instead search your library for a basic land card and a Shrine card. Reveal those cards, put them into your hand, then shuffle.\nYou gain 2 life.";
    let basic = || Op::SearchBasicLand {
        destination: Some(Destination::Hand),
        subtype: None,
    };

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.aang_s_journey",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::Branch {
                    condition: Condition::Kicked,
                    otherwise: vec![basic()],
                    then: vec![
                        basic(),
                        Op::SearchBasicLand {
                            destination: Some(Destination::Hand),
                            subtype: Some("Shrine"),
                        },
                    ],
                },
                Op::GainLife { amount: 2 },
            ],
        )
        .cost(Cost {
            mana: Some("{2}"),
            kicker: Some("{2}"),
            ..Default::default()
        })],
        registry_name: "Aang's Journey",
        text_box,
    }
}

fn sparring_dummy() -> Card {
    let text_box = "Defender\n{T}: Mill a card. You may put a land card milled this way into your hand. You gain 2 life if a Lesson card is milled this way. (To mill a card, put the top card of your library into your graveyard.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{G}"),
            subtypes: Some(vec!["Scarecrow"]),
            types: vec![CardType::Artifact, CardType::Creature],
            keywords: Some(vec![Keyword::Defender]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.sparring_dummy",
        programs: vec![Program::new(
            "train",
            ProgramKind::Activated,
            vec![Op::MillAndSelect {
                count: 1,
                if_declined: Vec::new(),
                predicate: Predicate::CardType {
                    value: CardType::Land,
                },
                milled_any: Some(Predicate::Subtype { value: "Lesson" }),
                then: vec![Op::GainLife { amount: 2 }],
            }],
        )
        .cost(Cost {
            mana: Some("0"),
            tap_source: Some(true),
            ..Default::default()
        })],
        registry_name: "Sparring Dummy",
        text_box,
    }
}

fn lost_days() -> Card {
    let text_box = "The owner of target creature or enchantment puts it into their library second from the top or on the bottom. You create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{U}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.lost_days",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::PutIntoLibrarySecondOrBottom { target: "lost" },
                Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: None,
                },
            ],
        )
        .cost(Cost::mana("{4}{U}"))
        .targets(vec![Target::one(
            "lost",
            Selector::Permanent {
                predicate: Predicate::TypesAny {
                    values: vec![CardType::Creature, CardType::Enchantment],
                },
            },
        )])],
        registry_name: "Lost Days",
        text_box,
    }
}

fn serpent_of_the_pass() -> Card {
    let text_box = "If there are three or more Lesson cards in your graveyard, you may cast this spell as though it had flash.\nThis spell costs {1} less to cast for each noncreature, nonland card in your graveyard.";
    let noncreature_nonland = || Predicate::NotCardTypes {
        values: vec![CardType::Creature, CardType::Land],
    };

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{5}{U}{U}"),
            subtypes: Some(vec!["Serpent"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(6)),
            toughness: Some(5),
            ..Default::default()
        },
        key: "tla.serpent_of_the_pass",
        programs: vec![
            Program::new(
                "flash",
                ProgramKind::Static,
                vec![Op::Branch {
                    condition: Condition::GraveyardAtLeast {
                        count: 3,
                        predicate: Predicate::Subtype { value: "Lesson" },
                    },
                    otherwise: Vec::new(),
                    then: vec![Op::GrantKeywords {
                        duration: Duration::Static,
                        keywords: vec![Keyword::Flash],
                        target: "source",
                    }],
                }],
            ),
            Program::new("spell", ProgramKind::Spell, Vec::new()).cost(Cost {
                mana: Some("{5}{U}{U}"),
                affinity: Some(noncreature_nonland()),
                affinity_zone: Some(Zone::YourGraveyard),
                ..Default::default()
            }),
        ],
        registry_name: "Serpent of the Pass",
        text_box,
    }
}

fn elemental_teachings() -> Card {
    let text_box = "Search your library for up to four land cards with different names and reveal them. An opponent chooses two of those cards. Put the chosen cards into your graveyard and the rest onto the battlefield tapped, then shuffle.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{4}{G}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.elemental_teachings",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::SearchLandsOpponentSplits {
                count: 4,
                opponent_picks: 2,
            }],
        )
        .cost(Cost::mana("{4}{G}"))],
        registry_name: "Elemental Teachings",
        text_box,
    }
}

fn dai_li_indoctrination() -> Card {
    let text_box = "Choose one —\n• Target opponent reveals their hand. You choose a nonland permanent card from it. That player discards that card.\n• Earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.dai_li_indoctrination",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Modal {
                modes: vec![
                    vec![Op::RevealHandDiscardChosen {
                        predicate: Predicate::NotCardTypes {
                            values: vec![CardType::Land, CardType::Instant, CardType::Sorcery],
                        },
                        target: "opponent",
                    }],
                    vec![Op::Earthbend {
                        count: Count::Fixed(2),
                        target: "land",
                    }],
                ],
            }],
        )
        .cost(Cost::mana("{1}{B}"))
        .targets(vec![
            Target::one("opponent", Selector::Opponent),
            Target::one("land", Selector::LandYouControl),
        ])
        .inferred()],
        registry_name: "Dai Li Indoctrination",
        text_box,
    }
}

fn professor_zei_anthropologist() -> Card {
    let text_box = "{T}, Discard a card: Draw a card.\n{1}, {T}, Sacrifice Professor Zei: Return target instant or sorcery card from your graveyard to your hand. Activate only during your turn.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{U/R}{U/R}"),
            subtypes: Some(vec!["Human", "Advisor", "Ally"]),
            supertypes: Some(vec!["Legendary"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(0)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.professor_zei_anthropologist",
        programs: vec![
            Program::new(
                "draw",
                ProgramKind::Activated,
                vec![Op::DrawCards { count: 1 }],
            )
            .cost(Cost {
                mana: Some("0"),
                tap_source: Some(true),
                discard_card: Some(true),
                ..Default::default()
            }),
            Program::new(
                "recall",
                ProgramKind::Activated,
                vec![Op::ReturnGraveyardTarget {
                    target: "spell",
                    destination: None,
                }],
            )
            .cost(Cost {
                mana: Some("{1}"),
                tap_source: Some(true),
                sacrifice_source: Some(true),
                own_turn_only: Some(true),
                ..Default::default()
            })
            .targets(vec![Target::one(
                "spell",
                Selector::GraveyardCardYouOwn {
                    predicate: Predicate::TypesAny {
                        values: vec![CardType::Instant, CardType::Sorcery],
                    },
                },
            )]),
        ],
        registry_name: "Professor Zei, Anthropologist",
        text_box,
    }
}

fn united_front() -> Card {
    let text_box = "Create X 1/1 white Ally creature tokens, then put a +1/+1 counter on each creature you control.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{X}{W}{W}"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.united_front",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::CreateTokensValue {
                    count: Quantity::X,
                    definition_ref: "token.ally",
                },
                Op::PutCounters {
                    count: 1,
                    target: "each:creature_you_control",
                },
            ],
        )
        .cost(Cost::mana("{X}{W}{W}"))
        .inferred()],
        registry_name: "United Front",
        text_box,
    }
}

fn spirit_water_revival() -> Card {
    let text_box = "As an additional cost to cast this spell, you may waterbend {6}. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)\nDraw two cards. If this spell's additional cost was paid, instead shuffle your graveyard into your library, draw seven cards, and you have no maximum hand size for the rest of the game.\nExile Spirit Water Revival.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{U}{U}"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "tla.spirit_water_revival",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::Branch {
                    condition: Condition::Kicked,
                    otherwise: vec![Op::DrawCards { count: 2 }],
                    then: vec![
                        Op::ShuffleGraveyardIntoLibrary,
                        Op::DrawCards { count: 7 },
                        Op::NoMaximumHandSize,
                    ],
                },
                Op::Exile { target: "source" },
            ],
        )
        .cost(Cost {
            mana: Some("{1}{U}{U}"),
            kicker: Some("{6}"),
            waterbend: Some(true),
            ..Default::default()
        })],
        registry_name: "Spirit Water Revival",
        text_box,
    }
}

fn yue_the_moon_spirit() -> Card {
    let text_box = "Flying, vigilance\nWaterbend {5}, {T}: You may cast a noncreature spell from your hand without paying its mana cost. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{3}{U}"),
            subtypes: Some(vec!["Spirit", "Ally"]),
            supertypes: Some(vec!["Legendary"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying, Keyword::Vigilance]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "tla.yue_the_moon_spirit",
        programs: vec![Program::new(
            "free_cast",
            ProgramKind::Activated,
            vec![Op::CastFromHandFree {
                predicate: Predicate::NotCardTypes {
                    values: vec![CardType::Creature, CardType::Land],
                },
            }],
        )
        .cost(Cost {
            mana: Some("{5}"),
            tap_source: Some(true),
            waterbend: Some(true),
            ..Default::default()
        })],
        registry_name: "Yue, the Moon Spirit",
        text_box,
    }
}

fn solstice_revelations() -> Card {
    let text_box = "Exile cards from the top of your library until you exile a nonland card. You may cast that card without paying its mana cost if the spell's mana value is less than the number of Mountains you control. If you don't cast that card this way, put it into your hand.\nFlashback {6}{R} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{R}"),
            subtypes: Some(vec!["Lesson"]),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "tla.solstice_revelations",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::ExileUntilNonlandCastFree {
                below: Quantity::BattlefieldMatching {
                    predicate: Predicate::Subtype { value: "Mountain" },
                },
            }],
        )
        .cost(Cost {
            mana: Some("{2}{R}"),
            flashback: Some("{6}{R}"),
            ..Default::default()
        })],
        registry_name: "Solstice Revelations",
        text_box,
    }
}
