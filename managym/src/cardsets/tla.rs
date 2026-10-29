//! Current TLA cards and their fixed tokens.
//! Edit these typed definitions; `semantic::compile` lowers them in process.
use crate::state::{
    ability::{
        Ability, Effect, EffectValue, LinkedChooser, LinkedGrant, MilledBonus, PaymentCost,
        StaticCondition, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, ActivationZone, CardDefinition, CardType, CardTypes, Keywords,
        ManaAbility, PowerCda, ReplacementEffect, SacrificeCost, SpellCostReduction, StaticPtBuff,
        StaticScope, TriggeredManaAbility,
    },
    mana::{Color, Colors, Mana, ManaCost, ManaUse},
    predicate::CardPredicate,
    zone::ZoneType,
};

/// Defined cards whose behavior is incomplete. The fixed setups that name them
/// still run them; the world does not support them for deck entry until their
/// batch completes the missing behavior.
pub(super) const INCOMPLETE: &[super::Incomplete] = &[
    ("tla.accumulate_wisdom", "Controller must order the library remainder (CR 401.4); it is randomized"),
    ("tla.cat_gator", "Any target must include planeswalkers and battles (CR 115.4)"),
    ("tla.dragonfly_swarm", "Unreviewed until T4: Flying; spell-and-ability ward; graveyard CDA power; conditional death draw."),
    ("tla.great_divide_guide", "Unreviewed until T3: Continuous any-color mana grant to lands and Allies; source removal and creature sickness."),
    ("ths.lightning_strike", "Any target must include planeswalkers and battles (CR 115.4)"),
    ("tla.ran_and_shaw", "Unreviewed until T5: Flying/firebending; cast-only conditional ETB nonlegendary self copy; Dragon anthem activation."),
    ("tla.suki_kyoshi_warrior", "CDA power must apply outside the battlefield (CR 604.3)"),
    ("tla.water_tribe_rallier", "Library power filter must evaluate CDAs, such as Suki's (CR 604.3)"),
];

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        aang_s_iceberg(),
        aang_s_journey(),
        aang_the_last_airbender(),
        abandon_attachments(),
        accumulate_wisdom(),
        airbender_s_reversal(),
        airbending_lesson(),
        airship_engine_room(),
        allies_at_last(),
        ally(),
        avatar_enthusiasts(),
        avatar_s_wrath(),
        azula_always_lies(),
        azula_cunning_usurper(),
        badgermole_cub(),
        ballistic_boulder(),
        barrels_of_blasting_jelly(),
        bear(),
        boar_q_pine(),
        boiling_rock_prison(),
        boomerang_basics(),
        callous_inspector(),
        canyon_crawler(),
        cat_gator(),
        cat_owl(),
        clue(),
        combustion_technique(),
        compassionate_healer(),
        cunning_maneuver(),
        curious_farm_animals(),
        cycle_of_renewal(),
        dai_li_agents(),
        dai_li_indoctrination(),
        deadly_precision(),
        deserter_s_disciple(),
        dragon(),
        dragonfly_swarm(),
        earth_kingdom_jailer(),
        earth_kingdom_protectors(),
        earth_kingdom_soldier(),
        earth_kings_lieutenant(),
        earth_village_ruffians(),
        earthbending_lesson(),
        elemental_teachings(),
        enter_the_avatar_state(),
        fancy_footwork(),
        fire_nation_attacks(),
        fire_nation_cadets(),
        fire_sages(),
        firebending_lesson(),
        first_time_flyer(),
        foggy_bottom_swamp(),
        foggy_swamp_spirit_keeper(),
        foggy_swamp_vinebender(),
        food(),
        forecasting_fortune_teller(),
        gather_the_white_lotus(),
        geyser_leaper(),
        giant_koi(),
        glider_kids(),
        gran_gran(),
        great_divide_guide(),
        hama_the_bloodbender(),
        haru_hidden_talent(),
        hermitic_herbalist(),
        hog_monkey(),
        iguana_parrot(),
        invasion_reinforcements(),
        iroh_grand_lotus(),
        iroh_s_demonstration(),
        itll_quench_ya(),
        jasmine_dragon_tea_shop(),
        jeong_jeongs_deserters(),
        jet_freedom_fighter(),
        june_bounty_hunter(),
        knowledge_seeker(),
        kyoshi_village(),
        kyoshi_warriors(),
        lost_days(),
        master_piandao(),
        meditation_pools(),
        merchant_of_many_hats(),
        messenger_hawk(),
        misty_palms_oasis(),
        momo_friendly_flier(),
        mongoose_lizard(),
        monk(),
        north_pole_gates(),
        north_pole_patrol(),
        octopus_form(),
        omashu_city(),
        origin_of_metalbending(),
        ostrich_horse(),
        otter_penguin(),
        ozai_s_cruelty(),
        pillar_launch(),
        pretending_poxbearers(),
        professor_zei_anthropologist(),
        rabaroo_troop(),
        ran_and_shaw(),
        rebellious_captives(),
        rockalanche(),
        rough_rhino_cavalry(),
        rumble_arena(),
        saber_tooth_moose_lion(),
        sandbenders_storm(),
        seismic_sense(),
        serpent_of_the_pass(),
        serpents_pass(),
        shared_roots(),
        sokka_bold_boomeranger(),
        sokka_s_haiku(),
        soldier(),
        solstice_revelations(),
        south_pole_voyager(),
        sozins_comet(),
        sparring_dummy(),
        spirit(),
        spirit_water_revival(),
        suki_kyoshi_warrior(),
        sun_blessed_peak(),
        team_avatar(),
        tiger_dillo(),
        tiger_seal(),
        tolls_of_war(),
        toph_the_blind_bandit(),
        treetop_freedom_fighters(),
        true_ancestry(),
        uncle_iroh(),
        united_front(),
        unlucky_cabbage_merchant(),
        walltop_sentries(),
        wandering_musicians(),
        wartime_protestors(),
        water_tribe_captain(),
        water_tribe_rallier(),
        waterbending_lesson(),
        waterbending_scroll(),
        white_lotus_hideout(),
        white_lotus_reinforcements(),
        yip_yip(),
        yue_the_moon_spirit(),
        yuyan_archers(),
        zuko_exiled_prince(),
        zuko_s_conviction(),
        zuko_s_exile(),
    ]
}

fn aang_s_iceberg() -> CardDefinition {
    let text_box = "Flash\nWhen this enchantment enters, exile up to one other target nonland permanent until this enchantment leaves the battlefield.\nWaterbend {3}: Sacrifice this enchantment. If you do, scry 2. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.aang_s_iceberg",
        scryfall_id: "720fbd87-b1c1-4b3b-97a1-46b943b115e3",
        name: "Aang's Iceberg".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Enchantment]),
        keywords: Keywords {
            flash: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    spec: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    },
                    min: 0,
                    max: 1,
                    another: true,
                },
                effects: vec![Effect::ExileUntilSourceLeaves {
                    target: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    },
                }],
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}"),
            waterbend: true,
            effects: vec![Effect::SacrificeSource {
                then: vec![Effect::Scry { count: 2 }],
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn aang_s_journey() -> CardDefinition {
    let text_box = "Kicker {2} (You may pay an additional {2} as you cast this spell.)\nSearch your library for a basic land card. If this spell was kicked, instead search your library for a basic land card and a Shrine card. Reveal those cards, put them into your hand, then shuffle.\nYou gain 2 life.";

    CardDefinition {
        key: "tla.aang_s_journey",
        scryfall_id: "5e51f727-5a9b-4bc7-83a9-dbcf1c933e15",
        name: "Aang's Journey".into(),
        mana_cost: Some(ManaCost::parse("{2}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::IfKicked {
                then: vec![
                    Effect::SearchBasicLand {
                        subtype: None,
                        count: 1,
                        tapped: true,
                        untap_if_lands: None,
                        bottom_source: false,
                        to_hand: true,
                    },
                    Effect::SearchBasicLand {
                        subtype: Some("Shrine".into()),
                        count: 1,
                        tapped: true,
                        untap_if_lands: None,
                        bottom_source: false,
                        to_hand: true,
                    },
                ],
                otherwise: vec![Effect::SearchBasicLand {
                    subtype: None,
                    count: 1,
                    tapped: true,
                    untap_if_lands: None,
                    bottom_source: false,
                    to_hand: true,
                }],
            },
            Effect::GainLife { amount: 2 },
        ],
        kicker: Some(ManaCost::parse("{2}")),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn aang_the_last_airbender() -> CardDefinition {
    let text_box = "Flying\nWhen Aang enters, airbend up to one other target nonland permanent. (Exile it. While it's exiled, its owner may cast it for {2} rather than its mana cost.)\nWhenever you cast a Lesson spell, Aang gains lifelink until end of turn.";

    CardDefinition {
        key: "tla.aang_the_last_airbender",
        scryfall_id: "245e008c-e073-443f-9592-6f628c0026ec",
        name: "Aang, the Last Airbender".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Avatar".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(2),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        spec: TargetSpec::Permanent {
                            predicate: CardPredicate {
                                not_card_types: vec![CardType::Land],
                                ..Default::default()
                            },
                        },
                        min: 0,
                        max: 1,
                        another: true,
                    },
                    effects: vec![Effect::Airbend {
                        target: TargetSpec::Permanent {
                            predicate: CardPredicate {
                                not_card_types: vec![CardType::Land],
                                ..Default::default()
                            },
                        },
                    }],
                }],
            },
            Ability::Triggered {
                condition: TriggerCondition::YouCastSpell {
                    predicate: CardPredicate {
                        subtype: Some("Lesson".into()),
                        ..Default::default()
                    },
                },
                effects: vec![Effect::GrantKeywordsToSource {
                    keywords: Keywords {
                        lifelink: true,
                        ..Default::default()
                    },
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn abandon_attachments() -> CardDefinition {
    let text_box = "You may discard a card. If you do, draw two cards.";

    CardDefinition {
        key: "tla.abandon_attachments",
        scryfall_id: "74ca45a4-97ab-4255-9129-884e8b42b984",
        name: "Abandon Attachments".into(),
        mana_cost: Some(ManaCost::parse("{1}{U/R}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::OptionalDiscardDraw { count: 2, up_to: 1 }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn accumulate_wisdom() -> CardDefinition {
    let text_box = "Look at the top three cards of your library. Put one of those cards into your hand and the rest on the bottom of your library in any order. Put each of those cards into your hand instead if there are three or more Lesson cards in your graveyard.";

    CardDefinition {
        key: "tla.accumulate_wisdom",
        scryfall_id: "a6335319-6c92-40d4-ab2d-c06c79049c30",
        name: "Accumulate Wisdom".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::IfGraveyardAtLeast {
            count: 3,
            predicate: CardPredicate {
                subtype: Some("Lesson".into()),
                ..Default::default()
            },
            then: vec![Effect::PutTopCardsInHand { count: 3 }],
            otherwise: vec![Effect::LookAndSelect {
                reveal: false,
                look: EffectValue::Fixed(3),
                min_select: 1,
                max_select: 1,
                predicate: CardPredicate::default(),
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn airbender_s_reversal() -> CardDefinition {
    let text_box = "Choose one —\n• Destroy target attacking creature.\n• Airbend target creature you control. (Exile it. While it's exiled, its owner may cast it for {2} rather than its mana cost.)";

    CardDefinition {
        key: "tla.airbender_s_reversal",
        scryfall_id: "6b7078e4-2892-4b5f-83ab-90369e0d6dba",
        name: "Airbender's Reversal".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::Destroy {
                    target: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            attacking: true,
                            card_type: Some(CardType::Creature),
                            ..Default::default()
                        },
                    },
                }],
                vec![Effect::Airbend {
                    target: TargetSpec::CreatureYouControl,
                }],
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn airbending_lesson() -> CardDefinition {
    let text_box = "Airbend target nonland permanent. (Exile it. While it's exiled, its owner may cast it for {2} rather than its mana cost.)\nDraw a card.";

    CardDefinition {
        key: "tla.airbending_lesson",
        scryfall_id: "e2bcd0a6-e94d-4a21-a334-a57459c1b8cc",
        name: "Airbending Lesson".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::Airbend {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Land],
                        ..Default::default()
                    },
                },
            },
            Effect::DrawCards { count: 1 },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn airship_engine_room() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {U} or {R}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.airship_engine_room",
        scryfall_id: "0b98ba34-f251-4bd0-9d75-1d4445be5cfd",
        name: "Airship Engine Room".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("U"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("R"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn allies_at_last() -> CardDefinition {
    let text_box = "Affinity for Allies (This spell costs {1} less to cast for each Ally you control.)\nUp to two target creatures you control each deal damage equal to their power to target creature an opponent controls.";

    CardDefinition {
        key: "tla.allies_at_last",
        scryfall_id: "11a77897-2aba-4a9b-bbe6-1768ca9f12cb",
        name: "Allies at Last".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::TargetCreaturesDealPowerDamageToLastTarget],
        targeting: vec![
            TargetRequirement {
                spec: TargetSpec::CreatureYouControl,
                min: 0,
                max: 2,
                another: false,
            },
            TargetRequirement {
                spec: TargetSpec::CreatureOpponentControls,
                min: 1,
                max: 1,
                another: false,
            },
        ],
        cost_reduction_per: Some(CardPredicate {
            subtype: Some("Ally".into()),
            ..Default::default()
        }),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn ally() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "token.ally",
        scryfall_id: "f842c52a-04b6-4625-ba58-45b043ada340",
        name: "Ally".into(),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Ally".into()],
        power: Some(1),
        toughness: Some(1),
        is_token: true,
        color_override: Some(Colors::from([Color::White])),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn avatar_enthusiasts() -> CardDefinition {
    let text_box =
        "Whenever another Ally you control enters, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "tla.avatar_enthusiasts",
        scryfall_id: "7292e5f2-d5a1-4e39-8f75-eba471819f04",
        name: "Avatar Enthusiasts".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Peasant".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnotherYouControl(CardPredicate {
                    subtype: Some("Ally".into()),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::PutCountersOnSource { count: 1 }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn avatar_s_wrath() -> CardDefinition {
    let text_box = "Choose up to one target creature, then airbend all other creatures. (Exile them. While each one is exiled, its owner may cast it for {2} rather than its mana cost.)\nUntil your next turn, your opponents can't cast spells from anywhere other than their hands.\nExile Avatar's Wrath.";

    CardDefinition {
        key: "tla.avatar_s_wrath",
        scryfall_id: "4811072d-fac0-40dd-a5cf-9694d51b12cf",
        name: "Avatar's Wrath".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::TargetClause {
                requirement: TargetRequirement {
                    spec: TargetSpec::Creature,
                    min: 0,
                    max: 1,
                    another: false,
                },
                effects: vec![Effect::AirbendAll {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                }],
            },
            Effect::OpponentsCastOnlyFromHand,
            Effect::ExileSourceSpell,
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::Creature,
            min: 0,
            max: 1,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn azula_always_lies() -> CardDefinition {
    let text_box = "Choose one or both —\n• Target creature gets -1/-1 until end of turn.\n• Put a +1/+1 counter on target creature.";

    CardDefinition {
        key: "tla.azula_always_lies",
        scryfall_id: "416b9207-a2af-44d1-9b87-4943f6e46d42",
        name: "Azula Always Lies".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        spec: TargetSpec::Creature,
                        min: 1,
                        max: 1,
                        another: false,
                    },
                    effects: vec![Effect::BuffTarget {
                        power: -1,
                        toughness: -1,
                        target: TargetSpec::Creature,
                    }],
                }],
                vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        spec: TargetSpec::Creature,
                        min: 1,
                        max: 1,
                        another: false,
                    },
                    effects: vec![Effect::PutCounters {
                        count: 1,
                        target: TargetSpec::Creature,
                    }],
                }],
                vec![
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            spec: TargetSpec::Creature,
                            min: 1,
                            max: 1,
                            another: false,
                        },
                        effects: vec![Effect::BuffTarget {
                            power: -1,
                            toughness: -1,
                            target: TargetSpec::Creature,
                        }],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            spec: TargetSpec::Creature,
                            min: 1,
                            max: 1,
                            another: false,
                        },
                        effects: vec![Effect::PutCounters {
                            count: 1,
                            target: TargetSpec::Creature,
                        }],
                    },
                ],
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn azula_cunning_usurper() -> CardDefinition {
    let text_box = "Firebending 2 (Whenever this creature attacks, add {R}{R}. This mana lasts until end of combat.)\nWhen Azula enters, target opponent exiles a nontoken creature they control, then they exile a nonland card from their graveyard.\nDuring your turn, you may cast cards exiled with Azula and you may cast them as though they had flash. Mana of any type can be spent to cast those spells.";

    CardDefinition {
        key: "tla.azula_cunning_usurper",
        scryfall_id: "daf30e1c-436d-4f23-b1d2-570619a4b7f5",
        name: "Azula, Cunning Usurper".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}{B}{B}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into(), "Rogue".into()],
        power: Some(4),
        toughness: Some(4),
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RR"),
                    until_end_of_combat: true,
                }],
            },
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![
                    Effect::ExileLinked {
                        chooser: LinkedChooser::TargetPlayer,
                        zone: ZoneType::Battlefield,
                        predicate: CardPredicate {
                            card_type: Some(CardType::Creature),
                            ..Default::default()
                        },
                        up_to: false,
                        grant: LinkedGrant::FlashAnyMana,
                        target: TargetSpec::Opponent,
                    },
                    Effect::ExileLinked {
                        chooser: LinkedChooser::TargetPlayer,
                        zone: ZoneType::Graveyard,
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                        up_to: false,
                        grant: LinkedGrant::FlashAnyMana,
                        target: TargetSpec::Opponent,
                    },
                ],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn badgermole_cub() -> CardDefinition {
    let text_box = "When this creature enters, earthbend 1. (Target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)\nWhenever you tap a creature for mana, add an additional {G}.";

    CardDefinition {
        key: "tla.badgermole_cub",
        scryfall_id: "340c5799-4964-44dd-8c48-8f3f3aba5211",
        name: "Badgermole Cub".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Badger".into(), "Mole".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(1),
                target: TargetSpec::LandYouControl,
            }],
        }],
        triggered_mana_abilities: vec![TriggeredManaAbility {
            predicate: CardPredicate {
                card_type: Some(CardType::Creature),
                ..Default::default()
            },
            mana: Mana::parse("G"),
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn ballistic_boulder() -> CardDefinition {
    let text_box = "Flying";

    CardDefinition {
        key: "token.ballistic_boulder",
        scryfall_id: "0b1b8ff0-0b1e-4a55-a937-fde7ec15c41a",
        name: "Ballistic Boulder".into(),
        types: CardTypes::new([CardType::Creature, CardType::Artifact]),
        subtypes: vec!["Construct".into()],
        power: Some(2),
        toughness: Some(1),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        is_token: true,
        color_override: Some(Colors::from([])),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn barrels_of_blasting_jelly() -> CardDefinition {
    let text_box = "{1}: Add one mana of any color. Activate only once each turn.\n{5}, {T}, Sacrifice this artifact: It deals 5 damage to target creature.";

    CardDefinition {
        key: "tla.barrels_of_blasting_jelly",
        scryfall_id: "4fd83083-3779-41f0-90a0-a00e3270e4e6",
        name: "Barrels of Blasting Jelly".into(),
        mana_cost: Some(ManaCost::parse("{1}")),
        types: CardTypes::new([CardType::Artifact]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                once_each_turn: true,
                mana_cost: ManaCost::parse("{1}"),
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("W"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("U"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("B"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("R"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("G"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{5}"),
                sacrifice_source: true,
                effects: vec![Effect::DealDamage {
                    amount: 5,
                    target: TargetSpec::Creature,
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn bear() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "token.bear",
        scryfall_id: "1dbbdb0f-57a4-4b85-ba5c-1383f0efb3af",
        name: "Bear".into(),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Bear".into()],
        power: Some(4),
        toughness: Some(4),
        is_token: true,
        color_override: Some(Colors::from([Color::Green])),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn boar_q_pine() -> CardDefinition {
    let text_box = "Whenever you cast a noncreature spell, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "tla.boar_q_pine",
        scryfall_id: "087a445c-5f63-45d6-83e8-20d22650db20",
        name: "Boar-q-pine".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Boar".into(), "Porcupine".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::YouCastSpell {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            effects: vec![Effect::PutCountersOnSource { count: 1 }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn boiling_rock_prison() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {B} or {R}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.boiling_rock_prison",
        scryfall_id: "1c2e2220-54d1-4180-93a0-964e3b0ba8b8",
        name: "Boiling Rock Prison".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("B"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("R"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn boomerang_basics() -> CardDefinition {
    let text_box = "Return target nonland permanent to its owner's hand. If you controlled that permanent, draw a card.";

    CardDefinition {
        key: "tla.boomerang_basics",
        scryfall_id: "17ab958a-abc6-472e-ad6a-97c731d89c74",
        name: "Boomerang Basics".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::ReturnToHandDrawIfControlled {
            target: TargetSpec::Permanent {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Land],
                    ..Default::default()
                },
            },
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn callous_inspector() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nWhen this creature dies, it deals 1 damage to you. Create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.callous_inspector",
        scryfall_id: "f22610e9-6b14-4914-bb67-bd6723bec9aa",
        name: "Callous Inspector".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn canyon_crawler() -> CardDefinition {
    let text_box = "Deathtouch\nWhen this creature enters, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nSwampcycling {2} ({2}, Discard this card: Search your library for a Swamp card, reveal it, put it into your hand, then shuffle.)";

    CardDefinition {
        key: "tla.canyon_crawler",
        scryfall_id: "30a30bfb-f0f3-425b-b37d-20079ee27046",
        name: "Canyon Crawler".into(),
        mana_cost: Some(ManaCost::parse("{4}{B}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Spider".into(), "Beast".into()],
        power: Some(6),
        toughness: Some(6),
        keywords: Keywords {
            deathtouch: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Food".into(),
                count: 1,
                tapped_and_attacking: false,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            zone: ActivationZone::Hand,
            discard_source: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::SearchBasicLand {
                subtype: Some("Swamp".into()),
                count: 1,
                tapped: true,
                untap_if_lands: None,
                bottom_source: false,
                to_hand: true,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn cat_gator() -> CardDefinition {
    let text_box = "Lifelink\nWhen this creature enters, it deals damage equal to the number of Swamps you control to any target.";

    CardDefinition {
        key: "tla.cat_gator",
        scryfall_id: "ec83a825-79e6-42a4-919c-3bc498ff4433",
        name: "Cat-Gator".into(),
        mana_cost: Some(ManaCost::parse("{6}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Fish".into(), "Crocodile".into()],
        power: Some(3),
        toughness: Some(2),
        keywords: Keywords {
            lifelink: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DealDamageValue {
                amount: EffectValue::BattlefieldMatching {
                    predicate: CardPredicate {
                        subtype: Some("Swamp".into()),
                        ..Default::default()
                    },
                },
                target: TargetSpec::CreatureOrPlayer,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn cat_owl() -> CardDefinition {
    let text_box = "Flying\nWhenever this creature attacks, untap target artifact or creature.";

    CardDefinition {
        key: "tla.cat_owl",
        scryfall_id: "d7df3391-dbe5-4eb4-85c1-037c5f3ea971",
        name: "Cat-Owl".into(),
        mana_cost: Some(ManaCost::parse("{3}{W/U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Cat".into(), "Bird".into()],
        power: Some(3),
        toughness: Some(3),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::UntapTarget {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Artifact, CardType::Creature],
                        ..Default::default()
                    },
                },
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn clue() -> CardDefinition {
    let text_box = "{2}, Sacrifice this token: Draw a card.";

    CardDefinition {
        key: "token.clue",
        scryfall_id: "82016bc3-8905-454a-b4aa-b9eac69d42c4",
        name: "Clue".into(),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Clue".into()],
        is_token: true,
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{2}"),
            sacrifice_source: true,
            effects: vec![Effect::DrawCards { count: 1 }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn combustion_technique() -> CardDefinition {
    let text_box = "Combustion Technique deals damage equal to 2 plus the number of Lesson cards in your graveyard to target creature. If that creature would die this turn, exile it instead.";

    CardDefinition {
        key: "tla.combustion_technique",
        scryfall_id: "fdcca576-2ef2-44cc-9944-a92bd146444a",
        name: "Combustion Technique".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::DealDamageValue {
                amount: EffectValue::GraveyardMatching {
                    base: 2,
                    predicate: CardPredicate {
                        subtype: Some("Lesson".into()),
                        ..Default::default()
                    },
                },
                target: TargetSpec::Creature,
            },
            Effect::ExileIfDiesThisTurn {
                target: TargetSpec::Creature,
            },
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn compassionate_healer() -> CardDefinition {
    let text_box = "Whenever this creature becomes tapped, you gain 1 life and scry 1. (Look at the top card of your library. You may put it on the bottom.)";

    CardDefinition {
        key: "tla.compassionate_healer",
        scryfall_id: "88d5f8fd-d4de-4e64-9b74-e53719ffbcdc",
        name: "Compassionate Healer".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Cleric".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::BecomesTapped {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }, Effect::Scry { count: 1 }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn cunning_maneuver() -> CardDefinition {
    let text_box = "Target creature gets +3/+1 until end of turn.\nCreate a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.cunning_maneuver",
        scryfall_id: "0ff7c993-ba29-43b0-9639-7c4bc0292fa2",
        name: "Cunning Maneuver".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::BuffTarget {
                power: 3,
                toughness: 1,
                target: TargetSpec::Creature,
            },
            Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
            },
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn curious_farm_animals() -> CardDefinition {
    let text_box = "When this creature dies, you gain 3 life.\n{2}, Sacrifice this creature: Destroy up to one target artifact or enchantment.";

    CardDefinition {
        key: "tla.curious_farm_animals",
        scryfall_id: "2402d759-84b6-41d2-ad78-9333974e9222",
        name: "Curious Farm Animals".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Boar".into(), "Elk".into(), "Bird".into(), "Ox".into()],
        power: Some(1),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 3 }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{2}"),
            sacrifice_source: true,
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    spec: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                            ..Default::default()
                        },
                    },
                    min: 0,
                    max: 1,
                    another: false,
                },
                effects: vec![Effect::Destroy {
                    target: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                            ..Default::default()
                        },
                    },
                }],
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn cycle_of_renewal() -> CardDefinition {
    let text_box = "Sacrifice a land. Search your library for up to two basic land cards, put them onto the battlefield tapped, then shuffle.";

    CardDefinition {
        key: "tla.cycle_of_renewal",
        scryfall_id: "233f4a05-057b-48f4-8f56-8d4a060a40e4",
        name: "Cycle of Renewal".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
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
                to_hand: false,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn dai_li_agents() -> CardDefinition {
    let text_box = "When this creature enters, earthbend 1, then earthbend 1. (To earthbend 1, target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)\nWhenever this creature attacks, each opponent loses X life and you gain X life, where X is the number of creatures you control with +1/+1 counters on them.";

    CardDefinition {
        key: "tla.dai_li_agents",
        scryfall_id: "9ddb8715-341b-400a-b6eb-f3f90518157c",
        name: "Dai Li Agents".into(),
        mana_cost: Some(ManaCost::parse("{3}{B}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(3),
        toughness: Some(4),
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            spec: TargetSpec::LandYouControl,
                            min: 1,
                            max: 1,
                            another: false,
                        },
                        effects: vec![Effect::Earthbend {
                            count: EffectValue::Fixed(1),
                            target: TargetSpec::LandYouControl,
                        }],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            spec: TargetSpec::LandYouControl,
                            min: 1,
                            max: 1,
                            another: false,
                        },
                        effects: vec![Effect::Earthbend {
                            count: EffectValue::Fixed(1),
                            target: TargetSpec::LandYouControl,
                        }],
                    },
                ],
            },
            Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrainLifeValue {
                    amount: EffectValue::CreaturesWithCounters,
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn dai_li_indoctrination() -> CardDefinition {
    let text_box = "Choose one —\n• Target opponent reveals their hand. You choose a nonland permanent card from it. That player discards that card.\n• Earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.dai_li_indoctrination",
        scryfall_id: "eca652b8-44f1-4bd9-b4bf-036eeead13aa",
        name: "Dai Li Indoctrination".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::RevealHandDiscardChosen {
                    target: TargetSpec::Opponent,
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Land, CardType::Instant, CardType::Sorcery],
                        ..Default::default()
                    },
                }],
                vec![Effect::Earthbend {
                    count: EffectValue::Fixed(2),
                    target: TargetSpec::LandYouControl,
                }],
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn deadly_precision() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, pay {4} or sacrifice an artifact or creature.\nDestroy target creature.";

    CardDefinition {
        key: "tla.deadly_precision",
        scryfall_id: "4661daf6-d960-4278-8946-70948efaf99d",
        name: "Deadly Precision".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        sacrifice_cost: Some(SacrificeCost {
            predicate: CardPredicate {
                card_types_any: vec![CardType::Artifact, CardType::Creature],
                ..Default::default()
            },
            another: false,
            or_mana: Some(ManaCost::parse("{4}")),
        }),
        spell_effects: vec![Effect::Destroy {
            target: TargetSpec::Creature,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn deserter_s_disciple() -> CardDefinition {
    let text_box =
        "{T}: Another target creature you control with power 2 or less can't be blocked this turn.";

    CardDefinition {
        key: "tla.deserter_s_disciple",
        scryfall_id: "14f2ed5a-042b-4cce-82ad-cfb4bd511d98",
        name: "Deserter's Disciple".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::CantBeBlockedTarget {
                target: TargetSpec::PermanentYouControl {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        max_power: Some(2),
                        ..Default::default()
                    },
                },
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn dragon() -> CardDefinition {
    let text_box = "Flying, firebending 4 (Whenever this token attacks, add {R}{R}{R}{R} to your mana pool. This mana lasts until end of combat.)";

    CardDefinition {
        key: "token.dragon",
        scryfall_id: "cddc0746-6d3e-441f-866f-28587bf54801",
        name: "Dragon".into(),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Dragon".into()],
        power: Some(4),
        toughness: Some(4),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        is_token: true,
        color_override: Some(Colors::from([Color::Red])),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("RRRR"),
                until_end_of_combat: true,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn dragonfly_swarm() -> CardDefinition {
    let text_box = "Flying, ward {1} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {1}.)\nThis creature's power is equal to the number of noncreature, nonland cards in your graveyard.\nWhen this creature dies, if there's a Lesson card in your graveyard, draw a card.";

    CardDefinition {
        key: "tla.dragonfly_swarm",
        scryfall_id: "16be5cc4-b15a-4c5e-8a6e-05c87b519127",
        name: "Dragonfly Swarm".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Dragon".into(), "Insect".into()],
        toughness: Some(3),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::ActiveIf {
                active_if: StaticCondition::GraveyardAtLeast {
                    count: 1,
                    predicate: CardPredicate {
                        subtype: Some("Lesson".into()),
                        ..Default::default()
                    },
                },
                condition: Box::new(TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                }),
            },
            effects: vec![Effect::IfGraveyardAtLeast {
                count: 1,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
                then: vec![Effect::DrawCards { count: 1 }],
                otherwise: vec![],
            }],
        }],
        ward: Some(ManaCost::parse("{1}")),
        power_cda: Some(PowerCda::GraveyardMatching(CardPredicate {
            not_card_types: vec![CardType::Creature, CardType::Land],
            ..Default::default()
        })),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn earth_kingdom_jailer() -> CardDefinition {
    let text_box = "When this creature enters, exile up to one target artifact, creature, or enchantment an opponent controls with mana value 3 or greater until this creature leaves the battlefield.";

    CardDefinition {
        key: "tla.earth_kingdom_jailer",
        scryfall_id: "9207362c-3605-4794-803f-ad1d0175fcca",
        name: "Earth Kingdom Jailer".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(3),
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
                        ..Default::default()
                    },
                },
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn earth_kingdom_protectors() -> CardDefinition {
    let text_box = "Vigilance\nSacrifice this creature: Another target Ally you control gains indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)";

    CardDefinition {
        key: "tla.earth_kingdom_protectors",
        scryfall_id: "d263472e-1d75-4cb9-8f7b-986fa22bc841",
        name: "Earth Kingdom Protectors".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            sacrifice_source: true,
            effects: vec![Effect::GrantKeywordsToTarget {
                keywords: Keywords {
                    indestructible: true,
                    ..Default::default()
                },
                target: TargetSpec::PermanentYouControl {
                    predicate: CardPredicate {
                        subtype: Some("Ally".into()),
                        ..Default::default()
                    },
                },
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn earth_kingdom_soldier() -> CardDefinition {
    let text_box = "Vigilance\nWhen this creature enters, put a +1/+1 counter on each of up to two target creatures you control.";

    CardDefinition {
        key: "tla.earth_kingdom_soldier",
        scryfall_id: "2d543d35-945a-4ffa-beb7-7c5d4f894f79",
        name: "Earth Kingdom Soldier".into(),
        mana_cost: Some(ManaCost::parse("{4}{G/W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(3),
        toughness: Some(4),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    spec: TargetSpec::CreatureYouControl,
                    min: 0,
                    max: 2,
                    another: false,
                },
                effects: vec![Effect::ForEachTarget {
                    effects: vec![Effect::PutCounters {
                        count: 1,
                        target: TargetSpec::CreatureYouControl,
                    }],
                }],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn earth_kings_lieutenant() -> CardDefinition {
    let text_box = "Trample\nWhen this creature enters, put a +1/+1 counter on each other Ally creature you control.\nWhenever another Ally you control enters, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "tla.earth_kings_lieutenant",
        scryfall_id: "4533d155-5c56-41a5-9d76-2d1414ac47c9",
        name: "Earth King's Lieutenant".into(),
        mana_cost: Some(ManaCost::parse("{W}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            trample: true,
            ..Default::default()
        },
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate {
                        subtype: Some("Ally".into()),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::PutCountersOnSource { count: 1 }],
            },
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::PutCountersOnEachMatching {
                    count: 1,
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        subtype: Some("Ally".into()),
                        ..Default::default()
                    },
                    other: true,
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn earth_village_ruffians() -> CardDefinition {
    let text_box = "When this creature dies, earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.earth_village_ruffians",
        scryfall_id: "56d4e0b7-232f-4953-84a3-e6f17781aa45",
        name: "Earth Village Ruffians".into(),
        mana_cost: Some(ManaCost::parse("{2}{B/G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Rogue".into()],
        power: Some(3),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(2),
                target: TargetSpec::LandYouControl,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn earthbending_lesson() -> CardDefinition {
    let text_box = "Earthbend 4. (Target land you control becomes a 0/0 creature with haste that's still a land. Put four +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.earthbending_lesson",
        scryfall_id: "eccd63b3-3a3a-4661-9d6e-fb8152429bdb",
        name: "Earthbending Lesson".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::Earthbend {
            count: EffectValue::Fixed(4),
            target: TargetSpec::LandYouControl,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn elemental_teachings() -> CardDefinition {
    let text_box = "Search your library for up to four land cards with different names and reveal them. An opponent chooses two of those cards. Put the chosen cards into your graveyard and the rest onto the battlefield tapped, then shuffle.";

    CardDefinition {
        key: "tla.elemental_teachings",
        scryfall_id: "cac8ac35-6860-4839-ab85-93d409206c08",
        name: "Elemental Teachings".into(),
        mana_cost: Some(ManaCost::parse("{4}{G}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::SearchLandsOpponentSplits {
            count: 4,
            opponent_picks: 2,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn enter_the_avatar_state() -> CardDefinition {
    let text_box = "Until end of turn, target creature you control becomes an Avatar in addition to its other types and gains flying, first strike, lifelink, and hexproof. (A creature with hexproof can't be the target of spells or abilities your opponents control.)";

    CardDefinition {
        key: "tla.enter_the_avatar_state",
        scryfall_id: "a8f3d2cb-d073-4df8-8769-49612628a377",
        name: "Enter the Avatar State".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::AddSubtypeToTarget {
                target: TargetSpec::CreatureYouControl,
                subtype: "Avatar".into(),
            },
            Effect::GrantKeywordsToTarget {
                keywords: Keywords {
                    flying: true,
                    first_strike: true,
                    lifelink: true,
                    hexproof: true,
                    ..Default::default()
                },
                target: TargetSpec::CreatureYouControl,
            },
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::CreatureYouControl,
            min: 1,
            max: 1,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fancy_footwork() -> CardDefinition {
    let text_box = "Untap one or two target creatures. They each get +2/+2 until end of turn.";

    CardDefinition {
        key: "tla.fancy_footwork",
        scryfall_id: "9c59f8c5-4063-4e5a-b684-fde175c4a981",
        name: "Fancy Footwork".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
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
        targeting: vec![TargetRequirement {
            spec: TargetSpec::Creature,
            min: 1,
            max: 2,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_attacks() -> CardDefinition {
    let text_box = "Create two 2/2 red Soldier creature tokens with firebending 1. (Whenever a creature with firebending 1 attacks, add {R}. This mana lasts until end of combat.)\nFlashback {8}{R} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";

    CardDefinition {
        key: "tla.fire_nation_attacks",
        scryfall_id: "9cc10845-8989-46ca-a57a-fd728fca0729",
        name: "Fire Nation Attacks".into(),
        mana_cost: Some(ManaCost::parse("{4}{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::CreateToken {
            token_name: "Soldier".into(),
            count: 2,
            tapped_and_attacking: false,
        }],
        flashback: Some(ManaCost::parse("{8}{R}")),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_cadets() -> CardDefinition {
    let text_box = "This creature has firebending 2 as long as there's a Lesson card in your graveyard. (Whenever this creature attacks, add {R}{R}. This mana lasts until end of combat.)\n{2}: This creature gets +1/+0 until end of turn.";

    CardDefinition {
        key: "tla.fire_nation_cadets",
        scryfall_id: "1a3a862c-9c9a-41cd-94d1-2d0cff22a6cd",
        name: "Fire Nation Cadets".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(1),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::ActiveIf {
                active_if: StaticCondition::GraveyardAtLeast {
                    count: 1,
                    predicate: CardPredicate {
                        subtype: Some("Lesson".into()),
                        ..Default::default()
                    },
                },
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
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::ModifyUntilEot {
                power_delta: 1,
                toughness_delta: 0,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_sages() -> CardDefinition {
    let text_box = "Firebending 1 (Whenever this creature attacks, add {R}. This mana lasts until end of combat.)\n{1}{R}{R}: Put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "tla.fire_sages",
        scryfall_id: "71a9fa44-306f-417a-a9d9-991aff95025c",
        name: "Fire Sages".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Cleric".into()],
        power: Some(2),
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
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{1}{R}{R}"),
            effects: vec![Effect::PutCountersOnSource { count: 1 }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn firebending_lesson() -> CardDefinition {
    let text_box = "Kicker {4} (You may pay an additional {4} as you cast this spell.)\nFirebending Lesson deals 2 damage to target creature. If this spell was kicked, it deals 5 damage to that creature instead.";

    CardDefinition {
        key: "tla.firebending_lesson",
        scryfall_id: "fa940e68-010e-4b68-be8a-555d7068f7b4",
        name: "Firebending Lesson".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
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
        targeting: vec![TargetRequirement {
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
        }],
        kicker: Some(ManaCost::parse("{4}")),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn first_time_flyer() -> CardDefinition {
    let text_box =
        "Flying\nThis creature gets +1/+1 as long as there's a Lesson card in your graveyard.";

    CardDefinition {
        key: "tla.first_time_flyer",
        scryfall_id: "ea22d29d-3a62-4ae2-91d5-21a678be9b48",
        name: "First-Time Flyer".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Pilot".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(2),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        static_pt_buffs: vec![StaticPtBuff {
            scope: StaticScope::This,
            condition: Some(StaticCondition::GraveyardAtLeast {
                count: 1,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            }),
            power: 1,
            toughness: 1,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn foggy_bottom_swamp() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {B} or {G}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.foggy_bottom_swamp",
        scryfall_id: "226af626-873d-4bd2-b889-000f8316786f",
        name: "Foggy Bottom Swamp".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("B"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("G"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn foggy_swamp_spirit_keeper() -> CardDefinition {
    let text_box = "Lifelink\nWhenever you draw your second card each turn, create a 1/1 colorless Spirit creature token with \"This token can't block or be blocked by non-Spirit creatures.\"";

    CardDefinition {
        key: "tla.foggy_swamp_spirit_keeper",
        scryfall_id: "cd624be9-e960-4485-a8a9-2d5f2dd51777",
        name: "Foggy Swamp Spirit Keeper".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Druid".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(4),
        keywords: Keywords {
            lifelink: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
            effects: vec![Effect::CreateToken {
                token_name: "Spirit".into(),
                count: 1,
                tapped_and_attacking: false,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn foggy_swamp_vinebender() -> CardDefinition {
    let text_box = "This creature can't be blocked by creatures with power 2 or less.\nWaterbend {5}: Put a +1/+1 counter on this creature. Activate only during your turn. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.foggy_swamp_vinebender",
        scryfall_id: "78a75317-94f7-47a4-b4da-5a027fa73248",
        name: "Foggy Swamp Vinebender".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Plant".into(), "Ally".into()],
        power: Some(4),
        toughness: Some(3),
        activated_abilities: vec![ActivatedAbilityDefinition {
            own_turn_only: true,
            mana_cost: ManaCost::parse("{5}"),
            waterbend: true,
            effects: vec![Effect::PutCountersOnSource { count: 1 }],
            ..Default::default()
        }],
        block_restriction: Some(CardPredicate {
            max_power: Some(2),
            ..Default::default()
        }),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn food() -> CardDefinition {
    let text_box = "{2}, {T}, Sacrifice this token: You gain 3 life.";

    CardDefinition {
        key: "token.food",
        scryfall_id: "db202f04-61e7-4e12-848e-6c1761956a58",
        name: "Food".into(),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Food".into()],
        is_token: true,
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{2}"),
            sacrifice_source: true,
            effects: vec![Effect::GainLife { amount: 3 }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn forecasting_fortune_teller() -> CardDefinition {
    let text_box = "When this creature enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.forecasting_fortune_teller",
        scryfall_id: "023e7ad4-6af0-4f29-a716-27a313974227",
        name: "Forecasting Fortune Teller".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Advisor".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(3),
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn gather_the_white_lotus() -> CardDefinition {
    let text_box = "Create a 1/1 white Ally creature token for each Plains you control. Scry 2. (Look at the top two cards of your library, then put any number of them on the bottom and the rest on top in any order.)";

    CardDefinition {
        key: "tla.gather_the_white_lotus",
        scryfall_id: "96e2b3d0-f060-4f3c-9d12-a0444b202008",
        name: "Gather the White Lotus".into(),
        mana_cost: Some(ManaCost::parse("{4}{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::CreateTokensValue {
                token_name: "Ally".into(),
                count: EffectValue::BattlefieldMatching {
                    predicate: CardPredicate {
                        subtype: Some("Plains".into()),
                        ..Default::default()
                    },
                },
            },
            Effect::Scry { count: 2 },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn geyser_leaper() -> CardDefinition {
    let text_box = "Flying\nWaterbend {4}: Draw a card, then discard a card. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.geyser_leaper",
        scryfall_id: "4d25428d-e5a5-43c4-8544-43120ed1c4d5",
        name: "Geyser Leaper".into(),
        mana_cost: Some(ManaCost::parse("{4}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(4),
        toughness: Some(3),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{4}"),
            waterbend: true,
            effects: vec![Effect::DrawCards { count: 1 }, Effect::Discard],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn giant_koi() -> CardDefinition {
    let text_box = "Waterbend {3}: This creature can't be blocked this turn. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)\nIslandcycling {2} ({2}, Discard this card: Search your library for an Island card, reveal it, put it into your hand, then shuffle.)";

    CardDefinition {
        key: "tla.giant_koi",
        scryfall_id: "3938ec1f-979c-48e5-bb13-bebace006a8f",
        name: "Giant Koi".into(),
        mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Fish".into()],
        power: Some(5),
        toughness: Some(7),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                zone: ActivationZone::Hand,
                discard_source: true,
                mana_cost: ManaCost::parse("{2}"),
                effects: vec![Effect::SearchBasicLand {
                    subtype: Some("Island".into()),
                    count: 1,
                    tapped: true,
                    untap_if_lands: None,
                    bottom_source: false,
                    to_hand: true,
                }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}"),
                waterbend: true,
                effects: vec![Effect::CantBeBlockedThisTurnSource],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn glider_kids() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, scry 1. (Look at the top card of your library. You may put it on the bottom.)";

    CardDefinition {
        key: "tla.glider_kids",
        scryfall_id: "0613afb5-4cee-4132-8adf-404767ed8d07",
        name: "Glider Kids".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Pilot".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(3),
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn gran_gran() -> CardDefinition {
    let text_box = "Whenever Gran-Gran becomes tapped, draw a card, then discard a card.\nNoncreature spells you cast cost {1} less to cast as long as there are three or more Lesson cards in your graveyard.";

    CardDefinition {
        key: "tla.gran_gran",
        scryfall_id: "fa434b41-e5f7-4989-865a-95db67b05cb1",
        name: "Gran-Gran".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Peasant".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(2),
        spell_cost_reductions: vec![SpellCostReduction {
            first_on_own_turn: false,
            predicate: CardPredicate {
                not_card_types: vec![CardType::Creature],
                ..Default::default()
            },
            condition: StaticCondition::GraveyardAtLeast {
                count: 3,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            },
            generic: 1,
        }],
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::BecomesTapped {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DrawCards { count: 1 }, Effect::Discard],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn great_divide_guide() -> CardDefinition {
    let text_box = "Each land and Ally you control has \"{T}: Add one mana of any color.\"";

    CardDefinition {
        key: "tla.great_divide_guide",
        scryfall_id: "cc3063ec-5ea6-46c1-8331-c740cbaf6c76",
        name: "Great Divide Guide".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Scout".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(3),
        mana_grant: vec![
            CardPredicate {
                card_type: Some(CardType::Land),
                ..Default::default()
            },
            CardPredicate {
                subtype: Some("Ally".into()),
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn hama_the_bloodbender() -> CardDefinition {
    let text_box = "When Hama enters, target opponent mills three cards. Exile up to one noncreature, nonland card from that player's graveyard. For as long as you control Hama, you may cast the exiled card during your turn by waterbending {X} rather than paying its mana cost, where X is its mana value. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.hama_the_bloodbender",
        scryfall_id: "6fa1197d-7b19-4d86-81e2-5c87de87757b",
        name: "Hama, the Bloodbender".into(),
        mana_cost: Some(ManaCost::parse("{2}{U/B}{U/B}{U/B}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Warlock".into()],
        power: Some(3),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![
                Effect::MillCards {
                    count: 3,
                    target: Some(TargetSpec::Opponent),
                },
                Effect::ExileLinked {
                    chooser: LinkedChooser::Controller,
                    zone: ZoneType::Graveyard,
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature, CardType::Land],
                        ..Default::default()
                    },
                    up_to: true,
                    grant: LinkedGrant::WaterbendManaValue,
                    target: TargetSpec::Opponent,
                },
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn haru_hidden_talent() -> CardDefinition {
    let text_box = "Whenever another Ally you control enters, earthbend 1. (Target land you control becomes a 0/0 creature with haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.haru_hidden_talent",
        scryfall_id: "9657a0c0-2c26-4331-8e62-f032265a01af",
        name: "Haru, Hidden Talent".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Human".into(), "Peasant".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnotherYouControl(CardPredicate {
                    subtype: Some("Ally".into()),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(1),
                target: TargetSpec::LandYouControl,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn hermitic_herbalist() -> CardDefinition {
    let text_box = "{T}: Add one mana of any color.\n{T}: Add two mana in any combination of colors. Spend this mana only to cast Lesson spells.";

    CardDefinition {
        key: "tla.hermitic_herbalist",
        scryfall_id: "5d0ee441-a057-41b5-abb8-dc86864ac248",
        name: "Hermitic Herbalist".into(),
        mana_cost: Some(ManaCost::parse("{U}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Druid".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(3),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("W"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("U"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("B"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("R"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("G"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [2, 0, 0, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [1, 1, 0, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [1, 0, 1, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [1, 0, 0, 1, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [1, 0, 0, 0, 1, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 2, 0, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 1, 1, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 1, 0, 1, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 1, 0, 0, 1, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 0, 2, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 0, 1, 1, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 0, 1, 0, 1, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 0, 0, 2, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 0, 0, 1, 1, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson"]),
                                [0, 0, 0, 0, 2, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn hog_monkey() -> CardDefinition {
    let text_box = "At the beginning of combat on your turn, target creature you control with a +1/+1 counter on it gains menace until end of turn. (It can't be blocked except by two or more creatures.)\nExhaust — {5}: Put two +1/+1 counters on this creature. (Activate each exhaust ability only once.)";

    CardDefinition {
        key: "tla.hog_monkey",
        scryfall_id: "4f442970-5355-4abf-8684-17daaa8e469b",
        name: "Hog-Monkey".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Boar".into(), "Monkey".into()],
        power: Some(3),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::BeginningOfYourCombat,
            effects: vec![Effect::GrantKeywordsToTarget {
                keywords: Keywords {
                    menace: true,
                    ..Default::default()
                },
                target: TargetSpec::PermanentYouControl {
                    predicate: CardPredicate {
                        with_counter: true,
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            exhaust: true,
            mana_cost: ManaCost::parse("{5}"),
            effects: vec![Effect::PutCountersOnSource { count: 2 }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn iguana_parrot() -> CardDefinition {
    let text_box = "Flying, vigilance\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";

    CardDefinition {
        key: "tla.iguana_parrot",
        scryfall_id: "7faa3e3d-c08a-4908-85ef-1370d3f90815",
        name: "Iguana Parrot".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Lizard".into(), "Bird".into(), "Pirate".into()],
        power: Some(2),
        toughness: Some(2),
        keywords: Keywords {
            flying: true,
            vigilance: true,
            ..Default::default()
        },
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn invasion_reinforcements() -> CardDefinition {
    let text_box = "Flash\nWhen this creature enters, create a 1/1 white Ally creature token.";

    CardDefinition {
        key: "tla.invasion_reinforcements",
        scryfall_id: "b8845bba-d116-43cd-8ae0-e553f3324d66",
        name: "Invasion Reinforcements".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            flash: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn iroh_grand_lotus() -> CardDefinition {
    let text_box = "Firebending 2\nDuring your turn, each non-Lesson instant and sorcery card in your graveyard has flashback. The flashback cost is equal to that card's mana cost. (You may cast a card from your graveyard for its flashback cost. Then exile it.)\nDuring your turn, each Lesson card in your graveyard has flashback {1}.";

    CardDefinition {
        key: "tla.iroh_grand_lotus",
        scryfall_id: "879b73d3-4552-4fdc-baee-c4d097ae9a4f",
        name: "Iroh, Grand Lotus".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}{R}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into(), "Ally".into()],
        power: Some(5),
        toughness: Some(5),
        grants_flashback: true,
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("RR"),
                until_end_of_combat: true,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn iroh_s_demonstration() -> CardDefinition {
    let text_box = "Choose one —\n• Iroh's Demonstration deals 1 damage to each creature your opponents control.\n• Iroh's Demonstration deals 4 damage to target creature.";

    CardDefinition {
        key: "tla.iroh_s_demonstration",
        scryfall_id: "18d15fed-1f8f-4407-a221-a47ce75001a8",
        name: "Iroh's Demonstration".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::DamageOpponentCreatures { amount: 1 }],
                vec![Effect::DealDamage {
                    amount: 4,
                    target: TargetSpec::Creature,
                }],
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn itll_quench_ya() -> CardDefinition {
    let text_box = "Counter target spell unless its controller pays {2}.";

    CardDefinition {
        key: "tla.itll_quench_ya",
        scryfall_id: "47c25e41-f43c-4447-81b5-b9631448bd29",
        name: "It'll Quench Ya!".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::CounterUnlessPays {
            cost: ManaCost::parse("{2}"),
        }],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::Spell,
            min: 1,
            max: 1,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn jasmine_dragon_tea_shop() -> CardDefinition {
    let text_box = "{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast an Ally spell or activate an ability of an Ally source.\n{5}, {T}: Create a 1/1 white Ally creature token.";

    CardDefinition {
        key: "tla.jasmine_dragon_tea_shop",
        scryfall_id: "da2c83d4-a95f-47ff-a08f-694eb78d6b9b",
        name: "Jasmine Dragon Tea Shop".into(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("1"),
        }],
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::new(["Ally"], ["Ally"]),
                                [1, 0, 0, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::new(["Ally"], ["Ally"]),
                                [0, 1, 0, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::new(["Ally"], ["Ally"]),
                                [0, 0, 1, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::new(["Ally"], ["Ally"]),
                                [0, 0, 0, 1, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::new(["Ally"], ["Ally"]),
                                [0, 0, 0, 0, 1, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{5}"),
                effects: vec![Effect::CreateToken {
                    token_name: "Ally".into(),
                    count: 1,
                    tapped_and_attacking: false,
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn jeong_jeongs_deserters() -> CardDefinition {
    let text_box = "When this creature enters, put a +1/+1 counter on target creature.";

    CardDefinition {
        key: "tla.jeong_jeongs_deserters",
        scryfall_id: "060966f9-5e56-4512-9ef3-2b216daf093c",
        name: "Jeong Jeong's Deserters".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::PutCounters {
                count: 1,
                target: TargetSpec::Creature,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn jet_freedom_fighter() -> CardDefinition {
    let text_box = "When Jet enters, he deals damage equal to the number of creatures you control to target creature an opponent controls.\nWhen Jet dies, put a +1/+1 counter on each of up to two target creatures.";

    CardDefinition {
        key: "tla.jet_freedom_fighter",
        scryfall_id: "9202c044-15e4-4218-a94d-16287ba19d69",
        name: "Jet, Freedom Fighter".into(),
        mana_cost: Some(ManaCost::parse("{2}{R/W}{R/W}{R/W}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(1),
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DealDamageValue {
                    amount: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Creature),
                            ..Default::default()
                        },
                    },
                    target: TargetSpec::CreatureOpponentControls,
                }],
            },
            Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        spec: TargetSpec::Creature,
                        min: 0,
                        max: 2,
                        another: false,
                    },
                    effects: vec![Effect::ForEachTarget {
                        effects: vec![Effect::PutCounters {
                            count: 1,
                            target: TargetSpec::Creature,
                        }],
                    }],
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn june_bounty_hunter() -> CardDefinition {
    let text_box = "June can't be blocked as long as you've drawn two or more cards this turn.\n{1}, Sacrifice another creature: Create a Clue token. Activate only during your turn. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.june_bounty_hunter",
        scryfall_id: "148bbaab-bc7b-46ab-8e18-ade69d71d847",
        name: "June, Bounty Hunter".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Mercenary".into()],
        power: Some(2),
        toughness: Some(2),
        unblockable_if: Some(StaticCondition::CardsDrawnAtLeast { count: 2 }),
        activated_abilities: vec![ActivatedAbilityDefinition {
            sacrifice: Some(SacrificeCost {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                another: true,
                or_mana: None,
            }),
            own_turn_only: true,
            mana_cost: ManaCost::parse("{1}"),
            effects: vec![Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn knowledge_seeker() -> CardDefinition {
    let text_box = "Vigilance\nWhenever you draw your second card each turn, put a +1/+1 counter on this creature.\nWhen this creature dies, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.knowledge_seeker",
        scryfall_id: "8554e862-f78a-42f5-b876-076aac6c9504",
        name: "Knowledge Seeker".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Fox".into(), "Spirit".into()],
        power: Some(2),
        toughness: Some(1),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn kyoshi_village() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {G} or {W}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.kyoshi_village",
        scryfall_id: "8d5f3008-2af8-4e81-8847-1c91f524e747",
        name: "Kyoshi Village".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("G"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("W"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn kyoshi_warriors() -> CardDefinition {
    let text_box = "When this creature enters, create a 1/1 white Ally creature token.";

    CardDefinition {
        key: "tla.kyoshi_warriors",
        scryfall_id: "211045e1-85c7-4088-b830-a2afa0fe520b",
        name: "Kyoshi Warriors".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn lost_days() -> CardDefinition {
    let text_box = "The owner of target creature or enchantment puts it into their library second from the top or on the bottom. You create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.lost_days",
        scryfall_id: "db8e88e3-0931-403f-90e7-3c22c0b61dac",
        name: "Lost Days".into(),
        mana_cost: Some(ManaCost::parse("{4}{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::OwnerPutsIntoLibrary {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Creature, CardType::Enchantment],
                        ..Default::default()
                    },
                },
            },
            Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
            },
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::Permanent {
                predicate: CardPredicate {
                    card_types_any: vec![CardType::Creature, CardType::Enchantment],
                    ..Default::default()
                },
            },
            min: 1,
            max: 1,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn master_piandao() -> CardDefinition {
    let text_box = "First strike\nWhenever Master Piandao attacks, look at the top four cards of your library. You may reveal an Ally, Equipment, or Lesson card from among them and put it into your hand. Put the rest on the bottom of your library in a random order.";

    CardDefinition {
        key: "tla.master_piandao",
        scryfall_id: "3e4a042c-f2a6-45e7-9444-acd1ca838b87",
        name: "Master Piandao".into(),
        mana_cost: Some(ManaCost::parse("{4}{W}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(4),
        toughness: Some(4),
        keywords: Keywords {
            first_strike: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::LookAndSelect {
                reveal: true,
                look: EffectValue::Fixed(4),
                min_select: 0,
                max_select: 1,
                predicate: CardPredicate {
                    subtypes_any: vec!["Ally".into(), "Equipment".into(), "Lesson".into()],
                    ..Default::default()
                },
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn meditation_pools() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {G} or {U}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.meditation_pools",
        scryfall_id: "cc078fc6-4470-473f-9e24-d2c470e09141",
        name: "Meditation Pools".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("G"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("U"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn merchant_of_many_hats() -> CardDefinition {
    let text_box = "{2}{B}: Return this card from your graveyard to your hand.";

    CardDefinition {
        key: "tla.merchant_of_many_hats",
        scryfall_id: "752ffa24-93b6-4b33-bf10-7222357ac472",
        name: "Merchant of Many Hats".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Peasant".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        activated_abilities: vec![ActivatedAbilityDefinition {
            zone: ActivationZone::Graveyard,
            mana_cost: ManaCost::parse("{2}{B}"),
            effects: vec![Effect::ReturnSourceToHand],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn messenger_hawk() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nThis creature gets +2/+0 as long as you've drawn two or more cards this turn.";

    CardDefinition {
        key: "tla.messenger_hawk",
        scryfall_id: "6a2946c0-0a40-4cdb-92c8-af2bafecca09",
        name: "Messenger Hawk".into(),
        mana_cost: Some(ManaCost::parse("{2}{U/B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Bird".into(), "Scout".into()],
        power: Some(1),
        toughness: Some(2),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn misty_palms_oasis() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {W} or {B}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.misty_palms_oasis",
        scryfall_id: "7cc29ddb-8f44-4493-985d-2fac35f9f34d",
        name: "Misty Palms Oasis".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("W"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("B"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn momo_friendly_flier() -> CardDefinition {
    let text_box = "Flying\nThe first non-Lemur creature spell with flying you cast during each of your turns costs {1} less to cast.\nWhenever another creature you control with flying enters, Momo gets +1/+1 until end of turn.";

    CardDefinition {
        key: "tla.momo_friendly_flier",
        scryfall_id: "c472ef84-a632-4ad7-853c-60588a7a4b12",
        name: "Momo, Friendly Flier".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Lemur".into(), "Bat".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        spell_cost_reductions: vec![SpellCostReduction {
            first_on_own_turn: true,
            predicate: CardPredicate {
                flying: true,
                not_subtypes: vec!["Lemur".into()],
                card_type: Some(CardType::Creature),
                ..Default::default()
            },
            condition: StaticCondition::Always,
            generic: 1,
        }],
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnotherYouControl(CardPredicate {
                    flying: true,
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::ModifyUntilEot {
                power_delta: 1,
                toughness_delta: 1,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn mongoose_lizard() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nWhen this creature enters, it deals 1 damage to any target.\nMountaincycling {2} ({2}, Discard this card: Search your library for a Mountain card, reveal it, put it into your hand, then shuffle.)";

    CardDefinition {
        key: "tla.mongoose_lizard",
        scryfall_id: "7d40b9cb-82ef-44c5-8d3c-c8bb4ce9891c",
        name: "Mongoose Lizard".into(),
        mana_cost: Some(ManaCost::parse("{4}{R}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Mongoose".into(), "Lizard".into()],
        power: Some(5),
        toughness: Some(6),
        keywords: Keywords {
            menace: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DealDamage {
                amount: 1,
                target: TargetSpec::CreatureOrPlayer,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            zone: ActivationZone::Hand,
            discard_source: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::SearchBasicLand {
                subtype: Some("Mountain".into()),
                count: 1,
                tapped: true,
                untap_if_lands: None,
                bottom_source: false,
                to_hand: true,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn monk() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";

    CardDefinition {
        key: "token.monk",
        scryfall_id: "e49c80ac-5c90-4fd2-ad40-cc77af160f64",
        name: "Monk".into(),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Monk".into()],
        power: Some(1),
        toughness: Some(1),
        is_token: true,
        color_override: Some(Colors::from([Color::Red])),
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn north_pole_gates() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {W} or {U}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.north_pole_gates",
        scryfall_id: "bcf7b9da-d35d-4338-b4d9-83e47dc058ee",
        name: "North Pole Gates".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("W"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("U"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn north_pole_patrol() -> CardDefinition {
    let text_box = "{T}: Untap another target permanent you control.\nWaterbend {3}, {T}: Tap target creature an opponent controls. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.north_pole_patrol",
        scryfall_id: "c2c0c138-5b61-4949-8431-4a6d458ead6a",
        name: "North Pole Patrol".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(3),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{3}"),
                waterbend: true,
                effects: vec![Effect::TapTarget {
                    target: TargetSpec::CreatureOpponentControls,
                }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::UntapTarget {
                    target: TargetSpec::PermanentYouControl {
                        predicate: CardPredicate::default(),
                    },
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn octopus_form() -> CardDefinition {
    let text_box = "Target creature you control gets +1/+1 and gains hexproof until end of turn. Untap it. (It can't be the target of spells or abilities your opponents control.)";

    CardDefinition {
        key: "tla.octopus_form",
        scryfall_id: "ce96a826-53a0-4029-bf34-46779d133b13",
        name: "Octopus Form".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn omashu_city() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {R} or {G}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.omashu_city",
        scryfall_id: "c21e413b-96ed-4ded-b567-f1647074ce5f",
        name: "Omashu City".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("R"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("G"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn origin_of_metalbending() -> CardDefinition {
    let text_box = "Choose one —\n• Destroy target artifact or enchantment.\n• Put a +1/+1 counter on target creature you control. It gains indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)";

    CardDefinition {
        key: "tla.origin_of_metalbending",
        scryfall_id: "25749f9a-260d-4bdc-bd28-429c12faa4a5",
        name: "Origin of Metalbending".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn ostrich_horse() -> CardDefinition {
    let text_box = "When this creature enters, mill three cards. You may put a land card from among them into your hand. If you don't, put a +1/+1 counter on this creature. (To mill three cards, put the top three cards of your library into your graveyard.)";

    CardDefinition {
        key: "tla.ostrich_horse",
        scryfall_id: "5ca3fd45-9301-44ef-afc0-4d7999d66d36",
        name: "Ostrich-Horse".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Bird".into(), "Horse".into()],
        power: Some(3),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::MillAndSelect {
                count: 3,
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                },
                if_declined: vec![Effect::PutCountersOnSource { count: 1 }],
                if_milled: None,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn otter_penguin() -> CardDefinition {
    let text_box = "Whenever you draw your second card each turn, this creature gets +1/+2 until end of turn and can't be blocked this turn.";

    CardDefinition {
        key: "tla.otter_penguin",
        scryfall_id: "480b6279-bb34-4b3f-a639-868b52af92b9",
        name: "Otter-Penguin".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Otter".into(), "Bird".into()],
        power: Some(2),
        toughness: Some(1),
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn ozai_s_cruelty() -> CardDefinition {
    let text_box =
        "Ozai's Cruelty deals 2 damage to target player. That player discards two cards.";

    CardDefinition {
        key: "tla.ozai_s_cruelty",
        scryfall_id: "22cad680-d46a-4589-a633-b6ee9a78d61e",
        name: "Ozai's Cruelty".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn pillar_launch() -> CardDefinition {
    let text_box = "Target creature gets +2/+2 and gains reach until end of turn. Untap it.";

    CardDefinition {
        key: "tla.pillar_launch",
        scryfall_id: "f177e9cc-0c41-417d-9215-5ff23a1a0657",
        name: "Pillar Launch".into(),
        mana_cost: Some(ManaCost::parse("{G}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::BuffTarget {
                power: 2,
                toughness: 2,
                target: TargetSpec::Creature,
            },
            Effect::GrantKeywordsToTarget {
                keywords: Keywords {
                    reach: true,
                    ..Default::default()
                },
                target: TargetSpec::Creature,
            },
            Effect::UntapTarget {
                target: TargetSpec::Creature,
            },
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn pretending_poxbearers() -> CardDefinition {
    let text_box = "When this creature dies, create a 1/1 white Ally creature token.";

    CardDefinition {
        key: "tla.pretending_poxbearers",
        scryfall_id: "d1af91a5-8681-4a05-910d-96f7a819bfaa",
        name: "Pretending Poxbearers".into(),
        mana_cost: Some(ManaCost::parse("{1}{W/B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into(), "Ally".into()],
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn professor_zei_anthropologist() -> CardDefinition {
    let text_box = "{T}, Discard a card: Draw a card.\n{1}, {T}, Sacrifice Professor Zei: Return target instant or sorcery card from your graveyard to your hand. Activate only during your turn.";

    CardDefinition {
        key: "tla.professor_zei_anthropologist",
        scryfall_id: "27baccc0-7e25-4f39-be0d-31cd98ca0dc5",
        name: "Professor Zei, Anthropologist".into(),
        mana_cost: Some(ManaCost::parse("{U/R}{U/R}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Advisor".into(), "Ally".into()],
        power: Some(0),
        toughness: Some(3),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                discard_card: true,
                tap_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                own_turn_only: true,
                tap_source: true,
                mana_cost: ManaCost::parse("{1}"),
                sacrifice_source: true,
                effects: vec![Effect::ReturnGraveyardTarget {
                    target: TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Instant, CardType::Sorcery],
                            ..Default::default()
                        },
                    },
                    tapped_onto_battlefield: false,
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn rabaroo_troop() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, this creature gains flying until end of turn and you gain 1 life.\nPlainscycling {2} ({2}, Discard this card: Search your library for a Plains card, reveal it, put it into your hand, then shuffle.)";

    CardDefinition {
        key: "tla.rabaroo_troop",
        scryfall_id: "62a983b4-ac73-4949-9317-05a75a8ce164",
        name: "Rabaroo Troop".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Rabbit".into(), "Kangaroo".into()],
        power: Some(3),
        toughness: Some(5),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
            },
            effects: vec![
                Effect::GrantKeywordsToSource {
                    keywords: Keywords {
                        flying: true,
                        ..Default::default()
                    },
                },
                Effect::GainLife { amount: 1 },
            ],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            zone: ActivationZone::Hand,
            discard_source: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::SearchBasicLand {
                subtype: Some("Plains".into()),
                count: 1,
                tapped: true,
                untap_if_lands: None,
                bottom_source: false,
                to_hand: true,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn ran_and_shaw() -> CardDefinition {
    let text_box = "Flying, firebending 2\nWhen Ran and Shaw enter, if you cast them and there are three or more Dragon and/or Lesson cards in your graveyard, create a token that's a copy of Ran and Shaw, except it's not legendary.\n{3}{R}: Dragons you control get +2/+0 until end of turn.";

    CardDefinition {
        key: "tla.ran_and_shaw",
        scryfall_id: "6436e2d0-989f-47e3-90bc-9cde82a2ddb4",
        name: "Ran and Shaw".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Dragon".into()],
        power: Some(4),
        toughness: Some(4),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RR"),
                    until_end_of_combat: true,
                }],
            },
            Ability::Triggered {
                condition: TriggerCondition::ActiveIf {
                    active_if: StaticCondition::GraveyardAtLeast {
                        count: 3,
                        predicate: CardPredicate {
                            subtypes_any: vec!["Dragon".into(), "Lesson".into()],
                            ..Default::default()
                        },
                    },
                    condition: Box::new(TriggerCondition::WasCast {
                        condition: Box::new(TriggerCondition::EntersTheBattlefield {
                            subject: TriggerSubject::This,
                        }),
                    }),
                },
                effects: vec![Effect::IfGraveyardAtLeast {
                    count: 3,
                    predicate: CardPredicate {
                        subtypes_any: vec!["Dragon".into(), "Lesson".into()],
                        ..Default::default()
                    },
                    then: vec![Effect::CreateNonlegendaryCopy],
                    otherwise: vec![],
                }],
            },
        ],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}{R}"),
            effects: vec![Effect::ModifyControlled {
                predicate: CardPredicate {
                    subtype: Some("Dragon".into()),
                    ..Default::default()
                },
                power: 2,
                toughness: 0,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn rebellious_captives() -> CardDefinition {
    let text_box = "Exhaust — {6}: Put two +1/+1 counters on this creature, then earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped. Activate each exhaust ability only once.)";

    CardDefinition {
        key: "tla.rebellious_captives",
        scryfall_id: "e64a4a8b-323c-4c8b-92af-1a3295799e65",
        name: "Rebellious Captives".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Peasant".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        activated_abilities: vec![ActivatedAbilityDefinition {
            exhaust: true,
            mana_cost: ManaCost::parse("{6}"),
            effects: vec![
                Effect::PutCountersOnSource { count: 2 },
                Effect::Earthbend {
                    count: EffectValue::Fixed(2),
                    target: TargetSpec::LandYouControl,
                },
            ],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn rockalanche() -> CardDefinition {
    let text_box = "Earthbend X, where X is the number of Forests you control. (Target land you control becomes a 0/0 creature with haste that's still a land. Put X +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)\nFlashback {5}{G} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";

    CardDefinition {
        key: "tla.rockalanche",
        scryfall_id: "52b213d3-6f68-43e6-91e9-435d8fe1f34c",
        name: "Rockalanche".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::Earthbend {
            count: EffectValue::BattlefieldMatching {
                predicate: CardPredicate {
                    subtype: Some("Forest".into()),
                    ..Default::default()
                },
            },
            target: TargetSpec::LandYouControl,
        }],
        flashback: Some(ManaCost::parse("{5}{G}")),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn rough_rhino_cavalry() -> CardDefinition {
    let text_box = "Firebending 2 (Whenever this creature attacks, add {R}{R}. This mana lasts until end of combat.)\nExhaust — {8}: Put two +1/+1 counters on this creature. It gains trample until end of turn. (Activate each exhaust ability only once.)";

    CardDefinition {
        key: "tla.rough_rhino_cavalry",
        scryfall_id: "a8bc29cf-f6f8-4bb8-b722-f016b58b6d2d",
        name: "Rough Rhino Cavalry".into(),
        mana_cost: Some(ManaCost::parse("{4}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Mercenary".into()],
        power: Some(5),
        toughness: Some(5),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("RR"),
                until_end_of_combat: true,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            exhaust: true,
            mana_cost: ManaCost::parse("{8}"),
            effects: vec![
                Effect::PutCountersOnSource { count: 2 },
                Effect::GrantKeywordsToSource {
                    keywords: Keywords {
                        trample: true,
                        ..Default::default()
                    },
                },
            ],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn rumble_arena() -> CardDefinition {
    let text_box = "Vigilance\nWhen this land enters, scry 1. (Look at the top card of your library. You may put it on the bottom.)\n{T}: Add {C}.\n{1}, {T}: Add one mana of any color.";

    CardDefinition {
        key: "tla.rumble_arena",
        scryfall_id: "3f5d2c38-663c-4f64-8cfc-c102462e82ef",
        name: "Rumble Arena".into(),
        types: CardTypes::new([CardType::Land]),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Scry { count: 1 }],
        }],
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("1"),
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{1}"),
            effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::AddMana {
                        mana: Mana::parse("W"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("U"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("B"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("R"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("G"),
                        until_end_of_combat: false,
                    }],
                ],
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn saber_tooth_moose_lion() -> CardDefinition {
    let text_box = "Reach\nForestcycling {2} ({2}, Discard this card: Search your library for a Forest card, reveal it, put it into your hand, then shuffle.)";

    CardDefinition {
        key: "tla.saber_tooth_moose_lion",
        scryfall_id: "b15f302d-c451-4e79-a5af-75ee3ba4cadf",
        name: "Saber-Tooth Moose-Lion".into(),
        mana_cost: Some(ManaCost::parse("{4}{G}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Elk".into(), "Cat".into()],
        power: Some(7),
        toughness: Some(7),
        keywords: Keywords {
            reach: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            zone: ActivationZone::Hand,
            discard_source: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::SearchBasicLand {
                subtype: Some("Forest".into()),
                count: 1,
                tapped: true,
                untap_if_lands: None,
                bottom_source: false,
                to_hand: true,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sandbenders_storm() -> CardDefinition {
    let text_box = "Choose one —\n• Destroy target creature with power 4 or greater.\n• Earthbend 3. (Target land you control becomes a 0/0 creature with haste that's still a land. Put three +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.sandbenders_storm",
        scryfall_id: "b3bf4a25-4329-4318-870a-2b06aa620dc4",
        name: "Sandbenders' Storm".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::Destroy {
                    target: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            min_power: Some(4),
                            card_type: Some(CardType::Creature),
                            ..Default::default()
                        },
                    },
                }],
                vec![Effect::Earthbend {
                    count: EffectValue::Fixed(3),
                    target: TargetSpec::LandYouControl,
                }],
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn seismic_sense() -> CardDefinition {
    let text_box = "Look at the top X cards of your library, where X is the number of lands you control. You may reveal a creature or land card from among them and put it into your hand. Put the rest on the bottom of your library in a random order.";

    CardDefinition {
        key: "tla.seismic_sense",
        scryfall_id: "a7b93c2a-c478-4734-8aca-03f3e9c11b20",
        name: "Seismic Sense".into(),
        mana_cost: Some(ManaCost::parse("{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::LookAndSelect {
            reveal: true,
            look: EffectValue::BattlefieldMatching {
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                },
            },
            min_select: 0,
            max_select: 1,
            predicate: CardPredicate {
                card_types_any: vec![CardType::Creature, CardType::Land],
                ..Default::default()
            },
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn serpent_of_the_pass() -> CardDefinition {
    let text_box = "If there are three or more Lesson cards in your graveyard, you may cast this spell as though it had flash.\nThis spell costs {1} less to cast for each noncreature, nonland card in your graveyard.";

    CardDefinition {
        key: "tla.serpent_of_the_pass",
        scryfall_id: "87595843-15bc-48bb-8a81-3e6ad924ed44",
        name: "Serpent of the Pass".into(),
        mana_cost: Some(ManaCost::parse("{5}{U}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Serpent".into()],
        power: Some(6),
        toughness: Some(5),
        flash_if: Some(StaticCondition::GraveyardAtLeast {
            count: 3,
            predicate: CardPredicate {
                subtype: Some("Lesson".into()),
                ..Default::default()
            },
        }),
        cost_reduction_in_graveyard: true,
        cost_reduction_per: Some(CardPredicate {
            not_card_types: vec![CardType::Creature, CardType::Land],
            ..Default::default()
        }),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn serpents_pass() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {U} or {B}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.serpents_pass",
        scryfall_id: "ad87bff5-9b8c-44e4-a6d3-8cc71be9640a",
        name: "Serpent's Pass".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("U"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("B"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn shared_roots() -> CardDefinition {
    let text_box = "Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.";

    CardDefinition {
        key: "tla.shared_roots",
        scryfall_id: "e7847ba5-e85e-417f-96c0-aef2e6f83994",
        name: "Shared Roots".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::SearchBasicLand {
            subtype: None,
            count: 1,
            tapped: true,
            untap_if_lands: None,
            bottom_source: false,
            to_hand: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sokka_bold_boomeranger() -> CardDefinition {
    let text_box = "When Sokka enters, discard up to two cards, then draw that many cards.\nWhenever you cast an artifact or Lesson spell, put a +1/+1 counter on Sokka.";

    CardDefinition {
        key: "tla.sokka_bold_boomeranger",
        scryfall_id: "4a1f1472-55b4-450d-8e4d-7297130a0cf3",
        name: "Sokka, Bold Boomeranger".into(),
        mana_cost: Some(ManaCost::parse("{U}{R}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(1),
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::OptionalDiscardDraw { count: 1, up_to: 2 }],
            },
            Ability::Triggered {
                condition: TriggerCondition::YouCastSpell {
                    predicate: CardPredicate {
                        any_of: vec![
                            CardPredicate {
                                card_type: Some(CardType::Artifact),
                                ..Default::default()
                            },
                            CardPredicate {
                                subtype: Some("Lesson".into()),
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                },
                effects: vec![Effect::PutCountersOnSource { count: 1 }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sokka_s_haiku() -> CardDefinition {
    let text_box = "Counter target spell.\nDraw a card, then mill three cards.\nUntap target land.";

    CardDefinition {
        key: "tla.sokka_s_haiku",
        scryfall_id: "004be4bc-ac3c-4026-8d36-f4687ab18c70",
        name: "Sokka's Haiku".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::TargetClause {
                requirement: TargetRequirement {
                    spec: TargetSpec::Spell,
                    min: 1,
                    max: 1,
                    another: false,
                },
                effects: vec![Effect::CounterSpell {
                    target: TargetSpec::Spell,
                }],
            },
            Effect::DrawCards { count: 1 },
            Effect::MillCards {
                count: 3,
                target: None,
            },
            Effect::TargetClause {
                requirement: TargetRequirement {
                    spec: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Land),
                            ..Default::default()
                        },
                    },
                    min: 1,
                    max: 1,
                    another: false,
                },
                effects: vec![Effect::UntapTarget {
                    target: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Land),
                            ..Default::default()
                        },
                    },
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn soldier() -> CardDefinition {
    let text_box = "Firebending 1 (Whenever this token attacks, add {R}. This mana lasts until end of combat.)";

    CardDefinition {
        key: "token.soldier",
        scryfall_id: "2de43b03-9ac5-4292-ab29-2dc6210ef3d9",
        name: "Soldier".into(),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Soldier".into()],
        power: Some(2),
        toughness: Some(2),
        is_token: true,
        color_override: Some(Colors::from([Color::Red])),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("R"),
                until_end_of_combat: true,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn solstice_revelations() -> CardDefinition {
    let text_box = "Exile cards from the top of your library until you exile a nonland card. You may cast that card without paying its mana cost if the spell's mana value is less than the number of Mountains you control. If you don't cast that card this way, put it into your hand.\nFlashback {6}{R} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";

    CardDefinition {
        key: "tla.solstice_revelations",
        scryfall_id: "f22ac19f-66fb-4d54-9f09-495a20a29577",
        name: "Solstice Revelations".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::ExileUntilNonlandCastFree {
            below: EffectValue::BattlefieldMatching {
                predicate: CardPredicate {
                    subtype: Some("Mountain".into()),
                    ..Default::default()
                },
            },
        }],
        flashback: Some(ManaCost::parse("{6}{R}")),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn south_pole_voyager() -> CardDefinition {
    let text_box = "Whenever this creature or another Ally you control enters, you gain 1 life. If this is the second time this ability has resolved this turn, draw a card.";

    CardDefinition {
        key: "tla.south_pole_voyager",
        scryfall_id: "4b5ad895-be8d-476b-91ca-22fad7a3cc58",
        name: "South Pole Voyager".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Scout".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    subtype: Some("Ally".into()),
                    ..Default::default()
                }),
            },
            effects: vec![
                Effect::GainLife { amount: 1 },
                Effect::OnNthResolutionThisTurn {
                    n: 2,
                    effect: Box::new(Effect::DrawCards { count: 1 }),
                },
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sozins_comet() -> CardDefinition {
    let text_box = "Each creature you control gains firebending 5 until end of turn. (Whenever it attacks, add {R}{R}{R}{R}{R}. This mana lasts until end of combat.)\nForetell {2}{R} (During your turn, you may pay {2} and exile this card from your hand face down. Cast it on a later turn for its foretell cost.)";

    CardDefinition {
        key: "tla.sozins_comet",
        scryfall_id: "649e50e5-299b-4191-87a8-36e9378795be",
        name: "Sozin's Comet".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::GrantFirebendingUntilEot { amount: 5 }],
        foretell: Some(ManaCost::parse("{2}{R}")),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sparring_dummy() -> CardDefinition {
    let text_box = "Defender\n{T}: Mill a card. You may put a land card milled this way into your hand. You gain 2 life if a Lesson card is milled this way. (To mill a card, put the top card of your library into your graveyard.)";

    CardDefinition {
        key: "tla.sparring_dummy",
        scryfall_id: "520f5a03-ff5a-43ed-8f15-cf24596d60cd",
        name: "Sparring Dummy".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature, CardType::Artifact]),
        subtypes: vec!["Scarecrow".into()],
        power: Some(1),
        toughness: Some(3),
        keywords: Keywords {
            defender: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::MillAndSelect {
                count: 1,
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                },
                if_declined: vec![],
                if_milled: Some(MilledBonus {
                    predicate: CardPredicate {
                        subtype: Some("Lesson".into()),
                        ..Default::default()
                    },
                    effects: vec![Effect::GainLife { amount: 2 }],
                }),
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn spirit() -> CardDefinition {
    let text_box = "This token can't block or be blocked by non-Spirit creatures.";

    CardDefinition {
        key: "token.spirit",
        scryfall_id: "f59eba51-458a-40e0-b754-999f91d5d839",
        name: "Spirit".into(),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Spirit".into()],
        power: Some(1),
        toughness: Some(1),
        is_token: true,
        color_override: Some(Colors::from([])),
        blocking_restriction: Some(CardPredicate {
            not_subtypes: vec!["Spirit".into()],
            ..Default::default()
        }),
        block_restriction: Some(CardPredicate {
            not_subtypes: vec!["Spirit".into()],
            ..Default::default()
        }),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn spirit_water_revival() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, you may waterbend {6}. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)\nDraw two cards. If this spell's additional cost was paid, instead shuffle your graveyard into your library, draw seven cards, and you have no maximum hand size for the rest of the game.\nExile Spirit Water Revival.";

    CardDefinition {
        key: "tla.spirit_water_revival",
        scryfall_id: "0c019e76-c88e-4d1b-a546-0f4e462ef44a",
        name: "Spirit Water Revival".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::IfKicked {
                then: vec![
                    Effect::ShuffleGraveyardIntoLibrary,
                    Effect::DrawCards { count: 7 },
                    Effect::NoMaximumHandSize,
                ],
                otherwise: vec![Effect::DrawCards { count: 2 }],
            },
            Effect::ExileSourceSpell,
        ],
        kicker: Some(ManaCost::parse("{6}")),
        kicker_waterbend: true,
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn suki_kyoshi_warrior() -> CardDefinition {
    let text_box = "Suki's power is equal to the number of creatures you control.\nWhenever Suki attacks, create a 1/1 white Ally creature token that's tapped and attacking.";

    CardDefinition {
        key: "tla.suki_kyoshi_warrior",
        scryfall_id: "3e311480-04f8-4b8f-938e-1cc6d06b9902",
        name: "Suki, Kyoshi Warrior".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        toughness: Some(4),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: true,
            }],
        }],
        power_cda: Some(PowerCda::CreaturesYouControl),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sun_blessed_peak() -> CardDefinition {
    let text_box = "This land enters tapped.\n{T}: Add {R} or {W}.\n{4}, {T}, Sacrifice this land: Draw a card.";

    CardDefinition {
        key: "tla.sun_blessed_peak",
        scryfall_id: "ae04c862-095f-41d2-8aef-21036d5a0cdc",
        name: "Sun-Blessed Peak".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("R"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("W"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn team_avatar() -> CardDefinition {
    let text_box = "Whenever a creature you control attacks alone, it gets +X/+X until end of turn, where X is the number of creatures you control.\n{2}{W}, Discard this card: It deals damage equal to the number of creatures you control to target creature.";

    CardDefinition {
        key: "tla.team_avatar",
        scryfall_id: "a9a5f6f7-f04d-477f-b67f-fd01a5dcc0f5",
        name: "Team Avatar".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::AttacksAlone {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::BuffTriggeringPermanentValue {
                power: EffectValue::BattlefieldMatching {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
                toughness: EffectValue::BattlefieldMatching {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            zone: ActivationZone::Hand,
            discard_source: true,
            mana_cost: ManaCost::parse("{2}{W}"),
            effects: vec![Effect::DealDamageValue {
                amount: EffectValue::BattlefieldMatching {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
                target: TargetSpec::Creature,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn tiger_dillo() -> CardDefinition {
    let text_box = "This creature can't attack or block unless you control another creature with power 4 or greater.";

    CardDefinition {
        key: "tla.tiger_dillo",
        scryfall_id: "067bd117-04e6-410e-89c4-6d431d627751",
        name: "Tiger-Dillo".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Cat".into(), "Armadillo".into()],
        power: Some(4),
        toughness: Some(3),
        combat_requires_other: Some(CardPredicate {
            min_power: Some(4),
            card_type: Some(CardType::Creature),
            ..Default::default()
        }),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn tiger_seal() -> CardDefinition {
    let text_box = "Vigilance\nAt the beginning of your upkeep, tap this creature.\nWhenever you draw your second card each turn, untap this creature.";

    CardDefinition {
        key: "tla.tiger_seal",
        scryfall_id: "7aef2891-1269-4a1a-acea-0aa37ef544c4",
        name: "Tiger-Seal".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Cat".into(), "Seal".into()],
        power: Some(3),
        toughness: Some(3),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
                effects: vec![Effect::UntapSource],
            },
            Ability::Triggered {
                condition: TriggerCondition::BeginningOfYourUpkeep,
                effects: vec![Effect::TapSource],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn tolls_of_war() -> CardDefinition {
    let text_box = "When this enchantment enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you sacrifice a permanent during your turn, create a 1/1 white Ally creature token. This ability triggers only once each turn.";

    CardDefinition {
        key: "tla.tolls_of_war",
        scryfall_id: "d822b9ad-a787-4355-88c2-07ae2bd3a78e",
        name: "Tolls of War".into(),
        mana_cost: Some(ManaCost::parse("{W}{B}")),
        types: CardTypes::new([CardType::Enchantment]),
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
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn toph_the_blind_bandit() -> CardDefinition {
    let text_box = "When Toph enters, earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)\nToph's power is equal to the number of +1/+1 counters on lands you control.";

    CardDefinition {
        key: "tla.toph_the_blind_bandit",
        scryfall_id: "ff68fa7b-8065-407b-a8b4-bfbb14f1c99c",
        name: "Toph, the Blind Bandit".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(0),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(2),
                target: TargetSpec::LandYouControl,
            }],
        }],
        power_cda: Some(PowerCda::CountersOnLandsYouControl),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn treetop_freedom_fighters() -> CardDefinition {
    let text_box = "Haste\nWhen this creature enters, create a 1/1 white Ally creature token.";

    CardDefinition {
        key: "tla.treetop_freedom_fighters",
        scryfall_id: "a9394200-7ffc-440f-8b5f-c08b7930133c",
        name: "Treetop Freedom Fighters".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(1),
        keywords: Keywords {
            haste: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn true_ancestry() -> CardDefinition {
    let text_box = "Return up to one target permanent card from your graveyard to your hand.\nCreate a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.true_ancestry",
        scryfall_id: "8c55b333-dc8b-4332-895b-eec5eb45543f",
        name: "True Ancestry".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::TargetClause {
                requirement: TargetRequirement {
                    spec: TargetSpec::GraveyardCardYouOwn {
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
                    },
                    min: 0,
                    max: 1,
                    another: false,
                },
                effects: vec![Effect::ReturnGraveyardTarget {
                    target: TargetSpec::GraveyardCardYouOwn {
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
                    },
                    tapped_onto_battlefield: false,
                }],
            },
            Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn uncle_iroh() -> CardDefinition {
    let text_box = "Firebending 1 (Whenever this creature attacks, add {R}. This mana lasts until end of combat.)\nLesson spells you cast cost {1} less to cast.";

    CardDefinition {
        key: "tla.uncle_iroh",
        scryfall_id: "5c88672c-0eba-4e93-a0ec-30bbb1bb7661",
        name: "Uncle Iroh".into(),
        mana_cost: Some(ManaCost::parse("{1}{R/G}{R/G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into(), "Ally".into()],
        power: Some(4),
        toughness: Some(2),
        spell_cost_reductions: vec![SpellCostReduction {
            first_on_own_turn: false,
            predicate: CardPredicate {
                subtype: Some("Lesson".into()),
                ..Default::default()
            },
            condition: StaticCondition::Always,
            generic: 1,
        }],
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("R"),
                until_end_of_combat: true,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn united_front() -> CardDefinition {
    let text_box = "Create X 1/1 white Ally creature tokens, then put a +1/+1 counter on each creature you control.";

    CardDefinition {
        key: "tla.united_front",
        scryfall_id: "d347caf8-0c12-401f-9e33-9978cb347f89",
        name: "United Front".into(),
        mana_cost: Some(ManaCost::parse("{X}{W}{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::CreateTokensValue {
                token_name: "Ally".into(),
                count: EffectValue::X,
            },
            Effect::PutCountersOnEachMatching {
                count: 1,
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                other: false,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn unlucky_cabbage_merchant() -> CardDefinition {
    let text_box = "When this creature enters, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nWhenever you sacrifice a Food, you may search your library for a basic land card and put it onto the battlefield tapped. If you search your library this way, put this creature on the bottom of its owner's library, then shuffle.";

    CardDefinition {
        key: "tla.unlucky_cabbage_merchant",
        scryfall_id: "0a838009-b115-4497-b35d-682e41ead7db",
        name: "Unlucky Cabbage Merchant".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
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
                    predicate: CardPredicate {
                        subtype: Some("Food".into()),
                        ..Default::default()
                    },
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
                            to_hand: false,
                        }],
                        vec![],
                    ],
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn walltop_sentries() -> CardDefinition {
    let text_box = "Reach, deathtouch\nWhen this creature dies, if there's a Lesson card in your graveyard, you gain 2 life.";

    CardDefinition {
        key: "tla.walltop_sentries",
        scryfall_id: "e55f1c63-58f6-4e2b-aaeb-f2d5faca47a2",
        name: "Walltop Sentries".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(3),
        keywords: Keywords {
            reach: true,
            deathtouch: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::ActiveIf {
                active_if: StaticCondition::GraveyardAtLeast {
                    count: 1,
                    predicate: CardPredicate {
                        subtype: Some("Lesson".into()),
                        ..Default::default()
                    },
                },
                condition: Box::new(TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                }),
            },
            effects: vec![Effect::IfGraveyardAtLeast {
                count: 1,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
                then: vec![Effect::GainLife { amount: 2 }],
                otherwise: vec![],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn wandering_musicians() -> CardDefinition {
    let text_box =
        "Whenever this creature attacks, creatures you control get +1/+0 until end of turn.";

    CardDefinition {
        key: "tla.wandering_musicians",
        scryfall_id: "3da81dbd-dc80-47ff-a33b-270d23a24e0a",
        name: "Wandering Musicians".into(),
        mana_cost: Some(ManaCost::parse("{3}{R/W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Bard".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(5),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::ModifyControlled {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                power: 1,
                toughness: 0,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn wartime_protestors() -> CardDefinition {
    let text_box = "Haste\nWhenever another Ally you control enters, put a +1/+1 counter on that creature and it gains haste until end of turn.";

    CardDefinition {
        key: "tla.wartime_protestors",
        scryfall_id: "bac81940-d717-49ff-83b2-16a22bb2c988",
        name: "Wartime Protestors".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Rebel".into(), "Ally".into()],
        power: Some(4),
        toughness: Some(4),
        keywords: Keywords {
            haste: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnotherYouControl(CardPredicate {
                    subtype: Some("Ally".into()),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::ModifyTriggeringPermanent {
                counters: 1,
                keywords: Keywords {
                    haste: true,
                    ..Default::default()
                },
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn water_tribe_captain() -> CardDefinition {
    let text_box = "{5}: Creatures you control get +1/+1 until end of turn.";

    CardDefinition {
        key: "tla.water_tribe_captain",
        scryfall_id: "0a0cdf97-1927-47c6-8ef1-29969e3567ee",
        name: "Water Tribe Captain".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(3),
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{5}"),
            effects: vec![Effect::ModifyControlled {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                power: 1,
                toughness: 1,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn water_tribe_rallier() -> CardDefinition {
    let text_box = "Waterbend {5}: Look at the top four cards of your library. You may reveal a creature card with power 3 or less from among them and put it into your hand. Put the rest on the bottom of your library in a random order. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.water_tribe_rallier",
        scryfall_id: "4e744b6c-1c2f-451a-818a-5ee7785b5213",
        name: "Water Tribe Rallier".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{5}"),
            waterbend: true,
            effects: vec![Effect::LookAndSelect {
                reveal: true,
                look: EffectValue::Fixed(4),
                min_select: 0,
                max_select: 1,
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    max_power: Some(3),
                    ..Default::default()
                },
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn waterbending_lesson() -> CardDefinition {
    let text_box = "Draw three cards. Then discard a card unless you waterbend {2}. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.waterbending_lesson",
        scryfall_id: "4f81e3b5-a0a9-4764-8c2d-b499ce1740b4",
        name: "Waterbending Lesson".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::DrawCards { count: 3 },
            Effect::OptionalPayment {
                cost: PaymentCost::Waterbend { generic: 2 },
                if_paid: vec![],
                if_declined: vec![Effect::Discard],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn waterbending_scroll() -> CardDefinition {
    let text_box = "{6}, {T}: Draw a card. This ability costs {1} less to activate for each Island you control.";

    CardDefinition {
        key: "tla.waterbending_scroll",
        scryfall_id: "50d0be73-a2c8-44a0-9178-9949f342f6f9",
        name: "Waterbending Scroll".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Artifact]),
        activated_abilities: vec![ActivatedAbilityDefinition {
            cost_reduction_per: Some(CardPredicate {
                subtype: Some("Island".into()),
                ..Default::default()
            }),
            tap_source: true,
            mana_cost: ManaCost::parse("{6}"),
            effects: vec![Effect::DrawCards { count: 1 }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn white_lotus_hideout() -> CardDefinition {
    let text_box = "{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast a Lesson or Shrine spell.\n{1}, {T}: Add one mana of any color.";

    CardDefinition {
        key: "tla.white_lotus_hideout",
        scryfall_id: "24d1e22c-0d99-4ed5-94f0-fd055fd8e2be",
        name: "White Lotus Hideout".into(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("1"),
        }],
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson", "Shrine"]),
                                [1, 0, 0, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson", "Shrine"]),
                                [0, 1, 0, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson", "Shrine"]),
                                [0, 0, 1, 0, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson", "Shrine"]),
                                [0, 0, 0, 1, 0, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::restricted_to(
                                ManaUse::spell(["Lesson", "Shrine"]),
                                [0, 0, 0, 0, 1, 0],
                            ),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{1}"),
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana::parse("W"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("U"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("B"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("R"),
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana::parse("G"),
                            until_end_of_combat: false,
                        }],
                    ],
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn white_lotus_reinforcements() -> CardDefinition {
    let text_box = "Vigilance\nOther Allies you control get +1/+1.";

    CardDefinition {
        key: "tla.white_lotus_reinforcements",
        scryfall_id: "4c7e3e2f-d7ac-4d40-ab84-cb7efe06a316",
        name: "White Lotus Reinforcements".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(3),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        static_pt_buffs: vec![StaticPtBuff {
            scope: StaticScope::OtherYouControl(CardPredicate {
                subtype: Some("Ally".into()),
                ..Default::default()
            }),
            condition: None,
            power: 1,
            toughness: 1,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn yip_yip() -> CardDefinition {
    let text_box = "Target creature you control gets +2/+2 until end of turn. If that creature is an Ally, it also gains flying until end of turn.";

    CardDefinition {
        key: "tla.yip_yip",
        scryfall_id: "43f4b10e-165b-4100-82f3-728e1b0c78ed",
        name: "Yip Yip!".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::BuffTarget {
                power: 2,
                toughness: 2,
                target: TargetSpec::CreatureYouControl,
            },
            Effect::IfTargetMatches {
                predicate: CardPredicate {
                    subtype: Some("Ally".into()),
                    ..Default::default()
                },
                then: vec![Effect::GrantKeywordsToTarget {
                    keywords: Keywords {
                        flying: true,
                        ..Default::default()
                    },
                    target: TargetSpec::CreatureYouControl,
                }],
            },
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::CreatureYouControl,
            min: 1,
            max: 1,
            another: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn yue_the_moon_spirit() -> CardDefinition {
    let text_box = "Flying, vigilance\nWaterbend {5}, {T}: You may cast a noncreature spell from your hand without paying its mana cost. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.yue_the_moon_spirit",
        scryfall_id: "ecdac50f-c639-43fc-a03e-d488fac96ae2",
        name: "Yue, the Moon Spirit".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Spirit".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(3),
        keywords: Keywords {
            flying: true,
            vigilance: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{5}"),
            waterbend: true,
            effects: vec![Effect::CastFromHandFree {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature, CardType::Land],
                    ..Default::default()
                },
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn yuyan_archers() -> CardDefinition {
    let text_box =
        "Reach\nWhen this creature enters, you may discard a card. If you do, draw a card.";

    CardDefinition {
        key: "tla.yuyan_archers",
        scryfall_id: "99244462-a996-4a5b-91fb-947045647d6d",
        name: "Yuyan Archers".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Archer".into()],
        power: Some(3),
        toughness: Some(1),
        keywords: Keywords {
            reach: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::OptionalDiscardDraw { count: 1, up_to: 1 }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn zuko_exiled_prince() -> CardDefinition {
    let text_box = "Firebending 3 (Whenever this creature attacks, add {R}{R}{R}. This mana lasts until end of combat.)\n{3}: Exile the top card of your library. You may play that card this turn.";

    CardDefinition {
        key: "tla.zuko_exiled_prince",
        scryfall_id: "6a73b372-9c0e-4a85-89d2-440163330687",
        name: "Zuko, Exiled Prince".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into()],
        power: Some(4),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("RRR"),
                until_end_of_combat: true,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}"),
            effects: vec![Effect::ExileTopCardPlayThisTurn],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn zuko_s_conviction() -> CardDefinition {
    let text_box = "Kicker {4} (You may pay an additional {4} as you cast this spell.)\nReturn target creature card from your graveyard to your hand. If this spell was kicked, instead put that card onto the battlefield tapped.";

    CardDefinition {
        key: "tla.zuko_s_conviction",
        scryfall_id: "ad8933d6-cdc7-4d60-a78e-b43ffecfb136",
        name: "Zuko's Conviction".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::IfKicked {
            then: vec![Effect::ReturnGraveyardTarget {
                target: TargetSpec::GraveyardCardYouOwn {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
                tapped_onto_battlefield: true,
            }],
            otherwise: vec![Effect::ReturnGraveyardTarget {
                target: TargetSpec::GraveyardCardYouOwn {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
                tapped_onto_battlefield: false,
            }],
        }],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::GraveyardCardYouOwn {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
            },
            min: 1,
            max: 1,
            another: false,
        }],
        kicker: Some(ManaCost::parse("{4}")),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn zuko_s_exile() -> CardDefinition {
    let text_box = "Exile target artifact, creature, or enchantment. Its controller creates a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.zuko_s_exile",
        scryfall_id: "9090b055-3406-4fed-a8c6-3f6353a9600e",
        name: "Zuko's Exile".into(),
        mana_cost: Some(ManaCost::parse("{5}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![Effect::ExileWithControllerToken {
            target: TargetSpec::Permanent {
                predicate: CardPredicate {
                    card_types_any: vec![
                        CardType::Artifact,
                        CardType::Creature,
                        CardType::Enchantment,
                    ],
                    ..Default::default()
                },
            },
            token_name: "Clue".into(),
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
