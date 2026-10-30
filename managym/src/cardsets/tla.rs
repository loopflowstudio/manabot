//! Current TLA cards.
//! Each card is one function returning its engine `CardDefinition`.
use crate::state::ability::AffectedSpec;
use crate::state::effect::Modification;
use crate::state::{
    ability::{
        Ability, Condition, Effect, EffectValue, LinkedChooser, LinkedGrant, MilledBonus,
        PaymentCost, PlayerSelector, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, ActivationZone, AdditionalCost, BlockRestriction,
        CardDefinition, CardType, CardTypes, CounterThreshold, Keywords, ManaAbility, PowerCda,
        ReplacementEffect, Restriction, SacrificeCost, SetCharacteristics, SpellCostReduction,
        StaticAbility, StaticKeywordGrant, StaticPtBuff, StaticRestriction, StaticScope,
        StaticTypeGrant, TargetCostReduction, TriggeredManaAbility,
    },
    mana::{Color, Mana, ManaCost, ManaUse},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

/// Defined cards whose behavior is incomplete. The fixed setups that name them
/// still run them; the world does not support them for deck entry until their
/// batch completes the missing behavior.
pub(super) const INCOMPLETE: &[super::Incomplete] = &[
    ("tla.accumulate_wisdom", "Controller must order the library remainder (CR 401.4); it is randomized"),
    ("tla.cat_gator", "Any target must include planeswalkers and battles (CR 115.4)"),
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
        abandoned_air_temple(),
        accumulate_wisdom(),
        agna_qel_a(),
        air_nomad_legacy(),
        airbender_s_reversal(),
        airbending_lesson(),
        airship_engine_room(),
        allies_at_last(),
        appa_steadfast_guardian(),
        avatar_enthusiasts(),
        avatar_s_wrath(),
        azula_always_lies(),
        azula_cunning_usurper(),
        azula_on_the_hunt(),
        ba_sing_se(),
        badgermole(),
        badgermole_cub(),
        barrels_of_blasting_jelly(),
        beetle_headed_merchants(),
        benevolent_river_spirit(),
        bitter_work(),
        boar_q_pine(),
        boiling_rock_prison(),
        boomerang_basics(),
        bumi_bash(),
        callous_inspector(),
        canyon_crawler(),
        cat_gator(),
        cat_owl(),
        combustion_technique(),
        compassionate_healer(),
        crashing_wave(),
        crescent_island_temple(),
        cruel_administrator(),
        cunning_maneuver(),
        curious_farm_animals(),
        cycle_of_renewal(),
        dai_li_agents(),
        dai_li_indoctrination(),
        day_of_black_sun(),
        deadly_precision(),
        deserter_s_disciple(),
        diligent_zookeeper(),
        dragonfly_swarm(),
        earth_kingdom_jailer(),
        earth_kingdom_protectors(),
        earth_kingdom_soldier(),
        earth_kings_lieutenant(),
        earth_rumble_wrestlers(),
        earth_village_ruffians(),
        earthbending_lesson(),
        earthen_ally(),
        elemental_teachings(),
        energybending(),
        enter_the_avatar_state(),
        fancy_footwork(),
        fated_firepower(),
        fire_lord_zuko(),
        fire_nation_attacks(),
        fire_nation_cadets(),
        fire_nation_engineer(),
        fire_nation_raider(),
        fire_nation_palace(),
        fire_nation_warship(),
        fire_sages(),
        firebending_lesson(),
        firebending_student(),
        first_time_flyer(),
        flexible_waterbender(),
        flopsie_bumi_s_buddy(),
        foggy_bottom_swamp(),
        foggy_swamp_hunters(),
        foggy_swamp_spirit_keeper(),
        foggy_swamp_vinebender(),
        forecasting_fortune_teller(),
        gather_the_white_lotus(),
        geyser_leaper(),
        giant_koi(),
        glider_kids(),
        glider_staff(),
        gran_gran(),
        great_divide_guide(),
        hakoda_selfless_commander(),
        hama_the_bloodbender(),
        haru_hidden_talent(),
        heartless_act(),
        hermitic_herbalist(),
        hog_monkey(),
        honest_work(),
        how_to_start_a_riot(),
        iguana_parrot(),
        invasion_reinforcements(),
        invasion_submersible(),
        invasion_tactics(),
        iroh_grand_lotus(),
        iroh_s_demonstration(),
        itll_quench_ya(),
        jasmine_dragon_tea_shop(),
        jeong_jeongs_deserters(),
        jet_freedom_fighter(),
        june_bounty_hunter(),
        katara_water_tribe_s_hope(),
        knowledge_seeker(),
        kyoshi_battle_fan(),
        kyoshi_island_plaza(),
        kyoshi_village(),
        kyoshi_warriors(),
        lo_and_li_twin_tutors(),
        lost_days(),
        mai_jaded_edge(),
        mai_scornful_striker(),
        master_pakku(),
        master_piandao(),
        meditation_pools(),
        merchant_of_many_hats(),
        messenger_hawk(),
        meteor_sword(),
        misty_palms_oasis(),
        momo_friendly_flier(),
        mongoose_lizard(),
        north_pole_gates(),
        north_pole_patrol(),
        northern_air_temple(),
        obsessive_pursuit(),
        octopus_form(),
        omashu_city(),
        origin_of_metalbending(),
        ostrich_horse(),
        otter_penguin(),
        ozai_s_cruelty(),
        ozai_the_phoenix_king(),
        path_to_redemption(),
        pillar_launch(),
        pirate_peddlers(),
        platypus_bear(),
        pretending_poxbearers(),
        professor_zei_anthropologist(),
        rabaroo_troop(),
        ran_and_shaw(),
        raucous_audience(),
        razor_rings(),
        realm_of_koh(),
        rebellious_captives(),
        rockalanche(),
        rocky_rebuke(),
        rough_rhino_cavalry(),
        rowdy_snowballers(),
        rumble_arena(),
        saber_tooth_moose_lion(),
        sandbenders_storm(),
        secret_tunnel(),
        seismic_sense(),
        serpent_of_the_pass(),
        serpents_pass(),
        shared_roots(),
        sold_out(),
        sokka_bold_boomeranger(),
        sokka_lateral_strategist(),
        sokka_s_haiku(),
        solstice_revelations(),
        south_pole_voyager(),
        southern_air_temple(),
        sozins_comet(),
        sparring_dummy(),
        spirit_water_revival(),
        suki_kyoshi_warrior(),
        sun_blessed_peak(),
        sun_warriors(),
        swampsnare_trap(),
        team_avatar(),
        the_boulder_ready_to_rumble(),
        the_earth_king(),
        the_last_agni_kai(),
        the_lion_turtle(),
        the_mechanist_aerial_artisan(),
        the_spirit_oasis(),
        the_walls_of_ba_sing_se(),
        tiger_dillo(),
        tiger_seal(),
        tolls_of_war(),
        toph_the_blind_bandit(),
        toph_the_first_metalbender(),
        treetop_freedom_fighters(),
        true_ancestry(),
        trusty_boomerang(),
        tundra_tank(),
        turtle_duck(),
        twin_blades(),
        uncle_iroh(),
        united_front(),
        unlucky_cabbage_merchant(),
        vengeful_villagers(),
        vindictive_warden(),
        walltop_sentries(),
        wan_shi_tong_librarian(),
        wandering_musicians(),
        war_balloon(),
        wartime_protestors(),
        water_tribe_captain(),
        water_tribe_rallier(),
        waterbender_ascension(),
        waterbending_lesson(),
        waterbending_scroll(),
        watery_grasp(),
        white_lotus_hideout(),
        white_lotus_reinforcements(),
        white_lotus_tile(),
        yip_yip(),
        yue_the_moon_spirit(),
        yuyan_archers(),
        zhao_ruthless_admiral(),
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    shares_creature_type: false,
                    spec: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    },
                    min: 0,
                    max: 1,
                    another: true,
                    max_x: false,
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
            Effect::If {
                condition: Condition::Kicked,
                then: vec![
                    Effect::SearchBasicLand {
                        subtype: None,
                        count: EffectValue::Fixed(1),
                        tapped: true,
                        untap_if_lands: None,
                        bottom_source: false,
                        to_hand: true,
                    },
                    Effect::SearchBasicLand {
                        subtype: Some("Shrine".into()),
                        count: EffectValue::Fixed(1),
                        tapped: true,
                        untap_if_lands: None,
                        bottom_source: false,
                        to_hand: true,
                    },
                ],
                otherwise: vec![Effect::SearchBasicLand {
                    subtype: None,
                    count: EffectValue::Fixed(1),
                    tapped: true,
                    untap_if_lands: None,
                    bottom_source: false,
                    to_hand: true,
                }],
            },
            Effect::GainLife { amount: 2 },
        ],
        additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{2}"))],
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
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        shares_creature_type: false,
                        spec: TargetSpec::Permanent {
                            predicate: CardPredicate {
                                not_card_types: vec![CardType::Land],
                                ..Default::default()
                            },
                        },
                        min: 0,
                        max: 1,
                        another: true,
                        max_x: false,
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
                intervening_if: None,
                condition: TriggerCondition::YouCastSpell {
                    predicate: CardPredicate {
                        subtype: Some("Lesson".into()),
                        ..Default::default()
                    },
                },
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::AddKeywords(Keywords {
                        lifelink: true,
                        ..Default::default()
                    }),
                )],
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

fn abandoned_air_temple() -> CardDefinition {
    let text_box = "This land enters tapped unless you control a basic land.\n{T}: Add {W}.\n{3}{W}, {T}: Put a +1/+1 counter on each creature you control.";

    CardDefinition {
        key: "tla.abandoned_air_temple",
        scryfall_id: "9c0433f9-8f1e-4a19-a83f-a41925f1b1a9",
        name: "Abandoned Air Temple".into(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("W"),
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{3}{W}"),
            effects: vec![Effect::PutCountersOnEachMatching {
                count: EffectValue::Fixed(1),
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                other: false,
            }],
            ..Default::default()
        }],
        replacement_effects: vec![ReplacementEffect::EntersTappedUnlessControls {
            predicate: CardPredicate {
                supertype: Some("basic".into()),
                card_type: Some(CardType::Land),
                ..Default::default()
            },
        }],
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
        spell_effects: vec![Effect::If {
            condition: Condition::GraveyardAtLeast {
                count: 3,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
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

fn agna_qel_a() -> CardDefinition {
    let text_box = "This land enters tapped unless you control a basic land.\n{T}: Add {U}.\n{2}{U}, {T}: Draw a card, then discard a card.";

    CardDefinition {
        key: "tla.agna_qel_a",
        scryfall_id: "6b885829-a323-4f7d-87c9-aa4615dcbe5c",
        name: "Agna Qel'a".into(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("U"),
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{2}{U}"),
            effects: vec![
                Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                },
                Effect::Discard,
            ],
            ..Default::default()
        }],
        replacement_effects: vec![ReplacementEffect::EntersTappedUnlessControls {
            predicate: CardPredicate {
                supertype: Some("basic".into()),
                card_type: Some(CardType::Land),
                ..Default::default()
            },
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn air_nomad_legacy() -> CardDefinition {
    let text_box = "When this enchantment enters, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nCreatures you control with flying get +1/+1.";

    CardDefinition {
        key: "tla.air_nomad_legacy",
        scryfall_id: "10874416-fa4c-4aa3-b885-4351d011d208",
        name: "Air Nomad Legacy".into(),
        mana_cost: Some(ManaCost::parse("{W}{U}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::YouControl(CardPredicate {
                flying: true,
                card_type: Some(CardType::Creature),
                ..Default::default()
            }),
            condition: None,
            power: 1,
            toughness: 1,
            per_creature_type_max: None,
            per_color_among: None,
        })],
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
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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
                shares_creature_type: false,
                spec: TargetSpec::CreatureYouControl,
                min: 0,
                max: 2,
                another: false,
                max_x: false,
            },
            TargetRequirement {
                shares_creature_type: false,
                spec: TargetSpec::CreatureOpponentControls,
                min: 1,
                max: 1,
                another: false,
                max_x: false,
            },
        ],
        statics: vec![StaticAbility::CostsLessPerPermanent(CardPredicate {
            subtype: Some("Ally".into()),
            ..Default::default()
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn appa_steadfast_guardian() -> CardDefinition {
    let text_box = "Flash\nFlying\nWhen Appa enters, airbend any number of other target nonland permanents you control. (Exile them. While each one is exiled, its owner may cast it for {2} rather than its mana cost.)\nWhenever you cast a spell from exile, create a 1/1 white Ally creature token.";
    let nonland_you_control = TargetSpec::PermanentYouControl {
        predicate: CardPredicate {
            not_card_types: vec![CardType::Land],
            ..Default::default()
        },
    };

    CardDefinition {
        key: "tla.appa_steadfast_guardian",
        scryfall_id: "829d91e9-4878-4e55-a262-ac0d55b65d4e",
        name: "Appa, Steadfast Guardian".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}{W}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Bison".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(4),
        keywords: Keywords {
            flash: true,
            flying: true,
            ..Default::default()
        },
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        shares_creature_type: false,
                        spec: nonland_you_control.clone(),
                        min: 0,
                        max: TargetRequirement::ANY_NUMBER,
                        another: true,
                        max_x: false,
                    },
                    effects: vec![Effect::ForEachTarget {
                        effects: vec![Effect::Airbend {
                            target: nonland_you_control,
                        }],
                    }],
                }],
            },
            Ability::Triggered {
                condition: TriggerCondition::FromZone {
                    zone: ZoneType::Exile,
                    condition: Box::new(TriggerCondition::YouCastSpell {
                        predicate: CardPredicate::default(),
                    }),
                },
                intervening_if: None,
                effects: vec![Effect::CreateToken {
                    token_name: "Ally".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                }],
            },
        ],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnotherYouControl(CardPredicate {
                    subtype: Some("Ally".into()),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::PutCountersOnSource {
                count: EffectValue::Fixed(1),
                kind: CounterKind::PlusOnePlusOne,
            }],
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
                    shares_creature_type: false,
                    spec: TargetSpec::Creature,
                    min: 0,
                    max: 1,
                    another: false,
                    max_x: false,
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
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 0,
            max: 1,
            another: false,
            max_x: false,
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
                        shares_creature_type: false,
                        spec: TargetSpec::Creature,
                        min: 1,
                        max: 1,
                        another: false,
                        max_x: false,
                    },
                    effects: vec![Effect::until_end_of_turn(
                        AffectedSpec::Target(TargetSpec::Creature),
                        Modification::pump(-1, -1),
                    )],
                }],
                vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        shares_creature_type: false,
                        spec: TargetSpec::Creature,
                        min: 1,
                        max: 1,
                        another: false,
                        max_x: false,
                    },
                    effects: vec![Effect::PutCounters {
                        count: 1,
                        kind: CounterKind::PlusOnePlusOne,
                        target: TargetSpec::Creature,
                    }],
                }],
                vec![
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            shares_creature_type: false,
                            spec: TargetSpec::Creature,
                            min: 1,
                            max: 1,
                            another: false,
                            max_x: false,
                        },
                        effects: vec![Effect::until_end_of_turn(
                            AffectedSpec::Target(TargetSpec::Creature),
                            Modification::pump(-1, -1),
                        )],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            shares_creature_type: false,
                            spec: TargetSpec::Creature,
                            min: 1,
                            max: 1,
                            another: false,
                            max_x: false,
                        },
                        effects: vec![Effect::PutCounters {
                            count: 1,
                            kind: CounterKind::PlusOnePlusOne,
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
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RR"),
                    until_end_of_combat: true,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
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

fn azula_on_the_hunt() -> CardDefinition {
    let text_box = "Firebending 2 (Whenever this creature attacks, add {R}{R}. This mana lasts until end of combat.)\nWhenever Azula attacks, you lose 1 life and create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.azula_on_the_hunt",
        scryfall_id: "1335a145-248a-4f1e-8760-9a5d531e14e3",
        name: "Azula, On the Hunt".into(),
        mana_cost: Some(ManaCost::parse("{3}{B}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into()],
        power: Some(4),
        toughness: Some(3),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RR"),
                    until_end_of_combat: true,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![
                    Effect::LoseLife {
                        player: PlayerSelector::You,
                        amount: EffectValue::Fixed(1),
                    },
                    Effect::CreateToken {
                        token_name: "Clue".into(),
                        count: 1,
                        tapped_and_attacking: false,
                        attach_source: false,
                    },
                ],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn ba_sing_se() -> CardDefinition {
    let text_box = "This land enters tapped unless you control a basic land.\n{T}: Add {G}.\n{2}{G}, {T}: Earthbend 2. Activate only as a sorcery. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.ba_sing_se",
        scryfall_id: "bdf3b2be-d0cd-4a3c-a10e-82d32c12d3bd",
        name: "Ba Sing Se".into(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("G"),
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            sorcery_timing: true,
            tap_source: true,
            mana_cost: ManaCost::parse("{2}{G}"),
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(2),
                target: TargetSpec::LandYouControl,
            }],
            ..Default::default()
        }],
        replacement_effects: vec![ReplacementEffect::EntersTappedUnlessControls {
            predicate: CardPredicate {
                supertype: Some("basic".into()),
                card_type: Some(CardType::Land),
                ..Default::default()
            },
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn badgermole() -> CardDefinition {
    let text_box = "When this creature enters, earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)\nCreatures you control with +1/+1 counters on them have trample.";

    CardDefinition {
        key: "tla.badgermole",
        scryfall_id: "aabbd420-b7fc-496f-9168-bad823d51d9e",
        name: "Badgermole".into(),
        mana_cost: Some(ManaCost::parse("{4}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Badger".into(), "Mole".into()],
        power: Some(4),
        toughness: Some(4),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(2),
                target: TargetSpec::LandYouControl,
            }],
        }],
        statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
            scope: StaticScope::YouControl(CardPredicate {
                with_counter: true,
                card_type: Some(CardType::Creature),
                ..Default::default()
            }),
            condition: None,
            keywords: Keywords {
                trample: true,
                ..Default::default()
            },
        })],
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
            intervening_if: None,
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
                effects: vec![one_mana_of_any_color(None)],
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

fn beetle_headed_merchants() -> CardDefinition {
    let text_box = "Whenever this creature attacks, you may sacrifice another creature or artifact. If you do, draw a card and put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "tla.beetle_headed_merchants",
        scryfall_id: "c2eed79f-38c1-4aac-9525-d54cb114f17f",
        name: "Beetle-Headed Merchants".into(),
        mana_cost: Some(ManaCost::parse("{4}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
        power: Some(5),
        toughness: Some(4),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            intervening_if: None,
            effects: vec![Effect::MaySacrifice {
                predicate: CardPredicate {
                    card_types_any: vec![CardType::Creature, CardType::Artifact],
                    ..Default::default()
                },
                another: true,
                then: vec![
                    Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    },
                    Effect::PutCountersOnSource {
                        count: EffectValue::Fixed(1),
                        kind: CounterKind::PlusOnePlusOne,
                    },
                ],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn benevolent_river_spirit() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, waterbend {5}. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)\nFlying, ward {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)\nWhen this creature enters, scry 2.";

    CardDefinition {
        key: "tla.benevolent_river_spirit",
        scryfall_id: "7ffb79cd-d170-4047-89c8-6e85188f30da",
        name: "Benevolent River Spirit".into(),
        mana_cost: Some(ManaCost::parse("{U}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Spirit".into()],
        power: Some(4),
        toughness: Some(5),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Scry { count: 2 }],
        }],
        statics: vec![StaticAbility::Ward(ManaCost::parse("{2}"))],
        additional_costs: vec![AdditionalCost::RequiredWaterbend(ManaCost::parse("{5}"))],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn bitter_work() -> CardDefinition {
    let text_box = "Whenever you attack a player with one or more creatures with power 4 or greater, draw a card.\nExhaust — {4}: Earthbend 4. Activate only during your turn. (Target land you control becomes a 0/0 creature with haste that's still a land. Put four +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped. Activate each exhaust ability only once.)";

    CardDefinition {
        key: "tla.bitter_work",
        scryfall_id: "3e8de6a9-9859-4ed2-8ece-ef80a7209be9",
        name: "Bitter Work".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}{G}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    card_type: Some(CardType::Creature),
                    min_power: Some(4),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            exhaust: true,
            own_turn_only: true,
            mana_cost: ManaCost::parse("{4}"),
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(4),
                target: TargetSpec::LandYouControl,
            }],
            ..Default::default()
        }],
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
            intervening_if: None,
            condition: TriggerCondition::YouCastSpell {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            effects: vec![Effect::PutCountersOnSource {
                count: EffectValue::Fixed(1),
                kind: CounterKind::PlusOnePlusOne,
            }],
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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

fn bumi_bash() -> CardDefinition {
    let text_box = "Choose one —\n• Bumi Bash deals damage equal to the number of lands you control to target creature.\n• Destroy target land creature or nonbasic land.";

    CardDefinition {
        key: "tla.bumi_bash",
        scryfall_id: "e8895479-26cb-4ee8-98ca-6d46c43f0dbd",
        name: "Bumi Bash".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::DealDamageValue {
                    amount: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Land),
                            ..Default::default()
                        },
                    },
                    target: TargetSpec::Creature,
                }],
                vec![Effect::Destroy {
                    target: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Land),
                            any_of: vec![
                                CardPredicate {
                                    card_type: Some(CardType::Creature),
                                    ..Default::default()
                                },
                                CardPredicate {
                                    nonbasic: true,
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        },
                    },
                }],
            ],
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
            intervening_if: None,
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![
                Effect::DealDamageToPlayer {
                    player: PlayerSelector::You,
                    amount: EffectValue::Fixed(1),
                },
                Effect::CreateToken {
                    token_name: "Clue".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Food".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            zone: ActivationZone::Hand,
            discard_source: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::SearchBasicLand {
                subtype: Some("Swamp".into()),
                count: EffectValue::Fixed(1),
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
            intervening_if: None,
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
            intervening_if: None,
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
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::ExileIfWouldDie,
            ),
        ],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
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
            intervening_if: None,
            condition: TriggerCondition::BecomesTapped {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }, Effect::Scry { count: 1 }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn crashing_wave() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, waterbend {X}. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)\nTap up to X target creatures, then distribute three stun counters among any number of tapped creatures your opponents control. (If a permanent with a stun counter would become untapped, remove one from it instead.)";

    CardDefinition {
        key: "tla.crashing_wave",
        scryfall_id: "9fd02eb4-1ef5-4a14-89a8-b25e720e8016",
        name: "Crashing Wave".into(),
        mana_cost: Some(ManaCost::parse("{U}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::ForEachTarget {
                effects: vec![Effect::TapTarget {
                    target: TargetSpec::Creature,
                }],
            },
            Effect::DistributeCountersToTappedOpponentCreatures {
                kind: CounterKind::Stun,
                count: 3,
            },
        ],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 0,
            max: 0,
            another: false,
            max_x: true,
        }],
        additional_costs: vec![AdditionalCost::RequiredWaterbend(ManaCost::parse("{X}"))],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn crescent_island_temple() -> CardDefinition {
    let text_box = "When Crescent Island Temple enters, for each Shrine you control, create a 1/1 red Monk creature token with prowess. (Whenever you cast a noncreature spell, it gets +1/+1 until end of turn.)\nWhenever another Shrine you control enters, create a 1/1 red Monk creature token with prowess.";

    CardDefinition {
        key: "tla.crescent_island_temple",
        scryfall_id: "04c85150-269e-4158-a1b9-bd11bc6b7d79",
        name: "Crescent Island Temple".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}")),
        types: CardTypes::new([CardType::Enchantment]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Shrine".into()],
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateTokensValue {
                    token_name: "Monk".into(),
                    count: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate {
                            subtype: Some("Shrine".into()),
                            ..Default::default()
                        },
                    },
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate {
                        subtype: Some("Shrine".into()),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Monk".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn cruel_administrator() -> CardDefinition {
    let text_box = "Raid — This creature enters with a +1/+1 counter on it if you attacked this turn.\nWhenever this creature attacks, create a 2/2 red Soldier creature token with firebending 1. (Whenever it attacks, add {R}. This mana lasts until end of combat.)";

    CardDefinition {
        key: "tla.cruel_administrator",
        scryfall_id: "9d70707b-2e25-40b1-80a8-9a5492bca7e2",
        name: "Cruel Administrator".into(),
        mana_cost: Some(ManaCost::parse("{3}{B}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(5),
        toughness: Some(4),
        replacement_effects: vec![ReplacementEffect::EntersWithPlusOneCounters {
            count: 1,
            condition: Some(Condition::AttackedThisTurn),
        }],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Soldier".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
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
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::pump(3, 1),
            ),
            Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            },
        ],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
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
            intervening_if: None,
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
                    shares_creature_type: false,
                    spec: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                            ..Default::default()
                        },
                    },
                    min: 0,
                    max: 1,
                    another: false,
                    max_x: false,
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
                count: EffectValue::Fixed(2),
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
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            shares_creature_type: false,
                            spec: TargetSpec::LandYouControl,
                            min: 1,
                            max: 1,
                            another: false,
                            max_x: false,
                        },
                        effects: vec![Effect::Earthbend {
                            count: EffectValue::Fixed(1),
                            target: TargetSpec::LandYouControl,
                        }],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            shares_creature_type: false,
                            spec: TargetSpec::LandYouControl,
                            min: 1,
                            max: 1,
                            another: false,
                            max_x: false,
                        },
                        effects: vec![Effect::Earthbend {
                            count: EffectValue::Fixed(1),
                            target: TargetSpec::LandYouControl,
                        }],
                    },
                ],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrainLife {
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

fn day_of_black_sun() -> CardDefinition {
    let text_box = "Each creature with mana value X or less loses all abilities until end of turn. Destroy those creatures.";

    CardDefinition {
        key: "tla.day_of_black_sun",
        scryfall_id: "d0e24797-3e45-4c2b-b4b0-1ef44d42eaee",
        name: "Day of Black Sun".into(),
        mana_cost: Some(ManaCost::parse("{X}{B}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::DestroyCreaturesLosingAbilities {
            max_mana_value: EffectValue::X,
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
        additional_costs: vec![AdditionalCost::Sacrifice(Box::new(SacrificeCost {
            predicate: CardPredicate {
                card_types_any: vec![CardType::Artifact, CardType::Creature],
                ..Default::default()
            },
            another: false,
            or_mana: Some(ManaCost::parse("{4}")),
        }))],
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
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::PermanentYouControl {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        max_power: Some(2),
                        ..Default::default()
                    },
                }),
                Modification::CantBeBlocked,
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn diligent_zookeeper() -> CardDefinition {
    let text_box = "Each non-Human creature you control gets +1/+1 for each of its creature types, to a maximum of 10.";

    CardDefinition {
        key: "tla.diligent_zookeeper",
        scryfall_id: "21f52564-9820-42dc-a08d-459d51afc397",
        name: "Diligent Zookeeper".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into(), "Ally".into()],
        power: Some(4),
        toughness: Some(4),
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::YouControl(CardPredicate {
                not_subtypes: vec!["Human".into()],
                card_type: Some(CardType::Creature),
                ..Default::default()
            }),
            condition: None,
            power: 1,
            toughness: 1,
            per_creature_type_max: Some(10),
            per_color_among: None,
        })],
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
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            intervening_if: Some(Condition::GraveyardAtLeast {
                count: 1,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            }),
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        }],
        statics: vec![
            StaticAbility::Ward(ManaCost::parse("{1}")),
            StaticAbility::CharacteristicPower(PowerCda::GraveyardMatching(CardPredicate {
                not_card_types: vec![CardType::Creature, CardType::Land],
                ..Default::default()
            })),
        ],
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
            intervening_if: None,
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
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::PermanentYouControl {
                    predicate: CardPredicate {
                        subtype: Some("Ally".into()),
                        ..Default::default()
                    },
                }),
                Modification::AddKeywords(Keywords {
                    indestructible: true,
                    ..Default::default()
                }),
            )],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    shares_creature_type: false,
                    spec: TargetSpec::CreatureYouControl,
                    min: 0,
                    max: 2,
                    another: false,
                    max_x: false,
                },
                effects: vec![Effect::ForEachTarget {
                    effects: vec![Effect::PutCounters {
                        count: 1,
                        kind: CounterKind::PlusOnePlusOne,
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
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate {
                        subtype: Some("Ally".into()),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(1),
                    kind: CounterKind::PlusOnePlusOne,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::PutCountersOnEachMatching {
                    count: EffectValue::Fixed(1),
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

fn earth_rumble_wrestlers() -> CardDefinition {
    let text_box = "Reach\nThis creature gets +1/+0 and has trample as long as you control a land creature or a land entered the battlefield under your control this turn.";
    let condition = || {
        Some(Condition::AnyOf(vec![
            Condition::ControlsMatching {
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    card_types_any: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            Condition::LandEnteredThisTurn,
        ]))
    };

    CardDefinition {
        key: "tla.earth_rumble_wrestlers",
        scryfall_id: "74fca0b8-12e1-405f-b999-160d65fe0ade",
        name: "Earth Rumble Wrestlers".into(),
        mana_cost: Some(ManaCost::parse("{3}{R/G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Performer".into()],
        power: Some(3),
        toughness: Some(4),
        keywords: Keywords {
            reach: true,
            ..Default::default()
        },
        statics: vec![
            StaticAbility::PtBuff(StaticPtBuff {
                scope: StaticScope::This,
                condition: condition(),
                power: 1,
                toughness: 0,
                per_creature_type_max: None,
                per_color_among: None,
            }),
            StaticAbility::KeywordGrant(StaticKeywordGrant {
                scope: StaticScope::This,
                condition: condition(),
                keywords: Keywords {
                    trample: true,
                    ..Default::default()
                },
            }),
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
            intervening_if: None,
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

fn earthen_ally() -> CardDefinition {
    let text_box = "This creature gets +1/+0 for each color among Allies you control.\n{2}{W}{U}{B}{R}{G}: Earthbend 5. (Target land you control becomes a 0/0 creature with haste that's still a land. Put five +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.earthen_ally",
        scryfall_id: "d7a0f2d9-efa4-4a53-a24b-36694457bd1c",
        name: "Earthen Ally".into(),
        mana_cost: Some(ManaCost::parse("{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into(), "Ally".into()],
        power: Some(0),
        toughness: Some(2),
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{2}{W}{U}{B}{R}{G}"),
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(5),
                target: TargetSpec::LandYouControl,
            }],
            ..Default::default()
        }],
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::This,
            condition: None,
            power: 1,
            toughness: 0,
            per_creature_type_max: None,
            per_color_among: Some(CardPredicate {
                subtype: Some("Ally".into()),
                ..Default::default()
            }),
        })],
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

fn energybending() -> CardDefinition {
    let text_box = "Lands you control gain all basic land types until end of turn.\nDraw a card.";

    CardDefinition {
        key: "tla.energybending",
        scryfall_id: "6c085441-e023-4032-89a0-d24ce5060ac3",
        name: "Energybending".into(),
        mana_cost: Some(ManaCost::parse("{2}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::until_end_of_turn(
                AffectedSpec::EachYouControl(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                Modification::AddSubtypes(vec![
                    "Plains".into(),
                    "Island".into(),
                    "Swamp".into(),
                    "Mountain".into(),
                    "Forest".into(),
                ]),
            ),
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
        ],
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
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::CreatureYouControl),
                Modification::AddSubtypes(vec!["Avatar".into()]),
            ),
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::CreatureYouControl),
                Modification::AddKeywords(Keywords {
                    flying: true,
                    first_strike: true,
                    lifelink: true,
                    hexproof: true,
                    ..Default::default()
                }),
            ),
        ],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::CreatureYouControl,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
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
                Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::Creature),
                    Modification::pump(2, 2),
                ),
            ],
        }],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 1,
            max: 2,
            another: false,
            max_x: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fated_firepower() -> CardDefinition {
    let text_box = "Flash\nThis enchantment enters with X fire counters on it.\nIf a source you control would deal damage to an opponent or a permanent an opponent controls, it deals that much damage plus an amount of damage equal to the number of fire counters on this enchantment instead.";

    CardDefinition {
        key: "tla.fated_firepower",
        scryfall_id: "51352127-1f86-42e9-b4ca-2fd58b14e86b",
        name: "Fated Firepower".into(),
        mana_cost: Some(ManaCost::parse("{X}{R}{R}{R}")),
        types: CardTypes::new([CardType::Enchantment]),
        keywords: Keywords {
            flash: true,
            ..Default::default()
        },
        replacement_effects: vec![
            ReplacementEffect::EntersWithXCounters {
                kind: CounterKind::Fire,
            },
            ReplacementEffect::OpponentDamagePlusCounters {
                kind: CounterKind::Fire,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_lord_zuko() -> CardDefinition {
    let text_box = "Firebending X, where X is Fire Lord Zuko's power. (Whenever this creature attacks, add X {R}. This mana lasts until end of combat.)\nWhenever you cast a spell from exile and whenever a permanent you control enters from exile, put a +1/+1 counter on each creature you control.";
    let counter_each_creature = || {
        vec![Effect::PutCountersOnEachMatching {
            count: EffectValue::Fixed(1),
            predicate: CardPredicate::creature(),
            other: false,
        }]
    };

    CardDefinition {
        key: "tla.fire_lord_zuko",
        scryfall_id: "e62d3bcc-7bb4-42be-90a9-caf3c1caa29d",
        name: "Fire Lord Zuko".into(),
        mana_cost: Some(ManaCost::parse("{R}{W}{B}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(4),
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::AddManaValue {
                    mana: Mana::parse("R"),
                    amount: EffectValue::SourcePower,
                    until_end_of_combat: true,
                }],
            },
            // One printed ability with two trigger events (CR 603.2): no
            // single event is both a cast and an entry from exile, so two
            // triggered abilities fire exactly when it does.
            Ability::Triggered {
                condition: TriggerCondition::FromZone {
                    zone: ZoneType::Exile,
                    condition: Box::new(TriggerCondition::YouCastSpell {
                        predicate: CardPredicate::default(),
                    }),
                },
                intervening_if: None,
                effects: counter_each_creature(),
            },
            Ability::Triggered {
                condition: TriggerCondition::FromZone {
                    zone: ZoneType::Exile,
                    condition: Box::new(TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::AnyYouControl(CardPredicate::default()),
                    }),
                },
                intervening_if: None,
                effects: counter_each_creature(),
            },
        ],
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
            attach_source: false,
        }],
        statics: vec![StaticAbility::Flashback(ManaCost::parse("{8}{R}"))],
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
            intervening_if: None,
            condition: TriggerCondition::ActiveIf {
                active_if: Condition::GraveyardAtLeast {
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
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::pump(1, 0),
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_engineer() -> CardDefinition {
    let text_box = "Raid — At the beginning of your end step, if you attacked this turn, put a +1/+1 counter on another target creature or Vehicle you control.";
    let creature_or_vehicle = CardPredicate {
        any_of: vec![CardPredicate::creature(), CardPredicate::subtype("Vehicle")],
        ..Default::default()
    };

    CardDefinition {
        key: "tla.fire_nation_engineer",
        scryfall_id: "5daba108-277d-46c8-ac35-7610f4786813",
        name: "Fire Nation Engineer".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Artificer".into()],
        power: Some(2),
        toughness: Some(3),
        // Attacking this turn cannot be undone, so the intervening-if
        // condition still holds as the ability resolves (CR 603.4).
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::BeginningOfYourEndStep,
            intervening_if: Some(Condition::AttackedThisTurn),
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    shares_creature_type: false,
                    spec: TargetSpec::PermanentYouControl {
                        predicate: creature_or_vehicle.clone(),
                    },
                    min: 1,
                    max: 1,
                    another: true,
                    max_x: false,
                },
                effects: vec![Effect::PutCounters {
                    count: 1,
                    kind: CounterKind::PlusOnePlusOne,
                    target: TargetSpec::PermanentYouControl {
                        predicate: creature_or_vehicle,
                    },
                }],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_raider() -> CardDefinition {
    let text_box = "Raid — When this creature enters, if you attacked this turn, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.fire_nation_raider",
        scryfall_id: "ede1bbdb-c726-46e3-aaf1-b8cf2be2c341",
        name: "Fire Nation Raider".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(4),
        toughness: Some(2),
        // Attacking this turn cannot be undone, so the intervening-if
        // condition still holds as the ability resolves (CR 603.4).
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            intervening_if: Some(Condition::AttackedThisTurn),
            effects: vec![Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_palace() -> CardDefinition {
    let text_box = "This land enters tapped unless you control a basic land.\n{T}: Add {R}.\n{1}{R}, {T}: Target creature you control gains firebending 4 until end of turn. (Whenever it attacks, add {R}{R}{R}{R}. This mana lasts until end of combat.)";

    CardDefinition {
        key: "tla.fire_nation_palace",
        scryfall_id: "c91ac3f2-9fcd-4b41-a168-ae7f70b67d3c",
        name: "Fire Nation Palace".into(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("R"),
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{1}{R}"),
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::CreatureYouControl),
                Modification::Firebending(4),
            )],
            ..Default::default()
        }],
        replacement_effects: vec![ReplacementEffect::EntersTappedUnlessControls {
            predicate: CardPredicate {
                supertype: Some("basic".into()),
                card_type: Some(CardType::Land),
                ..Default::default()
            },
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_warship() -> CardDefinition {
    let text_box = "Reach\nWhen this Vehicle dies, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nCrew 2 (Tap any number of creatures you control with total power 2 or more: This Vehicle becomes an artifact creature until end of turn.)";

    CardDefinition {
        key: "tla.fire_nation_warship",
        scryfall_id: "41ecc1de-353e-4131-b735-d2ce18c9c2d5",
        name: "Fire Nation Warship".into(),
        mana_cost: Some(ManaCost::parse("{3}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Vehicle".into()],
        power: Some(4),
        toughness: Some(4),
        keywords: Keywords {
            reach: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            crew: Some(2),
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::BecomeArtifactCreature,
            )],
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
            intervening_if: None,
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
            effects: vec![Effect::PutCountersOnSource {
                count: EffectValue::Fixed(1),
                kind: CounterKind::PlusOnePlusOne,
            }],
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
        spell_effects: vec![Effect::If {
            condition: Condition::Kicked,
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
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{4}"))],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn firebending_student() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\nFirebending X, where X is this creature's power. (Whenever this creature attacks, add X {R}. This mana lasts until end of combat.)";

    CardDefinition {
        key: "tla.firebending_student",
        scryfall_id: "3b366f59-16fe-43cb-888d-1f93ef8fd332",
        name: "Firebending Student".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Monk".into()],
        power: Some(1),
        toughness: Some(2),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::YouCastSpell {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                },
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::pump(1, 1),
                )],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddManaValue {
                    mana: Mana::parse("R"),
                    amount: EffectValue::SourcePower,
                    until_end_of_combat: true,
                }],
            },
        ],
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
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::This,
            condition: Some(Condition::GraveyardAtLeast {
                count: 1,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            }),
            power: 1,
            toughness: 1,
            per_creature_type_max: None,
            per_color_among: None,
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn flexible_waterbender() -> CardDefinition {
    let text_box = "Vigilance\nWaterbend {3}: This creature has base power and toughness 5/2 until end of turn. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.flexible_waterbender",
        scryfall_id: "1447ebce-7e48-4b21-a39c-740920538bdd",
        name: "Flexible Waterbender".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(5),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}"),
            waterbend: true,
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::SetBasePt {
                    power: Some(EffectValue::Fixed(5)),
                    toughness: Some(EffectValue::Fixed(2)),
                },
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn flopsie_bumi_s_buddy() -> CardDefinition {
    let text_box = "When Flopsie enters, put a +1/+1 counter on each creature you control.\nEach creature you control with power 4 or greater can't be blocked by more than one creature.";

    CardDefinition {
        key: "tla.flopsie_bumi_s_buddy",
        scryfall_id: "082f4abc-c09f-42cf-977f-060875dfd411",
        name: "Flopsie, Bumi's Buddy".into(),
        mana_cost: Some(ManaCost::parse("{4}{G}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Ape".into(), "Goat".into()],
        power: Some(4),
        toughness: Some(4),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::PutCountersOnEachMatching {
                count: EffectValue::Fixed(1),
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                other: false,
            }],
        }],
        statics: vec![StaticAbility::Restriction(StaticRestriction {
            scope: StaticScope::YouControl(CardPredicate {
                min_power: Some(4),
                card_type: Some(CardType::Creature),
                ..Default::default()
            }),
            kind: Restriction::CantBeBlockedByMoreThanOne,
            unless: None,
        })],
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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

fn foggy_swamp_hunters() -> CardDefinition {
    let text_box = "As long as you've drawn two or more cards this turn, this creature has lifelink and menace. (It can't be blocked except by two or more creatures.)";

    CardDefinition {
        key: "tla.foggy_swamp_hunters",
        scryfall_id: "3f6e5869-ca25-4b98-a844-8a498cb40aab",
        name: "Foggy Swamp Hunters".into(),
        mana_cost: Some(ManaCost::parse("{3}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Ranger".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(4),
        statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
            scope: StaticScope::This,
            condition: Some(Condition::CardsDrawnAtLeast { count: 2 }),
            keywords: Keywords {
                lifelink: true,
                menace: true,
                ..Default::default()
            },
        })],
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
            intervening_if: None,
            condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
            effects: vec![Effect::CreateToken {
                token_name: "Spirit".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
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
            effects: vec![Effect::PutCountersOnSource {
                count: EffectValue::Fixed(1),
                kind: CounterKind::PlusOnePlusOne,
            }],
            ..Default::default()
        }],
        statics: vec![StaticAbility::BlockRestriction(
            BlockRestriction::CantBeBlockedBy(CardPredicate {
                max_power: Some(2),
                ..Default::default()
            }),
        )],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
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
            effects: vec![
                Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                },
                Effect::Discard,
            ],
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
                    count: EffectValue::Fixed(1),
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
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::CantBeBlocked,
                )],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Scry { count: 1 }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn glider_staff() -> CardDefinition {
    let text_box = "When this Equipment enters, airbend up to one target creature. (Exile it. While it's exiled, its owner may cast it for {2} rather than its mana cost.)\nEquipped creature gets +1/+1 and has flying.\nEquip {2}";

    CardDefinition {
        key: "tla.glider_staff",
        scryfall_id: "7517f2eb-a24d-49f6-82bf-08de55d3789a",
        name: "Glider Staff".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Equipment".into()],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    shares_creature_type: false,
                    spec: TargetSpec::Creature,
                    min: 0,
                    max: 1,
                    another: false,
                    max_x: false,
                },
                effects: vec![Effect::Airbend {
                    target: TargetSpec::Creature,
                }],
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            sorcery_timing: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::AttachSourceTo {
                target: TargetSpec::CreatureYouControl,
            }],
            ..Default::default()
        }],
        statics: vec![
            StaticAbility::PtBuff(StaticPtBuff {
                scope: StaticScope::Attached,
                condition: None,
                power: 1,
                toughness: 1,
                per_creature_type_max: None,
                per_color_among: None,
            }),
            StaticAbility::KeywordGrant(StaticKeywordGrant {
                scope: StaticScope::Attached,
                condition: None,
                keywords: Keywords {
                    flying: true,
                    ..Default::default()
                },
            }),
        ],
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
        statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
            first_on_own_turn: false,
            predicate: CardPredicate {
                not_card_types: vec![CardType::Creature],
                ..Default::default()
            },
            condition: Condition::GraveyardAtLeast {
                count: 3,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            },
            generic: 1,
        })],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::BecomesTapped {
                subject: TriggerSubject::This,
            },
            effects: vec![
                Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                },
                Effect::Discard,
            ],
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
        statics: vec![
            StaticAbility::GrantsAnyColorManaAbility(CardPredicate {
                card_type: Some(CardType::Land),
                ..Default::default()
            }),
            StaticAbility::GrantsAnyColorManaAbility(CardPredicate {
                subtype: Some("Ally".into()),
                ..Default::default()
            }),
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn hakoda_selfless_commander() -> CardDefinition {
    let text_box = "Vigilance\nYou may look at the top card of your library any time.\nYou may cast Ally spells from the top of your library.\nSacrifice Hakoda: Creatures you control get +0/+5 and gain indestructible until end of turn.";

    CardDefinition {
        key: "tla.hakoda_selfless_commander",
        scryfall_id: "9aef3ddb-9bb7-42c8-975b-b1917955a416",
        name: "Hakoda, Selfless Commander".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(5),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        statics: vec![
            StaticAbility::LookAtLibraryTop,
            StaticAbility::CastFromLibraryTop(CardPredicate {
                subtype: Some("Ally".into()),
                ..Default::default()
            }),
        ],
        activated_abilities: vec![ActivatedAbilityDefinition {
            sacrifice_source: true,
            effects: vec![
                Effect::until_end_of_turn(
                    AffectedSpec::EachYouControl(CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    }),
                    Modification::pump(0, 5),
                ),
                Effect::until_end_of_turn(
                    AffectedSpec::EachYouControl(CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    }),
                    Modification::AddKeywords(Keywords {
                        indestructible: true,
                        ..Default::default()
                    }),
                ),
            ],
            ..Default::default()
        }],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![
                Effect::MillCards {
                    count: EffectValue::Fixed(3),
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
            intervening_if: None,
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

fn heartless_act() -> CardDefinition {
    let text_box = "Choose one —\n• Destroy target creature with no counters on it.\n• Remove up to three counters from target creature.";

    CardDefinition {
        key: "tla.heartless_act",
        scryfall_id: "b57c57a8-d72b-4c6c-a2db-a7ef61190f42",
        name: "Heartless Act".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        shares_creature_type: false,
                        spec: TargetSpec::Permanent {
                            predicate: CardPredicate {
                                no_counters: true,
                                card_type: Some(CardType::Creature),
                                ..Default::default()
                            },
                        },
                        min: 1,
                        max: 1,
                        another: false,
                        max_x: false,
                    },
                    effects: vec![Effect::Destroy {
                        target: TargetSpec::Permanent {
                            predicate: CardPredicate {
                                no_counters: true,
                                card_type: Some(CardType::Creature),
                                ..Default::default()
                            },
                        },
                    }],
                }],
                vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        shares_creature_type: false,
                        spec: TargetSpec::Creature,
                        min: 1,
                        max: 1,
                        another: false,
                        max_x: false,
                    },
                    effects: vec![Effect::RemoveCounters {
                        up_to: 3,
                        chosen: None,
                        target: TargetSpec::Creature,
                    }],
                }],
            ],
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
                effects: vec![one_mana_of_any_color(None)],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![two_mana_in_any_combination(ManaUse::spell(["Lesson"]))],
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
            intervening_if: None,
            condition: TriggerCondition::BeginningOfYourCombat,
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::PermanentYouControl {
                    predicate: CardPredicate {
                        with_counter: true,
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                }),
                Modification::AddKeywords(Keywords {
                    menace: true,
                    ..Default::default()
                }),
            )],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            exhaust: true,
            mana_cost: ManaCost::parse("{5}"),
            effects: vec![Effect::PutCountersOnSource {
                count: EffectValue::Fixed(2),
                kind: CounterKind::PlusOnePlusOne,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn honest_work() -> CardDefinition {
    let text_box = "Enchant creature an opponent controls\nWhen this Aura enters, tap enchanted creature and remove all counters from it.\nEnchanted creature loses all abilities and is a Citizen with base power and toughness 1/1 and \"{T}: Add {C}\" named Humble Merchant. (It loses all other creature types and names.)";

    CardDefinition {
        key: "tla.honest_work",
        scryfall_id: "57e0ccb0-22f0-4e4d-9a47-c4fec2c7f251",
        name: "Honest Work".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Enchantment]),
        subtypes: vec!["Aura".into()],
        statics: vec![
            StaticAbility::AttachedHas(ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("1"),
                    until_end_of_combat: false,
                }],
                ..Default::default()
            }),
            StaticAbility::AttachedSetsCharacteristics(SetCharacteristics {
                named: "Humble Merchant".into(),
                creature_types: vec!["Citizen".into()],
                power: 1,
                toughness: 1,
            }),
        ],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::TapAttached, Effect::RemoveAllCountersFromAttached],
        }],
        spell_effects: vec![Effect::AttachSourceTo {
            target: TargetSpec::CreatureOpponentControls,
        }],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::CreatureOpponentControls,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn how_to_start_a_riot() -> CardDefinition {
    let text_box = "Target creature gains menace until end of turn. (It can't be blocked except by two or more creatures.)\nCreatures target player controls get +2/+0 until end of turn.";

    CardDefinition {
        key: "tla.how_to_start_a_riot",
        scryfall_id: "23b3bf1e-ea85-47f5-8473-4d16615f68d7",
        name: "How to Start a Riot".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::TargetClause {
                requirement: TargetRequirement::one(TargetSpec::Creature),
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::Creature),
                    Modification::AddKeywords(Keywords {
                        menace: true,
                        ..Default::default()
                    }),
                )],
            },
            Effect::TargetClause {
                requirement: TargetRequirement::one(TargetSpec::Player),
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::EachTargetPlayerControls(CardPredicate::creature()),
                    Modification::pump(2, 0),
                )],
            },
        ],
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
            intervening_if: None,
            condition: TriggerCondition::YouCastSpell {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::pump(1, 1),
            )],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn invasion_submersible() -> CardDefinition {
    let text_box = "When this Vehicle enters, return up to one other target nonland permanent to its owner's hand.\nExhaust — Waterbend {3}: This Vehicle becomes an artifact creature. Put three +1/+1 counters on it. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}. Activate each exhaust ability only once.)";

    CardDefinition {
        key: "tla.invasion_submersible",
        scryfall_id: "af5299f5-1633-4c07-ade5-fd47e29ea4aa",
        name: "Invasion Submersible".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Vehicle".into()],
        power: Some(0),
        toughness: Some(0),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    shares_creature_type: false,
                    spec: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    },
                    min: 0,
                    max: 1,
                    another: true,
                    max_x: false,
                },
                effects: vec![Effect::ReturnToHand {
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
            exhaust: true,
            mana_cost: ManaCost::parse("{3}"),
            waterbend: true,
            effects: vec![
                Effect::indefinitely(AffectedSpec::Source, Modification::BecomeArtifactCreature),
                Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(3),
                    kind: CounterKind::PlusOnePlusOne,
                },
            ],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn invasion_tactics() -> CardDefinition {
    let text_box = "When this enchantment enters, creatures you control get +2/+2 until end of turn.\nWhenever one or more Allies you control deal combat damage to a player, draw a card.";

    CardDefinition {
        key: "tla.invasion_tactics",
        scryfall_id: "27c6330d-49f6-4707-b4d4-d1411fa422eb",
        name: "Invasion Tactics".into(),
        mana_cost: Some(ManaCost::parse("{4}{G}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::EachYouControl(CardPredicate::creature()),
                    Modification::pump(2, 2),
                )],
            },
            Ability::Triggered {
                condition: TriggerCondition::OneOrMoreDealCombatDamageToPlayer {
                    predicate: CardPredicate {
                        subtype: Some("Ally".into()),
                        ..Default::default()
                    },
                },
                intervening_if: None,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            },
        ],
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
        statics: vec![StaticAbility::GraveyardSpellsHaveFlashback],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
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
            shares_creature_type: false,
            spec: TargetSpec::Spell,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
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
                effects: vec![one_mana_of_any_color(Some(ManaUse::new(
                    ["Ally"],
                    ["Ally"],
                )))],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{5}"),
                effects: vec![Effect::CreateToken {
                    token_name: "Ally".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::PutCounters {
                count: 1,
                kind: CounterKind::PlusOnePlusOne,
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
                intervening_if: None,
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
                intervening_if: None,
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        shares_creature_type: false,
                        spec: TargetSpec::Creature,
                        min: 0,
                        max: 2,
                        another: false,
                        max_x: false,
                    },
                    effects: vec![Effect::ForEachTarget {
                        effects: vec![Effect::PutCounters {
                            count: 1,
                            kind: CounterKind::PlusOnePlusOne,
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
        statics: vec![StaticAbility::CantBeBlockedIf(
            Condition::CardsDrawnAtLeast { count: 2 },
        )],
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
                attach_source: false,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn katara_water_tribe_s_hope() -> CardDefinition {
    let text_box = "Vigilance\nWhen Katara enters, create a 1/1 white Ally creature token.\nWaterbend {X}: Creatures you control have base power and toughness X/X until end of turn. X can't be 0. Activate only during your turn. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.katara_water_tribe_s_hope",
        scryfall_id: "9ec03308-59a2-417a-938f-bdda75080e43",
        name: "Katara, Water Tribe's Hope".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}{U}{U}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(3),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            own_turn_only: true,
            mana_cost: ManaCost::parse("{X}"),
            waterbend: true,
            x_min: 1,
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::EachYouControl(CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                }),
                Modification::SetBasePt {
                    power: Some(EffectValue::X),
                    toughness: Some(EffectValue::X),
                },
            )],
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
                intervening_if: None,
                condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
                effects: vec![Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(1),
                    kind: CounterKind::PlusOnePlusOne,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Clue".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn kyoshi_battle_fan() -> CardDefinition {
    let text_box = "When this Equipment enters, create a 1/1 white Ally creature token, then attach this Equipment to it.\nEquipped creature gets +1/+0.\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";

    CardDefinition {
        key: "tla.kyoshi_battle_fan",
        scryfall_id: "257e6862-96fe-4312-aff2-509d0696843a",
        name: "Kyoshi Battle Fan".into(),
        mana_cost: Some(ManaCost::parse("{2}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Equipment".into()],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: true,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            sorcery_timing: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::AttachSourceTo {
                target: TargetSpec::CreatureYouControl,
            }],
            ..Default::default()
        }],
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::Attached,
            condition: None,
            power: 1,
            toughness: 0,
            per_creature_type_max: None,
            per_color_among: None,
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn kyoshi_island_plaza() -> CardDefinition {
    let text_box = "When Kyoshi Island Plaza enters, search your library for up to X basic land cards, where X is the number of Shrines you control. Put those cards onto the battlefield tapped, then shuffle.\nWhenever another Shrine you control enters, search your library for a basic land card, put it onto the battlefield tapped, then shuffle.";

    CardDefinition {
        key: "tla.kyoshi_island_plaza",
        scryfall_id: "9c25ea98-9823-4487-bf6b-ee29ce5a4ed8",
        name: "Kyoshi Island Plaza".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}")),
        types: CardTypes::new([CardType::Enchantment]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Shrine".into()],
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::SearchBasicLand {
                    subtype: None,
                    count: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate {
                            subtype: Some("Shrine".into()),
                            ..Default::default()
                        },
                    },
                    tapped: true,
                    untap_if_lands: None,
                    bottom_source: false,
                    to_hand: false,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate {
                        subtype: Some("Shrine".into()),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::SearchBasicLand {
                    subtype: None,
                    count: EffectValue::Fixed(1),
                    tapped: true,
                    untap_if_lands: None,
                    bottom_source: false,
                    to_hand: false,
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn lo_and_li_twin_tutors() -> CardDefinition {
    let text_box = "When Lo and Li enter, search your library for a Lesson or Noble card, reveal it, put it into your hand, then shuffle.\nNoble creatures you control and Lesson spells you control have lifelink.";

    CardDefinition {
        key: "tla.lo_and_li_twin_tutors",
        scryfall_id: "cba9260d-ef07-4429-99f9-6004cc1fea0f",
        name: "Lo and Li, Twin Tutors".into(),
        mana_cost: Some(ManaCost::parse("{4}{B}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Advisor".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::SearchLibraryToHand {
                predicate: CardPredicate {
                    subtypes_any: vec!["Lesson".into(), "Noble".into()],
                    ..Default::default()
                },
            }],
        }],
        statics: vec![
            StaticAbility::KeywordGrant(StaticKeywordGrant {
                scope: StaticScope::SpellsYouControl(CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                }),
                condition: None,
                keywords: Keywords {
                    lifelink: true,
                    ..Default::default()
                },
            }),
            StaticAbility::KeywordGrant(StaticKeywordGrant {
                scope: StaticScope::YouControl(CardPredicate {
                    card_type: Some(CardType::Creature),
                    subtype: Some("Noble".into()),
                    ..Default::default()
                }),
                condition: None,
                keywords: Keywords {
                    lifelink: true,
                    ..Default::default()
                },
            }),
        ],
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
                attach_source: false,
            },
        ],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Permanent {
                predicate: CardPredicate {
                    card_types_any: vec![CardType::Creature, CardType::Enchantment],
                    ..Default::default()
                },
            },
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn mai_jaded_edge() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\nExhaust — {3}: Put a double strike counter on Mai. (Activate each exhaust ability only once.)";

    CardDefinition {
        key: "tla.mai_jaded_edge",
        scryfall_id: "732e6bc9-0798-4c00-aea0-5ef4298b45f5",
        name: "Mai, Jaded Edge".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into()],
        power: Some(1),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::YouCastSpell {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::pump(1, 1),
            )],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            exhaust: true,
            mana_cost: ManaCost::parse("{3}"),
            effects: vec![Effect::PutCountersOnSource {
                count: EffectValue::Fixed(1),
                kind: CounterKind::DoubleStrike,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn mai_scornful_striker() -> CardDefinition {
    let text_box = "First strike\nWhenever a player casts a noncreature spell, they lose 2 life.";

    CardDefinition {
        key: "tla.mai_scornful_striker",
        scryfall_id: "74dd4c0e-27b8-4c47-b7a6-a281413cd6b4",
        name: "Mai, Scornful Striker".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        keywords: Keywords {
            first_strike: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::PlayerCastsSpell {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            effects: vec![Effect::LoseLife {
                player: PlayerSelector::TriggeringPlayer,
                amount: EffectValue::Fixed(2),
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn master_pakku() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\nWhenever Master Pakku becomes tapped, target player mills X cards, where X is the number of Lesson cards in your graveyard. (They put the top X cards of their library into their graveyard.)";

    CardDefinition {
        key: "tla.master_pakku",
        scryfall_id: "91e30df5-63f7-4281-8d64-8d522d663652",
        name: "Master Pakku".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Advisor".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(3),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::YouCastSpell {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                },
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::pump(1, 1),
                )],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::BecomesTapped {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::MillCards {
                    count: EffectValue::GraveyardMatching {
                        base: 0,
                        predicate: CardPredicate {
                            subtype: Some("Lesson".into()),
                            ..Default::default()
                        },
                    },
                    target: Some(TargetSpec::Player),
                }],
            },
        ],
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
            intervening_if: None,
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::This,
            condition: Some(Condition::CardsDrawnAtLeast { count: 2 }),
            power: 2,
            toughness: 0,
            per_creature_type_max: None,
            per_color_among: None,
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn meteor_sword() -> CardDefinition {
    let text_box = "When this Equipment enters, destroy target permanent.\nEquipped creature gets +3/+3.\nEquip {3} ({3}: Attach to target creature you control. Equip only as a sorcery.)";

    CardDefinition {
        key: "tla.meteor_sword",
        scryfall_id: "68b1c746-a4b8-4a45-9f0e-3962f0ae40e3",
        name: "Meteor Sword".into(),
        mana_cost: Some(ManaCost::parse("{7}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Equipment".into()],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Destroy {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate::default(),
                },
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            sorcery_timing: true,
            mana_cost: ManaCost::parse("{3}"),
            effects: vec![Effect::AttachSourceTo {
                target: TargetSpec::CreatureYouControl,
            }],
            ..Default::default()
        }],
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::Attached,
            condition: None,
            power: 3,
            toughness: 3,
            per_creature_type_max: None,
            per_color_among: None,
        })],
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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
        statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
            first_on_own_turn: true,
            predicate: CardPredicate {
                flying: true,
                not_subtypes: vec!["Lemur".into()],
                card_type: Some(CardType::Creature),
                ..Default::default()
            },
            condition: Condition::Always,
            generic: 1,
        })],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnotherYouControl(CardPredicate {
                    flying: true,
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::pump(1, 1),
            )],
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
            intervening_if: None,
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
                count: EffectValue::Fixed(1),
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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

fn northern_air_temple() -> CardDefinition {
    let text_box = "When Northern Air Temple enters, each opponent loses X life and you gain X life, where X is the number of Shrines you control.\nWhenever another Shrine you control enters, each opponent loses 1 life and you gain 1 life.";

    CardDefinition {
        key: "tla.northern_air_temple",
        scryfall_id: "06b7675e-e665-49cb-a8d9-7092a654d464",
        name: "Northern Air Temple".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Enchantment]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Shrine".into()],
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrainLife {
                    amount: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate {
                            subtype: Some("Shrine".into()),
                            ..Default::default()
                        },
                    },
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate {
                        subtype: Some("Shrine".into()),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::DrainLife {
                    amount: EffectValue::Fixed(1),
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn obsessive_pursuit() -> CardDefinition {
    let text_box = "When this enchantment enters and at the beginning of your upkeep, you lose 1 life and create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you attack, put X +1/+1 counters on target attacking creature, where X is the number of permanents you've sacrificed this turn. If X is three or more, that creature gains lifelink until end of turn.";
    let clue = || {
        vec![
            Effect::LoseLife {
                player: PlayerSelector::You,
                amount: EffectValue::Fixed(1),
            },
            Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            },
        ]
    };
    let attacking_creature = || TargetSpec::Permanent {
        predicate: CardPredicate {
            attacking: true,
            card_type: Some(CardType::Creature),
            ..Default::default()
        },
    };

    CardDefinition {
        key: "tla.obsessive_pursuit",
        scryfall_id: "e837e29c-d241-43c8-8f45-05056e082b60",
        name: "Obsessive Pursuit".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: clue(),
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::BeginningOfYourUpkeep,
                effects: clue(),
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::AnyYouControl(CardPredicate::creature()),
                },
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(attacking_creature()),
                    effects: vec![
                        Effect::PutCountersValue {
                            count: EffectValue::PermanentsSacrificedThisTurn,
                            target: attacking_creature(),
                        },
                        Effect::If {
                            condition: Condition::SacrificedThisTurnAtLeast { count: 3 },
                            then: vec![Effect::until_end_of_turn(
                                AffectedSpec::Target(attacking_creature()),
                                Modification::AddKeywords(Keywords {
                                    lifelink: true,
                                    ..Default::default()
                                }),
                            )],
                            otherwise: vec![],
                        },
                    ],
                }],
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
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::CreatureYouControl),
                Modification::pump(1, 1),
            ),
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::CreatureYouControl),
                Modification::AddKeywords(Keywords {
                    hexproof: true,
                    ..Default::default()
                }),
            ),
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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
                        kind: CounterKind::PlusOnePlusOne,
                        target: TargetSpec::CreatureYouControl,
                    },
                    Effect::until_end_of_turn(
                        AffectedSpec::Target(TargetSpec::CreatureYouControl),
                        Modification::AddKeywords(Keywords {
                            indestructible: true,
                            ..Default::default()
                        }),
                    ),
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::MillAndSelect {
                count: 3,
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                },
                if_declined: vec![Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(1),
                    kind: CounterKind::PlusOnePlusOne,
                }],
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
            intervening_if: None,
            condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
            effects: vec![
                Effect::until_end_of_turn(AffectedSpec::Source, Modification::pump(1, 2)),
                Effect::until_end_of_turn(AffectedSpec::Source, Modification::CantBeBlocked),
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

fn ozai_the_phoenix_king() -> CardDefinition {
    let text_box = "Trample, firebending 4, haste\nIf you would lose unspent mana, that mana becomes red instead.\nOzai has flying and indestructible as long as you have six or more unspent mana.";

    CardDefinition {
        key: "tla.ozai_the_phoenix_king",
        scryfall_id: "a98b1550-4609-4a2f-9371-4afe1cdc613e",
        name: "Ozai, the Phoenix King".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}{B}{R}{R}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into()],
        power: Some(7),
        toughness: Some(7),
        keywords: Keywords {
            haste: true,
            trample: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("RRRR"),
                until_end_of_combat: true,
            }],
        }],
        statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
            scope: StaticScope::This,
            condition: Some(Condition::UnspentManaAtLeast { count: 6 }),
            keywords: Keywords {
                flying: true,
                indestructible: true,
                ..Default::default()
            },
        })],
        replacement_effects: vec![ReplacementEffect::UnspentManaBecomesRed],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn path_to_redemption() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature can't attack or block.\n{5}, Sacrifice this Aura: Exile enchanted creature. Create a 1/1 white Ally creature token. Activate only during your turn.";

    CardDefinition {
        key: "tla.path_to_redemption",
        scryfall_id: "f936d64a-0db4-49c1-8a57-6d99e012a555",
        name: "Path to Redemption".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Enchantment]),
        subtypes: vec!["Aura".into()],
        activated_abilities: vec![ActivatedAbilityDefinition {
            own_turn_only: true,
            mana_cost: ManaCost::parse("{5}"),
            sacrifice_source: true,
            effects: vec![
                Effect::ExileAttached,
                Effect::CreateToken {
                    token_name: "Ally".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                },
            ],
            ..Default::default()
        }],
        spell_effects: vec![Effect::AttachSourceTo {
            target: TargetSpec::Creature,
        }],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        statics: vec![StaticAbility::Restriction(StaticRestriction {
            scope: StaticScope::Attached,
            kind: Restriction::CantAttackOrBlock,
            unless: None,
        })],
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
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::pump(2, 2),
            ),
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords {
                    reach: true,
                    ..Default::default()
                }),
            ),
            Effect::UntapTarget {
                target: TargetSpec::Creature,
            },
        ],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn pirate_peddlers() -> CardDefinition {
    let text_box = "Deathtouch\nWhenever you sacrifice another permanent, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "tla.pirate_peddlers",
        scryfall_id: "5c998994-6c3b-4f34-9b31-babba2b17266",
        name: "Pirate Peddlers".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Pirate".into()],
        power: Some(2),
        toughness: Some(2),
        keywords: Keywords {
            deathtouch: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Sacrificed {
                predicate: CardPredicate::default(),
                controller_only: true,
                another: true,
            },
            effects: vec![Effect::PutCountersOnSource {
                count: EffectValue::Fixed(1),
                kind: CounterKind::PlusOnePlusOne,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn platypus_bear() -> CardDefinition {
    let text_box = "Defender\nWhen this creature enters, mill two cards. (Put the top two cards of your library into your graveyard.)\nAs long as there is a Lesson card in your graveyard, this creature can attack as though it didn't have defender.";

    CardDefinition {
        key: "tla.platypus_bear",
        scryfall_id: "7ebf742a-3995-4b22-9dc7-b05f57889683",
        name: "Platypus-Bear".into(),
        mana_cost: Some(ManaCost::parse("{1}{G/U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Platypus".into(), "Bear".into()],
        power: Some(2),
        toughness: Some(3),
        keywords: Keywords {
            defender: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::MillCards {
                count: EffectValue::Fixed(2),
                target: None,
            }],
        }],
        statics: vec![StaticAbility::AttacksDespiteDefender(
            Condition::GraveyardAtLeast {
                count: 1,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            },
        )],
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
            intervening_if: None,
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
            },
            effects: vec![
                Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::AddKeywords(Keywords {
                        flying: true,
                        ..Default::default()
                    }),
                ),
                Effect::GainLife { amount: 1 },
            ],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            zone: ActivationZone::Hand,
            discard_source: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::SearchBasicLand {
                subtype: Some("Plains".into()),
                count: EffectValue::Fixed(1),
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
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RR"),
                    until_end_of_combat: true,
                }],
            },
            Ability::Triggered {
                condition: TriggerCondition::WasCast {
                    condition: Box::new(TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    }),
                },
                intervening_if: Some(Condition::GraveyardAtLeast {
                    count: 3,
                    predicate: CardPredicate {
                        subtypes_any: vec!["Dragon".into(), "Lesson".into()],
                        ..Default::default()
                    },
                }),
                effects: vec![Effect::CreateNonlegendaryCopy],
            },
        ],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}{R}"),
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::EachYouControl(CardPredicate {
                    subtype: Some("Dragon".into()),
                    ..Default::default()
                }),
                Modification::pump(2, 0),
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn raucous_audience() -> CardDefinition {
    let text_box =
        "{T}: Add {G}. If you control a creature with power 4 or greater, add {G}{G} instead.";

    CardDefinition {
        key: "tla.raucous_audience",
        scryfall_id: "ccd03ec3-0fae-48b2-9f04-9c57d2f94267",
        name: "Raucous Audience".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
        power: Some(2),
        toughness: Some(1),
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::If {
                condition: Condition::ControlsMatching {
                    predicate: CardPredicate {
                        min_power: Some(4),
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
                then: vec![Effect::AddMana {
                    mana: Mana::parse("GG"),
                    until_end_of_combat: false,
                }],
                otherwise: vec![Effect::AddMana {
                    mana: Mana::parse("G"),
                    until_end_of_combat: false,
                }],
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn razor_rings() -> CardDefinition {
    let text_box = "Razor Rings deals 4 damage to target attacking or blocking creature. You gain life equal to the excess damage dealt this way.";

    CardDefinition {
        key: "tla.razor_rings",
        scryfall_id: "b05cfee5-ee59-4df6-a5f6-d9ef0fa7f98a",
        name: "Razor Rings".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::DealDamage {
                amount: 4,
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        in_combat: true,
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
            },
            Effect::GainLifeValue {
                amount: EffectValue::ExcessDamage,
            },
        ],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Permanent {
                predicate: CardPredicate {
                    in_combat: true,
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
            },
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn realm_of_koh() -> CardDefinition {
    let text_box = "This land enters tapped unless you control a basic land.\n{T}: Add {B}.\n{3}{B}, {T}: Create a 1/1 colorless Spirit creature token with \"This token can't block or be blocked by non-Spirit creatures.\"";

    CardDefinition {
        key: "tla.realm_of_koh",
        scryfall_id: "061ac694-610c-479f-b038-a4ef5270d5d7",
        name: "Realm of Koh".into(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("B"),
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{3}{B}"),
            effects: vec![Effect::CreateToken {
                token_name: "Spirit".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
            ..Default::default()
        }],
        replacement_effects: vec![ReplacementEffect::EntersTappedUnlessControls {
            predicate: CardPredicate {
                supertype: Some("basic".into()),
                card_type: Some(CardType::Land),
                ..Default::default()
            },
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
                Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(2),
                    kind: CounterKind::PlusOnePlusOne,
                },
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
        statics: vec![StaticAbility::Flashback(ManaCost::parse("{5}{G}"))],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn rocky_rebuke() -> CardDefinition {
    let text_box = "Target creature you control deals damage equal to its power to target creature an opponent controls.";

    CardDefinition {
        key: "tla.rocky_rebuke",
        scryfall_id: "52fd910f-6d42-41f5-a4be-f375aa254ea2",
        name: "Rocky Rebuke".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::TargetCreaturesDealPowerDamageToLastTarget],
        targeting: vec![
            TargetRequirement::one(TargetSpec::CreatureYouControl),
            TargetRequirement::one(TargetSpec::CreatureOpponentControls),
        ],
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
            intervening_if: None,
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
                Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(2),
                    kind: CounterKind::PlusOnePlusOne,
                },
                Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::AddKeywords(Keywords {
                        trample: true,
                        ..Default::default()
                    }),
                ),
            ],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn rowdy_snowballers() -> CardDefinition {
    let text_box = "When this creature enters, tap target creature an opponent controls and put a stun counter on it. (If a permanent with a stun counter would become untapped, remove one from it instead.)";

    CardDefinition {
        key: "tla.rowdy_snowballers",
        scryfall_id: "17d71522-b133-4003-b787-0c742a2fd70e",
        name: "Rowdy Snowballers".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Peasant".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![
                Effect::TapTarget {
                    target: TargetSpec::CreatureOpponentControls,
                },
                Effect::PutCounters {
                    count: 1,
                    kind: CounterKind::Stun,
                    target: TargetSpec::CreatureOpponentControls,
                },
            ],
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
            intervening_if: None,
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
            effects: vec![one_mana_of_any_color(None)],
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
                count: EffectValue::Fixed(1),
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

fn secret_tunnel() -> CardDefinition {
    let text_box = "This land can't be blocked.\n{T}: Add {C}.\n{4}, {T}: Two target creatures you control that share a creature type can't be blocked this turn.";

    CardDefinition {
        key: "tla.secret_tunnel",
        scryfall_id: "2d39a0e1-6484-409c-ab05-5b276925a949",
        name: "Secret Tunnel".into(),
        types: CardTypes::new([CardType::Land]),
        subtypes: vec!["Cave".into()],
        statics: vec![StaticAbility::CantBeBlockedIf(Condition::Always)],
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("1"),
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{4}"),
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    shares_creature_type: true,
                    spec: TargetSpec::CreatureYouControl,
                    min: 2,
                    max: 2,
                    another: false,
                    max_x: false,
                },
                effects: vec![Effect::ForEachTarget {
                    effects: vec![Effect::until_end_of_turn(
                        AffectedSpec::Target(TargetSpec::CreatureYouControl),
                        Modification::CantBeBlocked,
                    )],
                }],
            }],
            ..Default::default()
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
        statics: vec![
            StaticAbility::FlashIf(Condition::GraveyardAtLeast {
                count: 3,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            }),
            StaticAbility::CostsLessPerGraveyardCard(CardPredicate {
                not_card_types: vec![CardType::Creature, CardType::Land],
                ..Default::default()
            }),
        ],
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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
            count: EffectValue::Fixed(1),
            tapped: true,
            untap_if_lands: None,
            bottom_source: false,
            to_hand: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sold_out() -> CardDefinition {
    let text_box = "Exile target creature. If it was dealt damage this turn, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";

    CardDefinition {
        key: "tla.sold_out",
        scryfall_id: "1affba6d-2cca-4cdf-8690-1e23ffbe8462",
        name: "Sold Out".into(),
        mana_cost: Some(ManaCost::parse("{3}{B}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::Exile {
                target: TargetSpec::Creature,
            },
            Effect::If {
                condition: Condition::TargetDealtDamageThisTurn,
                then: vec![Effect::CreateToken {
                    token_name: "Clue".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                }],
                otherwise: vec![],
            },
        ],
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
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::OptionalDiscardDraw { count: 1, up_to: 2 }],
            },
            Ability::Triggered {
                intervening_if: None,
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
                effects: vec![Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(1),
                    kind: CounterKind::PlusOnePlusOne,
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sokka_lateral_strategist() -> CardDefinition {
    let text_box = "Vigilance\nWhenever Sokka and at least one other creature attack, draw a card.";

    CardDefinition {
        key: "tla.sokka_lateral_strategist",
        scryfall_id: "1326c5db-4615-495f-8e30-376243d91352",
        name: "Sokka, Lateral Strategist".into(),
        mana_cost: Some(ManaCost::parse("{1}{W/U}{W/U}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(4),
        keywords: Keywords {
            vigilance: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::AttacksWithAnother,
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        }],
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
                    shares_creature_type: false,
                    spec: TargetSpec::Spell,
                    min: 1,
                    max: 1,
                    another: false,
                    max_x: false,
                },
                effects: vec![Effect::CounterSpell {
                    target: TargetSpec::Spell,
                }],
            },
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
            Effect::MillCards {
                count: EffectValue::Fixed(3),
                target: None,
            },
            Effect::TargetClause {
                requirement: TargetRequirement {
                    shares_creature_type: false,
                    spec: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Land),
                            ..Default::default()
                        },
                    },
                    min: 1,
                    max: 1,
                    another: false,
                    max_x: false,
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
        statics: vec![StaticAbility::Flashback(ManaCost::parse("{6}{R}"))],
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
            intervening_if: None,
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
                    effect: Box::new(Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }),
                },
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn southern_air_temple() -> CardDefinition {
    let text_box = "When Southern Air Temple enters, put X +1/+1 counters on each creature you control, where X is the number of Shrines you control.\nWhenever another Shrine you control enters, put a +1/+1 counter on each creature you control.";

    CardDefinition {
        key: "tla.southern_air_temple",
        scryfall_id: "5c172f9b-2184-4736-9f90-74ce08596292",
        name: "Southern Air Temple".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}")),
        types: CardTypes::new([CardType::Enchantment]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Shrine".into()],
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::PutCountersOnEachMatching {
                    count: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate {
                            subtype: Some("Shrine".into()),
                            ..Default::default()
                        },
                    },
                    predicate: CardPredicate::creature(),
                    other: false,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate {
                        subtype: Some("Shrine".into()),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::PutCountersOnEachMatching {
                    count: EffectValue::Fixed(1),
                    predicate: CardPredicate::creature(),
                    other: false,
                }],
            },
        ],
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
        spell_effects: vec![Effect::until_end_of_turn(
            AffectedSpec::EachYouControl(CardPredicate::creature()),
            Modification::Firebending(5),
        )],
        statics: vec![StaticAbility::Foretell(ManaCost::parse("{2}{R}"))],
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

fn spirit_water_revival() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, you may waterbend {6}. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)\nDraw two cards. If this spell's additional cost was paid, instead shuffle your graveyard into your library, draw seven cards, and you have no maximum hand size for the rest of the game.\nExile Spirit Water Revival.";

    CardDefinition {
        key: "tla.spirit_water_revival",
        scryfall_id: "0c019e76-c88e-4d1b-a546-0f4e462ef44a",
        name: "Spirit Water Revival".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::If {
                condition: Condition::Kicked,
                then: vec![
                    Effect::ShuffleGraveyardIntoLibrary,
                    Effect::DrawCards {
                        count: EffectValue::Fixed(7),
                    },
                    Effect::indefinitely(AffectedSpec::You, Modification::NoMaximumHandSize),
                ],
                otherwise: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                }],
            },
            Effect::ExileSourceSpell,
        ],
        additional_costs: vec![AdditionalCost::Waterbend(ManaCost::parse("{6}"))],
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
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: true,
                attach_source: false,
            }],
        }],
        statics: vec![StaticAbility::CharacteristicPower(
            PowerCda::CreaturesYouControl,
        )],
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
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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

fn sun_warriors() -> CardDefinition {
    let text_box = "Firebending X, where X is the number of creatures you control. (Whenever this creature attacks, add X {R}. This mana lasts until end of combat.)\n{5}: Create a 1/1 white Ally creature token.";

    CardDefinition {
        key: "tla.sun_warriors",
        scryfall_id: "e477d750-42d2-48c8-bc1f-8148d15d5f53",
        name: "Sun Warriors".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(5),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddManaValue {
                mana: Mana::parse("R"),
                amount: EffectValue::BattlefieldMatching {
                    predicate: CardPredicate::creature(),
                },
                until_end_of_combat: true,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{5}"),
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn swampsnare_trap() -> CardDefinition {
    let text_box = "This spell costs {1} less to cast if it targets a creature with flying.\nEnchant creature\nEnchanted creature gets -5/-3.";

    CardDefinition {
        key: "tla.swampsnare_trap",
        scryfall_id: "6348b6bf-08a7-45f0-8b2d-2827ab89f21c",
        name: "Swampsnare Trap".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Enchantment]),
        subtypes: vec!["Aura".into()],
        spell_effects: vec![Effect::AttachSourceTo {
            target: TargetSpec::Creature,
        }],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        statics: vec![
            StaticAbility::CostsLessIfTargets(TargetCostReduction {
                predicate: CardPredicate {
                    flying: true,
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                generic: 1,
            }),
            StaticAbility::PtBuff(StaticPtBuff {
                scope: StaticScope::Attached,
                condition: None,
                power: -5,
                toughness: -3,
                per_creature_type_max: None,
                per_color_among: None,
            }),
        ],
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
            intervening_if: None,
            condition: TriggerCondition::AttacksAlone {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::TriggeringPermanent,
                Modification::ModifyPt {
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
                },
            )],
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

fn the_boulder_ready_to_rumble() -> CardDefinition {
    let text_box = "Whenever The Boulder attacks, earthbend X, where X is the number of creatures you control with power 4 or greater. (Target land you control becomes a 0/0 creature with haste that's still a land. Put X +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.the_boulder_ready_to_rumble",
        scryfall_id: "ec27a466-5457-44c6-a842-1de7d3788d66",
        name: "The Boulder, Ready to Rumble".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Performer".into()],
        power: Some(4),
        toughness: Some(4),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Earthbend {
                count: EffectValue::BattlefieldMatching {
                    predicate: CardPredicate {
                        min_power: Some(4),
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                },
                target: TargetSpec::LandYouControl,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn the_earth_king() -> CardDefinition {
    let text_box = "When The Earth King enters, create a 4/4 green Bear creature token.\nWhenever one or more creatures you control with power 4 or greater attack, search your library for up to that many basic land cards, put them onto the battlefield tapped, then shuffle.";

    CardDefinition {
        key: "tla.the_earth_king",
        scryfall_id: "a8d5dca6-381a-4361-b265-27de8d04335c",
        name: "The Earth King".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Noble".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Bear".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::AnyYouControl(CardPredicate {
                        card_type: Some(CardType::Creature),
                        min_power: Some(4),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::SearchBasicLand {
                    subtype: None,
                    count: EffectValue::AttackingMatching {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Creature),
                            min_power: Some(4),
                            ..Default::default()
                        },
                    },
                    tapped: true,
                    untap_if_lands: None,
                    bottom_source: false,
                    to_hand: false,
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn the_last_agni_kai() -> CardDefinition {
    let text_box = "Target creature you control fights target creature an opponent controls. If the creature the opponent controls is dealt excess damage this way, add that much {R}.\nUntil end of turn, you don't lose unspent red mana as steps and phases end.";

    CardDefinition {
        key: "tla.the_last_agni_kai",
        scryfall_id: "61eaebc6-7575-48ed-b212-ff8b0c7ae694",
        name: "The Last Agni Kai".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::FightLastTarget,
            Effect::AddManaValue {
                mana: Mana::parse("R"),
                amount: EffectValue::ExcessDamage,
                until_end_of_combat: false,
            },
            Effect::until_end_of_turn(AffectedSpec::You, Modification::KeepUnspentMana(Color::Red)),
        ],
        targeting: vec![
            TargetRequirement {
                shares_creature_type: false,
                spec: TargetSpec::CreatureYouControl,
                min: 1,
                max: 1,
                another: false,
                max_x: false,
            },
            TargetRequirement {
                shares_creature_type: false,
                spec: TargetSpec::CreatureOpponentControls,
                min: 1,
                max: 1,
                another: false,
                max_x: false,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn the_lion_turtle() -> CardDefinition {
    let text_box = "Reach, vigilance\nWhen The Lion-Turtle enters, you gain 3 life.\nThe Lion-Turtle can't attack or block unless there are three or more Lesson cards in your graveyard.\n{T}: Add one mana of any color.";

    CardDefinition {
        key: "tla.the_lion_turtle",
        scryfall_id: "8491585c-81d0-48be-8ed0-832e42fad9c6",
        name: "The Lion-Turtle".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Elder".into(), "Cat".into(), "Turtle".into()],
        power: Some(3),
        toughness: Some(6),
        keywords: Keywords {
            reach: true,
            vigilance: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 3 }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
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
        }],
        statics: vec![StaticAbility::Restriction(StaticRestriction {
            scope: StaticScope::This,
            kind: Restriction::CantAttackOrBlock,
            unless: Some(Condition::GraveyardAtLeast {
                count: 3,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            }),
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn the_mechanist_aerial_artisan() -> CardDefinition {
    let text_box = "Whenever you cast a noncreature spell, create a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\n{T}: Until end of turn, target artifact token you control becomes a 3/1 Construct artifact creature with flying.";

    CardDefinition {
        key: "tla.the_mechanist_aerial_artisan",
        scryfall_id: "b910851e-9332-4c83-a790-d379667cabfc",
        name: "The Mechanist, Aerial Artisan".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Artificer".into(), "Ally".into()],
        power: Some(1),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::YouCastSpell {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            effects: vec![Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![
                Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::PermanentYouControl {
                        predicate: CardPredicate {
                            token: Some(true),
                            card_type: Some(CardType::Artifact),
                            ..Default::default()
                        },
                    }),
                    Modification::BecomeArtifactCreature,
                ),
                Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::PermanentYouControl {
                        predicate: CardPredicate {
                            token: Some(true),
                            card_type: Some(CardType::Artifact),
                            ..Default::default()
                        },
                    }),
                    Modification::SetBasePt {
                        power: Some(EffectValue::Fixed(3)),
                        toughness: Some(EffectValue::Fixed(1)),
                    },
                ),
                Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::PermanentYouControl {
                        predicate: CardPredicate {
                            token: Some(true),
                            card_type: Some(CardType::Artifact),
                            ..Default::default()
                        },
                    }),
                    Modification::AddSubtypes(vec!["Construct".into()]),
                ),
                Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::PermanentYouControl {
                        predicate: CardPredicate {
                            token: Some(true),
                            card_type: Some(CardType::Artifact),
                            ..Default::default()
                        },
                    }),
                    Modification::AddKeywords(Keywords {
                        flying: true,
                        ..Default::default()
                    }),
                ),
            ],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn the_spirit_oasis() -> CardDefinition {
    let text_box = "When The Spirit Oasis enters, draw a card for each Shrine you control.\nWhenever another Shrine you control enters, draw a card.";

    CardDefinition {
        key: "tla.the_spirit_oasis",
        scryfall_id: "3a1b1329-531c-47ed-9802-c505501f8fd9",
        name: "The Spirit Oasis".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Enchantment]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Shrine".into()],
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrawCards {
                    count: EffectValue::BattlefieldMatching {
                        predicate: CardPredicate {
                            subtype: Some("Shrine".into()),
                            ..Default::default()
                        },
                    },
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnotherYouControl(CardPredicate {
                        subtype: Some("Shrine".into()),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn the_walls_of_ba_sing_se() -> CardDefinition {
    let text_box = "Defender\nOther permanents you control have indestructible.";

    CardDefinition {
        key: "tla.the_walls_of_ba_sing_se",
        scryfall_id: "24a14ac5-0b7e-4441-9963-782a7cda3f4a",
        name: "The Walls of Ba Sing Se".into(),
        mana_cost: Some(ManaCost::parse("{8}")),
        types: CardTypes::new([CardType::Creature, CardType::Artifact]),
        supertypes: vec!["legendary".into()],
        subtypes: vec!["Wall".into()],
        power: Some(0),
        toughness: Some(30),
        keywords: Keywords {
            defender: true,
            ..Default::default()
        },
        statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
            scope: StaticScope::OtherYouControl(CardPredicate::default()),
            condition: None,
            keywords: Keywords {
                indestructible: true,
                ..Default::default()
            },
        })],
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
        statics: vec![StaticAbility::CantAttackOrBlockUnlessYouControlAnother(
            CardPredicate {
                min_power: Some(4),
                card_type: Some(CardType::Creature),
                ..Default::default()
            },
        )],
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
                intervening_if: None,
                condition: TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
                effects: vec![Effect::UntapSource],
            },
            Ability::Triggered {
                intervening_if: None,
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
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Clue".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::OnceEachTurn {
                    condition: Box::new(TriggerCondition::ActiveIf {
                        active_if: Condition::YourTurn,
                        condition: Box::new(TriggerCondition::Sacrificed {
                            predicate: CardPredicate::default(),
                            controller_only: true,
                            another: false,
                        }),
                    }),
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Ally".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(2),
                target: TargetSpec::LandYouControl,
            }],
        }],
        statics: vec![StaticAbility::CharacteristicPower(
            PowerCda::CountersOnLandsYouControl,
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn toph_the_first_metalbender() -> CardDefinition {
    let text_box = "Nontoken artifacts you control are lands in addition to their other types. (They don't gain the ability to {T} for mana.)\nAt the beginning of your end step, earthbend 2. (Target land you control becomes a 0/0 creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped.)";

    CardDefinition {
        key: "tla.toph_the_first_metalbender",
        scryfall_id: "70b6670f-a9fa-4d75-b0a7-c01b5071a514",
        name: "Toph, the First Metalbender".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}{R}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(3),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::BeginningOfYourEndStep,
            effects: vec![Effect::Earthbend {
                count: EffectValue::Fixed(2),
                target: TargetSpec::LandYouControl,
            }],
        }],
        statics: vec![StaticAbility::TypeGrant(StaticTypeGrant {
            scope: StaticScope::YouControl(CardPredicate {
                token: Some(false),
                card_type: Some(CardType::Artifact),
                ..Default::default()
            }),
            types: vec![CardType::Land],
        })],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
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
                    shares_creature_type: false,
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
                    max_x: false,
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
                attach_source: false,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn trusty_boomerang() -> CardDefinition {
    let text_box = "Equipped creature has \"{1}, {T}: Tap target creature. Return Trusty Boomerang to its owner's hand.\"\nEquip {1} ({1}: Attach to target creature you control. Equip only as a sorcery.)";

    CardDefinition {
        key: "tla.trusty_boomerang",
        scryfall_id: "df99a166-6d58-4d92-9037-2fdfbaf65629",
        name: "Trusty Boomerang".into(),
        mana_cost: Some(ManaCost::parse("{1}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Equipment".into()],
        statics: vec![StaticAbility::AttachedHas(ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{1}"),
            effects: vec![
                Effect::TapTarget {
                    target: TargetSpec::Creature,
                },
                Effect::ReturnGrantorToHand,
            ],
            ..Default::default()
        })],
        activated_abilities: vec![ActivatedAbilityDefinition {
            sorcery_timing: true,
            mana_cost: ManaCost::parse("{1}"),
            effects: vec![Effect::AttachSourceTo {
                target: TargetSpec::CreatureYouControl,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn tundra_tank() -> CardDefinition {
    let text_box = "Firebending 1 (Whenever this creature attacks, add {R}. This mana lasts until end of combat.)\nWhen this Vehicle enters, target creature you control gains indestructible until end of turn.\nCrew 1 (Tap any number of creatures you control with total power 1 or more: This Vehicle becomes an artifact creature until end of turn.)";

    CardDefinition {
        key: "tla.tundra_tank",
        scryfall_id: "13c66fce-8f8e-4a98-8413-9f05aebf6e13",
        name: "Tundra Tank".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Vehicle".into()],
        power: Some(4),
        toughness: Some(4),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("R"),
                    until_end_of_combat: true,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::CreatureYouControl),
                    Modification::AddKeywords(Keywords {
                        indestructible: true,
                        ..Default::default()
                    }),
                )],
            },
        ],
        activated_abilities: vec![ActivatedAbilityDefinition {
            crew: Some(1),
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::BecomeArtifactCreature,
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn turtle_duck() -> CardDefinition {
    let text_box = "{3}: Until end of turn, this creature has base power 4 and gains trample.";

    CardDefinition {
        key: "tla.turtle_duck",
        scryfall_id: "aefcd734-3916-4c77-9d98-3ea2c2795658",
        name: "Turtle-Duck".into(),
        mana_cost: Some(ManaCost::parse("{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Turtle".into(), "Bird".into()],
        power: Some(0),
        toughness: Some(4),
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}"),
            effects: vec![
                Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::SetBasePt {
                        power: Some(EffectValue::Fixed(4)),
                        toughness: None,
                    },
                ),
                Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::AddKeywords(Keywords {
                        trample: true,
                        ..Default::default()
                    }),
                ),
            ],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn twin_blades() -> CardDefinition {
    let text_box = "Flash\nWhen this Equipment enters, attach it to target creature you control. That creature gains double strike until end of turn.\nEquipped creature gets +1/+1.\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";

    CardDefinition {
        key: "tla.twin_blades",
        scryfall_id: "dc399ae0-36b5-4c92-9fe2-138caf8d7a86",
        name: "Twin Blades".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Equipment".into()],
        keywords: Keywords {
            flash: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![
                Effect::AttachSourceTo {
                    target: TargetSpec::CreatureYouControl,
                },
                Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::CreatureYouControl),
                    Modification::AddKeywords(Keywords {
                        double_strike: true,
                        ..Default::default()
                    }),
                ),
            ],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            sorcery_timing: true,
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::AttachSourceTo {
                target: TargetSpec::CreatureYouControl,
            }],
            ..Default::default()
        }],
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::Attached,
            condition: None,
            power: 1,
            toughness: 1,
            per_creature_type_max: None,
            per_color_among: None,
        })],
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
        statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
            first_on_own_turn: false,
            predicate: CardPredicate {
                subtype: Some("Lesson".into()),
                ..Default::default()
            },
            condition: Condition::Always,
            generic: 1,
        })],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
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
                count: EffectValue::Fixed(1),
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
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Food".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Sacrificed {
                    predicate: CardPredicate {
                        subtype: Some("Food".into()),
                        ..Default::default()
                    },
                    controller_only: true,
                    another: false,
                },
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::SearchBasicLand {
                            subtype: None,
                            count: EffectValue::Fixed(1),
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

fn vengeful_villagers() -> CardDefinition {
    let text_box = "Whenever this creature attacks, choose target creature an opponent controls. Tap it, then you may sacrifice an artifact or creature. If you do, put a stun counter on the chosen creature. (If a permanent with a stun counter would become untapped, remove one from it instead.)";

    CardDefinition {
        key: "tla.vengeful_villagers",
        scryfall_id: "8cabb6ed-5c80-4dab-b96c-9d4d9fe72db7",
        name: "Vengeful Villagers".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
        power: Some(3),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            intervening_if: None,
            effects: vec![
                Effect::TapTarget {
                    target: TargetSpec::CreatureOpponentControls,
                },
                Effect::MaySacrifice {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Artifact, CardType::Creature],
                        ..Default::default()
                    },
                    another: false,
                    then: vec![Effect::PutCounters {
                        count: 1,
                        kind: CounterKind::Stun,
                        target: TargetSpec::CreatureOpponentControls,
                    }],
                },
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn vindictive_warden() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nFirebending 1 (Whenever this creature attacks, add {R}. This mana lasts until end of combat.)\n{3}: This creature deals 1 damage to each opponent.";

    CardDefinition {
        key: "tla.vindictive_warden",
        scryfall_id: "0f34de05-39a9-425f-a3b4-3a9d46c917ac",
        name: "Vindictive Warden".into(),
        mana_cost: Some(ManaCost::parse("{2}{B/R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(2),
        toughness: Some(3),
        keywords: Keywords {
            menace: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("R"),
                until_end_of_combat: true,
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}"),
            effects: vec![Effect::DealDamageToPlayer {
                player: PlayerSelector::EachOpponent,
                amount: EffectValue::Fixed(1),
            }],
            ..Default::default()
        }],
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
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            intervening_if: Some(Condition::GraveyardAtLeast {
                count: 1,
                predicate: CardPredicate {
                    subtype: Some("Lesson".into()),
                    ..Default::default()
                },
            }),
            effects: vec![Effect::GainLife { amount: 2 }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn wan_shi_tong_librarian() -> CardDefinition {
    let text_box = "Flash\nFlying, vigilance\nWhen Wan Shi Tong enters, put X +1/+1 counters on him. Then draw half X cards, rounded down.\nWhenever an opponent searches their library, put a +1/+1 counter on Wan Shi Tong and draw a card.";

    CardDefinition {
        key: "tla.wan_shi_tong_librarian",
        scryfall_id: "e20da6b5-1057-4a28-9e85-07de714e262f",
        name: "Wan Shi Tong, Librarian".into(),
        mana_cost: Some(ManaCost::parse("{X}{U}{U}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Bird".into(), "Spirit".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            flash: true,
            flying: true,
            vigilance: true,
            ..Default::default()
        },
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![
                    Effect::PutCountersOnSource {
                        count: EffectValue::X,
                        kind: CounterKind::PlusOnePlusOne,
                    },
                    Effect::DrawCards {
                        count: EffectValue::Half(Box::new(EffectValue::X)),
                    },
                ],
            },
            Ability::Triggered {
                condition: TriggerCondition::OpponentSearchesLibrary,
                intervening_if: None,
                effects: vec![
                    Effect::PutCountersOnSource {
                        count: EffectValue::Fixed(1),
                        kind: CounterKind::PlusOnePlusOne,
                    },
                    Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    },
                ],
            },
        ],
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
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::EachYouControl(CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                }),
                Modification::pump(1, 0),
            )],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn war_balloon() -> CardDefinition {
    let text_box = "Flying\n{1}: Put a fire counter on this Vehicle.\nAs long as this Vehicle has three or more fire counters on it, it's an artifact creature.\nCrew 3 (Tap any number of creatures you control with total power 3 or more: This Vehicle becomes an artifact creature until end of turn.)";

    CardDefinition {
        key: "tla.war_balloon",
        scryfall_id: "6829b20d-c2fa-41a6-89ca-f21c522d8866",
        name: "War Balloon".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Artifact]),
        subtypes: vec!["Vehicle".into()],
        power: Some(4),
        toughness: Some(3),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                crew: Some(3),
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::Source,
                    Modification::BecomeArtifactCreature,
                )],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                effects: vec![Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(1),
                    kind: CounterKind::Fire,
                }],
                ..Default::default()
            },
        ],
        statics: vec![StaticAbility::CreatureWithCounters(CounterThreshold {
            kind: CounterKind::Fire,
            count: 3,
        })],
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnotherYouControl(CardPredicate {
                    subtype: Some("Ally".into()),
                    ..Default::default()
                }),
            },
            effects: vec![
                Effect::PutCountersOnTriggeringPermanent { count: 1 },
                Effect::until_end_of_turn(
                    AffectedSpec::TriggeringPermanent,
                    Modification::AddKeywords(Keywords {
                        haste: true,
                        ..Default::default()
                    }),
                ),
            ],
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
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::EachYouControl(CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                }),
                Modification::pump(1, 1),
            )],
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

fn waterbender_ascension() -> CardDefinition {
    let text_box = "Whenever a creature you control deals combat damage to a player, put a quest counter on this enchantment. Then if it has four or more quest counters on it, draw a card.\nWaterbend {4}: Target creature can't be blocked this turn. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.waterbender_ascension",
        scryfall_id: "3f57e0f9-e232-489c-b991-d0d23f75d8dd",
        name: "Waterbender Ascension".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::CreatureDealsCombatDamageToPlayer {
                subject: TriggerSubject::AnyYouControl(CardPredicate::creature()),
            },
            intervening_if: None,
            effects: vec![
                Effect::PutCountersOnSource {
                    count: EffectValue::Fixed(1),
                    kind: CounterKind::Quest,
                },
                Effect::If {
                    condition: Condition::SourceCountersAtLeast {
                        kind: CounterKind::Quest,
                        count: 4,
                    },
                    then: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                    otherwise: vec![],
                },
            ],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{4}"),
            waterbend: true,
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::CantBeBlocked,
            )],
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
            Effect::DrawCards {
                count: EffectValue::Fixed(3),
            },
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
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn watery_grasp() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature doesn't untap during its controller's untap step.\nWaterbend {5}: Enchanted creature's owner shuffles it into their library. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)";

    CardDefinition {
        key: "tla.watery_grasp",
        scryfall_id: "26bcff57-428e-4f30-a153-a778fbfc437d",
        name: "Watery Grasp".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Enchantment]),
        subtypes: vec!["Aura".into()],
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{5}"),
            waterbend: true,
            effects: vec![Effect::ShuffleAttachedIntoLibrary],
            ..Default::default()
        }],
        spell_effects: vec![Effect::AttachSourceTo {
            target: TargetSpec::Creature,
        }],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::Creature,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        statics: vec![StaticAbility::Restriction(StaticRestriction {
            scope: StaticScope::Attached,
            kind: Restriction::DoesntUntap,
            unless: None,
        })],
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
                effects: vec![one_mana_of_any_color(Some(ManaUse::spell([
                    "Lesson", "Shrine",
                ])))],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{1}"),
                effects: vec![one_mana_of_any_color(None)],
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
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::OtherYouControl(CardPredicate {
                subtype: Some("Ally".into()),
                ..Default::default()
            }),
            condition: None,
            power: 1,
            toughness: 1,
            per_creature_type_max: None,
            per_color_among: None,
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn white_lotus_tile() -> CardDefinition {
    let text_box = "This artifact enters tapped.\n{T}: Add X mana of any one color, where X is the greatest number of creatures you control that have a creature type in common.";

    CardDefinition {
        key: "tla.white_lotus_tile",
        scryfall_id: "1d70f9ec-fdc1-4219-b89b-c030d712c1fc",
        name: "White Lotus Tile".into(),
        mana_cost: Some(ManaCost::parse("{4}")),
        types: CardTypes::new([CardType::Artifact]),
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::AddOneColorManaValue {
                amount: EffectValue::GreatestSharedCreatureType,
            }],
            ..Default::default()
        }],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
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
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::CreatureYouControl),
                Modification::pump(2, 2),
            ),
            Effect::If {
                condition: Condition::TargetMatches(CardPredicate {
                    subtype: Some("Ally".into()),
                    ..Default::default()
                }),
                then: vec![Effect::until_end_of_turn(
                    AffectedSpec::Target(TargetSpec::CreatureYouControl),
                    Modification::AddKeywords(Keywords {
                        flying: true,
                        ..Default::default()
                    }),
                )],
                otherwise: vec![],
            },
        ],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::CreatureYouControl,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
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
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::OptionalDiscardDraw { count: 1, up_to: 1 }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn zhao_ruthless_admiral() -> CardDefinition {
    let text_box = "Firebending 2 (Whenever this creature attacks, add {R}{R}. This mana lasts until end of combat.)\nWhenever you sacrifice another permanent, creatures you control get +1/+0 until end of turn.";

    CardDefinition {
        key: "tla.zhao_ruthless_admiral",
        scryfall_id: "3fd48a57-b0bb-4177-a0f3-bd317a179cbe",
        name: "Zhao, Ruthless Admiral".into(),
        mana_cost: Some(ManaCost::parse("{2}{B/R}{B/R}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(3),
        toughness: Some(4),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RR"),
                    until_end_of_combat: true,
                }],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Sacrificed {
                    predicate: CardPredicate::default(),
                    controller_only: true,
                    another: true,
                },
                effects: vec![Effect::until_end_of_turn(
                    AffectedSpec::EachYouControl(CardPredicate::creature()),
                    Modification::pump(1, 0),
                )],
            },
        ],
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
            intervening_if: None,
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
        spell_effects: vec![Effect::If {
            condition: Condition::Kicked,
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
            shares_creature_type: false,
            spec: TargetSpec::GraveyardCardYouOwn {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
            },
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{4}"))],
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

// ---- Mana constructors ----

/// The colored slots of `Mana::mana`, in WUBRG order; slot 5 is colorless.
const COLOR_SLOTS: std::ops::Range<usize> = 0..5;

/// Mana of `amounts` by slot, restricted to `spend` when set (CR 106.6).
fn mana_of(amounts: [u8; 6], spend: Option<ManaUse>) -> Mana {
    match spend {
        Some(spend) => Mana::restricted_to(spend, amounts),
        None => Mana {
            mana: amounts,
            ..Mana::default()
        },
    }
}

/// A modal mana effect with one mode per amount it may add.
fn add_one_of(options: impl Iterator<Item = Mana>) -> Effect {
    Effect::Modal {
        modes: options
            .map(|mana| {
                vec![Effect::AddMana {
                    mana,
                    until_end_of_combat: false,
                }]
            })
            .collect(),
    }
}

/// "Add one mana of any color": one mode per color, W through G.
fn one_mana_of_any_color(spend: Option<ManaUse>) -> Effect {
    add_one_of(COLOR_SLOTS.map(|slot| {
        let mut amounts = [0; 6];
        amounts[slot] = 1;
        mana_of(amounts, spend)
    }))
}

/// "Add two mana in any combination of colors": one mode per unordered pair
/// (WW, WU, ..., GG), each restricted to `spend`.
fn two_mana_in_any_combination(spend: ManaUse) -> Effect {
    add_one_of(COLOR_SLOTS.flat_map(|first| {
        (first..COLOR_SLOTS.end).map(move |second| {
            let mut amounts = [0; 6];
            amounts[first] += 1;
            amounts[second] += 1;
            mana_of(amounts, Some(spend))
        })
    }))
}
