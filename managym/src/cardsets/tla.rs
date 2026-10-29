//! Current TLA cards and their fixed tokens.
//! Edit these typed definitions; `semantic::compile` lowers them in process.
use crate::state::{
    ability::{
        Ability, Effect, EffectValue, PaymentCost, StaticCondition, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords, ManaAbility,
        PowerCda, ReplacementEffect, SacrificeCost, SpellCostReduction, StaticPtBuff, StaticScope,
        TriggeredManaAbility,
    },
    mana::{Color, Colors, Mana, ManaCost},
    predicate::CardPredicate,
};

/// Defined cards whose behavior is incomplete. The fixed setups that name them
/// still run them; the world does not support them for deck entry until their
/// batch completes the missing behavior.
pub(super) const INCOMPLETE: &[super::Incomplete] = &[
    ("tla.accumulate_wisdom", "Controller must order the library remainder (CR 401.4); it is randomized"),
    ("tla.badgermole_cub", "Unreviewed until T2: ETB earthbend; creature mana taps add green; waterbend taps do not produce mana."),
    ("tla.cat_gator", "Any target must include planeswalkers and battles (CR 115.4)"),
    ("tla.dai_li_agents", "Unreviewed until T2: Two earthbend clauses may share target; attack drain counts creatures with counters."),
    ("tla.dragonfly_swarm", "Unreviewed until T4: Flying; spell-and-ability ward; graveyard CDA power; conditional death draw."),
    ("tla.great_divide_guide", "Unreviewed until T3: Continuous any-color mana grant to lands and Allies; source removal and creature sickness."),
    ("tla.haru_hidden_talent", "Unreviewed until T2: Another Ally entry earthbend; repeated targets and independent delayed returns."),
    ("tla.hermitic_herbalist", "Unreviewed until T2: Any-color one mana or all unordered two-color combinations restricted to Lesson casting."),
    ("tla.iroh_grand_lotus", "Unreviewed until T2: Firebending; own-turn graveyard flashback grants differing by Lesson type; retention after source leaves."),
    ("ths.lightning_strike", "Any target must include planeswalkers and battles (CR 115.4)"),
    ("tla.ran_and_shaw", "Unreviewed until T5: Flying/firebending; cast-only conditional ETB nonlegendary self copy; Dragon anthem activation."),
    ("tla.suki_kyoshi_warrior", "CDA power must apply outside the battlefield (CR 604.3)"),
    ("tla.toph_the_blind_bandit", "Unreviewed until T2: ETB earthbend; CDA power counts counters on controlled lands."),
    ("tla.water_tribe_rallier", "Library power filter must evaluate CDAs, such as Suki's (CR 604.3)"),
];

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        abandon_attachments(),
        accumulate_wisdom(),
        airship_engine_room(),
        allies_at_last(),
        ally(),
        avatar_enthusiasts(),
        azula_always_lies(),
        badgermole_cub(),
        ballistic_boulder(),
        bear(),
        boar_q_pine(),
        boiling_rock_prison(),
        boomerang_basics(),
        callous_inspector(),
        cat_gator(),
        cat_owl(),
        clue(),
        combustion_technique(),
        compassionate_healer(),
        cunning_maneuver(),
        curious_farm_animals(),
        cycle_of_renewal(),
        dai_li_agents(),
        deadly_precision(),
        deserter_s_disciple(),
        dragon(),
        dragonfly_swarm(),
        earth_kingdom_jailer(),
        earth_kingdom_protectors(),
        earth_kingdom_soldier(),
        earth_kings_lieutenant(),
        enter_the_avatar_state(),
        fancy_footwork(),
        fire_nation_cadets(),
        fire_sages(),
        firebending_lesson(),
        first_time_flyer(),
        foggy_bottom_swamp(),
        foggy_swamp_spirit_keeper(),
        food(),
        forecasting_fortune_teller(),
        gather_the_white_lotus(),
        glider_kids(),
        gran_gran(),
        great_divide_guide(),
        haru_hidden_talent(),
        hermitic_herbalist(),
        iguana_parrot(),
        invasion_reinforcements(),
        iroh_grand_lotus(),
        iroh_s_demonstration(),
        itll_quench_ya(),
        jeong_jeongs_deserters(),
        jet_freedom_fighter(),
        june_bounty_hunter(),
        knowledge_seeker(),
        kyoshi_village(),
        kyoshi_warriors(),
        master_piandao(),
        meditation_pools(),
        messenger_hawk(),
        misty_palms_oasis(),
        momo_friendly_flier(),
        monk(),
        north_pole_gates(),
        octopus_form(),
        omashu_city(),
        origin_of_metalbending(),
        ostrich_horse(),
        otter_penguin(),
        ozai_s_cruelty(),
        pillar_launch(),
        pretending_poxbearers(),
        ran_and_shaw(),
        rumble_arena(),
        serpents_pass(),
        shared_roots(),
        sokka_s_haiku(),
        soldier(),
        south_pole_voyager(),
        spirit(),
        suki_kyoshi_warrior(),
        sun_blessed_peak(),
        tiger_dillo(),
        tiger_seal(),
        tolls_of_war(),
        toph_the_blind_bandit(),
        treetop_freedom_fighters(),
        true_ancestry(),
        uncle_iroh(),
        unlucky_cabbage_merchant(),
        walltop_sentries(),
        wandering_musicians(),
        wartime_protestors(),
        water_tribe_captain(),
        water_tribe_rallier(),
        waterbending_lesson(),
        white_lotus_reinforcements(),
        yip_yip(),
        yuyan_archers(),
        zuko_s_exile(),
    ]
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
        spell_effects: vec![Effect::OptionalDiscardDraw { count: 2 }],
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
                look: 3,
                min_select: 1,
                max_select: 1,
                predicate: CardPredicate::default(),
            }],
        }],
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
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
            },
            TargetRequirement {
                spec: TargetSpec::CreatureOpponentControls,
                min: 1,
                max: 1,
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
                count: 1,
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
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
            effect: Effect::DrawCards { count: 1 },
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
            effect: Effect::TargetClause {
                requirement: TargetRequirement {
                    spec: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                            ..Default::default()
                        },
                    },
                    min: 0,
                    max: 1,
                },
                effects: vec![Effect::Destroy {
                    target: TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                            ..Default::default()
                        },
                    },
                }],
            },
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
                        },
                        effects: vec![Effect::Earthbend {
                            count: 1,
                            target: TargetSpec::LandYouControl,
                        }],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            spec: TargetSpec::LandYouControl,
                            min: 1,
                            max: 1,
                        },
                        effects: vec![Effect::Earthbend {
                            count: 1,
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
            effect: Effect::CantBeBlockedTarget {
                target: TargetSpec::PermanentYouControl {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        max_power: Some(2),
                        ..Default::default()
                    },
                },
            },
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
            effect: Effect::GrantKeywordsToTarget {
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
            },
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
        }],
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
            effect: Effect::ModifyUntilEot {
                power_delta: 1,
                toughness_delta: 0,
            },
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
            effect: Effect::PutCountersOnSource { count: 1 },
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
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
            effect: Effect::GainLife { amount: 3 },
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
                count: 1,
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
                effect: Effect::Modal {
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
                },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
                    modes: vec![
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [2, 0, 0, 0, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [1, 1, 0, 0, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [1, 0, 1, 0, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [1, 0, 0, 1, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [1, 0, 0, 0, 1, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 2, 0, 0, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 1, 1, 0, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 1, 0, 1, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 1, 0, 0, 1, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 0, 2, 0, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 0, 1, 1, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 0, 1, 0, 1, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 0, 0, 2, 0, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 0, 0, 1, 1, 0],
                            },
                            until_end_of_combat: false,
                        }],
                        vec![Effect::AddMana {
                            mana: Mana {
                                mana: [0, 0, 0, 0, 0, 0],
                                lesson: [0, 0, 0, 0, 2, 0],
                            },
                            until_end_of_combat: false,
                        }],
                    ],
                },
                ..Default::default()
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
        }],
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
            effect: Effect::CreateToken {
                token_name: "Clue".into(),
                count: 1,
                tapped_and_attacking: false,
            },
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
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
                look: 4,
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
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
            effect: Effect::ModifyControlled {
                predicate: CardPredicate {
                    subtype: Some("Dragon".into()),
                    ..Default::default()
                },
                power: 2,
                toughness: 0,
            },
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
            effect: Effect::Modal {
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
            },
            ..Default::default()
        }],
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
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
                    spec: TargetSpec::Spell,
                    min: 1,
                    max: 1,
                },
                effects: vec![Effect::CounterSpell {
                    target: TargetSpec::Spell,
                }],
            },
            Effect::DrawCards { count: 1 },
            Effect::MillCards { count: 3 },
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
                effect: Effect::DrawCards { count: 1 },
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                tap_source: true,
                effect: Effect::Modal {
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
                },
                ..Default::default()
            },
        ],
        replacement_effects: vec![ReplacementEffect::EntersTapped],
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
                count: 2,
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
            effect: Effect::ModifyControlled {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                power: 1,
                toughness: 1,
            },
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
            effect: Effect::LookAndSelect {
                reveal: true,
                look: 4,
                min_select: 0,
                max_select: 1,
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    max_power: Some(3),
                    ..Default::default()
                },
            },
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
            effects: vec![Effect::OptionalDiscardDraw { count: 1 }],
        }],
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
